/*
 * depth_estimation.cu
 *
 * High-performance CUDA kernels for sphere sweeping stereo depth estimation
 * Implements kernel fusion strategy to minimize intermediate memory writes
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>
#include <cuda_fp16.h>
#include <vector>
#include <algorithm>
#include "my_stereo_pkg/depth_estimation.hpp"
#include "utils_kernel.cuh"  // For Double Sphere projection functions
#include "my_stereo_pkg/vec_utils.cuh"  // For vector operators

// Type alias for compatibility
typedef unsigned char uchar;

// ============================================================================
// Constant Memory for Camera Calibrations
// ============================================================================

__constant__ DoubleSphereCalibration d_calib_constant[8];  // Up to 8 cameras
__constant__ CameraConfig d_config_constant;

// ============================================================================
// Texture Memory for Image Sampling
// ============================================================================

// Texture objects will be bound at runtime for each image
// Using float textures for normalized coordinates with hardware bilinear filtering

// ============================================================================
// Device-side Unprojection (Double Sphere Model)
// ============================================================================

/**
 * Unproject a 2D image coordinate to 3D unit direction vector
 * Using Double Sphere distortion model
 * Applies matching_scale as per Python implementation
 * 
 * @param uv Image coordinates (normalized to [0, width-1], [0, height-1])
 * @param calib Camera calibration parameters
 * @return 3D unit direction vector, or invalid if outside valid region
 */
__device__ float3 unproject_double_sphere(float2 uv, const DoubleSphereCalibration& calib) {
    // Convert to CameraCalibration format and apply matching_scale
    // Python: m_xy = (uv - principal * matching_scale) / (fl * matching_scale)
    CameraCalibration cam_calib;
    cam_calib.intrinsics.fx = calib.fx * calib.matching_scale;
    cam_calib.intrinsics.fy = calib.fy * calib.matching_scale;
    cam_calib.intrinsics.cx = calib.cx * calib.matching_scale;
    cam_calib.intrinsics.cy = calib.cy * calib.matching_scale;
    cam_calib.intrinsics.xi = calib.xi;
    cam_calib.intrinsics.alpha = calib.alpha;
    cam_calib.matching_scale = calib.matching_scale;
    
    float3 point_out;
    char valid;
    unproject_double_sphere(uv, cam_calib, point_out, valid);
    return point_out;
}

/**
 * Project a 3D point to 2D image coordinates
 * Using Double Sphere distortion model
 * Applies matching_scale as per Python implementation
 * 
 * @param point 3D point in camera frame
 * @param calib Camera calibration parameters
 * @return 2D image coordinates, or negative if behind camera
 */
__device__ float2 project_double_sphere(float3 point, const DoubleSphereCalibration& calib) {
    // Convert to CameraCalibration format and apply matching_scale
    // Python: uv = (fl * matching_scale * point.xy) / norm + principal * matching_scale
    CameraCalibration cam_calib;
    cam_calib.intrinsics.fx = calib.fx * calib.matching_scale;
    cam_calib.intrinsics.fy = calib.fy * calib.matching_scale;
    cam_calib.intrinsics.cx = calib.cx * calib.matching_scale;
    cam_calib.intrinsics.cy = calib.cy * calib.matching_scale;
    cam_calib.intrinsics.xi = calib.xi;
    cam_calib.intrinsics.alpha = calib.alpha;
    cam_calib.matching_scale = calib.matching_scale;
    
    float2 uv_out;
    char valid;
    project_double_sphere(point, cam_calib, uv_out, valid);
    return uv_out;
}

/**
 * Transform a 3D point via 4x4 rigid body transformation
 * Uses DoubleSphereCalibration's R (3x3) and t (3) arrays
 * R is row-major 3x3: [r00 r01 r02 r10 r11 r12 r20 r21 r22]
 */
__device__ float3 transform_point(float3 p, const DoubleSphereCalibration& calib) {
    // Apply: R @ p + t (using utils_kernel.cuh's apply_rotation and vec_utils.cuh operators)
    return apply_rotation(calib.R, p) + make_float3(calib.t[0], calib.t[1], calib.t[2]);
}

// ============================================================================
// Adaptive Camera Selection Kernel (Preprocessing)
// ============================================================================

/**
 * select_best_cameras_kernel
 * 
 * For each pixel in reference camera, determine which of the available cameras
 * provides the best baseline for stereo matching (maximum parallax).
 * 
 * Block: (32, 32) threads
 * Grid: ceil(width/32) x ceil(height/32)
 * 
 * Thread granularity: 1 pixel per thread
 * Each thread iterates over all cameras and selects the one with max displacement
 */
__global__ void select_best_cameras_kernel_impl(
    int* d_selected_cameras,
    float* d_max_displacement,
    const DoubleSphereCalibration* d_calibrations,
    const float* const* d_masks,
    const DoubleSphereCalibration ref_calib,
    int num_cameras
) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (pixel_x >= d_config_constant.matching_width || pixel_y >= d_config_constant.matching_height)
        return;
    
    int pixel_idx = pixel_y * d_config_constant.matching_width + pixel_x;
    
    // Unproject reference camera pixel to unit direction
    float2 uv = make_float2((float)pixel_x, (float)pixel_y);
    float3 pt_unit = unproject_double_sphere(uv, ref_calib);
    
    // Near and far sphere points (using vec_utils.cuh scalar multiplication)
    float3 pt_near = d_config_constant.min_dist * pt_unit;
    float3 pt_far = d_config_constant.max_dist * pt_unit;
    
    int best_camera = -1;
    float max_disp = 0.0f;
    
    // Iterate through all cameras to find best baseline
    for (int cam_idx = 0; cam_idx < num_cameras; cam_idx++) {
        if (cam_idx == 0) continue;  // Skip reference camera
        
        const DoubleSphereCalibration& cam_calib = d_calibrations[cam_idx];
        
        // Transform points to target camera frame
        float3 pt_near_cam = transform_point(pt_near, cam_calib);
        float3 pt_far_cam = transform_point(pt_far, cam_calib);
        
        // Project to image
        float2 uv_near = project_double_sphere(pt_near_cam, cam_calib);
        float2 uv_far = project_double_sphere(pt_far_cam, cam_calib);
        
        if (uv_near.x < 0 || uv_far.x < 0) continue;
        
        // Compute displacement
        float disp = sqrtf((uv_near.x - uv_far.x) * (uv_near.x - uv_far.x) +
                           (uv_near.y - uv_far.y) * (uv_near.y - uv_far.y));
        
        if (disp > max_disp) {
            max_disp = disp;
            best_camera = cam_idx;
        }
    }
    
    d_selected_cameras[pixel_idx] = best_camera;
    d_max_displacement[pixel_idx] = max_disp;
}

// ============================================================================
// Fused Depth Estimation Kernel (Main Computation)
// ============================================================================

/**
 * estimate_fisheye_distance_fused_kernel
 * 
 * Main fused kernel integrating:
 * 1. Unprojection: Reference image pixel -> 3D ray
 * 2. Distance sweep: Multiple distance candidates
 * 3. Reprojection: 3D point -> target camera images
 * 4. Sampling: Bilinear interpolation via texture memory
 * 5. Cost computation: SAD (Sum of Absolute Differences)
 * 6. Winner selection: Minimum cost distance
 * 
 * Optimization Strategy:
 * - Each thread processes 1 pixel and all candidates
 * - Cost values kept in registers (not global memory)
 * - Only final distance written to global memory
 * - Shared memory used for reference image data prefetch (if needed)
 * 
 * Block: (16, 16) threads for occupancy
 * Grid: ceil(width/16) x ceil(height/16)
 * 
 * Thread granularity: 1 pixel per thread
 * Register pressure: ~40-50 registers per thread (acceptable for Ampere/Turing)
 */
__global__ void estimate_fisheye_distance_fused_kernel_impl(
    float* d_distance_map,
    const uchar4* d_reference_image,
    const uchar4* const* d_images,
    const int* d_selected_cameras,
    const DoubleSphereCalibration* d_calibrations,
    const DoubleSphereCalibration reference_calib,
    const uchar* d_guide,
    int num_cameras,
    cudaTextureObject_t* d_texobjs
) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (pixel_x >= d_config_constant.matching_width || pixel_y >= d_config_constant.matching_height)
        return;
    
    int pixel_idx = pixel_y * d_config_constant.matching_width + pixel_x;
    
    // ========================================================================
    // Step 1: Unprojection in reference camera
    // ========================================================================
    
    // Use reference calibration passed as argument (not from constant memory)
    const DoubleSphereCalibration& ref_calib = reference_calib;
    float2 uv_ref = make_float2((float)pixel_x, (float)pixel_y);
    float3 pt_unit = unproject_double_sphere(uv_ref, ref_calib);
    
    if (pt_unit.z <= 0.0f) {
        d_distance_map[pixel_idx] = d_config_constant.max_dist;
        return;
    }
    
    // Reference image color (already in device memory)
    uchar4 ref_color = d_reference_image[pixel_idx];
    float ref_r = (float)ref_color.x;
    float ref_g = (float)ref_color.y;
    float ref_b = (float)ref_color.z;
    
    // ========================================================================
    // Step 2: Distance sweep with cost accumulation
    // ========================================================================
    
    int selected_cam = d_selected_cameras[pixel_idx];
    if (selected_cam < 0) {
        d_distance_map[pixel_idx] = d_config_constant.max_dist;
        return;
    }
    
    const DoubleSphereCalibration& selected_calib = d_calib_constant[selected_cam];
    
    float min_cost = FLT_MAX;
    float min_cost_index = (float)(d_config_constant.candidate_count - 1);
    float left_cost = FLT_MAX;   // Cost at min_index - 1
    float right_cost = FLT_MAX;  // Cost at min_index + 1
    float prev_cost = FLT_MAX;   // Previous iteration's cost
    
    for (int dist_idx = 0; dist_idx < d_config_constant.candidate_count; dist_idx++) {
        // Inverse distance parameterization for numerical stability
        float inv_dist_min = 1.0f / d_config_constant.min_dist;
        float inv_dist_max = 1.0f / d_config_constant.max_dist;
        float inv_dist = inv_dist_min - (inv_dist_min - inv_dist_max) * 
                         ((float)dist_idx / (float)(d_config_constant.candidate_count - 1));
        float distance = 1.0f / inv_dist;
        
        // 3D point at this distance (using vec_utils.cuh scalar multiplication)
        float3 pt_3d = distance * pt_unit;
        
        // Transform to selected camera
        float3 pt_cam = transform_point(pt_3d, selected_calib);
        
        // Project to image
        float2 uv_proj = project_double_sphere(pt_cam, selected_calib);
        
        if (uv_proj.x < 0 || uv_proj.x >= selected_calib.width || 
            uv_proj.y < 0 || uv_proj.y >= selected_calib.height) {
            continue;
        }
        
        // ====================================================================
        // Step 3: Bilinear Sampling via Texture Memory
        // ====================================================================
        
        // Normalize coordinates for texture sampling [0, 1]
        // Python's grid_sample with align_corners=False uses pixel centers:
        // normalized_coord = (pixel_coord + 0.5) / image_size
        // This matches CUDA texture sampling with normalizedCoords=true
        float2 uv_normalized = make_float2(
            (uv_proj.x + 0.5f) / selected_calib.width,
            (uv_proj.y + 0.5f) / selected_calib.height
        );
        
        // Sample from texture object (hardware bilinear interpolation)
        float4 sampled_color = tex2D<float4>(d_texobjs[selected_cam], uv_normalized.x, uv_normalized.y);
        
        // ====================================================================
        // Step 4: Cost Computation (SAD - Sum of Absolute Differences)
        // ====================================================================
        
        // Using vec_utils.cuh's absSum for cleaner code
        float3 ref_rgb = make_float3(ref_r, ref_g, ref_b);
        float3 tgt_rgb = make_float3(sampled_color.x, sampled_color.y, sampled_color.z);
        float cost = absSum(ref_rgb - tgt_rgb);
        
        // Python equivalent: cost_volume = torch.clamp(cost_volume, max=500)
        cost = fminf(cost, 500.0f);
        
        // Update minimum and track neighboring costs for quadratic fitting
        if (cost < min_cost) {
            // Shift costs: previous min becomes left neighbor
            if (dist_idx > 0) {
                left_cost = prev_cost;
            }
            min_cost = cost;
            min_cost_index = (float)dist_idx;
        } else if (dist_idx == (int)(min_cost_index + 1.0f)) {
            // This is the right neighbor
            right_cost = cost;
        }
        
        prev_cost = cost;
    }
    
    // ========================================================================
    // Step 5: Subpixel Refinement via Quadratic Fitting
    // ========================================================================
    // Python implementation:
    // variation = 0.5 * (left_cost - right_cost) / (left_cost + right_cost - 2*min_cost + 1e-8)
    // variation = clamp(variation, -0.5, 0.5)
    // selected_index_map = selected_index_map + variation
    
    float variation = 0.0f;
    int min_idx_int = (int)min_cost_index;
    
    // Only apply quadratic fitting if we have valid neighbors
    if (min_idx_int > 0 && min_idx_int < d_config_constant.candidate_count - 1 &&
        left_cost < FLT_MAX && right_cost < FLT_MAX) {
        
        float denominator = left_cost + right_cost - 2.0f * min_cost + 1e-8f;
        if (fabsf(denominator) > 1e-6f) {
            variation = 0.5f * (left_cost - right_cost) / denominator;
            variation = fmaxf(-0.5f, fminf(0.5f, variation));  // Clamp to [-0.5, 0.5]
        }
    }
    
    // Add sub-pixel offset
    min_cost_index += variation;
    
    // ========================================================================
    // Step 6: Distance Conversion
    // ========================================================================
    
    float inv_dist_min = 1.0f / d_config_constant.min_dist;
    float inv_dist_max = 1.0f / d_config_constant.max_dist;
    float inv_dist_at_min_idx = inv_dist_min - (inv_dist_min - inv_dist_max) * 
                                (min_cost_index / (float)(d_config_constant.candidate_count - 1));
    
    d_distance_map[pixel_idx] = 1.0f / inv_dist_at_min_idx;
}

// ============================================================================
// Kernel Launchers (C++ callable)
// ============================================================================

void select_best_cameras_kernel(
    int* d_selected_cameras,
    float* d_max_displacement,
    const DoubleSphereCalibration* d_calibrations,
    const float* const* d_masks,
    const DoubleSphereCalibration& reference_calib,
    const CameraConfig& config,
    cudaStream_t stream
) {
    // Copy config to constant memory
    cudaMemcpyToSymbolAsync(d_config_constant, &config, sizeof(CameraConfig), 0, cudaMemcpyHostToDevice, stream);
    
    // Skip copying calibrations to constant memory - use global memory instead
    
    dim3 block(32, 32);
    dim3 grid((config.matching_width + 31) / 32, (config.matching_height + 31) / 32);
    
    select_best_cameras_kernel_impl<<<grid, block, 0, stream>>>(
        d_selected_cameras, d_max_displacement, d_calibrations, d_masks, 
        reference_calib, config.num_cameras
    );
}

void estimate_fisheye_distance_fused_kernel(
    float* d_distance_map,
    const uchar4* d_reference_image,
    const uchar4* const* d_images,
    const int* d_selected_cameras,
    const DoubleSphereCalibration* d_calibrations,
    const DoubleSphereCalibration& reference_calib,
    const uchar* d_guide,
    const CameraConfig& config,
    cudaStream_t stream,
    cudaTextureObject_t* d_texobjs
) {
    // Copy config to constant memory (skip calibrations - use global memory)
    cudaMemcpyToSymbolAsync(d_config_constant, &config, sizeof(CameraConfig), 0, cudaMemcpyHostToDevice, stream);
    
    dim3 block(16, 16);
    dim3 grid((config.matching_width + 15) / 16, (config.matching_height + 15) / 16);
    
    estimate_fisheye_distance_fused_kernel_impl<<<grid, block, 0, stream>>>(
        d_distance_map, d_reference_image, d_images, d_selected_cameras,
        d_calibrations, reference_calib, d_guide, config.num_cameras, d_texobjs
    );
}
// ============================================================================
// Cost Volume Based Depth Estimation (Python-equivalent pipeline)
// ============================================================================

/**
 * Generate full cost volume for all distance candidates
 * Output: [candidate_count, height, width] cost values
 */
__global__ void compute_cost_volume_kernel_impl(
    float* d_cost_volume,           // [candidate_count * height * width]
    const uchar4* d_reference_image,
    const uchar4* const* d_images,
    const int* d_selected_cameras,
    const DoubleSphereCalibration* d_calibrations,
    const DoubleSphereCalibration reference_calib,
    int num_cameras,
    cudaTextureObject_t* d_texobjs
) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    int dist_idx = blockIdx.z;  // Depth candidate index
    
    if (pixel_x >= d_config_constant.matching_width || pixel_y >= d_config_constant.matching_height)
        return;
    
    int pixel_idx = pixel_y * d_config_constant.matching_width + pixel_x;
    int cost_idx = dist_idx * (d_config_constant.matching_width * d_config_constant.matching_height) + pixel_idx;
    
    // Unproject reference pixel
    float2 uv_ref = make_float2((float)pixel_x, (float)pixel_y);
    float3 pt_unit = unproject_double_sphere(uv_ref, reference_calib);
    
    if (pt_unit.z <= 0.0f) {
        d_cost_volume[cost_idx] = 500.0f;  // Max cost for invalid
        return;
    }
    
    // Get selected camera
    int selected_cam = d_selected_cameras[pixel_idx];
    if (selected_cam < 0) {
        d_cost_volume[cost_idx] = 500.0f;
        return;
    }
    
    const DoubleSphereCalibration& selected_calib = d_calib_constant[selected_cam];
    
    // Compute distance for this candidate
    float inv_dist_min = 1.0f / d_config_constant.min_dist;
    float inv_dist_max = 1.0f / d_config_constant.max_dist;
    float inv_dist = inv_dist_min - (inv_dist_min - inv_dist_max) * 
                     ((float)dist_idx / (float)(d_config_constant.candidate_count - 1));
    float distance = 1.0f / inv_dist;
    
    // 3D point
    float3 pt_3d = distance * pt_unit;
    
    // Transform to selected camera
    float3 pt_cam = transform_point(pt_3d, selected_calib);
    
    // Project
    float2 uv_proj = project_double_sphere(pt_cam, selected_calib);
    
    if (uv_proj.x < 0 || uv_proj.x >= selected_calib.width || 
        uv_proj.y < 0 || uv_proj.y >= selected_calib.height) {
        d_cost_volume[cost_idx] = 500.0f;
        return;
    }
    
    // Sample via texture
    float2 uv_normalized = make_float2(
        (uv_proj.x + 0.5f) / selected_calib.width,
        (uv_proj.y + 0.5f) / selected_calib.height
    );
    float4 sampled_color = tex2D<float4>(d_texobjs[selected_cam], uv_normalized.x, uv_normalized.y);
    
    // Compute SAD cost
    uchar4 ref_color = d_reference_image[pixel_idx];
    float3 ref_rgb = make_float3((float)ref_color.x, (float)ref_color.y, (float)ref_color.z);
    float3 tgt_rgb = make_float3(sampled_color.x, sampled_color.y, sampled_color.z);
    float cost = absSum(ref_rgb - tgt_rgb);
    
    // Python: cost_volume = torch.clamp(cost_volume, max=500)
    cost = fminf(cost, 500.0f);
    
    d_cost_volume[cost_idx] = cost;
}

/**
 * Select distance from filtered cost volume with quadratic fitting
 */
__global__ void select_distance_from_cost_volume_kernel_impl(
    float* d_distance_map,
    const float* d_cost_volume,
    int width,
    int height
) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (pixel_x >= width || pixel_y >= height)
        return;
    
    int pixel_idx = pixel_y * width + pixel_x;
    
    // Find minimum cost
    float min_cost = FLT_MAX;
    int min_idx = d_config_constant.candidate_count - 1;
    
    for (int d = 0; d < d_config_constant.candidate_count; d++) {
        int cost_idx = d * (width * height) + pixel_idx;
        float cost = d_cost_volume[cost_idx];
        if (cost < min_cost) {
            min_cost = cost;
            min_idx = d;
        }
    }
    
    // Quadratic fitting
    float variation = 0.0f;
    if (min_idx > 0 && min_idx < d_config_constant.candidate_count - 1) {
        int left_idx = (min_idx - 1) * (width * height) + pixel_idx;
        int center_idx = min_idx * (width * height) + pixel_idx;
        int right_idx = (min_idx + 1) * (width * height) + pixel_idx;
        
        float left_cost = d_cost_volume[left_idx];
        float center_cost = d_cost_volume[center_idx];
        float right_cost = d_cost_volume[right_idx];
        
        float denominator = left_cost + right_cost - 2.0f * center_cost + 1e-8f;
        if (fabsf(denominator) > 1e-6f) {
            variation = 0.5f * (left_cost - right_cost) / denominator;
            variation = fmaxf(-0.5f, fminf(0.5f, variation));
        }
    }
    
    float refined_idx = (float)min_idx + variation;
    
    // Convert to distance
    float inv_dist_min = 1.0f / d_config_constant.min_dist;
    float inv_dist_max = 1.0f / d_config_constant.max_dist;
    float inv_dist = inv_dist_min - (inv_dist_min - inv_dist_max) * 
                     (refined_idx / (float)(d_config_constant.candidate_count - 1));
    float distance = 1.0f / inv_dist;
    
    d_distance_map[pixel_idx] = distance;
}


// Wrapper for cost volume generation
void compute_cost_volume_kernel(
    float* d_cost_volume,
    const uchar4* d_reference_image,
    const uchar4* const* d_images,
    const int* d_selected_cameras,
    const DoubleSphereCalibration* d_calibrations,
    const DoubleSphereCalibration& reference_calib,
    const CameraConfig& config,
    cudaStream_t stream,
    cudaTextureObject_t* d_texobjs
) {
    cudaMemcpyToSymbolAsync(d_config_constant, &config, sizeof(CameraConfig), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(d_calib_constant, d_calibrations, config.num_cameras * sizeof(DoubleSphereCalibration), 0, cudaMemcpyDeviceToDevice, stream);
    
    dim3 block(16, 16);
    dim3 grid((config.matching_width + 15) / 16, (config.matching_height + 15) / 16, config.candidate_count);
    
    compute_cost_volume_kernel_impl<<<grid, block, 0, stream>>>(
        d_cost_volume, d_reference_image, d_images, d_selected_cameras,
        d_calibrations, reference_calib, config.num_cameras, d_texobjs
    );
}

// Wrapper for distance selection
void select_distance_from_cost_volume_kernel(
    float* d_distance_map,
    const float* d_cost_volume,
    const CameraConfig& config,
    cudaStream_t stream
) {
    cudaMemcpyToSymbolAsync(d_config_constant, &config, sizeof(CameraConfig), 0, cudaMemcpyHostToDevice, stream);
    
    dim3 block(16, 16);
    dim3 grid((config.matching_width + 15) / 16, (config.matching_height + 15) / 16);
    
    select_distance_from_cost_volume_kernel_impl<<<grid, block, 0, stream>>>(
        d_distance_map, d_cost_volume, config.matching_width, config.matching_height
    );
}

// ============================================================================
// ISB Filter Implementation
// ============================================================================

#include "isb_filter.cuh"
#include <cmath>

// Instantiate template for 64 candidates (typical config)
template __global__ void isb_downsample_kernel<64>(
    const uchar3*, const float*, int, int, uchar3*, float*, int, int, float);
template __global__ void isb_upsample_kernel<64>(
    const uchar3*, const float*, int, int, uchar3*, float*, int, int, float, float, float);

/**
 * Apply ISB Filter to cost volume
 * Multi-scale bilateral filtering with edge-preserving downsampling/upsampling
 */
void apply_isb_filter(
    uchar3* d_guide,
    float* d_cost_volume,
    int width,
    int height,
    int candidate_count,
    float sigma_i,
    float sigma_s,
    cudaStream_t stream
) {
    // Compute variance inverses
    float var_inv_s = 1.0f / (2.0f * sigma_s * sigma_s);
    float var_inv_i = 1.0f / (2.0f * sigma_i * sigma_i);
    
    // Calculate number of scales
    int scale_count = std::min((int)std::log2(width), (int)std::log2(height)) - 1;
    scale_count = std::max(1, scale_count);
    
    // Allocate pyramid buffers
    std::vector<uchar3*> d_guides(scale_count);
    std::vector<float*> d_costs(scale_count);
    std::vector<int> widths(scale_count);
    std::vector<int> heights(scale_count);
    
    d_guides[0] = d_guide;
    d_costs[0] = d_cost_volume;
    widths[0] = width;
    heights[0] = height;
    
    // Allocate downsampled scales
    for (int scale = 1; scale < scale_count; scale++) {
        widths[scale] = (widths[scale - 1] + 1) / 2;
        heights[scale] = (heights[scale - 1] + 1) / 2;
        
        cudaMalloc(&d_guides[scale], widths[scale] * heights[scale] * sizeof(uchar3));
        cudaMalloc(&d_costs[scale], widths[scale] * heights[scale] * candidate_count * sizeof(float));
    }
    
    // Downsample pyramid
    for (int scale = 1; scale < scale_count; scale++) {
        int block_size = 256;
        int grid_size = (widths[scale] * heights[scale] + block_size - 1) / block_size;
        
        if (candidate_count == 64) {
            isb_downsample_kernel<64><<<grid_size, block_size, 0, stream>>>(
                d_guides[scale - 1], d_costs[scale - 1], heights[scale - 1], widths[scale - 1],
                d_guides[scale], d_costs[scale], heights[scale], widths[scale], var_inv_i
            );
        }
    }
    
    // Upsample pyramid
    for (int scale = scale_count - 2; scale >= 0; scale--) {
        float distance = std::pow(2.0f, scale) - 0.5f;
        float weight_down = std::exp(-(distance * distance) * var_inv_s);
        float weight_up = 1.0f - weight_down;
        
        int block_size = 256;
        int grid_size = (widths[scale + 1] * heights[scale + 1] + block_size - 1) / block_size;
        
        if (candidate_count == 64) {
            isb_upsample_kernel<64><<<grid_size, block_size, 0, stream>>>(
                d_guides[scale + 1], d_costs[scale + 1], heights[scale + 1], widths[scale + 1],
                d_guides[scale], d_costs[scale], heights[scale], widths[scale],
                weight_up, weight_down, var_inv_i
            );
        }
    }
    
    // Free intermediate scales
    for (int scale = 1; scale < scale_count; scale++) {
        cudaFree(d_guides[scale]);
        cudaFree(d_costs[scale]);
    }
}
