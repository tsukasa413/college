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
#include "my_stereo_pkg/depth_estimation.hpp"

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
 * 
 * @param uv Image coordinates (normalized to [0, width-1], [0, height-1])
 * @param calib Camera calibration parameters
 * @return 3D unit direction vector, or invalid if outside valid region
 */
__device__ float3 unproject_double_sphere(float2 uv, const DoubleSphereCalibration& calib) {
    // Normalized image coordinates
    float x = (uv.x - calib.cx) / calib.fx;
    float y = (uv.y - calib.cy) / calib.fy;
    
    // Fisheye (spherical) coordinate
    float rho_sq = x * x + y * y;
    float rho = sqrtf(rho_sq);
    
    // Double sphere inverse (Kannala-Branden model variant)
    // r = xi + sqrt(1 + (1-xi^2)*rho^2)
    float xi = calib.xi;
    float alpha = calib.alpha;
    
    float numer = 1.0f - xi * xi * rho_sq;
    if (numer < 0.0f) return make_float3(0.0f, 0.0f, 0.0f);  // Invalid
    
    float r = xi + sqrtf(numer) / (1.0f + alpha * rho_sq);
    
    float3 point = make_float3(x / r, y / r, 1.0f / r);
    float norm = sqrtf(point.x * point.x + point.y * point.y + point.z * point.z);
    return make_float3(point.x / norm, point.y / norm, point.z / norm);
}

/**
 * Project a 3D point to 2D image coordinates
 * Using Double Sphere distortion model
 * 
 * @param point 3D point in camera frame
 * @param calib Camera calibration parameters
 * @return 2D image coordinates, or negative if behind camera
 */
__device__ float2 project_double_sphere(float3 point, const DoubleSphereCalibration& calib) {
    // Normalize to unit sphere
    float norm = sqrtf(point.x * point.x + point.y * point.y + point.z * point.z);
    if (norm < 1e-6f) return make_float2(-1.0f, -1.0f);
    
    point = make_float3(point.x / norm, point.y / norm, point.z / norm);
    
    // Double sphere projection
    float xi = calib.xi;
    float alpha = calib.alpha;
    
    // Intermediate sphere projection
    float z1 = (xi * norm + sqrtf(1.0f - xi * xi + xi * xi * norm * norm));
    if (z1 <= 0.0f) return make_float2(-1.0f, -1.0f);
    
    float x1 = point.x / z1;
    float y1 = point.y / z1;
    
    // Second sphere / distortion
    float rho_sq = x1 * x1 + y1 * y1;
    float r = alpha + sqrtf(1.0f - alpha * alpha * (1.0f + rho_sq));
    
    float x_proj = r * x1;
    float y_proj = r * y1;
    
    // Camera intrinsics
    float u = calib.fx * x_proj + calib.cx;
    float v = calib.fy * y_proj + calib.cy;
    
    return make_float2(u, v);
}

/**
 * Transform a 3D point via 4x4 rigid body transformation
 * RT matrix stored as row-major
 */
__device__ float3 transform_point(float3 p, const float rt[16]) {
    float4 p_homog = make_float4(p.x, p.y, p.z, 1.0f);
    
    // Matrix multiplication (row-major)
    float x = rt[0*4+0] * p_homog.x + rt[0*4+1] * p_homog.y + rt[0*4+2] * p_homog.z + rt[0*4+3] * p_homog.w;
    float y = rt[1*4+0] * p_homog.x + rt[1*4+1] * p_homog.y + rt[1*4+2] * p_homog.z + rt[1*4+3] * p_homog.w;
    float z = rt[2*4+0] * p_homog.x + rt[2*4+1] * p_homog.y + rt[2*4+2] * p_homog.z + rt[2*4+3] * p_homog.w;
    
    return make_float3(x, y, z);
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
    
    // Near and far sphere points
    float3 pt_near = make_float3(pt_unit.x * d_config_constant.min_dist,
                                  pt_unit.y * d_config_constant.min_dist,
                                  pt_unit.z * d_config_constant.min_dist);
    float3 pt_far = make_float3(pt_unit.x * d_config_constant.max_dist,
                                 pt_unit.y * d_config_constant.max_dist,
                                 pt_unit.z * d_config_constant.max_dist);
    
    int best_camera = -1;
    float max_disp = 0.0f;
    
    // Iterate through all cameras to find best baseline
    for (int cam_idx = 0; cam_idx < num_cameras; cam_idx++) {
        if (cam_idx == 0) continue;  // Skip reference camera
        
        const DoubleSphereCalibration& cam_calib = d_calibrations[cam_idx];
        
        // Transform points to target camera frame
        float3 pt_near_cam = transform_point(pt_near, cam_calib.rt);
        float3 pt_far_cam = transform_point(pt_far, cam_calib.rt);
        
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
    
    const DoubleSphereCalibration& ref_calib = d_calib_constant[0];
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
    
    for (int dist_idx = 0; dist_idx < d_config_constant.candidate_count; dist_idx++) {
        // Inverse distance parameterization for numerical stability
        float inv_dist_min = 1.0f / d_config_constant.min_dist;
        float inv_dist_max = 1.0f / d_config_constant.max_dist;
        float inv_dist = inv_dist_min - (inv_dist_min - inv_dist_max) * 
                         ((float)dist_idx / (float)(d_config_constant.candidate_count - 1));
        float distance = 1.0f / inv_dist;
        
        // 3D point at this distance
        float3 pt_3d = make_float3(pt_unit.x * distance,
                                    pt_unit.y * distance,
                                    pt_unit.z * distance);
        
        // Transform to selected camera
        float3 pt_cam = transform_point(pt_3d, selected_calib.rt);
        
        // Project to image
        float2 uv_proj = project_double_sphere(pt_cam, selected_calib);
        
        if (uv_proj.x < 0 || uv_proj.x >= selected_calib.width || 
            uv_proj.y < 0 || uv_proj.y >= selected_calib.height) {
            continue;
        }
        
        // ====================================================================
        // Step 3: Bilinear Sampling via Texture Memory
        // ====================================================================
        
        // Normalize coordinates for texture sampling [-1, 1]
        float2 uv_normalized = make_float2(
            2.0f * (uv_proj.x + 0.5f) / selected_calib.width - 1.0f,
            2.0f * (uv_proj.y + 0.5f) / selected_calib.height - 1.0f
        );
        
        // Sample from texture object (hardware bilinear interpolation)
        float4 sampled_color = tex2D<float4>(d_texobjs[selected_cam], uv_normalized.x, uv_normalized.y);
        
        // ====================================================================
        // Step 4: Cost Computation (SAD - Sum of Absolute Differences)
        // ====================================================================
        
        float cost = fabsf(ref_r - sampled_color.x) +
                     fabsf(ref_g - sampled_color.y) +
                     fabsf(ref_b - sampled_color.z);
        
        // Update minimum
        if (cost < min_cost) {
            min_cost = cost;
            min_cost_index = (float)dist_idx;
        }
    }
    
    // ========================================================================
    // Step 5: Subpixel Refinement via Quadratic Fitting (Register-based)
    // ========================================================================
    // Note: Full refinement requires cost_volume; simplified here
    // In production, compute nearby costs and fit quadratic
    
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
    
    // Copy calibrations to constant memory
    cudaMemcpyToSymbolAsync(d_calib_constant, d_calibrations, 
                            config.num_cameras * sizeof(DoubleSphereCalibration), 0, 
                            cudaMemcpyDeviceToDevice, stream);
    
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
    const uchar* d_guide,
    const CameraConfig& config,
    cudaStream_t stream,
    cudaTextureObject_t* d_texobjs
) {
    // Copy config and calibrations to constant memory
    cudaMemcpyToSymbolAsync(d_config_constant, &config, sizeof(CameraConfig), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(d_calib_constant, d_calibrations,
                            config.num_cameras * sizeof(DoubleSphereCalibration), 0,
                            cudaMemcpyDeviceToDevice, stream);
    
    dim3 block(16, 16);
    dim3 grid((config.matching_width + 15) / 16, (config.matching_height + 15) / 16);
    
    estimate_fisheye_distance_fused_kernel_impl<<<grid, block, 0, stream>>>(
        d_distance_map, d_reference_image, d_images, d_selected_cameras,
        d_calibrations, d_guide, config.num_cameras, d_texobjs
    );
}

void refine_distance_quadratic_kernel(
    float* d_distance_map,
    const float* d_cost_volume,
    const CameraConfig& config,
    cudaStream_t stream
) {
    // Placeholder for subpixel refinement
    // In full implementation, perform quadratic fitting on neighboring cost values
    // to refine distance to sub-candidate precision
    cudaStreamSynchronize(stream);
}

void grid_sample_texture_kernel(
    float* d_output,
    cudaTextureObject_t d_texobj,
    const float2* d_sample_coords,
    int width, int height,
    int total_pixels,
    cudaStream_t stream
) {
    // Utility kernel for explicit grid sampling if needed
    // In fused kernel, this is integrated directly
    // Placeholder implementation
}
