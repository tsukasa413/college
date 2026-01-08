/**
 * @file cost_volume.cu
 * @brief CUDA kernel implementations for cost volume computation
 * 
 * Implements sphere sweeping stereo matching with adaptive camera selection.
 * 
 * Based on: "Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images"
 * Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
 * CVPR 2021
 * 
 * @copyright CC BY-NC-SA 3.0
 */

#include "sphere_stereo_ros/cuda/cost_volume.cuh"
#include "vec_utils.cuh"

#include <cstdio>
#include <cmath>
#include <cfloat>

namespace sphere_stereo_ros {
namespace cuda {

// =============================================================================
// Error Checking Macro
// =============================================================================

#define CUDA_CHECK_ERROR(call)                                                  \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            printf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,             \
                   cudaGetErrorString(err));                                    \
        }                                                                       \
    } while (0)

// =============================================================================
// Device Helper Functions (from vec_utils.cuh)
// =============================================================================

/**
 * @brief Double sphere unprojection (pixel to unit ray)
 */
__device__ __forceinline__
bool unprojectDoubleSphere(
    float u, float v,
    const Intrinsics& intr,
    float3& ray)
{
    // Normalize coordinates
    float mx = (u - intr.principal.x) / intr.fl.x;
    float my = (v - intr.principal.y) / intr.fl.y;
    float r2 = mx * mx + my * my;
    
    // Check validity
    float xi = intr.xi;
    float alpha = intr.alpha;
    
    // Validity check for Double Sphere model
    if (alpha > 0.5f) {
        float r2_max = 1.0f / (2.0f * alpha - 1.0f);
        if (r2 > r2_max) {
            return false;
        }
    }
    
    // Compute z
    float beta = 1.0f - (2.0f * alpha - 1.0f) * r2;
    if (beta < 0.0f) return false;
    
    float mz = (1.0f - alpha * alpha * r2) / (alpha * sqrtf(beta) + 1.0f - alpha);
    
    // Second step for xi
    float scale_denom = mz * mz + r2;
    if (scale_denom < 1e-10f) return false;
    
    float scale = (mz * xi + sqrtf(mz * mz + (1.0f - xi * xi) * r2)) / scale_denom;
    
    ray.x = scale * mx;
    ray.y = scale * my;
    ray.z = scale * mz - xi;
    
    // Normalize
    float norm = sqrtf(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
    if (norm > 1e-10f) {
        ray.x /= norm;
        ray.y /= norm;
        ray.z /= norm;
    }
    
    return ray.z > 0.0f;
}

/**
 * @brief Double sphere projection (3D point to pixel)
 */
__device__ __forceinline__
bool projectDoubleSphere(
    const float3& point,
    const Intrinsics& intr,
    float& u, float& v)
{
    float d1 = sqrtf(point.x * point.x + point.y * point.y + point.z * point.z);
    if (d1 < 1e-10f) return false;
    
    float z_shifted = point.z + intr.xi * d1;
    if (z_shifted < 1e-10f) return false;
    
    float d2 = sqrtf(point.x * point.x + point.y * point.y + z_shifted * z_shifted);
    if (d2 < 1e-10f) return false;
    
    float alpha = intr.alpha;
    float denom = alpha * d2 + (1.0f - alpha) * z_shifted;
    if (fabsf(denom) < 1e-10f) return false;
    
    float mx = point.x / denom;
    float my = point.y / denom;
    
    u = intr.fl.x * mx + intr.principal.x;
    v = intr.fl.y * my + intr.principal.y;
    
    return true;
}

/**
 * @brief Apply 3x3 rotation to float3
 */
__device__ __forceinline__
float3 applyRotation(const Rotation& rot, const float3& v)
{
    return make_float3(
        rot.r[0][0] * v.x + rot.r[0][1] * v.y + rot.r[0][2] * v.z,
        rot.r[1][0] * v.x + rot.r[1][1] * v.y + rot.r[1][2] * v.z,
        rot.r[2][0] * v.x + rot.r[2][1] * v.y + rot.r[2][2] * v.z
    );
}

/**
 * @brief Bilinear interpolation for uchar3
 */
__device__ __forceinline__
float3 bilinearSampleUchar3(
    const uchar3* image,
    float u, float v,
    int cols, int rows)
{
    // Clamp coordinates
    u = fmaxf(0.0f, fminf(u, static_cast<float>(cols - 1)));
    v = fmaxf(0.0f, fminf(v, static_cast<float>(rows - 1)));
    
    int x0 = static_cast<int>(floorf(u));
    int y0 = static_cast<int>(floorf(v));
    int x1 = min(x0 + 1, cols - 1);
    int y1 = min(y0 + 1, rows - 1);
    
    float fx = u - x0;
    float fy = v - y0;
    
    uchar3 c00 = image[y0 * cols + x0];
    uchar3 c01 = image[y0 * cols + x1];
    uchar3 c10 = image[y1 * cols + x0];
    uchar3 c11 = image[y1 * cols + x1];
    
    float3 result;
    result.x = (1.0f - fx) * (1.0f - fy) * c00.x + fx * (1.0f - fy) * c01.x +
               (1.0f - fx) * fy * c10.x + fx * fy * c11.x;
    result.y = (1.0f - fx) * (1.0f - fy) * c00.y + fx * (1.0f - fy) * c01.y +
               (1.0f - fx) * fy * c10.y + fx * fy * c11.y;
    result.z = (1.0f - fx) * (1.0f - fy) * c00.z + fx * (1.0f - fy) * c01.z +
               (1.0f - fx) * fy * c10.z + fx * fy * c11.z;
    
    return result;
}

/**
 * @brief Sample mask with bounds checking
 */
__device__ __forceinline__
float sampleMask(
    const float* mask,
    float u, float v,
    int cols, int rows)
{
    if (u < 0 || u >= cols - 1 || v < 0 || v >= rows - 1) {
        return 0.0f;
    }
    
    int x0 = static_cast<int>(floorf(u));
    int y0 = static_cast<int>(floorf(v));
    int x1 = min(x0 + 1, cols - 1);
    int y1 = min(y0 + 1, rows - 1);
    
    float fx = u - x0;
    float fy = v - y0;
    
    float m00 = mask[y0 * cols + x0];
    float m01 = mask[y0 * cols + x1];
    float m10 = mask[y1 * cols + x0];
    float m11 = mask[y1 * cols + x1];
    
    return (1.0f - fx) * (1.0f - fy) * m00 + fx * (1.0f - fy) * m01 +
           (1.0f - fx) * fy * m10 + fx * fy * m11;
}

// =============================================================================
// CUDA Kernels
// =============================================================================

/**
 * @brief Kernel: Select best matching camera per pixel
 * 
 * For each pixel, projects to near/far distances in each camera
 * and selects the camera with maximum displacement (best stereo baseline).
 */
__global__ void selectCameraKernel(
    int* selected_camera,
    const float* masks,
    const Intrinsics* intrinsics,
    const Rotation* rotations,
    const float3* translations,
    const Intrinsics reference_intrinsics,
    const int reference_index,
    const int cols,
    const int rows,
    const int num_cameras,
    const float min_dist,
    const float max_dist)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cols * rows) return;
    
    int x = idx % cols;
    int y = idx / cols;
    
    // Initialize with no selection
    selected_camera[idx] = -1;
    
    // Unproject pixel to unit ray in reference camera
    float3 ray_unit;
    if (!unprojectDoubleSphere(static_cast<float>(x), static_cast<float>(y),
                                reference_intrinsics, ray_unit)) {
        return;
    }
    
    // Reference mask check
    float ref_mask = masks[reference_index * rows * cols + idx];
    if (ref_mask < 0.9f) {
        return;
    }
    
    float max_displacement = -1.0f;
    int best_camera = -1;
    
    // Check each camera
    for (int cam = 0; cam < num_cameras; ++cam) {
        // Project to near and far distances
        float3 pt_near = make_float3(ray_unit.x * min_dist, 
                                     ray_unit.y * min_dist, 
                                     ray_unit.z * min_dist);
        float3 pt_far = make_float3(ray_unit.x * max_dist, 
                                    ray_unit.y * max_dist, 
                                    ray_unit.z * max_dist);
        
        // Transform to target camera coordinate system
        // pt_in_cam = R^T * (pt_in_ref - t)  where R, t are ref_to_cam transform
        // But we have cam_to_ref, so we need inverse:
        // pt_in_cam = R_cam_to_ref^T * pt_in_ref + (-R_cam_to_ref^T * t_cam_to_ref)
        // The rotations/translations are already relative (ref to cam)
        float3 pt_near_cam = applyRotation(rotations[cam], pt_near);
        pt_near_cam.x += translations[cam].x;
        pt_near_cam.y += translations[cam].y;
        pt_near_cam.z += translations[cam].z;
        
        float3 pt_far_cam = applyRotation(rotations[cam], pt_far);
        pt_far_cam.x += translations[cam].x;
        pt_far_cam.y += translations[cam].y;
        pt_far_cam.z += translations[cam].z;
        
        // Normalize for projection
        float norm_near = sqrtf(pt_near_cam.x * pt_near_cam.x + 
                               pt_near_cam.y * pt_near_cam.y + 
                               pt_near_cam.z * pt_near_cam.z);
        float norm_far = sqrtf(pt_far_cam.x * pt_far_cam.x + 
                              pt_far_cam.y * pt_far_cam.y + 
                              pt_far_cam.z * pt_far_cam.z);
        
        if (norm_near < 1e-10f || norm_far < 1e-10f) continue;
        
        pt_near_cam.x /= norm_near;
        pt_near_cam.y /= norm_near;
        pt_near_cam.z /= norm_near;
        
        pt_far_cam.x /= norm_far;
        pt_far_cam.y /= norm_far;
        pt_far_cam.z /= norm_far;
        
        // Project to camera image plane
        float u_near, v_near, u_far, v_far;
        if (!projectDoubleSphere(pt_near_cam, intrinsics[cam], u_near, v_near)) continue;
        if (!projectDoubleSphere(pt_far_cam, intrinsics[cam], u_far, v_far)) continue;
        
        // Check mask validity at projected locations
        float mask_near = sampleMask(masks + cam * rows * cols, 
                                     u_near, v_near, cols, rows);
        float mask_far = sampleMask(masks + cam * rows * cols,
                                    u_far, v_far, cols, rows);
        
        if (mask_near < 0.9f || mask_far < 0.9f) continue;
        
        // Compute displacement
        float du = u_near - u_far;
        float dv = v_near - v_far;
        float displacement = sqrtf(du * du + dv * dv);
        
        if (displacement > max_displacement) {
            max_displacement = displacement;
            best_camera = cam;
        }
    }
    
    selected_camera[idx] = best_camera;
}

/**
 * @brief Kernel: Compute cost volume via sphere sweeping
 * 
 * For each pixel and depth candidate, reprojects to the selected camera
 * and computes SAD cost in YCbCr space.
 */
__global__ void computeCostVolumeKernel(
    float* cost_volume,
    const uchar3* reference_image,
    const uchar3* images,
    const int* selected_camera,
    const Intrinsics* intrinsics,
    const Rotation* rotations,
    const float3* translations,
    const Intrinsics reference_intrinsics,
    const float* distance_candidates,
    const int cols,
    const int rows,
    const int num_depths,
    const int num_cameras,
    const float cost_clamp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cols * rows) return;
    
    int x = idx % cols;
    int y = idx / cols;
    
    // Get selected camera for this pixel
    int cam = selected_camera[idx];
    
    // Reference pixel color (YCbCr)
    uchar3 ref_color = reference_image[idx];
    float3 ref_ycbcr = make_float3(ref_color.x, ref_color.y, ref_color.z);
    
    // Unproject reference pixel to ray
    float3 ray_unit;
    bool valid_ray = unprojectDoubleSphere(static_cast<float>(x), static_cast<float>(y),
                                           reference_intrinsics, ray_unit);
    
    // Process each depth candidate
    for (int d = 0; d < num_depths; ++d) {
        float cost = cost_clamp;  // Default to max cost
        
        if (valid_ray && cam >= 0 && cam < num_cameras) {
            float dist = distance_candidates[d];
            
            // 3D point at this distance
            float3 pt = make_float3(ray_unit.x * dist, 
                                    ray_unit.y * dist, 
                                    ray_unit.z * dist);
            
            // Transform to target camera
            float3 pt_cam = applyRotation(rotations[cam], pt);
            pt_cam.x += translations[cam].x;
            pt_cam.y += translations[cam].y;
            pt_cam.z += translations[cam].z;
            
            // Normalize for projection
            float norm = sqrtf(pt_cam.x * pt_cam.x + 
                              pt_cam.y * pt_cam.y + 
                              pt_cam.z * pt_cam.z);
            
            if (norm > 1e-10f) {
                pt_cam.x /= norm;
                pt_cam.y /= norm;
                pt_cam.z /= norm;
                
                // Project to camera
                float u_cam, v_cam;
                if (projectDoubleSphere(pt_cam, intrinsics[cam], u_cam, v_cam)) {
                    // Check bounds
                    if (u_cam >= 0 && u_cam < cols - 1 && 
                        v_cam >= 0 && v_cam < rows - 1) {
                        
                        // Sample target camera image (bilinear)
                        const uchar3* target_image = images + cam * rows * cols;
                        float3 target_ycbcr = bilinearSampleUchar3(
                            target_image, u_cam, v_cam, cols, rows);
                        
                        // Compute SAD (Sum of Absolute Differences)
                        float sad = fabsf(ref_ycbcr.x - target_ycbcr.x) +
                                   fabsf(ref_ycbcr.y - target_ycbcr.y) +
                                   fabsf(ref_ycbcr.z - target_ycbcr.z);
                        
                        cost = fminf(sad, cost_clamp);
                    }
                }
            }
        }
        
        // Write cost to volume (CHW layout: depth x rows x cols)
        cost_volume[d * rows * cols + idx] = cost;
    }
}

/**
 * @brief Kernel: Select depth with quadratic sub-pixel refinement
 */
__global__ void selectDepthKernel(
    float* distance_map,
    const float* cost_volume,
    const float* distance_candidates,
    const int cols,
    const int rows,
    const int num_depths)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cols * rows) return;
    
    // Find minimum cost depth
    float min_cost = FLT_MAX;
    float max_cost = -FLT_MAX;
    int min_idx = num_depths - 1;
    
    for (int d = 0; d < num_depths; ++d) {
        float cost = cost_volume[d * rows * cols + idx];
        if (cost < min_cost) {
            min_cost = cost;
            min_idx = d;
        }
        if (cost > max_cost) {
            max_cost = cost;
        }
    }
    
    // Quadratic sub-pixel refinement (Eq. 5 from paper)
    float selected_idx = static_cast<float>(min_idx);
    
    if (min_idx > 0 && min_idx < num_depths - 1 && max_cost > min_cost + 1e-8f) {
        float left_cost = cost_volume[(min_idx - 1) * rows * cols + idx];
        float right_cost = cost_volume[(min_idx + 1) * rows * cols + idx];
        
        float denom = (left_cost + right_cost) - 2.0f * min_cost + 1e-8f;
        float variation = 0.5f * (left_cost - right_cost) / denom;
        variation = fmaxf(-0.5f, fminf(0.5f, variation));
        
        selected_idx += variation;
    }
    
    // Convert index to distance (inverse depth sampling)
    // distance = d0 / ((d0/d_max - 1) * idx / (num_depths - 1) + 1)
    float d0 = distance_candidates[0];
    float d_max = distance_candidates[num_depths - 1];
    
    float t = selected_idx / static_cast<float>(num_depths - 1);
    float distance = d0 / ((d0 / d_max - 1.0f) * t + 1.0f);
    
    // Handle flat cost case
    if (fabsf(max_cost - min_cost) < 1e-8f) {
        distance = d_max;
    }
    
    distance_map[idx] = distance;
}

/**
 * @brief Kernel: Convert RGB to YCbCr
 */
__global__ void rgb2YCbCrKernel(
    uchar3* output,
    const uchar3* input,
    const int cols,
    const int rows)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cols * rows) return;
    
    uchar3 rgb = input[idx];
    
    // RGB to YCbCr conversion
    // Y  =  0.299*R + 0.587*G + 0.114*B
    // Cb = -0.169*R - 0.331*G + 0.500*B + 128
    // Cr =  0.500*R - 0.419*G - 0.081*B + 128
    float r = rgb.x;
    float g = rgb.y;
    float b = rgb.z;
    
    float y  = 0.299f * r + 0.587f * g + 0.114f * b;
    float cb = -0.169f * r - 0.331f * g + 0.500f * b + 128.0f;
    float cr = 0.500f * r - 0.419f * g - 0.081f * b + 128.0f;
    
    output[idx] = make_uchar3(
        static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, y))),
        static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, cb))),
        static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, cr)))
    );
}

/**
 * @brief Kernel: Clear cost volume
 */
__global__ void clearCostVolumeKernel(
    float* cost_volume,
    const int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    cost_volume[idx] = 0.0f;
}

// =============================================================================
// Kernel Launch Functions
// =============================================================================

void launchSelectCameraKernel(
    int* selected_camera,
    const float* masks,
    const Intrinsics* intrinsics,
    const Rotation* rotations,
    const float3* translations,
    const Intrinsics& reference_intrinsics,
    int reference_index,
    const CostVolumeConfig& config,
    cudaStream_t stream)
{
    int total_pixels = config.cols * config.rows;
    int block_size = 256;
    int grid_size = (total_pixels + block_size - 1) / block_size;
    
    selectCameraKernel<<<grid_size, block_size, 0, stream>>>(
        selected_camera,
        masks,
        intrinsics,
        rotations,
        translations,
        reference_intrinsics,
        reference_index,
        config.cols,
        config.rows,
        config.num_cameras,
        config.min_dist,
        config.max_dist
    );
    
    CUDA_CHECK_ERROR(cudaGetLastError());
}

void launchComputeCostVolumeKernel(
    float* cost_volume,
    const uchar3* reference_image,
    const uchar3* images,
    const int* selected_camera,
    const Intrinsics* intrinsics,
    const Rotation* rotations,
    const float3* translations,
    const Intrinsics& reference_intrinsics,
    const float* distance_candidates,
    const CostVolumeConfig& config,
    cudaStream_t stream)
{
    int total_pixels = config.cols * config.rows;
    int block_size = 256;
    int grid_size = (total_pixels + block_size - 1) / block_size;
    
    computeCostVolumeKernel<<<grid_size, block_size, 0, stream>>>(
        cost_volume,
        reference_image,
        images,
        selected_camera,
        intrinsics,
        rotations,
        translations,
        reference_intrinsics,
        distance_candidates,
        config.cols,
        config.rows,
        config.num_depths,
        config.num_cameras,
        config.cost_clamp
    );
    
    CUDA_CHECK_ERROR(cudaGetLastError());
}

void launchSelectDepthKernel(
    float* distance_map,
    const float* cost_volume,
    const float* distance_candidates,
    const CostVolumeConfig& config,
    cudaStream_t stream)
{
    int total_pixels = config.cols * config.rows;
    int block_size = 256;
    int grid_size = (total_pixels + block_size - 1) / block_size;
    
    selectDepthKernel<<<grid_size, block_size, 0, stream>>>(
        distance_map,
        cost_volume,
        distance_candidates,
        config.cols,
        config.rows,
        config.num_depths
    );
    
    CUDA_CHECK_ERROR(cudaGetLastError());
}

void launchRgb2YCbCrKernel(
    uchar3* output,
    const uchar3* input,
    int cols,
    int rows,
    cudaStream_t stream)
{
    int total_pixels = cols * rows;
    int block_size = 256;
    int grid_size = (total_pixels + block_size - 1) / block_size;
    
    rgb2YCbCrKernel<<<grid_size, block_size, 0, stream>>>(
        output, input, cols, rows
    );
    
    CUDA_CHECK_ERROR(cudaGetLastError());
}

void launchClearCostVolumeKernel(
    float* cost_volume,
    int num_depths,
    int cols,
    int rows,
    cudaStream_t stream)
{
    int total_size = num_depths * cols * rows;
    int block_size = 256;
    int grid_size = (total_size + block_size - 1) / block_size;
    
    clearCostVolumeKernel<<<grid_size, block_size, 0, stream>>>(
        cost_volume, total_size
    );
    
    CUDA_CHECK_ERROR(cudaGetLastError());
}

}  // namespace cuda
}  // namespace sphere_stereo_ros
