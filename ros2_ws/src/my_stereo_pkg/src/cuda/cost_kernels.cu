/**
=======================================================================
Cost Volume Computation CUDA Kernels
-------------------
Implements cost volume computation and quadratic fitting for depth estimation
from the Sphere Sweeping Stereo paper
=======================================================================
**/

#include "my_stereo_pkg/cuda_kernels.hpp"
#include <torch/torch.h>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Compute cost volume kernel
 * Calculates absolute difference between sweeping volume and reference image
 * Implements: torch.sum(torch.abs(sweeping_volume - reference_image), dim=1)
 */
__global__ void computeCostVolumeKernel(
    const float* sweeping_volume,  // [1, 3, candidate_count, H, W]
    const float* reference_image,  // [1, 3, 1, H, W]
    float* cost_volume,            // [candidate_count, H, W]
    int candidate_count,
    int rows,
    int cols
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = candidate_count * rows * cols;
    
    if (idx < total_pixels) {
        int d = idx / (rows * cols);        // distance candidate index
        int pixel_idx = idx % (rows * cols); // pixel index in H*W
        int y = pixel_idx / cols;
        int x = pixel_idx % cols;
        
        float cost = 0.0f;
        
        // Sum absolute differences across 3 color channels
        for (int c = 0; c < 3; ++c) {
            // sweeping_volume: [1, 3, candidate_count, H, W]
            // Access pattern: batch=0, channel=c, depth=d, row=y, col=x
            int sweep_idx = c * candidate_count * rows * cols + d * rows * cols + y * cols + x;
            
            // reference_image: [1, 3, 1, H, W]
            // Access pattern: batch=0, channel=c, depth=0, row=y, col=x
            int ref_idx = c * rows * cols + y * cols + x;
            
            float diff = sweeping_volume[sweep_idx] - reference_image[ref_idx];
            cost += fabsf(diff);
        }
        
        // Output: [candidate_count, H, W]
        cost_volume[d * rows * cols + y * cols + x] = cost;
    }
}

/**
 * Quadratic fitting kernel
 * Finds minimum cost and applies parabolic interpolation for sub-pixel accuracy
 * Implements Python logic from estimate_fisheye_distance
 */
__global__ void quadraticFittingKernel(
    const float* cost_volume,         // [candidate_count, H, W]
    const float* distance_candidates, // [candidate_count]
    float* distance_map,              // [H, W]
    int candidate_count,
    int rows,
    int cols
)
{
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = rows * cols;
    
    if (pixel_idx < total_pixels) {
        int y = pixel_idx / cols;
        int x = pixel_idx % cols;
        
        // Find minimum cost
        float min_cost = INFINITY;
        float max_cost = -INFINITY;
        int min_idx = 0;
        
        for (int d = 0; d < candidate_count; ++d) {
            float cost = cost_volume[d * rows * cols + pixel_idx];
            if (cost < min_cost) {
                min_cost = cost;
                min_idx = d;
            }
            if (cost > max_cost) {
                max_cost = cost;
            }
        }
        
        // If all costs are equal, use maximum distance
        if (fabsf(max_cost - min_cost) < 1e-8f) {
            distance_map[pixel_idx] = distance_candidates[candidate_count - 1];
            return;
        }
        
        // Quadratic fitting for sub-candidate accuracy
        // Get left and right neighbor costs
        int left_idx = max(0, min_idx - 1);
        int right_idx = min(candidate_count - 1, min_idx + 1);
        
        float left_cost = cost_volume[left_idx * rows * cols + pixel_idx];
        float right_cost = cost_volume[right_idx * rows * cols + pixel_idx];
        
        // Compute parabolic variation
        // variation = 0.5 * (left_cost - right_cost) / ((left_cost + right_cost) - 2 * min_cost + epsilon)
        float denominator = (left_cost + right_cost) - 2.0f * min_cost + 1e-8f;
        float variation = 0.5f * (left_cost - right_cost) / denominator;
        
        // Clamp variation to [-0.5, 0.5]
        variation = fmaxf(-0.5f, fminf(0.5f, variation));
        
        // Don't apply variation at boundaries
        if (min_idx == 0 || min_idx == candidate_count - 1) {
            variation = 0.0f;
        }
        
        // Compute fractional index
        float selected_index = static_cast<float>(min_idx) + variation;
        
        // Convert index to distance using inverse linear interpolation
        // distance_map = dist_0 / ((dist_0 / dist_last - 1) * selected_index / (candidate_count - 1) + 1)
        float dist_0 = distance_candidates[0];
        float dist_last = distance_candidates[candidate_count - 1];
        float ratio = (dist_0 / dist_last - 1.0f) * selected_index / (candidate_count - 1) + 1.0f;
        
        distance_map[pixel_idx] = dist_0 / ratio;
    }
}

// ============================================================================
// C++ Wrapper Functions (LibTorch interface)
// ============================================================================

template<typename T>
T* get_device_ptr(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA device");
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
    return tensor.data_ptr<T>();
}

void launch_compute_cost_volume(
    const at::Tensor& sweeping_volume,
    const at::Tensor& reference_image,
    const at::Tensor& cost_volume,
    int candidate_count,
    int rows,
    int cols
)
{
    // Validate inputs
    TORCH_CHECK(sweeping_volume.is_cuda(), "sweeping_volume must be on CUDA device");
    TORCH_CHECK(reference_image.is_cuda(), "reference_image must be on CUDA device");
    TORCH_CHECK(cost_volume.is_cuda(), "cost_volume must be on CUDA device");
    
    TORCH_CHECK(sweeping_volume.dtype() == at::kFloat, "sweeping_volume must be float32");
    TORCH_CHECK(reference_image.dtype() == at::kFloat, "reference_image must be float32");
    TORCH_CHECK(cost_volume.dtype() == at::kFloat, "cost_volume must be float32");
    
    // Validate dimensions
    TORCH_CHECK(sweeping_volume.dim() == 5, "sweeping_volume must be 5D [1, 3, D, H, W]");
    TORCH_CHECK(reference_image.dim() == 5, "reference_image must be 5D [1, 3, 1, H, W]");
    TORCH_CHECK(cost_volume.dim() == 3, "cost_volume must be 3D [D, H, W]");
    
    // Get device pointers
    const float* d_sweeping = get_device_ptr<float>(sweeping_volume);
    const float* d_reference = get_device_ptr<float>(reference_image);
    float* d_cost = get_device_ptr<float>(cost_volume);
    
    // Calculate grid and block sizes
    int total_pixels = candidate_count * rows * cols;
    int blockSize = 256;
    int gridSize = (total_pixels + blockSize - 1) / blockSize;
    
    // Launch kernel
    computeCostVolumeKernel<<<gridSize, blockSize>>>(
        d_sweeping, d_reference, d_cost,
        candidate_count, rows, cols
    );
    
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_quadratic_fitting(
    const at::Tensor& cost_volume,
    const at::Tensor& distance_candidates,
    const at::Tensor& distance_map,
    int candidate_count,
    int rows,
    int cols
)
{
    // Validate inputs
    TORCH_CHECK(cost_volume.is_cuda(), "cost_volume must be on CUDA device");
    TORCH_CHECK(distance_candidates.is_cuda(), "distance_candidates must be on CUDA device");
    TORCH_CHECK(distance_map.is_cuda(), "distance_map must be on CUDA device");
    
    TORCH_CHECK(cost_volume.dtype() == at::kFloat, "cost_volume must be float32");
    TORCH_CHECK(distance_candidates.dtype() == at::kFloat, "distance_candidates must be float32");
    TORCH_CHECK(distance_map.dtype() == at::kFloat, "distance_map must be float32");
    
    // Validate dimensions
    TORCH_CHECK(cost_volume.dim() == 3, "cost_volume must be 3D [D, H, W]");
    TORCH_CHECK(distance_candidates.dim() == 1, "distance_candidates must be 1D [D]");
    TORCH_CHECK(distance_map.dim() == 2, "distance_map must be 2D [H, W]");
    
    TORCH_CHECK(cost_volume.size(0) == candidate_count, "cost_volume depth mismatch");
    TORCH_CHECK(distance_candidates.size(0) == candidate_count, "distance_candidates size mismatch");
    
    // Get device pointers
    const float* d_cost = get_device_ptr<float>(cost_volume);
    const float* d_distances = get_device_ptr<float>(distance_candidates);
    float* d_distance_map = get_device_ptr<float>(distance_map);
    
    // Calculate grid and block sizes
    int total_pixels = rows * cols;
    int blockSize = 256;
    int gridSize = (total_pixels + blockSize - 1) / blockSize;
    
    // Launch kernel
    quadraticFittingKernel<<<gridSize, blockSize>>>(
        d_cost, d_distances, d_distance_map,
        candidate_count, rows, cols
    );
    
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
