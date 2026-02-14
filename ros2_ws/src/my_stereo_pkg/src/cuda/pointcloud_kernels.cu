/**
 * CUDA Kernels for Point Cloud Generation
 */

#include <cuda_runtime.h>
#include <cmath>

namespace my_stereo_pkg {

/**
 * CUDA kernel to convert panorama RGB-D to 3D points
 * 
 * Maps each pixel (u, v) to 3D coordinates (x, y, z) using:
 *   θ = 2π * u / width          (longitude)
 *   φ = π * (0.5 - v / height)  (latitude, -π/2 to π/2)
 *   
 * ROS standard coordinate system (right-handed):
 *   x = d * cos(φ) * sin(θ)  (forward)
 *   y = d * cos(φ) * cos(θ)  (left)
 *   z = d * sin(φ)           (up)
 */
__global__ void panoramaToPointCloudKernel(
    const float* __restrict__ distance,      // [H, W]
    const unsigned char* __restrict__ rgb,   // [H, W, 3]
    const float* __restrict__ cos_lat,       // [H] - precomputed cos(φ)
    const float* __restrict__ sin_lat,       // [H] - precomputed sin(φ)
    float* __restrict__ points,              // [H*W, 6] output
    bool* __restrict__ valid_mask,           // [H*W] output
    int width,
    int height,
    float min_depth,
    float max_depth
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (u >= width || v >= height) return;
    
    int pixel_idx = v * width + u;
    
    // Get distance value
    float d = distance[pixel_idx];
    
    // Check depth validity
    if (d < min_depth || d > max_depth || !isfinite(d)) {
        valid_mask[pixel_idx] = false;
        return;
    }
    
    valid_mask[pixel_idx] = true;
    
    // Calculate longitude θ (0 to 2π)
    float theta = (2.0f * M_PI * u) / width;
    
    // Get precomputed latitude values
    float cos_phi = cos_lat[v];
    float sin_phi = sin_lat[v];
    
    // Calculate sin(θ) and cos(θ)
    float sin_theta = sinf(theta);
    float cos_theta = cosf(theta);
    
    // Convert to 3D coordinates (ROS standard: x=forward, y=left, z=up)
    float x = d * cos_phi * sin_theta;
    float y = d * cos_phi * cos_theta;  // Left direction
    float z = d * sin_phi;              // Up direction
    
    // Write to output (x, y, z)
    points[pixel_idx * 6 + 0] = x;
    points[pixel_idx * 6 + 1] = y;
    points[pixel_idx * 6 + 2] = z;
    
    // Write RGB values
    int rgb_idx = pixel_idx * 3;
    points[pixel_idx * 6 + 3] = rgb[rgb_idx + 0];  // R
    points[pixel_idx * 6 + 4] = rgb[rgb_idx + 1];  // G
    points[pixel_idx * 6 + 5] = rgb[rgb_idx + 2];  // B
}

// Wrapper function to launch kernel from C++
void launchPanoramaToPointCloudKernel(
    const float* distance,
    const unsigned char* rgb,
    const float* cos_lat,
    const float* sin_lat,
    float* points,
    bool* valid_mask,
    int width,
    int height,
    float min_depth,
    float max_depth,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
    
    panoramaToPointCloudKernel<<<grid, block, 0, stream>>>(
        distance,
        rgb,
        cos_lat,
        sin_lat,
        points,
        valid_mask,
        width,
        height,
        min_depth,
        max_depth
    );
}

} // namespace my_stereo_pkg
