/**
=======================================================================
GPU-Accelerated Utilities - CUDA Kernel Implementation
Optimized for NVIDIA Jetson AGX Orin (sm_87, JetPack 6.0)
=======================================================================
*/

#include "utils_kernel.cuh"
#include <stdio.h>

// ============================================================================
// CUDA Error Checking Macros
// ============================================================================

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
    } \
}

// ============================================================================
// Kernel: Unproject Pixels to 3D Points
// ============================================================================

/**
 * Grid: (ceil(W/32), ceil(H/32))
 * Block: (32, 32) = 1024 threads per block (optimal for Jetson AGX Orin)
 * 
 * Memory layout:
 * - uv_in:     [H*W*2] float (contiguous: x0, y0, x1, y1, ...)
 * - points_out: [H*W*3] float (contiguous: x0, y0, z0, x1, y1, z1, ...)
 * - valid_out: [H*W] uint8
 */
__global__ void kernel_unproject(
    const float* uv_in,
    const CameraCalibration* calib,
    float* points_out,
    uint8_t* valid_out,
    int width,
    int height
) {
    // Global thread ID
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Linear index
    int idx = y * width + x;
    
    // Load UV coordinates
    float2 uv;
    uv.x = uv_in[idx * 2 + 0];
    uv.y = uv_in[idx * 2 + 1];
    
    // Unproject using Double Sphere model (pass calib by value)
    float3 point;
    char valid;
    unproject_double_sphere(uv, *calib, point, valid);  // ← Dereferencing pointer to pass by value
    
    // Write outputs
    points_out[idx * 3 + 0] = point.x;
    points_out[idx * 3 + 1] = point.y;
    points_out[idx * 3 + 2] = point.z;
    valid_out[idx] = valid;
}

// ============================================================================
// Kernel: Project 3D Points to Pixels
// ============================================================================

__global__ void kernel_project(
    const float* points_in,
    const CameraCalibration* calib,
    float* uv_out,
    uint8_t* valid_out,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Load 3D point
    float3 point;
    point.x = points_in[idx * 3 + 0];
    point.y = points_in[idx * 3 + 1];
    point.z = points_in[idx * 3 + 2];
    
    // Project
    float2 uv;
    char valid;
    project_double_sphere(point, *calib, uv, valid);
    
    // Write outputs
    uv_out[idx * 2 + 0] = uv.x;
    uv_out[idx * 2 + 1] = uv.y;
    valid_out[idx] = valid;
}

// ============================================================================
// Kernel: RGB to YCbCr Conversion
// ============================================================================

/**
 * Grid: (ceil(W/32), ceil(H/32))
 * Block: (32, 32) = 1024 threads per block
 * 
 * Processes RGB image (uint8, 3-channel) to YCbCr
 * - rgb_in:     [H*W*3] uint8 (RGB interleaved)
 * - ycbcr_out: [H*W*3] uint8 (YCbCr interleaved)
 */
__global__ void kernel_rgb2ycbcr(
    const uint8_t* rgb_in,
    uint8_t* ycbcr_out,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * 3;
    
    // Load RGB pixel
    uchar3 rgb = {rgb_in[idx], rgb_in[idx + 1], rgb_in[idx + 2]};
    
    // Convert to YCbCr
    uchar3 ycbcr = rgb2ycbcr(rgb);
    
    // Write YCbCr pixel
    ycbcr_out[idx]     = ycbcr.x;
    ycbcr_out[idx + 1] = ycbcr.y;
    ycbcr_out[idx + 2] = ycbcr.z;
}

// ============================================================================
// Kernel: Bilinear Resampling via Texture Objects
// ============================================================================

/**
 * Uses CUDA texture objects for hardware-accelerated bilinear interpolation
 * Grid: (ceil(out_W/32), ceil(out_H/32))
 * Block: (32, 32)
 * 
 * @param tex: Texture object bound to input image
 * @param output: Output buffer [out_H*out_W*3] float (RGB)
 * @param sample_coords: Normalized coordinates [out_H*out_W*2] float ∈ [0, 1]
/**
 * Bilinear Resampling Kernel
 * @param d_image: Input image [H*W*3] (uint8 RGB)
 * @param image_width, image_height: Input image dimensions
 * @param d_coords: Normalized sampling coordinates [out_H*out_W*2] in [0,1]
 * @param d_output: Output image [out_H*out_W*3] (uint8 RGB)
 * @param output_width, output_height: Output image dimensions
 */
__global__ void kernel_resample_bilinear(
    const uint8_t* d_image,
    int image_width,
    int image_height,
    const float* d_coords,
    uint8_t* d_output,
    int output_width,
    int output_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= output_width || y >= output_height) return;
    
    int out_idx = (y * output_width + x);
    int coord_idx = out_idx * 2;
    
    // Load normalized sampling coordinates [0, 1]
    float u = d_coords[coord_idx];
    float v = d_coords[coord_idx + 1];
    
    // Convert to pixel coordinates
    float px = u * (image_width - 1);
    float py = v * (image_height - 1);
    
    // Get integer and fractional parts
    int ix = (int)px;
    int iy = (int)py;
    float fx = px - ix;
    float fy = py - iy;
    
    // Clamp to valid range
    ix = max(0, min(ix, image_width - 2));
    iy = max(0, min(iy, image_height - 2));
    
    // Bilinear interpolation for each channel
    for (int c = 0; c < 3; ++c) {
        int idx00 = (iy * image_width + ix) * 3 + c;
        int idx10 = (iy * image_width + (ix + 1)) * 3 + c;
        int idx01 = ((iy + 1) * image_width + ix) * 3 + c;
        int idx11 = ((iy + 1) * image_width + (ix + 1)) * 3 + c;
        
        uint8_t v00 = d_image[idx00];
        uint8_t v10 = d_image[idx10];
        uint8_t v01 = d_image[idx01];
        uint8_t v11 = d_image[idx11];
        
        float val0 = v00 * (1 - fx) + v10 * fx;
        float val1 = v01 * (1 - fx) + v11 * fx;
        float val = val0 * (1 - fy) + val1 * fy;
        
        d_output[out_idx * 3 + c] = (uint8_t)roundf(val);
    }
}

// ============================================================================
// Wrapper Functions for Host-side Calls
// ============================================================================

/**
 * Host wrapper for unproject kernel
 * Allocates temporary GPU memory, launches kernel, and returns results
 */
extern "C" {

cudaError_t launch_unproject_kernel(
    const float* d_uv,
    const void* d_calib,
    float* d_points,
    uint8_t* d_valid,
    int width,
    int height
) {
    // Grid and block dimensions optimized for Jetson AGX Orin
    dim3 block(32, 32, 1);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y,
        1
    );
    
    kernel_unproject<<<grid, block>>>(
        d_uv, (const CameraCalibration*)d_calib, d_points, d_valid, width, height
    );
    
    return cudaGetLastError();
}

cudaError_t launch_project_kernel(
    const float* d_points,
    const void* d_calib,
    float* d_uv,
    uint8_t* d_valid,
    int width,
    int height
) {
    dim3 block(32, 32, 1);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y,
        1
    );
    
    kernel_project<<<grid, block>>>(
        d_points, (const CameraCalibration*)d_calib, d_uv, d_valid, width, height
    );
    
    return cudaGetLastError();
}

cudaError_t launch_rgb2ycbcr_kernel(
    const uint8_t* d_rgb,
    uint8_t* d_ycbcr,
    int width,
    int height
) {
    dim3 block(32, 32, 1);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y,
        1
    );
    
    kernel_rgb2ycbcr<<<grid, block>>>(d_rgb, d_ycbcr, width, height);
    
    return cudaGetLastError();
}

cudaError_t launch_resample_bilinear_kernel(
    const uint8_t* d_image,
    const float* d_coords,
    uint8_t* d_output,
    int image_width,
    int image_height,
    int channels,
    int output_width,
    int output_height
) {
    dim3 block(32, 32, 1);
    dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        1
    );
    
    kernel_resample_bilinear<<<grid, block>>>(
        d_image, image_width, image_height,
        d_coords, d_output, output_width, output_height
    );
    
    return cudaGetLastError();
}

} // extern "C"
