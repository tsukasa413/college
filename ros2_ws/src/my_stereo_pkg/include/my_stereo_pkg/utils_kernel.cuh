/**
=======================================================================
GPU-Accelerated Utilities for Sphere Sweeping Stereo
Based on Python utils.py from CVPR 2021 implementation
Optimized for NVIDIA Jetson AGX Orin (JetPack 6.0)

Author: Optimized Implementation
License: CC BY-NC-SA 3.0 (matching sphere-stereo)
=======================================================================
*/

#ifndef UTILS_KERNEL_CUH
#define UTILS_KERNEL_CUH

#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>

// ============================================================================
// Double Sphere Camera Model Structures
// ============================================================================

struct Intrinsics {
    float fx, fy;       // Focal lengths
    float cx, cy;       // Principal point
    float xi;           // 1st sphere parameter
    float alpha;        // 2nd sphere parameter
};

struct CameraExtrinsics {
    float R[9];         // Rotation matrix [3x3] row-major
    float t[3];         // Translation vector
};

struct CameraCalibration {
    Intrinsics intrinsics;
    CameraExtrinsics extrinsics;
    float matching_scale;      // Scale factor for matching resolution
    int resolution_x, resolution_y;  // Original image resolution
};

// ============================================================================
// Device-side Helper Functions (Math Operations)
// ============================================================================

// Note: make_float3, make_float2 are already defined by CUDA
// Use inline operations instead

__device__ inline float length_f3(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ inline float3 add_f3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 sub_f3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 scale_f3(float3 v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

__device__ inline float dot_f3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// 2D Vector Operations use CUDA built-in make_float2
__device__ inline float length_f2(float2 v) {
    return sqrtf(v.x * v.x + v.y * v.y);
}

__device__ inline float2 add_f2(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ inline float2 sub_f2(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ inline float2 scale_f2(float2 v, float s) {
    return make_float2(v.x * s, v.y * s);
}

// ============================================================================
// Double Sphere Model Implementation
// ============================================================================

/**
 * Unproject pixel coordinates to 3D unit sphere
 * Based on Double Sphere Camera Model (https://arxiv.org/abs/1807.08957)
 * 
 * @param uv: Pixel coordinates (in matching resolution space)
 * @param calib: Camera calibration with intrinsics and scale
 * @param point_out: Output 3D point on sphere
 * @param valid_out: Valid flag (1 if valid, 0 otherwise)
 */
__device__ inline void unproject_double_sphere(
    float2 uv,
    const CameraCalibration calib,  // 値渡しに変更
    float3& point_out,
    char& valid_out
) {
    // Apply matching scale to intrinsics
    float2 fl = {calib.intrinsics.fx * calib.matching_scale,
                 calib.intrinsics.fy * calib.matching_scale};
    float2 principal = {calib.intrinsics.cx * calib.matching_scale,
                        calib.intrinsics.cy * calib.matching_scale};
    
    // Compute normalized image coordinates
    float2 m_xy = scale_f2(sub_f2(uv, principal), 1.0f / fl.x);
    m_xy.y = (uv.y - principal.y) / fl.y;
    
    // Compute radius squared
    float r2 = m_xy.x * m_xy.x + m_xy.y * m_xy.y;
    
    // Check validity constraint
    float discriminant = 1.0f - (2.0f * calib.intrinsics.alpha - 1.0f) * r2;
    valid_out = (discriminant >= 0) ? 1 : 0;
    
    if (valid_out == 0) {
        point_out = {0, 0, 0};
        return;
    }
    
    // Compute z-coordinate on first sphere
    float sqrt_term = sqrtf(fmaxf(discriminant, 0.0f));
    float denom = calib.intrinsics.alpha * sqrt_term + 1.0f - calib.intrinsics.alpha;
    float m_z = (1.0f - calib.intrinsics.alpha * calib.intrinsics.alpha * r2) / denom;
    
    // Project to 3D
    float3 point = {m_xy.x, m_xy.y, m_z};
    
    // Apply interspherical scaling
    float d_z = dot_f3(point, point);
    float coeff = (m_z * calib.intrinsics.xi + sqrtf(m_z * m_z + (1.0f - calib.intrinsics.xi * calib.intrinsics.xi) * r2)) 
                  / (d_z + 1e-10f);
    
    point = scale_f3(point, coeff);
    point.z -= calib.intrinsics.xi;
    
    point_out = point;
}

/**
 * Project 3D point to pixel coordinates
 * Based on Double Sphere Camera Model
 * 
 * @param point: 3D point in camera frame
 * @param calib: Camera calibration
 * @param uv_out: Output pixel coordinates
 * @param valid_out: Valid flag
 */
__device__ inline void project_double_sphere(
    float3 point,
    const CameraCalibration calib,  // 値渡しに変更
    float2& uv_out,
    char& valid_out
) {
    float d1 = length_f3(point);
    
    float c = calib.intrinsics.xi * d1 + point.z;
    float3 norm_vec = {point.x, point.y, c};
    float d2 = length_f3(norm_vec);
    
    float norm_denom = calib.intrinsics.alpha * d2 + (1.0f - calib.intrinsics.alpha) * c + 1e-10f;
    
    // Compute visibility weight
    float w1 = (calib.intrinsics.alpha > 0.5f) 
               ? (1.0f - calib.intrinsics.alpha) / calib.intrinsics.alpha
               : calib.intrinsics.alpha / (1.0f - calib.intrinsics.alpha);
    float w2 = (w1 + calib.intrinsics.xi) / sqrtf(2.0f * w1 * calib.intrinsics.xi 
                                                    + calib.intrinsics.xi * calib.intrinsics.xi + 1.0f);
    
    valid_out = (point.z > -w2 * d1) ? 1 : 0;
    
    // Apply matching scale to intrinsics
    float2 fl = {calib.intrinsics.fx * calib.matching_scale,
                 calib.intrinsics.fy * calib.matching_scale};
    float2 principal = {calib.intrinsics.cx * calib.matching_scale,
                        calib.intrinsics.cy * calib.matching_scale};
    
    // Project to image plane
    float2 proj = {fl.x * point.x / norm_denom, fl.y * point.y / norm_denom};
    uv_out = add_f2(proj, principal);
}

/**
 * Apply rotation matrix to 3D point
 * @param R: Rotation matrix (3x3, row-major)
 * @param p: Input point
 * @return Rotated point
 */
__device__ inline float3 apply_rotation(const float* R, float3 p) {
    float3 result;
    result.x = R[0] * p.x + R[1] * p.y + R[2] * p.z;
    result.y = R[3] * p.x + R[4] * p.y + R[5] * p.z;
    result.z = R[6] * p.x + R[7] * p.y + R[8] * p.z;
    return result;
}

/**
 * Apply camera extrinsics (rotation + translation)
 */
__device__ inline float3 apply_extrinsics(float3 point, const CameraExtrinsics& ext) {
    float3 rotated = apply_rotation(ext.R, point);
    return {rotated.x + ext.t[0], rotated.y + ext.t[1], rotated.z + ext.t[2]};
}

// ============================================================================
// Color Space Conversion (RGB <-> YCbCr)
// ============================================================================

/**
 * Convert RGB to YCbCr (BT.601 standard)
 * Input: RGB values in [0, 255]
 * Output: YCbCr with Y in [16, 235], CbCr in [16, 240]
 * 
 * @param rgb: Input RGB pixel (uint8x3)
 * @return YCbCr pixel (uint8x3)
 */
__device__ inline uchar3 rgb2ycbcr(uchar3 rgb) {
    float r = rgb.x;
    float g = rgb.y;
    float b = rgb.z;
    
    float y  = fmaxf(16.0f, fminf(235.0f, 16.0f + 0.1826f * r + 0.6142f * g + 0.062f * b));
    float cb = fmaxf(16.0f, fminf(240.0f, 128.0f - 0.1006f * r - 0.3386f * g + 0.4392f * b));
    float cr = fmaxf(16.0f, fminf(240.0f, 128.0f + 0.4392f * r - 0.3989f * g - 0.0403f * b));
    
    return {(unsigned char)y, (unsigned char)cb, (unsigned char)cr};
}

/**
 * Convert single float RGB to YCbCr (for normalized [0,1] input)
 */
__device__ inline float3 rgb2ycbcr_float(float3 rgb) {
    rgb.x *= 255.0f;
    rgb.y *= 255.0f;
    rgb.z *= 255.0f;
    
    float y  = fmaxf(16.0f, fminf(235.0f, 16.0f + 0.1826f * rgb.x + 0.6142f * rgb.y + 0.062f * rgb.z));
    float cb = fmaxf(16.0f, fminf(240.0f, 128.0f - 0.1006f * rgb.x - 0.3386f * rgb.y + 0.4392f * rgb.z));
    float cr = fmaxf(16.0f, fminf(240.0f, 128.0f + 0.4392f * rgb.x - 0.3989f * rgb.y - 0.0403f * rgb.z));
    
    return {y / 255.0f, cb / 255.0f, cr / 255.0f};
}

// ============================================================================
// Kernel Function Declarations
// ============================================================================

/**
 * Unproject kernel: Convert pixel coordinates to 3D points
 * Grid: (W, H), Block: (32, 32)
 * 
 * @param uv: Input UV coordinates [H, W, 2] float32
 * @param calib: Device pointer to calibration (constant memory)
 * @param points_out: Output 3D points [H, W, 3] float32
 * @param valid_out: Output valid mask [H, W] uint8
 * @param width, height: Image dimensions
 */
__global__ void kernel_unproject(
    const float* uv_in,
    const CameraCalibration* calib,
    float* points_out,
    uint8_t* valid_out,
    int width,
    int height
);

/**
 * Project kernel: Convert 3D points to pixel coordinates
 */
__global__ void kernel_project(
    const float* points_in,
    const CameraCalibration* calib,
    float* uv_out,
    uint8_t* valid_out,
    int width,
    int height
);

/**
 * RGB to YCbCr conversion kernel
 * Processes entire image in-place or to output buffer
 */
__global__ void kernel_rgb2ycbcr(
    const uint8_t* rgb_in,
    uint8_t* ycbcr_out,
    int width,
    int height
);

/**
 * Bilinear resampling kernel using texture coordinates
 * Applies fisheye-to-panorama mapping via texture interpolation
 */
__global__ void kernel_resample_bilinear(
    cudaTextureObject_t tex,
    float* output,
    const float2* sample_coords,
    int width,
    int height,
    int out_width,
    int out_height
);

// ============================================================================
// Host-callable Wrapper Functions (defined in utils_kernel.cu)
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

void launch_unproject_kernel(
    const float* d_uv,
    const CameraCalibration* d_calib,
    float* d_points,
    uint8_t* d_valid,
    int width, int height
);

void launch_project_kernel(
    const float* d_points,
    const CameraCalibration* d_calib,
    float* d_uv,
    uint8_t* d_valid,
    int width, int height
);

void launch_rgb2ycbcr_kernel(
    const uint8_t* d_rgb,
    uint8_t* d_ycbcr,
    int width, int height
);

void launch_resample_bilinear_kernel(
    const uint8_t* d_image,
    const float* d_coords,
    uint8_t* d_output,
    int image_width, int image_height, int channels,
    int output_width, int output_height
);

#ifdef __cplusplus
}
#endif

#endif // UTILS_KERNEL_CUH
