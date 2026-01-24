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
    float matching_scale;       // Single scale factor (now scalar, not float2)
    int2 original_resolution; // Original image resolution [width, height]
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
    const CameraCalibration calib,    // ← VALUE PASSING
    float3& point_out,
    char& valid_out
) {
    const Intrinsics& intr_ref = calib.intrinsics;
    
    // Normalized image coordinates
    float mx = (uv.x - intr_ref.cx) / intr_ref.fx;
    float my = (uv.y - intr_ref.cy) / intr_ref.fy;
    
    // Radius in normalized image space
    float r2 = mx * mx + my * my;
    
    // Double Sphere model inverse (matching CPU reference)
    float alpha = intr_ref.alpha;
    float xi = intr_ref.xi;
    
    // Validity check
    float denom = alpha * r2 + 1.0f - (2.0f * alpha - 1.0f) * r2 * xi;
    if (denom < 0.0001f) {
        point_out = make_float3(0, 0, 0);
        valid_out = 0;
        return;
    }
    
    // Compute z on first sphere
    float numerator = 1.0f - xi * xi * r2;
    float z_sphere = numerator / denom;
    
    // 3D point
    float x_3d = mx * z_sphere;
    float y_3d = my * z_sphere;
    float z_3d = z_sphere - xi;
    
    point_out = make_float3(x_3d, y_3d, z_3d);
    valid_out = 1;
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
    const CameraCalibration calib,    // ← VALUE PASSING
    float2& uv_out,
    char& valid_out
) {
    const Intrinsics& intr = calib.intrinsics;
    
    // Translate by xi (second sphere center)
    float z_shifted = point.z + intr.xi;
    
    // Project to first sphere
    float r_xy2 = point.x * point.x + point.y * point.y;
    float r = sqrtf(r_xy2 + z_shifted * z_shifted);
    
    if (r < 0.0001f) {
        uv_out = make_float2(intr.cx, intr.cy);
        valid_out = 0;
        return;
    }
    
    // Project to second sphere (image plane)
    float alpha = intr.alpha;
    float m = alpha * r + (1.0f - alpha) * z_shifted;
    
    if (m < 0.0001f) {
        uv_out = make_float2(intr.cx, intr.cy);
        valid_out = 0;
        return;
    }
    
    float mx = point.x / m;
    float my = point.y / m;
    
    float u = intr.fx * mx + intr.cx;
    float v = intr.fy * my + intr.cy;
    
    uv_out = make_float2(u, v);
    valid_out = 1;
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

#endif // UTILS_KERNEL_CUH
