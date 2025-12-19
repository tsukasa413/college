/**
 * @file vec_utils.cuh
 * @brief CUDA vector utility functions for sphere stereo vision
 * 
 * Ported from Python implementation for ROS 2 / C++ project.
 * Based on: "Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images"
 * Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
 * CVPR 2021
 * 
 * @copyright CC BY-NC-SA 3.0
 */

#ifndef SPHERE_STEREO_ROS_VEC_UTILS_CUH
#define SPHERE_STEREO_ROS_VEC_UTILS_CUH

#include <cuda_runtime.h>
#include <cmath>

namespace sphere_stereo_ros {
namespace cuda {

// =============================================================================
// Type Definitions
// =============================================================================
using uchar = unsigned char;

// =============================================================================
// float3 Operations
// =============================================================================

/**
 * @brief Compute the Euclidean length of a float3 vector
 */
__device__ __forceinline__ float length(float3 a)
{
    return norm3df(a.x, a.y, a.z);
}

/**
 * @brief Scalar multiplication: scalar * float3
 */
__device__ __forceinline__ float3 operator*(float b, float3 a)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

/**
 * @brief Scalar multiplication: float3 * scalar
 */
__device__ __forceinline__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

/**
 * @brief Vector addition: float3 + float3
 */
__device__ __forceinline__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/**
 * @brief Vector subtraction: float3 - float3
 */
__device__ __forceinline__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

/**
 * @brief Scalar division: float3 / scalar
 */
__device__ __forceinline__ float3 operator/(float3 a, float b)
{
    float inv = 1.0f / b;
    return make_float3(a.x * inv, a.y * inv, a.z * inv);
}

/**
 * @brief Dot product of two float3 vectors
 */
__device__ __forceinline__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
 * @brief Cross product of two float3 vectors
 */
__device__ __forceinline__ float3 cross(float3 a, float3 b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

/**
 * @brief Normalize a float3 vector
 */
__device__ __forceinline__ float3 normalize(float3 a)
{
    float inv_len = rsqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
    return make_float3(a.x * inv_len, a.y * inv_len, a.z * inv_len);
}

/**
 * @brief Sum of absolute values of float3 components
 */
__device__ __forceinline__ float absSum(float3 a)
{
    return fabsf(a.x) + fabsf(a.y) + fabsf(a.z);
}

// =============================================================================
// float2 Operations
// =============================================================================

/**
 * @brief Compute the Euclidean length of a float2 vector
 */
__device__ __forceinline__ float length(float2 a)
{
    return hypotf(a.x, a.y);
}

/**
 * @brief Dot product of two float2 vectors
 */
__device__ __forceinline__ float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}

/**
 * @brief Element-wise multiplication: float2 * float2
 */
__device__ __forceinline__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}

/**
 * @brief Scalar multiplication: float2 * scalar
 */
__device__ __forceinline__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}

/**
 * @brief Scalar multiplication: scalar * float2
 */
__device__ __forceinline__ float2 operator*(float b, float2 a)
{
    return make_float2(a.x * b, a.y * b);
}

/**
 * @brief Vector addition: float2 + float2
 */
__device__ __forceinline__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

/**
 * @brief Vector subtraction: float2 - float2
 */
__device__ __forceinline__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

/**
 * @brief Element-wise division: float2 / float2
 */
__device__ __forceinline__ float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}

/**
 * @brief Scalar division: float2 / scalar
 */
__device__ __forceinline__ float2 operator/(float2 a, float b)
{
    float inv = 1.0f / b;
    return make_float2(a.x * inv, a.y * inv);
}

/**
 * @brief Normalize a float2 vector
 */
__device__ __forceinline__ float2 normalize(float2 a)
{
    float inv_len = rsqrtf(a.x * a.x + a.y * a.y);
    return make_float2(a.x * inv_len, a.y * inv_len);
}

// =============================================================================
// int2 Operations
// =============================================================================

/**
 * @brief Vector addition: int2 + int2
 */
__device__ __forceinline__ int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}

/**
 * @brief Scalar subtraction: int2 - scalar
 */
__device__ __forceinline__ int2 operator-(int2 a, int b)
{
    return make_int2(a.x - b, a.y - b);
}

/**
 * @brief Vector subtraction: int2 - int2
 */
__device__ __forceinline__ int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}

// =============================================================================
// Type Conversion Utilities
// =============================================================================

/**
 * @brief Convert uchar3 to float3
 */
__device__ __forceinline__ float3 uchar3ToFloat3(uchar3 a)
{
    return make_float3(
        static_cast<float>(a.x),
        static_cast<float>(a.y),
        static_cast<float>(a.z)
    );
}

/**
 * @brief Convert float3 to uchar3 (with clamping)
 */
__device__ __forceinline__ uchar3 float3ToUchar3(float3 a)
{
    return make_uchar3(
        static_cast<uchar>(fminf(fmaxf(a.x, 0.0f), 255.0f)),
        static_cast<uchar>(fminf(fmaxf(a.y, 0.0f), 255.0f)),
        static_cast<uchar>(fminf(fmaxf(a.z, 0.0f), 255.0f))
    );
}

/**
 * @brief Convert float3 to uchar3 (without clamping, assumes valid range)
 */
__device__ __forceinline__ uchar3 float3ToUchar3Fast(float3 a)
{
    return make_uchar3(
        static_cast<uchar>(a.x),
        static_cast<uchar>(a.y),
        static_cast<uchar>(a.z)
    );
}

// =============================================================================
// Matrix Operations (3x3)
// =============================================================================

/**
 * @brief Multiply a 3x3 matrix (row-major) by a float3 vector
 */
__device__ __forceinline__ float3 matMul3x3(const float r[3][3], float3 v)
{
    return make_float3(
        r[0][0] * v.x + r[0][1] * v.y + r[0][2] * v.z,
        r[1][0] * v.x + r[1][1] * v.y + r[1][2] * v.z,
        r[2][0] * v.x + r[2][1] * v.y + r[2][2] * v.z
    );
}

/**
 * @brief Multiply a 3x3 matrix (as flat array, row-major) by a float3 vector
 */
__device__ __forceinline__ float3 matMul3x3Flat(const float* r, float3 v)
{
    return make_float3(
        r[0] * v.x + r[1] * v.y + r[2] * v.z,
        r[3] * v.x + r[4] * v.y + r[5] * v.z,
        r[6] * v.x + r[7] * v.y + r[8] * v.z
    );
}

// =============================================================================
// Clamping Utilities
// =============================================================================

/**
 * @brief Clamp a float value to [min_val, max_val]
 */
__device__ __forceinline__ float clampf(float val, float min_val, float max_val)
{
    return fminf(fmaxf(val, min_val), max_val);
}

/**
 * @brief Clamp a float2 value component-wise
 */
__device__ __forceinline__ float2 clamp2f(float2 val, float min_val, float max_val)
{
    return make_float2(
        fminf(fmaxf(val.x, min_val), max_val),
        fminf(fmaxf(val.y, min_val), max_val)
    );
}

/**
 * @brief Clamp a float2 value to image bounds [0.1, cols-1.1] x [0.1, rows-1.1]
 */
__device__ __forceinline__ float2 clampToImage(float2 uv, int cols, int rows)
{
    return make_float2(
        fminf(fmaxf(uv.x, 0.1f), static_cast<float>(cols) - 1.1f),
        fminf(fmaxf(uv.y, 0.1f), static_cast<float>(rows) - 1.1f)
    );
}

// =============================================================================
// Constants
// =============================================================================
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;
constexpr float kHalfPi = 1.57079632679489661923f;

}  // namespace cuda
}  // namespace sphere_stereo_ros

#endif  // SPHERE_STEREO_ROS_VEC_UTILS_CUH
