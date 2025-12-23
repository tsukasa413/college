/**
 * @file cost_volume.cuh
 * @brief CUDA kernel declarations for cost volume computation
 * 
 * Implements sphere sweeping stereo matching with adaptive camera selection.
 * 
 * Based on: "Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images"
 * Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
 * CVPR 2021
 * 
 * @copyright CC BY-NC-SA 3.0
 */

#ifndef SPHERE_STEREO_ROS_COST_VOLUME_CUH
#define SPHERE_STEREO_ROS_COST_VOLUME_CUH

#include <cuda_runtime.h>
#include "sphere_stereo_ros/cuda/stitcher.cuh"  // For Intrinsics, Rotation

namespace sphere_stereo_ros {
namespace cuda {

// =============================================================================
// Data Structures
// =============================================================================

/**
 * @brief Configuration for cost volume computation
 */
struct CostVolumeConfig {
    int cols;                   ///< Image width (matching resolution)
    int rows;                   ///< Image height (matching resolution)
    int num_depths;             ///< Number of depth/distance candidates
    int num_cameras;            ///< Total number of cameras
    float min_dist;             ///< Minimum distance (meters)
    float max_dist;             ///< Maximum distance (meters)
    float cost_clamp;           ///< Maximum cost value (clamp)
};

// =============================================================================
// Kernel Launch Functions (Host-side)
// =============================================================================

/**
 * @brief Compute selected camera map for adaptive matching
 * 
 * For each pixel, selects the best matching camera based on maximum
 * displacement between near and far projections (Section 3.1 of paper).
 * 
 * @param selected_camera Output: selected camera index per pixel [rows x cols], int
 * @param masks Validity masks for all cameras [num_cameras x rows x cols], float
 * @param intrinsics Camera intrinsics [num_cameras]
 * @param rotations Relative rotations from reference [num_cameras]
 * @param translations Relative translations from reference [num_cameras]
 * @param reference_intrinsics Intrinsics of reference camera
 * @param reference_index Index of the reference camera
 * @param config Cost volume configuration
 * @param stream CUDA stream
 */
void launchSelectCameraKernel(
    int* selected_camera,
    const float* masks,
    const Intrinsics* intrinsics,
    const Rotation* rotations,
    const float3* translations,
    const Intrinsics& reference_intrinsics,
    int reference_index,
    const CostVolumeConfig& config,
    cudaStream_t stream = 0);

/**
 * @brief Compute cost volume using sphere sweeping
 * 
 * For each pixel and each depth candidate, computes the matching cost
 * by reprojecting to the selected camera and computing SAD (Sum of 
 * Absolute Differences) in YCbCr color space.
 * 
 * @param cost_volume Output: cost volume [num_depths x rows x cols], float
 * @param reference_image Reference image (YCbCr) [rows x cols], uchar3 (HWC)
 * @param images All camera images (YCbCr) [num_cameras x rows x cols], uchar3 (HWC)
 * @param selected_camera Selected camera per pixel [rows x cols], int
 * @param intrinsics Camera intrinsics [num_cameras]
 * @param rotations Relative rotations from reference [num_cameras]
 * @param translations Relative translations from reference [num_cameras]
 * @param reference_intrinsics Intrinsics of reference camera
 * @param distance_candidates Pre-computed distance candidates [num_depths], float
 * @param config Cost volume configuration
 * @param stream CUDA stream
 */
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
    cudaStream_t stream = 0);

/**
 * @brief Select minimum cost depth with quadratic sub-pixel refinement
 * 
 * Performs winner-take-all depth selection with quadratic fitting
 * for sub-candidate accuracy (Eq. 5 in paper).
 * 
 * @param distance_map Output: estimated distance map [rows x cols], float
 * @param cost_volume Input: filtered cost volume [num_depths x rows x cols], float
 * @param distance_candidates Pre-computed distance candidates [num_depths], float
 * @param config Cost volume configuration
 * @param stream CUDA stream
 */
void launchSelectDepthKernel(
    float* distance_map,
    const float* cost_volume,
    const float* distance_candidates,
    const CostVolumeConfig& config,
    cudaStream_t stream = 0);

/**
 * @brief Convert RGB image to YCbCr color space
 * 
 * @param output Output YCbCr image [rows x cols], uchar3 (HWC)
 * @param input Input RGB image [rows x cols], uchar3 (HWC)
 * @param cols Image width
 * @param rows Image height
 * @param stream CUDA stream
 */
void launchRgb2YCbCrKernel(
    uchar3* output,
    const uchar3* input,
    int cols,
    int rows,
    cudaStream_t stream = 0);

/**
 * @brief Initialize cost volume with zeros
 * 
 * @param cost_volume Cost volume to clear [num_depths x rows x cols], float
 * @param num_depths Number of depth planes
 * @param cols Image width
 * @param rows Image height
 * @param stream CUDA stream
 */
void launchClearCostVolumeKernel(
    float* cost_volume,
    int num_depths,
    int cols,
    int rows,
    cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace sphere_stereo_ros

#endif  // SPHERE_STEREO_ROS_COST_VOLUME_CUH
