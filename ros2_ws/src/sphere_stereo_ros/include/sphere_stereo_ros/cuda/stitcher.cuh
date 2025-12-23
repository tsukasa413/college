/**
 * @file stitcher.cuh
 * @brief CUDA kernel declarations for panorama stitching
 * 
 * Ported from Python implementation for ROS 2 / C++ project.
 * Based on: "Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images"
 * Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
 * CVPR 2021
 * 
 * @copyright CC BY-NC-SA 3.0
 */

#ifndef SPHERE_STEREO_ROS_STITCHER_CUH
#define SPHERE_STEREO_ROS_STITCHER_CUH

#include <cuda_runtime.h>

namespace sphere_stereo_ros {
namespace cuda {

// =============================================================================
// Data Structures
// =============================================================================

/**
 * @brief Double Sphere camera intrinsic parameters
 */
struct Intrinsics {
    float2 fl;         ///< Focal length (fx, fy)
    float2 principal;  ///< Principal point (cx, cy)
    float xi;          ///< Double sphere xi parameter
    float alpha;       ///< Double sphere alpha parameter
};

/**
 * @brief 3x3 Rotation matrix (row-major)
 */
struct Rotation {
    float r[3][3];
};

/**
 * @brief Stitcher configuration parameters
 */
struct StitcherConfig {
    int pano_cols;           ///< Panorama width
    int pano_rows;           ///< Panorama height
    int fisheye_cols;        ///< Fisheye image width (matching resolution)
    int fisheye_rows;        ///< Fisheye image height (matching resolution)
    int stitch_cols;         ///< Stitching image width (may differ from fisheye)
    int stitch_rows;         ///< Stitching image height
    int num_references;      ///< Number of reference cameras
    float min_dist;          ///< Minimum expected distance
    float max_dist;          ///< Maximum expected distance
};

// =============================================================================
// Kernel Launch Functions (Host-side)
// =============================================================================

/**
 * @brief Reproject distance map from camera view to reference viewpoint
 * 
 * Uses z-buffering to handle occlusions. Should be called twice:
 * 1. Initialize distance_out to a large value (e.g., max_dist + 1)
 * 2. Call this kernel twice to fill in values
 * 
 * @param distance_in Input distance map at camera viewpoint [rows x cols]
 * @param distance_out Output distance map at reference viewpoint [rows x cols]
 * @param intrinsics Camera intrinsic parameters
 * @param translation Translation from reference to camera (float3)
 * @param cols Image width
 * @param rows Image height
 * @param stream CUDA stream for async execution
 */
void launchReprojectDistanceKernel(
    const float* distance_in,
    float* distance_out,
    const Intrinsics* intrinsics,
    const float3* translation,
    int cols,
    int rows,
    cudaStream_t stream = 0);

/**
 * @brief Create inpainting weights based on occlusion direction
 * 
 * Computes the optimal neighbor pixels for inpainting based on the 
 * direction from near to far projections.
 * 
 * @param inpaint_weights Output inpainting weights [rows x cols], uchar2 encoded
 * @param intrinsics Camera intrinsic parameters
 * @param translation Translation from reference to camera
 * @param cols Image width
 * @param rows Image height
 * @param min_dist Minimum distance
 * @param max_dist Maximum distance
 * @param stream CUDA stream
 */
void launchCreateInpaintingWeightsKernel(
    uchar2* inpaint_weights,
    const Intrinsics* intrinsics,
    const float3* translation,
    int cols,
    int rows,
    float min_dist,
    float max_dist,
    cudaStream_t stream = 0);

/**
 * @brief Apply inpainting to fill holes in reprojected distance map
 * 
 * Uses precomputed inpainting weights to propagate valid distance values
 * to neighboring invalid pixels. Should be called iteratively.
 * 
 * @param distance_map Distance map with holes (in/out) [rows x cols]
 * @param inpaint_weights Precomputed inpainting weights [rows x cols]
 * @param cols Image width
 * @param rows Image height
 * @param max_dist Maximum distance (pixels >= max_dist + 0.1 are invalid)
 * @param stream CUDA stream
 */
void launchInpaintKernel(
    float* distance_map,
    const uchar2* inpaint_weights,
    int cols,
    int rows,
    float max_dist,
    cudaStream_t stream = 0);

/**
 * @brief Create blending lookup tables for panorama stitching
 * 
 * Computes sampling locations and blending weights for each panorama pixel
 * based on warp-aware blending.
 * 
 * @param sampling_lut Output sampling locations [num_refs x pano_rows x pano_cols x 2]
 * @param blending_weights Output blending weights [num_refs x pano_rows x pano_cols]
 * @param masks Input validity masks [num_refs x rows x cols]
 * @param intrinsics Camera intrinsics array [num_refs]
 * @param rotations Camera rotations array [num_refs]
 * @param translations Camera translations array [num_refs]
 * @param config Stitcher configuration
 * @param stream CUDA stream
 */
void launchCreateBlendingLutKernel(
    float2* sampling_lut,
    float* blending_weights,
    const float* masks,
    const Intrinsics* intrinsics,
    const Rotation* rotations,
    const float3* translations,
    const StitcherConfig& config,
    cudaStream_t stream = 0);

/**
 * @brief Merge fisheye images into RGB-D panorama
 * 
 * Uses precomputed lookup tables and blending weights to stitch
 * multiple fisheye images and distance maps into a panorama.
 * 
 * @param sampling_lut Sampling locations [num_refs x pano_rows x pano_cols x 2]
 * @param blending_weights Blending weights [num_refs x pano_rows x pano_cols]
 * @param reprojected_distances Reprojected distance maps [num_refs x rows x cols]
 * @param distance_maps Original distance maps [num_refs x rows x cols]
 * @param stitch_images Fisheye images for stitching [num_refs x stitch_rows x stitch_cols x 3]
 * @param translations Camera translations [num_refs]
 * @param intrinsics Camera intrinsics [num_refs]
 * @param distance_panorama Output distance panorama [pano_rows x pano_cols]
 * @param rgb_panorama Output RGB panorama [pano_rows x pano_cols x 3]
 * @param config Stitcher configuration
 * @param stream CUDA stream
 */
void launchMergeRGBDPanoramaKernel(
    const float2* sampling_lut,
    const float* blending_weights,
    const float* reprojected_distances,
    const float* distance_maps,
    const uchar3* stitch_images,
    const float3* translations,
    const Intrinsics* intrinsics,
    float* distance_panorama,
    uchar3* rgb_panorama,
    const StitcherConfig& config,
    cudaStream_t stream = 0);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Fill a float array with a constant value
 */
void launchFillKernel(
    float* data,
    float value,
    int count,
    cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace sphere_stereo_ros

#endif  // SPHERE_STEREO_ROS_STITCHER_CUH
