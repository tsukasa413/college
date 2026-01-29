/**
=======================================================================
General Information
-------------------
C++ implementation of the Sphere Sweeping Stereo pipeline from:
Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images
Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
Proc. IEEE Computer Vision and Pattern Recognition (CVPR 2021, Oral)
=======================================================================
**/

#pragma once

#include "my_stereo_pkg/calibration.hpp"
#include "my_stereo_pkg/stitcher.hpp"
#include "my_stereo_pkg/isb_filter.hpp"
#include "my_stereo_pkg/cuda_kernels.hpp"
#include <torch/torch.h>
#include <vector>
#include <memory>

// Forward declarations for CUDA structures
struct DoubleSphereParams;
struct CameraExtrinsics;

namespace my_stereo_pkg {

// Import Calibration and Stitcher from my_stereo namespace
using my_stereo::Calibration;
using my_stereo::Stitcher;

/**
 * Complete RGBD estimation pipeline from fisheye images
 * Implements adaptive spherical matching with hierarchical filtering
 * Matches Python RGBD_Estimator class
 */
class RGBDEstimator {
public:
    /**
     * Constructor
     * @param calibrations Camera calibration parameters for all cameras
     * @param min_dist Minimum distance for sphere sweep volume
     * @param max_dist Maximum distance for sphere sweep volume
     * @param candidate_count Number of distance candidates between min_dist and max_dist
     * @param references_indices Indices of reference cameras for distance estimation
     * @param reprojection_viewpoint Reference viewpoint for RGB-D panorama [x, y, z]
     * @param masks Valid area masks for each camera [num_cameras][H, W]
     * @param matching_resolution Resolution (cols, rows) for matching
     * @param rgb_to_stitch_resolution Resolution (cols, rows) for color stitching
     * @param panorama_resolution Resolution (cols, rows) for output panorama
     * @param sigma_i Edge preservation parameter (lower = preserve edges more)
     * @param sigma_s Smoothing parameter (higher = more smoothing from coarse scales)
     * @param device CUDA device for processing
     */
    RGBDEstimator(
        const std::vector<Calibration>& calibrations,
        float min_dist,
        float max_dist,
        int candidate_count,
        const std::vector<int>& references_indices,
        const at::Tensor& reprojection_viewpoint,
        const std::vector<at::Tensor>& masks,
        const std::pair<int, int>& matching_resolution,
        const std::pair<int, int>& rgb_to_stitch_resolution,
        const std::pair<int, int>& panorama_resolution,
        float sigma_i,
        float sigma_s,
        const at::Device& device
    );

    /**
     * Destructor - Free unified memory buffers
     */
    ~RGBDEstimator();

    /**
     * Execute complete RGBD estimation pipeline
     * @param images_to_match Fisheye images for matching [num_cameras][H, W, 3] float32 [0-255]
     * @param images_to_stitch Fisheye images for stitching [num_refs][H, W, 3] float32 [0-255]
     * @return Pair of (RGB panorama [H, W, 3] uint8, distance panorama [H, W] float32)
     */
    std::pair<at::Tensor, at::Tensor> run(
        const std::vector<at::Tensor>& images_to_match,
        const std::vector<at::Tensor>& images_to_stitch
    );

    /**
     * Estimate distance map for a single fisheye reference image
     * ZERO-COPY: Writes directly to output_buffer (eliminates copy_ overhead)
     * @param reference_image Reference fisheye image [H, W, 3]
     * @param guide Guide image for filtering [H, W, 3] uint8
     * @param reference_calibration Calibration for reference camera
     * @param selected_camera Camera selection map [1, H, W] int
     * @param images All fisheye images [num_cameras][H, W, 3]
     * @param ref_idx_local Local index in references_indices_ (for precomputed RT)
     * @param stream CUDA stream for async execution
     * @param output_buffer Pre-allocated output buffer [H, W] (writes directly here)
     */
    void estimate_fisheye_distance(
        const at::Tensor& reference_image,
        const at::Tensor& guide,
        const Calibration& reference_calibration,
        const at::Tensor& selected_camera,
        const std::vector<at::Tensor>& images,
        int ref_idx_local,
        cudaStream_t stream,
        at::Tensor& output_buffer
    );

    /**
     * Select best matching camera for each pixel (adaptive matching)
     * Populates selected_cameras_ member variable
     * @param masks Valid area masks for each camera
     */
    void select_camera(const std::vector<at::Tensor>& masks);

    /**
     * Convert RGB image to YCbCr color space
     * @param rgb_image RGB image [H, W, 3] float32 [0-255]
     * @return YCbCr image [H, W, 3] uint8
     */
    void rgb_to_ycbcr(const at::Tensor& rgb_image, at::Tensor& ycbcr_out);

private:
    // Configuration parameters
    std::vector<Calibration> calibrations_;
    float min_dist_;
    float max_dist_;
    int candidate_count_;
    std::vector<int> references_indices_;
    at::Tensor reprojection_viewpoint_;
    std::pair<int, int> matching_resolution_;  // (cols, rows)
    at::Device device_;
    float sigma_i_;
    float sigma_s_;

    // Pre-computed data
    at::Tensor distance_candidates_;  // [candidate_count] Pre-computed distance values
    std::vector<float> distance_candidates_cpu_;  // CPU copy for constant memory (init-time only)
    std::vector<at::Tensor> selected_cameras_;  // [num_refs][1, H, W] Camera selection per pixel

    // Processing modules (PARALLELIZED: one filter per camera to eliminate buffer contention)
    std::vector<std::unique_ptr<ISBFilter>> cost_filters_;      // Per-camera filters for cost volumes
    std::vector<std::unique_ptr<ISBFilter>> distance_filters_;  // Per-camera filters for distance maps
    std::unique_ptr<Stitcher> fisheye_stitcher_;  // Stitcher for panorama creation

    // CUDA structures for fused kernel
    std::vector<DoubleSphereParams> camera_params_;  // Camera intrinsics for all cameras
    std::vector<CameraExtrinsics> camera_rts_;       // Relative RT matrices for all cameras
    
    // Pre-computed RT matrices per reference camera (eliminates 34.5% CPU/GPU sync bottleneck)
    std::vector<std::vector<CameraExtrinsics>> precomputed_camera_rts_;  // [num_refs][num_cameras]
    
    // CUDA Streams for async parallel execution (eliminates serialization)
    std::vector<cudaStream_t> camera_streams_;  // [num_refs] One stream per reference camera

    // Unified Memory Buffers (Zero-Copy Architecture)
    // CRITICAL: Per-camera buffers to eliminate memory contention (true parallelism)
    float* unified_sweeping_volume_ptr_;  // [1, 3, D, H, W]
    std::vector<float*> unified_cost_volume_ptrs_;      // [num_refs][D, H, W] - ONE PER CAMERA
    float* unified_distance_map_ptr_;     // [H, W]
    float* unified_input_buffer_ptr_;     // [num_cameras, H, W, 3]
    
    // Per-camera pre-allocated buffers (FP16 for memory efficiency)
    std::vector<at::Tensor> per_camera_distance_maps_;  // [num_refs][H, W] float32
    std::vector<at::Tensor> per_camera_guide_buffers_;  // [num_refs][H, W, 3] uint8
    std::vector<at::Tensor> per_camera_cost_volumes_;   // [num_refs][D, H, W] float32 - ONE PER CAMERA
    std::vector<at::Tensor> per_camera_temp_distance_buffers_;  // [num_refs][H, W] - ONE PER CAMERA (eliminates contention)
    
    // Texture management (CRITICAL: Pre-create to avoid .cpu() sync in hot path)
    std::vector<std::vector<cudaArray*>> texture_arrays_;  // [num_refs][num_cameras] Pre-allocated CUDA arrays
    std::vector<std::vector<cudaTextureObject_t>> texture_objects_;  // [num_refs][num_cameras] Reusable textures
    
    // Wrapped tensors from unified memory
    at::Tensor unified_sweeping_volume_;
    std::vector<at::Tensor> unified_cost_volumes_;  // [num_refs] - ONE PER CAMERA
    at::Tensor unified_distance_map_;
    at::Tensor unified_input_buffer_;
    
    // Buffer sizes
    size_t sweeping_volume_size_;
    size_t cost_volume_size_;
    size_t distance_map_size_;
    size_t input_buffer_size_;

    /**
     * Allocate unified memory buffers
     */
    void allocate_unified_buffers();

    /**
     * Free unified memory buffers
     */
    void free_unified_buffers();
    
    /**
     * Pre-compute relative RT matrices for all reference cameras
     * Eliminates runtime CPU/GPU sync bottleneck (34.5% overhead)
     */
    void precompute_relative_rt_matrices();

    /**
     * Unproject pixel coordinates to 3D unit vectors
     * @param uv Pixel coordinates [N, 2]
     * @param calib Camera calibration
     * @return Pair of (unit vectors [N, 3], validity mask [N])
     */
    std::pair<at::Tensor, at::Tensor> unproject(
        const at::Tensor& uv,
        const Calibration& calib
    ) const;

    /**
     * Project 3D points to pixel coordinates
     * @param points 3D points [N, 3]
     * @param calib Camera calibration
     * @return Pair of (pixel coordinates [N, 2], validity mask [N])
     */
    std::pair<at::Tensor, at::Tensor> project(
        const at::Tensor& points,
        const Calibration& calib
    ) const;
};

} // namespace my_stereo_pkg
