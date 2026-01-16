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
#include <torch/torch.h>
#include <vector>
#include <memory>

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
     * @param reference_image Reference fisheye image [1, 3, 1, H, W]
     * @param guide Guide image for filtering [H, W, 3] uint8
     * @param reference_calibration Calibration for reference camera
     * @param selected_camera Camera selection map [1, H, W] int
     * @param images All fisheye images [num_cameras][1, 3, 1, H, W]
     * @return Distance map [H, W] float32
     */
    at::Tensor estimate_fisheye_distance(
        const at::Tensor& reference_image,
        const at::Tensor& guide,
        const Calibration& reference_calibration,
        const at::Tensor& selected_camera,
        const std::vector<at::Tensor>& images
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
    static at::Tensor rgb_to_ycbcr(const at::Tensor& rgb_image);

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
    std::vector<at::Tensor> selected_cameras_;  // [num_refs][1, H, W] Camera selection per pixel

    // Processing modules
    std::unique_ptr<ISBFilter> cost_filter_;      // Filter for cost volumes
    std::unique_ptr<ISBFilter> distance_filter_;  // Filter for distance maps
    std::unique_ptr<Stitcher> fisheye_stitcher_;  // Stitcher for panorama creation

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
