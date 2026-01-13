#pragma once

#include <torch/torch.h>
#include <vector>
#include <memory>
#include "my_stereo_pkg/calibration.hpp"
#include "my_stereo_pkg/cuda_kernels.hpp"

namespace my_stereo {

/**
 * Stitcher to create RGB-D panoramas from RGB-D fisheye images
 * C++ implementation using LibTorch
 */
class Stitcher {
public:
    /**
     * Constructor
     * @param calibrations Vector of calibration parameters for each camera
     * @param reprojection_viewpoint [3] Reference viewpoint where the RGB-D panorama will be created
     * @param masks [num_cameras, matching_rows, matching_cols] Mask of valid areas
     * @param min_dist Minimum distance expected in distance maps
     * @param max_dist Maximum distance expected in distance maps
     * @param matching_cols Width of matching resolution
     * @param matching_rows Height of matching resolution
     * @param rgb_to_stitch_cols Width of color images for stitching
     * @param rgb_to_stitch_rows Height of color images for stitching
     * @param panorama_cols Width of output panorama
     * @param panorama_rows Height of output panorama
     * @param device CUDA device
     * @param smoothing_radius Radius for blending weight smoothing (default: 15)
     * @param inpainting_iterations Number of inpainting passes (default: 32)
     */
    Stitcher(
        const std::vector<Calibration>& calibrations,
        const at::Tensor& reprojection_viewpoint,
        const at::Tensor& masks,
        float min_dist,
        float max_dist,
        int matching_cols,
        int matching_rows,
        int rgb_to_stitch_cols,
        int rgb_to_stitch_rows,
        int panorama_cols,
        int panorama_rows,
        const at::Device& device,
        int smoothing_radius = 15,
        int inpainting_iterations = 32
    );

    /**
     * Stitch fisheye images and distance maps into RGB-D panorama
     * @param images Vector of [H, W, 3] uint8 color fisheye images
     * @param distance_maps Vector of [H, W] float32 distance maps
     * @return Pair of (RGB panorama [H, W, 3] uint8, distance panorama [H, W] float32)
     */
    std::pair<at::Tensor, at::Tensor> stitch(
        const std::vector<at::Tensor>& images,
        const std::vector<at::Tensor>& distance_maps
    );

private:
    // Helper function to vectorize calibration (equivalent to Python's vectorize_calibration)
    Intrinsics vectorize_calibration(const Calibration& calib);
    
    // Helper function to create convolution kernel for smoothing
    at::Tensor create_box_filter(int radius, const at::Device& device);

    // Parameters
    float max_dist_;
    int inpainting_iterations_;
    int matching_cols_;
    int matching_rows_;
    int panorama_cols_;
    int panorama_rows_;
    int num_cameras_;
    at::Device device_;

    // Intermediate tensors
    at::Tensor reprojected_distances_;      // [num_cameras, matching_rows, matching_cols]
    at::Tensor distances_stacked_;          // [num_cameras, matching_rows, matching_cols]
    at::Tensor images_to_stitch_;           // [num_cameras, rgb_rows, rgb_cols, 3] uint8
    
    // Inpainting weights for each camera
    std::vector<at::Tensor> inpainting_weights_list_;  // Each: [matching_rows, matching_cols, 2] uint8
    
    // Blending lookup tables
    at::Tensor blending_sampling_;          // [num_cameras, pano_rows, pano_cols, 2]
    at::Tensor blending_weights_;           // [num_cameras, pano_rows, pano_cols]
    
    // Output buffers
    at::Tensor RGB_panorama_;               // [pano_rows, pano_cols, 3] uint8
    at::Tensor distance_panorama_;          // [pano_rows, pano_cols] float32
    
    // Camera parameters
    std::vector<at::Tensor> translations_list_;        // Each: [3]
    std::vector<Intrinsics> calibration_vectors_list_;  // Vectorized calibrations
    at::Tensor calibration_vectors_;        // [num_cameras * 6] all calibrations concatenated
    at::Tensor translations_;               // [num_cameras * 3] all translations concatenated
};

} // namespace my_stereo
