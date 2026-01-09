#pragma once

#include <torch/torch.h>
#include <memory>
#include <vector>
#include "my_stereo_pkg/cuda_kernels.hpp"

namespace my_stereo_pkg {

/**
 * Stitcher class for creating RGB-D panoramas from RGB-D fisheye images
 * C++ implementation of the sphere sweeping stereo algorithm
 */
class Stitcher {
public:
    /**
     * Constructor
     * @param params Complete stitcher parameters
     * @param masks Masks for valid areas in fisheye images [num_cameras][rows][cols]
     */
    Stitcher(const StitcherParams& params, const std::vector<torch::Tensor>& masks);
    
    /**
     * Destructor
     */
    ~Stitcher();

    /**
     * Main processing function - stitch fisheye images into RGB-D panorama
     * @param input Concatenated RGB images and distance maps
     *              Format: [rgb_images..., distance_maps...]
     *              RGB: [num_cameras, height, width, 3] uint8
     *              Distance: [num_cameras, height, width] float32
     * @return Stitched panorama [panorama_height, panorama_width, 4] 
     *         where channels are [R, G, B, Distance]
     */
    at::Tensor process(const at::Tensor& input);

    /**
     * Stitch RGB-D fisheye images into panorama
     * @param rgb_images Vector of RGB fisheye images [height, width, 3] uint8
     * @param distance_maps Vector of distance maps [height, width] float32
     * @param rgb_panorama_out Output RGB panorama [pano_height, pano_width, 3] uint8
     * @param distance_panorama_out Output distance panorama [pano_height, pano_width] float32
     */
    void stitch(
        const std::vector<torch::Tensor>& rgb_images,
        const std::vector<torch::Tensor>& distance_maps,
        torch::Tensor& rgb_panorama_out,
        torch::Tensor& distance_panorama_out
    );

    /**
     * Get stitcher parameters
     */
    const StitcherParams& getParams() const { return params_; }

    /**
     * Update reprojection viewpoint
     */
    void setReprojectionViewpoint(const torch::Tensor& viewpoint);

private:
    // Core parameters
    StitcherParams params_;
    
    // Pre-allocated intermediate tensors
    torch::Tensor reprojected_distances_;      // [num_cameras, rows, cols]
    torch::Tensor distances_stacked_;          // [num_cameras, rows, cols]
    torch::Tensor images_to_stitch_;          // [num_cameras, rgb_rows, rgb_cols, 3]
    
    // Lookup tables and weights
    torch::Tensor blending_sampling_;         // [num_cameras, pano_rows, pano_cols, 2]
    torch::Tensor blending_weights_;          // [num_cameras, pano_rows, pano_cols]
    std::vector<torch::Tensor> inpainting_weights_list_;  // [num_cameras][rows, cols, 2]
    
    // Output tensors
    torch::Tensor rgb_panorama_;             // [pano_rows, pano_cols, 3]
    torch::Tensor distance_panorama_;        // [pano_rows, pano_cols]
    
    // Camera-specific data
    std::vector<torch::Tensor> translations_list_;
    std::vector<Intrinsics> calibrations_list_;
    torch::Tensor calibration_vectors_;      // Vectorized calibrations
    torch::Tensor translations_;             // Concatenated translations
    torch::Tensor rotations_;               // Concatenated rotation matrices

    /**
     * Initialize all intermediate tensors and lookup tables
     */
    void initialize();
    
    /**
     * Create inpainting weights for all cameras
     */
    void createInpaintingWeights();
    
    /**
     * Create blending lookup tables for panorama generation
     */
    void createBlendingLuts(const std::vector<torch::Tensor>& masks);
    
    /**
     * Convert calibration parameters to vectorized format
     */
    torch::Tensor vectorizeCalibration(const Intrinsics& calibration);
    
    /**
     * Validate input tensors
     */
    void validateInputs(
        const std::vector<torch::Tensor>& rgb_images,
        const std::vector<torch::Tensor>& distance_maps
    ) const;
    
    /**
     * Copy input data to pre-allocated buffers
     */
    void copyInputData(
        const std::vector<torch::Tensor>& rgb_images,
        const std::vector<torch::Tensor>& distance_maps
    );
    
    /**
     * Reproject distance maps to reference viewpoint
     */
    void reprojectDistanceMaps();
    
    /**
     * Apply inpainting to fill holes in distance maps
     */
    void inpaintDistanceMaps();
    
    /**
     * Merge fisheye images into final RGB-D panorama
     */
    void mergeRGBDPanorama();
};

/**
 * Factory function to create Stitcher instance
 * @param calibrations Vector of camera calibrations
 * @param reprojection_viewpoint Reference viewpoint [3]
 * @param masks Masks for valid areas [num_cameras][rows][cols]
 * @param min_dist Minimum expected distance
 * @param max_dist Maximum expected distance
 * @param matching_resolution Resolution for matching computation
 * @param rgb_to_stitch_resolution Resolution for RGB stitching
 * @param panorama_resolution Output panorama resolution
 * @param device CUDA device
 * @param smoothing_radius Blending smoothing radius (default: 15)
 * @param inpainting_iterations Number of inpainting iterations (default: 32)
 * @return Unique pointer to Stitcher instance
 */
std::unique_ptr<Stitcher> createStitcher(
    const std::vector<CameraCalibration>& calibrations,
    const torch::Tensor& reprojection_viewpoint,
    const std::vector<torch::Tensor>& masks,
    float min_dist,
    float max_dist,
    const Resolution& matching_resolution,
    const Resolution& rgb_to_stitch_resolution,
    const Resolution& panorama_resolution,
    const torch::Device& device,
    int smoothing_radius = 15,
    int inpainting_iterations = 32
);

} // namespace my_stereo_pkg