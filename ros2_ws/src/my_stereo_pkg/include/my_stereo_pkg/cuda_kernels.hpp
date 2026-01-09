#pragma once

#include <vector>
#include <memory>
#include <torch/torch.h>

namespace my_stereo_pkg {

/**
 * Intrinsics structure matching the CUDA kernel definition
 * Based on the Double Sphere Camera Model (https://arxiv.org/abs/1807.08957)
 */
struct Intrinsics {
    float2 fl;         // focal length (x, y)
    float2 principal;  // principal point (x, y)  
    float xi;          // xi parameter of double sphere model
    float alpha;       // alpha parameter of double sphere model
};

/**
 * Camera calibration parameters for a single camera
 */
struct CameraCalibration {
    Intrinsics intrinsics;
    torch::Tensor rotation_matrix;     // 3x3 rotation matrix
    torch::Tensor translation_vector;  // 3x1 translation vector
    float matching_scale;              // scale factor for matching resolution
};

/**
 * Resolution structure for different image sizes
 */
struct Resolution {
    int cols;  // width
    int rows;  // height
    
    Resolution() : cols(0), rows(0) {}
    Resolution(int w, int h) : cols(w), rows(h) {}
};

/**
 * Complete stitcher parameters structure
 * Contains all parameters needed for sphere sweeping stereo stitching
 */
struct StitcherParams {
    // Camera parameters
    std::vector<CameraCalibration> calibrations;
    torch::Tensor reprojection_viewpoint;  // [3] reference viewpoint
    
    // Distance parameters
    float min_dist;
    float max_dist;
    
    // Resolution parameters
    Resolution matching_resolution;        // resolution for matching computation
    Resolution rgb_to_stitch_resolution;  // resolution for RGB stitching
    Resolution panorama_resolution;       // output panorama resolution
    
    // Processing parameters
    int smoothing_radius;        // blending smoothing radius (default: 15)
    int inpainting_iterations;   // number of inpainting iterations (default: 32)
    int block_size;              // CUDA block size (default: 256)
    
    // GPU parameters
    torch::Device device;
    
    // Computed grid sizes
    int fisheye_grid_size;
    int panorama_grid_size;
    
    // Constructor with default values
    StitcherParams(
        const std::vector<CameraCalibration>& cams,
        const torch::Tensor& viewpoint,
        float min_distance,
        float max_distance,
        const Resolution& matching_res,
        const Resolution& rgb_res, 
        const Resolution& pano_res,
        const torch::Device& dev,
        int smooth_radius = 15,
        int inpaint_iters = 32,
        int block_sz = 256
    ) : calibrations(cams),
        reprojection_viewpoint(viewpoint),
        min_dist(min_distance),
        max_dist(max_distance), 
        matching_resolution(matching_res),
        rgb_to_stitch_resolution(rgb_res),
        panorama_resolution(pano_res),
        smoothing_radius(smooth_radius),
        inpainting_iterations(inpaint_iters),
        block_size(block_sz),
        device(dev) {
        
        // Calculate grid sizes for CUDA kernels
        int matching_pixels = matching_resolution.cols * matching_resolution.rows;
        int panorama_pixels = panorama_resolution.cols * panorama_resolution.rows;
        fisheye_grid_size = (matching_pixels + block_size - 1) / block_size;
        panorama_grid_size = (panorama_pixels + block_size - 1) / panorama_grid_size;
    }
};

/**
 * CUDA kernel launch functions
 */

// Reproject distance map to reference viewpoint 
void launch_reproject_distance_kernel(
    const torch::Tensor& distance_in,
    torch::Tensor& distance_out,
    const Intrinsics& calibration,
    const torch::Tensor& translation,
    const StitcherParams& params
);

// Create inpainting weights for hole filling
void launch_create_inpainting_weights_kernel(
    torch::Tensor& inpainting_weights,
    const Intrinsics& calibration,
    const torch::Tensor& translation,
    const StitcherParams& params
);

// Apply inpainting to fill holes in distance map
void launch_inpaint_kernel(
    torch::Tensor& distance_map,
    const torch::Tensor& inpainting_weights,
    const StitcherParams& params
);

// Create blending lookup tables for panorama stitching
void launch_create_blending_lut_kernel(
    torch::Tensor& sampling_lut,
    torch::Tensor& blending_weights,
    const torch::Tensor& masks,
    const std::vector<Intrinsics>& calibrations,
    const torch::Tensor& rotations,
    const torch::Tensor& translations,
    const StitcherParams& params
);

// Main stitching kernel to merge RGB-D fisheye images into panorama
void launch_merge_rgbd_panorama_kernel(
    const torch::Tensor& sampling_lut,
    const torch::Tensor& blending_weights,
    const torch::Tensor& reprojected_distance_maps,
    const torch::Tensor& distance_maps,
    const torch::Tensor& stitching_images,
    const torch::Tensor& translations,
    const std::vector<Intrinsics>& calibrations,
    torch::Tensor& distance_panorama,
    torch::Tensor& rgb_panorama,
    const StitcherParams& params
);

// Main stitching function - high level API
void launch_stitch_kernel(
    const std::vector<torch::Tensor>& rgb_images,
    const std::vector<torch::Tensor>& distance_maps,
    torch::Tensor& rgb_panorama_out,
    torch::Tensor& distance_panorama_out,
    const StitcherParams& params
);

/**
 * Utility functions
 */

// Convert calibration parameters to Intrinsics struct
Intrinsics create_intrinsics(
    const torch::Tensor& focal_length,      // [2] (fx, fy)
    const torch::Tensor& principal_point,   // [2] (cx, cy)
    float xi,
    float alpha,
    float matching_scale = 1.0f
);

// Create vectorized calibration for CUDA kernels (matches vectorize_calibration in Python)
torch::Tensor vectorize_calibration(
    const Intrinsics& calibration,
    const torch::Device& device
);

} // namespace my_stereo_pkg