#include "my_stereo_pkg/stitcher.hpp"
#include "my_stereo_pkg/cuda_kernels.hpp"
#include <torch/torch.h>
#include <stdexcept>
#include <iostream>

namespace my_stereo_pkg {

Stitcher::Stitcher(const StitcherParams& params, const std::vector<torch::Tensor>& masks)
    : params_(params) {
    
    // Validate parameters
    if (params_.calibrations.empty()) {
        throw std::runtime_error("No calibrations provided");
    }
    
    if (masks.size() != params_.calibrations.size()) {
        throw std::runtime_error("Number of masks must match number of calibrations");
    }
    
    // Initialize all intermediate tensors and lookup tables
    initialize();
    
    // Create inpainting weights for all cameras
    createInpaintingWeights();
    
    // Create blending lookup tables
    createBlendingLuts(masks);
}

Stitcher::~Stitcher() {
    // Tensors will be automatically cleaned up
}

at::Tensor Stitcher::process(const at::Tensor& input) {
    // Step 1: Check if input tensor is on GPU and contiguous
    if (!input.is_cuda()) {
        throw std::runtime_error("Input tensor must be on CUDA device");
    }
    
    if (!input.is_contiguous()) {
        throw std::runtime_error("Input tensor must be contiguous");
    }
    
    // Validate input dimensions
    // Expected format: [batch_size, channels, height, width] where channels = (3 + 1) * num_cameras
    // RGB channels (3) + Distance channel (1) for each camera
    auto input_sizes = input.sizes();
    if (input_sizes.size() != 4) {
        throw std::runtime_error("Input tensor must be 4D: [batch, channels, height, width]");
    }
    
    int batch_size = input_sizes[0];
    int total_channels = input_sizes[1];
    int height = input_sizes[2];
    int width = input_sizes[3];
    
    int num_cameras = params_.calibrations.size();
    int expected_channels = (3 + 1) * num_cameras;  // RGB + Distance for each camera
    
    if (total_channels != expected_channels) {
        throw std::runtime_error("Input channels (" + std::to_string(total_channels) + 
                                ") must match (RGB+D) * num_cameras (" + std::to_string(expected_channels) + ")");
    }
    
    // Step 2: Allocate output tensor on GPU
    auto options = torch::TensorOptions()
                    .dtype(torch::kUInt8)
                    .device(params_.device)
                    .requires_grad(false);
    
    // Output format: [batch, 4, panorama_height, panorama_width] where channels are [R, G, B, Distance]
    torch::Tensor output = torch::empty({
        batch_size, 
        4,  // R, G, B, Distance
        params_.panorama_resolution.rows, 
        params_.panorama_resolution.cols
    }, options);
    
    // Process each item in the batch
    for (int b = 0; b < batch_size; ++b) {
        // Extract RGB images and distance maps from input
        std::vector<torch::Tensor> rgb_images;
        std::vector<torch::Tensor> distance_maps;
        
        rgb_images.reserve(num_cameras);
        distance_maps.reserve(num_cameras);
        
        for (int c = 0; c < num_cameras; ++c) {
            // Extract RGB channels for camera c
            torch::Tensor rgb = input[b].slice(0, c * 4, c * 4 + 3);  // [3, H, W]
            rgb = rgb.permute({1, 2, 0});  // Convert to [H, W, 3]
            rgb_images.push_back(rgb);
            
            // Extract distance channel for camera c  
            torch::Tensor distance = input[b].slice(0, c * 4 + 3, c * 4 + 4).squeeze(0);  // [H, W]
            distance_maps.push_back(distance);
        }
        
        // Create output tensors for this batch item
        auto rgb_panorama_out = torch::empty({
            params_.panorama_resolution.rows, 
            params_.panorama_resolution.cols, 
            3
        }, torch::TensorOptions().dtype(torch::kUInt8).device(params_.device));
        
        auto distance_panorama_out = torch::empty({
            params_.panorama_resolution.rows, 
            params_.panorama_resolution.cols
        }, torch::TensorOptions().dtype(torch::kFloat32).device(params_.device));
        
        // Step 3: Call launch_stitch_kernel to execute processing
        try {
            launch_stitch_kernel(
                rgb_images,
                distance_maps, 
                rgb_panorama_out,
                distance_panorama_out,
                params_
            );
        } catch (const std::exception& e) {
            throw std::runtime_error("CUDA kernel execution failed: " + std::string(e.what()));
        }
        
        // Copy results to output tensor
        // RGB channels
        output[b].slice(0, 0, 3) = rgb_panorama_out.permute({2, 0, 1});  // [3, H, W]
        
        // Distance channel (convert float32 to uint8 for output consistency)
        auto distance_normalized = (distance_panorama_out - params_.min_dist) / 
                                  (params_.max_dist - params_.min_dist) * 255.0;
        distance_normalized = torch::clamp(distance_normalized, 0, 255);
        output[b][3] = distance_normalized.to(torch::kUInt8);
    }
    
    // Step 4: Return output tensor
    return output;
}

void Stitcher::stitch(
    const std::vector<torch::Tensor>& rgb_images,
    const std::vector<torch::Tensor>& distance_maps,
    torch::Tensor& rgb_panorama_out,
    torch::Tensor& distance_panorama_out
) {
    // Validate inputs
    validateInputs(rgb_images, distance_maps);
    
    // Copy input data to pre-allocated buffers
    copyInputData(rgb_images, distance_maps);
    
    // Step 1: Reproject distance maps to reference viewpoint
    reprojectDistanceMaps();
    
    // Step 2: Apply inpainting to fill holes
    inpaintDistanceMaps();
    
    // Step 3: Merge fisheye images into final panorama
    mergeRGBDPanorama();
    
    // Copy results to output tensors
    rgb_panorama_out.copy_(rgb_panorama_);
    distance_panorama_out.copy_(distance_panorama_);
}

void Stitcher::setReprojectionViewpoint(const torch::Tensor& viewpoint) {
    if (viewpoint.sizes() != torch::IntArrayRef{3}) {
        throw std::runtime_error("Reprojection viewpoint must be 3D vector");
    }
    
    params_.reprojection_viewpoint = viewpoint.to(params_.device);
    
    // Recreate inpainting weights and blending LUTs with new viewpoint
    createInpaintingWeights();
}

// Private implementation methods

void Stitcher::initialize() {
    int num_cameras = params_.calibrations.size();
    
    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(params_.device);
    auto uint8_opts = torch::TensorOptions().dtype(torch::kUInt8).device(params_.device);
    
    // Allocate intermediate tensors
    reprojected_distances_ = torch::zeros({
        num_cameras, 
        params_.matching_resolution.rows, 
        params_.matching_resolution.cols
    }, float_opts);
    
    distances_stacked_ = torch::zeros({
        num_cameras,
        params_.matching_resolution.rows, 
        params_.matching_resolution.cols
    }, float_opts);
    
    images_to_stitch_ = torch::zeros({
        num_cameras,
        params_.rgb_to_stitch_resolution.rows,
        params_.rgb_to_stitch_resolution.cols,
        3
    }, uint8_opts);
    
    // Allocate output tensors
    rgb_panorama_ = torch::zeros({
        params_.panorama_resolution.rows,
        params_.panorama_resolution.cols, 
        3
    }, uint8_opts);
    
    distance_panorama_ = torch::zeros({
        params_.panorama_resolution.rows,
        params_.panorama_resolution.cols
    }, float_opts);
    
    // Initialize camera-specific data
    translations_list_.clear();
    calibrations_list_.clear();
    
    // Extract calibrations and compute translations
    torch::Tensor reprojection_viewpoint_homogeneous = torch::cat({
        params_.reprojection_viewpoint,
        torch::ones({1}, float_opts)
    });
    
    for (const auto& calib : params_.calibrations) {
        calibrations_list_.push_back(calib.intrinsics);
        
        // Compute translation from reference viewpoint to camera
        torch::Tensor rt_inv = torch::inverse(calib.rotation_matrix.cat({calib.translation_vector}, 1));
        torch::Tensor translation = torch::matmul(rt_inv, reprojection_viewpoint_homogeneous).slice(0, 0, 3);
        translations_list_.push_back(translation);
    }
}

void Stitcher::createInpaintingWeights() {
    int num_cameras = params_.calibrations.size();
    inpainting_weights_list_.clear();
    inpainting_weights_list_.reserve(num_cameras);
    
    auto uint8_opts = torch::TensorOptions().dtype(torch::kUInt8).device(params_.device);
    
    for (int i = 0; i < num_cameras; ++i) {
        torch::Tensor inpainting_weight = torch::zeros({
            params_.matching_resolution.rows,
            params_.matching_resolution.cols, 
            2
        }, uint8_opts);
        
        launch_create_inpainting_weights_kernel(
            inpainting_weight,
            calibrations_list_[i],
            translations_list_[i],
            params_
        );
        
        inpainting_weights_list_.push_back(inpainting_weight);
    }
}

void Stitcher::createBlendingLuts(const std::vector<torch::Tensor>& masks) {
    int num_cameras = params_.calibrations.size();
    
    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(params_.device);
    
    // Allocate blending tensors
    blending_sampling_ = torch::zeros({
        num_cameras,
        params_.panorama_resolution.rows,
        params_.panorama_resolution.cols,
        2
    }, float_opts);
    
    blending_weights_ = torch::zeros({
        num_cameras,
        params_.panorama_resolution.rows,
        params_.panorama_resolution.cols
    }, float_opts);
    
    // Apply smoothing to masks
    torch::Tensor masks_concat = torch::cat(masks, 0).to(params_.device);
    
    // Create rotation matrices tensor
    std::vector<torch::Tensor> rotation_matrices;
    for (const auto& calib : params_.calibrations) {
        rotation_matrices.push_back(torch::inverse(calib.rotation_matrix));
    }
    rotations_ = torch::cat(rotation_matrices, 0).contiguous();
    translations_ = torch::cat(translations_list_, 0).contiguous();
    
    launch_create_blending_lut_kernel(
        blending_sampling_,
        blending_weights_,
        masks_concat,
        calibrations_list_,
        rotations_,
        translations_,
        params_
    );
    
    // Apply smoothing to blending weights
    auto conv_kernel = torch::ones({1, 1, 2 * params_.smoothing_radius + 1, 2 * params_.smoothing_radius + 1}, float_opts);
    conv_kernel /= torch::sum(conv_kernel);
    
    blending_weights_ = torch::nn::functional::conv2d(
        blending_weights_.unsqueeze(1), 
        conv_kernel,
        torch::nn::functional::Conv2dFuncOptions().padding(params_.smoothing_radius)
    ).squeeze(1);
    
    // Normalize weights
    blending_weights_ /= torch::sum(blending_weights_, 0, true);
}

void Stitcher::validateInputs(
    const std::vector<torch::Tensor>& rgb_images,
    const std::vector<torch::Tensor>& distance_maps
) const {
    if (rgb_images.size() != params_.calibrations.size()) {
        throw std::runtime_error("Number of RGB images must match number of calibrations");
    }
    
    if (distance_maps.size() != params_.calibrations.size()) {
        throw std::runtime_error("Number of distance maps must match number of calibrations");
    }
    
    for (const auto& img : rgb_images) {
        if (!img.is_cuda()) {
            throw std::runtime_error("All RGB images must be on CUDA device");
        }
        if (!img.is_contiguous()) {
            throw std::runtime_error("All RGB images must be contiguous");
        }
    }
    
    for (const auto& dist : distance_maps) {
        if (!dist.is_cuda()) {
            throw std::runtime_error("All distance maps must be on CUDA device");
        }
        if (!dist.is_contiguous()) {
            throw std::runtime_error("All distance maps must be contiguous");
        }
    }
}

void Stitcher::copyInputData(
    const std::vector<torch::Tensor>& rgb_images,
    const std::vector<torch::Tensor>& distance_maps
) {
    for (size_t i = 0; i < rgb_images.size(); ++i) {
        images_to_stitch_[i].copy_(rgb_images[i]);
        distances_stacked_[i].copy_(distance_maps[i]);
    }
}

void Stitcher::reprojectDistanceMaps() {
    for (size_t i = 0; i < calibrations_list_.size(); ++i) {
        // Initialize with large values
        reprojected_distances_[i].fill_(1e8);
        
        // Run reprojection kernel twice for better coverage
        for (int pass = 0; pass < 2; ++pass) {
            launch_reproject_distance_kernel(
                distances_stacked_[i],
                reprojected_distances_[i],
                calibrations_list_[i],
                translations_list_[i],
                params_
            );
        }
    }
}

void Stitcher::inpaintDistanceMaps() {
    for (size_t i = 0; i < calibrations_list_.size(); ++i) {
        for (int iter = 0; iter < params_.inpainting_iterations; ++iter) {
            launch_inpaint_kernel(
                reprojected_distances_[i],
                inpainting_weights_list_[i],
                params_
            );
        }
    }
}

void Stitcher::mergeRGBDPanorama() {
    launch_merge_rgbd_panorama_kernel(
        blending_sampling_,
        blending_weights_,
        reprojected_distances_,
        distances_stacked_,
        images_to_stitch_,
        translations_,
        calibrations_list_,
        distance_panorama_,
        rgb_panorama_,
        params_
    );
}

// Factory function implementation
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
    int smoothing_radius,
    int inpainting_iterations
) {
    StitcherParams params(
        calibrations,
        reprojection_viewpoint,
        min_dist,
        max_dist,
        matching_resolution,
        rgb_to_stitch_resolution,
        panorama_resolution,
        device,
        smoothing_radius,
        inpainting_iterations
    );
    
    return std::make_unique<Stitcher>(params, masks);
}

} // namespace my_stereo_pkg