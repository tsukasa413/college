#include "my_stereo_pkg/stitcher.hpp"
#include <torch/torch.h>
#include <cmath>
#include <iostream>

namespace my_stereo {

Stitcher::Stitcher(
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
    int smoothing_radius,
    int inpainting_iterations
) : max_dist_(max_dist),
    inpainting_iterations_(inpainting_iterations),
    matching_cols_(matching_cols),
    matching_rows_(matching_rows),
    panorama_cols_(panorama_cols),
    panorama_rows_(panorama_rows),
    num_cameras_(calibrations.size()),
    device_(device)
{
    std::cout << "Initializing Stitcher with " << num_cameras_ << " cameras..." << std::endl;
    
    // Extend reprojection viewpoint to homogeneous coordinates [4]
    at::Tensor reprojection_viewpoint_homo = at::cat({
        reprojection_viewpoint,
        at::ones({1}, at::TensorOptions().device(device))
    }, 0);
    
    // Allocate intermediate arrays
    reprojected_distances_ = at::zeros({num_cameras_, matching_rows, matching_cols}, 
                                       at::TensorOptions().dtype(at::kFloat).device(device));
    distances_stacked_ = at::zeros({num_cameras_, matching_rows, matching_cols}, 
                                    at::TensorOptions().dtype(at::kFloat).device(device));
    images_to_stitch_ = at::zeros({num_cameras_, rgb_to_stitch_rows, rgb_to_stitch_cols, 3}, 
                                   at::TensorOptions().dtype(at::kByte).device(device));
    
    // Allocate blending lookup tables
    blending_sampling_ = at::zeros({num_cameras_, panorama_rows, panorama_cols, 2}, 
                                    at::TensorOptions().dtype(at::kFloat).device(device));
    blending_weights_ = at::zeros({num_cameras_, panorama_rows, panorama_cols}, 
                                   at::TensorOptions().dtype(at::kFloat).device(device));
    
    // Allocate output buffers
    RGB_panorama_ = at::zeros({panorama_rows, panorama_cols, 3}, 
                              at::TensorOptions().dtype(at::kByte).device(device));
    distance_panorama_ = at::zeros({panorama_rows, panorama_cols}, 
                                    at::TensorOptions().dtype(at::kFloat).device(device));
    
    // Prepare calibrations and translations
    std::vector<at::Tensor> rotation_list;
    std::vector<float> calib_flat_list;
    std::vector<float> translation_flat_list;
    
    // Reserve space to avoid reallocation
    calibration_vectors_list_.reserve(num_cameras_);
    translations_list_.reserve(num_cameras_);
    inpainting_weights_list_.reserve(num_cameras_);
    rotation_list.reserve(num_cameras_);
    calib_flat_list.reserve(num_cameras_ * 6);
    translation_flat_list.reserve(num_cameras_ * 3);
    
    for (const auto& calibration : calibrations) {
        // Vectorize calibration and store in list
        Intrinsics intrinsics = vectorize_calibration(calibration);
        calibration_vectors_list_.push_back(intrinsics);
        
        // Flatten intrinsics to float array for concatenation
        calib_flat_list.push_back(intrinsics.fl.x);
        calib_flat_list.push_back(intrinsics.fl.y);
        calib_flat_list.push_back(intrinsics.principal.x);
        calib_flat_list.push_back(intrinsics.principal.y);
        calib_flat_list.push_back(intrinsics.xi);
        calib_flat_list.push_back(intrinsics.alpha);
        
        // Compute translation: translation = inv(rt) @ reprojection_viewpoint
        at::Tensor rt_inv = at::inverse(calibration.rt);
        at::Tensor translation = at::matmul(rt_inv, reprojection_viewpoint_homo).slice(0, 0, 3);
        
        // Move to device and store
        at::Tensor translation_device = translation.to(device);
        translations_list_.push_back(translation_device);
        
        // Flatten translation (access on CPU)
        at::Tensor translation_cpu = translation.cpu().contiguous();
        auto translation_acc = translation_cpu.accessor<float, 1>();
        translation_flat_list.push_back(translation_acc[0]);
        translation_flat_list.push_back(translation_acc[1]);
        translation_flat_list.push_back(translation_acc[2]);
        
        // Extract rotation matrix (inverse of rt[:3, :3])
        at::Tensor rotation = at::inverse(calibration.rt.slice(0, 0, 3).slice(1, 0, 3));
        rotation_list.push_back(rotation);
        
        // Allocate inpainting weights for this camera
        at::Tensor inpaint_weights = at::zeros({matching_rows, matching_cols, 2}, 
                                                at::TensorOptions().dtype(at::kByte).device(device));
        inpainting_weights_list_.push_back(inpaint_weights);
        
        // Create inpainting weights table using CUDA kernel
        std::cout << "Creating inpainting weights for camera " << (inpainting_weights_list_.size()) << "..." << std::endl;
        
        // Call CUDA wrapper with proper dimensions
        // Note: inpaint_weights is already [matching_rows, matching_cols, 2]
        launch_create_inpainting_weights(
            inpaint_weights,
            intrinsics,
            translation_device,
            matching_cols, matching_rows,
            min_dist, max_dist
        );
    }
    
    // Concatenate all rotations
    at::Tensor rotations = at::cat(rotation_list, 0).contiguous().to(device);
    
    // Create calibration and translation tensors for batch processing
    // Use at::tensor to create a new tensor that owns the data
    calibration_vectors_ = at::tensor(
        calib_flat_list,
        at::TensorOptions().dtype(at::kFloat).device(device)
    );
    
    translations_ = at::tensor(
        translation_flat_list,
        at::TensorOptions().dtype(at::kFloat).device(device)
    );
    
    // Process masks for blending weights
    std::cout << "Processing masks for blending..." << std::endl;
    at::Tensor masks_processed = masks.to(device).to(at::kFloat);
    
    // Pad masks for smoothing
    masks_processed = at::constant_pad_nd(
        masks_processed.unsqueeze(1),
        {smoothing_radius, smoothing_radius, smoothing_radius, smoothing_radius},
        1.0
    );
    
    // Apply box filter for smoothing
    at::Tensor conv_kernel = create_box_filter(smoothing_radius, device);
    masks_processed = at::conv2d(masks_processed, conv_kernel);
    masks_processed = masks_processed.squeeze(1).contiguous();
    
    // Create blending lookup tables using CUDA kernel
    std::cout << "Creating blending lookup tables..." << std::endl;
    
    // Prepare rotation parameters for CUDA kernel
    // Convert rotation matrices to RotationParams structure
    std::vector<RotationParams> rotation_params_list;
    rotation_params_list.reserve(num_cameras_);
    
    for (int i = 0; i < num_cameras_; i++) {
        RotationParams rot_params;
        at::Tensor rot_slice = rotations.slice(0, i * 3, (i + 1) * 3);
        
        // Ensure tensor is contiguous and on CPU for safe access
        rot_slice = rot_slice.contiguous().cpu();
        auto rot_acc = rot_slice.accessor<float, 2>();
        
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                rot_params.rotation.r[r][c] = rot_acc[r][c];
            }
        }
        rotation_params_list.push_back(rot_params);
    }
    
    // Call CUDA wrapper to create blending LUT
    // calibration_vectors_list_ contains Intrinsics structs (already in host memory)
    // rotation_params_list contains Rotation structs (already in host memory)
    // translations_ is a flat tensor on device
    auto [sampling_lut, blending_weights] = launch_create_blending_lut(
        blending_sampling_,
        blending_weights_,
        masks_processed,
        calibration_vectors_list_,  // std::vector<Intrinsics> - will be copied to device in wrapper
        rotation_params_list,        // std::vector<RotationParams> - will be copied to device in wrapper
        translations_,               // at::Tensor on device
        panorama_cols, panorama_rows,
        matching_cols, matching_rows,
        min_dist, max_dist
    );
    
    blending_sampling_ = sampling_lut;
    blending_weights_ = blending_weights;
    
    // Smooth blending weights to avoid seams
    std::cout << "Smoothing blending weights..." << std::endl;
    blending_weights_ = at::conv2d(
        blending_weights_.unsqueeze(1),
        conv_kernel,
        at::nullopt, 1, smoothing_radius
    ).squeeze(1);
    
    // Normalize blending weights
    at::Tensor weight_sum = at::sum(blending_weights_, 0, true);
    blending_weights_ = blending_weights_ / (weight_sum + 1e-8);
    
    std::cout << "Stitcher initialization complete!" << std::endl;
    std::cout << "  - Number of cameras: " << num_cameras_ << std::endl;
    std::cout << "  - Matching resolution: " << matching_cols_ << " x " << matching_rows_ << std::endl;
    std::cout << "  - Panorama resolution: " << panorama_cols_ << " x " << panorama_rows_ << std::endl;
    std::cout << "  - Inpainting iterations: " << inpainting_iterations_ << std::endl;
    std::cout << "  - Distance range: [" << min_dist << ", " << max_dist << "]" << std::endl;
}

Intrinsics Stitcher::vectorize_calibration(const Calibration& calib) {
    /**
     * Convert calibration to Intrinsics structure used by CUDA kernels
     * Scale focal length and principal point by matching_scale
     */
    Intrinsics intrinsics;
    
    intrinsics.fl.x = calib.fl.x * calib.matching_scale.x;
    intrinsics.fl.y = calib.fl.y * calib.matching_scale.y;
    intrinsics.principal.x = calib.principal.x * calib.matching_scale.x;
    intrinsics.principal.y = calib.principal.y * calib.matching_scale.y;
    intrinsics.xi = calib.xi;
    intrinsics.alpha = calib.alpha;
    
    return intrinsics;
}

at::Tensor Stitcher::create_box_filter(int radius, const at::Device& device) {
    /**
     * Create a box filter kernel for smoothing
     * Returns [1, 1, kernel_size, kernel_size] tensor
     */
    int kernel_size = 2 * radius + 1;
    at::Tensor kernel = at::ones({1, 1, kernel_size, kernel_size}, 
                                 at::TensorOptions().dtype(at::kFloat).device(device));
    kernel = kernel / at::sum(kernel);
    return kernel;
}

std::pair<at::Tensor, at::Tensor> Stitcher::stitch(
    const std::vector<at::Tensor>& images,
    const std::vector<at::Tensor>& distance_maps
) {
    /**
     * Stitch fisheye images and distance maps into RGB-D panorama
     */
    TORCH_CHECK(static_cast<int>(images.size()) == num_cameras_, 
                "Number of images must match number of cameras");
    TORCH_CHECK(static_cast<int>(distance_maps.size()) == num_cameras_, 
                "Number of distance maps must match number of cameras");
    
    // Process each camera view
    for (int i = 0; i < num_cameras_; i++) {
        const auto& image = images[i];
        const auto& distance_map = distance_maps[i];
        
        // Get corresponding tensors
        at::Tensor reprojected_distance = reprojected_distances_[i];
        at::Tensor distance_stack = distances_stacked_[i];
        at::Tensor image_to_stitch = images_to_stitch_[i];
        const Intrinsics& intrinsics = calibration_vectors_list_[i];
        const at::Tensor& translation = translations_list_[i];
        const at::Tensor& inpaint_weights = inpainting_weights_list_[i];
        
        // Reproject distance map to reference viewpoint using z-buffering
        // Run twice as in Python version for better occlusion handling
        reprojected_distance.fill_(1e8);
        
        for (int pass = 0; pass < 2; pass++) {
            // ZERO-COPY: Assume distance_map is already on device_ and contiguous
            launch_reproject_distance(
                distance_map,
                reprojected_distance,
                intrinsics,
                translation,
                matching_cols_, matching_rows_
            );
        }
        
        // Fill holes using inpainting (background-to-foreground)
        for (int iter = 0; iter < inpainting_iterations_; iter++) {
            launch_inpaint(
                reprojected_distance,
                inpaint_weights.contiguous(),
                matching_cols_, matching_rows_,
                max_dist_
            );
        }
        
        // ZERO-COPY: Assume image and distance_map are already on device_ and contiguous
        image_to_stitch.copy_(image);
        distance_stack.copy_(distance_map);
    }
    
    // Merge all views into panorama using CUDA kernel
    auto [distance_pano, rgb_pano] = launch_merge_rgbd_panorama(
        blending_sampling_,
        blending_weights_,
        reprojected_distances_,
        distances_stacked_,
        images_to_stitch_,
        translations_,
        calibration_vectors_list_,
        distance_panorama_,
        RGB_panorama_,
        panorama_cols_, panorama_rows_,
        matching_cols_, matching_rows_,
        images_to_stitch_.size(1), images_to_stitch_.size(2)
    );
    
    distance_panorama_ = distance_pano;
    RGB_panorama_ = rgb_pano;
    
    return std::make_pair(RGB_panorama_, distance_panorama_);
}

} // namespace my_stereo
