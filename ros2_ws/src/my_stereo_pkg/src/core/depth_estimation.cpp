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

#include "my_stereo_pkg/depth_estimation.hpp"
#include <cmath>
#include <stdexcept>
#include <cuda_runtime.h>

namespace my_stereo_pkg {

RGBDEstimator::RGBDEstimator(
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
    const at::Device& device)
    : calibrations_(calibrations)
    , min_dist_(min_dist)
    , max_dist_(max_dist)
    , candidate_count_(candidate_count)
    , references_indices_(references_indices)
    , reprojection_viewpoint_(reprojection_viewpoint)
    , matching_resolution_(matching_resolution)
    , device_(device)
    , sigma_i_(sigma_i)
    , sigma_s_(sigma_s)
{
    // Pre-compute distance candidates using inverse linear interpolation
    // distance_candidates = 1 / linspace(1/min_dist, 1/max_dist, candidate_count)
    auto inv_distances = at::linspace(1.0f / min_dist_, 1.0f / max_dist_, 
                                      candidate_count_, 
                                      at::TensorOptions().dtype(at::kFloat).device(device_));
    distance_candidates_ = 1.0f / inv_distances;

    // Initialize ISB filters
    cost_filter_ = std::make_unique<ISBFilter>(candidate_count, matching_resolution, device);
    distance_filter_ = std::make_unique<ISBFilter>(1, matching_resolution, device);

    // Prepare calibrations and masks for stitcher
    std::vector<Calibration> calibrations_for_stitch;
    std::vector<at::Tensor> masks_for_stitching;
    for (int ref_idx : references_indices_) {
        calibrations_for_stitch.push_back(calibrations_[ref_idx]);
        // masks[ref_idx] is [1, H, W], squeeze to [H, W]
        masks_for_stitching.push_back(masks[ref_idx].squeeze(0));
    }
    
    // Stack masks into single tensor: [num_references, H, W]
    auto stacked_masks = at::stack(masks_for_stitching, /*dim=*/0);

    // Initialize stitcher
    fisheye_stitcher_ = std::make_unique<Stitcher>(
        calibrations_for_stitch,
        reprojection_viewpoint_,
        stacked_masks,
        min_dist_,
        max_dist_,
        matching_resolution.first,   // cols
        matching_resolution.second,  // rows
        rgb_to_stitch_resolution.first,
        rgb_to_stitch_resolution.second,
        panorama_resolution.first,
        panorama_resolution.second,
        device_
    );

    // Perform camera selection
    select_camera(masks);
    
    // Initialize camera parameters for fused CUDA kernel
    camera_params_.resize(calibrations_.size());
    camera_rts_.resize(calibrations_.size());
    
    for (size_t i = 0; i < calibrations_.size(); ++i) {
        const auto& calib = calibrations_[i];
        
        // Initialize DoubleSphereParams
        camera_params_[i].fx = calib.fl.x;
        camera_params_[i].fy = calib.fl.y;
        camera_params_[i].cx = calib.principal.x;
        camera_params_[i].cy = calib.principal.y;
        camera_params_[i].xi = calib.xi;
        camera_params_[i].alpha = calib.alpha;
        camera_params_[i].scale_x = calib.matching_scale.x;
        camera_params_[i].scale_y = calib.matching_scale.y;
        
        // Initialize CameraExtrinsics (identity for now, will be computed per reference)
        // RT matrix will be computed as inv(cam_rt) @ ref_rt in estimate_fisheye_distance
        for (int j = 0; j < 12; ++j) {
            camera_rts_[i].rt[j] = 0.0f;
        }
        // Set identity rotation
        camera_rts_[i].rt[0] = 1.0f;  // R00
        camera_rts_[i].rt[5] = 1.0f;  // R11
        camera_rts_[i].rt[10] = 1.0f; // R22
    }
    
    // Allocate unified memory buffers for zero-copy architecture
    allocate_unified_buffers();
}

std::pair<at::Tensor, at::Tensor> RGBDEstimator::unproject(
    const at::Tensor& uv,
    const Calibration& calib
) const
{
    /**
     * Unproject pixels to unit sphere using Double Sphere Camera Model
     * Reference: https://arxiv.org/abs/1807.08957
     * Python: utils.py unproject()
     */
    
    // Extract calibration parameters
    float fx = calib.fl.x;
    float fy = calib.fl.y;
    float cx = calib.principal.x;
    float cy = calib.principal.y;
    float xi = calib.xi;
    float alpha = calib.alpha;
    float scale_x = calib.matching_scale.x;
    float scale_y = calib.matching_scale.y;
    
    // m_xy = (uv - principal * matching_scale) / (fl * matching_scale)
    auto principal_scaled = at::tensor({cx * scale_x, cy * scale_y}, 
                                       at::TensorOptions().dtype(at::kFloat).device(device_));
    auto fl_scaled = at::tensor({fx * scale_x, fy * scale_y},
                                at::TensorOptions().dtype(at::kFloat).device(device_));
    
    auto m_xy = (uv - principal_scaled) / fl_scaled;
    
    // r2 = sum(m_xy^2, dim=-1, keepdim=True)
    auto r2 = at::sum(m_xy.pow(2), /*dim=*/-1, /*keepdim=*/true);
    
    // m_z = (1 - alpha^2 * r2) / (alpha * sqrt(clamp(1 - (2*alpha - 1)*r2, min=0)) + 1 - alpha)
    auto alpha_sq = alpha * alpha;
    auto two_alpha_minus_1 = 2.0f * alpha - 1.0f;
    auto sqrt_arg = at::clamp(1.0f - two_alpha_minus_1 * r2, /*min=*/0.0f);
    auto denominator = alpha * at::sqrt(sqrt_arg) + 1.0f - alpha;
    auto m_z = (1.0f - alpha_sq * r2) / denominator;
    
    // point = [m_xy, m_z]
    auto point = at::cat({m_xy, m_z}, /*dim=*/-1);
    
    // point = ((m_z * xi + sqrt(m_z^2 + (1 - xi^2) * r2)) / (m_z^2 + r2)) * point
    auto xi_sq = xi * xi;
    auto m_z_sq = m_z.pow(2);
    auto numerator = m_z * xi + at::sqrt(m_z_sq + (1.0f - xi_sq) * r2);
    auto denominator2 = m_z_sq + r2;
    point = (numerator / denominator2) * point;
    
    // point[..., 2] -= xi
    // Use select to ensure correct in-place modification
    auto point_z = point.select(-1, 2);
    point_z.sub_(xi);
    
    // valid = (1 - (2*alpha - 1)*r2 >= 0)
    auto valid = (1.0f - two_alpha_minus_1 * r2 >= 0.0f).squeeze(-1);
    
    return {point, valid};
}

RGBDEstimator::~RGBDEstimator() {
    free_unified_buffers();
}

void RGBDEstimator::allocate_unified_buffers() {
    /**
     * Allocate unified memory buffers using cudaMallocManaged
     * Zero-copy architecture for Jetson devices
     */
    
    int cols = matching_resolution_.first;
    int rows = matching_resolution_.second;
    int num_cameras = calibrations_.size();
    
    // Calculate buffer sizes
    sweeping_volume_size_ = 1 * 3 * candidate_count_ * rows * cols * sizeof(float);
    cost_volume_size_ = candidate_count_ * rows * cols * sizeof(float);
    distance_map_size_ = 1 * rows * cols * sizeof(float);
    input_buffer_size_ = num_cameras * rows * cols * 3 * sizeof(float);
    
    // Allocate unified memory
    cudaMallocManaged(&unified_sweeping_volume_ptr_, sweeping_volume_size_);
    cudaMallocManaged(&unified_cost_volume_ptr_, cost_volume_size_);
    cudaMallocManaged(&unified_distance_map_ptr_, distance_map_size_);
    cudaMallocManaged(&unified_input_buffer_ptr_, input_buffer_size_);
    
    // Wrap with LibTorch tensors using at::from_blob
    auto options = at::TensorOptions().dtype(at::kFloat).device(device_);
    
    unified_sweeping_volume_ = at::from_blob(
        unified_sweeping_volume_ptr_,
        {1, 3, candidate_count_, rows, cols},
        options
    );
    
    unified_cost_volume_ = at::from_blob(
        unified_cost_volume_ptr_,
        {candidate_count_, rows, cols},
        options
    );
    
    unified_distance_map_ = at::from_blob(
        unified_distance_map_ptr_,
        {1, rows, cols},
        options
    );
    
    unified_input_buffer_ = at::from_blob(
        unified_input_buffer_ptr_,
        {num_cameras, rows, cols, 3},
        options
    );
    
    std::cout << "\n=== Unified Memory Buffers Allocated ===" << std::endl;
    std::cout << "  Sweeping volume: " << sweeping_volume_size_ / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Cost volume: " << cost_volume_size_ / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Distance map: " << distance_map_size_ / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Input buffer: " << input_buffer_size_ / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Total: " << (sweeping_volume_size_ + cost_volume_size_ + 
                                distance_map_size_ + input_buffer_size_) / (1024.0 * 1024.0) 
              << " MB" << std::endl;
}

void RGBDEstimator::free_unified_buffers() {
    /**
     * Free unified memory buffers
     */
    if (unified_sweeping_volume_ptr_) {
        cudaFree(unified_sweeping_volume_ptr_);
        unified_sweeping_volume_ptr_ = nullptr;
    }
    if (unified_cost_volume_ptr_) {
        cudaFree(unified_cost_volume_ptr_);
        unified_cost_volume_ptr_ = nullptr;
    }
    if (unified_distance_map_ptr_) {
        cudaFree(unified_distance_map_ptr_);
        unified_distance_map_ptr_ = nullptr;
    }
    if (unified_input_buffer_ptr_) {
        cudaFree(unified_input_buffer_ptr_);
        unified_input_buffer_ptr_ = nullptr;
    }
}

std::pair<at::Tensor, at::Tensor> RGBDEstimator::project(
    const at::Tensor& points,
    const Calibration& calib
) const
{
    /**
     * Project 3D points to pixel coordinates using Double Sphere Camera Model
     * Reference: https://arxiv.org/abs/1807.08957
     * Python: utils.py project()
     */
    
    // Extract calibration parameters
    float fx = calib.fl.x;
    float fy = calib.fl.y;
    float cx = calib.principal.x;
    float cy = calib.principal.y;
    float xi = calib.xi;
    float alpha = calib.alpha;
    float scale_x = calib.matching_scale.x;
    float scale_y = calib.matching_scale.y;
    
    // d1 = norm(point, dim=-1, keepdim=True)
    auto d1 = at::norm(points, 2, /*dim=*/-1, /*keepdim=*/true);
    
    // c = xi * d1 + point[..., 2:3]
    auto c = xi * d1 + points.index({"...", at::indexing::Slice(2, 3)});
    
    // d2 = norm([point[..., :2], c], dim=-1, keepdim=True)
    auto point_xy = points.index({"...", at::indexing::Slice(at::indexing::None, 2)});
    auto d2 = at::norm(at::cat({point_xy, c}, /*dim=*/-1), 2, /*dim=*/-1, /*keepdim=*/true);
    
    // norm = alpha * d2 + (1 - alpha) * c
    auto norm = alpha * d2 + (1.0f - alpha) * c;
    
    // Compute validity threshold w2
    float w1, w2;
    if (alpha > 0.5f) {
        w1 = (1.0f - alpha) / alpha;
    } else {
        w1 = alpha / (1.0f - alpha);
    }
    w2 = (w1 + xi) / std::sqrt(2.0f * w1 * xi + xi * xi + 1.0f);
    
    // valid = point[..., 2:3] > -w2 * d1
    auto point_z = points.index({"...", at::indexing::Slice(2, 3)});
    auto valid = (point_z > -w2 * d1).squeeze(-1);
    
    // uv = (fl * matching_scale * point[..., :2]) / norm + principal * matching_scale
    auto principal_scaled = at::tensor({cx * scale_x, cy * scale_y},
                                       at::TensorOptions().dtype(at::kFloat).device(device_));
    auto fl_scaled = at::tensor({fx * scale_x, fy * scale_y},
                                at::TensorOptions().dtype(at::kFloat).device(device_));
    
    auto uv = (fl_scaled * point_xy) / norm + principal_scaled;
    
    return {uv, valid};
}

at::Tensor RGBDEstimator::rgb_to_ycbcr(const at::Tensor& rgb_image)
{
    // RGB to YCbCr conversion - MUST match Python utils.py rgb2yCbCr exactly
    // Note: Input is BGR from OpenCV, treated as RGB by Python code
    // Y  = 16  + 0.1826*R + 0.6142*G + 0.062*B,  clamp [16, 235]
    // Cb = 128 - 0.1006*R - 0.3386*G + 0.4392*B, clamp [16, 240]
    // Cr = 128 + 0.4392*R - 0.3989*G - 0.0403*B, clamp [16, 240]
    
    TORCH_CHECK(rgb_image.dim() == 3 && rgb_image.size(2) == 3,
                "RGB image must be [H, W, 3]");
    
    auto rgb = rgb_image.to(at::kFloat);
    auto r = rgb.index({at::indexing::Slice(), at::indexing::Slice(), 0});
    auto g = rgb.index({at::indexing::Slice(), at::indexing::Slice(), 1});
    auto b = rgb.index({at::indexing::Slice(), at::indexing::Slice(), 2});
    
    // Exact coefficients from Python utils.py
    auto y  = (16.0f   + 0.1826f * r + 0.6142f * g + 0.062f  * b).clamp(16.0f, 235.0f);
    auto cb = (128.0f  - 0.1006f * r - 0.3386f * g + 0.4392f * b).clamp(16.0f, 240.0f);
    auto cr = (128.0f  + 0.4392f * r - 0.3989f * g - 0.0403f * b).clamp(16.0f, 240.0f);
    
    auto ycbcr = at::stack({y, cb, cr}, /*dim=*/2);
    return ycbcr.to(at::kByte);
}

void RGBDEstimator::select_camera(const std::vector<at::Tensor>& masks)
{
    /**
     * Adaptive camera selection (Section 3.1 of the paper)
     * For each reference camera, select the best matching camera per pixel
     * based on maximum displacement from min_dist to max_dist
     */
    
    selected_cameras_.clear();
    selected_cameras_.reserve(references_indices_.size());
    
    int cols = matching_resolution_.first;
    int rows = matching_resolution_.second;
    
    for (int ref_idx : references_indices_) {
        const auto& reference_calibration = calibrations_[ref_idx];
        
        // Initialize with invalid camera index (-1)
        auto selected_camera = -at::ones({1, rows, cols}, 
                                         at::TensorOptions().dtype(at::kInt).device(device_));
        // Initialize max_displacement to -1.0 to ensure first valid result is selected
        auto max_displacement = -at::ones({1, rows, cols},
                                          at::TensorOptions().dtype(at::kFloat).device(device_));
        
        // Create pixel grid
        // Python: meshgrid([v, u], indexing='ij') then stack([v, u])
        // C++: meshgrid({v, u}, "ij") gives grid[0]=v, grid[1]=u, then stack({u, v})
        auto u = at::arange(0, cols, at::TensorOptions().dtype(at::kFloat).device(device_));
        auto v = at::arange(0, rows, at::TensorOptions().dtype(at::kFloat).device(device_));
        auto grid = at::meshgrid({v, u}, "ij");  // grid[0]=v (rows), grid[1]=u (cols)
        auto uv = at::stack({grid[1], grid[0]}, /*dim=*/-1).unsqueeze(0);  // [1, H, W, 2] as (u, v)
        
        // Unproject to unit vectors
        auto unproject_result = unproject(uv.reshape({-1, 2}), reference_calibration);
        auto pt_unit = unproject_result.first.reshape({1, rows, cols, 3});
        auto reference_valid = unproject_result.second.reshape({1, rows, cols});
        
        // Iterate through all cameras to find best match per pixel
        for (int cam_index = 0; cam_index < static_cast<int>(calibrations_.size()); ++cam_index) {
            const auto& calibration = calibrations_[cam_index];
            const auto& mask = masks[cam_index];
            
            // Compute points at near and far distances
            auto pt_near = pt_unit * min_dist_;
            auto pt_far = pt_unit * max_dist_;
            
            // Transform to matched camera coordinate system
            auto rt = at::matmul(at::inverse(calibration.rt), reference_calibration.rt);
            
            // Convert to homogeneous coordinates and transform
            // PyTorch matmul handles broadcasting: [1, H, W, 4] @ [4, 4] -> [1, H, W, 4]
            auto ones = at::ones_like(pt_near.index({"...", at::indexing::Slice(at::indexing::None, 1)}));
            auto pt_near_homo = at::cat({pt_near, ones}, /*dim=*/-1);  // [1, H, W, 4]
            auto pt_far_homo = at::cat({pt_far, ones}, /*dim=*/-1);    // [1, H, W, 4]
            
            pt_near_homo = at::matmul(pt_near_homo, rt.t());
            pt_far_homo = at::matmul(pt_far_homo, rt.t());
            
            // Normalize to unit vectors
            auto pt_near_xyz = pt_near_homo.index({"...", at::indexing::Slice(at::indexing::None, 3)});
            auto pt_far_xyz = pt_far_homo.index({"...", at::indexing::Slice(at::indexing::None, 3)});
            pt_near_xyz = pt_near_xyz / at::norm(pt_near_xyz, 2, /*dim=*/-1, /*keepdim=*/true);
            pt_far_xyz = pt_far_xyz / at::norm(pt_far_xyz, 2, /*dim=*/-1, /*keepdim=*/true);
            
            // Project to matched camera
            auto project_near_result = project(pt_near_xyz.reshape({-1, 3}), calibration);
            auto uv_near = project_near_result.first;
            auto valid_near = project_near_result.second;
            auto project_far_result = project(pt_far_xyz.reshape({-1, 3}), calibration);
            auto uv_far = project_far_result.first;
            auto valid_far = project_far_result.second;
            
            uv_near = uv_near.reshape({1, rows, cols, 2});
            uv_far = uv_far.reshape({1, rows, cols, 2});
            valid_near = valid_near.reshape({1, rows, cols});
            valid_far = valid_far.reshape({1, rows, cols});
            
            // Calculate displacement
            auto displacement = at::norm(uv_near - uv_far, 2, /*dim=*/-1);
            
            // Normalize UV coordinates to [-1, 1] for grid_sample (align_corners=False)
            // Python: ((uv + 0.5) / [cols, rows]) * 2 - 1
            auto resolution_tensor = at::tensor({static_cast<float>(cols), static_cast<float>(rows)},
                                               at::TensorOptions().dtype(at::kFloat).device(device_));
            uv_near = ((uv_near + 0.5f) / resolution_tensor) * 2.0f - 1.0f;
            uv_far = ((uv_far + 0.5f) / resolution_tensor) * 2.0f - 1.0f;
            
            // Sample mask at projected locations
            auto mask_near = at::grid_sampler(mask.unsqueeze(0), uv_near, 
                                             /*interpolation_mode=*/0, 
                                             /*padding_mode=*/0, 
                                             /*align_corners=*/false).squeeze(0);
            auto mask_far = at::grid_sampler(mask.unsqueeze(0), uv_far,
                                            /*interpolation_mode=*/0,
                                            /*padding_mode=*/0,
                                            /*align_corners=*/false).squeeze(0);
            
            // Determine which pixels should use this camera
            auto current_best = (displacement > max_displacement)
                              & reference_valid
                              & valid_near & valid_far
                              & (masks[ref_idx] >= 0.9f)
                              & (mask_near >= 0.9f)
                              & (mask_far >= 0.9f);
            
            // Update selection using where() to selectively update values
            max_displacement = at::where(current_best, displacement, max_displacement);
            selected_camera = at::where(current_best, 
                                       at::full_like(selected_camera, cam_index), 
                                       selected_camera);
        }
        
        selected_cameras_.push_back(selected_camera);
    }
}

at::Tensor RGBDEstimator::estimate_fisheye_distance(
    const at::Tensor& reference_image,
    const at::Tensor& guide,
    const Calibration& reference_calibration,
    const at::Tensor& selected_camera,
    const std::vector<at::Tensor>& images)
{
    /**
     * Estimate distance map for a single fisheye reference image
     * Uses fused CUDA kernel with texture acceleration (30+ FPS)
     */
    
    int cols = matching_resolution_.first;
    int rows = matching_resolution_.second;
    
    // Find reference camera index
    int ref_camera_idx = -1;
    for (size_t i = 0; i < calibrations_.size(); ++i) {
        if (&calibrations_[i] == &reference_calibration) {
            ref_camera_idx = static_cast<int>(i);
            break;
        }
    }
    TORCH_CHECK(ref_camera_idx >= 0, "Reference camera not found in calibrations");
    
    // Compute relative RT matrices for all cameras
    // RT = inv(cam_rt) @ ref_rt
    auto ref_rt = reference_calibration.rt;
    auto ref_rt_inv = at::inverse(ref_rt);
    
    for (size_t i = 0; i < calibrations_.size(); ++i) {
        auto rt_relative = at::matmul(at::inverse(calibrations_[i].rt), ref_rt);
        
        // Copy to CameraExtrinsics structure (row-major 3x4)
        auto rt_cpu = rt_relative.cpu();
        auto rt_accessor = rt_cpu.accessor<float, 2>();
        
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 4; ++col) {
                camera_rts_[i].rt[row * 4 + col] = rt_accessor[row][col];
            }
        }
    }
    
    // Convert images from [1, 3, 1, H, W] to [H, W, 3] for CUDA kernel
    std::vector<at::Tensor> images_hwc;
    images_hwc.reserve(images.size());
    for (const auto& img : images) {
        // [1, 3, 1, H, W] -> squeeze -> [3, H, W] -> permute -> [H, W, 3]
        auto img_squeezed = img.squeeze(0).squeeze(1);  // [3, H, W]
        auto img_hwc = img_squeezed.permute({1, 2, 0}).contiguous();  // [H, W, 3]
        images_hwc.push_back(img_hwc);
    }
    
    // Convert reference_image from [1, 3, 1, H, W] to [H, W, 3]
    auto ref_image_hwc = reference_image.squeeze(0).squeeze(1).permute({1, 2, 0}).contiguous();
    
    // Convert selected_camera from [1, H, W] to [H, W]
    auto selected_camera_squeezed = selected_camera.squeeze(0).contiguous();
    
    // ========================================================================
    // 3-Pass Pipeline: Cost Volume -> ISB Filter -> Final Depth
    // ========================================================================
    
    // Pass 1: Compute raw cost volume [D, H, W]
    // Uses texture acceleration and Double Sphere projection
    launch_compute_costs(
        images_hwc,
        ref_image_hwc,
        selected_camera_squeezed,
        distance_candidates_,
        camera_params_,
        camera_rts_,
        unified_cost_volume_,  // Output: [candidate_count, H, W]
        ref_camera_idx,
        rows,
        cols
    );
    
    // Pass 2: Apply ISB Filter to cost volume
    // Edge-preserving smoothing in cost space (critical for accuracy)
    // Python: cost_volume, _ = self.cost_filter.apply(guide, cost_volume, sigma_i, sigma_s)
    auto filtered_cost_result = cost_filter_->apply(
        guide.clone(), 
        unified_cost_volume_,  // Input: raw cost [D, H, W]
        sigma_i_,              // Color similarity threshold
        sigma_s_               // Spatial similarity threshold
    );
    auto filtered_cost_volume = filtered_cost_result.first;  // [D, H, W]
    
    // Pass 3: Winner-Take-All + Quadratic Fitting
    // Compute final depth from ISB-filtered cost volume
    auto distance_map_raw = at::zeros({rows, cols}, 
                                     at::TensorOptions().dtype(at::kFloat).device(device_));
    
    launch_final_depth(
        filtered_cost_volume,
        distance_candidates_,
        distance_map_raw,
        rows,
        cols
    );
    
    // Optional: Light post-filtering on distance map (much weaker than Python version)
    // Python version doesn't apply distance filter after cost filter
    // We apply very light filtering only for noise reduction
    auto distance_map_batched = distance_map_raw.unsqueeze(0);
    auto distance_filter_result = distance_filter_->apply(
        guide.clone(), 
        distance_map_batched, 
        sigma_i_ * 2.0f,  // Much weaker color preservation
        sigma_s_ * 2.0f   // Much weaker spatial preservation
    );
    auto filtered_distance = distance_filter_result.first;
    
    return filtered_distance.squeeze(0);
}

std::pair<at::Tensor, at::Tensor> RGBDEstimator::run(
    const std::vector<at::Tensor>& images_to_match,
    const std::vector<at::Tensor>& images_to_stitch)
{
    /**
     * Complete RGBD estimation pipeline
     * Faithfully ported from depth_estimation.py estimate_RGBD_panorama()
     * 
     * Processing flow (Zero-Copy Architecture):
     * 1. Prepare images for matching (permute to [1, C, 1, H, W] format)
     * 2. For each reference camera:
     *    - Create YCbCr guide image (uint8)
     *    - Call estimate_fisheye_distance with ISBFilter (uses unified buffers)
     * 3. Stitch distance maps into RGB-D panorama using Stitcher
     * 
     * Memory management: Unified memory buffers pre-allocated in constructor
     */
    
    // Prepare images for matching: permute to [1, C, 1, H, W] format
    // Matches Python: images_to_match_permuted = [image.unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(2)
    //                                              for image in images_to_match]
    std::vector<at::Tensor> images_to_match_permuted;
    images_to_match_permuted.reserve(images_to_match.size());
    for (const auto& image : images_to_match) {
        // image: [H, W, 3] -> unsqueeze(0) -> [1, H, W, 3]
        // -> permute(0,3,1,2) -> [1, 3, H, W]
        // -> unsqueeze(2) -> [1, 3, 1, H, W]
        auto permuted = image.unsqueeze(0).permute({0, 3, 1, 2}).unsqueeze(2);
        images_to_match_permuted.push_back(permuted);
    }
    
    // Estimate distance for each reference camera
    // Matches Python: for reference_index, selected_camera in zip(self.references_indices, self.selected_cameras)
    std::vector<at::Tensor> distance_maps;
    distance_maps.reserve(references_indices_.size());
    
    for (size_t i = 0; i < references_indices_.size(); ++i) {
        int ref_idx = references_indices_[i];
        const auto& selected_camera = selected_cameras_[i];
        
        // Create YCbCr guide image
        // Matches Python: guide = rgb2yCbCr(images_to_match[reference_index]).type(torch.uint8)
        auto guide = rgb_to_ycbcr(images_to_match[ref_idx]).to(at::kByte);
        
        // Estimate distance for this reference camera (uses unified buffers)
        // Calls ISBFilter::apply internally
        auto distance_map = estimate_fisheye_distance(
            images_to_match_permuted[ref_idx],
            guide,
            calibrations_[ref_idx],
            selected_camera,
            images_to_match_permuted
        );
        
        // CRITICAL: Ensure distance_map is independent by cloning
        // This prevents all elements pointing to the same memory
        auto distance_map_independent = distance_map.clone();
        
        distance_maps.push_back(distance_map_independent);
    }
    
    // Prepare images for stitching: convert to uint8
    // Matches Python: images_to_stitch = [reference_image.type(torch.uint8)
    //                                      for reference_image in images_to_stitch]
    std::vector<at::Tensor> images_to_stitch_uint8;
    images_to_stitch_uint8.reserve(images_to_stitch.size());
    for (const auto& image : images_to_stitch) {
        images_to_stitch_uint8.push_back(image.to(at::kByte));
    }
    
    // Stitch into RGB-D panorama
    // Matches Python: rgb, distance = self.fisheye_stitcher.stitch(images_to_stitch, distance_maps)
    auto [rgb, distance] = fisheye_stitcher_->stitch(images_to_stitch_uint8, distance_maps);
    
    return {rgb, distance};
}

} // namespace my_stereo_pkg
