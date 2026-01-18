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
     * Uses adaptive spherical matching with cost volume filtering
     */
    
    int cols = matching_resolution_.first;
    int rows = matching_resolution_.second;
    
    // Create pixel grid and unproject to unit vectors
    // Python: meshgrid([v, u], indexing='ij') then stack([v, u])
    // C++: meshgrid({v, u}, "ij") gives grid[0]=v, grid[1]=u, then stack({u, v})
    auto u = at::arange(0, cols, at::TensorOptions().dtype(at::kFloat).device(device_));
    auto v = at::arange(0, rows, at::TensorOptions().dtype(at::kFloat).device(device_));
    auto grid = at::meshgrid({v, u}, "ij");  // grid[0]=v (rows), grid[1]=u (cols)
    auto uv = at::stack({grid[1], grid[0]}, /*dim=*/-1).unsqueeze(0);  // [1, H, W, 2] as (u, v)
    
    auto unproject_result = unproject(uv.reshape({-1, 2}), reference_calibration);
    auto pt_unit = unproject_result.first.reshape({1, rows, cols, 3});
    
    // Create sweeping volume: [candidate_count, H, W, 3]
    auto point_volume = distance_candidates_.view({candidate_count_, 1, 1, 1}) * 
                       pt_unit.view({1, rows, cols, 3});
    
    // Initialize sweeping volume for color matching
    auto sweeping_volume = at::zeros({1, 3, candidate_count_, rows, cols},
                                     at::TensorOptions().dtype(at::kFloat).device(device_));
    
    // Build sweeping volume using adaptive camera selection
    for (int cam_index = 0; cam_index < static_cast<int>(calibrations_.size()); ++cam_index) {
        const auto& calibration = calibrations_[cam_index];
        
        // Transform points to matched camera coordinate system
        auto rt = at::matmul(at::inverse(calibration.rt), reference_calibration.rt);
        auto ones = at::ones_like(point_volume.index({"...", at::indexing::Slice(at::indexing::None, 1)}));
        // PyTorch matmul handles broadcasting: [candidate_count, H, W, 4] @ [4, 4] -> [candidate_count, H, W, 4]
        auto point_volume_in_cam = at::matmul(
            at::cat({point_volume, ones}, /*dim=*/-1), rt.t());
        
        // Project to camera
        auto project_result = project(
            point_volume_in_cam.index({"...", at::indexing::Slice(at::indexing::None, 3)}).reshape({-1, 3}),
            calibration);
        auto uv_proj = project_result.first.reshape({candidate_count_, rows, cols, 2});
        
        // Normalize to [-1, 1] for grid_sample (align_corners=False)
        // Python: ((uv + 0.5) / [cols, rows]) * 2 - 1
        // This maps pixel [0, 0] center to [-1 + 1/res, -1 + 1/res]
        auto resolution_tensor = at::tensor({static_cast<float>(cols), static_cast<float>(rows)},
                                           at::TensorOptions().dtype(at::kFloat).device(device_));
        uv_proj = ((uv_proj + 0.5f) / resolution_tensor) * 2.0f - 1.0f;
        uv_proj = uv_proj.unsqueeze(0);
        uv_proj = at::cat({uv_proj, at::zeros_like(uv_proj.index({"...", at::indexing::Slice(at::indexing::None, 1)}))}, 
                         /*dim=*/-1);
        
        // Sample colors from matched camera
        auto image = images[cam_index];
        auto sweeping_volume_for_cam = at::grid_sampler(image, uv_proj,
                                                        /*interpolation_mode=*/0,
                                                        /*padding_mode=*/0,
                                                        /*align_corners=*/false);
        
        // Apply adaptive selection mask
        auto selected_mask = (selected_camera == cam_index);
        selected_mask = selected_mask.repeat({1, 3, candidate_count_, 1, 1});
        sweeping_volume.masked_scatter_(selected_mask, 
                                       sweeping_volume_for_cam.masked_select(selected_mask));
    }
    
    // Compute cost volume (photometric error)
    auto cost_volume = at::sum(at::abs(sweeping_volume - reference_image), /*dim=*/1).squeeze(0);
    cost_volume = at::clamp(cost_volume, /*min=*/at::nullopt, /*max=*/500.0f);
    
    // Filter cost volume
    auto cost_filter_result = cost_filter_->apply(guide.clone(), cost_volume.clone(), 
                                                  sigma_i_, sigma_s_);
    auto filtered_cost = cost_filter_result.first;
    
    // Select minimum cost
    auto min_result = at::min(filtered_cost, /*dim=*/0, /*keepdim=*/true);
    auto min_cost = std::get<0>(min_result);
    auto selected_index_map = std::get<1>(min_result);
    auto max_result = at::max(filtered_cost, /*dim=*/0, /*keepdim=*/true);
    auto max_cost = std::get<0>(max_result);
    
    // Quadratic fitting for sub-candidate accuracy
    auto left_cost = at::gather(filtered_cost, /*dim=*/0,
                                at::clamp(selected_index_map - 1, 0, candidate_count_ - 1));
    auto right_cost = at::gather(filtered_cost, /*dim=*/0,
                                 at::clamp(selected_index_map + 1, 0, candidate_count_ - 1));
    
    auto variation = 0.5f * (left_cost - right_cost) / 
                    ((left_cost + right_cost) - 2.0f * min_cost + 1e-8f);
    variation = at::clamp(variation, -0.5f, 0.5f);
    variation.masked_fill_(selected_index_map == (candidate_count_ - 1), 0.0f);
    variation.masked_fill_(selected_index_map == 0, 0.0f);
    
    auto selected_index_map_float = selected_index_map.to(at::kFloat) + variation;
    selected_index_map_float.masked_fill_(max_cost == min_cost, 
                                         static_cast<float>(candidate_count_ - 1));
    
    // Convert index to distance
    auto dist_0 = distance_candidates_[0];
    auto dist_last = distance_candidates_[candidate_count_ - 1];
    auto distance_map = dist_0 / ((dist_0 / dist_last - 1.0f) * 
                                  selected_index_map_float / (candidate_count_ - 1) + 1.0f);
    distance_map.masked_fill_(at::abs(max_cost - min_cost) < 1e-8f, dist_last);
    
    // Post-filter distance map with stronger edge preservation
    auto distance_filter_result = distance_filter_->apply(
        guide.clone(), distance_map.clone(), sigma_i_ / 2.0f, sigma_s_ / 2.0f);
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
     * Processing flow:
     * 1. Prepare images for matching (permute to [1, C, 1, H, W] format)
     * 2. For each reference camera:
     *    - Create YCbCr guide image (uint8)
     *    - Call estimate_fisheye_distance with ISBFilter
     * 3. Stitch distance maps into RGB-D panorama using Stitcher
     * 
     * Memory management: All operations on GPU with at::Tensor
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
    
    std::cout << "\n=== Estimating distance for each reference camera ===" << std::endl;
    for (size_t i = 0; i < references_indices_.size(); ++i) {
        int ref_idx = references_indices_[i];
        const auto& selected_camera = selected_cameras_[i];
        
        std::cout << "  Camera " << i << " (ref_idx=" << ref_idx << "):" << std::endl;
        
        // Verify calibration uniqueness
        auto& calib = calibrations_[ref_idx];
        std::cout << "    RT translation: [" 
                  << calib.rt[0][3].item<float>() << ", "
                  << calib.rt[1][3].item<float>() << ", "
                  << calib.rt[2][3].item<float>() << "]" << std::endl;
        
        // Verify selected_camera map has valid selections
        auto valid_selections = (selected_camera >= 0).sum().item<int>();
        auto total_pixels = selected_camera.numel();
        std::cout << "    Selected camera valid pixels: " << valid_selections 
                  << " / " << total_pixels << std::endl;
        
        // Create YCbCr guide image
        // Matches Python: guide = rgb2yCbCr(images_to_match[reference_index]).type(torch.uint8)
        auto guide = rgb_to_ycbcr(images_to_match[ref_idx]).to(at::kByte);
        
        // Estimate distance for this reference camera
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
        
        // Debug output: verify distance map statistics
        auto dist_mean = distance_map_independent.mean().item<float>();
        auto dist_min = distance_map_independent.min().item<float>();
        auto dist_max = distance_map_independent.max().item<float>();
        std::cout << "    Distance map stats: mean=" << dist_mean 
                  << ", min=" << dist_min 
                  << ", max=" << dist_max << std::endl;
        
        distance_maps.push_back(distance_map_independent);
    }
    
    // Verify distance_maps are unique
    if (distance_maps.size() > 1) {
        bool all_same = true;
        for (size_t i = 1; i < distance_maps.size(); ++i) {
            if (!at::equal(distance_maps[0], distance_maps[i])) {
                all_same = false;
                break;
            }
        }
        if (all_same) {
            std::cerr << "\n⚠️  WARNING: All distance_maps are identical! Bug detected.\n" << std::endl;
        } else {
            std::cout << "\n✓ Distance maps are unique for each camera.\n" << std::endl;
        }
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

RGBDEstimator::~RGBDEstimator() {
    free_unified_buffers();
}

void RGBDEstimator::allocate_unified_buffers() {
    int cols = matching_resolution_.first;
    int rows = matching_resolution_.second;
    int num_cameras = calibrations_.size();
    
    sweeping_volume_size_ = 1 * 3 * candidate_count_ * rows * cols * sizeof(float);
    cost_volume_size_ = candidate_count_ * rows * cols * sizeof(float);
    distance_map_size_ = 1 * rows * cols * sizeof(float);
    input_buffer_size_ = num_cameras * rows * cols * 3 * sizeof(float);
    
    cudaMallocManaged(&unified_sweeping_volume_ptr_, sweeping_volume_size_);
    cudaMallocManaged(&unified_cost_volume_ptr_, cost_volume_size_);
    cudaMallocManaged(&unified_distance_map_ptr_, distance_map_size_);
    cudaMallocManaged(&unified_input_buffer_ptr_, input_buffer_size_);
    
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
}

void RGBDEstimator::free_unified_buffers() {
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
} // namespace my_stereo_pkg
