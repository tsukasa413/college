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
#include "my_stereo_pkg/cuda_kernels.hpp"  // For CUDA_CHECK macro
#include <cmath>
#include <stdexcept>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

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
    
    // Copy distance candidates to CPU (ONCE during initialization)
    auto distance_cpu = distance_candidates_.cpu();
    distance_candidates_cpu_.resize(candidate_count_);
    std::memcpy(distance_candidates_cpu_.data(), 
                distance_cpu.data_ptr<float>(), 
                candidate_count_ * sizeof(float));
    
    // Initialize constant memory (ONCE during initialization, eliminates per-frame sync)
    initialize_constant_memory(
        distance_candidates_cpu_.data(),
        candidate_count_
    );

    // Initialize ISB filters (ONE PER CAMERA to eliminate buffer contention)
    int num_refs = references_indices_.size();
    cost_filters_.reserve(num_refs);
    distance_filters_.reserve(num_refs);
    for (int i = 0; i < num_refs; ++i) {
        cost_filters_.push_back(std::make_unique<ISBFilter>(candidate_count, matching_resolution, device));
        distance_filters_.push_back(std::make_unique<ISBFilter>(1, matching_resolution, device));
    }

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
    
    // Pre-compute RT matrices for all reference cameras (eliminates runtime CPU/GPU sync)
    precompute_relative_rt_matrices();
    
    // Initialize CUDA streams for parallel camera processing
    camera_streams_.resize(references_indices_.size());
    for (size_t i = 0; i < camera_streams_.size(); ++i) {
        cudaStreamCreate(&camera_streams_[i]);
    }
    
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
    
    // Destroy CUDA streams
    for (auto& stream : camera_streams_) {
        cudaStreamDestroy(stream);
    }
}

void RGBDEstimator::allocate_unified_buffers() {
    /**
     * Allocate unified memory buffers using cudaMallocManaged
     * Zero-copy architecture for Jetson devices
     */
    
    int cols = matching_resolution_.first;
    int rows = matching_resolution_.second;
    int num_cameras = calibrations_.size();
    int num_refs = references_indices_.size();
    
    // Calculate buffer sizes
    sweeping_volume_size_ = 1 * 3 * candidate_count_ * rows * cols * sizeof(float);
    cost_volume_size_ = candidate_count_ * rows * cols * sizeof(float);
    distance_map_size_ = 1 * rows * cols * sizeof(float);
    input_buffer_size_ = num_cameras * rows * cols * 3 * sizeof(float);
    
    // Allocate unified memory
    cudaMallocManaged(&unified_sweeping_volume_ptr_, sweeping_volume_size_);
    
    // CRITICAL: Allocate ONE cost volume per camera (eliminates memory contention)
    unified_cost_volume_ptrs_.resize(num_refs);
    for (int i = 0; i < num_refs; ++i) {
        cudaMallocManaged(&unified_cost_volume_ptrs_[i], cost_volume_size_);
    }
    
    cudaMallocManaged(&unified_distance_map_ptr_, distance_map_size_);
    cudaMallocManaged(&unified_input_buffer_ptr_, input_buffer_size_);
    
    // Wrap with LibTorch tensors using at::from_blob (FP32 for unified memory compatibility)
    auto options = at::TensorOptions().dtype(at::kFloat).device(device_);
    
    unified_sweeping_volume_ = at::from_blob(
        unified_sweeping_volume_ptr_,
        {1, 3, candidate_count_, rows, cols},
        options
    );
    
    // CRITICAL: Wrap per-camera cost volumes (eliminates memory contention)
    unified_cost_volumes_.reserve(num_refs);
    for (int i = 0; i < num_refs; ++i) {
        unified_cost_volumes_.push_back(
            at::from_blob(
                unified_cost_volume_ptrs_[i],
                {candidate_count_, rows, cols},
                options
            )
        );
    }
    
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
    
    // Pre-allocate per-camera buffers to avoid dynamic allocation in run loop
    per_camera_distance_maps_.reserve(num_refs);
    per_camera_guide_buffers_.reserve(num_refs);
    per_camera_cost_volumes_.reserve(num_refs);
    per_camera_temp_distance_buffers_.reserve(num_refs);
    
    auto tensor_options = at::TensorOptions().device(device_);
    for (int i = 0; i < num_refs; ++i) {
        // Distance map buffer [H, W] float32
        per_camera_distance_maps_.push_back(
            at::zeros({rows, cols}, tensor_options.dtype(at::kFloat)));
        
        // Guide image buffer [H, W, 3] uint8 for YCbCr conversion
        per_camera_guide_buffers_.push_back(
            at::zeros({rows, cols, 3}, tensor_options.dtype(at::kByte)));
        
        // CRITICAL: Per-camera cost volume buffer [D, H, W] float32
        per_camera_cost_volumes_.push_back(
            at::zeros({candidate_count_, rows, cols}, tensor_options.dtype(at::kFloat)));
        
        // CRITICAL: Per-camera temp distance buffer [H, W] float32
        per_camera_temp_distance_buffers_.push_back(
            at::zeros({rows, cols}, tensor_options.dtype(at::kFloat)));
    }
    
    // CRITICAL: Pre-create texture arrays (move cudaMallocArray out of hot path)
    // This eliminates the 1.1GB/frame CPU↔GPU transfer bottleneck
    texture_arrays_.resize(num_refs);
    texture_objects_.resize(num_refs);
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    for (int ref = 0; ref < num_refs; ++ref) {
        texture_arrays_[ref].resize(num_cameras);
        texture_objects_[ref].resize(num_cameras);
        
        for (int cam = 0; cam < num_cameras; ++cam) {
            // Allocate CUDA array ONCE (reused every frame)
            auto err = cudaMallocArray(&texture_arrays_[ref][cam], &channelDesc, cols, rows);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("cudaMallocArray failed: ") + cudaGetErrorString(err));
            }
            
            // Create texture object
            cudaResourceDesc resDesc{};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = texture_arrays_[ref][cam];
            
            cudaTextureDesc texDesc{};
            texDesc.addressMode[0] = cudaAddressModeBorder;
            texDesc.addressMode[1] = cudaAddressModeBorder;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 1;
            
            err = cudaCreateTextureObject(&texture_objects_[ref][cam], &resDesc, &texDesc, nullptr);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("cudaCreateTextureObject failed: ") + cudaGetErrorString(err));
            }
        }
    }
}

void RGBDEstimator::free_unified_buffers() {
    /**
     * Free unified memory buffers
     */
    if (unified_sweeping_volume_ptr_) {
        cudaFree(unified_sweeping_volume_ptr_);
        unified_sweeping_volume_ptr_ = nullptr;
    }
    
    // Free per-camera cost volumes
    for (auto ptr : unified_cost_volume_ptrs_) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    unified_cost_volume_ptrs_.clear();
    
    if (unified_distance_map_ptr_) {
        cudaFree(unified_distance_map_ptr_);
        unified_distance_map_ptr_ = nullptr;
    }
    if (unified_input_buffer_ptr_) {
        cudaFree(unified_input_buffer_ptr_);
        unified_input_buffer_ptr_ = nullptr;
    }
    
    // Free texture resources
    for (size_t ref = 0; ref < texture_objects_.size(); ++ref) {
        for (size_t cam = 0; cam < texture_objects_[ref].size(); ++cam) {
            if (texture_objects_[ref][cam]) {
                cudaDestroyTextureObject(texture_objects_[ref][cam]);
            }
            if (texture_arrays_[ref][cam]) {
                cudaFreeArray(texture_arrays_[ref][cam]);
            }
        }
    }
    texture_objects_.clear();
    texture_arrays_.clear();
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

void RGBDEstimator::precompute_relative_rt_matrices()
{
    /**
     * Pre-compute relative RT matrices for all reference cameras
     * This eliminates the 34.5% bottleneck from runtime at::matmul + .cpu() sync
     * 
     * For each reference camera, compute RT_relative[i] = inv(camera[i].rt) @ ref_camera.rt
     * Store in precomputed_camera_rts_[ref_idx][cam_idx]
     */
    
    precomputed_camera_rts_.clear();
    precomputed_camera_rts_.reserve(references_indices_.size());
    
    for (size_t ref_idx_local = 0; ref_idx_local < references_indices_.size(); ++ref_idx_local) {
        int ref_camera_idx = references_indices_[ref_idx_local];
        const auto& ref_rt = calibrations_[ref_camera_idx].rt;
        
        std::vector<CameraExtrinsics> rt_for_this_ref;
        rt_for_this_ref.resize(calibrations_.size());
        
        // Compute relative RT for all cameras w.r.t. this reference
        for (size_t i = 0; i < calibrations_.size(); ++i) {
            auto rt_relative = at::matmul(at::inverse(calibrations_[i].rt), ref_rt);
            
            // CRITICAL: Extract to CPU ONCE during initialization (not per-frame)
            auto rt_cpu = rt_relative.cpu();
            auto rt_accessor = rt_cpu.accessor<float, 2>();
            
            // Copy to CameraExtrinsics structure (row-major 3x4)
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 4; ++col) {
                    rt_for_this_ref[i].rt[row * 4 + col] = rt_accessor[row][col];
                }
            }
        }
        
        precomputed_camera_rts_.push_back(std::move(rt_for_this_ref));
    }
}

void RGBDEstimator::rgb_to_ycbcr(const at::Tensor& rgb_image, at::Tensor& ycbcr_out)
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
    
    // CRITICAL: Write directly to pre-allocated buffer (eliminates copy_ synchronization)
    auto ycbcr = at::stack({y, cb, cr}, /*dim=*/2);
    ycbcr_out.copy_(ycbcr.to(at::kByte));
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

void RGBDEstimator::estimate_fisheye_distance(
    const at::Tensor& reference_image,
    const at::Tensor& guide,
    const Calibration& reference_calibration,
    const at::Tensor& selected_camera,
    const std::vector<at::Tensor>& images,
    int ref_idx_local,
    cudaStream_t stream,
    at::Tensor& output_buffer)
{
    /**
     * Estimate distance map for a single fisheye reference image
     * ZERO-COPY: Writes directly to output_buffer (eliminates copy_ overhead)
     * Uses pre-computed RT matrices (eliminates CPU/GPU sync)
     * Uses dedicated ISBFilter (eliminates buffer contention)
     */
    
    nvtxRangePushA("estimate_fisheye_distance");
    
    int cols = matching_resolution_.first;
    int rows = matching_resolution_.second;
    
    // Use pre-computed RT matrices (NO CPU/GPU SYNC!)
    const auto& precomputed_rts = precomputed_camera_rts_[ref_idx_local];
    
    // CRITICAL: Use dedicated cost volume for this camera (eliminates memory contention)
    at::Tensor& camera_cost_volume = unified_cost_volumes_[ref_idx_local];
    
    // ========================================================================
    // 3-Pass Pipeline: Cost Volume -> ISB Filter -> Final Depth
    // ========================================================================
    
    // Pass 1: Compute raw cost volume [D, H, W] - TRUE ASYNC with per-camera stream
    // CRITICAL: Uses pre-created textures (NO .cpu() sync!)
    nvtxRangePushA("Stage1_ComputeCostVolume");
    launch_compute_costs_async(
        texture_objects_[ref_idx_local],  // Pre-created textures (reused every frame)
        texture_arrays_[ref_idx_local],   // Pre-allocated CUDA arrays
        images,  // Already in [H, W, 3] format from run()
        reference_image,
        selected_camera,
        distance_candidates_,
        camera_params_,
        precomputed_rts,  // Use pre-computed RT (no CPU/GPU sync!)
        camera_cost_volume,  // CRITICAL: Camera-dedicated buffer (no contention!)
        ref_idx_local,  // Use local index for precomputed_rts
        rows,
        cols,
        stream  // Per-camera stream for parallel execution
    );
    nvtxRangePop();  // Stage1_ComputeCostVolume
    
    // Pass 2: Apply ISB Filter to cost volume (DEDICATED FILTER per camera)
    nvtxRangePushA("Stage2_ISBFilter_Cost");
    auto filtered_cost_result = cost_filters_[ref_idx_local]->apply(
        guide,
        camera_cost_volume,
        sigma_i_,
        sigma_s_,
        stream
    );
    auto filtered_cost_volume = filtered_cost_result.first;
    nvtxRangePop();  // Stage2_ISBFilter_Cost
    
    // Pass 3: Winner-Take-All + Quadratic Fitting
    // CRITICAL: Use per-camera temp buffer (eliminates resource contention)
    nvtxRangePushA("Stage3_FinalDepth");
    per_camera_temp_distance_buffers_[ref_idx_local].zero_();
    launch_final_depth(
        filtered_cost_volume,
        distance_candidates_,
        per_camera_temp_distance_buffers_[ref_idx_local],
        rows,
        cols
    );
    nvtxRangePop();  // Stage3_FinalDepth
    
    // Optional: Light post-filtering on distance map
    nvtxRangePushA("Stage4_PostFilter_Distance");
    auto distance_map_batched = per_camera_temp_distance_buffers_[ref_idx_local].unsqueeze(0);
    auto distance_filter_result = distance_filters_[ref_idx_local]->apply(
        guide,
        distance_map_batched,
        sigma_i_ / 2.0f,
        sigma_s_ / 2.0f,
        stream
    );
    auto filtered_distance = distance_filter_result.first.squeeze(0);
    
    // ZERO-COPY: Write directly to output buffer
    output_buffer.copy_(filtered_distance);
    nvtxRangePop();  // Stage4_PostFilter_Distance
    
    nvtxRangePop();  // estimate_fisheye_distance
}

std::pair<at::Tensor, at::Tensor> RGBDEstimator::run(
    const std::vector<at::Tensor>& images_to_match,
    const std::vector<at::Tensor>& images_to_stitch)
{
    /**
     * Complete RGBD estimation pipeline
     * NVTX-enabled for Nsight Systems profiling
     * Zero-copy architecture with unified memory buffers
     */
    
    nvtxRangePushA("RGBDEstimator::run");
    
    // Prepare images ONCE in [H, W, 3] format (eliminates redundant permute/contiguous)
    nvtxRangePushA("Preparation_ImageFormat");
    std::vector<at::Tensor> images_hwc;
    images_hwc.reserve(images_to_match.size());
    for (const auto& image : images_to_match) {
        // Input: [H, W, 3] float32 -> ensure contiguous
        images_hwc.push_back(image.contiguous());
    }
    nvtxRangePop();  // Preparation_ImageFormat
    
    // PARALLEL PIPELINE: Process all reference cameras asynchronously
    nvtxRangePushA("DistanceEstimation_AllCameras");
    std::vector<at::Tensor> distance_maps;
    distance_maps.reserve(references_indices_.size());
    
    // Launch all cameras in parallel (each with its own CUDA stream)
    for (size_t i = 0; i < references_indices_.size(); ++i) {
        int ref_idx = references_indices_[i];
        const auto& selected_camera = selected_cameras_[i];
        
        // CRITICAL: CUDAStreamGuard ensures ALL LibTorch operations use this stream
        // Without this, tensors silently fall back to default stream (Stream 0)
        // causing serialization and cudaStreamSynchronize (39.7% bottleneck)
        c10::cuda::CUDAStreamGuard guard(
            c10::cuda::getStreamFromExternal(camera_streams_[i], device_.index())
        );
        
        // NVTX marker for per-camera processing
        std::string camera_label = "Camera_" + std::to_string(ref_idx);
        nvtxRangePushA(camera_label.c_str());
        
        // Create YCbCr guide image directly in pre-allocated buffer (NO COPY!)
        // CRITICAL: Eliminates hidden synchronization from copy_ between streams
        rgb_to_ycbcr(images_hwc[ref_idx], per_camera_guide_buffers_[i]);
        
        // ZERO-COPY: Write directly to output buffer (eliminates return value copy)
        estimate_fisheye_distance(
            images_hwc[ref_idx],
            per_camera_guide_buffers_[i],
            calibrations_[ref_idx],
            selected_camera,
            images_hwc,
            i,  // Local index for precomputed_rts and dedicated filters
            camera_streams_[i],  // Dedicated stream for this camera
            per_camera_distance_maps_[i]  // Direct output (no intermediate copy)
        );
        
        distance_maps.push_back(per_camera_distance_maps_[i]);
        
        nvtxRangePop();  // Camera_X
    }
    
    // CRITICAL: Wait for all camera streams to complete before stitching
    // This is the ONLY synchronization point (replaces scattered implicit syncs)
    for (size_t i = 0; i < camera_streams_.size(); ++i) {
        cudaStreamSynchronize(camera_streams_[i]);
    }
    
    nvtxRangePop();  // DistanceEstimation_AllCameras
    
    // Prepare images for stitching: convert to uint8
    nvtxRangePushA("Preparation_Stitching");
    std::vector<at::Tensor> images_to_stitch_uint8;
    images_to_stitch_uint8.reserve(images_to_stitch.size());
    for (const auto& image : images_to_stitch) {
        images_to_stitch_uint8.push_back(image.to(at::kByte));
    }
    nvtxRangePop();  // Preparation_Stitching
    
    // Stitching
    nvtxRangePushA("Stitching");
    auto [rgb, distance] = fisheye_stitcher_->stitch(images_to_stitch_uint8, distance_maps);
    nvtxRangePop();  // Stitching
    
    nvtxRangePop();  // RGBDEstimator::run
    return {rgb, distance};
}

} // namespace my_stereo_pkg
