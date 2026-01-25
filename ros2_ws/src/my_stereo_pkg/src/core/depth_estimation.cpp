/*
 * depth_estimation.cpp
 *
 * CPU-side implementation of RGBD_Estimator
 * Manages GPU memory, coordinates kernel launches, and handles data transfers
 * 
 * Optimization Strategy:
 * - Zero-Copy: Minimize CPU-GPU transfers
 * - cudaMallocPitch: Aligned 2D arrays for coalesced memory access
 * - Async Streams: Overlap processing of multiple reference cameras
 * - Constant Memory: Store camera calibrations for fast access
 */

#include "my_stereo_pkg/depth_estimation.hpp"
#include "my_stereo_pkg/isb_filter.hpp"
#include "my_stereo_pkg/stitcher.hpp"
#include "my_stereo_pkg/calibration.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <torch/torch.h>

// CUDA error checking macro
#define CUDA_CHECK(err) \
    do { \
        cudaError_t error = (err); \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(error)); \
        } \
    } while(0)

// ============================================================================
// Forward Declarations of Kernel Launchers
// ============================================================================

extern void select_best_cameras_kernel(
    int* d_selected_cameras,
    float* d_max_displacement,
    const DoubleSphereCalibration* d_calibrations,
    const float* const* d_masks,
    const DoubleSphereCalibration& reference_calib,
    const CameraConfig& config,
    cudaStream_t stream
);

extern void estimate_fisheye_distance_fused_kernel(
    float* d_distance_map,
    const uchar4* d_reference_image,
    const uchar4* const* d_images,
    const int* d_selected_cameras,
    const DoubleSphereCalibration* d_calibrations,
    const uchar* d_guide,
    const CameraConfig& config,
    cudaStream_t stream,
    cudaTextureObject_t* d_texobjs
);

// ============================================================================
// RGBD_Estimator Constructor
// ============================================================================

RGBD_Estimator::RGBD_Estimator(
    const std::vector<float>& calibrations_rt,
    const std::vector<float>& calibrations_intrinsics,
    const std::vector<float>& calibrations_sphere,
    const std::vector<float>& calibrations_resolution,
    float min_dist,
    float max_dist,
    int candidate_count,
    const std::vector<int>& references_indices,
    const std::vector<float>& reprojection_viewpoint,
    const std::vector<int>& image_widths,
    const std::vector<int>& image_heights,
    int matching_width,
    int matching_height,
    int rgb_to_stitch_width,
    int rgb_to_stitch_height,
    int panorama_width,
    int panorama_height,
    float sigma_i,
    float sigma_s,
    int device
)
    : device_id_(device),
      num_cameras_(image_widths.size()),
      image_widths_(image_widths),
      image_heights_(image_heights),
      rgb_to_stitch_width_(rgb_to_stitch_width),
      rgb_to_stitch_height_(rgb_to_stitch_height),
      panorama_width_(panorama_width),
      panorama_height_(panorama_height),
      sigma_i_(sigma_i),
      sigma_s_(sigma_s),
      references_indices_(references_indices),
      d_calibrations_(nullptr),
      d_distance_map_(nullptr),
      distance_pitch_(0),
      d_cost_volume_(nullptr),
      cost_volume_pitch_(0)
{
    CUDA_CHECK(cudaSetDevice(device_id_));
    
    // ========================================================================
    // Initialize Configuration
    // ========================================================================
    
    config_.num_cameras = num_cameras_;
    config_.matching_width = matching_width;
    config_.matching_height = matching_height;
    config_.min_dist = min_dist;
    config_.max_dist = max_dist;
    config_.candidate_count = candidate_count;
    
    matching_pixels_ = matching_width * matching_height;
    
    // ========================================================================
    // Build Calibration Structures
    // ========================================================================
    
    calibrations_.resize(num_cameras_);
    for (int i = 0; i < num_cameras_; i++) {
        DoubleSphereCalibration& calib = calibrations_[i];
        
        // Copy RT matrix (row-major 4x4)
        for (int j = 0; j < 16; j++) {
            calib.rt[j] = calibrations_rt[i * 16 + j];
        }
        
        // Extract R (3x3) and t (3) from rt matrix for new format
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                calib.R[row * 3 + col] = calib.rt[row * 4 + col];
            }
            calib.t[row] = calib.rt[row * 4 + 3];
        }
        
        // Copy intrinsics [fx, fy, cx, cy]
        calib.fx = calibrations_intrinsics[i * 4 + 0];
        calib.fy = calibrations_intrinsics[i * 4 + 1];
        calib.cx = calibrations_intrinsics[i * 4 + 2];
        calib.cy = calibrations_intrinsics[i * 4 + 3];
        
        // Copy sphere parameters [xi, alpha]
        calib.xi = calibrations_sphere[i * 2 + 0];
        calib.alpha = calibrations_sphere[i * 2 + 1];
        
        // Original resolution from calibration data
        float original_width = calibrations_resolution[i * 2 + 0];
        float original_height = calibrations_resolution[i * 2 + 1];
        
        // Matching resolution (target size for depth estimation)
        calib.width = matching_width;
        calib.height = matching_height;
        
        // Matching scale: ratio of matching to original resolution
        // This matches Python's calib.matching_scale
        calib.matching_scale = (original_width > 0) ? 
            static_cast<float>(matching_width) / original_width : 1.0f;
        
        calib.padding = 0;
    }
    
    // ========================================================================
    // Allocate GPU Memory
    // ========================================================================
    
    allocate_gpu_memory();
    
    // ========================================================================
    // Pre-compute Relative RT Matrices (avoid runtime computation)
    // ========================================================================
    
    precompute_relative_calibrations();
    upload_calibrations();
    
    // ========================================================================
    // Create CUDA Streams for Async Processing
    // ========================================================================
    
    streams_.resize(references_indices_.size());
    for (size_t i = 0; i < streams_.size(); i++) {
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }
    
    // ========================================================================
    // Allocate Host-side Temporary Buffers
    // ========================================================================
    
    h_reference_image_.resize(matching_width * matching_height * 3);
    h_guide_image_.resize(matching_width * matching_height);
    
    h_target_images_.resize(num_cameras_);
    for (int i = 0; i < num_cameras_; i++) {
        h_target_images_[i].resize(matching_width * matching_height * 3);
    }
}

// ============================================================================
// RGBD_Estimator Destructor
// ============================================================================

RGBD_Estimator::~RGBD_Estimator() {
    CUDA_CHECK(cudaSetDevice(device_id_));
    
    deallocate_gpu_memory();
    
    // Destroy streams
    for (auto& stream : streams_) {
        cudaStreamDestroy(stream);
    }
}

// ============================================================================
// Memory Allocation
// ============================================================================

void RGBD_Estimator::allocate_gpu_memory() {
    CUDA_CHECK(cudaSetDevice(device_id_));
    
    // Allocate calibrations in device memory (will be copied to constant memory)
    CUDA_CHECK(cudaMalloc(&d_calibrations_, num_cameras_ * sizeof(DoubleSphereCalibration)));
    
    // Distance map: [matching_height, matching_width] with pitched memory
    CUDA_CHECK(cudaMallocPitch(&d_distance_map_, &distance_pitch_,
                               config_.matching_width * sizeof(float),
                               config_.matching_height));
    
    // Cost volume with pitched memory for better coalescing
    CUDA_CHECK(cudaMallocPitch(&d_cost_volume_, &cost_volume_pitch_,
                               config_.matching_width * config_.candidate_count * sizeof(float),
                               config_.matching_height));
    
    // Pre-allocate target images to avoid runtime malloc/free
    d_target_images_.resize(num_cameras_);
    d_target_arrays_.resize(num_cameras_);
    d_image_texobjs_.resize(num_cameras_);
    
    for (int i = 0; i < num_cameras_; i++) {
        // Allocate image memory
        CUDA_CHECK(cudaMalloc(&d_target_images_[i],
                              config_.matching_width * config_.matching_height * sizeof(uchar4)));
        
        // Create CUDA array for texture
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
        CUDA_CHECK(cudaMallocArray(&d_target_arrays_[i], &channel_desc,
                                   config_.matching_width, config_.matching_height));
    }
    
    // Allocate validity masks
    d_masks_.resize(num_cameras_);
    for (int i = 0; i < num_cameras_; i++) {
        size_t mask_pitch;
        CUDA_CHECK(cudaMallocPitch(&d_masks_[i], &mask_pitch,
                                   config_.matching_width * sizeof(float),
                                   config_.matching_height));
    }
}

void RGBD_Estimator::deallocate_gpu_memory() {
    CUDA_CHECK(cudaSetDevice(device_id_));
    
    if (d_calibrations_) {
        CUDA_CHECK(cudaFree(d_calibrations_));
        d_calibrations_ = nullptr;
    }
    
    if (d_distance_map_) {
        CUDA_CHECK(cudaFree(d_distance_map_));
        d_distance_map_ = nullptr;
    }
    
    if (d_cost_volume_) {
        CUDA_CHECK(cudaFree(d_cost_volume_));
        d_cost_volume_ = nullptr;
    }
    
    // Free pre-allocated target images
    for (auto& img_ptr : d_target_images_) {
        if (img_ptr) {
            CUDA_CHECK(cudaFree(img_ptr));
        }
    }
    d_target_images_.clear();
    
    // Destroy texture objects
    for (auto& texobj : d_image_texobjs_) {
        if (texobj) {
            CUDA_CHECK(cudaDestroyTextureObject(texobj));
        }
    }
    d_image_texobjs_.clear();
    
    // Free CUDA arrays
    for (auto& array_ptr : d_target_arrays_) {
        if (array_ptr) {
            CUDA_CHECK(cudaFreeArray(array_ptr));
        }
    }
    d_target_arrays_.clear();
    
    for (auto& mask_ptr : d_masks_) {
        if (mask_ptr) {
            CUDA_CHECK(cudaFree(mask_ptr));
        }
    }
    d_masks_.clear();
}

void RGBD_Estimator::upload_calibrations() {
    CUDA_CHECK(cudaSetDevice(device_id_));
    
    // Copy to device memory first (for parameter access)
    CUDA_CHECK(cudaMemcpy(d_calibrations_, calibrations_.data(),
                          num_cameras_ * sizeof(DoubleSphereCalibration),
                          cudaMemcpyHostToDevice));
    
    // TODO: Migrate to constant memory for better performance
    // When ready: CUDA_CHECK(cudaMemcpyToSymbol(c_calibrations, calibrations_.data(), ...));
}

// ============================================================================
// RGB to YCbCr Conversion (Host-side)
// ============================================================================

/**
 * 4x4 Matrix inverse (row-major)
 * Simplified for rigid body transformations (rotation + translation)
 */
static void invert_rt_matrix(const float src[16], float dst[16]) {
    // For rigid body transformation [R | t]
    //                                [0 | 1]
    // Inverse is [R^T | -R^T*t]
    //            [0   | 1      ]
    
    // Transpose rotation part (3x3)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            dst[i*4 + j] = src[j*4 + i];
        }
    }
    
    // Compute -R^T * t
    dst[0*4 + 3] = -(dst[0*4+0] * src[0*4+3] + dst[0*4+1] * src[1*4+3] + dst[0*4+2] * src[2*4+3]);
    dst[1*4 + 3] = -(dst[1*4+0] * src[0*4+3] + dst[1*4+1] * src[1*4+3] + dst[1*4+2] * src[2*4+3]);
    dst[2*4 + 3] = -(dst[2*4+0] * src[0*4+3] + dst[2*4+1] * src[1*4+3] + dst[2*4+2] * src[2*4+3]);
    
    // Bottom row
    dst[3*4 + 0] = 0.0f;
    dst[3*4 + 1] = 0.0f;
    dst[3*4 + 2] = 0.0f;
    dst[3*4 + 3] = 1.0f;
}

/**
 * 4x4 Matrix multiplication (row-major): result = A * B
 */
static void multiply_rt_matrix(const float A[16], const float B[16], float result[16]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result[i*4 + j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                result[i*4 + j] += A[i*4 + k] * B[k*4 + j];
            }
        }
    }
}

// ============================================================================
// RGB to YCbCr Conversion (Host-side)
// ============================================================================

/**
 * Convert RGB image to Y channel (grayscale for filtering)
 */
static void rgb_to_y_channel(const std::vector<float>& rgb_image, 
                              std::vector<uint8_t>& y_image,
                              int width, int height) {
    y_image.resize(width * height);
    
    for (int i = 0; i < width * height; i++) {
        float r = rgb_image[i * 3 + 0];
        float g = rgb_image[i * 3 + 1];
        float b = rgb_image[i * 3 + 2];
        
        // ITU-R BT.601 conversion
        float y = 0.299f * r + 0.587f * g + 0.114f * b;
        y_image[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, y)));
    }
}

/**
 * Convert RGB to YCbCr (3-channel)
 */
static void rgb_to_ycbcr(const std::vector<float>& rgb_image,
                         std::vector<uchar3>& ycbcr_image,
                         int width, int height) {
    ycbcr_image.resize(width * height);
    
    for (int i = 0; i < width * height; i++) {
        float r = rgb_image[i * 3 + 0];
        float g = rgb_image[i * 3 + 1];
        float b = rgb_image[i * 3 + 2];
        
        // ITU-R BT.601 conversion (Python: utils.py rgb2yCbCr)
        float y  = std::min(235.0f, std::max(16.0f, 16.0f + 0.1826f * r + 0.6142f * g + 0.062f * b));
        float cb = std::min(240.0f, std::max(16.0f, 128.0f - 0.1006f * r - 0.3386f * g + 0.4392f * b));
        float cr = std::min(240.0f, std::max(16.0f, 128.0f + 0.4392f * r - 0.3989f * g - 0.0403f * b));
        
        ycbcr_image[i] = make_uchar3(
            static_cast<unsigned char>(y),
            static_cast<unsigned char>(cb),
            static_cast<unsigned char>(cr)
        );
    }
}

/**
 * Convert std::vector<float> distance map to LibTorch tensor
 */
static at::Tensor vector_to_tensor_2d(const std::vector<float>& vec, int rows, int cols, const at::Device& device) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    at::Tensor tensor = at::from_blob(const_cast<float*>(vec.data()), {rows, cols}, options).clone();
    return tensor.to(device);
}

/**
 * Convert RGB image std::vector<float> [H*W*3] to LibTorch tensor [H, W, 3] uint8
 */
static at::Tensor rgb_vector_to_tensor(const std::vector<float>& rgb_vec, int rows, int cols, const at::Device& device) {
    std::vector<uint8_t> uint8_vec(rgb_vec.size());
    for (size_t i = 0; i < rgb_vec.size(); i++) {
        uint8_vec[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, rgb_vec[i])));
    }
    auto options = at::TensorOptions().dtype(at::kByte).device(at::kCPU);
    at::Tensor tensor = at::from_blob(uint8_vec.data(), {rows, cols, 3}, options).clone();
    return tensor.to(device);
}

/**
 * Convert LibTorch tensor to std::vector
 */
static std::vector<uint8_t> tensor_to_vector_uint8(const at::Tensor& tensor) {
    at::Tensor cpu_tensor = tensor.to(at::kCPU).contiguous();
    uint8_t* data_ptr = cpu_tensor.data_ptr<uint8_t>();
    size_t size = cpu_tensor.numel();
    return std::vector<uint8_t>(data_ptr, data_ptr + size);
}

static std::vector<float> tensor_to_vector_float(const at::Tensor& tensor) {
    at::Tensor cpu_tensor = tensor.to(at::kCPU).contiguous();
    float* data_ptr = cpu_tensor.data_ptr<float>();
    size_t size = cpu_tensor.numel();
    return std::vector<float>(data_ptr, data_ptr + size);
}

/**
 * Pre-compute all relative RT matrices to avoid runtime computation
 * For each reference camera, compute: relative_rt[cam] = inv(cam.rt) @ ref.rt
 */
void RGBD_Estimator::precompute_relative_calibrations() {
    relative_calibrations_.resize(references_indices_.size());
    
    for (size_t ref_idx = 0; ref_idx < references_indices_.size(); ref_idx++) {
        int reference_index = references_indices_[ref_idx];
        const DoubleSphereCalibration& ref_calib = calibrations_[reference_index];
        
        relative_calibrations_[ref_idx].resize(num_cameras_);
        
        // Compute inverse of reference camera RT
        float ref_rt_inv[16];
        invert_rt_matrix(ref_calib.rt, ref_rt_inv);
        
        for (int cam_idx = 0; cam_idx < num_cameras_; cam_idx++) {
            relative_calibrations_[ref_idx][cam_idx] = calibrations_[cam_idx];
            
            if (cam_idx == reference_index) {
                // Reference camera relative to itself is identity
                float identity[16] = {
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1
                };
                std::copy(identity, identity + 16, relative_calibrations_[ref_idx][cam_idx].rt);
                // Set R/t arrays to identity
                float identity_R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
                float identity_t[3] = {0, 0, 0};
                std::copy(identity_R, identity_R + 9, relative_calibrations_[ref_idx][cam_idx].R);
                std::copy(identity_t, identity_t + 3, relative_calibrations_[ref_idx][cam_idx].t);
            } else {
                // Compute: inv(target_RT) @ reference_RT
                float target_rt_inv[16];
                invert_rt_matrix(calibrations_[cam_idx].rt, target_rt_inv);
                multiply_rt_matrix(target_rt_inv, ref_calib.rt, 
                                   relative_calibrations_[ref_idx][cam_idx].rt);
                
                // Extract R/t from computed relative RT matrix
                for (int row = 0; row < 3; row++) {
                    for (int col = 0; col < 3; col++) {
                        relative_calibrations_[ref_idx][cam_idx].R[row * 3 + col] = 
                            relative_calibrations_[ref_idx][cam_idx].rt[row * 4 + col];
                    }
                    relative_calibrations_[ref_idx][cam_idx].t[row] = 
                        relative_calibrations_[ref_idx][cam_idx].rt[row * 4 + 3];
                }
            }
        }
    }
}

/**
 * Convert DoubleSphereCalibration to my_stereo::Calibration
 */
static my_stereo::Calibration convert_calibration(
    const DoubleSphereCalibration& ds_calib,
    float matching_scale,
    const at::Device& device
) {
    my_stereo::Calibration calib;
    
    // Copy RT matrix (4x4 row-major)
    auto rt_tensor = at::from_blob(
        const_cast<float*>(ds_calib.rt),
        {4, 4},
        at::TensorOptions().dtype(at::kFloat)
    ).clone().to(device);
    calib.rt = rt_tensor;
    
    // Copy intrinsics
    calib.fl = {ds_calib.fx, ds_calib.fy};
    calib.principal = {ds_calib.cx, ds_calib.cy};
    calib.xi = ds_calib.xi;
    calib.alpha = ds_calib.alpha;
    calib.matching_scale = matching_scale;
    
    return calib;
}

/**
 * Convert RGB float32 [0, 255] to uchar4 for GPU processing
 */
static void rgb_to_uchar4(const std::vector<float>& rgb_image,
                          std::vector<uchar4>& uchar4_image,
                          int width, int height) {
    uchar4_image.resize(width * height);
    
    for (int i = 0; i < width * height; i++) {
        uint8_t r = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, rgb_image[i * 3 + 0])));
        uint8_t g = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, rgb_image[i * 3 + 1])));
        uint8_t b = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, rgb_image[i * 3 + 2])));
        uint8_t a = 255;
        
        uchar4_image[i] = make_uchar4(r, g, b, a);
    }
}

// ============================================================================
// Main Estimation Function
// ============================================================================

std::pair<std::vector<uint8_t>, std::vector<float>>
RGBD_Estimator::estimate_RGBD_panorama(
    const std::vector<std::vector<float>>& images_to_match,
    const std::vector<std::vector<float>>& images_to_stitch
) {
    std::cout << "[RGBD_Estimator] Starting RGBD panorama estimation\n";
    std::cout << "[RGBD_Estimator] Input: " << images_to_match.size() << " matching images, "
              << images_to_stitch.size() << " stitching images\n";
    
    CUDA_CHECK(cudaSetDevice(device_id_));
    
    at::Device device(at::kCUDA, device_id_);
    
    // ========================================================================
    // Step 1: Estimate Distance Maps for Each Reference Camera
    // ========================================================================
    
    std::cout << "[RGBD_Estimator] Step 1: Estimating distance maps for " 
              << references_indices_.size() << " reference cameras\n";
    
    std::vector<std::vector<float>> distance_maps_vec;
    distance_maps_vec.reserve(references_indices_.size());
    
    for (size_t i = 0; i < references_indices_.size(); i++) {
        int ref_idx = references_indices_[i];
        std::cout << "[RGBD_Estimator]   Processing reference camera " << ref_idx << "\n";
        
        // Call existing estimate_fisheye_distance implementation
        std::vector<float> distance_map = estimate_fisheye_distance(ref_idx, images_to_match);
        distance_maps_vec.push_back(std::move(distance_map));
    }
    
    std::cout << "[RGBD_Estimator] Depth estimation complete\n";
    
    // ========================================================================
    // Step 2: Apply ISB Distance Filter (Post-processing)
    // ========================================================================
    
    std::cout << "[RGBD_Estimator] Step 2: Applying ISB distance filtering\n";
    
    // Initialize ISB filter for distance maps (1 channel)
    my_stereo_pkg::ISBFilter distance_filter(
        1,  // 1 channel (distance only)
        {config_.matching_width, config_.matching_height},
        device
    );
    
    std::vector<at::Tensor> filtered_distance_tensors;
    filtered_distance_tensors.reserve(references_indices_.size());
    
    for (size_t i = 0; i < references_indices_.size(); i++) {
        int ref_idx = references_indices_[i];
        
        // Convert to tensor
        at::Tensor distance_tensor = vector_to_tensor_2d(
            distance_maps_vec[i],
            config_.matching_height,
            config_.matching_width,
            device
        ).unsqueeze(0);  // [1, H, W]
        
        // Create guide image from matching image
        at::Tensor guide_tensor = rgb_vector_to_tensor(
            images_to_match[ref_idx],
            config_.matching_height,
            config_.matching_width,
            device
        );
        
        // Apply ISB filter with edge preservation (sigma_i/2 for stronger preservation)
        auto [filtered_dist, _] = distance_filter.apply(
            guide_tensor,
            distance_tensor,
            sigma_i_ / 2.0f,  // Stronger edge preservation for distance
            sigma_s_ / 2.0f
        );
        
        filtered_distance_tensors.push_back(filtered_dist.squeeze(0));  // [H, W]
    }
    
    std::cout << "[RGBD_Estimator] Distance filtering complete\n";
    
    // ========================================================================
    // Step 3: Stitch into RGB-D Panorama
    // ========================================================================
    
    std::cout << "[RGBD_Estimator] Step 3: Stitching RGB-D panorama\n";
    
    // Convert calibrations
    std::vector<my_stereo::Calibration> stitcher_calibrations;
    stitcher_calibrations.reserve(references_indices_.size());
    
    float matching_scale = 1.0f;  // Assuming no scaling between capture and matching resolution
    
    for (int ref_idx : references_indices_) {
        stitcher_calibrations.push_back(
            convert_calibration(calibrations_[ref_idx], matching_scale, device)
        );
    }
    
    // Create reprojection viewpoint tensor (typically origin [0, 0, 0])
    at::Tensor reprojection_viewpoint = at::zeros({3}, at::TensorOptions().dtype(at::kFloat).device(device));
    
    // Create masks (all ones - assuming full validity)
    at::Tensor masks = at::ones(
        {static_cast<int>(references_indices_.size()), config_.matching_height, config_.matching_width},
        at::TensorOptions().dtype(at::kFloat).device(device)
    );
    
    // Initialize stitcher
    my_stereo::Stitcher stitcher(
        stitcher_calibrations,
        reprojection_viewpoint,
        masks,
        config_.min_dist,
        config_.max_dist,
        config_.matching_width,
        config_.matching_height,
        rgb_to_stitch_width_,
        rgb_to_stitch_height_,
        panorama_width_,
        panorama_height_,
        device,
        15,  // smoothing_radius
        32   // inpainting_iterations
    );
    
    // Convert RGB images to tensors
    std::vector<at::Tensor> rgb_tensors;
    rgb_tensors.reserve(references_indices_.size());
    
    for (size_t i = 0; i < references_indices_.size(); i++) {
        int ref_idx = references_indices_[i];
        rgb_tensors.push_back(
            rgb_vector_to_tensor(
                images_to_stitch[i],
                config_.matching_height,
                config_.matching_width,
                device
            )
        );
    }
    
    // Perform stitching
    auto [rgb_panorama_tensor, distance_panorama_tensor] = stitcher.stitch(
        rgb_tensors,
        filtered_distance_tensors
    );
    
    std::cout << "[RGBD_Estimator] Stitching complete\n";
    
    // ========================================================================
    // Step 4: Convert Results to std::vector
    // ========================================================================
    
    std::vector<uint8_t> rgb_panorama = tensor_to_vector_uint8(rgb_panorama_tensor);
    std::vector<float> distance_panorama = tensor_to_vector_float(distance_panorama_tensor);
    
    std::cout << "[RGBD_Estimator] Output: " << rgb_panorama.size() << " byte RGB, "
              << distance_panorama.size() << " float distances\n";
    std::cout << "[RGBD_Estimator] Estimation complete\n";
    
    return {rgb_panorama, distance_panorama};
}

// ============================================================================
// Helper: Estimate Distance for Single Reference Camera
// ============================================================================

std::vector<float> RGBD_Estimator::estimate_fisheye_distance(
    int reference_index,
    const std::vector<std::vector<float>>& images_to_match
) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    
    // Find reference index in precomputed relative calibrations
    size_t ref_calib_idx = 0;
    for (size_t i = 0; i < references_indices_.size(); i++) {
        if (references_indices_[i] == reference_index) {
            ref_calib_idx = i;
            break;
        }
    }
    
    // Convert and upload images to pre-allocated GPU memory
    std::vector<uchar4> h_ref_image_uchar4;
    rgb_to_uchar4(images_to_match[reference_index], h_ref_image_uchar4,
                  config_.matching_width, config_.matching_height);
    
    // Upload target images to pre-allocated memory and update texture objects
    for (int cam_idx = 0; cam_idx < num_cameras_; cam_idx++) {
        std::vector<uchar4> h_target_image_uchar4;
        rgb_to_uchar4(images_to_match[cam_idx], h_target_image_uchar4,
                      config_.matching_width, config_.matching_height);
        
        // Copy to pre-allocated GPU memory (avoiding runtime malloc/free)
        CUDA_CHECK(cudaMemcpy(d_target_images_[cam_idx], h_target_image_uchar4.data(),
                              config_.matching_width * config_.matching_height * sizeof(uchar4),
                              cudaMemcpyHostToDevice));
        
        // Update CUDA array for texture
        CUDA_CHECK(cudaMemcpy2DToArray(d_target_arrays_[cam_idx], 0, 0,
                                       d_target_images_[cam_idx],
                                       config_.matching_width * sizeof(uchar4),
                                       config_.matching_width * sizeof(uchar4),
                                       config_.matching_height,
                                       cudaMemcpyDeviceToDevice));
        
        // Destroy existing texture object if it exists
        if (d_image_texobjs_[cam_idx]) {
            CUDA_CHECK(cudaDestroyTextureObject(d_image_texobjs_[cam_idx]));
        }
        
        // Create texture object with correct normalization settings
        struct cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = d_target_arrays_[cam_idx];
        
        struct cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.addressMode[0] = cudaAddressModeBorder;
        tex_desc.addressMode[1] = cudaAddressModeBorder;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        // CRITICAL FIX: Use pixel coordinates (0 to width/height) instead of normalized
        tex_desc.normalizedCoords = 0;  // Use pixel coordinates, not [0,1] normalized
        
        CUDA_CHECK(cudaCreateTextureObject(&d_image_texobjs_[cam_idx], &res_desc, &tex_desc, nullptr));
    }
    
    // Allocate reference image
    uchar4* d_ref_image;
    CUDA_CHECK(cudaMalloc(&d_ref_image, config_.matching_width * config_.matching_height * sizeof(uchar4)));
    CUDA_CHECK(cudaMemcpy(d_ref_image, h_ref_image_uchar4.data(),
                          config_.matching_width * config_.matching_height * sizeof(uchar4),
                          cudaMemcpyHostToDevice));
    
    // Create guide image (YCbCr format for ISB Filter)
    std::vector<uchar3> h_guide_ycbcr;
    rgb_to_ycbcr(images_to_match[reference_index], h_guide_ycbcr,
                 config_.matching_width, config_.matching_height);
    
    uchar3* d_guide_ycbcr;
    CUDA_CHECK(cudaMalloc(&d_guide_ycbcr, config_.matching_width * config_.matching_height * sizeof(uchar3)));
    CUDA_CHECK(cudaMemcpy(d_guide_ycbcr, h_guide_ycbcr.data(),
                          config_.matching_width * config_.matching_height * sizeof(uchar3),
                          cudaMemcpyHostToDevice));
    
    // Also create single-channel Y guide for legacy kernel
    std::vector<uint8_t> h_guide;
    for (const auto& ycbcr : h_guide_ycbcr) {
        h_guide.push_back(ycbcr.x);  // Y channel only
    }
    uchar* d_guide;
    CUDA_CHECK(cudaMalloc(&d_guide, config_.matching_width * config_.matching_height));
    CUDA_CHECK(cudaMemcpy(d_guide, h_guide.data(),
                          config_.matching_width * config_.matching_height,
                          cudaMemcpyHostToDevice));
    
    // Allocate selected cameras
    int* d_selected_cameras;
    float* d_max_displacement;
    CUDA_CHECK(cudaMalloc(&d_selected_cameras,
                          config_.matching_width * config_.matching_height * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max_displacement,
                          config_.matching_width * config_.matching_height * sizeof(float)));
    
    // ========================================================================
    // Use Pre-computed Relative RT Matrices (avoids runtime computation)
    // ========================================================================
    
    const std::vector<DoubleSphereCalibration>& relative_calibrations = 
        relative_calibrations_[ref_calib_idx];
    const DoubleSphereCalibration& ref_calib = calibrations_[reference_index];
    
    // ========================================================================
    // Launch Adaptive Camera Selection Kernel
    // ========================================================================
    
    select_best_cameras_kernel(d_selected_cameras, d_max_displacement,
                               d_calibrations_, nullptr,
                               ref_calib, config_, NULL);
    
    // ========================================================================
    // Create Device-side Pointer Arrays for Kernels
    // ========================================================================
    // CRITICAL FIX: GPU kernels cannot access host-side std::vector data
    // We must copy pointer arrays to device memory
    
    uchar4** d_images_ptr_array;
    cudaTextureObject_t* d_texobjs_ptr_array;
    
    CUDA_CHECK(cudaMalloc(&d_images_ptr_array, num_cameras_ * sizeof(uchar4*)));
    CUDA_CHECK(cudaMalloc(&d_texobjs_ptr_array, num_cameras_ * sizeof(cudaTextureObject_t)));
    
    CUDA_CHECK(cudaMemcpy(d_images_ptr_array, d_target_images_.data(),
                          num_cameras_ * sizeof(uchar4*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_texobjs_ptr_array, d_image_texobjs_.data(),
                          num_cameras_ * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));
    
    // ========================================================================
    // Python-Equivalent Pipeline: Cost Volume -> Filter -> Distance Selection
    // ========================================================================
    
    // Upload relative calibrations to device
    DoubleSphereCalibration* d_relative_calibrations;
    CUDA_CHECK(cudaMalloc(&d_relative_calibrations, num_cameras_ * sizeof(DoubleSphereCalibration)));
    CUDA_CHECK(cudaMemcpy(d_relative_calibrations, relative_calibrations.data(),
                          num_cameras_ * sizeof(DoubleSphereCalibration),
                          cudaMemcpyHostToDevice));
    
    // Step 1: Generate cost volume (using pre-allocated pitched memory)
    compute_cost_volume_kernel(d_cost_volume_, d_ref_image,
                               (const uchar4* const*)d_images_ptr_array,
                               d_selected_cameras,
                               d_relative_calibrations,
                               relative_calibrations[reference_index],
                               config_, NULL,
                               d_texobjs_ptr_array);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Step 2: Apply ISB Filter to cost volume
    apply_isb_filter(d_guide_ycbcr, d_cost_volume_,
                     config_.matching_width, config_.matching_height,
                     config_.candidate_count,
                     sigma_i_, sigma_s_, NULL);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Step 3: Select minimum distance with quadratic fitting
    select_distance_from_cost_volume_kernel(d_distance_map_, d_cost_volume_, config_, NULL);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // ========================================================================
    // Copy Distance Map from GPU to Host (CRITICAL: Use Pitched Memory Transfer)
    // ========================================================================
    
    std::vector<float> h_distance_map(config_.matching_width * config_.matching_height);
    
    // CRITICAL FIX: Use cudaMemcpy2D to handle pitched memory properly
    CUDA_CHECK(cudaMemcpy2D(h_distance_map.data(),                    // dst
                            config_.matching_width * sizeof(float),   // dst_pitch (row width)
                            d_distance_map_,                          // src (pitched memory)
                            distance_pitch_,                          // src_pitch (includes padding)
                            config_.matching_width * sizeof(float),   // width_in_bytes
                            config_.matching_height,                  // height
                            cudaMemcpyDeviceToHost));
    
    // ========================================================================
    // Cleanup (reduced cleanup since most memory is pre-allocated)
    // ========================================================================
    
    CUDA_CHECK(cudaFree(d_images_ptr_array));
    CUDA_CHECK(cudaFree(d_texobjs_ptr_array));
    CUDA_CHECK(cudaFree(d_relative_calibrations));
    
    CUDA_CHECK(cudaFree(d_ref_image));
    CUDA_CHECK(cudaFree(d_guide));
    CUDA_CHECK(cudaFree(d_guide_ycbcr));
    CUDA_CHECK(cudaFree(d_selected_cameras));
    CUDA_CHECK(cudaFree(d_max_displacement));
    
    return h_distance_map;
}
