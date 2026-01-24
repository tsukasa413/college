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
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>
#include <algorithm>

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
      sigma_i_(sigma_i),
      sigma_s_(sigma_s),
      references_indices_(references_indices),
      d_calibrations_(nullptr),
      d_distance_map_(nullptr),
      d_cost_volume_(nullptr)
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
        
        // Copy intrinsics [fx, fy, cx, cy]
        calib.fx = calibrations_intrinsics[i * 4 + 0];
        calib.fy = calibrations_intrinsics[i * 4 + 1];
        calib.cx = calibrations_intrinsics[i * 4 + 2];
        calib.cy = calibrations_intrinsics[i * 4 + 3];
        
        // Copy sphere parameters [xi, alpha]
        calib.xi = calibrations_sphere[i * 2 + 0];
        calib.alpha = calibrations_sphere[i * 2 + 1];
        
        // Copy resolution [width, height]
        calib.width = calibrations_resolution[i * 2 + 0];
        calib.height = calibrations_resolution[i * 2 + 1];
        
        calib.padding = 0;
    }
    
    // ========================================================================
    // Allocate GPU Memory
    // ========================================================================
    
    allocate_gpu_memory();
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
    
    // Distance map: [matching_height, matching_width]
    size_t distance_pitch;
    CUDA_CHECK(cudaMallocPitch(&d_distance_map_, &distance_pitch,
                               config_.matching_width * sizeof(float),
                               config_.matching_height));
    
    // Cost volume (for filtering, may be optional)
    size_t cost_pitch;
    CUDA_CHECK(cudaMallocPitch(&d_cost_volume_, &cost_pitch,
                               config_.matching_width * config_.candidate_count * sizeof(float),
                               config_.matching_height));
    
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
    
    for (auto& mask_ptr : d_masks_) {
        if (mask_ptr) {
            CUDA_CHECK(cudaFree(mask_ptr));
        }
    }
    d_masks_.clear();
}

void RGBD_Estimator::upload_calibrations() {
    CUDA_CHECK(cudaSetDevice(device_id_));
    
    CUDA_CHECK(cudaMemcpy(d_calibrations_, calibrations_.data(),
                          num_cameras_ * sizeof(DoubleSphereCalibration),
                          cudaMemcpyHostToDevice));
}

void RGBD_Estimator::upload_masks() {
    // Placeholder: In full implementation, upload validity masks to GPU
    // For now, assume masks are already on GPU or not used
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
    CUDA_CHECK(cudaSetDevice(device_id_));
    
    // Temporary containers
    std::vector<uchar4> h_reference_image_uchar4;
    std::vector<std::vector<uchar4>> h_target_images_uchar4(num_cameras_);
    
    std::vector<float> distance_maps_combined;
    std::vector<std::vector<float>> distance_maps;
    
    // ========================================================================
    // Process each reference camera
    // ========================================================================
    
    for (size_t ref_idx = 0; ref_idx < references_indices_.size(); ref_idx++) {
        int reference_index = references_indices_[ref_idx];
        
        // Convert images to appropriate formats
        rgb_to_uchar4(images_to_match[reference_index], h_reference_image_uchar4,
                      config_.matching_width, config_.matching_height);
        
        for (int cam_idx = 0; cam_idx < num_cameras_; cam_idx++) {
            rgb_to_uchar4(images_to_match[cam_idx], h_target_images_uchar4[cam_idx],
                          config_.matching_width, config_.matching_height);
        }
        
        // Compute distance for this reference
        std::vector<float> distance = estimate_fisheye_distance(reference_index, images_to_match);
        distance_maps.push_back(distance);
    }
    
    // ========================================================================
    // Placeholder for Stitching
    // ========================================================================
    
    std::vector<uint8_t> rgb_panorama;
    std::vector<float> distance_panorama;
    
    // In full implementation, call fishey_stitcher.stitch() here
    // For now, return empty results
    
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
    
    // Convert reference image to GPU format
    std::vector<uchar4> h_ref_image_uchar4;
    rgb_to_uchar4(images_to_match[reference_index], h_ref_image_uchar4,
                  config_.matching_width, config_.matching_height);
    
    // Convert target images
    std::vector<std::vector<uchar4>> h_target_images_uchar4(num_cameras_);
    std::vector<uchar4*> d_target_images(num_cameras_);
    
    for (int cam_idx = 0; cam_idx < num_cameras_; cam_idx++) {
        rgb_to_uchar4(images_to_match[cam_idx], h_target_images_uchar4[cam_idx],
                      config_.matching_width, config_.matching_height);
        
        // Allocate GPU memory for target image
        CUDA_CHECK(cudaMalloc(&d_target_images[cam_idx],
                              config_.matching_width * config_.matching_height * sizeof(uchar4)));
        
        // Copy to GPU
        CUDA_CHECK(cudaMemcpy(d_target_images[cam_idx], h_target_images_uchar4[cam_idx].data(),
                              config_.matching_width * config_.matching_height * sizeof(uchar4),
                              cudaMemcpyHostToDevice));
    }
    
    // Allocate GPU memory for reference image
    uchar4* d_ref_image;
    CUDA_CHECK(cudaMalloc(&d_ref_image, config_.matching_width * config_.matching_height * sizeof(uchar4)));
    
    // Copy reference image to GPU
    CUDA_CHECK(cudaMemcpy(d_ref_image, h_ref_image_uchar4.data(),
                          config_.matching_width * config_.matching_height * sizeof(uchar4),
                          cudaMemcpyHostToDevice));
    
    // Create guide image (Y channel)
    std::vector<uint8_t> h_guide;
    rgb_to_y_channel(images_to_match[reference_index], h_guide,
                     config_.matching_width, config_.matching_height);
    
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
    // Launch Adaptive Camera Selection Kernel
    // ========================================================================
    
    const DoubleSphereCalibration& ref_calib = calibrations_[reference_index];
    select_best_cameras_kernel(d_selected_cameras, d_max_displacement,
                               d_calibrations_, nullptr,
                               ref_calib, config_, NULL);
    
    // ========================================================================
    // Create Texture Objects for Images
    // ========================================================================
    
    std::vector<cudaTextureObject_t> d_texobjs(num_cameras_);
    std::vector<cudaArray*> d_cuarrays(num_cameras_);
    
    for (int cam_idx = 0; cam_idx < num_cameras_; cam_idx++) {
        // Create CUDA array from device memory
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
        CUDA_CHECK(cudaMallocArray(&d_cuarrays[cam_idx], &channel_desc,
                                   config_.matching_width, config_.matching_height));
        
        // Copy image to array
        CUDA_CHECK(cudaMemcpyToArray(d_cuarrays[cam_idx], 0, 0,
                                     d_target_images[cam_idx],
                                     config_.matching_width * config_.matching_height * sizeof(uchar4),
                                     cudaMemcpyDeviceToDevice));
        
        // Create texture object
        struct cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = d_cuarrays[cam_idx];
        
        struct cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.addressMode[0] = cudaAddressModeBorder;
        tex_desc.addressMode[1] = cudaAddressModeBorder;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        
        CUDA_CHECK(cudaCreateTextureObject(&d_texobjs[cam_idx], &res_desc, &tex_desc, nullptr));
    }
    
    // ========================================================================
    // Launch Main Fused Depth Estimation Kernel
    // ========================================================================
    
    estimate_fisheye_distance_fused_kernel(d_distance_map_, d_ref_image,
                                           (const uchar4* const*)d_target_images.data(),
                                           d_selected_cameras,
                                           d_calibrations_, d_guide, config_, NULL,
                                           d_texobjs.data());
    
    // ========================================================================
    // Copy Distance Map from GPU to Host
    // ========================================================================
    
    std::vector<float> h_distance_map(config_.matching_width * config_.matching_height);
    CUDA_CHECK(cudaMemcpy(h_distance_map.data(), d_distance_map_,
                          config_.matching_width * config_.matching_height * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    
    CUDA_CHECK(cudaFree(d_ref_image));
    CUDA_CHECK(cudaFree(d_guide));
    CUDA_CHECK(cudaFree(d_selected_cameras));
    CUDA_CHECK(cudaFree(d_max_displacement));
    
    for (auto d_img : d_target_images) {
        CUDA_CHECK(cudaFree(d_img));
    }
    
    for (auto d_texobj : d_texobjs) {
        CUDA_CHECK(cudaDestroyTextureObject(d_texobj));
    }
    
    for (auto d_arr : d_cuarrays) {
        CUDA_CHECK(cudaFreeArray(d_arr));
    }
    
    return h_distance_map;
}
