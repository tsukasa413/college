/**
 * @file DepthEstimator.hpp
 * @brief RGB-D panorama estimation from multi-view fisheye images
 * 
 * Main depth estimation pipeline combining:
 * - Adaptive camera selection
 * - Sphere sweeping cost volume computation
 * - ISB filter for edge-preserving aggregation
 * - Panorama stitching
 * 
 * Based on: "Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images"
 * Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
 * CVPR 2021
 * 
 * @copyright CC BY-NC-SA 3.0
 */

#ifndef SPHERE_STEREO_ROS_DEPTH_ESTIMATOR_HPP
#define SPHERE_STEREO_ROS_DEPTH_ESTIMATOR_HPP

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

#include "sphere_stereo_ros/Utils.hpp"
#include "sphere_stereo_ros/Stitcher.hpp"
#include "sphere_stereo_ros/IsbFilter.hpp"
#include "sphere_stereo_ros/cuda/cost_volume.cuh"

namespace sphere_stereo_ros {

/**
 * @brief Configuration for depth estimation pipeline
 */
struct DepthEstimatorConfig {
    // Resolution settings
    int matching_width = 224;      ///< Matching resolution width
    int matching_height = 224;     ///< Matching resolution height
    int stitch_width = 672;        ///< RGB stitching resolution width
    int stitch_height = 672;       ///< RGB stitching resolution height
    int pano_width = 256;          ///< Output panorama width
    int pano_height = 128;         ///< Output panorama height
    
    // Depth estimation settings
    int num_depth_candidates = 64; ///< Number of depth planes
    float min_dist = 0.4f;         ///< Minimum distance (meters)
    float max_dist = 1000.0f;      ///< Maximum distance (meters)
    
    // ISB filter settings
    float sigma_i = 10.0f;         ///< Edge preservation for cost volume
    float sigma_s = 3.0f;          ///< Spatial smoothing for cost volume
    float sigma_i_dist = 5.0f;     ///< Edge preservation for distance map (sigma_i / 2)
    float sigma_s_dist = 1.5f;     ///< Spatial smoothing for distance map (sigma_s / 2)
    
    // Cost computation
    float cost_clamp = 500.0f;     ///< Maximum cost value
    
    // Reference cameras (cameras where depth is estimated before stitching)
    std::vector<int> reference_indices = {0, 2};  ///< Default: cameras 0 and 2
};

/**
 * @brief Depth estimator for multi-view fisheye systems
 * 
 * This class implements the complete RGB-D panorama estimation pipeline:
 * 1. For each reference camera:
 *    a. Select best matching camera per pixel (adaptive matching)
 *    b. Compute cost volume via sphere sweeping
 *    c. Filter cost volume with ISB filter
 *    d. Select depth via WTA + quadratic refinement
 *    e. Filter distance map with ISB filter
 * 2. Stitch reference views into panorama
 * 
 * Memory Management:
 * - All GPU memory is pre-allocated at construction
 * - No dynamic allocation during update() calls
 * - Uses CUDA streams for async execution
 */
class DepthEstimator {
public:
    /**
     * @brief Construct depth estimator
     * @param calibration Camera calibration data
     * @param config Depth estimation configuration
     */
    DepthEstimator(const CalibrationSet& calibration, const DepthEstimatorConfig& config);
    
    /**
     * @brief Destructor - frees all GPU memory
     */
    ~DepthEstimator();
    
    // Non-copyable, movable
    DepthEstimator(const DepthEstimator&) = delete;
    DepthEstimator& operator=(const DepthEstimator&) = delete;
    DepthEstimator(DepthEstimator&&) noexcept;
    DepthEstimator& operator=(DepthEstimator&&) noexcept;
    
    /**
     * @brief Initialize all GPU resources
     * 
     * Must be called before update(). Allocates:
     * - Input image buffers
     * - Cost volume buffers
     * - Distance map buffers
     * - Precomputes camera selection maps
     * - Initializes Stitcher and ISB filters
     */
    void initialize();
    
    /**
     * @brief Check if initialized
     */
    bool isInitialized() const { return initialized_; }
    
    /**
     * @brief Process a frame and estimate RGB-D panorama
     * 
     * Complete pipeline from fisheye images to RGB-D panorama.
     * No dynamic GPU allocation is performed during this call.
     * 
     * @param images Input fisheye images (CV_8UC3, BGR format)
     *               Vector of num_cameras images, each at original resolution
     * @param[out] rgb_panorama Output RGB panorama (CV_8UC3)
     * @param[out] distance_panorama Output distance panorama (CV_32FC1)
     */
    void update(
        const std::vector<cv::Mat>& images,
        cv::Mat& rgb_panorama,
        cv::Mat& distance_panorama);
    
    /**
     * @brief Get the Stitcher instance
     */
    Stitcher& getStitcher() { return *stitcher_; }
    const Stitcher& getStitcher() const { return *stitcher_; }
    
    /**
     * @brief Get configuration
     */
    const DepthEstimatorConfig& getConfig() const { return config_; }
    
    /**
     * @brief Get number of cameras
     */
    int getNumCameras() const { return num_cameras_; }
    
    /**
     * @brief Synchronize CUDA operations
     */
    void synchronize();

private:
    // Configuration
    CalibrationSet calibration_;
    DepthEstimatorConfig config_;
    int num_cameras_;
    int num_references_;
    bool initialized_;
    
    // CUDA stream
    cudaStream_t stream_;
    
    // Sub-components
    std::unique_ptr<Stitcher> stitcher_;
    std::unique_ptr<IsbFilter> cost_filter_;      ///< Filter for cost volume
    std::unique_ptr<IsbFilter> distance_filter_;  ///< Filter for distance map
    
    // GPU buffers - Calibration data
    cuda::Intrinsics* d_intrinsics_;      ///< [num_cameras]
    cuda::Rotation* d_rotations_;         ///< Relative rotations [num_cameras x num_references]
    float3* d_translations_;              ///< Relative translations [num_cameras x num_references]
    float* d_masks_;                      ///< Validity masks [num_cameras x matching_h x matching_w]
    
    // GPU buffers - Camera selection (precomputed)
    int* d_selected_cameras_;             ///< [num_references x matching_h x matching_w]
    
    // GPU buffers - Input images
    uchar3* d_images_matching_;           ///< [num_cameras x matching_h x matching_w] YCbCr
    uchar3* d_images_stitch_;             ///< [num_references x stitch_h x stitch_w] RGB
    
    // GPU buffers - Intermediate results
    float* d_cost_volume_;                ///< [num_depths x matching_h x matching_w]
    float* d_distance_candidates_;        ///< [num_depths]
    float* d_distance_maps_;              ///< [num_references x matching_h x matching_w]
    uchar3* d_guide_images_;              ///< [num_references x matching_h x matching_w] YCbCr
    
    // Pinned host memory for transfers (pre-allocated)
    uchar3* h_images_matching_pinned_;    ///< [num_cameras x matching_h x matching_w]
    uchar3* h_images_stitch_pinned_;      ///< [num_references x stitch_h x stitch_w]
    float* h_distances_pinned_;           ///< [num_references x matching_h x matching_w]
    uchar3* h_stitch_download_pinned_;    ///< [num_references x stitch_h x stitch_w]
    
    // Pre-allocated cv::Mat for update() (Zero-Allocation)
    std::vector<cv::Mat> distance_maps_cpu_;   ///< Reusable cv::Mat wrappers for distances
    std::vector<cv::Mat> stitch_images_cpu_;   ///< Reusable cv::Mat for stitch images
    
    // Cost volume config
    cuda::CostVolumeConfig cv_config_;
    
    // Helper methods
    void allocateMemory();
    void freeMemory();
    void uploadCalibration();
    void computeMasks();
    void computeCameraSelection();
    void computeDistanceCandidates();
    void preprocessImages(const std::vector<cv::Mat>& images);
    void estimateFisheyeDistance(int reference_idx);
};

}  // namespace sphere_stereo_ros

#endif  // SPHERE_STEREO_ROS_DEPTH_ESTIMATOR_HPP
