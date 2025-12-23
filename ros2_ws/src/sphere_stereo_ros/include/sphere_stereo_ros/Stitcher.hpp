/**
 * @file Stitcher.hpp
 * @brief C++ wrapper class for CUDA-based panorama stitching
 * 
 * Ported from Python implementation for ROS 2 / C++ project.
 * Based on: "Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images"
 * Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
 * CVPR 2021
 * 
 * @copyright CC BY-NC-SA 3.0
 */

#ifndef SPHERE_STEREO_ROS_STITCHER_HPP
#define SPHERE_STEREO_ROS_STITCHER_HPP

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

#include "sphere_stereo_ros/Utils.hpp"
#include "sphere_stereo_ros/cuda/stitcher.cuh"

namespace sphere_stereo_ros {

/**
 * @brief Panorama stitcher for multi-camera fisheye systems
 * 
 * This class handles the stitching of multiple fisheye images and their
 * distance maps into an equirectangular RGB-D panorama. It uses precomputed
 * blending lookup tables for efficient runtime operation.
 * 
 * Memory Layout:
 * - All GPU buffers are pre-allocated at construction
 * - No dynamic allocation during update loop
 * - Uses cudaStream for async execution
 * 
 * Usage:
 * @code
 * Stitcher stitcher(calibration, config);
 * stitcher.initialize();  // Call once
 * 
 * // In real-time loop:
 * stitcher.uploadImages(rgb_images);
 * stitcher.uploadDistances(distance_maps);
 * stitcher.stitch();
 * cv::Mat panorama = stitcher.downloadPanorama();
 * @endcode
 */
class Stitcher {
public:
    /**
     * @brief Configuration for stitcher dimensions
     */
    struct Config {
        int pano_width = 256;      ///< Output panorama width
        int pano_height = 128;     ///< Output panorama height
        int fisheye_width = 224;   ///< Input fisheye distance map width
        int fisheye_height = 224;  ///< Input fisheye distance map height
        int stitch_width = 672;    ///< Input RGB image width (may be higher res)
        int stitch_height = 672;   ///< Input RGB image height
        float min_dist = 0.4f;     ///< Minimum expected distance (meters)
        float max_dist = 1000.0f;  ///< Maximum expected distance (meters)
        int inpaint_iterations = 10;  ///< Number of inpainting passes
    };

    /**
     * @brief Construct a new Stitcher
     * @param calibration Calibration data for all cameras
     * @param config Stitcher configuration
     */
    Stitcher(const CalibrationSet& calibration, const Config& config);

    /**
     * @brief Destructor - frees all GPU memory
     */
    ~Stitcher();

    // Non-copyable, movable
    Stitcher(const Stitcher&) = delete;
    Stitcher& operator=(const Stitcher&) = delete;
    Stitcher(Stitcher&&) noexcept;
    Stitcher& operator=(Stitcher&&) noexcept;

    /**
     * @brief Initialize GPU buffers and precompute lookup tables
     * 
     * Must be called before any stitching operations.
     * Allocates all GPU memory and computes:
     * - Blending lookup tables
     * - Inpainting weights for each camera
     * - Validity masks
     */
    void initialize();

    /**
     * @brief Check if initialized
     */
    bool isInitialized() const { return initialized_; }

    /**
     * @brief Upload RGB images to GPU
     * 
     * @param images Vector of RGB images (BGR format, 8-bit per channel)
     *               Size: num_cameras x [stitch_height x stitch_width x 3]
     */
    void uploadImages(const std::vector<cv::Mat>& images);

    /**
     * @brief Upload distance maps to GPU
     * 
     * @param distances Vector of distance maps (CV_32FC1)
     *                  Size: num_cameras x [fisheye_height x fisheye_width]
     */
    void uploadDistances(const std::vector<cv::Mat>& distances);

    /**
     * @brief Perform stitching operation
     * 
     * Executes the full stitching pipeline:
     * 1. Reproject distance maps to reference viewpoints
     * 2. Apply inpainting to fill occlusion holes
     * 3. Blend RGB and distance into panorama
     */
    void stitch();

    /**
     * @brief Download RGB panorama from GPU
     * @return RGB panorama (CV_8UC3, BGR format)
     */
    cv::Mat downloadRGBPanorama();

    /**
     * @brief Download distance panorama from GPU
     * @return Distance panorama (CV_32FC1)
     */
    cv::Mat downloadDistancePanorama();

    /**
     * @brief Synchronize CUDA stream
     * 
     * Call this before accessing downloaded results if needed.
     */
    void synchronize();

    /**
     * @brief Get configuration
     */
    const Config& getConfig() const { return config_; }

    /**
     * @brief Get number of cameras
     */
    int getNumCameras() const { return num_cameras_; }

private:
    // Configuration
    CalibrationSet calibration_;
    Config config_;
    cuda::StitcherConfig cuda_config_;
    int num_cameras_;
    bool initialized_;

    // CUDA stream
    cudaStream_t stream_;

    // GPU buffers - Calibration (constant after init)
    cuda::Intrinsics* d_intrinsics_;       // [num_cameras]
    cuda::Rotation* d_rotations_;          // [num_cameras]
    float3* d_translations_;               // [num_cameras]
    float* d_masks_;                       // [num_cameras x fisheye_height x fisheye_width]

    // GPU buffers - Lookup tables (precomputed)
    float2* d_sampling_lut_;               // [num_cameras x pano_height x pano_width]
    float* d_blending_weights_;            // [num_cameras x pano_height x pano_width]
    uchar2* d_inpaint_weights_;            // [num_cameras x fisheye_height x fisheye_width]

    // GPU buffers - Per-frame data
    uchar3* d_stitch_images_;              // [num_cameras x stitch_height x stitch_width]
    float* d_distance_maps_;               // [num_cameras x fisheye_height x fisheye_width]
    float* d_reprojected_distances_;       // [num_cameras x fisheye_height x fisheye_width]

    // GPU buffers - Output
    uchar3* d_rgb_panorama_;               // [pano_height x pano_width]
    float* d_distance_panorama_;           // [pano_height x pano_width]

    // Host buffers for output (pinned memory for faster transfer)
    uchar3* h_rgb_panorama_;
    float* h_distance_panorama_;

    // Helper methods
    void allocateGPUMemory();
    void freeGPUMemory();
    void uploadCalibration();
    void computeMasks();
    void computeLookupTables();
    void computeInpaintingWeights();
    void reprojectDistances();
    void applyInpainting();
};

}  // namespace sphere_stereo_ros

#endif  // SPHERE_STEREO_ROS_STITCHER_HPP
