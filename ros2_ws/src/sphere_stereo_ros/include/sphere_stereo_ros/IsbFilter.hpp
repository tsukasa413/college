/**
 * @file IsbFilter.hpp
 * @brief Inter-Scale Bilateral (ISB) Filter for edge-preserving cost volume aggregation
 * 
 * Ported from Python implementation for ROS 2 / C++ project.
 * Based on: "Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images"
 * Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
 * CVPR 2021
 * 
 * @copyright CC BY-NC-SA 3.0
 */

#ifndef SPHERE_STEREO_ROS__ISB_FILTER_HPP_
#define SPHERE_STEREO_ROS__ISB_FILTER_HPP_

#include <opencv2/core.hpp>
#include <cuda_runtime.h>

#include <vector>
#include <memory>
#include <cmath>

namespace sphere_stereo_ros {

/**
 * @brief Configuration parameters for ISB filter
 */
struct IsbFilterConfig {
    float sigma_i = 10.0f;   ///< Edge preservation (color similarity), lower = more edges
    float sigma_s = 3.0f;    ///< Spatial smoothing, higher = more weight to coarser scales
    
    /**
     * @brief Calculate var_inv_i from sigma_i
     * @return 1 / (2 * sigma_i^2)
     */
    float varInvI() const {
        return 1.0f / (2.0f * sigma_i * sigma_i);
    }
    
    /**
     * @brief Calculate var_inv_s from sigma_s
     * @return 1 / (2 * sigma_s^2)
     */
    float varInvS() const {
        return 1.0f / (2.0f * sigma_s * sigma_s);
    }
};

/**
 * @brief Inter-Scale Bilateral Filter for edge-preserving filtering
 * 
 * This filter implements the multi-scale bilateral filtering approach from
 * Section 3.2 of the CVPR 2021 paper. It uses a guide image (typically in YUV/YCbCr
 * color space) to preserve edges while smoothing the cost volume.
 * 
 * The filter builds a scale pyramid and performs:
 * 1. Downsampling pass: bilateral downsample from fine to coarse
 * 2. Upsampling pass: edge-aware upsample from coarse to fine, merging scales
 * 
 * Memory layout:
 * - Guide: HWC format (rows x cols x 3), uint8
 * - Cost volume: CHW format (candidate_count x rows x cols), float32
 * 
 * Memory management:
 * - All GPU buffers are pre-allocated at construction
 * - No dynamic allocation during filter() calls
 * - Uses cudaStream for async execution
 * 
 * Usage:
 * @code
 * IsbFilter filter(candidate_count, cols, rows);
 * 
 * // GPU-side processing (preferred):
 * filter.filter(d_guide, d_cost, config, stream);
 * 
 * // Or with host-side cv::Mat (for debugging):
 * filter.filterFromHost(guide_mat, cost_mats, config);
 * @endcode
 */
class IsbFilter {
public:
    /**
     * @brief Construct ISB filter
     * @param candidate_count Number of distance candidates in cost volume
     * @param cols Image width
     * @param rows Image height
     */
    IsbFilter(int candidate_count, int cols, int rows);
    
    /**
     * @brief Destructor - releases all CUDA resources
     */
    ~IsbFilter();
    
    // Disable copy
    IsbFilter(const IsbFilter&) = delete;
    IsbFilter& operator=(const IsbFilter&) = delete;
    
    // Enable move
    IsbFilter(IsbFilter&& other) noexcept;
    IsbFilter& operator=(IsbFilter&& other) noexcept;
    
    /**
     * @brief Apply ISB filter to GPU buffers (in-place)
     * 
     * This is the primary interface for GPU-side processing.
     * Both input buffers are modified in place.
     * 
     * @param d_guide Device pointer to guide image [rows x cols], uchar3 (HWC)
     * @param d_cost Device pointer to cost volume [candidate_count x rows x cols], float
     * @param config Filter configuration parameters
     * @param stream CUDA stream for async execution (default: 0)
     * 
     * @note Caller retains ownership of d_guide and d_cost
     */
    void filter(
        uchar3* d_guide,
        float* d_cost,
        const IsbFilterConfig& config,
        cudaStream_t stream = 0);
    
    /**
     * @brief Apply ISB filter with host-side cv::Mat input/output
     * 
     * Convenience method that handles H2D/D2H transfers internally.
     * Uses pre-allocated device buffers (no runtime allocation).
     * 
     * @param guide_io Input/output guide image (HWC, CV_8UC3). Modified in place.
     * @param cost_io Input/output cost volume slices (vector of CV_32FC1). Modified in place.
     * @param config Filter configuration parameters
     * @param stream CUDA stream for async execution (default: 0)
     */
    void filterFromHost(
        cv::Mat& guide_io,
        std::vector<cv::Mat>& cost_io,
        const IsbFilterConfig& config,
        cudaStream_t stream = 0);
    
    /**
     * @brief Get device pointer for input guide (for zero-copy input)
     * 
     * Call uploadGuide() first, then use this pointer as input.
     * @return Device pointer to internal guide buffer
     */
    uchar3* getDeviceGuidePtr() { return d_input_guide_; }
    
    /**
     * @brief Get device pointer for input cost (for zero-copy input)
     * 
     * Call uploadCost() first, then use this pointer as input.
     * @return Device pointer to internal cost buffer
     */
    float* getDeviceCostPtr() { return d_input_cost_; }
    
    /**
     * @brief Upload guide image from host to device
     * @param guide Host-side guide image (CV_8UC3, continuous)
     * @param stream CUDA stream
     */
    void uploadGuide(const cv::Mat& guide, cudaStream_t stream = 0);
    
    /**
     * @brief Upload cost volume from host to device
     * @param cost_slices Vector of cost volume slices (CV_32FC1)
     * @param stream CUDA stream
     */
    void uploadCost(const std::vector<cv::Mat>& cost_slices, cudaStream_t stream = 0);
    
    /**
     * @brief Download guide image from device to host
     * @param guide Output host-side guide image (CV_8UC3)
     * @param stream CUDA stream
     */
    void downloadGuide(cv::Mat& guide, cudaStream_t stream = 0);
    
    /**
     * @brief Download cost volume from device to host
     * @param cost_slices Output vector of cost volume slices (CV_32FC1)
     * @param stream CUDA stream
     */
    void downloadCost(std::vector<cv::Mat>& cost_slices, cudaStream_t stream = 0);
    
    /**
     * @brief Synchronize the internal CUDA stream
     */
    void synchronize();
    
    // Getters
    int getScaleCount() const { return scale_count_; }
    int getCandidateCount() const { return candidate_count_; }
    int getCols() const { return cols_; }
    int getRows() const { return rows_; }
    std::pair<int, int> getResolution() const { return {cols_, rows_}; }

private:
    int candidate_count_;
    int cols_;
    int rows_;
    int scale_count_;
    
    // GPU memory for scale pyramid (each scale level)
    std::vector<uchar3*> d_guides_;    // [scale_count]
    std::vector<float*> d_costs_;      // [scale_count]
    
    // Pre-allocated input/output buffers (for host transfer methods)
    uchar3* d_input_guide_;            // [rows x cols]
    float* d_input_cost_;              // [candidate_count x rows x cols]
    
    // Pinned host memory for faster transfers
    uchar3* h_guide_pinned_;           // [rows x cols]
    float* h_cost_pinned_;             // [candidate_count x rows x cols]
    
    // Dimensions at each scale
    std::vector<int> scale_cols_;
    std::vector<int> scale_rows_;
    
    void allocateMemory();
    void freeMemory();
};

}  // namespace sphere_stereo_ros

#endif  // SPHERE_STEREO_ROS__ISB_FILTER_HPP_
