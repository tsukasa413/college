/**
 * @file isb_filter.cuh
 * @brief CUDA kernel declarations for Inter-Scale Bilateral (ISB) Filter
 * 
 * Ported from Python implementation for ROS 2 / C++ project.
 * Based on: "Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images"
 * Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
 * CVPR 2021
 * 
 * The ISB filter performs edge-preserving cost volume aggregation using
 * a multi-scale bilateral filtering approach.
 * 
 * @copyright CC BY-NC-SA 3.0
 */

#ifndef SPHERE_STEREO_ROS_ISB_FILTER_CUH
#define SPHERE_STEREO_ROS_ISB_FILTER_CUH

#include <cuda_runtime.h>

namespace sphere_stereo_ros {
namespace cuda {

// =============================================================================
// Configuration
// =============================================================================

/**
 * @brief ISB Filter configuration
 */
struct IsbFilterConfig {
    int candidate_count;    ///< Number of distance candidates in cost volume
    float sigma_i;          ///< Edge preservation parameter (intensity)
    float sigma_s;          ///< Smoothing parameter (spatial)
};

// =============================================================================
// Kernel Launch Functions
// =============================================================================

/**
 * @brief Edge-preserving 2x downsampling with bilateral weighting
 * 
 * Downsamples both guide image and cost volume using edge-aware bilateral
 * weights computed from the guide image. This preserves edges during
 * the multi-scale decomposition.
 * 
 * @param guide_in Higher resolution guide image [rows_in x cols_in], uchar3 (RGB)
 * @param cost_in Higher resolution cost volume [candidate_count x rows_in x cols_in]
 * @param rows_in Input height
 * @param cols_in Input width
 * @param guide_out Output downsampled guide [rows_out x cols_out]
 * @param cost_out Output downsampled cost volume [candidate_count x rows_out x cols_out]
 * @param rows_out Output height (should be ~rows_in/2)
 * @param cols_out Output width (should be ~cols_in/2)
 * @param var_inv_i Inverse variance for intensity weighting (1/(2*sigma_i^2))
 * @param candidate_count Number of distance candidates
 * @param stream CUDA stream
 */
void launchGuideDownsample2xKernel(
    const uchar3* guide_in,
    const float* cost_in,
    int rows_in,
    int cols_in,
    uchar3* guide_out,
    float* cost_out,
    int rows_out,
    int cols_out,
    float var_inv_i,
    int candidate_count,
    cudaStream_t stream = 0);

/**
 * @brief Edge-aware 2x upsampling with inter-scale bilateral filtering
 * 
 * Merges a coarse scale with a finer scale using bilateral weights.
 * Preserves edges from the finer scale while incorporating smoothed
 * information from the coarser scale.
 * 
 * @param guide_in Lower resolution guide [rows_in x cols_in]
 * @param cost_in Lower resolution cost volume [candidate_count x rows_in x cols_in]
 * @param rows_in Coarse scale height
 * @param cols_in Coarse scale width
 * @param guide_inout Higher resolution guide (input/output) [rows_out x cols_out]
 * @param cost_inout Higher resolution cost volume (input/output)
 * @param rows_out Fine scale height
 * @param cols_out Fine scale width
 * @param weight_up Weight for finer scale contribution (1 - weight_down)
 * @param weight_down Weight for coarser scale contribution
 * @param var_inv_i Inverse variance for intensity weighting
 * @param candidate_count Number of distance candidates
 * @param stream CUDA stream
 */
void launchGuideUpsample2xKernel(
    const uchar3* guide_in,
    const float* cost_in,
    int rows_in,
    int cols_in,
    uchar3* guide_inout,
    float* cost_inout,
    int rows_out,
    int cols_out,
    float weight_up,
    float weight_down,
    float var_inv_i,
    int candidate_count,
    cudaStream_t stream = 0);

}  // namespace cuda
}  // namespace sphere_stereo_ros

#endif  // SPHERE_STEREO_ROS_ISB_FILTER_CUH
