/**
=======================================================================
General Information
-------------------
This is a C++ implementation of the ISB Filter from the following paper:
Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images
Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
Proc. IEEE Computer Vision and Pattern Recognition (CVPR 2021, Oral)
=======================================================================
**/

#include "my_stereo_pkg/isb_filter.hpp"
#include "my_stereo_pkg/cuda_kernels.hpp"
#include <cmath>
#include <algorithm>

namespace my_stereo_pkg {

ISBFilter::ISBFilter(int candidate_count, const std::pair<int, int>& resolution, 
                     const at::Device& device)
    : candidate_count_(candidate_count)
    , device_(device)
{
    int cols = resolution.first;
    int rows = resolution.second;
    
    // Calculate number of pyramid scales
    // scale_count = int(min(log2(cols), log2(rows)) - 1)
    scale_count_ = static_cast<int>(
        std::min(std::log2(static_cast<double>(cols)), 
                 std::log2(static_cast<double>(rows))) - 1.0);
    
    // Allocate guide images and cost volumes for each scale
    guides_.reserve(scale_count_);
    costs_.reserve(scale_count_);
    
    for (int scale = 0; scale < scale_count_; ++scale) {
        // Calculate dimensions at this scale
        // ceil(rows / 2^scale), ceil(cols / 2^scale)
        int scale_rows = static_cast<int>(
            std::ceil(static_cast<double>(rows) / std::pow(2.0, scale)));
        int scale_cols = static_cast<int>(
            std::ceil(static_cast<double>(cols) / std::pow(2.0, scale)));
        
        // Allocate guide image: [H, W, 3] uint8
        guides_.push_back(
            at::zeros({scale_rows, scale_cols, 3}, 
                     at::TensorOptions().dtype(at::kByte).device(device_)));
        
        // Allocate cost volume: [candidate_count, H, W] float32
        costs_.push_back(
            at::zeros({candidate_count_, scale_rows, scale_cols},
                     at::TensorOptions().dtype(at::kFloat).device(device_)));
    }
}

std::pair<at::Tensor, at::Tensor> ISBFilter::apply(
    const at::Tensor& guide,
    const at::Tensor& cost,
    float sigma_i,
    float sigma_s,
    cudaStream_t stream)
{
    // Validate inputs
    TORCH_CHECK(guide.dim() == 3 && guide.size(2) == 3, 
                "Guide must be [H, W, 3] tensor");
    TORCH_CHECK(guide.dtype() == at::kByte, 
                "Guide must be uint8 tensor");
    TORCH_CHECK(cost.dim() == 3 && cost.size(0) == candidate_count_,
                "Cost must be [candidate_count, H, W] tensor");
    TORCH_CHECK(cost.dtype() == at::kFloat,
                "Cost must be float32 tensor");
    TORCH_CHECK(guide.is_cuda() && cost.is_cuda(),
                "Inputs must be on CUDA device");
    
    // Copy input to finest scale
    guides_[0].copy_(guide);
    costs_[0].copy_(cost);
    
    // Calculate variance inverses for bilateral filtering
    float var_inv_s = 1.0f / (2.0f * sigma_s * sigma_s);
    float var_inv_i = 1.0f / (2.0f * sigma_i * sigma_i);
    
    // ========================================================================
    // Downsampling pass: Build image pyramid from fine to coarse
    // ========================================================================
    for (int scale = 1; scale < scale_count_; ++scale) {
        int rows_in = guides_[scale - 1].size(0);
        int cols_in = guides_[scale - 1].size(1);
        int rows_out = guides_[scale].size(0);
        int cols_out = guides_[scale].size(1);
        
        // weight_down parameter (not used in downsample, but kept for consistency)
        float weight_down = 1.0f;
        
        // Launch downsampling kernel on specified stream
        launch_guide_downsample_2x(
            guides_[scale - 1],  // guide_in
            costs_[scale - 1],   // values_in
            guides_[scale],      // guide_out
            costs_[scale],       // values_out
            rows_in,
            cols_in,
            rows_out,
            cols_out,
            candidate_count_,
            var_inv_i,
            weight_down,
            stream  // CRITICAL: Pass stream for async execution
        );
    }
    
    // ========================================================================
    // Upsampling pass: Merge scales from coarse to fine with bilateral weights
    // ========================================================================
    for (int scale = scale_count_ - 2; scale >= 0; --scale) {
        int rows_low = guides_[scale + 1].size(0);
        int cols_low = guides_[scale + 1].size(1);
        int rows_high = guides_[scale].size(0);
        int cols_high = guides_[scale].size(1);
        
        // Calculate bilateral weights based on spatial distance
        // distance = 2^scale - 0.5
        float distance = std::pow(2.0f, static_cast<float>(scale)) - 0.5f;
        float weight_down = std::exp(-(distance * distance) * var_inv_s);
        float weight_up = 1.0f - weight_down;
        
        // Launch upsampling kernel on specified stream
        launch_guide_upsample_2x(
            guides_[scale + 1],  // guide_low
            costs_[scale + 1],   // values_low
            guides_[scale],      // guide_high (input/output)
            costs_[scale],       // values_high (input/output)
            rows_low,
            cols_low,
            rows_high,
            cols_high,
            candidate_count_,
            var_inv_i,
            weight_up,
            weight_down,
            stream  // CRITICAL: Pass stream for async execution
        );
    }
    
    // Return filtered cost volume and guide image at finest scale
    return {costs_[0], guides_[0]};
}

} // namespace my_stereo_pkg
