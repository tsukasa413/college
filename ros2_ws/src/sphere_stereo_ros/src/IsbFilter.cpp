/**
 * @file IsbFilter.cpp
 * @brief Implementation of Inter-Scale Bilateral (ISB) Filter
 * 
 * @copyright CC BY-NC-SA 3.0
 */

#include "sphere_stereo_ros/IsbFilter.hpp"
#include "sphere_stereo_ros/cuda/isb_filter.cuh"

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <cstring>

namespace sphere_stereo_ros {

// =============================================================================
// Constructor / Destructor
// =============================================================================

IsbFilter::IsbFilter(int candidate_count, int cols, int rows)
    : candidate_count_(candidate_count)
    , cols_(cols)
    , rows_(rows)
    , d_input_guide_(nullptr)
    , d_input_cost_(nullptr)
    , h_guide_pinned_(nullptr)
    , h_cost_pinned_(nullptr)
{
    // Calculate number of scales: min(log2(cols), log2(rows)) - 1
    // This ensures even the coarsest scale has at least 2x2 pixels
    scale_count_ = static_cast<int>(
        std::min(std::log2(static_cast<double>(cols)), 
                 std::log2(static_cast<double>(rows)))) - 1;
    
    if (scale_count_ < 2) {
        throw std::invalid_argument("Image resolution too small for ISB filter. Need at least 4x4.");
    }
    
    // Calculate dimensions at each scale
    for (int scale = 0; scale < scale_count_; ++scale) {
        int divisor = 1 << scale;  // 2^scale
        scale_cols_.push_back((cols + divisor - 1) / divisor);  // ceil division
        scale_rows_.push_back((rows + divisor - 1) / divisor);
    }
    
    allocateMemory();
}

IsbFilter::~IsbFilter()
{
    freeMemory();
}

IsbFilter::IsbFilter(IsbFilter&& other) noexcept
    : candidate_count_(other.candidate_count_)
    , cols_(other.cols_)
    , rows_(other.rows_)
    , scale_count_(other.scale_count_)
    , d_guides_(std::move(other.d_guides_))
    , d_costs_(std::move(other.d_costs_))
    , d_input_guide_(other.d_input_guide_)
    , d_input_cost_(other.d_input_cost_)
    , h_guide_pinned_(other.h_guide_pinned_)
    , h_cost_pinned_(other.h_cost_pinned_)
    , scale_cols_(std::move(other.scale_cols_))
    , scale_rows_(std::move(other.scale_rows_))
{
    other.d_guides_.clear();
    other.d_costs_.clear();
    other.d_input_guide_ = nullptr;
    other.d_input_cost_ = nullptr;
    other.h_guide_pinned_ = nullptr;
    other.h_cost_pinned_ = nullptr;
    other.scale_count_ = 0;
}

IsbFilter& IsbFilter::operator=(IsbFilter&& other) noexcept
{
    if (this != &other) {
        freeMemory();
        
        candidate_count_ = other.candidate_count_;
        cols_ = other.cols_;
        rows_ = other.rows_;
        scale_count_ = other.scale_count_;
        d_guides_ = std::move(other.d_guides_);
        d_costs_ = std::move(other.d_costs_);
        d_input_guide_ = other.d_input_guide_;
        d_input_cost_ = other.d_input_cost_;
        h_guide_pinned_ = other.h_guide_pinned_;
        h_cost_pinned_ = other.h_cost_pinned_;
        scale_cols_ = std::move(other.scale_cols_);
        scale_rows_ = std::move(other.scale_rows_);
        
        other.d_guides_.clear();
        other.d_costs_.clear();
        other.d_input_guide_ = nullptr;
        other.d_input_cost_ = nullptr;
        other.h_guide_pinned_ = nullptr;
        other.h_cost_pinned_ = nullptr;
        other.scale_count_ = 0;
    }
    return *this;
}

// =============================================================================
// Memory Management
// =============================================================================

void IsbFilter::allocateMemory()
{
    cudaError_t err;
    
    // Allocate scale pyramid buffers
    d_guides_.resize(scale_count_, nullptr);
    d_costs_.resize(scale_count_, nullptr);
    
    for (int scale = 0; scale < scale_count_; ++scale) {
        int c = scale_cols_[scale];
        int r = scale_rows_[scale];
        
        // Guide: HWC format (rows * cols * 3 bytes as uchar3)
        err = cudaMalloc(&d_guides_[scale], r * c * sizeof(uchar3));
        if (err != cudaSuccess) {
            freeMemory();
            throw std::runtime_error("Failed to allocate GPU memory for guide pyramid: " +
                                    std::string(cudaGetErrorString(err)));
        }
        
        // Cost: CHW format (candidate_count * rows * cols floats)
        err = cudaMalloc(&d_costs_[scale], candidate_count_ * r * c * sizeof(float));
        if (err != cudaSuccess) {
            freeMemory();
            throw std::runtime_error("Failed to allocate GPU memory for cost pyramid: " +
                                    std::string(cudaGetErrorString(err)));
        }
    }
    
    // Allocate input/output device buffers for host transfer methods
    err = cudaMalloc(&d_input_guide_, rows_ * cols_ * sizeof(uchar3));
    if (err != cudaSuccess) {
        freeMemory();
        throw std::runtime_error("Failed to allocate d_input_guide_");
    }
    
    err = cudaMalloc(&d_input_cost_, candidate_count_ * rows_ * cols_ * sizeof(float));
    if (err != cudaSuccess) {
        freeMemory();
        throw std::runtime_error("Failed to allocate d_input_cost_");
    }
    
    // Allocate pinned host memory for faster transfers
    err = cudaMallocHost(&h_guide_pinned_, rows_ * cols_ * sizeof(uchar3));
    if (err != cudaSuccess) {
        freeMemory();
        throw std::runtime_error("Failed to allocate h_guide_pinned_");
    }
    
    err = cudaMallocHost(&h_cost_pinned_, candidate_count_ * rows_ * cols_ * sizeof(float));
    if (err != cudaSuccess) {
        freeMemory();
        throw std::runtime_error("Failed to allocate h_cost_pinned_");
    }
}

void IsbFilter::freeMemory()
{
    for (auto& ptr : d_guides_) {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
    }
    for (auto& ptr : d_costs_) {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
    }
    d_guides_.clear();
    d_costs_.clear();
    
    if (d_input_guide_) { cudaFree(d_input_guide_); d_input_guide_ = nullptr; }
    if (d_input_cost_) { cudaFree(d_input_cost_); d_input_cost_ = nullptr; }
    if (h_guide_pinned_) { cudaFreeHost(h_guide_pinned_); h_guide_pinned_ = nullptr; }
    if (h_cost_pinned_) { cudaFreeHost(h_cost_pinned_); h_cost_pinned_ = nullptr; }
}

// =============================================================================
// Filter Operations
// =============================================================================

void IsbFilter::filter(
    uchar3* d_guide,
    float* d_cost,
    const IsbFilterConfig& config,
    cudaStream_t stream)
{
    if (!d_guide || !d_cost) {
        throw std::invalid_argument("Null pointer passed to IsbFilter::filter");
    }
    
    // Copy input to scale 0 of pyramid
    cudaMemcpyAsync(d_guides_[0], d_guide,
                    rows_ * cols_ * sizeof(uchar3),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_costs_[0], d_cost,
                    candidate_count_ * rows_ * cols_ * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    
    float var_inv_i = config.varInvI();
    float var_inv_s = config.varInvS();
    
    // ========== Downsampling pass ==========
    // From fine (scale 0) to coarse (scale_count - 1)
    for (int scale = 1; scale < scale_count_; ++scale) {
        cuda::launchGuideDownsample2xKernel(
            d_guides_[scale - 1],
            d_costs_[scale - 1],
            scale_rows_[scale - 1],
            scale_cols_[scale - 1],
            d_guides_[scale],
            d_costs_[scale],
            scale_rows_[scale],
            scale_cols_[scale],
            var_inv_i,
            candidate_count_,
            stream);
    }
    
    // ========== Upsampling pass ==========
    // From coarse (scale_count - 1) to fine (scale 0)
    // Merges information from coarser scale into finer scale
    for (int scale = scale_count_ - 2; scale >= 0; --scale) {
        // Calculate weights based on "distance" at this scale
        // Python: distance = 2**scale - 0.5
        float distance = static_cast<float>(1 << scale) - 0.5f;
        float weight_down = std::exp(-distance * distance * var_inv_s);
        float weight_up = 1.0f - weight_down;
        
        cuda::launchGuideUpsample2xKernel(
            d_guides_[scale + 1],
            d_costs_[scale + 1],
            scale_rows_[scale + 1],
            scale_cols_[scale + 1],
            d_guides_[scale],
            d_costs_[scale],
            scale_rows_[scale],
            scale_cols_[scale],
            weight_up,
            weight_down,
            var_inv_i,
            candidate_count_,
            stream);
    }
    
    // Copy result back from scale 0 to caller's buffers
    cudaMemcpyAsync(d_guide, d_guides_[0],
                    rows_ * cols_ * sizeof(uchar3),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_cost, d_costs_[0],
                    candidate_count_ * rows_ * cols_ * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
}

void IsbFilter::filterFromHost(
    cv::Mat& guide_io,
    std::vector<cv::Mat>& cost_io,
    const IsbFilterConfig& config,
    cudaStream_t stream)
{
    // Validate input dimensions
    if (guide_io.rows != rows_ || guide_io.cols != cols_) {
        throw std::invalid_argument("Guide dimensions do not match filter configuration");
    }
    if (guide_io.type() != CV_8UC3) {
        throw std::invalid_argument("Guide must be CV_8UC3");
    }
    if (static_cast<int>(cost_io.size()) != candidate_count_) {
        throw std::invalid_argument("Cost vector size does not match candidate_count");
    }
    
    // Upload guide and cost
    uploadGuide(guide_io, stream);
    uploadCost(cost_io, stream);
    
    // Apply filter on device buffers
    filter(d_input_guide_, d_input_cost_, config, stream);
    
    // Download results
    downloadGuide(guide_io, stream);
    downloadCost(cost_io, stream);
    
    // Synchronize to ensure data is ready
    cudaStreamSynchronize(stream);
}

// =============================================================================
// Upload / Download Methods
// =============================================================================

void IsbFilter::uploadGuide(const cv::Mat& guide, cudaStream_t stream)
{
    if (guide.rows != rows_ || guide.cols != cols_ || guide.type() != CV_8UC3) {
        throw std::invalid_argument("Invalid guide dimensions or type");
    }
    
    const cv::Mat* src = &guide;
    cv::Mat contiguous;
    if (!guide.isContinuous()) {
        contiguous = guide.clone();
        src = &contiguous;
    }
    
    // Copy through pinned memory for better performance
    std::memcpy(h_guide_pinned_, src->data, rows_ * cols_ * sizeof(uchar3));
    cudaMemcpyAsync(d_input_guide_, h_guide_pinned_,
                    rows_ * cols_ * sizeof(uchar3),
                    cudaMemcpyHostToDevice, stream);
}

void IsbFilter::uploadCost(const std::vector<cv::Mat>& cost_slices, cudaStream_t stream)
{
    if (static_cast<int>(cost_slices.size()) != candidate_count_) {
        throw std::invalid_argument("Cost slice count does not match candidate_count");
    }
    
    // Pack cost slices into CHW format in pinned memory
    for (int z = 0; z < candidate_count_; ++z) {
        const cv::Mat& slice = cost_slices[z];
        if (slice.rows != rows_ || slice.cols != cols_ || slice.type() != CV_32FC1) {
            throw std::invalid_argument("Invalid cost slice dimensions or type at index " + std::to_string(z));
        }
        
        const cv::Mat* src = &slice;
        cv::Mat contiguous;
        if (!slice.isContinuous()) {
            contiguous = slice.clone();
            src = &contiguous;
        }
        
        std::memcpy(h_cost_pinned_ + z * rows_ * cols_,
                   src->data,
                   rows_ * cols_ * sizeof(float));
    }
    
    cudaMemcpyAsync(d_input_cost_, h_cost_pinned_,
                    candidate_count_ * rows_ * cols_ * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
}

void IsbFilter::downloadGuide(cv::Mat& guide, cudaStream_t stream)
{
    if (guide.rows != rows_ || guide.cols != cols_ || guide.type() != CV_8UC3) {
        guide = cv::Mat(rows_, cols_, CV_8UC3);
    }
    
    cudaMemcpyAsync(h_guide_pinned_, d_input_guide_,
                    rows_ * cols_ * sizeof(uchar3),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    std::memcpy(guide.data, h_guide_pinned_, rows_ * cols_ * sizeof(uchar3));
}

void IsbFilter::downloadCost(std::vector<cv::Mat>& cost_slices, cudaStream_t stream)
{
    // Ensure output vector has correct size
    if (static_cast<int>(cost_slices.size()) != candidate_count_) {
        cost_slices.resize(candidate_count_);
    }
    
    cudaMemcpyAsync(h_cost_pinned_, d_input_cost_,
                    candidate_count_ * rows_ * cols_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Unpack from CHW to vector of slices
    for (int z = 0; z < candidate_count_; ++z) {
        if (cost_slices[z].rows != rows_ || cost_slices[z].cols != cols_ || 
            cost_slices[z].type() != CV_32FC1) {
            cost_slices[z] = cv::Mat(rows_, cols_, CV_32FC1);
        }
        
        std::memcpy(cost_slices[z].data,
                   h_cost_pinned_ + z * rows_ * cols_,
                   rows_ * cols_ * sizeof(float));
    }
}

void IsbFilter::synchronize()
{
    cudaDeviceSynchronize();
}

}  // namespace sphere_stereo_ros
