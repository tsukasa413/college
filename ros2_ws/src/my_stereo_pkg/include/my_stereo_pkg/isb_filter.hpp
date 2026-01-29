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

#pragma once

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <vector>

namespace my_stereo_pkg {

/**
 * Fast edge-preserving filter for cost volume aggregation
 * Implements Inverse-Square Bilateral (ISB) filtering with hierarchical processing
 */
class ISBFilter {
public:
    /**
     * Constructor
     * @param candidate_count Number of depth candidates (cost volume channels)
     * @param resolution Image resolution (cols, rows)
     * @param device CUDA device for processing
     */
    ISBFilter(int candidate_count, const std::pair<int, int>& resolution, 
              const at::Device& device);

    /**
     * Apply edge-preserving filter to cost volume
     * @param guide Guide image [H, W, 3] (uint8) for edge preservation
     * @param cost Cost volume [candidate_count, H, W] (float32) to be filtered
     * @param sigma_i Edge preservation parameter (lower = preserve edges more)
     * @param sigma_s Smoothing parameter (higher = more smoothing from coarse scales)
     * @param stream CUDA stream for async execution (default: current stream)
     * @return Pair of filtered cost volume and guide image
     */
    std::pair<at::Tensor, at::Tensor> apply(
        const at::Tensor& guide,
        const at::Tensor& cost,
        float sigma_i,
        float sigma_s,
        cudaStream_t stream = nullptr);

    /**
     * Get the number of pyramid scales used
     */
    int get_scale_count() const { return scale_count_; }

private:
    int candidate_count_;           // Number of depth candidates
    int scale_count_;               // Number of pyramid scales
    at::Device device_;             // CUDA device
    
    std::vector<at::Tensor> guides_;  // Guide images at each scale
    std::vector<at::Tensor> costs_;   // Cost volumes at each scale
};

} // namespace my_stereo_pkg
