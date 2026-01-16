#pragma once

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <vector_types.h>

namespace my_stereo {

/**
 * Double Sphere Camera Model calibration parameters
 */
struct Calibration {
    float2 fl;                    // Focal length (fx, fy)
    float2 principal;             // Principal point (cx, cy)
    float xi;                     // First distortion parameter
    float alpha;                  // Second distortion parameter
    float2 matching_scale;        // Scale factor for matching resolution (scale_x, scale_y)
    at::Tensor rt;                // [4, 4] Extrinsic matrix (rotation + translation)
    
    Calibration() : xi(0.0f), alpha(0.0f) {
        fl = {0.0f, 0.0f};
        principal = {0.0f, 0.0f};
        matching_scale = {1.0f, 1.0f};
    }
};

} // namespace my_stereo
