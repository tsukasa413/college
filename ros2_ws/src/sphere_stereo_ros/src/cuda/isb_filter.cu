/**
 * @file isb_filter.cu
 * @brief CUDA kernels for Inter-Scale Bilateral (ISB) Filter
 * 
 * Ported from Python implementation for ROS 2 / C++ project.
 * Based on: "Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images"
 * Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
 * CVPR 2021
 * 
 * @copyright CC BY-NC-SA 3.0
 */

#include "sphere_stereo_ros/cuda/isb_filter.cuh"
#include "vec_utils.cuh"

namespace sphere_stereo_ros {
namespace cuda {

// =============================================================================
// Constants
// =============================================================================
constexpr int kBlockSize = 256;
constexpr float kEps = 0.01f;

// =============================================================================
// CUDA Kernels
// =============================================================================

/**
 * @brief Edge-preserving 2x downsampling kernel
 * 
 * See Section 3.2.1 of the paper for details.
 */
__global__ void guideDownsample2xKernel(
    const uchar3* __restrict__ guide_in,
    const float* __restrict__ cost_in,
    int rows_in,
    int cols_in,
    uchar3* __restrict__ guide_out,
    float* __restrict__ cost_out,
    int rows_out,
    int cols_out,
    float var_inv_i,
    int candidate_count)
{
    int index_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (index_out < cols_out * rows_out) {
        int x = index_out % cols_out;
        int y = index_out / cols_out;

        int x2 = 2 * x;
        int y2 = 2 * y;

        // Clamp to valid range
        int2 plus_up;
        plus_up.x = min(x2 + 1, cols_in - 1);
        plus_up.y = min(y2 + 1, rows_in - 1);

        int2 minus_up;
        minus_up.x = max(x2 - 1, 0);
        minus_up.y = max(y2 - 1, 0);

        // Read current pixel and guide
        float3 current_guide = uchar3ToFloat3(guide_in[y2 * cols_in + x2]);
        
        float3 neighbours_guide[8] = {
            uchar3ToFloat3(guide_in[minus_up.y * cols_in + minus_up.x]),
            uchar3ToFloat3(guide_in[y2 * cols_in + minus_up.x]),
            uchar3ToFloat3(guide_in[plus_up.y * cols_in + minus_up.x]),
            uchar3ToFloat3(guide_in[plus_up.y * cols_in + x2]),
            uchar3ToFloat3(guide_in[plus_up.y * cols_in + plus_up.x]),
            uchar3ToFloat3(guide_in[y2 * cols_in + plus_up.x]),
            uchar3ToFloat3(guide_in[minus_up.y * cols_in + plus_up.x]),
            uchar3ToFloat3(guide_in[minus_up.y * cols_in + x2])
        };

        // Compute bilateral downsampling coefficients
        float weights[8] = {0.25f, 0.5f, 0.25f, 0.5f, 0.25f, 0.5f, 0.25f, 0.5f};
        float weight_sum = 1.0f;
        float3 in_sum_guide = current_guide;

        for (int i = 0; i < 8; i++) {
            float diff = absSum(current_guide - neighbours_guide[i]);
            weights[i] *= (__expf(-diff * diff * var_inv_i) + kEps);
            weight_sum += weights[i];
            in_sum_guide = in_sum_guide + weights[i] * neighbours_guide[i];
        }

        // Downsample the guide
        guide_out[index_out] = float3ToUchar3(in_sum_guide / weight_sum);

        // Downsample the cost volume using previously computed coefficients
        for (int z = 0; z < candidate_count; z++) {
            float current_cost = cost_in[z * rows_in * cols_in + y2 * cols_in + x2];
            
            float neighbours_cost[8] = {
                cost_in[z * rows_in * cols_in + minus_up.y * cols_in + minus_up.x],
                cost_in[z * rows_in * cols_in + y2 * cols_in + minus_up.x],
                cost_in[z * rows_in * cols_in + plus_up.y * cols_in + minus_up.x],
                cost_in[z * rows_in * cols_in + plus_up.y * cols_in + x2],
                cost_in[z * rows_in * cols_in + plus_up.y * cols_in + plus_up.x],
                cost_in[z * rows_in * cols_in + y2 * cols_in + plus_up.x],
                cost_in[z * rows_in * cols_in + minus_up.y * cols_in + plus_up.x],
                cost_in[z * rows_in * cols_in + minus_up.y * cols_in + x2]
            };

            float in_sum_cost = current_cost;
            for (int i = 0; i < 8; i++) {
                in_sum_cost += neighbours_cost[i] * weights[i];
            }

            cost_out[z * rows_out * cols_out + y * cols_out + x] = in_sum_cost / weight_sum;
        }
    }
}

/**
 * @brief Edge-aware 2x upsampling kernel
 * 
 * See Section 3.2.1 of the main paper and Section 1 of the supplemental document.
 * Merges coarse scale (guide_in/cost_in) with finer scale (guide_inout/cost_inout).
 */
__global__ void guideUpsample2xKernel(
    const uchar3* __restrict__ guide_in,
    const float* __restrict__ cost_in,
    int rows_in,
    int cols_in,
    uchar3* __restrict__ guide_inout,
    float* __restrict__ cost_inout,
    int rows_out,
    int cols_out,
    float weight_up,
    float weight_down,
    float var_inv_i,
    int candidate_count)
{
    int index_in = blockIdx.x * blockDim.x + threadIdx.x;

    if (index_in < cols_in * rows_in) {
        int x = index_in % cols_in;
        int y = index_in / cols_in;

        // Neighbor indices at coarser scale
        int2 minus_down;
        minus_down.x = max(x - 1, 0);
        minus_down.y = max(y - 1, 0);
        
        int2 plus_down;
        plus_down.x = min(x + 1, cols_in - 1);
        plus_down.y = min(y + 1, rows_in - 1);

        // Corresponding indices at finer scale
        int x2 = 2 * x;
        int y2 = 2 * y;
        
        int2 plus_up;
        plus_up.x = min(x2 + 1, cols_out - 1);
        plus_up.y = min(y2 + 1, rows_out - 1);

        // Initial weights for different pixel configurations
        float weights1[9] = {1.0f, 1.0f, 1.0f, 1.0f, 8.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        float weights2[6] = {8.0f, 8.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        float weights3[6] = {8.0f, 8.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        float weights4[4] = {1.0f, 1.0f, 1.0f, 1.0f};

        // Read neighbor pixel values at coarser scale (3x3 neighborhood)
        float3 neighbours_guide_down[3][3];
        neighbours_guide_down[0][0] = uchar3ToFloat3(guide_in[minus_down.y * cols_in + minus_down.x]);
        neighbours_guide_down[0][1] = uchar3ToFloat3(guide_in[minus_down.y * cols_in + x]);
        neighbours_guide_down[0][2] = uchar3ToFloat3(guide_in[minus_down.y * cols_in + plus_down.x]);
        neighbours_guide_down[1][0] = uchar3ToFloat3(guide_in[y * cols_in + minus_down.x]);
        neighbours_guide_down[1][1] = uchar3ToFloat3(guide_in[y * cols_in + x]);
        neighbours_guide_down[1][2] = uchar3ToFloat3(guide_in[y * cols_in + plus_down.x]);
        neighbours_guide_down[2][0] = uchar3ToFloat3(guide_in[plus_down.y * cols_in + minus_down.x]);
        neighbours_guide_down[2][1] = uchar3ToFloat3(guide_in[plus_down.y * cols_in + x]);
        neighbours_guide_down[2][2] = uchar3ToFloat3(guide_in[plus_down.y * cols_in + plus_down.x]);

        // Arrange neighbors for each of the 4 output pixels
        float3 neighbours_guide1[9] = {
            neighbours_guide_down[0][0], neighbours_guide_down[0][1], neighbours_guide_down[0][2],
            neighbours_guide_down[1][0], neighbours_guide_down[1][1], neighbours_guide_down[1][2],
            neighbours_guide_down[2][0], neighbours_guide_down[2][1], neighbours_guide_down[2][2]
        };

        float3 neighbours_guide2[6] = {
            neighbours_guide_down[1][1], neighbours_guide_down[2][1],
            neighbours_guide_down[1][0], neighbours_guide_down[2][0],
            neighbours_guide_down[1][2], neighbours_guide_down[2][2]
        };

        float3 neighbours_guide3[6] = {
            neighbours_guide_down[1][1], neighbours_guide_down[1][2],
            neighbours_guide_down[0][1], neighbours_guide_down[0][2],
            neighbours_guide_down[2][1], neighbours_guide_down[2][2]
        };

        float3 neighbours_guide4[4] = {
            neighbours_guide_down[1][1], neighbours_guide_down[1][2],
            neighbours_guide_down[2][1], neighbours_guide_down[2][2]
        };

        // Read pixel values at finer scale
        float3 current_guide_up1 = uchar3ToFloat3(guide_inout[y2 * cols_out + x2]);
        float3 current_guide_up2 = uchar3ToFloat3(guide_inout[plus_up.y * cols_out + x2]);
        float3 current_guide_up3 = uchar3ToFloat3(guide_inout[y2 * cols_out + plus_up.x]);
        float3 current_guide_up4 = uchar3ToFloat3(guide_inout[plus_up.y * cols_out + plus_up.x]);

        float3 in_sum_guide1 = weight_up * current_guide_up1;
        float3 in_sum_guide2 = weight_up * current_guide_up2;
        float3 in_sum_guide3 = weight_up * current_guide_up3;
        float3 in_sum_guide4 = weight_up * current_guide_up4;
        
        float weight_sum1 = weight_up;
        float weight_sum2 = weight_up;
        float weight_sum3 = weight_up;
        float weight_sum4 = weight_up;

        // Inter-scale bilateral coefficient computation
        for (int i = 0; i < 9; i++) {
            float diff = absSum(neighbours_guide1[i] - current_guide_up1);
            weights1[i] *= weight_down * (__expf(-diff * diff * var_inv_i) + kEps);
            in_sum_guide1 = in_sum_guide1 + weights1[i] * neighbours_guide1[i];
            weight_sum1 += weights1[i];
        }

        for (int i = 0; i < 6; i++) {
            float diff = absSum(neighbours_guide2[i] - current_guide_up2);
            weights2[i] *= weight_down * (__expf(-diff * diff * var_inv_i) + kEps);
            in_sum_guide2 = in_sum_guide2 + weights2[i] * neighbours_guide2[i];
            weight_sum2 += weights2[i];

            diff = absSum(neighbours_guide3[i] - current_guide_up3);
            weights3[i] *= weight_down * (__expf(-diff * diff * var_inv_i) + kEps);
            in_sum_guide3 = in_sum_guide3 + weights3[i] * neighbours_guide3[i];
            weight_sum3 += weights3[i];
        }

        for (int i = 0; i < 4; i++) {
            float diff = absSum(neighbours_guide4[i] - current_guide_up4);
            weights4[i] *= weight_down * (__expf(-diff * diff * var_inv_i) + kEps);
            in_sum_guide4 = in_sum_guide4 + weights4[i] * neighbours_guide4[i];
            weight_sum4 += weights4[i];
        }

        weight_sum1 = 1.0f / weight_sum1;
        weight_sum2 = 1.0f / weight_sum2;
        weight_sum3 = 1.0f / weight_sum3;
        weight_sum4 = 1.0f / weight_sum4;

        bool set_plus_y = (y2 != plus_up.y);
        bool set_plus_x = (x2 != plus_up.x);

        // Guide upsampling
        guide_inout[y2 * cols_out + x2] = float3ToUchar3(weight_sum1 * in_sum_guide1);
        
        if (set_plus_y) {
            guide_inout[plus_up.y * cols_out + x2] = float3ToUchar3(weight_sum2 * in_sum_guide2);
        }
        if (set_plus_x) {
            guide_inout[y2 * cols_out + plus_up.x] = float3ToUchar3(weight_sum3 * in_sum_guide3);
        }
        if (set_plus_y && set_plus_x) {
            guide_inout[plus_up.y * cols_out + plus_up.x] = float3ToUchar3(weight_sum4 * in_sum_guide4);
        }

        // Cost volume upsampling
        for (int z = 0; z < candidate_count; z++) {
            float neighbours_cost_down[3][3];
            neighbours_cost_down[0][0] = cost_in[z * rows_in * cols_in + minus_down.y * cols_in + minus_down.x];
            neighbours_cost_down[0][1] = cost_in[z * rows_in * cols_in + minus_down.y * cols_in + x];
            neighbours_cost_down[0][2] = cost_in[z * rows_in * cols_in + minus_down.y * cols_in + plus_down.x];
            neighbours_cost_down[1][0] = cost_in[z * rows_in * cols_in + y * cols_in + minus_down.x];
            neighbours_cost_down[1][1] = cost_in[z * rows_in * cols_in + y * cols_in + x];
            neighbours_cost_down[1][2] = cost_in[z * rows_in * cols_in + y * cols_in + plus_down.x];
            neighbours_cost_down[2][0] = cost_in[z * rows_in * cols_in + plus_down.y * cols_in + minus_down.x];
            neighbours_cost_down[2][1] = cost_in[z * rows_in * cols_in + plus_down.y * cols_in + x];
            neighbours_cost_down[2][2] = cost_in[z * rows_in * cols_in + plus_down.y * cols_in + plus_down.x];

            // Pixel 1 (top-left)
            cost_inout[z * rows_out * cols_out + y2 * cols_out + x2] =
                (weight_up * cost_inout[z * rows_out * cols_out + y2 * cols_out + x2] +
                 weights1[0] * neighbours_cost_down[0][0] +
                 weights1[1] * neighbours_cost_down[0][1] +
                 weights1[2] * neighbours_cost_down[0][2] +
                 weights1[3] * neighbours_cost_down[1][0] +
                 weights1[4] * neighbours_cost_down[1][1] +
                 weights1[5] * neighbours_cost_down[1][2] +
                 weights1[6] * neighbours_cost_down[2][0] +
                 weights1[7] * neighbours_cost_down[2][1] +
                 weights1[8] * neighbours_cost_down[2][2]
                ) * weight_sum1;

            // Pixel 2 (bottom-left)
            if (set_plus_y) {
                cost_inout[z * rows_out * cols_out + plus_up.y * cols_out + x2] =
                    (weight_up * cost_inout[z * rows_out * cols_out + plus_up.y * cols_out + x2] +
                     weights2[0] * neighbours_cost_down[1][1] +
                     weights2[1] * neighbours_cost_down[2][1] +
                     weights2[2] * neighbours_cost_down[1][0] +
                     weights2[3] * neighbours_cost_down[2][0] +
                     weights2[4] * neighbours_cost_down[1][2] +
                     weights2[5] * neighbours_cost_down[2][2]
                    ) * weight_sum2;
            }

            // Pixel 3 (top-right)
            if (set_plus_x) {
                cost_inout[z * rows_out * cols_out + y2 * cols_out + plus_up.x] =
                    (weight_up * cost_inout[z * rows_out * cols_out + y2 * cols_out + plus_up.x] +
                     weights3[0] * neighbours_cost_down[1][1] +
                     weights3[1] * neighbours_cost_down[1][2] +
                     weights3[2] * neighbours_cost_down[0][1] +
                     weights3[3] * neighbours_cost_down[0][2] +
                     weights3[4] * neighbours_cost_down[2][1] +
                     weights3[5] * neighbours_cost_down[2][2]
                    ) * weight_sum3;
            }

            // Pixel 4 (bottom-right)
            if (set_plus_y && set_plus_x) {
                cost_inout[z * rows_out * cols_out + plus_up.y * cols_out + plus_up.x] =
                    (weight_up * cost_inout[z * rows_out * cols_out + plus_up.y * cols_out + plus_up.x] +
                     weights4[0] * neighbours_cost_down[1][1] +
                     weights4[1] * neighbours_cost_down[1][2] +
                     weights4[2] * neighbours_cost_down[2][1] +
                     weights4[3] * neighbours_cost_down[2][2]
                    ) * weight_sum4;
            }
        }
    }
}

// =============================================================================
// Host Launch Functions
// =============================================================================

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
    cudaStream_t stream)
{
    int total_out = cols_out * rows_out;
    int grid_size = (total_out + kBlockSize - 1) / kBlockSize;

    guideDownsample2xKernel<<<grid_size, kBlockSize, 0, stream>>>(
        guide_in, cost_in, rows_in, cols_in,
        guide_out, cost_out, rows_out, cols_out,
        var_inv_i, candidate_count);
}

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
    cudaStream_t stream)
{
    int total_in = cols_in * rows_in;
    int grid_size = (total_in + kBlockSize - 1) / kBlockSize;

    guideUpsample2xKernel<<<grid_size, kBlockSize, 0, stream>>>(
        guide_in, cost_in, rows_in, cols_in,
        guide_inout, cost_inout, rows_out, cols_out,
        weight_up, weight_down, var_inv_i, candidate_count);
}

}  // namespace cuda
}  // namespace sphere_stereo_ros
