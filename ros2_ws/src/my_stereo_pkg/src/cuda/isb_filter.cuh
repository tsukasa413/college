/**
 * isb_filter.cuh
 * 
 * CUDA implementation of Iterative Spatial Bilateral Filter
 * for cost volume filtering in sphere sweeping stereo
 * 
 * Based on:
 * Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images
 * Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
 * CVPR 2021 (Oral)
 */

#ifndef ISB_FILTER_CUH
#define ISB_FILTER_CUH

#include <cuda_runtime.h>
// Use existing vec_utils from utils_kernel.cuh (included by depth_estimation.cu)

#define ISB_EPS 0.01f

// ============================================================================
// ISB Filter CUDA Kernels
// ============================================================================

/**
 * Edge-preserving downsampling (see Section 3.2.1)
 * Bilateral filtering with 2x downsampling
 */
template<int CANDIDATE_COUNT>
__global__ void isb_downsample_kernel(
    const uchar3* guide_in,
    const float* cost_in,
    int rows_in,
    int cols_in,
    uchar3* guide_out,
    float* cost_out,
    int rows_out,
    int cols_out,
    float var_inv_i
) {
    int index_out = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index_out >= cols_out * rows_out)
        return;
    
    int x = index_out % cols_out;
    int y = index_out / cols_out;
    
    int x2 = 2 * x;
    int y2 = 2 * y;
    
    // Boundary clamping
    int2 plus_up = {
        min(x2 + 1, cols_in - 1),
        min(y2 + 1, rows_in - 1)
    };
    int2 minus_up = {
        max(x2 - 1, 0),
        max(y2 - 1, 0)
    };
    
    // Read current pixel and guide
    float3 current_guide = uchar3Tofloat3(guide_in[y2 * cols_in + x2]);
    float3 neighbours_guide[8] = {
        uchar3Tofloat3(guide_in[minus_up.y * cols_in + minus_up.x]),
        uchar3Tofloat3(guide_in[y2 * cols_in + minus_up.x]),
        uchar3Tofloat3(guide_in[plus_up.y * cols_in + minus_up.x]),
        uchar3Tofloat3(guide_in[plus_up.y * cols_in + x2]),
        uchar3Tofloat3(guide_in[plus_up.y * cols_in + plus_up.x]),
        uchar3Tofloat3(guide_in[y2 * cols_in + plus_up.x]),
        uchar3Tofloat3(guide_in[minus_up.y * cols_in + plus_up.x]),
        uchar3Tofloat3(guide_in[minus_up.y * cols_in + x2])
    };
    
    // Compute coefficients for bilateral downsampling
    float weights[8] = {0.25f, 0.5f, 0.25f, 0.5f, 0.25f, 0.5f, 0.25f, 0.5f};
    float weight_sum = 1.0f;
    float3 in_sum_guide = current_guide;
    
    for (int i = 0; i < 8; i++) {
        float diff = absSum(current_guide - neighbours_guide[i]);
        weights[i] *= (__expf(-diff * diff * var_inv_i) + ISB_EPS);
        weight_sum += weights[i];
        in_sum_guide = in_sum_guide + weights[i] * neighbours_guide[i];
    }
    
    // Downsample the guide
    guide_out[index_out] = float3Touchar3(in_sum_guide / weight_sum);
    
    // Downsample the cost volume using the previously set coefficients
    for (int z = 0; z < CANDIDATE_COUNT; z++) {
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

/**
 * Edge-aware Upsampling (See Section 3.2.1 of the main paper and Section 1. of the supplemental document)
 * The process merges a coarse scale with a finer scale while preserving edges
 */
template<int CANDIDATE_COUNT>
__global__ void isb_upsample_kernel(
    const uchar3* guide_in,
    const float* cost_in,
    int rows_in,
    int cols_in,
    uchar3* guide_in_out,
    float* cost_in_out,
    int rows_in_out,
    int cols_in_out,
    float weight_up,
    float weight_down,
    float var_inv_i
) {
    int index_in = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index_in >= cols_in * rows_in)
        return;
    
    int x = index_in % cols_in;
    int y = index_in / cols_in;
    
    // Boundary clamping for coarse scale
    int2 minus_down = {max(x - 1, 0), max(y - 1, 0)};
    int2 plus_down = {min(x + 1, cols_in - 1), min(y + 1, rows_in - 1)};
    
    // Upsampled coordinates
    int x2 = 2 * x;
    int y2 = 2 * y;
    int2 plus_up = {
        min(x2 + 1, cols_in_out - 1),
        min(y2 + 1, rows_in_out - 1)
    };
    
    // Initial weights for different upsampled pixels
    float weights1[9] = {1.f, 1.f, 1.f, 1.f, 8.f, 1.f, 1.f, 1.f, 1.f};
    float weights2[6] = {8.f, 8.f, 1.f, 1.f, 1.f, 1.f};
    float weights3[6] = {8.f, 8.f, 1.f, 1.f, 1.f, 1.f};
    float weights4[4] = {1.f, 1.f, 1.f, 1.f};
    
    // Read neighbour pixel values at coarser scale
    float3 neighbours_guide_down[3][3] = {
        {uchar3Tofloat3(guide_in[minus_down.y * cols_in + minus_down.x]),
         uchar3Tofloat3(guide_in[minus_down.y * cols_in + x]),
         uchar3Tofloat3(guide_in[minus_down.y * cols_in + plus_down.x])},
        {uchar3Tofloat3(guide_in[y * cols_in + minus_down.x]),
         uchar3Tofloat3(guide_in[y * cols_in + x]),
         uchar3Tofloat3(guide_in[y * cols_in + plus_down.x])},
        {uchar3Tofloat3(guide_in[plus_down.y * cols_in + minus_down.x]),
         uchar3Tofloat3(guide_in[plus_down.y * cols_in + x]),
         uchar3Tofloat3(guide_in[plus_down.y * cols_in + plus_down.x])}
    };
    
    // Prepare neighbor arrays for each upsampled pixel
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
    
    // Read neighbour pixel values at finer scale
    float3 current_guide_up1 = uchar3Tofloat3(guide_in_out[y2 * cols_in_out + x2]);
    float3 current_guide_up2 = uchar3Tofloat3(guide_in_out[plus_up.y * cols_in_out + x2]);
    float3 current_guide_up3 = uchar3Tofloat3(guide_in_out[y2 * cols_in_out + plus_up.x]);
    float3 current_guide_up4 = uchar3Tofloat3(guide_in_out[plus_up.y * cols_in_out + plus_up.x]);
    
    float3 in_sum_guide1 = weight_up * current_guide_up1;
    float3 in_sum_guide2 = weight_up * current_guide_up2;
    float3 in_sum_guide3 = weight_up * current_guide_up3;
    float3 in_sum_guide4 = weight_up * current_guide_up4;
    float weight_sum1 = weight_up;
    float weight_sum2 = weight_up;
    float weight_sum3 = weight_up;
    float weight_sum4 = weight_up;
    
    // Interscale bilateral upsampling coefficient computation
    for (int i = 0; i < 9; i++) {
        float diff = absSum(neighbours_guide1[i] - current_guide_up1);
        weights1[i] *= weight_down * (__expf(-diff * diff * var_inv_i) + ISB_EPS);
        in_sum_guide1 = in_sum_guide1 + weights1[i] * neighbours_guide1[i];
        weight_sum1 += weights1[i];
    }
    for (int i = 0; i < 6; i++) {
        float diff = absSum(neighbours_guide2[i] - current_guide_up2);
        weights2[i] *= weight_down * (__expf(-diff * diff * var_inv_i) + ISB_EPS);
        in_sum_guide2 = in_sum_guide2 + weights2[i] * neighbours_guide2[i];
        weight_sum2 += weights2[i];
        
        diff = absSum(neighbours_guide3[i] - current_guide_up3);
        weights3[i] *= weight_down * (__expf(-diff * diff * var_inv_i) + ISB_EPS);
        in_sum_guide3 = in_sum_guide3 + weights3[i] * neighbours_guide3[i];
        weight_sum3 += weights3[i];
    }
    for (int i = 0; i < 4; i++) {
        float diff = absSum(neighbours_guide4[i] - current_guide_up4);
        weights4[i] *= weight_down * (__expf(-diff * diff * var_inv_i) + ISB_EPS);
        in_sum_guide4 = in_sum_guide4 + weights4[i] * neighbours_guide4[i];
        weight_sum4 += weights4[i];
    }
    
    weight_sum1 = 1.f / weight_sum1;
    weight_sum2 = 1.f / weight_sum2;
    weight_sum3 = 1.f / weight_sum3;
    weight_sum4 = 1.f / weight_sum4;
    
    bool set_plus_y = (y2 != plus_up.y);
    bool set_plus_x = (x2 != plus_up.x);
    
    // Guide upsampling
    guide_in_out[y2 * cols_in_out + x2] = float3Touchar3(weight_sum1 * in_sum_guide1);
    if (set_plus_y)
        guide_in_out[plus_up.y * cols_in_out + x2] = float3Touchar3(weight_sum2 * in_sum_guide2);
    if (set_plus_x)
        guide_in_out[y2 * cols_in_out + plus_up.x] = float3Touchar3(weight_sum3 * in_sum_guide3);
    if (set_plus_y && set_plus_x)
        guide_in_out[plus_up.y * cols_in_out + plus_up.x] = float3Touchar3(weight_sum4 * in_sum_guide4);
    
    // Cost upsampling
    for (int z = 0; z < CANDIDATE_COUNT; z++) {
        float neighbours_cost_down[3][3] = {
            {cost_in[z * rows_in * cols_in + minus_down.y * cols_in + minus_down.x],
             cost_in[z * rows_in * cols_in + minus_down.y * cols_in + x],
             cost_in[z * rows_in * cols_in + minus_down.y * cols_in + plus_down.x]},
            {cost_in[z * rows_in * cols_in + y * cols_in + minus_down.x],
             cost_in[z * rows_in * cols_in + y * cols_in + x],
             cost_in[z * rows_in * cols_in + y * cols_in + plus_down.x]},
            {cost_in[z * rows_in * cols_in + plus_down.y * cols_in + minus_down.x],
             cost_in[z * rows_in * cols_in + plus_down.y * cols_in + x],
             cost_in[z * rows_in * cols_in + plus_down.y * cols_in + plus_down.x]}
        };
        
        cost_in_out[z * rows_in_out * cols_in_out + y2 * cols_in_out + x2] = 
            (weight_up * cost_in_out[z * rows_in_out * cols_in_out + y2 * cols_in_out + x2] +
             weights1[0] * neighbours_cost_down[0][0] + weights1[1] * neighbours_cost_down[0][1] +
             weights1[2] * neighbours_cost_down[0][2] + weights1[3] * neighbours_cost_down[1][0] +
             weights1[4] * neighbours_cost_down[1][1] + weights1[5] * neighbours_cost_down[1][2] +
             weights1[6] * neighbours_cost_down[2][0] + weights1[7] * neighbours_cost_down[2][1] +
             weights1[8] * neighbours_cost_down[2][2]) * weight_sum1;
        
        if (set_plus_y)
            cost_in_out[z * rows_in_out * cols_in_out + plus_up.y * cols_in_out + x2] = 
                (weight_up * cost_in_out[z * rows_in_out * cols_in_out + plus_up.y * cols_in_out + x2] +
                 weights2[0] * neighbours_cost_down[1][1] + weights2[1] * neighbours_cost_down[2][1] +
                 weights2[2] * neighbours_cost_down[1][0] + weights2[3] * neighbours_cost_down[2][0] +
                 weights2[4] * neighbours_cost_down[1][2] + weights2[5] * neighbours_cost_down[2][2]) * weight_sum2;
        
        if (set_plus_x)
            cost_in_out[z * rows_in_out * cols_in_out + y2 * cols_in_out + plus_up.x] = 
                (weight_up * cost_in_out[z * rows_in_out * cols_in_out + y2 * cols_in_out + plus_up.x] +
                 weights3[0] * neighbours_cost_down[1][1] + weights3[1] * neighbours_cost_down[1][2] +
                 weights3[2] * neighbours_cost_down[0][1] + weights3[3] * neighbours_cost_down[0][2] +
                 weights3[4] * neighbours_cost_down[2][1] + weights3[5] * neighbours_cost_down[2][2]) * weight_sum3;
        
        if (set_plus_y && set_plus_x)
            cost_in_out[z * rows_in_out * cols_in_out + plus_up.y * cols_in_out + plus_up.x] = 
                (weight_up * cost_in_out[z * rows_in_out * cols_in_out + plus_up.y * cols_in_out + plus_up.x] +
                 weights4[0] * neighbours_cost_down[1][1] + weights4[1] * neighbours_cost_down[1][2] +
                 weights4[2] * neighbours_cost_down[2][1] + weights4[3] * neighbours_cost_down[2][2]) * weight_sum4;
    }
}

#endif // ISB_FILTER_CUH
