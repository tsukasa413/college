/**
 * @file stitcher.cu
 * @brief CUDA kernels for panorama stitching
 * 
 * Ported from Python implementation for ROS 2 / C++ project.
 * Based on: "Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images"
 * Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
 * CVPR 2021
 * 
 * @copyright CC BY-NC-SA 3.0
 */

#include "sphere_stereo_ros/cuda/stitcher.cuh"
#include "vec_utils.cuh"

namespace sphere_stereo_ros {
namespace cuda {

// =============================================================================
// Constants
// =============================================================================
constexpr int kBlockSize = 256;
constexpr int kBlockSize2D = 16;

// =============================================================================
// Device Helper Functions
// =============================================================================

/**
 * @brief Linear interpolation in float image
 * @param sampled Image data pointer
 * @param uv Sampling coordinates
 * @param cols Image width
 * @param rows Image height (for bounds checking)
 */
__device__ __forceinline__ float interpF(
    const float* sampled, 
    float2 uv, 
    int cols)
{
    int u1 = __float2int_rd(uv.x);
    int v1 = __float2int_rd(uv.y);
    int u2 = u1 + 1;
    int v2 = v1 + 1;

    float u1f = static_cast<float>(u1);
    float u2f = static_cast<float>(u2);
    float v1f = static_cast<float>(v1);
    float v2f = static_cast<float>(v2);

    float w1 = (u2f - uv.x) * (v2f - uv.y);
    float w2 = (u2f - uv.x) * (uv.y - v1f);
    float w3 = (uv.x - u1f) * (v2f - uv.y);
    float w4 = (uv.x - u1f) * (uv.y - v1f);

    float p1 = sampled[v1 * cols + u1];
    float p2 = sampled[v2 * cols + u1];
    float p3 = sampled[v1 * cols + u2];
    float p4 = sampled[v2 * cols + u2];

    return w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4;
}

/**
 * @brief Linear interpolation in uchar3 image (HWC packed RGB format)
 * @param sampled Image data pointer (HWC layout)
 * @param uv Sampling coordinates
 * @param cols Image width
 */
__device__ __forceinline__ float3 interpRGB(
    const uchar3* sampled, 
    float2 uv, 
    int cols)
{
    int u1 = __float2int_rd(uv.x);
    int v1 = __float2int_rd(uv.y);
    int u2 = u1 + 1;
    int v2 = v1 + 1;

    float u1f = static_cast<float>(u1);
    float u2f = static_cast<float>(u2);
    float v1f = static_cast<float>(v1);
    float v2f = static_cast<float>(v2);

    float w1 = (u2f - uv.x) * (v2f - uv.y);
    float w2 = (u2f - uv.x) * (uv.y - v1f);
    float w3 = (uv.x - u1f) * (v2f - uv.y);
    float w4 = (uv.x - u1f) * (uv.y - v1f);

    float3 p1 = uchar3ToFloat3(sampled[v1 * cols + u1]);
    float3 p2 = uchar3ToFloat3(sampled[v2 * cols + u1]);
    float3 p3 = uchar3ToFloat3(sampled[v1 * cols + u2]);
    float3 p4 = uchar3ToFloat3(sampled[v2 * cols + u2]);

    return w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4;
}

/**
 * @brief Unproject pixel to unit sphere using Double Sphere model
 */
__device__ __forceinline__ float3 unproject(float2 uv, const Intrinsics& calib)
{
    float2 m = (uv - calib.principal) / calib.fl;

    float r2 = m.x * m.x + m.y * m.y;
    float denom = calib.alpha * sqrtf(fmaxf(1.0f - (2.0f * calib.alpha - 1.0f) * r2, 0.0f)) 
                  + 1.0f - calib.alpha;
    float mz = (1.0f - calib.alpha * calib.alpha * r2) / denom;
    float mz2 = mz * mz;

    float3 point = make_float3(m.x, m.y, mz);
    float scale = (mz * calib.xi + sqrtf(fmaxf(mz2 + (1.0f - calib.xi * calib.xi) * r2, 0.0f))) 
                  / (mz2 + r2);
    point = scale * point;
    point.z -= calib.xi;

    return point;
}

/**
 * @brief Project 3D point to pixel coordinates using Double Sphere model
 */
__device__ __forceinline__ float2 project(float3 point, const Intrinsics& calib)
{
    float d1 = length(point);
    float c = calib.xi * d1 + point.z;
    float d2 = norm3df(point.x, point.y, c);
    float norm = calib.alpha * d2 + (1.0f - calib.alpha) * c;

    return (calib.fl * make_float2(point.x, point.y)) / norm + calib.principal;
}

/**
 * @brief Project with validity check
 */
__device__ __forceinline__ float2 projectWithValid(
    float3 point, 
    const Intrinsics& calib, 
    bool& valid)
{
    float d1 = length(point);
    float c = calib.xi * d1 + point.z;
    float d2 = norm3df(point.x, point.y, c);
    float norm = calib.alpha * d2 + (1.0f - calib.alpha) * c;

    float w1 = (calib.alpha > 0.5f) 
               ? (1.0f - calib.alpha) / calib.alpha 
               : calib.alpha / (1.0f - calib.alpha);
    float w2 = (w1 + calib.xi) / sqrtf(2.0f * w1 * calib.xi + calib.xi * calib.xi + 1.0f);
    valid = valid && (point.z > -w2 * d1);

    return (calib.fl * make_float2(point.x, point.y)) / norm + calib.principal;
}

// =============================================================================
// CUDA Kernels
// =============================================================================

/**
 * @brief Reproject distance map using z-buffering
 */
__global__ void reprojectDistanceKernel(
    const float* __restrict__ distance_in,
    float* __restrict__ distance_out,
    const Intrinsics* __restrict__ calib,
    const float3* __restrict__ translation,
    int cols,
    int rows)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = cols * rows;

    if (idx < total) {
        float2 pixel = make_float2(
            static_cast<float>(idx % cols),
            static_cast<float>(idx / cols)
        );

        // Unproject to 3D and transform to reference viewpoint
        float3 pt = unproject(pixel, *calib);
        pt = distance_in[idx] * pt - *translation;

        // Project back to image
        float2 out_px = project(pt, *calib);
        float distance = length(pt);

        int out_idx = __float2int_rn(out_px.y) * cols + __float2int_rn(out_px.x);

        // Z-buffer update (atomic min would be better but float atomics are limited)
        if (out_idx >= 0 && out_idx < total) {
            if (distance < distance_out[out_idx]) {
                distance_out[out_idx] = distance;
            }
        }
    }
}

/**
 * @brief Create inpainting direction weights
 */
__global__ void createInpaintingWeightsKernel(
    uchar2* __restrict__ inpaint_weights,
    const Intrinsics* __restrict__ calib,
    const float3* __restrict__ translation,
    int cols,
    int rows,
    float min_dist,
    float max_dist)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = cols * rows;

    if (idx < total) {
        float2 pixel = make_float2(
            static_cast<float>(idx % cols),
            static_cast<float>(idx / cols)
        );

        // Get inpainting direction
        float3 unit = unproject(pixel, *calib);

        float2 px_close = project(min_dist * unit - *translation, *calib);
        float2 px_far = project(max_dist * unit - *translation, *calib);

        float2 inpaint_dir = px_far - px_close;
        float dir_len = length(inpaint_dir);
        if (dir_len > 1e-6f) {
            inpaint_dir = inpaint_dir / dir_len;
        }

        // Find two best neighbor pixels
        int2 neighbours[2];
        float weights[2] = {0.0f, 0.0f};

        for (int n = -1; n <= 1; n++) {
            for (int m = -1; m <= 1; m++) {
                if (n != 0 || m != 0) {
                    float2 pix_dir = make_float2(static_cast<float>(n), static_cast<float>(m));
                    float pix_len = length(pix_dir);
                    pix_dir = pix_dir / pix_len;

                    float weight = dot(pix_dir, inpaint_dir);

                    if (weight > weights[1]) {
                        weights[0] = weights[1];
                        neighbours[0] = neighbours[1];
                        weights[1] = weight;
                        neighbours[1] = make_int2(n, m);
                    } else if (weight > weights[0]) {
                        weights[0] = weight;
                        neighbours[0] = make_int2(n, m);
                    }
                }
            }
        }

        // Encode weights and offsets
        // Format: |weight(4bits)|y(2bits)|x(2bits)|
        uchar valx = (static_cast<uchar>(weights[0] * 255.0f) & 0xF0) 
                     + 4 * static_cast<uchar>(1 + neighbours[0].y) 
                     + static_cast<uchar>(1 + neighbours[0].x);
        uchar valy = (static_cast<uchar>(weights[1] * 255.0f) & 0xF0) 
                     + 4 * static_cast<uchar>(1 + neighbours[1].y) 
                     + static_cast<uchar>(1 + neighbours[1].x);

        inpaint_weights[idx] = make_uchar2(valx, valy);
    }
}

/**
 * @brief Inpainting kernel to fill holes
 */
__global__ void inpaintKernel(
    float* __restrict__ distance_map,
    const uchar2* __restrict__ inpaint_weights,
    int cols,
    int rows,
    float max_dist)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = cols * rows;
    float invalid_threshold = max_dist + 0.1f;

    if (idx < total && distance_map[idx] >= invalid_threshold) {
        int2 current = make_int2(idx % cols, idx / cols);

        if (current.x >= 1 && current.x < cols - 1 && 
            current.y >= 1 && current.y < rows - 1) {
            
            // Decode neighbor pixels
            uchar2 dir_weights = inpaint_weights[idx];
            int2 n0 = current + make_int2(dir_weights.x & 3, (dir_weights.x & 12) / 4) - 1;
            int2 n1 = current + make_int2(dir_weights.y & 3, (dir_weights.y & 12) / 4) - 1;

            float w0 = static_cast<float>(dir_weights.x & 0xF0);
            float w1 = static_cast<float>(dir_weights.y & 0xF0);

            float dist_val = 0.0f;
            float weight_sum = 0.0f;

            // Sample neighbor 0
            float d0 = distance_map[n0.y * cols + n0.x];
            if (d0 < invalid_threshold && w0 > 0.0f) {
                dist_val += w0 * d0;
                weight_sum += w0;
            }

            // Sample neighbor 1
            float d1 = distance_map[n1.y * cols + n1.x];
            if (d1 < invalid_threshold && w1 > 0.0f) {
                dist_val += w1 * d1;
                weight_sum += w1;
            }

            if (weight_sum > 0.0f) {
                distance_map[idx] = dist_val / weight_sum;
            }
        }
    }
}

/**
 * @brief Create blending lookup tables
 */
__global__ void createBlendingLutKernel(
    float2* __restrict__ sampling_lut,
    float* __restrict__ blending_weights,
    const float* __restrict__ masks,
    const Intrinsics* __restrict__ calibs,
    const Rotation* __restrict__ rotations,
    const float3* __restrict__ translations,
    int pano_cols,
    int pano_rows,
    int fisheye_cols,
    int fisheye_rows,
    int num_refs,
    float min_dist,
    float max_dist)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pano_total = pano_cols * pano_rows;

    if (idx < pano_total) {
        float2 pixel = make_float2(
            static_cast<float>(idx % pano_cols),
            static_cast<float>(idx / pano_cols)
        );

        // Convert panorama pixel to sphere point
        float phi = (pixel.y + 0.5f) * kPi / pano_rows - kHalfPi;
        float theta = (pixel.x + 0.5f) * kTwoPi / pano_cols + kPi;

        float3 unit_pano = make_float3(
            cosf(phi) * sinf(theta),
            sinf(phi),
            cosf(phi) * cosf(theta)
        );

        float weight_sum = 0.0f;

        for (int ref = 0; ref < num_refs; ref++) {
            // Transform to fisheye frame
            float3 unit_fisheye = matMul3x3(rotations[ref].r, unit_pano);

            // Project to fisheye image
            bool valid = true;
            float2 uv = projectWithValid(unit_fisheye, calibs[ref], valid);
            uv = clampToImage(uv, fisheye_cols, fisheye_rows);

            int lut_idx = ref * pano_total + idx;
            sampling_lut[lut_idx] = uv;

            // Compute blending weight
            float weight = 1e-8f;

            float2 px_near = projectWithValid(
                min_dist * unit_fisheye - translations[ref], calibs[ref], valid);
            px_near = clampToImage(px_near, fisheye_cols, fisheye_rows);
            px_near.y += static_cast<float>(ref * fisheye_rows);

            float2 px_far = projectWithValid(
                max_dist * unit_fisheye - translations[ref], calibs[ref], valid);
            px_far = clampToImage(px_far, fisheye_cols, fisheye_rows);
            px_far.y += static_cast<float>(ref * fisheye_rows);

            float mask_near = interpF(masks, px_near, fisheye_cols);
            float mask_far = interpF(masks, px_far, fisheye_cols);

            if (valid && mask_near > 0.99f && mask_far > 0.99f) {
                float2 disp = px_far - px_near;
                float disp_strength = length(disp);
                weight = expf(-disp_strength * disp_strength / 
                             (1e-4f * fisheye_rows * fisheye_cols));
            }

            blending_weights[lut_idx] = weight;
            weight_sum += weight;
        }

        // Normalize weights
        for (int ref = 0; ref < num_refs; ref++) {
            int lut_idx = ref * pano_total + idx;
            blending_weights[lut_idx] /= weight_sum;
        }
    }
}

/**
 * @brief Merge fisheye images into RGB-D panorama
 */
__global__ void mergeRGBDPanoramaKernel(
    const float2* __restrict__ sampling_lut,
    const float* __restrict__ blending_weights,
    const float* __restrict__ reprojected_distances,
    const float* __restrict__ distance_maps,
    const uchar3* __restrict__ stitch_images,
    const float3* __restrict__ translations,
    const Intrinsics* __restrict__ calibs,
    float* __restrict__ distance_panorama,
    uchar3* __restrict__ rgb_panorama,
    int pano_cols,
    int pano_rows,
    int fisheye_cols,
    int fisheye_rows,
    int stitch_cols,
    int stitch_rows,
    int num_refs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pano_total = pano_cols * pano_rows;
    int fisheye_total = fisheye_cols * fisheye_rows;
    int stitch_total = stitch_cols * stitch_rows;

    if (idx < pano_total) {
        float avg_inv_dist = 0.0f;
        float3 rgb = make_float3(0.0f, 0.0f, 0.0f);

        for (int ref = 0; ref < num_refs; ref++) {
            int lut_idx = ref * pano_total + idx;

            float weight = blending_weights[lut_idx];
            float2 uv = sampling_lut[lut_idx];

            // Sample reprojected distance
            float2 uv_dist = uv;
            uv_dist.y += static_cast<float>(ref * fisheye_rows);
            float reproj_dist = interpF(reprojected_distances, uv_dist, fisheye_cols);
            avg_inv_dist += weight / reproj_dist;

            // Sample original distance for color reprojection
            float orig_dist = interpF(distance_maps, uv_dist, fisheye_cols);
            float dist_for_color = fminf(reproj_dist, orig_dist);

            // Reproject to get RGB sampling location
            float3 pt = unproject(uv, calibs[ref]);
            pt = dist_for_color * pt + translations[ref];
            float2 rgb_uv = project(pt, calibs[ref]);

            // Scale for different resolution
            float2 scale_ratio = make_float2(
                static_cast<float>(stitch_cols) / fisheye_cols,
                static_cast<float>(stitch_rows) / fisheye_rows
            );
            rgb_uv = rgb_uv * scale_ratio;
            rgb_uv = clampToImage(rgb_uv, stitch_cols, stitch_rows);
            rgb_uv.y += static_cast<float>(ref * stitch_rows);

            // Sample and blend color
            rgb = rgb + weight * interpRGB(stitch_images, rgb_uv, stitch_cols);
        }

        distance_panorama[idx] = 1.0f / avg_inv_dist;
        rgb_panorama[idx] = float3ToUchar3(rgb);
    }
}

/**
 * @brief Fill kernel
 */
__global__ void fillKernel(float* data, float value, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = value;
    }
}

// =============================================================================
// Host Launch Functions
// =============================================================================

void launchReprojectDistanceKernel(
    const float* distance_in,
    float* distance_out,
    const Intrinsics* intrinsics,
    const float3* translation,
    int cols,
    int rows,
    cudaStream_t stream)
{
    int total = cols * rows;
    int grid_size = (total + kBlockSize - 1) / kBlockSize;

    reprojectDistanceKernel<<<grid_size, kBlockSize, 0, stream>>>(
        distance_in, distance_out, intrinsics, translation, cols, rows);
}

void launchCreateInpaintingWeightsKernel(
    uchar2* inpaint_weights,
    const Intrinsics* intrinsics,
    const float3* translation,
    int cols,
    int rows,
    float min_dist,
    float max_dist,
    cudaStream_t stream)
{
    int total = cols * rows;
    int grid_size = (total + kBlockSize - 1) / kBlockSize;

    createInpaintingWeightsKernel<<<grid_size, kBlockSize, 0, stream>>>(
        inpaint_weights, intrinsics, translation, cols, rows, min_dist, max_dist);
}

void launchInpaintKernel(
    float* distance_map,
    const uchar2* inpaint_weights,
    int cols,
    int rows,
    float max_dist,
    cudaStream_t stream)
{
    int total = cols * rows;
    int grid_size = (total + kBlockSize - 1) / kBlockSize;

    inpaintKernel<<<grid_size, kBlockSize, 0, stream>>>(
        distance_map, inpaint_weights, cols, rows, max_dist);
}

void launchCreateBlendingLutKernel(
    float2* sampling_lut,
    float* blending_weights,
    const float* masks,
    const Intrinsics* intrinsics,
    const Rotation* rotations,
    const float3* translations,
    const StitcherConfig& config,
    cudaStream_t stream)
{
    int pano_total = config.pano_cols * config.pano_rows;
    int grid_size = (pano_total + kBlockSize - 1) / kBlockSize;

    createBlendingLutKernel<<<grid_size, kBlockSize, 0, stream>>>(
        sampling_lut, blending_weights, masks,
        intrinsics, rotations, translations,
        config.pano_cols, config.pano_rows,
        config.fisheye_cols, config.fisheye_rows,
        config.num_references,
        config.min_dist, config.max_dist);
}

void launchMergeRGBDPanoramaKernel(
    const float2* sampling_lut,
    const float* blending_weights,
    const float* reprojected_distances,
    const float* distance_maps,
    const uchar3* stitch_images,
    const float3* translations,
    const Intrinsics* intrinsics,
    float* distance_panorama,
    uchar3* rgb_panorama,
    const StitcherConfig& config,
    cudaStream_t stream)
{
    int pano_total = config.pano_cols * config.pano_rows;
    int grid_size = (pano_total + kBlockSize - 1) / kBlockSize;

    mergeRGBDPanoramaKernel<<<grid_size, kBlockSize, 0, stream>>>(
        sampling_lut, blending_weights,
        reprojected_distances, distance_maps,
        stitch_images, translations, intrinsics,
        distance_panorama, rgb_panorama,
        config.pano_cols, config.pano_rows,
        config.fisheye_cols, config.fisheye_rows,
        config.stitch_cols, config.stitch_rows,
        config.num_references);
}

void launchFillKernel(
    float* data,
    float value,
    int count,
    cudaStream_t stream)
{
    int grid_size = (count + kBlockSize - 1) / kBlockSize;
    fillKernel<<<grid_size, kBlockSize, 0, stream>>>(data, value, count);
}

}  // namespace cuda
}  // namespace sphere_stereo_ros
