#pragma once

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector>

// CUDA構造体の定義 (stitcher.cuと同じ)

struct Intrinsics
{
    float2 fl, principal;
    float xi, alpha;
};

struct Rotation
{
    float r[3][3];
};

// 回転パラメータをラップする構造体
struct RotationParams
{
    Rotation rotation;
};

// カメラパラメータをまとめて扱う構造体
struct CamParams
{
    Intrinsics intrinsics;
    Rotation rotation;
    float3 translation;
};

// CUDAカーネルのラッパー関数宣言

/**
 * Reproject distance map from one camera to reference viewpoint
 * @param distance_in Input distance map tensor [H, W]
 * @param distance_out Output distance map tensor [H, W] (modified in-place)
 * @param intrinsics Camera intrinsics
 * @param translation Translation from reference to camera [3]
 * @param cols Image width
 * @param rows Image height
 */
at::Tensor launch_reproject_distance(
    const at::Tensor& distance_in,
    const at::Tensor& distance_out,
    const Intrinsics& intrinsics,
    const at::Tensor& translation,
    int cols,
    int rows
);

/**
 * Create inpainting weight lookup table
 * @param inpaint_weights Output inpainting weights tensor [H, W, 2] uint8
 * @param intrinsics Camera intrinsics
 * @param translation Translation from reference to camera [3]
 * @param cols Image width
 * @param rows Image height
 * @param min_dist Minimum distance
 * @param max_dist Maximum distance
 */
at::Tensor launch_create_inpainting_weights(
    const at::Tensor& inpaint_weights,
    const Intrinsics& intrinsics,
    const at::Tensor& translation,
    int cols,
    int rows,
    float min_dist,
    float max_dist
);

/**
 * Apply inpainting to fill holes in distance map
 * @param distance_map Distance map tensor [H, W] (modified in-place)
 * @param inpaint_weights Inpainting weight lookup table [H, W, 2] uint8
 * @param cols Image width
 * @param rows Image height
 * @param max_dist Maximum distance
 */
at::Tensor launch_inpaint(
    const at::Tensor& distance_map,
    const at::Tensor& inpaint_weights,
    int cols,
    int rows,
    float max_dist
);

/**
 * Create blending lookup table for panorama stitching
 * @param sampling_lut Output sampling coordinates [num_cams, pano_h, pano_w, 2]
 * @param blending_weights Output blending weights [num_cams, pano_h, pano_w]
 * @param masks Camera masks [num_cams, H, W]
 * @param calibrations Vector of camera intrinsics
 * @param rotations Vector of rotation parameters
 * @param translations Translations tensor [num_cams * 3]
 * @param pano_cols Panorama width
 * @param pano_rows Panorama height
 * @param cols Image width
 * @param rows Image height
 * @param min_dist Minimum distance
 * @param max_dist Maximum distance
 */
std::pair<at::Tensor, at::Tensor> launch_create_blending_lut(
    const at::Tensor& sampling_lut,
    const at::Tensor& blending_weights,
    const at::Tensor& masks,
    const std::vector<Intrinsics>& calibrations,
    const std::vector<RotationParams>& rotations,
    const at::Tensor& translations,
    int pano_cols, int pano_rows,
    int cols, int rows,
    float min_dist, float max_dist
);

/**
 * Merge multiple RGBD images into panorama
 * @param sampling_lut Sampling coordinates [num_cams, pano_h, pano_w, 2]
 * @param blending_weights Blending weights [num_cams, pano_h, pano_w]
 * @param reprojected_distance_maps Reprojected distance maps [num_cams, H, W]
 * @param distance_maps Original distance maps [num_cams, H, W]
 * @param stitching_imgs Input RGB images [num_cams, img_h, img_w, 3]
 * @param translations Translations tensor [num_cams * 3]
 * @param calibrations Vector of camera intrinsics
 * @param distance_panorama Output distance panorama [pano_h, pano_w]
 * @param rgb_panorama Output RGB panorama [pano_h, pano_w, 3]
 * @param pano_cols Panorama width
 * @param pano_rows Panorama height
 * @param cols Distance map width
 * @param rows Distance map height
 * @param stitching_imgs_rows RGB image height
 * @param stitching_imgs_cols RGB image width
 */
std::pair<at::Tensor, at::Tensor> launch_merge_rgbd_panorama(
    const at::Tensor& sampling_lut,
    const at::Tensor& blending_weights,
    const at::Tensor& reprojected_distance_maps,
    const at::Tensor& distance_maps,
    const at::Tensor& stitching_imgs,
    const at::Tensor& translations,
    const std::vector<Intrinsics>& calibrations,
    const at::Tensor& distance_panorama,
    const at::Tensor& rgb_panorama,
    int pano_cols, int pano_rows,
    int cols, int rows,
    int stitching_imgs_rows, int stitching_imgs_cols
);

// ヘルパー関数

/**
 * Convert camera parameters from torch tensors
 * @param intrinsics_tensor Intrinsics tensor [fx, fy, cx, cy, xi, alpha]
 * @param rotation_tensor Rotation tensor [3, 3]
 * @param translation_tensor Translation tensor [3]
 * @return CamParams structure
 */
CamParams tensor_to_cam_params(
    const at::Tensor& intrinsics_tensor,
    const at::Tensor& rotation_tensor,
    const at::Tensor& translation_tensor
);

/**
 * Check if CUDA is available and set device
 * @return true if CUDA is available
 */
bool check_cuda_availability();

// ISB Filter CUDA kernel wrappers

/**
 * Guided downsample 2x with Inverse-Square Bilateral weighting
 * @param guide_in Input guide image [rowsIn, colsIn, channels]
 * @param values_in Input values (depth candidates) [candidate_count, rowsIn, colsIn]
 * @param guide_out Output downsampled guide image [rowsOut, colsOut, channels]
 * @param values_out Output downsampled values [candidate_count, rowsOut, colsOut]
 * @param rowsIn Input height
 * @param colsIn Input width
 * @param rowsOut Output height (typically rowsIn / 2)
 * @param colsOut Output width (typically colsIn / 2)
 * @param candidate_count Number of depth candidates
 * @param var_inv_i Inverse variance for intensity difference (1 / sigma_i^2)
 * @param weight_down Spatial weight for downsampling (exp(-dist^2 / sigma_s^2))
 */
void launch_guide_downsample_2x(
    const at::Tensor& guide_in,
    const at::Tensor& values_in,
    const at::Tensor& guide_out,
    const at::Tensor& values_out,
    int rowsIn,
    int colsIn,
    int rowsOut,
    int colsOut,
    int candidate_count,
    float var_inv_i,
    float weight_down
);

/**
 * Guided upsample 2x with Inverse-Square Bilateral weighting
 * @param guide_low Low-resolution guide image [rowsIn, colsIn, channels]
 * @param values_low Low-resolution values (depth candidates) [candidate_count, rowsIn, colsIn]
 * @param guide_high High-resolution guide image [rowsOut, colsOut, channels]
 * @param values_high Output high-resolution values [candidate_count, rowsOut, colsOut]
 * @param rowsIn Input height (low-res)
 * @param colsIn Input width (low-res)
 * @param rowsOut Output height (high-res, typically rowsIn * 2)
 * @param colsOut Output width (high-res, typically colsIn * 2)
 * @param candidate_count Number of depth candidates
 * @param var_inv_i Inverse variance for intensity difference (1 / sigma_i^2)
 * @param weight_up Spatial weight for upsampling (exp(-dist^2 / sigma_s^2))
 */
void launch_guide_upsample_2x(
    const at::Tensor& guide_low,
    const at::Tensor& values_low,
    const at::Tensor& guide_high,
    const at::Tensor& values_high,
    int rowsIn,
    int colsIn,
    int rowsOut,
    int colsOut,
    int candidate_count,
    float var_inv_i,
    float weight_up,
    float weight_down
);

// Constants (should match stitcher.cu defines)
constexpr float MIN_DIST = 0.1f;
constexpr float MAX_DIST = 100.0f;