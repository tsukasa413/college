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

// ============================================================================
// Fused Depth Estimation Structures
// ============================================================================

/**
 * Double Sphere Camera Model Parameters
 * Reference: https://arxiv.org/abs/1807.08957
 */
struct DoubleSphereParams {
    float fx, fy;           // Focal lengths
    float cx, cy;           // Principal point
    float xi;               // First sphere projection parameter
    float alpha;            // Second sphere projection parameter
    float scale_x, scale_y; // Matching scale factors
};

/**
 * Camera Extrinsics (RT matrix in row-major order)
 * Stored as float[12] for efficient memory access
 * Layout: [R00, R01, R02, T0,
 *          R10, R11, R12, T1,
 *          R20, R21, R22, T2]
 */
struct CameraExtrinsics {
    float rt[12];  // 3x4 matrix in row-major
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
 * @param stream CUDA stream for async execution (nullptr = default stream)
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
    float weight_down,
    cudaStream_t stream = nullptr
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
 * @param weight_down Spatial weight for downsampling
 * @param stream CUDA stream for async execution (nullptr = default stream)
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
    float weight_down,
    cudaStream_t stream = nullptr
);

// Cost Volume Computation CUDA kernel wrappers

/**
 * Stage 1: Compute raw cost volume with texture acceleration
 * @param images All fisheye images [num_cameras][H, W, 3] float32
 * @param reference_image Reference image [H, W, 3] float32
 * @param selected_camera_map Camera selection map [H, W] int32
 * @param distance_candidates Distance values [candidate_count] float32
 * @param camera_params Camera intrinsics [num_cameras]
 * @param camera_rts Relative RT matrices [num_cameras]
 * @param cost_volume_out Output cost volume [D, H, W] float32
 * @param ref_camera_idx Reference camera index
 * @param rows Image height
 * @param cols Image width
 */
void launch_compute_costs(
    const std::vector<at::Tensor>& images,
    const at::Tensor& reference_image,
    const at::Tensor& selected_camera_map,
    const at::Tensor& distance_candidates,
    const std::vector<DoubleSphereParams>& camera_params,
    const std::vector<CameraExtrinsics>& camera_rts,
    const at::Tensor& cost_volume_out,
    int ref_camera_idx,
    int rows,
    int cols
);

/**
 * Initialize constant memory with distance candidates (CALL ONCE at startup)
 * Eliminates per-frame cudaMemcpyToSymbol synchronization overhead
 */
void initialize_constant_memory(
    const float* distance_candidates,
    int candidate_count
);

/**
 * Update CUDA texture from GPU tensor (ZERO CPU SYNC)
 * Critical: Eliminates .cpu() bottleneck in hot path
 */
void update_texture_from_gpu(
    const at::Tensor& image,
    cudaArray* cuArray,
    cudaStream_t stream
);

/**
 * ASYNC version with pre-created textures (ELIMINATES .cpu() sync)
 * @param texture_objects Pre-allocated texture objects [num_cameras]
 * @param images Source images for texture update [num_cameras][H,W,3]
 * @param stream CUDA stream for async execution
 */
void launch_compute_costs_async(
    const std::vector<cudaTextureObject_t>& texture_objects,
    const std::vector<cudaArray*>& texture_arrays,
    const std::vector<at::Tensor>& images,
    const at::Tensor& reference_image,
    const at::Tensor& selected_camera_map,
    const at::Tensor& distance_candidates,
    const std::vector<DoubleSphereParams>& camera_params,
    const std::vector<CameraExtrinsics>& camera_rts,
    const at::Tensor& cost_volume_out,
    int ref_camera_idx,
    int rows,
    int cols,
    cudaStream_t stream
);

/**
 * Stage 3: Compute final depth from ISB-filtered cost volume
 * @param cost_volume Filtered cost volume [D, H, W] float32
 * @param distance_candidates Distance values [candidate_count] float32
 * @param distance_map_out Output distance map [H, W] float32
 * @param rows Image height
 * @param cols Image width
 */
void launch_final_depth(
    const at::Tensor& cost_volume,
    const at::Tensor& distance_candidates,
    const at::Tensor& distance_map_out,
    int rows,
    int cols
);

/**
 * Stage 3 FP16: Compute final depth using __half for 2x memory bandwidth
 * @param cost_volume Filtered cost volume [D, H, W] FP16
 * @param distance_candidates Distance values [candidate_count] FP16
 * @param distance_map_out Output distance map [H, W] FP16
 * @param rows Image height
 * @param cols Image width
 */
void launch_final_depth_fp16(
    const at::Tensor& cost_volume,
    const at::Tensor& distance_candidates,
    const at::Tensor& distance_map_out,
    int rows,
    int cols
);

/**
 * RGB to YCbCr color space conversion (eliminates LibTorch overhead)
 * @param rgb_in RGB image [H, W, 3] uint8
 * @param ycbcr_out YCbCr output [H, W, 3] uint8 (pre-allocated)
 * @param stream CUDA stream for async execution
 */
void launch_rgb_to_ycbcr(
    const at::Tensor& rgb_in,
    at::Tensor& ycbcr_out,
    cudaStream_t stream = 0
);

// Constants (should match stitcher.cu defines)
constexpr float MIN_DIST = 0.1f;
constexpr float MAX_DIST = 100.0f;