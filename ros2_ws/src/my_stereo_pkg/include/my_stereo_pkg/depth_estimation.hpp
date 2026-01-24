/*
 * depth_estimation.hpp
 * 
 * High-performance CUDA implementation of:
 * Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images
 * Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
 * CVPR 2021 (Oral)
 *
 * Optimization Strategy:
 * - Zero-Copy Strategy: Minimize CPU-GPU transfers, complete all processing on GPU
 * - Kernel Fusion: Fuse grid generation -> projection -> sampling into single kernel
 * - Memory Optimization: Use shared memory for cost calculations, texture memory for bilinear interpolation
 */

#ifndef DEPTH_ESTIMATION_HPP
#define DEPTH_ESTIMATION_HPP

#include <cuda_runtime.h>
#include <vector>
#include <memory>

// Type aliases for CUDA compatibility
typedef unsigned char uchar;

// ============================================================================
// CUDA Device Type Definitions
// ============================================================================

/**
 * Double Sphere Distortion Model
 * Following the calibration model from EuRoC/TUM-VI datasets
 */
struct __align__(16) DoubleSphereCalibration {
    // 4x4 rigid body transformation matrix (row-major)
    float rt[16];           
    float fx, fy;           // focal lengths
    float cx, cy;           // principal point
    float xi;               // first sphere parameter
    float alpha;            // second sphere parameter
    float width, height;    // image dimensions
    int padding;            // alignment
};

/**
 * Camera projection/unprojection parameters stored in constant memory
 * for all threads to access efficiently
 */
struct CameraConfig {
    int num_cameras;
    int matching_width;
    int matching_height;
    float min_dist;
    float max_dist;
    int candidate_count;
};

// ============================================================================
// CUDA Kernel Function Declarations
// ============================================================================

/**
 * Fused kernel: Adaptive camera selection
 * Computes which cameras provide the best depth information for each pixel
 * 
 * @param d_selected_cameras Output: [height, width] best camera index per pixel
 * @param d_max_displacement Output: [height, width] displacement metric
 * @param d_calibrations Array of camera calibrations
 * @param d_masks Validity masks for each camera
 * @param reference_calib Reference camera calibration
 * @param config Camera configuration
 */
void select_best_cameras_kernel(
    int* d_selected_cameras,
    float* d_max_displacement,
    const DoubleSphereCalibration* d_calibrations,
    const float* const* d_masks,
    const DoubleSphereCalibration& reference_calib,
    const CameraConfig& config,
    cudaStream_t stream
);

/**
 * Main fused depth estimation kernel
 * Integrates: unprojection -> reprojection -> sampling -> cost computation -> distance selection
 * 
 * Key optimization: Each thread processes one pixel across all distance candidates,
 * maintaining cost in registers to avoid intermediate buffer writes
 * 
 * @param d_distance_map Output: [height, width] estimated distance
 * @param d_reference_image Reference image for comparison
 * @param d_images Array of target images for sweeping
 * @param d_selected_cameras Per-pixel camera selection (adaptive matching)
 * @param d_calibrations Camera calibration array
 * @param d_guide Guide image for filtering (Y channel)
 * @param config Camera configuration
 * @param stream CUDA stream
 * @param d_texobjs Texture objects for images
 */
void estimate_fisheye_distance_fused_kernel(
    float* d_distance_map,
    const uchar4* d_reference_image,
    const uchar4* const* d_images,
    const int* d_selected_cameras,
    const DoubleSphereCalibration* d_calibrations,
    const uchar* d_guide,
    const CameraConfig& config,
    cudaStream_t stream,
    cudaTextureObject_t* d_texobjs
);

/**
 * Subpixel refinement via quadratic fitting
 * Performed in-kernel for selected distance indices
 * 
 * @param d_distance_map In/Out: distance map with refined sub-pixel accuracy
 * @param d_cost_volume Cost volume for fitting
 * @param config Camera configuration
 */
void refine_distance_quadratic_kernel(
    float* d_distance_map,
    const float* d_cost_volume,
    const CameraConfig& config,
    cudaStream_t stream
);

/**
 * Texture-memory based bilinear grid sampling (replaces torch.nn.functional.grid_sample)
 * 
 * @param d_output Output buffer
 * @param d_texobj Texture object for source image
 * @param d_sample_coords Normalized sample coordinates [-1, 1]
 * @param width, height Image dimensions
 */
void grid_sample_texture_kernel(
    float* d_output,
    cudaTextureObject_t d_texobj,
    const float2* d_sample_coords,
    int width, int height,
    int total_pixels,
    cudaStream_t stream
);

// ============================================================================
// C++ Host Class Definition
// ============================================================================

/**
 * RGBD_Estimator: GPU-accelerated depth estimation
 * 
 * Manages GPU memory, coordinates kernel launches, and handles calibration data.
 * Uses async streams for overlapped processing of multiple reference cameras.
 */
class RGBD_Estimator {
public:
    /**
     * Initialize depth estimator with camera calibrations and parameters
     * 
     * @param calibrations_rt Camera RT matrices (16 floats each as row-major 4x4)
     * @param calibrations_intrinsics Camera intrinsics [fx, fy, cx, cy]
     * @param calibrations_sphere Sphere model params [xi, alpha]
     * @param calibrations_resolution Image resolutions [width, height]
     * @param min_dist, max_dist Sweep volume distance range
     * @param candidate_count Number of distance samples
     * @param references_indices Which cameras to estimate depth on
     * @param reprojection_viewpoint Output panorama viewpoint (3D)
     * @param image_widths, image_heights Per-camera dimensions
     * @param matching_resolution Resolution for matching (width, height)
     * @param rgb_to_stitch_resolution Resolution for stitching
     * @param panorama_resolution Output panorama resolution
     * @param sigma_i, sigma_s ISB Filter parameters
     * @param device CUDA device ID
     */
    RGBD_Estimator(
        const std::vector<float>& calibrations_rt,
        const std::vector<float>& calibrations_intrinsics,
        const std::vector<float>& calibrations_sphere,
        const std::vector<float>& calibrations_resolution,
        float min_dist,
        float max_dist,
        int candidate_count,
        const std::vector<int>& references_indices,
        const std::vector<float>& reprojection_viewpoint,
        const std::vector<int>& image_widths,
        const std::vector<int>& image_heights,
        int matching_width,
        int matching_height,
        int rgb_to_stitch_width,
        int rgb_to_stitch_height,
        int panorama_width,
        int panorama_height,
        float sigma_i,
        float sigma_s,
        int device = 0
    );

    ~RGBD_Estimator();

    /**
     * Estimate depth on reference fisheye images
     * 
     * @param images_to_match [num_cameras] Float32 images [H, W, 3] in [0, 255] range
     * @param images_to_stitch [num_references] Float32 images for color stitching
     * @return (rgb_panorama [H, W, 3] uint8, distance_panorama [H, W] float32)
     */
    std::pair<std::vector<uint8_t>, std::vector<float>>
    estimate_RGBD_panorama(
        const std::vector<std::vector<float>>& images_to_match,
        const std::vector<std::vector<float>>& images_to_stitch
    );

private:
    // ========================================================================
    // Memory Management (GPU)
    // ========================================================================
    
    void allocate_gpu_memory();
    void deallocate_gpu_memory();
    void upload_calibrations();
    void upload_masks();

    // ========================================================================
    // Processing Kernels
    // ========================================================================
    
    std::vector<float> estimate_fisheye_distance(
        int reference_index,
        const std::vector<std::vector<float>>& images_to_match
    );

    // ========================================================================
    // Internal State
    // ========================================================================
    
    // Configuration
    int device_id_;
    CameraConfig config_;
    std::vector<DoubleSphereCalibration> calibrations_;
    std::vector<int> references_indices_;
    
    // Dimensions
    int matching_pixels_;
    int num_cameras_;
    std::vector<int> image_widths_;
    std::vector<int> image_heights_;
    
    // GPU Memory Pointers (device)
    DoubleSphereCalibration* d_calibrations_;           // Constant memory
    std::vector<float*> d_masks_;                       // Validity masks
    
    float* d_distance_map_;                            // Output distance
    float* d_cost_volume_;                             // Intermediate cost volume
    std::vector<cudaTextureObject_t> d_image_texobjs_; // Texture objects for images
    
    // CPU-side temporary buffers for data transfer
    std::vector<uint8_t> h_reference_image_;
    std::vector<std::vector<uint8_t>> h_target_images_;
    std::vector<uint8_t> h_guide_image_;
    
    // CUDA Streams for async execution
    std::vector<cudaStream_t> streams_;
    
    // Filter parameters
    float sigma_i_;
    float sigma_s_;
};

#endif // DEPTH_ESTIMATION_HPP
