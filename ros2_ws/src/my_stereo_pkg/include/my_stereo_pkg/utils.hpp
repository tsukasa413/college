/**
=======================================================================
GPU-Accelerated Utilities Header - Host Side
Sphere Sweeping Stereo Preprocessing
=======================================================================
*/

#ifndef SPHERE_STEREO_UTILS_HPP
#define SPHERE_STEREO_UTILS_HPP

#include <vector>
#include <string>
#include <memory>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

namespace sphere_stereo {

// ============================================================================
// Double Sphere Camera Model Structures
// ============================================================================

struct Intrinsics {
    float fx, fy, cx, cy;
    float xi, alpha;
};

struct CameraExtrinsics {
    float rotation[3][3];
    float translation[3];
};

struct CameraCalibration {
    Intrinsics intrinsics;
    CameraExtrinsics extrinsics;
    float matching_scale;
    int resolution_x, resolution_y;
};

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

// ============================================================================
// Host-side Calibration Wrapper
// ============================================================================

/**
 * Encapsulates camera calibration and provides GPU memory management
 */
class CameraCalibrationGPU {
public:
    CameraCalibrationGPU() : d_calib_(nullptr) {}
    
    ~CameraCalibrationGPU() {
        if (d_calib_) {
            cudaFree(d_calib_);
        }
    }
    
    /**
     * Initialize from host calibration structure
     */
    void initialize_from_host(
        const CameraCalibration& h_calib
    );
    
    /**
     * Get device pointer for kernel calls
     */
    CameraCalibration* get_device_ptr() const {
        return d_calib_;
    }
    
    /**
     * Get CPU copy of calibration
     */
    CameraCalibration get_host_copy() const;
    
private:
    CameraCalibration* d_calib_;  // Device pointer
};

// ============================================================================
// Image Buffers and GPU Memory Management
// ============================================================================

/**
 * GPU Image Buffer with Unified Memory support
 */
template<typename T>
class GPUBuffer {
public:
    GPUBuffer(size_t width, size_t height, size_t channels = 1)
        : width_(width), height_(height), channels_(channels),
          size_(width * height * channels), ptr_(nullptr) {}
    
    ~GPUBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }
    
    /**
     * Allocate Unified Memory (accessible from both CPU and GPU)
     */
    void allocate_unified() {
        if (ptr_) cudaFree(ptr_);
        CUDA_CHECK(cudaMallocManaged((void**)&ptr_, size_ * sizeof(T)));
        CUDA_CHECK(cudaMemAdvise(ptr_, size_ * sizeof(T), 
                                 cudaMemAdviseSetReadMostly, 0));
    }
    
    /**
     * Allocate GPU memory only
     */
    void allocate_device() {
        if (ptr_) cudaFree(ptr_);
        CUDA_CHECK(cudaMalloc((void**)&ptr_, size_ * sizeof(T)));
    }
    
    /**
     * Get device pointer
     */
    T* device_ptr() { return ptr_; }
    const T* device_ptr() const { return ptr_; }
    
    /**
     * Get host pointer (for Unified Memory)
     */
    T* host_ptr() { return ptr_; }
    const T* host_ptr() const { return ptr_; }
    
    /**
     * Synchronize Unified Memory
     */
    void synchronize() {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    /**
     * Get buffer dimensions
     */
    size_t width() const { return width_; }
    size_t height() const { return height_; }
    size_t channels() const { return channels_; }
    size_t size() const { return size_; }
    
    /**
     * Copy from host to device
     */
    void copy_from_host(const T* h_ptr, cudaStream_t stream = 0) {
        CUDA_CHECK(cudaMemcpyAsync(ptr_, h_ptr, size_ * sizeof(T),
                                    cudaMemcpyHostToDevice, stream));
    }
    
    /**
     * Copy from device to host
     */
    void copy_to_host(T* h_ptr, cudaStream_t stream = 0) {
        CUDA_CHECK(cudaMemcpyAsync(h_ptr, ptr_, size_ * sizeof(T),
                                    cudaMemcpyDeviceToHost, stream));
    }
    
private:
    size_t width_, height_, channels_, size_;
    T* ptr_;
    
    #define CUDA_CHECK(call) \
        if ((call) != cudaSuccess) { \
            throw std::runtime_error("CUDA error"); \
        }
};

// ============================================================================
// Calibration Parsing (JSON to GPU)
// ============================================================================

/**
 * Parse Basalt-format JSON calibration file
 * Returns calibration data ready for GPU transfer
 */
class CalibrationParser {
public:
    /**
     * Load from JSON file (Basalt format)
     */
    static std::vector<CameraCalibrationGPU> load_json_basalt(
        const std::string& json_path,
        const std::vector<int>& matching_resolution,
        const std::vector<int>& original_resolution = {}
    );
    
    /**
     * Parse single camera from JSON object
     */
    static CameraCalibrationGPU parse_camera_json(
        const nlohmann::json& cam_json,
        const std::vector<int>& matching_resolution,
        const std::vector<int>& original_resolution
    );
};

// ============================================================================
// GPU Utility Functions
// ============================================================================

/**
 * Unproject pixel coordinates to 3D points using GPU
 * 
 * @param h_uv: Host UV coordinates [H*W*2] float32
 * @param calib: Camera calibration
 * @param h_points_out: Output 3D points [H*W*3] float32 (pre-allocated)
 * @param h_valid_out: Output validity mask [H*W] uint8 (pre-allocated)
 * @param width, height: Image dimensions
 */
void unproject_gpu(
    const float* h_uv,
    const CameraCalibrationGPU& calib,
    float* h_points_out,
    uint8_t* h_valid_out,
    int width, int height
);

/**
 * Project 3D points to pixel coordinates using GPU
 */
void project_gpu(
    const float* h_points,
    const CameraCalibrationGPU& calib,
    float* h_uv_out,
    uint8_t* h_valid_out,
    int width, int height
);

/**
 * RGB to YCbCr conversion on GPU
 * Faster than OpenCV for large batches
 */
void rgb2ycbcr_gpu(
    const uint8_t* h_rgb,
    uint8_t* h_ycbcr_out,
    int width, int height
);

/**
 * Bilinear resampling with texture interpolation
 */
void resample_bilinear_gpu(
    const uint8_t* h_image,
    const float* h_sample_coords,
    int image_width, int image_height,
    uint8_t* h_output,
    int output_width, int output_height
);

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Convert raw pointer to GPU buffer
 */
template<typename T>
GPUBuffer<T> ptr_to_gpu(const T* ptr, size_t width, size_t height, size_t channels = 1) {
    GPUBuffer<T> buffer(width, height, channels);
    buffer.allocate_device();
    buffer.copy_from_host(ptr);
    return buffer;
}

/**
 * Convert GPU buffer to raw pointer
 */
template<typename T>
void gpu_to_ptr(const GPUBuffer<T>& buffer, T* h_ptr) {
    const_cast<GPUBuffer<T>&>(buffer).copy_to_host(h_ptr);
}

} // namespace sphere_stereo

// Forward declarations for CUDA kernels
extern "C" {
    void launch_unproject_kernel(
        const float* d_uv,
        const void* d_calib_data,
        float* d_xyz,
        uint8_t* d_valid,
        int width, int height
    );
    
    void launch_project_kernel(
        const float* d_xyz,
        const void* d_calib_data,
        float* d_uv,
        uint8_t* d_valid,
        int width, int height
    );
    
    void launch_rgb2ycbcr_kernel(
        const uint8_t* d_rgb,
        uint8_t* d_ycbcr,
        int width, int height
    );
    
    void launch_resample_bilinear_kernel(
        const uint8_t* d_image,
        const float* d_coords,
        uint8_t* d_output,
        int image_width, int image_height, int channels,
        int output_width, int output_height
    );
}


#endif // SPHERE_STEREO_UTILS_HPP
