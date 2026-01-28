/**
=======================================================================
GPU-Accelerated Utilities Implementation - Host Side
Sphere Sweeping Stereo Preprocessing
=======================================================================
*/

#include "my_stereo_pkg/utils.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

namespace sphere_stereo {

// ============================================================================
// CameraCalibrationGPU Implementation - Unified Memory (Zero-Copy)
// ============================================================================

void CameraCalibrationGPU::initialize_unified(
    float fx, float fy, float cx, float cy,
    float xi, float alpha,
    const float rt_matrix[16],
    float matching_scale,
    int width, int height
) {
    if (managed_calib_) {
        cudaFree(managed_calib_);
    }
    
    // Allocate Unified Memory (accessible from both CPU and GPU)
    CUDA_CHECK(cudaMallocManaged((void**)&managed_calib_, sizeof(CameraCalibration)));
    
    // Initialize directly in Unified Memory (no cudaMemcpy needed!)
    managed_calib_->fx = fx;
    managed_calib_->fy = fy;
    managed_calib_->cx = cx;
    managed_calib_->cy = cy;
    managed_calib_->xi = xi;
    managed_calib_->alpha = alpha;
    managed_calib_->matching_scale = matching_scale;
    managed_calib_->width = width;
    managed_calib_->height = height;
    
    // Copy RT matrix
    for (int i = 0; i < 16; i++) {
        managed_calib_->rt[i] = rt_matrix[i];
    }
    
    // Prefetch to GPU for better performance (optional on Jetson)
    int device = 0;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(managed_calib_, sizeof(CameraCalibration), device, 0);
}

// ============================================================================
// CalibrationParser Implementation
// ============================================================================

std::vector<CameraCalibrationGPU> CalibrationParser::load_json_basalt(
    const std::string& json_path,
    const std::vector<int>& matching_resolution,
    const std::vector<int>& original_resolution
) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open calibration file: " + json_path);
    }
    
    nlohmann::json file_root = nlohmann::json::parse(file);
    
    // Extract value0 if it exists (Basalt format)
    nlohmann::json root;
    if (file_root.contains("value0")) {
        root = file_root["value0"];
    } else {
        root = file_root;
    }
    
    std::vector<CameraCalibrationGPU> calibrations;
    
    // Check if this is Basalt array format (T_imu_cam, intrinsics arrays)
    if (root.contains("T_imu_cam") && root.contains("intrinsics")) {
        // Basalt array format
        auto extrinsics_array = root["T_imu_cam"];
        auto intrinsics_array = root["intrinsics"];
        
        // Get resolutions
        std::vector<std::vector<int>> resolutions;
        if (root.contains("resolution")) {
            resolutions = root["resolution"].get<std::vector<std::vector<int>>>();
        } else {
            // Use provided original_resolution for all cameras
            for (size_t i = 0; i < extrinsics_array.size(); ++i) {
                resolutions.push_back(original_resolution);
            }
        }
        
        // Parse each camera
        for (size_t i = 0; i < extrinsics_array.size(); ++i) {
            auto calib_gpu = parse_basalt_camera(
                extrinsics_array[i],
                intrinsics_array[i],
                resolutions[i],
                matching_resolution
            );
            calibrations.push_back(std::move(calib_gpu));
        }
    } else {
        // Old format: cam0, cam1, etc.
        for (auto& [key, cam_json] : root.items()) {
            if (key.substr(0, 3) == "cam") {
                auto calib_gpu = parse_camera_json(
                    cam_json,
                    matching_resolution,
                    original_resolution
                );
                calibrations.push_back(std::move(calib_gpu));
            }
        }
    }
    
    return calibrations;
}

CameraCalibrationGPU CalibrationParser::parse_basalt_camera(
    const nlohmann::json& extrinsics_json,
    const nlohmann::json& intrinsics_json,
    const std::vector<int>& original_resolution,
    const std::vector<int>& matching_resolution
) {
    // Check camera type
    std::string camera_type = intrinsics_json["camera_type"].get<std::string>();
    if (camera_type != "ds") {
        throw std::runtime_error("Unsupported camera type: " + camera_type + ". Only 'ds' (double sphere) is supported.");
    }
    
    // Parse intrinsics
    auto cam_intrinsics = intrinsics_json["intrinsics"];
    float fx = cam_intrinsics["fx"].get<float>();
    float fy = cam_intrinsics["fy"].get<float>();
    float cx = cam_intrinsics["cx"].get<float>();
    float cy = cam_intrinsics["cy"].get<float>();
    float xi = cam_intrinsics["xi"].get<float>();
    float alpha = cam_intrinsics["alpha"].get<float>();
    
    // Parse extrinsics (quaternion + translation)
    float qx = extrinsics_json["qx"].get<float>();
    float qy = extrinsics_json["qy"].get<float>();
    float qz = extrinsics_json["qz"].get<float>();
    float qw = extrinsics_json["qw"].get<float>();
    
    float px = extrinsics_json["px"].get<float>();
    float py = extrinsics_json["py"].get<float>();
    float pz = extrinsics_json["pz"].get<float>();
    
    // Convert quaternion to rotation matrix
    // R = I + 2*q_v*q_v^T + 2*q_w*[q_v]_x
    float x2 = qx * qx;
    float y2 = qy * qy;
    float z2 = qz * qz;
    float xy = qx * qy;
    float xz = qx * qz;
    float yz = qy * qz;
    float wx = qw * qx;
    float wy = qw * qy;
    float wz = qw * qz;
    
    // Build 4x4 RT matrix (row-major)
    float rt_matrix[16];
    rt_matrix[0] = 1.0f - 2.0f * (y2 + z2);
    rt_matrix[1] = 2.0f * (xy - wz);
    rt_matrix[2] = 2.0f * (xz + wy);
    rt_matrix[3] = px;
    
    rt_matrix[4] = 2.0f * (xy + wz);
    rt_matrix[5] = 1.0f - 2.0f * (x2 + z2);
    rt_matrix[6] = 2.0f * (yz - wx);
    rt_matrix[7] = py;
    
    rt_matrix[8] = 2.0f * (xz - wy);
    rt_matrix[9] = 2.0f * (yz + wx);
    rt_matrix[10] = 1.0f - 2.0f * (x2 + y2);
    rt_matrix[11] = pz;
    
    rt_matrix[12] = 0.0f;
    rt_matrix[13] = 0.0f;
    rt_matrix[14] = 0.0f;
    rt_matrix[15] = 1.0f;
    
    // Compute matching resolution scale
    float matching_scale = 
        static_cast<float>(matching_resolution[0]) / 
        static_cast<float>(original_resolution[0]);
    
    std::cout << "[CalibrationParser] Creating CameraCalibrationGPU with Unified Memory..." << std::endl;
    std::cout << "  Intrinsics: fx=" << fx << ", fy=" << fy << std::endl;
    std::cout << "  Translation: [" << px << ", " << py << ", " << pz << "]" << std::endl;
    
    // Create GPU calibration object and initialize with Unified Memory
    CameraCalibrationGPU calib_gpu;
    calib_gpu.initialize_unified(
        fx, fy, cx, cy, xi, alpha,
        rt_matrix,
        matching_scale,
        original_resolution[0], original_resolution[1]
    );
    
    // CRITICAL: Synchronize to ensure Unified Memory is ready for CPU access
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "[CalibrationParser] CameraCalibrationGPU initialized successfully with Unified Memory" << std::endl;
    std::cout << "  Unified Memory address: " << calib_gpu.get_unified_ptr() << std::endl;
    
    return calib_gpu;
}

CameraCalibrationGPU CalibrationParser::parse_camera_json(
    const nlohmann::json& cam_json,
    const std::vector<int>& matching_resolution,
    const std::vector<int>& original_resolution
) {
    // Parse intrinsics
    auto calib_arr = cam_json["intrinsics"].get<std::vector<double>>();
    float fx = calib_arr[0];
    float fy = calib_arr[1];
    float cx = calib_arr[2];
    float cy = calib_arr[3];
    
    // Parse distortion parameters
    float xi = 0.0f, alpha = 0.5f;
    if (cam_json.contains("distortion_coeffs")) {
        auto dist_arr = cam_json["distortion_coeffs"].get<std::vector<double>>();
        xi = dist_arr[0];
        alpha = dist_arr.size() > 1 ? dist_arr[1] : 0.5f;
    }
    
    // Parse resolution
    int width = original_resolution[0];
    int height = original_resolution[1];
    if (cam_json.contains("resolution")) {
        auto res_arr = cam_json["resolution"].get<std::vector<int>>();
        width = res_arr[0];
        height = res_arr[1];
    }
    
    // Build RT matrix from extrinsics
    float rt_matrix[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    
    if (cam_json.contains("T_cn_cnm1")) {
        auto T_arr = cam_json["T_cn_cnm1"].get<std::vector<std::vector<double>>>();
        
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                rt_matrix[i * 4 + j] = T_arr[i][j];
            }
        }
    }
    
    // Compute matching resolution scale
    float matching_scale = (!matching_resolution.empty() && !original_resolution.empty()) ? 
        static_cast<float>(matching_resolution[0]) / static_cast<float>(original_resolution[0]) : 1.0f;
    
    // Create GPU calibration object and initialize with Unified Memory
    CameraCalibrationGPU calib_gpu;
    calib_gpu.initialize_unified(
        fx, fy, cx, cy, xi, alpha,
        rt_matrix,
        matching_scale,
        width, height
    );
    
    // CRITICAL: Synchronize to ensure Unified Memory is ready for CPU access
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return calib_gpu;
}

// ============================================================================
// GPU Unproject Implementation
// ============================================================================

void unproject_gpu(
    const float* h_uv,
    const CameraCalibrationGPU& calib,
    float* h_points_out,
    uint8_t* h_valid_out,
    int width, int height
) {
    size_t uv_size = width * height * 2 * sizeof(float);
    size_t points_size = width * height * 3 * sizeof(float);
    size_t valid_size = width * height * sizeof(uint8_t);
    
    float* d_uv = nullptr;
    float* d_points = nullptr;
    uint8_t* d_valid = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_uv, uv_size));
    CUDA_CHECK(cudaMalloc(&d_points, points_size));
    CUDA_CHECK(cudaMalloc(&d_valid, valid_size));
    
    CUDA_CHECK(cudaMemcpy(d_uv, h_uv, uv_size, cudaMemcpyHostToDevice));
    
    launch_unproject_kernel(
        d_uv,
        calib.get_unified_ptr(),
        d_points,
        d_valid,
        width, height
    );
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "ERROR after launch_unproject_kernel: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_points_out, d_points, points_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_valid_out, d_valid, valid_size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_uv);
    cudaFree(d_points);
    cudaFree(d_valid);
}

// ============================================================================
// GPU Project Implementation
// ============================================================================

void project_gpu(
    const float* h_points,
    const CameraCalibrationGPU& calib,
    float* h_uv_out,
    uint8_t* h_valid_out,
    int width, int height
) {
    size_t points_size = width * height * 3 * sizeof(float);
    size_t uv_size = width * height * 2 * sizeof(float);
    size_t valid_size = width * height * sizeof(uint8_t);
    
    float* d_points = nullptr;
    float* d_uv = nullptr;
    uint8_t* d_valid = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_points, points_size));
    CUDA_CHECK(cudaMalloc(&d_uv, uv_size));
    CUDA_CHECK(cudaMalloc(&d_valid, valid_size));
    
    CUDA_CHECK(cudaMemcpy(d_points, h_points, points_size, cudaMemcpyHostToDevice));
    
    launch_project_kernel(
        d_points,
        calib.get_unified_ptr(),
        d_uv,
        d_valid,
        width, height
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_uv_out, d_uv, uv_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_valid_out, d_valid, valid_size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_points);
    cudaFree(d_uv);
    cudaFree(d_valid);
}

// ============================================================================
// RGB to YCbCr Conversion
// ============================================================================

void rgb2ycbcr_gpu(
    const uint8_t* h_rgb,
    uint8_t* h_ycbcr_out,
    int width, int height
) {
    size_t img_size = width * height * 3 * sizeof(uint8_t);
    
    uint8_t* d_rgb = nullptr;
    uint8_t* d_ycbcr = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_rgb, img_size));
    CUDA_CHECK(cudaMalloc(&d_ycbcr, img_size));
    
    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb, img_size, cudaMemcpyHostToDevice));
    
    launch_rgb2ycbcr_kernel(d_rgb, d_ycbcr, width, height);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_ycbcr_out, d_ycbcr, img_size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_rgb);
    cudaFree(d_ycbcr);
}

// ============================================================================
// Bilinear Resampling
// ============================================================================

void resample_bilinear_gpu(
    const uint8_t* h_image,
    const float* h_sample_coords,
    int image_width, int image_height,
    uint8_t* h_output,
    int output_width, int output_height
) {
    size_t img_size = image_width * image_height * 3 * sizeof(uint8_t);
    size_t coord_size = output_width * output_height * 2 * sizeof(float);
    size_t out_size = output_width * output_height * 3 * sizeof(uint8_t);
    
    uint8_t* d_image = nullptr;
    float* d_coords = nullptr;
    uint8_t* d_output = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_image, img_size));
    CUDA_CHECK(cudaMalloc(&d_coords, coord_size));
    CUDA_CHECK(cudaMalloc(&d_output, out_size));
    
    CUDA_CHECK(cudaMemcpy(d_image, h_image, img_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coords, h_sample_coords, coord_size, cudaMemcpyHostToDevice));
    
    launch_resample_bilinear_kernel(
        d_image, d_coords, d_output,
        image_width, image_height, 3,
        output_width, output_height
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_image);
    cudaFree(d_coords);
    cudaFree(d_output);
}

} // namespace sphere_stereo
