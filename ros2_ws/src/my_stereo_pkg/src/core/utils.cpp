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
// CameraCalibrationGPU Implementation
// ============================================================================

void CameraCalibrationGPU::initialize_from_host(
    const CameraCalibration& h_calib
) {
    if (d_calib_) {
        cudaFree(d_calib_);
    }
    
    CUDA_CHECK(cudaMalloc((void**)&d_calib_, sizeof(CameraCalibration)));
    CUDA_CHECK(cudaMemcpy(d_calib_, &h_calib, sizeof(CameraCalibration),
                          cudaMemcpyHostToDevice));
}

CameraCalibration CameraCalibrationGPU::get_host_copy() const {
    if (!d_calib_) {
        std::cerr << "[CameraCalibrationGPU] Error: d_calib_ is nullptr!" << std::endl;
        throw std::runtime_error("CameraCalibrationGPU not initialized (d_calib_ is null)");
    }
    
    CameraCalibration h_calib;
    cudaError_t err = cudaMemcpy(&h_calib, d_calib_, sizeof(CameraCalibration),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[CameraCalibrationGPU] cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }
    
    return h_calib;
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
    CameraCalibration h_calib;
    
    // Check camera type
    std::string camera_type = intrinsics_json["camera_type"].get<std::string>();
    if (camera_type != "ds") {
        throw std::runtime_error("Unsupported camera type: " + camera_type + ". Only 'ds' (double sphere) is supported.");
    }
    
    // Parse intrinsics
    auto cam_intrinsics = intrinsics_json["intrinsics"];
    h_calib.intrinsics.fx = cam_intrinsics["fx"].get<float>();
    h_calib.intrinsics.fy = cam_intrinsics["fy"].get<float>();
    h_calib.intrinsics.cx = cam_intrinsics["cx"].get<float>();
    h_calib.intrinsics.cy = cam_intrinsics["cy"].get<float>();
    h_calib.intrinsics.xi = cam_intrinsics["xi"].get<float>();
    h_calib.intrinsics.alpha = cam_intrinsics["alpha"].get<float>();
    
    // Parse resolution
    h_calib.resolution_x = original_resolution[0];
    h_calib.resolution_y = original_resolution[1];
    
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
    
    h_calib.extrinsics.rotation[0][0] = 1.0f - 2.0f * (y2 + z2);
    h_calib.extrinsics.rotation[0][1] = 2.0f * (xy - wz);
    h_calib.extrinsics.rotation[0][2] = 2.0f * (xz + wy);
    
    h_calib.extrinsics.rotation[1][0] = 2.0f * (xy + wz);
    h_calib.extrinsics.rotation[1][1] = 1.0f - 2.0f * (x2 + z2);
    h_calib.extrinsics.rotation[1][2] = 2.0f * (yz - wx);
    
    h_calib.extrinsics.rotation[2][0] = 2.0f * (xz - wy);
    h_calib.extrinsics.rotation[2][1] = 2.0f * (yz + wx);
    h_calib.extrinsics.rotation[2][2] = 1.0f - 2.0f * (x2 + y2);
    
    h_calib.extrinsics.translation[0] = px;
    h_calib.extrinsics.translation[1] = py;
    h_calib.extrinsics.translation[2] = pz;
    
    // Compute matching resolution scale
    h_calib.matching_scale = 
        static_cast<float>(matching_resolution[0]) / 
        static_cast<float>(original_resolution[0]);
    
    std::cout << "[CalibrationParser] Creating CameraCalibrationGPU..." << std::endl;
    std::cout << "  Intrinsics: fx=" << h_calib.intrinsics.fx << ", fy=" << h_calib.intrinsics.fy << std::endl;
    std::cout << "  Translation: [" << h_calib.extrinsics.translation[0] << ", " 
              << h_calib.extrinsics.translation[1] << ", " 
              << h_calib.extrinsics.translation[2] << "]" << std::endl;
    
    CameraCalibrationGPU calib_gpu;
    calib_gpu.initialize_from_host(h_calib);
    
    std::cout << "[CalibrationParser] CameraCalibrationGPU initialized successfully" << std::endl;
    
    return calib_gpu;
}

CameraCalibrationGPU CalibrationParser::parse_camera_json(
    const nlohmann::json& cam_json,
    const std::vector<int>& matching_resolution,
    const std::vector<int>& original_resolution
) {
    CameraCalibration h_calib;
    
    // Parse intrinsics
    auto calib_arr = cam_json["intrinsics"].get<std::vector<double>>();
    h_calib.intrinsics.fx = calib_arr[0];
    h_calib.intrinsics.fy = calib_arr[1];
    h_calib.intrinsics.cx = calib_arr[2];
    h_calib.intrinsics.cy = calib_arr[3];
    
    // Parse distortion parameters
    if (cam_json.contains("distortion_coeffs")) {
        auto dist_arr = cam_json["distortion_coeffs"].get<std::vector<double>>();
        h_calib.intrinsics.xi = dist_arr[0];
        h_calib.intrinsics.alpha = dist_arr.size() > 1 ? dist_arr[1] : 0.5f;
    }
    
    // Parse resolution
    if (cam_json.contains("resolution")) {
        auto res_arr = cam_json["resolution"].get<std::vector<int>>();
        h_calib.resolution_x = res_arr[0];
        h_calib.resolution_y = res_arr[1];
    }
    
    // Parse camera extrinsics
    if (cam_json.contains("T_cn_cnm1")) {
        auto T_arr = cam_json["T_cn_cnm1"].get<std::vector<std::vector<double>>>();
        
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                h_calib.extrinsics.rotation[i][j] = T_arr[i][j];
            }
        }
        
        for (int i = 0; i < 3; ++i) {
            h_calib.extrinsics.translation[i] = T_arr[i][3];
        }
    }
    
    // Compute matching resolution scale
    if (!matching_resolution.empty() && !original_resolution.empty()) {
        h_calib.matching_scale = 
            static_cast<float>(matching_resolution[0]) / 
            static_cast<float>(original_resolution[0]);
    } else {
        h_calib.matching_scale = 1.0f;
    }
    
    CameraCalibrationGPU calib_gpu;
    calib_gpu.initialize_from_host(h_calib);
    
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
        calib.get_device_ptr(),
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
        calib.get_device_ptr(),
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
