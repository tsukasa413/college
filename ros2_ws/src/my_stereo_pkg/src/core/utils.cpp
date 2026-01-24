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
    CameraCalibration h_calib;
    CUDA_CHECK(cudaMemcpy(&h_calib, d_calib_, sizeof(CameraCalibration),
                          cudaMemcpyDeviceToHost));
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
    
    nlohmann::json root = nlohmann::json::parse(file);
    std::vector<CameraCalibrationGPU> calibrations;
    
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
    
    return calibrations;
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
