/**
 * Standalone C++ RGBD Estimator
 * 
 * Pure C++ implementation for full-sphere stereo depth estimation.
 * No Python dependencies - reads calibration JSON, loads images,
 * runs depth estimation, and saves results.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

// OpenCV for image I/O
#include <opencv2/opencv.hpp>

// Eigen for matrix operations
#include <Eigen/Dense>

// JSON parser
#include <nlohmann/json.hpp>

// LibTorch
#include <torch/torch.h>

// CUDA types
#include <cuda_runtime.h>

// Our RGBDEstimator
#include "my_stereo_pkg/depth_estimation.hpp"
#include "my_stereo_pkg/calibration.hpp"

using json = nlohmann::json;
using namespace my_stereo_pkg;


/**
 * Calculate matching scale from original to target resolution
 * Matches Python: matching_scale = torch.tensor([width, height], device=device) / original_resolution
 */
Eigen::Vector2f calculate_matching_scale(
    const std::pair<int, int>& original_resolution,  // (width, height)
    const std::pair<int, int>& matching_resolution   // (width, height)
) {
    float scale_x = static_cast<float>(matching_resolution.first) / static_cast<float>(original_resolution.first);
    float scale_y = static_cast<float>(matching_resolution.second) / static_cast<float>(original_resolution.second);
    return Eigen::Vector2f(scale_x, scale_y);
}


/**
 * Parse calibration JSON and create Calibration objects
 * Matches Python: parse_json_calib() in utils.py
 */
std::vector<Calibration> parse_calibration(
    const std::string& calib_path,
    const std::pair<int, int>& original_resolution,  // (width, height)
    const std::pair<int, int>& matching_resolution,  // (width, height)
    const at::Device& device
) {
    std::ifstream file(calib_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open calibration file: " + calib_path);
    }

    json calib_data;
    file >> calib_data;
    
    // Get value0 object (raw calibration)
    if (!calib_data.contains("value0") || !calib_data["value0"].is_object()) {
        throw std::runtime_error("Calibration JSON missing 'value0' object");
    }
    
    auto& raw_calib = calib_data["value0"];
    
    // Get T_imu_cam array (extrinsics)
    if (!raw_calib.contains("T_imu_cam") || !raw_calib["T_imu_cam"].is_array()) {
        throw std::runtime_error("Calibration missing 'T_imu_cam' array");
    }
    
    // Get intrinsics array
    if (!raw_calib.contains("intrinsics") || !raw_calib["intrinsics"].is_array()) {
        throw std::runtime_error("Calibration missing 'intrinsics' array");
    }
    
    auto& extrinsics_array = raw_calib["T_imu_cam"];
    auto& intrinsics_array = raw_calib["intrinsics"];
    
    if (extrinsics_array.size() != intrinsics_array.size()) {
        throw std::runtime_error("Mismatch between extrinsics and intrinsics array sizes");
    }
    
    std::vector<Calibration> calibrations;
    
    // Calculate matching scale (applies to all cameras)
    Eigen::Vector2f matching_scale = calculate_matching_scale(original_resolution, matching_resolution);
    
    for (size_t i = 0; i < extrinsics_array.size(); ++i) {
        const auto& extrinsics = extrinsics_array[i];
        const auto& intrinsics = intrinsics_array[i];
        
        // Check camera type
        if (!intrinsics.contains("camera_type") || intrinsics["camera_type"].get<std::string>() != "ds") {
            throw std::runtime_error("Only double sphere camera model is supported");
        }
        
        Calibration calib;
        
        // Parse intrinsics
        const auto& cam_intrinsics = intrinsics["intrinsics"];
        
        calib.fl = {
            cam_intrinsics["fx"].get<float>(),
            cam_intrinsics["fy"].get<float>()
        };
        
        calib.principal = {
            cam_intrinsics["cx"].get<float>(),
            cam_intrinsics["cy"].get<float>()
        };
        
        // Double sphere parameters
        calib.xi = cam_intrinsics["xi"].get<float>();
        calib.alpha = cam_intrinsics["alpha"].get<float>();
        
        // Parse extrinsics (quaternion + translation)
        float qx = extrinsics["qx"].get<float>();
        float qy = extrinsics["qy"].get<float>();
        float qz = extrinsics["qz"].get<float>();
        float qw = extrinsics["qw"].get<float>();
        
        float px = extrinsics["px"].get<float>();
        float py = extrinsics["py"].get<float>();
        float pz = extrinsics["pz"].get<float>();
        
        // Convert quaternion to rotation matrix
        // Quaternion formula: R = I + 2*q_skew*q_skew + 2*w*q_skew
        // where q_skew is the skew-symmetric matrix of [qx, qy, qz]
        float xx = qx * qx;
        float yy = qy * qy;
        float zz = qz * qz;
        float xy = qx * qy;
        float xz = qx * qz;
        float yz = qy * qz;
        float wx = qw * qx;
        float wy = qw * qy;
        float wz = qw * qz;
        
        // Build RT matrix
        calib.rt = torch::zeros({4, 4}, torch::dtype(torch::kFloat32).device(device));
        
        // Rotation part (3x3)
        calib.rt[0][0] = 1.0f - 2.0f * (yy + zz);
        calib.rt[0][1] = 2.0f * (xy - wz);
        calib.rt[0][2] = 2.0f * (xz + wy);
        
        calib.rt[1][0] = 2.0f * (xy + wz);
        calib.rt[1][1] = 1.0f - 2.0f * (xx + zz);
        calib.rt[1][2] = 2.0f * (yz - wx);
        
        calib.rt[2][0] = 2.0f * (xz - wy);
        calib.rt[2][1] = 2.0f * (yz + wx);
        calib.rt[2][2] = 1.0f - 2.0f * (xx + yy);
        
        // Translation part
        calib.rt[0][3] = px;
        calib.rt[1][3] = py;
        calib.rt[2][3] = pz;
        calib.rt[3][3] = 1.0f;
        
        // Set matching scale (convert Eigen::Vector2f to float2)
        calib.matching_scale = make_float2(matching_scale[0], matching_scale[1]);
        
        calibrations.push_back(calib);
    }
    
    std::cout << "Loaded " << calibrations.size() << " camera calibrations" << std::endl;
    std::cout << "Matching scale: [" << matching_scale[0] << ", " << matching_scale[1] << "]" << std::endl;
    
    return calibrations;
}


/**
 * Load and preprocess fisheye images
 * Matches Python: read_input_images() in utils.py
 */
std::vector<at::Tensor> load_images(
    const std::string& dataset_path,
    const std::string& filename,
    const std::vector<int>& camera_indices,  // Indices of cameras to load
    const std::pair<int, int>& target_resolution,  // (width, height)
    const at::Device& device
) {
    std::vector<at::Tensor> images;
    
    for (int cam_idx : camera_indices) {
        std::string image_path = dataset_path + "/cam" + std::to_string(cam_idx) + "/" + filename;
        
        // Load image with OpenCV (BGR format)
        cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            throw std::runtime_error("Failed to load image: " + image_path);
        }
        
        // Convert BGR to RGB
        cv::Mat rgb;
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        
        // Convert to float32 [0-255]
        cv::Mat rgb_float;
        rgb.convertTo(rgb_float, CV_32FC3);
        
        // Convert to torch tensor: HWC -> CHW for PyTorch interpolation
        at::Tensor tensor = torch::from_blob(
            rgb_float.data,
            {rgb_float.rows, rgb_float.cols, 3},
            torch::kFloat32
        ).clone();  // Clone to own the memory
        
        // Move to CUDA and permute to CHW format
        tensor = tensor.to(device).permute({2, 0, 1}).unsqueeze(0);  // [1, 3, H, W]
        
        // Resize using PyTorch's interpolate (use bilinear for better compatibility)
        tensor = torch::nn::functional::interpolate(
            tensor,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{target_resolution.second, target_resolution.first})  // {H, W}
                .mode(torch::kBilinear)
                .align_corners(false)
        );
        
        // Convert back to HWC format and remove batch dimension
        tensor = tensor.squeeze(0).permute({1, 2, 0});  // [H, W, 3]
        
        images.push_back(tensor);
    }
    
    std::cout << "Loaded " << images.size() << " images at resolution ["
              << target_resolution.first << ", " << target_resolution.second << "]" << std::endl;
    
    return images;
}


/**
 * Load mask images for each camera
 */
std::vector<at::Tensor> load_masks(
    const std::string& dataset_path,
    int num_cameras,
    const std::pair<int, int>& target_resolution,  // (width, height)
    const at::Device& device
) {
    std::vector<at::Tensor> masks;
    
    for (int cam_idx = 0; cam_idx < num_cameras; ++cam_idx) {
        std::string mask_path = dataset_path + "/cam" + std::to_string(cam_idx) + "/mask.png";
        
        // Try to load mask
        cv::Mat mask = cv::imread(mask_path, cv::IMREAD_UNCHANGED);
        
        at::Tensor mask_tensor;
        
        if (mask.empty()) {
            // No mask file - create all-ones mask on GPU
            std::cout << "No mask for cam" << cam_idx << ", using full mask" << std::endl;
            mask_tensor = torch::ones({1, target_resolution.second, target_resolution.first}, 
                                     torch::dtype(torch::kFloat32).device(device));
        } else {
            // Convert to float [0, 1]
            cv::Mat mask_float;
            mask.convertTo(mask_float, CV_32FC1, 1.0 / 255.0);
            
            // Convert to torch tensor [H, W]
            at::Tensor mask_cpu = torch::from_blob(
                mask_float.data,
                {mask_float.rows, mask_float.cols},
                torch::kFloat32
            ).clone();
            
            // Move to GPU and add batch/channel dimension -> [1, 1, H, W]
            mask_tensor = mask_cpu.to(device).unsqueeze(0).unsqueeze(0);
            
            // Resize using PyTorch's interpolate
            mask_tensor = torch::nn::functional::interpolate(
                mask_tensor,
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{target_resolution.second, target_resolution.first})
                    .mode(torch::kBilinear)
                    .align_corners(false)
            );
            
            // Remove extra dimension -> [1, H, W]
            mask_tensor = mask_tensor.squeeze(0);
        }
        
        masks.push_back(mask_tensor);
    }
    
    std::cout << "Loaded " << masks.size() << " masks" << std::endl;
    return masks;
}


/**
 * Calculate reprojection viewpoint (center of reference cameras)
 */
at::Tensor calculate_reprojection_viewpoint(
    const std::vector<Calibration>& calibrations,
    const std::vector<int>& references_indices,
    const at::Device& device
) {
    at::Tensor viewpoint = torch::zeros({3}, torch::dtype(torch::kFloat32).device(device));
    
    for (int ref_idx : references_indices) {
        // Extract translation from RT matrix [0:3, 3]
        viewpoint += calibrations[ref_idx].rt.slice(0, 0, 3).slice(1, 3, 4).squeeze();
    }
    
    viewpoint /= static_cast<float>(references_indices.size());
    
    return viewpoint;
}


/**
 * Colorize distance map for visualization
 * Uses inverse distance with MAGMA colormap
 */
cv::Mat colorize_distance_map(
    const at::Tensor& distance,  // [H, W] float32 on CPU
    float min_dist,
    float max_dist
) {
    auto distance_cpu = distance.cpu();
    auto distance_acc = distance_cpu.accessor<float, 2>();
    
    int height = distance_cpu.size(0);
    int width = distance_cpu.size(1);
    
    // Create normalized inverse distance map
    cv::Mat normalized(height, width, CV_8UC1);
    
    float inv_min = 1.0f / max_dist;
    float inv_max = 1.0f / min_dist;
    float inv_range = inv_max - inv_min;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float dist = distance_acc[y][x];
            
            // Convert to inverse distance
            float inv_dist = 1.0f / dist;
            
            // Normalize to [0, 255]
            float normalized_val = (inv_dist - inv_min) / inv_range;
            normalized_val = std::clamp(normalized_val * 255.0f, 0.0f, 255.0f);
            
            normalized.at<uint8_t>(y, x) = static_cast<uint8_t>(normalized_val);
        }
    }
    
    // Apply MAGMA colormap
    cv::Mat colored;
    cv::applyColorMap(normalized, colored, cv::COLORMAP_MAGMA);
    
    return colored;
}


/**
 * Main function
 */
int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Standalone C++ RGBD Estimator" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Parse command line arguments
    std::string dataset_path = "/home/motoken/college/sphere-stereo/resources";
    std::string output_dir = "/home/motoken/college/ros2_ws/output/standalone";
    
    if (argc > 1) {
        dataset_path = argv[1];
    }
    if (argc > 2) {
        output_dir = argv[2];
    }
    
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Dataset: " << dataset_path << std::endl;
    std::cout << "  Output: " << output_dir << std::endl;
    
    // Create output directory
    std::string mkdir_cmd = "mkdir -p " + output_dir;
    int ret = system(mkdir_cmd.c_str());
    (void)ret;  // Suppress unused warning
    
    // Pipeline parameters
    const float min_dist = 0.6f;
    const float max_dist = 10.0f;
    const int candidate_count = 64;
    const std::vector<int> references_indices = {0, 1, 2, 3};
    
    const std::pair<int, int> original_resolution = {1944, 1096};      // (width, height)
    const std::pair<int, int> matching_resolution = {1024, 1024};
    const std::pair<int, int> rgb_to_stitch_resolution = {1216, 1216};
    const std::pair<int, int> panorama_resolution = {2048, 1024};
    
    const float sigma_i = 10.0f;
    const float sigma_s = 25.0f;
    
    const std::string filename = "0.jpg";
    
    at::Device device(at::kCUDA, 0);
    
    std::cout << "  Distance range: [" << min_dist << ", " << max_dist << "]" << std::endl;
    std::cout << "  Candidates: " << candidate_count << std::endl;
    std::cout << "  Original resolution: [" << original_resolution.first << ", " << original_resolution.second << "]" << std::endl;
    std::cout << "  Matching resolution: [" << matching_resolution.first << ", " << matching_resolution.second << "]" << std::endl;
    std::cout << "  Panorama resolution: [" << panorama_resolution.first << ", " << panorama_resolution.second << "]" << std::endl;
    
    try {
        // Step 1: Load calibration
        std::cout << "\n[1/5] Loading calibration..." << std::endl;
        std::string calib_path = dataset_path + "/calibration.json";
        auto calibrations = parse_calibration(calib_path, original_resolution, matching_resolution, device);
        int num_cameras = calibrations.size();
        
        // Step 2: Load images
        std::cout << "\n[2/5] Loading images..." << std::endl;
        
        // Create camera indices vectors
        std::vector<int> all_camera_indices(num_cameras);
        for (int i = 0; i < num_cameras; ++i) {
            all_camera_indices[i] = i;
        }
        
        auto images_to_match = load_images(dataset_path, filename, all_camera_indices, matching_resolution, device);
        auto images_to_stitch = load_images(dataset_path, filename, references_indices, rgb_to_stitch_resolution, device);
        
        // Step 3: Load masks
        std::cout << "\n[3/5] Loading masks..." << std::endl;
        auto masks = load_masks(dataset_path, num_cameras, matching_resolution, device);
        
        // Step 4: Calculate reprojection viewpoint
        std::cout << "\n[4/5] Calculating reprojection viewpoint..." << std::endl;
        auto reprojection_viewpoint = calculate_reprojection_viewpoint(calibrations, references_indices, device);
        std::cout << "  Viewpoint: [" 
                  << reprojection_viewpoint[0].item<float>() << ", "
                  << reprojection_viewpoint[1].item<float>() << ", "
                  << reprojection_viewpoint[2].item<float>() << "]" << std::endl;
        
        // Step 5: Initialize estimator and run
        std::cout << "\n[5/5] Running RGBD estimation..." << std::endl;
        
        RGBDEstimator estimator(
            calibrations,
            min_dist,
            max_dist,
            candidate_count,
            references_indices,
            reprojection_viewpoint,
            masks,
            matching_resolution,
            rgb_to_stitch_resolution,
            panorama_resolution,
            sigma_i,
            sigma_s,
            device
        );
        
        // Warmup
        std::cout << "  Warming up..." << std::endl;
        for (int i = 0; i < 2; ++i) {
            auto [rgb, dist] = estimator.run(images_to_match, images_to_stitch);
        }
        torch::cuda::synchronize();
        
        // Timed run
        std::cout << "  Running timed inference..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        auto [rgb_panorama, distance_panorama] = estimator.run(images_to_match, images_to_stitch);
        
        torch::cuda::synchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "RESULTS" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Execution time: " << elapsed_ms << " ms" << std::endl;
        std::cout << "RGB panorama shape: [" << rgb_panorama.size(0) << ", " << rgb_panorama.size(1) << ", " << rgb_panorama.size(2) << "]" << std::endl;
        std::cout << "Distance panorama shape: [" << distance_panorama.size(0) << ", " << distance_panorama.size(1) << "]" << std::endl;
        
        // Save results
        std::cout << "\nSaving results..." << std::endl;
        
        // Save RGB panorama
        auto rgb_cpu = rgb_panorama.cpu();
        cv::Mat rgb_mat(rgb_cpu.size(0), rgb_cpu.size(1), CV_8UC3, rgb_cpu.data_ptr<uint8_t>());
        cv::Mat rgb_bgr;
        cv::cvtColor(rgb_mat, rgb_bgr, cv::COLOR_RGB2BGR);
        cv::imwrite(output_dir + "/rgb_panorama.png", rgb_bgr);
        std::cout << "  Saved: " << output_dir << "/rgb_panorama.png" << std::endl;
        
        // Save distance panorama (raw)
        auto distance_cpu = distance_panorama.cpu();
        cv::Mat distance_mat(distance_cpu.size(0), distance_cpu.size(1), CV_32FC1, distance_cpu.data_ptr<float>());
        cv::imwrite(output_dir + "/distance_panorama.exr", distance_mat);
        std::cout << "  Saved: " << output_dir << "/distance_panorama.exr" << std::endl;
        
        // Save colorized distance panorama
        cv::Mat distance_colored = colorize_distance_map(distance_panorama, min_dist, max_dist);
        cv::imwrite(output_dir + "/distance_panorama_colored.png", distance_colored);
        std::cout << "  Saved: " << output_dir << "/distance_panorama_colored.png" << std::endl;
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "✓ SUCCESS" << std::endl;
        std::cout << "========================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
