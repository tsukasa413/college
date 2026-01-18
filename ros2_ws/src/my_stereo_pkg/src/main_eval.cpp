/**
=======================================================================
General Information
-------------------
C++ evaluation program for sphere sweeping stereo pipeline
This program validates the C++ implementation against Python reference results
and measures performance metrics.

Verifies:
- Calibration parsing accuracy
- Image loading and tensor conversion
- RGBD estimation correctness (MAE, Max Error, Match Rate)
- GPU execution time

Based on: Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images
Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
CVPR 2021
=======================================================================
**/

#include "my_stereo_pkg/depth_estimation.hpp"
#include "my_stereo_pkg/calibration.hpp"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <cuda_runtime.h>
#include <chrono>
#include <filesystem>
#include <cuda_runtime.h>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace my_stereo_pkg;
using my_stereo::Calibration;

// ========================================================================
// Utility Functions
// ========================================================================

/**
 * Parse calibration.json with exact Python compatibility
 * Matches Python's utils.parse_json_calib behavior
 */
std::vector<Calibration> parse_json_calib(
    const std::string& json_path,
    const std::pair<int, int>& matching_resolution,
    const torch::Device& device,
    const std::pair<int, int>& original_resolution)
{
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open calibration file: " + json_path);
    }

    json calib_json;
    file >> calib_json;
    
    // Navigate to value0 (same as Python)
    auto raw_calibration = calib_json["value0"];
    
    std::vector<Calibration> calibrations;
    auto T_imu_cam = raw_calibration["T_imu_cam"];
    auto intrinsics_array = raw_calibration["intrinsics"];
    
    for (size_t cam_idx = 0; cam_idx < T_imu_cam.size(); ++cam_idx) {
        Calibration calib;
        
        // Parse extrinsics (rotation + translation)
        auto extrinsics = T_imu_cam[cam_idx];
        double qx = extrinsics["qx"];
        double qy = extrinsics["qy"];
        double qz = extrinsics["qz"];
        double qw = extrinsics["qw"];
        double px = extrinsics["px"];
        double py = extrinsics["py"];
        double pz = extrinsics["pz"];
        
        // Convert quaternion to rotation matrix (matches scipy.spatial.transform.Rotation)
        // Quaternion order: [x, y, z, w]
        double x = qx, y = qy, z = qz, w = qw;
        double x2 = x * x, y2 = y * y, z2 = z * z;
        double xy = x * y, xz = x * z, yz = y * z;
        double wx = w * x, wy = w * y, wz = w * z;
        
        // Build rotation matrix
        auto rt = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        auto rt_cpu = rt.cpu();
        
        rt_cpu[0][0] = 1.0 - 2.0 * (y2 + z2);
        rt_cpu[0][1] = 2.0 * (xy - wz);
        rt_cpu[0][2] = 2.0 * (xz + wy);
        
        rt_cpu[1][0] = 2.0 * (xy + wz);
        rt_cpu[1][1] = 1.0 - 2.0 * (x2 + z2);
        rt_cpu[1][2] = 2.0 * (yz - wx);
        
        rt_cpu[2][0] = 2.0 * (xz - wy);
        rt_cpu[2][1] = 2.0 * (yz + wx);
        rt_cpu[2][2] = 1.0 - 2.0 * (x2 + y2);
        
        // Translation
        rt_cpu[0][3] = px;
        rt_cpu[1][3] = py;
        rt_cpu[2][3] = pz;
        
        calib.rt = rt_cpu.to(device);
        
        // Parse intrinsics
        auto intrinsics = intrinsics_array[cam_idx];
        
        // Check camera type (must be double sphere)
        std::string camera_type = intrinsics["camera_type"];
        if (camera_type != "ds") {
            throw std::runtime_error("Unexpected camera model. Only double sphere (ds) is supported.");
        }
        
        auto cam_intrinsics = intrinsics["intrinsics"];
        
        // Extract intrinsic parameters (EXACT order and precision match Python)
        calib.principal.x = cam_intrinsics["cx"];
        calib.principal.y = cam_intrinsics["cy"];
        calib.fl.x = cam_intrinsics["fx"];
        calib.fl.y = cam_intrinsics["fy"];
        calib.xi = cam_intrinsics["xi"];
        calib.alpha = cam_intrinsics["alpha"];
        
        // Calculate matching_scale with EXACT floating point precision
        // matching_scale = [matching_width / original_width, matching_height / original_height]
        // This MUST match Python's calculation exactly
        calib.matching_scale.x = static_cast<float>(matching_resolution.first) / 
                                  static_cast<float>(original_resolution.first);
        calib.matching_scale.y = static_cast<float>(matching_resolution.second) / 
                                  static_cast<float>(original_resolution.second);
        
        calibrations.push_back(calib);
    }
    
    std::cout << "Loaded " << calibrations.size() << " camera calibrations" << std::endl;
    std::cout << "  Matching scale: [" << calibrations[0].matching_scale.x << ", " 
              << calibrations[0].matching_scale.y << "]" << std::endl;
    
    return calibrations;
}

/**
 * Read fisheye images with exact Python compatibility
 * Matches Python's read_input_images behavior
 */
struct ImageLoadResult {
    std::vector<at::Tensor> images_to_match;   // [num_cams][H, W, 3] float32 [0-255]
    std::vector<at::Tensor> images_to_stitch;  // [num_refs][H, W, 3] float32 [0-255]
    bool is_valid;
};

/**
 * Safe conversion from cv::Mat to torch::Tensor
 * Ensures continuous memory layout and proper copying
 */
at::Tensor mat_to_tensor_safe(cv::Mat& mat) {
    // 1. Ensure continuous memory layout (CRITICAL!)
    if (!mat.isContinuous()) {
        std::cout << "      [Debug] Mat not continuous, cloning..." << std::endl;
        mat = mat.clone();
    }
    
    // 2. Allocate Torch memory first
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto tensor = torch::empty({mat.rows, mat.cols, 3}, options);
    
    // 3. Safe memory copy (most reliable method)
    std::memcpy(tensor.data_ptr<float>(), mat.data, mat.total() * mat.elemSize());
    
    return tensor;
}

ImageLoadResult read_input_images(
    const std::string& filename,
    const std::string& dataset_path,
    const std::pair<int, int>& matching_resolution,
    const std::pair<int, int>& rgb_to_stitch_resolution,
    const std::vector<Calibration>& calibrations,
    const std::vector<int>& references_indices,
    const torch::Device& device)
{
    ImageLoadResult result;
    result.is_valid = true;
    
    std::cout << "  [Debug] Loading images for frame: " << filename << std::endl;
    
    for (size_t cam_index = 0; cam_index < calibrations.size(); ++cam_index) {
        // Construct file path: dataset_path/cam{cam_index}/{filename}
        std::string file_path = dataset_path + "/cam" + std::to_string(cam_index) + "/" + filename;
        
        std::cout << "    [Debug] Loading cam" << cam_index << ": " << file_path << std::endl;
        
        // Read image with OpenCV (BGR format by default)
        cv::Mat image = cv::imread(file_path, cv::IMREAD_UNCHANGED);
        
        if (image.empty()) {
            std::cerr << "    [Error] Cannot read image for file " << file_path << std::endl;
            result.is_valid = false;
            continue;
        }
        
        std::cout << "      Image size: " << image.cols << "x" << image.rows 
                  << ", Type: " << image.type() << std::endl;
        
        // Memory-efficient approach: Keep as uint8 until after resize
        // This reduces memory footprint by 4x during resize operations
        // Critical for Jetson's unified memory where LibTorch allocates heavily
        
        // Normalize to uint8 if needed (keep memory footprint small)
        cv::Mat image_u8;
        if (image.type() == CV_8UC3) {
            image_u8 = image;
        } else if (image.type() == CV_16UC3) {
            image.convertTo(image_u8, CV_8UC3, 255.0 / 65535.0);
        } else if (image.type() == CV_32FC3) {
            double min_val, max_val;
            cv::minMaxLoc(image, &min_val, &max_val);
            if (max_val <= 1.0) {
                image.convertTo(image_u8, CV_8UC3, 255.0);
            } else {
                image.convertTo(image_u8, CV_8UC3);
            }
        } else {
            std::cerr << "Warning: Invalid image type for file " << file_path << std::endl;
            result.is_valid = false;
            continue;
        }
        
        // Check which images are reference cameras (need higher resolution)
        bool is_reference = std::find(references_indices.begin(), references_indices.end(), 
                                       cam_index) != references_indices.end();
        
        // WORKAROUND: cv::resize causes segfault on Jetson with LibTorch
        // Solution: Convert to tensor first, then resize on GPU using torch::nn::functional::interpolate
        
        std::cout << "      [Debug] Converting to RGB and float..." << std::endl;
        cv::Mat image_rgb;
        cv::cvtColor(image_u8, image_rgb, cv::COLOR_BGR2RGB);
        cv::Mat image_float;
        image_rgb.convertTo(image_float, CV_32FC3, 1.0);
        
        std::cout << "      [Debug] Converting to tensor (original size: " << image_float.cols << "x" << image_float.rows << ")..." << std::endl;
        auto tensor = mat_to_tensor_safe(image_float);  // [H, W, C]
        
        std::cout << "      [Debug] Moving to GPU..." << std::endl;
        tensor = tensor.to(device);
        
        // Permute to [C, H, W] for torch interpolate
        tensor = tensor.permute({2, 0, 1});  // [C, H, W]
        
        // Add batch dimension [1, C, H, W]
        tensor = tensor.unsqueeze(0);
        
        // Resize on GPU using torch::nn::functional::interpolate
        if (is_reference) {
            std::cout << "      [Debug] Resizing for stitching on GPU (" 
                      << tensor.size(3) << "x" << tensor.size(2) << " -> "
                      << rgb_to_stitch_resolution.first << "x" << rgb_to_stitch_resolution.second << ")..." << std::endl;
            
            auto resized = torch::nn::functional::interpolate(tensor,
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{rgb_to_stitch_resolution.second, rgb_to_stitch_resolution.first})
                    .mode(torch::kBilinear)
                    .align_corners(false));
            
            // Remove batch dimension and permute back to [H, W, C]
            resized = resized.squeeze(0).permute({1, 2, 0});
            result.images_to_stitch.push_back(resized);
        }
        
        // Resize for matching
        std::cout << "      [Debug] Resizing for matching on GPU ("
                  << tensor.size(3) << "x" << tensor.size(2) << " -> "
                  << matching_resolution.first << "x" << matching_resolution.second << ")..." << std::endl;
        
        auto resized_match = torch::nn::functional::interpolate(tensor,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{matching_resolution.second, matching_resolution.first})
                .mode(torch::kBilinear)
                .align_corners(false));
        
        // Remove batch dimension and permute back to [H, W, C]
        resized_match = resized_match.squeeze(0).permute({1, 2, 0});
        result.images_to_match.push_back(resized_match);
        
        std::cout << "      [Debug] Cam" << cam_index << " completed successfully (GPU resize)" << std::endl;
    }  // End of camera loop
    
    return result;
}

// Original resize code - commented out for debugging
/*
void resize_code_disabled_for_debugging() {
        // ORIGINAL RESIZE CODE - DISABLED FOR DEBUGGING
        // Resize for stitching (reference cameras only) - uint8 is 4x lighter
        if (is_reference) {
            std::cout << "      [Debug] Resizing for stitching (uint8: " 
                      << image_u8.cols << "x" << image_u8.rows << " -> "
                      << rgb_to_stitch_resolution.first << "x" << rgb_to_stitch_resolution.second << ")..." << std::endl;
            cv::Mat image_resized_stitch;
            cv::resize(image_u8, image_resized_stitch, 
                      cv::Size(rgb_to_stitch_resolution.first, rgb_to_stitch_resolution.second),
                      0, 0, cv::INTER_LINEAR);
            
            std::cout << "      [Debug] Converting to RGB and float (stitch)..." << std::endl;
            cv::cvtColor(image_resized_stitch, image_resized_stitch, cv::COLOR_BGR2RGB);
            cv::Mat image_stitch_float;
            image_resized_stitch.convertTo(image_stitch_float, CV_32FC3, 1.0);  // 0-255 range
            
            std::cout << "      [Debug] Converting to tensor (stitch)..." << std::endl;
            auto tensor = mat_to_tensor_safe(image_stitch_float);
            
            std::cout << "      [Debug] Moving to GPU (stitch)..." << std::endl;
            result.images_to_stitch.push_back(tensor.to(device));
        }
        
        // Resize for matching - also in uint8 for efficiency
        std::cout << "      [Debug] Resizing for matching (uint8: "
                  << image_u8.cols << "x" << image_u8.rows << " -> "
                  << matching_resolution.first << "x" << matching_resolution.second << ")..." << std::endl;
        cv::Mat image_resized_match;
        cv::resize(image_u8, image_resized_match, 
                  cv::Size(matching_resolution.first, matching_resolution.second),
                  0, 0, cv::INTER_LINEAR);
        
        std::cout << "      [Debug] Converting to RGB and float (match)..." << std::endl;
        cv::cvtColor(image_resized_match, image_resized_match, cv::COLOR_BGR2RGB);
        cv::Mat image_match_float;
        image_resized_match.convertTo(image_match_float, CV_32FC3, 1.0);  // 0-255 range
        
        std::cout << "      [Debug] Converting to tensor (match)..." << std::endl;
        auto tensor_match = mat_to_tensor_safe(image_match_float);
        
        std::cout << "      [Debug] Moving to GPU (match)..." << std::endl;
        result.images_to_match.push_back(tensor_match.to(device));
        
        std::cout << "      [Debug] Cam" << cam_index << " completed successfully" << std::endl;
}
*/

// ========================================================================
// Mask Loading (commented out original complex version)
// ========================================================================

/**
 * Load masks for all cameras
 */
/**
 * Safe mask loading with proper memory management
 */
std::vector<at::Tensor> load_masks(
    const std::string& dataset_path,
    const std::vector<Calibration>& calibrations,
    const std::pair<int, int>& matching_resolution,
    const torch::Device& device)
{
    std::vector<at::Tensor> masks;
    
    std::cout << "  Loading masks for " << calibrations.size() << " cameras..." << std::endl;
    
    // TEMPORARY: Skip actual mask loading to debug image loading
    std::cout << "  [Debug] Using all-ones masks (skipping actual mask files for debugging)" << std::endl;
    for (size_t cam_index = 0; cam_index < calibrations.size(); ++cam_index) {
        auto ones = torch::ones(
            {1, matching_resolution.second, matching_resolution.first},
            torch::TensorOptions().dtype(torch::kFloat32).device(device)
        );
        masks.push_back(ones);
    }
    
    return masks;
}

// Original mask loading code disabled for debugging
/*
std::vector<at::Tensor> load_masks_original(
    const std::string& dataset_path,
    const std::vector<Calibration>& calibrations,
    const std::pair<int, int>& matching_resolution,
    const torch::Device& device)
{
    std::vector<at::Tensor> masks;
    std::cout << "  Loading masks for " << calibrations.size() << " cameras..." << std::endl;
    
    /* DISABLED FOR DEBUGGING
    for (size_t cam_index = 0; cam_index < calibrations.size(); ++cam_index) {
        std::string mask_path = dataset_path + "/cam" + std::to_string(cam_index) + "/mask.png";
        
        std::cout << "    [Debug] Mask for cam" << cam_index << ": " << mask_path;
        
        cv::Mat mask = cv::imread(mask_path, cv::IMREAD_UNCHANGED);
        
        if (mask.empty()) {
            std::cout << " (not found, using ones)" << std::endl;
            auto ones = torch::ones(
                {1, matching_resolution.second, matching_resolution.first},
                torch::TensorOptions().dtype(torch::kFloat32).device(device)
            );
            masks.push_back(ones);
        } else {
            std::cout << " (" << mask.cols << "x" << mask.rows << ", type=" << mask.type() << ")" << std::endl;
            
            try {
                std::cout << "      [Debug] Step 1: Converting to grayscale..." << std::endl;
                // Convert to grayscale if needed
                cv::Mat mask_gray;
                if (mask.channels() == 3) {
                    cv::cvtColor(mask, mask_gray, cv::COLOR_BGR2GRAY);
                } else if (mask.channels() == 4) {
                    cv::cvtColor(mask, mask_gray, cv::COLOR_BGRA2GRAY);
                } else {
                    mask_gray = mask.clone();  // Use clone() to ensure independent memory
                }
                std::cout << "      [Debug] After grayscale: " << mask_gray.cols << "x" << mask_gray.rows 
                          << ", depth=" << mask_gray.depth() << std::endl;
                
                // Convert to uint8 first if needed (CV_16U can cause issues with resize)
                if (mask_gray.depth() == CV_16U) {
                    std::cout << "      [Debug] Converting uint16 to uint8 before resize..." << std::endl;
                    cv::Mat mask_u8;
                    mask_gray.convertTo(mask_u8, CV_8U, 255.0 / 65535.0);
                    mask_gray = mask_u8.clone();  // Ensure fresh memory before resize
                    std::cout << "      [Debug] After uint8 conversion: " << mask_gray.cols << "x" << mask_gray.rows 
                              << ", depth=" << mask_gray.depth() << ", continuous=" << mask_gray.isContinuous() << std::endl;
                }
                
                std::cout << "      [Debug] Step 2: Resizing from " << mask_gray.size() 
                          << " to (" << matching_resolution.first << ", " << matching_resolution.second << ")..." << std::endl;
                // Resize
                cv::Mat mask_resized;
                try {
                    cv::resize(mask_gray, mask_resized,
                              cv::Size(matching_resolution.first, matching_resolution.second),
                              0, 0, cv::INTER_AREA);
                } catch (const cv::Exception& e) {
                    std::cerr << "      [Error] cv::resize failed: " << e.what() << std::endl;
                    throw;
                }
                std::cout << "      [Debug] After resize: " << mask_resized.cols << "x" << mask_resized.rows << std::endl;
                
                std::cout << "      [Debug] Step 3: Converting to float..." << std::endl;
                // Convert to float [0-1]
                cv::Mat mask_float;
                if (mask_resized.depth() == CV_16U) {
                    mask_resized.convertTo(mask_float, CV_32F, 1.0 / 65535.0);
                } else if (mask_resized.depth() == CV_8U) {
                    mask_resized.convertTo(mask_float, CV_32F, 1.0 / 255.0);
                } else {
                    mask_resized.convertTo(mask_float, CV_32F);
                }
                std::cout << "      [Debug] After float conversion, continuous=" << mask_float.isContinuous() << std::endl;
                
                std::cout << "      [Debug] Step 4: Ensuring continuous memory..." << std::endl;
                // Ensure continuous
                if (!mask_float.isContinuous()) {
                    mask_float = mask_float.clone();
                }
                
                std::cout << "      [Debug] Step 5: Creating tensor with memcpy..." << std::endl;
                // Safe tensor creation
                auto options = torch::TensorOptions().dtype(torch::kFloat32);
                auto tensor = torch::empty({mask_float.rows, mask_float.cols}, options);
                std::memcpy(tensor.data_ptr<float>(), mask_float.data, 
                           mask_float.total() * mask_float.elemSize());
                
                std::cout << "      [Debug] Step 6: Adding channel dimension..." << std::endl;
                // Add channel dimension [1, H, W]
                tensor = tensor.unsqueeze(0);
                
                std::cout << "      [Debug] Step 7: Moving to device..." << std::endl;
                masks.push_back(tensor.to(device));
                
                std::cout << "      [Debug] Mask loaded successfully" << std::endl;
                
            } catch (const std::exception& e) {
                std::cerr << "      [Error] " << e.what() << ", using ones" << std::endl;
                auto ones = torch::ones(
                    {1, matching_resolution.second, matching_resolution.first},
                    torch::TensorOptions().dtype(torch::kFloat32).device(device)
                );
                masks.push_back(ones);
            }
        }
    }
    
    return masks;
}
*/

// ========================================================================
// Validation Functions
// ========================================================================

/**
 * Calculate comparison metrics between C++ and Python outputs
 */
struct ComparisonMetrics {
    double rgb_mae;           // Mean Absolute Error for RGB
    double rgb_max_error;     // Maximum error for RGB
    int rgb_max_error_x;      // X coordinate of max error
    int rgb_max_error_y;      // Y coordinate of max error
    double rgb_match_rate;    // Percentage of pixels within threshold
    
    double depth_mae;         // Mean Absolute Error for inverse distance
    double depth_max_error;   // Maximum error for inverse distance
    int depth_max_error_x;    // X coordinate of max error
    int depth_max_error_y;    // Y coordinate of max error
    double depth_match_rate;  // Percentage of pixels within threshold
};

ComparisonMetrics compare_outputs(
    const at::Tensor& cpp_rgb,         // [H, W, 3] uint8
    const at::Tensor& cpp_inv_dist,    // [H, W] float32
    const at::Tensor& py_rgb,          // [H, W, 3] uint8
    const at::Tensor& py_inv_dist,     // [H, W] float32
    float rgb_threshold = 1.0f,        // RGB error threshold
    float depth_threshold = 0.01f)     // Depth error threshold (in inverse distance units)
{
    ComparisonMetrics metrics;
    
    // Move to CPU for analysis
    auto cpp_rgb_cpu = cpp_rgb.to(torch::kCPU);
    auto cpp_inv_dist_cpu = cpp_inv_dist.to(torch::kCPU);
    auto py_rgb_cpu = py_rgb.to(torch::kCPU);
    auto py_inv_dist_cpu = py_inv_dist.to(torch::kCPU);
    
    // RGB comparison
    auto rgb_diff = (cpp_rgb_cpu.to(torch::kFloat32) - py_rgb_cpu.to(torch::kFloat32)).abs();
    auto rgb_diff_max_channel = std::get<0>(rgb_diff.max(/*dim=*/2));  // Max over channels
    
    metrics.rgb_mae = rgb_diff.mean().item<double>();
    metrics.rgb_max_error = rgb_diff_max_channel.max().item<double>();
    
    // Find max error location
    auto max_loc = rgb_diff_max_channel.argmax();
    int w = cpp_rgb_cpu.size(1);
    metrics.rgb_max_error_y = max_loc.item<long>() / w;
    metrics.rgb_max_error_x = max_loc.item<long>() % w;
    
    // Match rate (pixels within threshold)
    auto rgb_within_threshold = (rgb_diff_max_channel <= rgb_threshold);
    metrics.rgb_match_rate = rgb_within_threshold.to(torch::kFloat32).mean().item<double>() * 100.0;
    
    // Depth comparison
    auto depth_diff = (cpp_inv_dist_cpu - py_inv_dist_cpu).abs();
    
    metrics.depth_mae = depth_diff.mean().item<double>();
    metrics.depth_max_error = depth_diff.max().item<double>();
    
    // Find max error location for depth
    max_loc = depth_diff.argmax();
    metrics.depth_max_error_y = max_loc.item<long>() / w;
    metrics.depth_max_error_x = max_loc.item<long>() % w;
    
    // Match rate
    auto depth_within_threshold = (depth_diff <= depth_threshold);
    metrics.depth_match_rate = depth_within_threshold.to(torch::kFloat32).mean().item<double>() * 100.0;
    
    return metrics;
}

/**
 * Save difference heatmaps for visualization
 */
void save_difference_heatmaps(
    const at::Tensor& cpp_rgb,
    const at::Tensor& cpp_inv_dist,
    const at::Tensor& py_rgb,
    const at::Tensor& py_inv_dist,
    const std::string& output_dir,
    const std::string& frame_name)
{
    // Create output directory if needed
    fs::create_directories(output_dir);
    
    // RGB difference heatmap
    auto rgb_diff = (cpp_rgb.to(torch::kFloat32).to(torch::kCPU) - 
                     py_rgb.to(torch::kFloat32).to(torch::kCPU)).abs();
    auto rgb_diff_max_channel = std::get<0>(rgb_diff.max(/*dim=*/2));  // [H, W]
    
    // Normalize to [0, 255]
    auto rgb_diff_norm = (rgb_diff_max_channel / rgb_diff_max_channel.max() * 255.0).to(torch::kUInt8);
    
    // Convert to OpenCV Mat
    cv::Mat rgb_diff_cv(rgb_diff_norm.size(0), rgb_diff_norm.size(1), CV_8UC1, 
                        rgb_diff_norm.data_ptr<uint8_t>());
    cv::Mat rgb_diff_colored;
    cv::applyColorMap(rgb_diff_cv.clone(), rgb_diff_colored, cv::COLORMAP_MAGMA);
    
    // Use simple names if frame_name is empty
    std::string rgb_suffix = frame_name.empty() ? ".png" : ("_" + frame_name + ".png");
    std::string depth_suffix = frame_name.empty() ? ".png" : ("_" + frame_name + ".png");
    
    std::string rgb_diff_path = output_dir + "/diff_rgb" + rgb_suffix;
    cv::imwrite(rgb_diff_path, rgb_diff_colored);
    
    // Depth difference heatmap
    auto depth_diff = (cpp_inv_dist.to(torch::kCPU) - py_inv_dist.to(torch::kCPU)).abs();
    
    // Normalize to [0, 255]
    auto depth_diff_norm = (depth_diff / depth_diff.max() * 255.0).to(torch::kUInt8);
    
    // Convert to OpenCV Mat
    cv::Mat depth_diff_cv(depth_diff_norm.size(0), depth_diff_norm.size(1), CV_8UC1,
                          depth_diff_norm.data_ptr<uint8_t>());
    cv::Mat depth_diff_colored;
    cv::applyColorMap(depth_diff_cv.clone(), depth_diff_colored, cv::COLORMAP_MAGMA);
    
    std::string depth_diff_path = output_dir + "/diff_depth" + depth_suffix;
    cv::imwrite(depth_diff_path, depth_diff_colored);
    
    std::cout << "  Saved difference heatmaps:" << std::endl;
    std::cout << "    RGB:   " << rgb_diff_path << std::endl;
    std::cout << "    Depth: " << depth_diff_path << std::endl;
}

// ========================================================================
// Main Evaluation Program
// ========================================================================

int main(int argc, char** argv) {
    try {
        // ====================================================================
        // Parse Command Line Arguments (match Python argparse)
        // ====================================================================
        
        std::string dataset_path = "evaluation_dataset";
        std::vector<int> references_indices = {0, 2};
        float min_dist = 0.55f;
        float max_dist = 100.0f;
        int candidate_count = 32;
        float sigma_i = 10.0f;
        float sigma_s = 25.0f;
        std::pair<int, int> matching_resolution = {1024, 1024};  // (width, height)
        std::pair<int, int> rgb_to_stitch_resolution = {1216, 1216};
        std::pair<int, int> panorama_resolution = {2048, 1024};
        std::pair<int, int> original_resolution = {1944, 1096};  // Based on actual images
        bool visualize = false;
        
        // Simple argument parsing (can be improved with a library)
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--dataset_path" && i + 1 < argc) {
                dataset_path = argv[++i];
            } else if (arg == "--min_dist" && i + 1 < argc) {
                min_dist = std::stof(argv[++i]);
            } else if (arg == "--max_dist" && i + 1 < argc) {
                max_dist = std::stof(argv[++i]);
            } else if (arg == "--visualize") {
                visualize = true;
            }
        }
        
        std::cout << "========================================" << std::endl;
        std::cout << "Sphere Sweeping Stereo - C++ Evaluation" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Dataset: " << dataset_path << std::endl;
        std::cout << "Distance range: [" << min_dist << ", " << max_dist << "]" << std::endl;
        std::cout << "Candidates: " << candidate_count << std::endl;
        std::cout << "Matching resolution: " << matching_resolution.first << "x" 
                  << matching_resolution.second << std::endl;
        std::cout << "Panorama resolution: " << panorama_resolution.first << "x" 
                  << panorama_resolution.second << std::endl;
        std::cout << std::endl;
        
        // ====================================================================
        // Initialize CUDA Device
        // ====================================================================
        
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available!");
        }
        
        torch::Device device(torch::kCUDA, 0);
        std::cout << "Using device: " << device << std::endl;
        
        // Get CUDA device name using cudaDeviceProp
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "CUDA device: " << prop.name << std::endl;
        std::cout << std::endl;
        
        // ====================================================================
        // Load Calibration (EXACT match to Python parse_json_calib)
        // ====================================================================
        
        std::string calib_path = dataset_path + "/calibration.json";
        std::cout << "Loading calibration from: " << calib_path << std::endl;
        
        auto calibrations = parse_json_calib(
            calib_path,
            matching_resolution,
            device,
            original_resolution
        );
        
        // Calculate reprojection viewpoint (center of reference cameras)
        auto reprojection_viewpoint = torch::zeros({3}, torch::TensorOptions().device(device));
        for (int ref_idx : references_indices) {
            reprojection_viewpoint += calibrations[ref_idx].rt.index({torch::indexing::Slice(0, 3), 3});
        }
        reprojection_viewpoint /= static_cast<float>(references_indices.size());
        
        std::cout << "Reprojection viewpoint: [" 
                  << reprojection_viewpoint[0].item<float>() << ", "
                  << reprojection_viewpoint[1].item<float>() << ", "
                  << reprojection_viewpoint[2].item<float>() << "]" << std::endl;
        std::cout << std::endl;
        
        // ====================================================================
        // Load Masks
        // ====================================================================
        
        std::cout << "Loading masks..." << std::endl;
        auto masks = load_masks(dataset_path, calibrations, matching_resolution, device);
        std::cout << "Loaded " << masks.size() << " masks" << std::endl;
        std::cout << std::endl;
        
        // ====================================================================
        // Initialize RGBD Estimator
        // ====================================================================
        
        std::cout << "Initializing RGBD Estimator..." << std::endl;
        RGBDEstimator rgbd_estimator(
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
        std::cout << "RGBD Estimator initialized" << std::endl;
        std::cout << std::endl;
        
        // ====================================================================
        // Get list of frames to process
        // ====================================================================
        
        std::string cam0_dir = dataset_path + "/cam0/";
        std::vector<std::string> filenames;
        
        for (const auto& entry : fs::directory_iterator(cam0_dir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (filename != "mask.png") {
                    filenames.push_back(filename);
                }
            }
        }
        
        std::sort(filenames.begin(), filenames.end());
        std::cout << "Found " << filenames.size() << " frames to process" << std::endl;
        std::cout << std::endl;
        
        // ====================================================================
        // Setup Output Directory
        // ====================================================================
        
        std::string output_dir = "/home/motoken/college/ros2_ws/output";
        fs::create_directories(output_dir);
        std::cout << "Output directory: " << output_dir << std::endl;
        std::cout << std::endl;
        
        // ====================================================================
        // Load Python Timing Information (if available)
        // ====================================================================
        
        double python_time_ms = 0.0;
        std::string timing_file = dataset_path + "/output/timing.txt";
        if (fs::exists(timing_file)) {
            std::ifstream timing_stream(timing_file);
            timing_stream >> python_time_ms;
            std::cout << "Python reference timing: " << python_time_ms << " ms" << std::endl;
            std::cout << std::endl;
        }
        
        // ====================================================================
        // Process Each Frame and Compare with Python Output
        // ====================================================================
        
        double total_gpu_time_ms = 0.0;
        int valid_frames = 0;
        
        // Accumulate metrics across all frames
        double total_rgb_mae = 0.0;
        double total_rgb_max_error = 0.0;
        double total_rgb_match_rate = 0.0;
        double total_depth_mae = 0.0;
        double total_depth_max_error = 0.0;
        double total_depth_match_rate = 0.0;
        
        for (const auto& filename : filenames) {
            std::cout << "Processing: " << filename << std::endl;
            
            // Load images
            auto images_result = read_input_images(
                filename,
                dataset_path,
                matching_resolution,
                rgb_to_stitch_resolution,
                calibrations,
                references_indices,
                device
            );
            
            if (!images_result.is_valid) {
                std::cerr << "  Skipping invalid frame" << std::endl;
                continue;
            }
            
            // ================================================================
            // Run C++ RGBD Estimation with GPU Timing
            // ================================================================
            
            std::cout << "  [Debug] Creating CUDA events for timing..." << std::endl;
            
            // Create CUDA events for timing
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            // Ensure all previous operations are complete before timing
            cudaDeviceSynchronize();
            
            std::cout << "  [Debug] Recording start event..." << std::endl;
            // Record start event
            cudaEventRecord(start);
            
            std::cout << "  [Debug] Running RGBD estimation..." << std::endl;
            // Run estimation
            auto [cpp_rgb, cpp_distance] = rgbd_estimator.run(
                images_result.images_to_match,
                images_result.images_to_stitch
            );
            
            std::cout << "  [Debug] Recording stop event..." << std::endl;
            // Record stop event and synchronize
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            std::cout << "  [Debug] Computing elapsed time..." << std::endl;
            // Calculate elapsed time
            float gpu_time_ms;
            cudaEventElapsedTime(&gpu_time_ms, start, stop);
            
            // Cleanup events
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            
            total_gpu_time_ms += gpu_time_ms;
            
            std::cout << "  C++ execution time: " << gpu_time_ms << " ms" << std::endl;
            
            // Calculate inverse distance (Python saves inverse distance)
            auto cpp_inv_distance = 1.0f / cpp_distance;
            
            // ================================================================
            // Load Python Reference Output
            // ================================================================
            
            std::string frame_name = filename.substr(0, filename.find_last_of('.'));
            std::string py_rgb_path = dataset_path + "/output/rgb_" + frame_name + ".png";
            std::string py_inv_dist_path = dataset_path + "/output/inv_distance_" + frame_name + ".png";
            
            // Check if Python output exists
            if (!fs::exists(py_rgb_path) || !fs::exists(py_inv_dist_path)) {
                std::cerr << "  Warning: Python reference output not found, skipping comparison" 
                          << std::endl;
                continue;
            }
            
            // Load Python RGB output
            cv::Mat py_rgb_cv = cv::imread(py_rgb_path, cv::IMREAD_UNCHANGED);
            if (py_rgb_cv.empty()) {
                std::cerr << "  Error: Failed to load Python RGB output" << std::endl;
                continue;
            }
            
            // Convert BGR to RGB and to tensor
            cv::cvtColor(py_rgb_cv, py_rgb_cv, cv::COLOR_BGR2RGB);
            auto py_rgb = torch::from_blob(
                py_rgb_cv.data,
                {py_rgb_cv.rows, py_rgb_cv.cols, 3},
                torch::kUInt8
            ).clone().to(device);
            
            // Load Python inverse distance output
            cv::Mat py_inv_dist_cv = cv::imread(py_inv_dist_path, cv::IMREAD_UNCHANGED);
            if (py_inv_dist_cv.empty()) {
                std::cerr << "  Error: Failed to load Python inverse distance output" << std::endl;
                continue;
            }
            
            // Python saves normalized PNG [0-255], convert back to float [0-1] range
            py_inv_dist_cv.convertTo(py_inv_dist_cv, CV_32F, 1.0 / 255.0);
            auto py_inv_dist = torch::from_blob(
                py_inv_dist_cv.data,
                {py_inv_dist_cv.rows, py_inv_dist_cv.cols},
                torch::kFloat32
            ).clone().to(device);
            
            // ================================================================
            // Compare C++ and Python Outputs (using normalized values)
            // ================================================================
            
            // Normalize C++ inverse distance to [0-1] for fair comparison
            auto cpp_inv_dist_normalized = (cpp_inv_distance - cpp_inv_distance.min()) / 
                                          (cpp_inv_distance.max() - cpp_inv_distance.min());
            
            auto metrics = compare_outputs(cpp_rgb, cpp_inv_dist_normalized, py_rgb, py_inv_dist);
            
            std::cout << "  RGB Comparison:" << std::endl;
            std::cout << "    MAE:        " << metrics.rgb_mae << std::endl;
            std::cout << "    Max Error:  " << metrics.rgb_max_error 
                      << " at (" << metrics.rgb_max_error_x << ", " 
                      << metrics.rgb_max_error_y << ")" << std::endl;
            std::cout << "    Match Rate: " << metrics.rgb_match_rate << "% (threshold=±1)" 
                      << std::endl;
            
            std::cout << "  Depth Comparison:" << std::endl;
            std::cout << "    MAE:        " << metrics.depth_mae << std::endl;
            std::cout << "    Max Error:  " << metrics.depth_max_error 
                      << " at (" << metrics.depth_max_error_x << ", " 
                      << metrics.depth_max_error_y << ")" << std::endl;
            std::cout << "    Match Rate: " << metrics.depth_match_rate 
                      << "% (threshold=0.01)" << std::endl;
            
            // Accumulate metrics
            total_rgb_mae += metrics.rgb_mae;
            total_rgb_max_error += metrics.rgb_max_error;
            total_rgb_match_rate += metrics.rgb_match_rate;
            total_depth_mae += metrics.depth_mae;
            total_depth_max_error += metrics.depth_max_error;
            total_depth_match_rate += metrics.depth_match_rate;
            
            // ================================================================
            // Save Outputs to ros2_ws/output/ directory
            // ================================================================
            
            // Save C++ RGB output
            auto cpp_rgb_u8 = cpp_rgb.to(torch::kCPU).to(torch::kUInt8);
            cv::Mat cpp_rgb_cv(cpp_rgb_u8.size(0), cpp_rgb_u8.size(1), CV_8UC3,
                              cpp_rgb_u8.data_ptr<uint8_t>());
            cv::Mat cpp_rgb_bgr;
            cv::cvtColor(cpp_rgb_cv.clone(), cpp_rgb_bgr, cv::COLOR_RGB2BGR);
            cv::imwrite(output_dir + "/cpp_rgb.png", cpp_rgb_bgr);
            
            // Save C++ inverse distance output (MATCH PYTHON EXACTLY!)
            // Python saves: inv_distance (1/distance) as normalized PNG [0-255]
            auto cpp_inv_dist_cpu = cpp_inv_distance.to(torch::kCPU);
            cv::Mat cpp_inv_dist_cv(cpp_inv_dist_cpu.size(0), cpp_inv_dist_cpu.size(1), CV_32F,
                                   cpp_inv_dist_cpu.data_ptr<float>());
            
            // Normalize inverse distance to 0-255 and apply MAGMA colormap
            cv::Mat cpp_inv_dist_png;
            double cpp_inv_min, cpp_inv_max;
            cv::minMaxLoc(cpp_inv_dist_cv, &cpp_inv_min, &cpp_inv_max);
            cpp_inv_dist_cv.convertTo(cpp_inv_dist_png, CV_8UC1, 
                                      255.0 / (cpp_inv_max - cpp_inv_min), 
                                      -cpp_inv_min * 255.0 / (cpp_inv_max - cpp_inv_min));
            cv::Mat cpp_inv_dist_colored;
            cv::applyColorMap(cpp_inv_dist_png, cpp_inv_dist_colored, cv::COLORMAP_MAGMA);
            cv::imwrite(output_dir + "/cpp_distance.png", cpp_inv_dist_colored);
            
            // Also save as TIFF for precise analysis (debugging)
            cv::imwrite(output_dir + "/cpp_inv_distance.tif", cpp_inv_dist_cv.clone());
            
            // Copy Python reference images to output directory
            if (valid_frames == 0) {  // Only copy once
                // Convert Python RGB to BGR for correct OpenCV saving
                cv::Mat py_rgb_bgr;
                cv::cvtColor(py_rgb_cv.clone(), py_rgb_bgr, cv::COLOR_RGB2BGR);
                cv::imwrite(output_dir + "/python_rgb.png", py_rgb_bgr);
                
                // Python also saves inverse distance as normalized PNG
                auto py_inv_dist_cpu = py_inv_dist.to(torch::kCPU);
                cv::Mat py_inv_dist_cv(py_inv_dist_cpu.size(0), py_inv_dist_cpu.size(1), CV_32F,
                                      py_inv_dist_cpu.data_ptr<float>());
                
                // Normalize Python inverse distance to 0-255 and apply MAGMA colormap
                cv::Mat py_inv_dist_png;
                double py_inv_min, py_inv_max;
                cv::minMaxLoc(py_inv_dist_cv, &py_inv_min, &py_inv_max);
                py_inv_dist_cv.convertTo(py_inv_dist_png, CV_8UC1, 
                                         255.0 / (py_inv_max - py_inv_min), 
                                         -py_inv_min * 255.0 / (py_inv_max - py_inv_min));
                cv::Mat py_inv_dist_colored;
                cv::applyColorMap(py_inv_dist_png, py_inv_dist_colored, cv::COLORMAP_MAGMA);
                cv::imwrite(output_dir + "/python_distance.png", py_inv_dist_colored);
                
                // Also save Python TIFF for debugging
                cv::imwrite(output_dir + "/python_inv_distance.tif", py_inv_dist_cv.clone());
            }
            
            // Save difference heatmaps with new naming
            save_difference_heatmaps(
                cpp_rgb, cpp_inv_distance,
                py_rgb, py_inv_dist,
                output_dir,
                ""  // Empty frame name to use diff_rgb.png, diff_depth.png
            );
            
            valid_frames++;
            std::cout << std::endl;
        }
        
        // ====================================================================
        // Print Final Summary in Requested Format
        // ====================================================================
        
        std::cout << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        
        if (valid_frames > 0) {
            double avg_cpp_time = total_gpu_time_ms / valid_frames;
            double speedup = (python_time_ms > 0.0) ? (python_time_ms / avg_cpp_time) : 0.0;
            
            std::cout << "[Performance Comparison]" << std::endl;
            if (python_time_ms > 0.0) {
                std::cout << "- Python Execution Time:  " << std::fixed << std::setprecision(1) 
                          << python_time_ms << " ms" << std::endl;
            } else {
                std::cout << "- Python Execution Time:  Not available (run with timing.txt)" << std::endl;
            }
            std::cout << "- C++ Execution Time:     " << std::fixed << std::setprecision(1) 
                      << avg_cpp_time << " ms" << std::endl;
            if (speedup > 0.0) {
                std::cout << "- Speedup:                " << std::fixed << std::setprecision(1) 
                          << speedup << " x (C++ is " << speedup << " times faster)" << std::endl;
            }
            std::cout << std::endl;
            
            std::cout << "[Accuracy Comparison]" << std::endl;
            std::cout << "- RGB Mean Absolute Error:   " << std::fixed << std::setprecision(1) 
                      << (total_rgb_mae / valid_frames) << std::endl;
            std::cout << "- Depth Mean Absolute Error: " << std::fixed << std::setprecision(4) 
                      << (total_depth_mae / valid_frames) << std::endl;
            std::cout << "- Match Rate (Tolerance):    " << std::fixed << std::setprecision(1) 
                      << (total_rgb_match_rate / valid_frames) << " %" << std::endl;
            std::cout << std::endl;
            
            std::cout << "[Output Files]" << std::endl;
            std::cout << "- Python RGB Reference:   " << output_dir << "/python_rgb.png" << std::endl;
            std::cout << "- Python Depth Reference: " << output_dir << "/python_distance.png" << std::endl;
            std::cout << "- C++ RGB Output:         " << output_dir << "/cpp_rgb.png" << std::endl;
            std::cout << "- C++ Depth Output:       " << output_dir << "/cpp_distance.png" << std::endl;
            std::cout << "- RGB Difference:         " << output_dir << "/diff_rgb.png" << std::endl;
            std::cout << "- Depth Difference:       " << output_dir << "/diff_depth.png" << std::endl;
        }
        
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << std::endl;
        std::cout << "Evaluation complete!" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
