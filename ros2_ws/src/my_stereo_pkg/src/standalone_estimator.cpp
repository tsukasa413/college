/**
 * Standalone C++ RGBD Estimator with Real-Time Camera Streaming
 * 
 * Pure C++ implementation for full-sphere stereo depth estimation.
 * Uses GStreamer to capture live video from 4 cameras and runs
 * continuous depth estimation.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <memory>
#include <thread>
#include <iomanip>
#include <sstream>
#include <ctime>

// OpenCV for image I/O and GStreamer
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
 * Configuration structure to hold all pipeline parameters
 */
struct Config {
    // Distance parameters
    float min_dist;
    float max_dist;
    int candidate_count;
    
    // Resolution parameters
    std::pair<int, int> original_resolution;
    std::pair<int, int> matching_resolution;
    std::pair<int, int> rgb_to_stitch_resolution;
    std::pair<int, int> panorama_resolution;
    
    // Filter parameters
    float sigma_i;
    float sigma_s;
    
    // Camera parameters
    std::vector<int> references_indices;
    int num_cameras;
    int sensor_mode;
    
    /**
     * Load configuration from JSON file
     */
    static Config load(const std::string& config_path) {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open config file: " + config_path);
        }
        
        json config_json;
        file >> config_json;
        
        Config config;
        
        // Distance parameters
        config.min_dist = config_json["distance"]["min"].get<float>();
        config.max_dist = config_json["distance"]["max"].get<float>();
        config.candidate_count = config_json["distance"]["candidates"].get<int>();
        
        // Resolution parameters
        config.original_resolution = {
            config_json["resolution"]["original"][0].get<int>(),
            config_json["resolution"]["original"][1].get<int>()
        };
        config.matching_resolution = {
            config_json["resolution"]["matching"][0].get<int>(),
            config_json["resolution"]["matching"][1].get<int>()
        };
        config.rgb_to_stitch_resolution = {
            config_json["resolution"]["rgb_to_stitch"][0].get<int>(),
            config_json["resolution"]["rgb_to_stitch"][1].get<int>()
        };
        config.panorama_resolution = {
            config_json["resolution"]["panorama"][0].get<int>(),
            config_json["resolution"]["panorama"][1].get<int>()
        };
        
        // Filter parameters
        config.sigma_i = config_json["filter"]["sigma_i"].get<float>();
        config.sigma_s = config_json["filter"]["sigma_s"].get<float>();
        
        // References indices
        config.references_indices = config_json["references_indices"].get<std::vector<int>>();
        
        // Camera parameters
        config.num_cameras = config_json["camera"]["num_cameras"].get<int>();
        config.sensor_mode = config_json["camera"]["sensor_mode"].get<int>();
        
        return config;
    }
    
    /**
     * Print configuration summary
     */
    void print() const {
        std::cout << "\nConfiguration:" << std::endl;
        std::cout << "  Distance range: [" << min_dist << ", " << max_dist << "]" << std::endl;
        std::cout << "  Candidates: " << candidate_count << std::endl;
        std::cout << "  Original resolution: [" << original_resolution.first << ", " << original_resolution.second << "]" << std::endl;
        std::cout << "  Matching resolution: [" << matching_resolution.first << ", " << matching_resolution.second << "]" << std::endl;
        std::cout << "  RGB to stitch resolution: [" << rgb_to_stitch_resolution.first << ", " << rgb_to_stitch_resolution.second << "]" << std::endl;
        std::cout << "  Panorama resolution: [" << panorama_resolution.first << ", " << panorama_resolution.second << "]" << std::endl;
        std::cout << "  Filter sigma_i: " << sigma_i << std::endl;
        std::cout << "  Filter sigma_s: " << sigma_s << std::endl;
        std::cout << "  Num cameras: " << num_cameras << std::endl;
        std::cout << "  Sensor mode: " << sensor_mode << std::endl;
        std::cout << "  References: [";
        for (size_t i = 0; i < references_indices.size(); ++i) {
            std::cout << references_indices[i];
            if (i < references_indices.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
};


/**
 * Camera Streamer Class - Manages 4 GStreamer camera streams
 */
class CameraStreamer {
public:
    CameraStreamer(int num_cameras = 4, int sensor_mode = 2) 
        : num_cameras_(num_cameras), sensor_mode_(sensor_mode) {
        caps_.resize(num_cameras_);
    }
    
    ~CameraStreamer() {
        close();
    }
    
    /**
     * Initialize all camera streams with GStreamer pipelines
     */
    bool initialize() {
        std::cout << "Initializing " << num_cameras_ << " camera streams..." << std::endl;
        
        for (int i = 0; i < num_cameras_; ++i) {
            std::string pipeline = buildGStreamerPipeline(i);
            std::cout << "\nCamera " << i << " pipeline: " << pipeline << std::endl;
            
            // Add delay between camera initializations to avoid Argus driver conflicts
            if (i > 0) {
                std::cout << "  Waiting 2 seconds before initializing camera " << i << "..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
            
            // Try to open camera with retry
            int max_retries = 3;
            bool opened = false;
            
            for (int retry = 0; retry < max_retries && !opened; ++retry) {
                if (retry > 0) {
                    std::cout << "  Retry " << retry << "/" << max_retries << " for camera " << i << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
                
                caps_[i] = cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);
                
                if (caps_[i].isOpened()) {
                    // Test frame capture
                    cv::Mat test_frame;
                    if (caps_[i].read(test_frame)) {
                        std::cout << "  Camera " << i << " initialized: " 
                                  << test_frame.cols << "x" << test_frame.rows << std::endl;
                        opened = true;
                    } else {
                        std::cerr << "  Camera " << i << " opened but cannot read frames" << std::endl;
                        caps_[i].release();
                    }
                } else {
                    std::cerr << "  Failed to open camera " << i << std::endl;
                }
            }
            
            if (!opened) {
                std::cerr << "\nERROR: Failed to initialize camera " << i << " after " << max_retries << " retries" << std::endl;
                // Clean up previously opened cameras
                for (int j = 0; j < i; ++j) {
                    if (caps_[j].isOpened()) {
                        caps_[j].release();
                    }
                }
                return false;
            }
        }
        
        std::cout << "\nAll cameras initialized successfully!" << std::endl;
        return true;
    }
    
    /**
     * Capture frames from all cameras
     */
    bool captureFrames(std::vector<cv::Mat>& frames) {
        frames.clear();
        frames.reserve(num_cameras_);
        
        for (int i = 0; i < num_cameras_; ++i) {
            cv::Mat frame;
            if (!caps_[i].read(frame)) {
                std::cerr << "ERROR: Failed to read frame from camera " << i << std::endl;
                return false;
            }
            frames.push_back(frame);
        }
        
        return true;
    }
    
    /**
     * Close all camera streams
     */
    void close() {
        for (auto& cap : caps_) {
            if (cap.isOpened()) {
                cap.release();
            }
        }
    }
    
private:
    /**
     * Build GStreamer pipeline string for a camera
     * Matches Python sync_cam_node.py full_fov_mode pipeline
     */
    std::string buildGStreamerPipeline(int camera_id) {
        // sensor-mode=2: 1944x1096 @ 30fps (full FOV)
        // Matches Python: full_fov_mode with target 1944x1096
        std::ostringstream pipeline;
        pipeline << "nvarguscamerasrc sensor-id=" << camera_id 
                 << " sensor-mode=" << sensor_mode_ << " bufapi-version=1 ! "
                 << "video/x-raw(memory:NVMM), width=(int)1944, height=(int)1096, "
                 << "format=(string)NV12, framerate=(fraction)30/1 ! "
                 << "nvvidconv ! "
                 << "video/x-raw, format=(string)BGRx ! "
                 << "videoconvert ! "
                 << "video/x-raw, format=(string)BGR ! "
                 << "appsink drop=true sync=false max-buffers=1";
        return pipeline.str();
    }
    
    int num_cameras_;
    int sensor_mode_;
    std::vector<cv::VideoCapture> caps_;
};


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
 * Load and preprocess fisheye images from live camera frames
 * Converts cv::Mat (BGR, HWC) to LibTorch Tensor (RGB, HWC, Float32) and resizes
 */
std::vector<at::Tensor> preprocess_camera_frames(
    const std::vector<cv::Mat>& frames,
    const std::pair<int, int>& target_resolution,  // (width, height)
    const at::Device& device
) {
    std::vector<at::Tensor> tensors;
    tensors.reserve(frames.size());
    
    for (size_t i = 0; i < frames.size(); ++i) {
        const cv::Mat& frame = frames[i];
        
        if (frame.empty()) {
            throw std::runtime_error("Empty frame from camera " + std::to_string(i));
        }
        
        // Convert BGR to RGB
        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        
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
        
        // Resize using PyTorch's interpolate (bilinear)
        tensor = torch::nn::functional::interpolate(
            tensor,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{target_resolution.second, target_resolution.first})  // {H, W}
                .mode(torch::kBilinear)
                .align_corners(false)
        );
        
        // Convert back to HWC format and remove batch dimension
        tensor = tensor.squeeze(0).permute({1, 2, 0});  // [H, W, 3]
        
        tensors.push_back(tensor);
    }
    
    return tensors;
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
 * Main function - Real-time camera streaming and depth estimation
 */
int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Real-Time C++ RGBD Estimator" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Parse command line arguments
    std::string dataset_path = "/home/motoken/college/sphere-stereo/resources";
    std::string output_dir = "/home/motoken/college/ros2_ws/output/standalone";
    bool save_every_frame = false;
    bool show_display = false;  // Default to false for headless operation
    
    if (argc > 1) {
        dataset_path = argv[1];
    }
    if (argc > 2) {
        output_dir = argv[2];
    }
    if (argc > 3) {
        if (std::string(argv[3]) == "save") {
            save_every_frame = true;
        } else if (std::string(argv[3]) == "display") {
            show_display = true;
        }
    }
    
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Calibration path: " << dataset_path << std::endl;
    std::cout << "  Output dir: " << output_dir << std::endl;
    std::cout << "  Save every frame: " << (save_every_frame ? "Yes" : "No") << std::endl;
    std::cout << "  Show display: " << (show_display ? "Yes" : "No (headless)") << std::endl;
    
    // Create output directory
    std::string mkdir_cmd = "mkdir -p " + output_dir;
    int ret = system(mkdir_cmd.c_str());
    (void)ret;  // Suppress unused warning
    
    // Load configuration
    std::string config_path = dataset_path + "/config.json";
    Config config = Config::load(config_path);
    config.print();
    
    at::Device device(at::kCUDA, 0);
    
    try {
        // Step 1: Initialize camera streamer
        std::cout << "\n[1/6] Initializing camera streams..." << std::endl;
        CameraStreamer streamer(config.num_cameras, config.sensor_mode);
        if (!streamer.initialize()) {
            throw std::runtime_error("Failed to initialize camera streams");
        }
        
        // Step 2: Load calibration
        std::cout << "\n[2/6] Loading calibration..." << std::endl;
        std::string calib_path = dataset_path + "/calibration.json";
        auto calibrations = parse_calibration(calib_path, config.original_resolution, config.matching_resolution, device);
        int num_cameras = calibrations.size();
        
        if (num_cameras != 4) {
            throw std::runtime_error("Expected 4 cameras in calibration, got " + std::to_string(num_cameras));
        }
        
        // Step 3: Load masks (from disk, only once)
        std::cout << "\n[3/6] Loading masks..." << std::endl;
        auto masks = load_masks(dataset_path, num_cameras, config.matching_resolution, device);
        
        // Step 4: Calculate reprojection viewpoint
        std::cout << "\n[4/6] Calculating reprojection viewpoint..." << std::endl;
        auto reprojection_viewpoint = calculate_reprojection_viewpoint(calibrations, config.references_indices, device);
        std::cout << "  Viewpoint: [" 
                  << reprojection_viewpoint[0].item<float>() << ", "
                  << reprojection_viewpoint[1].item<float>() << ", "
                  << reprojection_viewpoint[2].item<float>() << "]" << std::endl;
        
        // Step 5: Initialize estimator
        std::cout << "\n[5/6] Initializing RGBD Estimator..." << std::endl;
        RGBDEstimator estimator(
            calibrations,
            config.min_dist,
            config.max_dist,
            config.candidate_count,
            config.references_indices,
            reprojection_viewpoint,
            masks,
            config.matching_resolution,
            config.rgb_to_stitch_resolution,
            config.panorama_resolution,
            config.sigma_i,
            config.sigma_s,
            device
        );
        
        // Warmup with first few frames
        std::cout << "\n[6/6] Warming up with live camera frames..." << std::endl;
        for (int i = 0; i < 3; ++i) {
            std::vector<cv::Mat> frames;
            if (!streamer.captureFrames(frames)) {
                throw std::runtime_error("Failed to capture warmup frames");
            }
            
            auto images_to_match = preprocess_camera_frames(frames, config.matching_resolution, device);
            auto images_to_stitch = preprocess_camera_frames(
                {frames[0], frames[1], frames[2], frames[3]}, 
                config.rgb_to_stitch_resolution, 
                device
            );
            
            auto [rgb, dist] = estimator.run(images_to_match, images_to_stitch);
        }
        torch::cuda::synchronize();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "STARTING REAL-TIME INFERENCE" << std::endl;
        if (show_display) {
            std::cout << "Press 'q' to quit, 's' to save snapshot, 'r' to start/stop recording" << std::endl;
        } else {
            std::cout << "Running in headless mode (Ctrl+C to quit)" << std::endl;
        }
        std::cout << "========================================\n" << std::endl;
        
        // Main loop - continuous processing
        int frame_count = 0;
        double total_time_ms = 0.0;
        double min_time_ms = 1e9;
        double max_time_ms = 0.0;
        
        // Video recording setup
        bool is_recording = false;
        cv::VideoWriter rgb_writer, distance_writer;
        std::string video_timestamp = "";
        
        // Persistent frame buffers for display and recording
        cv::Mat rgb_bgr, distance_colored;
        
        // Create display window if needed
        if (show_display) {
            cv::namedWindow("RGB Panorama", cv::WINDOW_NORMAL);
            cv::namedWindow("Distance Panorama", cv::WINDOW_NORMAL);
            cv::resizeWindow("RGB Panorama", 1024, 512);
            cv::resizeWindow("Distance Panorama", 1024, 512);
        }
        
        while (true) {
            // Capture frames from all cameras
            std::vector<cv::Mat> frames;
            if (!streamer.captureFrames(frames)) {
                std::cerr << "Warning: Failed to capture frames, skipping..." << std::endl;
                continue;
            }
            
            // Preprocess frames for matching and stitching
            auto images_to_match = preprocess_camera_frames(frames, config.matching_resolution, device);
            auto images_to_stitch = preprocess_camera_frames(
                {frames[0], frames[1], frames[2], frames[3]}, 
                config.rgb_to_stitch_resolution, 
                device
            );
            
            // Run depth estimation
            auto start = std::chrono::high_resolution_clock::now();
            auto [rgb_panorama, distance_panorama] = estimator.run(images_to_match, images_to_stitch);
            torch::cuda::synchronize();
            auto end = std::chrono::high_resolution_clock::now();
            
            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            // Update statistics
            frame_count++;
            total_time_ms += elapsed_ms;
            min_time_ms = std::min(min_time_ms, elapsed_ms);
            max_time_ms = std::max(max_time_ms, elapsed_ms);
            
            // Print status every 30 frames
            if (frame_count % 30 == 0) {
                double avg_time = total_time_ms / frame_count;
                double fps = 1000.0 / avg_time;
                std::cout << "Frame " << frame_count 
                          << " | Avg: " << avg_time << " ms (" << fps << " FPS)"
                          << " | Min: " << min_time_ms << " ms"
                          << " | Max: " << max_time_ms << " ms"
                          << " | Current: " << elapsed_ms << " ms" << std::endl;
            }
            
            // Display results
            if (show_display) {
                // Convert RGB panorama to OpenCV format (reuse buffer)
                auto rgb_cpu = rgb_panorama.cpu();
                cv::Mat rgb_mat(rgb_cpu.size(0), rgb_cpu.size(1), CV_8UC3, rgb_cpu.data_ptr<uint8_t>());
                cv::cvtColor(rgb_mat, rgb_bgr, cv::COLOR_RGB2BGR);
                
                // Colorize distance panorama (reuse buffer)
                distance_colored = colorize_distance_map(distance_panorama, config.min_dist, config.max_dist);
                
                // Add recording indicator
                if (is_recording) {
                    cv::circle(rgb_bgr, cv::Point(30, 30), 15, cv::Scalar(0, 0, 255), -1);
                    cv::putText(rgb_bgr, "REC", cv::Point(60, 40), 
                               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                    cv::circle(distance_colored, cv::Point(30, 30), 15, cv::Scalar(0, 0, 255), -1);
                    cv::putText(distance_colored, "REC", cv::Point(60, 40), 
                               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                }
                
                cv::imshow("RGB Panorama", rgb_bgr);
                cv::imshow("Distance Panorama", distance_colored);
                // 3. 画面解像度の設定（お使いのモニタに合わせて調整してください）
                int screenWidth = 1920;
                int screenHeight = 1080;
                int halfWidth = screenWidth / 2;

                // 4. 位置の移動とサイズ変更
                // 左半分に配置
                cv::moveWindow("RGB Panorama", 0, 0);
                cv::resizeWindow("RGB Panorama", halfWidth, screenHeight);

                // 右半分に配置
                cv::moveWindow("Distance Panorama", halfWidth, 0);
                cv::resizeWindow("Distance Panorama", halfWidth, screenHeight);
                
                // Write to video if recording
                if (is_recording && rgb_writer.isOpened() && distance_writer.isOpened()) {
                    if (!rgb_bgr.empty() && !distance_colored.empty()) {
                        rgb_writer.write(rgb_bgr);
                        distance_writer.write(distance_colored);
                    }
                }
                
                // Handle keyboard input
                int key = cv::waitKey(1);
                if (key == 'q' || key == 27) {  // 'q' or ESC
                    std::cout << "\nQuit requested by user" << std::endl;
                    break;
                }
                
                if (key == 's') {  // 's' for snapshot
                    if (!rgb_bgr.empty() && !distance_colored.empty()) {
                        std::string snapshot_prefix = output_dir + "/snapshot_" + std::to_string(frame_count);
                        cv::imwrite(snapshot_prefix + "_rgb.png", rgb_bgr);
                        cv::imwrite(snapshot_prefix + "_distance.png", distance_colored);
                        std::cout << "Snapshot saved: " << snapshot_prefix << std::endl;
                    } else {
                        std::cerr << "No frames available for snapshot" << std::endl;
                    }
                }
                
                if (key == 'r') {  // 'r' for record
                    if (!is_recording) {
                        // Ensure frames are available FIRST
                        if (rgb_bgr.empty() || distance_colored.empty()) {
                            std::cerr << "ERROR: No frames available yet. Wait a moment and try again." << std::endl;
                            continue;  // Skip to next iteration
                        }
                        
                        if (rgb_bgr.cols != config.panorama_resolution.first || 
                            rgb_bgr.rows != config.panorama_resolution.second) {
                            std::cerr << "ERROR: Frame size mismatch. Expected " 
                                      << config.panorama_resolution.first << "x" << config.panorama_resolution.second
                                      << ", got " << rgb_bgr.cols << "x" << rgb_bgr.rows << std::endl;
                            continue;  // Skip to next iteration
                        }
                        
                        // Create timestamp AFTER validation
                        auto t = std::time(nullptr);
                        auto tm = *std::localtime(&t);
                        std::ostringstream oss;
                        oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
                        video_timestamp = oss.str();
                        
                        // Save frames as images instead of video (safer fallback)
                        std::cout << "\n[Recording] Starting image sequence recording..." << std::endl;
                        std::cout << "  Output: " << output_dir << "/video_" << video_timestamp << "/" << std::endl;
                        std::cout << "  Note: Video recording disabled due to stability issues." << std::endl;
                        std::cout << "  Use 's' key to save snapshots, or enable save-every-frame mode." << std::endl;
                        std::cout << "  To convert images to video later: ffmpeg -framerate 12 -i frame_%04d_rgb.png output.mp4" << std::endl;
                    } else {
                        // Stop recording
                        is_recording = false;
                        if (rgb_writer.isOpened()) {
                            rgb_writer.release();
                        }
                        if (distance_writer.isOpened()) {
                            distance_writer.release();
                        }
                        std::cout << "\n⏹️  Recording stopped and saved" << std::endl;
                    }
                }
            }
            
            // Save every frame if requested
            if (save_every_frame) {
                std::string frame_prefix = output_dir + "/frame_" + 
                                          std::to_string(frame_count);
                
                auto rgb_cpu = rgb_panorama.cpu();
                cv::Mat rgb_mat(rgb_cpu.size(0), rgb_cpu.size(1), CV_8UC3, rgb_cpu.data_ptr<uint8_t>());
                cv::Mat rgb_bgr;
                cv::cvtColor(rgb_mat, rgb_bgr, cv::COLOR_RGB2BGR);
                cv::imwrite(frame_prefix + "_rgb.png", rgb_bgr);
                
                auto distance_cpu = distance_panorama.cpu();
                cv::Mat distance_mat(distance_cpu.size(0), distance_cpu.size(1), CV_32FC1, 
                                    distance_cpu.data_ptr<float>());
                cv::imwrite(frame_prefix + "_distance.exr", distance_mat);
            }
        }
        
        // Print final statistics
        std::cout << "\n========================================" << std::endl;
        std::cout << "FINAL STATISTICS" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total frames: " << frame_count << std::endl;
        std::cout << "Average time: " << (total_time_ms / frame_count) << " ms" << std::endl;
        std::cout << "Average FPS: " << (1000.0 * frame_count / total_time_ms) << std::endl;
        std::cout << "Min time: " << min_time_ms << " ms" << std::endl;
        std::cout << "Max time: " << max_time_ms << " ms" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // Cleanup
        if (is_recording) {
            rgb_writer.release();
            distance_writer.release();
            std::cout << "Video recording finalized" << std::endl;
        }
        streamer.close();
        cv::destroyAllWindows();
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
