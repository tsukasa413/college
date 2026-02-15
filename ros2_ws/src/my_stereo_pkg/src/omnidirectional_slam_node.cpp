/**
 * Omnidirectional RGBD SLAM Node Implementation
 */

#include "my_stereo_pkg/omnidirectional_slam_node.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>

using json = nlohmann::json;
using namespace std::chrono_literals;

namespace my_stereo_pkg {

// ========================================
// Configuration Loading
// ========================================

SlamConfig SlamConfig::load(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }
    
    json config_json;
    file >> config_json;
    
    SlamConfig config;
    
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
    
    // Point cloud filtering (optional, with defaults)
    if (config_json.contains("pointcloud")) {
        config.pointcloud_min_depth = config_json["pointcloud"]["min_depth"].get<float>();
        config.pointcloud_max_depth = config_json["pointcloud"]["max_depth"].get<float>();
    } else {
        config.pointcloud_min_depth = 0.1f;
        config.pointcloud_max_depth = 100.0f;
    }
    
    // Publishing rate (optional, default to 10 Hz)
    if (config_json.contains("publish_rate_hz")) {
        config.publish_rate_hz = config_json["publish_rate_hz"].get<double>();
    } else {
        config.publish_rate_hz = 10.0;
    }
    
    return config;
}


// ========================================
// Camera Streamer Implementation
// ========================================

CameraStreamer::CameraStreamer(int num_cameras, int sensor_mode) 
    : num_cameras_(num_cameras), sensor_mode_(sensor_mode) {
    caps_.resize(num_cameras_);
}

CameraStreamer::~CameraStreamer() {
    close();
}

std::string CameraStreamer::buildGStreamerPipeline(int camera_id) {
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

bool CameraStreamer::initialize() {
    RCLCPP_INFO(rclcpp::get_logger("CameraStreamer"), 
                "Initializing %d camera streams...", num_cameras_);
    
    for (int i = 0; i < num_cameras_; ++i) {
        std::string pipeline = buildGStreamerPipeline(i);
        
        // Add delay between camera initializations
        if (i > 0) {
            RCLCPP_INFO(rclcpp::get_logger("CameraStreamer"),
                       "Waiting 2 seconds before initializing camera %d...", i);
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
        
        // Try to open camera with retry
        int max_retries = 3;
        bool opened = false;
        
        for (int retry = 0; retry < max_retries && !opened; ++retry) {
            if (retry > 0) {
                RCLCPP_WARN(rclcpp::get_logger("CameraStreamer"),
                           "Retry %d/%d for camera %d", retry, max_retries, i);
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            
            caps_[i] = cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);
            
            if (caps_[i].isOpened()) {
                cv::Mat test_frame;
                if (caps_[i].read(test_frame)) {
                    RCLCPP_INFO(rclcpp::get_logger("CameraStreamer"),
                               "Camera %d initialized: %dx%d", 
                               i, test_frame.cols, test_frame.rows);
                    opened = true;
                } else {
                    RCLCPP_ERROR(rclcpp::get_logger("CameraStreamer"),
                                "Camera %d opened but cannot read frames", i);
                    caps_[i].release();
                }
            } else {
                RCLCPP_ERROR(rclcpp::get_logger("CameraStreamer"),
                            "Failed to open camera %d", i);
            }
        }
        
        if (!opened) {
            RCLCPP_ERROR(rclcpp::get_logger("CameraStreamer"),
                        "Failed to initialize camera %d after %d retries", i, max_retries);
            // Clean up previously opened cameras
            for (int j = 0; j < i; ++j) {
                if (caps_[j].isOpened()) {
                    caps_[j].release();
                }
            }
            return false;
        }
    }
    
    RCLCPP_INFO(rclcpp::get_logger("CameraStreamer"), "All cameras initialized successfully!");
    return true;
}

bool CameraStreamer::captureFrames(std::vector<cv::Mat>& frames) {
    frames.clear();
    frames.reserve(num_cameras_);
    
    for (int i = 0; i < num_cameras_; ++i) {
        cv::Mat frame;
        if (!caps_[i].read(frame)) {
            RCLCPP_ERROR(rclcpp::get_logger("CameraStreamer"),
                        "Failed to read frame from camera %d", i);
            return false;
        }
        frames.push_back(frame);
    }
    
    return true;
}

void CameraStreamer::close() {
    for (auto& cap : caps_) {
        if (cap.isOpened()) {
            cap.release();
        }
    }
}


// ========================================
// Omnidirectional SLAM Node
// ========================================

OmnidirectionalSlamNode::OmnidirectionalSlamNode(const rclcpp::NodeOptions& options)
    : Node("omnidirectional_slam_node", options),
      device_(at::kCUDA, 0),
      frame_count_(0)
{
    // Declare and get parameters
    this->declare_parameter<std::string>("dataset_path", "/home/motoken/college/ros2_ws/src/my_stereo_pkg/resources");
    this->get_parameter("dataset_path", dataset_path_);
    
    RCLCPP_INFO(this->get_logger(), "========================================");
    RCLCPP_INFO(this->get_logger(), "Omnidirectional RGBD SLAM Node");
    RCLCPP_INFO(this->get_logger(), "========================================");
    RCLCPP_INFO(this->get_logger(), "Dataset path: %s", dataset_path_.c_str());
    
    // Create publishers
    rgb_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/omnidirectional/rgb_panorama", 10);
    depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/omnidirectional/depth_panorama", 10);
    camera_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
        "/rgb/camera_info", 10);
    pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/omnidirectional/point_cloud", 10);
    
    // Initialize TF2 broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    
    RCLCPP_INFO(this->get_logger(), "Publishers created:");
    RCLCPP_INFO(this->get_logger(), "  - /omnidirectional/rgb_panorama");
    RCLCPP_INFO(this->get_logger(), "  - /omnidirectional/depth_panorama");
    RCLCPP_INFO(this->get_logger(), "  - /rgb/camera_info");
    RCLCPP_INFO(this->get_logger(), "  - /omnidirectional/point_cloud");
    
    // Initialize components
    initialize();
}

OmnidirectionalSlamNode::~OmnidirectionalSlamNode() {
    if (streamer_) {
        streamer_->close();
    }
}

void OmnidirectionalSlamNode::initialize() {
    // Load configuration
    RCLCPP_INFO(this->get_logger(), "\n[1/7] Loading configuration...");
    std::string config_path = dataset_path_ + "/config.json";
    config_ = SlamConfig::load(config_path);
    
    RCLCPP_INFO(this->get_logger(), "Configuration loaded:");
    RCLCPP_INFO(this->get_logger(), "  Distance range: [%.2f, %.2f]", config_.min_dist, config_.max_dist);
    RCLCPP_INFO(this->get_logger(), "  Panorama resolution: [%d, %d]", 
                config_.panorama_resolution.first, config_.panorama_resolution.second);
    RCLCPP_INFO(this->get_logger(), "  Publish rate: %.1f Hz", config_.publish_rate_hz);
    
    // Initialize camera streamer
    RCLCPP_INFO(this->get_logger(), "\n[2/7] Initializing camera streams...");
    streamer_ = std::make_unique<CameraStreamer>(config_.num_cameras, config_.sensor_mode);
    if (!streamer_->initialize()) {
        throw std::runtime_error("Failed to initialize camera streams");
    }
    
    // Load calibration
    RCLCPP_INFO(this->get_logger(), "\n[3/7] Loading calibration...");
    calibrations_ = loadCalibration();
    RCLCPP_INFO(this->get_logger(), "Loaded %zu camera calibrations", calibrations_.size());
    
    // Load masks
    RCLCPP_INFO(this->get_logger(), "\n[4/7] Loading masks...");
    masks_ = loadMasks();
    RCLCPP_INFO(this->get_logger(), "Loaded %zu masks", masks_.size());
    
    // Calculate reprojection viewpoint
    RCLCPP_INFO(this->get_logger(), "\n[5/7] Calculating reprojection viewpoint...");
    at::Tensor viewpoint = torch::zeros({3}, torch::dtype(torch::kFloat32).device(device_));
    for (int ref_idx : config_.references_indices) {
        // Extract translation vector from RT matrix [0:3, 3]
        viewpoint += calibrations_[ref_idx].rt.slice(0, 0, 3).slice(1, 3, 4).squeeze();
    }
    viewpoint /= static_cast<float>(config_.references_indices.size());
    
    // Initialize RGBD estimator
    RCLCPP_INFO(this->get_logger(), "\n[6/7] Initializing RGBD Estimator...");
    estimator_ = std::make_unique<RGBDEstimator>(
        calibrations_,
        config_.min_dist,
        config_.max_dist,
        config_.candidate_count,
        config_.references_indices,
        viewpoint,
        masks_,
        config_.matching_resolution,
        config_.rgb_to_stitch_resolution,
        config_.panorama_resolution,
        config_.sigma_i,
        config_.sigma_s,
        device_
    );
    
    // Initialize point cloud generator
    RCLCPP_INFO(this->get_logger(), "\n[7/7] Initializing Point Cloud Generator...");
    pointcloud_generator_ = std::make_unique<PointCloudGenerator>(
        config_.panorama_resolution.first,
        config_.panorama_resolution.second,
        device_
    );
    
    // Warmup
    RCLCPP_INFO(this->get_logger(), "\nWarming up with live camera frames...");
    for (int i = 0; i < 3; ++i) {
        std::vector<cv::Mat> frames;
        if (!streamer_->captureFrames(frames)) {
            throw std::runtime_error("Failed to capture warmup frames");
        }
        
        auto images_to_match = preprocessFrames(frames, config_.matching_resolution);
        auto images_to_stitch = preprocessFrames(frames, config_.rgb_to_stitch_resolution);
        
        auto [rgb, dist] = estimator_->run(images_to_match, images_to_stitch);
    }
    torch::cuda::synchronize();
    
    RCLCPP_INFO(this->get_logger(), "\n========================================");
    RCLCPP_INFO(this->get_logger(), "Initialization complete!");
    RCLCPP_INFO(this->get_logger(), "Starting real-time processing at %.1f Hz", config_.publish_rate_hz);
    RCLCPP_INFO(this->get_logger(), "========================================\n");
    
    // Start processing loop timer
    auto period = std::chrono::duration<double>(1.0 / config_.publish_rate_hz);
    timer_ = this->create_wall_timer(
        std::chrono::duration_cast<std::chrono::milliseconds>(period),
        std::bind(&OmnidirectionalSlamNode::processingLoop, this)
    );
    
    last_time_ = std::chrono::high_resolution_clock::now();
}

std::vector<Calibration> OmnidirectionalSlamNode::loadCalibration() {
    std::string calib_path = dataset_path_ + "/calibration.json";
    std::ifstream file(calib_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open calibration file: " + calib_path);
    }
    
    json calib_data;
    file >> calib_data;
    
    if (!calib_data.contains("value0") || !calib_data["value0"].is_object()) {
        throw std::runtime_error("Calibration JSON missing 'value0' object");
    }
    
    auto& raw_calib = calib_data["value0"];
    
    if (!raw_calib.contains("T_imu_cam") || !raw_calib["T_imu_cam"].is_array()) {
        throw std::runtime_error("Calibration missing 'T_imu_cam' array");
    }
    
    if (!raw_calib.contains("intrinsics") || !raw_calib["intrinsics"].is_array()) {
        throw std::runtime_error("Calibration missing 'intrinsics' array");
    }
    
    auto& extrinsics_array = raw_calib["T_imu_cam"];
    auto& intrinsics_array = raw_calib["intrinsics"];
    
    std::vector<Calibration> calibrations;
    
    // Calculate matching scale (float2)
    float2 matching_scale;
    matching_scale.x = static_cast<float>(config_.matching_resolution.first) / 
                       static_cast<float>(config_.original_resolution.first);
    matching_scale.y = static_cast<float>(config_.matching_resolution.second) / 
                       static_cast<float>(config_.original_resolution.second);
    
    for (size_t i = 0; i < extrinsics_array.size(); ++i) {
        const auto& extrinsics = extrinsics_array[i];
        const auto& intrinsics = intrinsics_array[i];
        
        if (intrinsics["camera_type"].get<std::string>() != "ds") {
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
        
        calib.xi = cam_intrinsics["xi"].get<float>();
        calib.alpha = cam_intrinsics["alpha"].get<float>();
        
        // Set matching scale
        calib.matching_scale = matching_scale;
        
        // Parse extrinsics (quaternion + position format)
        // Extract quaternion (qx, qy, qz, qw)
        float qx = extrinsics["qx"].get<float>();
        float qy = extrinsics["qy"].get<float>();
        float qz = extrinsics["qz"].get<float>();
        float qw = extrinsics["qw"].get<float>();
        
        // Extract position
        float px = extrinsics["px"].get<float>();
        float py = extrinsics["py"].get<float>();
        float pz = extrinsics["pz"].get<float>();
        
        // Convert quaternion to rotation matrix using Eigen
        Eigen::Quaternionf quaternion(qw, qx, qy, qz);  // w, x, y, z order
        Eigen::Matrix3f rotation_matrix = quaternion.toRotationMatrix();
        
        // Build RT matrix (4x4 transformation matrix)
        calib.rt = torch::zeros({4, 4}, torch::dtype(torch::kFloat32).device(device_));
        
        // Copy rotation part
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                calib.rt[row][col] = rotation_matrix(row, col);
            }
        }
        
        // Copy translation part
        calib.rt[0][3] = px;
        calib.rt[1][3] = py;
        calib.rt[2][3] = pz;
        calib.rt[3][3] = 1.0f;
        
        calibrations.push_back(calib);
    }
    
    return calibrations;
}

std::vector<at::Tensor> OmnidirectionalSlamNode::loadMasks() {
    std::vector<at::Tensor> masks;
    
    for (int cam_idx = 0; cam_idx < config_.num_cameras; ++cam_idx) {
        std::ostringstream mask_path_stream;
        mask_path_stream << dataset_path_ << "/mask/cam" << cam_idx << ".png";
        std::string mask_path = mask_path_stream.str();
        
        cv::Mat mask_img = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
        if (mask_img.empty()) {
            RCLCPP_WARN(this->get_logger(), "Failed to load mask for camera %d using full mask", cam_idx);
            // Create all-ones mask directly on GPU with shape [1, H, W]
            at::Tensor mask = torch::ones({1, config_.matching_resolution.second, 
                                          config_.matching_resolution.first},
                                         torch::dtype(torch::kFloat32).device(device_));
            masks.push_back(mask);
        } else {
            cv::resize(mask_img, mask_img, 
                      cv::Size(config_.matching_resolution.first, config_.matching_resolution.second));
            
            // Convert to torch tensor [H, W]
            at::Tensor mask = torch::from_blob(
                mask_img.data,
                {mask_img.rows, mask_img.cols},
                torch::dtype(torch::kUInt8)
            ).clone().to(device_);
            
            // Convert to float32 [0, 1] and add batch dimension -> [1, H, W]
            mask = (mask.to(torch::kFloat32) / 255.0f).unsqueeze(0);
            masks.push_back(mask);
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "Loaded %zu masks", masks.size());
    return masks;
}

std::vector<at::Tensor> OmnidirectionalSlamNode::preprocessFrames(
    const std::vector<cv::Mat>& frames,
    const std::pair<int, int>& target_resolution
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
        tensor = tensor.to(device_).permute({2, 0, 1}).unsqueeze(0);  // [1, 3, H, W]
        
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

void OmnidirectionalSlamNode::processingLoop() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Capture frames from all cameras
        std::vector<cv::Mat> frames;
        if (!streamer_->captureFrames(frames)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to capture frames");
            return;
        }
        
        // Preprocess frames
        auto images_to_match = preprocessFrames(frames, config_.matching_resolution);
        auto images_to_stitch = preprocessFrames(frames, config_.rgb_to_stitch_resolution);
        
        // Run RGBD estimation
        auto [rgb_panorama, distance_panorama] = estimator_->run(images_to_match, images_to_stitch);
        
        // Calculate depth statistics for debugging
        auto distance_cpu = distance_panorama.cpu();
        auto depth_valid_mask = (distance_cpu > 0.0f) & (distance_cpu < 100.0f) & torch::isfinite(distance_cpu);
        auto valid_depths = distance_cpu.masked_select(depth_valid_mask);
        
        float depth_min = 0.0f, depth_max = 0.0f, depth_mean = 0.0f;
        int64_t valid_count = 0;
        if (valid_depths.numel() > 0) {
            depth_min = valid_depths.min().item<float>();
            depth_max = valid_depths.max().item<float>();
            depth_mean = valid_depths.mean().item<float>();
            valid_count = valid_depths.numel();
        }
        
        // Generate point cloud: returns [N, 6] tensor with [x, y, z, r, g, b] per row
        auto points_xyzrgb = pointcloud_generator_->generate(
            rgb_panorama,
            distance_panorama,
            config_.pointcloud_min_depth,
            config_.pointcloud_max_depth
        );
        
        torch::cuda::synchronize();
        
        // Create message header
        std_msgs::msg::Header header;
        header.stamp = this->now();
        header.frame_id = "camera_link";
        
        // Publish RGB panorama
        auto rgb_msg = tensorToImageMsg(rgb_panorama, header, "rgb8");
        rgb_pub_->publish(*rgb_msg);
        
        // Publish depth panorama
        auto depth_msg = tensorToImageMsg(distance_panorama, header, "32FC1");
        depth_pub_->publish(*depth_msg);
        
        // Publish camera info
        auto camera_info_msg = createCameraInfoMsg(header);
        camera_info_pub_->publish(*camera_info_msg);
        
        // Publish point cloud using static converter
        auto pointcloud_msg = PointCloudGenerator::toRosPointCloud2(
            points_xyzrgb, 
            header.frame_id, 
            rclcpp::Time(header.stamp)
        );
        pointcloud_pub_->publish(pointcloud_msg);
        
        // Publish TF transforms
        publishTransforms(header.stamp);
        
        // Performance monitoring
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        frame_count_++;
        
        if (frame_count_ % 30 == 0) {
            RCLCPP_INFO(this->get_logger(), 
                       "Frame %d | Time: %ld ms | Points: %ld | Depth: min=%.2f max=%.2f mean=%.2f valid=%ld/%ld (%.1f%%)",
                       frame_count_, duration.count(), points_xyzrgb.size(0),
                       depth_min, depth_max, depth_mean, valid_count, 
                       distance_panorama.numel(), 
                       100.0f * valid_count / distance_panorama.numel());
        }
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Processing error: %s", e.what());
    }
}

sensor_msgs::msg::Image::SharedPtr OmnidirectionalSlamNode::tensorToImageMsg(
    const torch::Tensor& tensor,
    const std_msgs::msg::Header& header,
    const std::string& encoding
) {
    auto msg = std::make_shared<sensor_msgs::msg::Image>();
    msg->header = header;
    
    auto tensor_cpu = tensor.cpu().contiguous();
    
    if (encoding == "rgb8") {
        // RGB image [H, W, 3], uint8
        msg->height = tensor_cpu.size(0);
        msg->width = tensor_cpu.size(1);
        msg->encoding = encoding;
        msg->step = msg->width * 3;
        
        size_t data_size = msg->height * msg->step;
        msg->data.resize(data_size);
        
        std::memcpy(msg->data.data(), tensor_cpu.data_ptr<uint8_t>(), data_size);
        
    } else if (encoding == "32FC1") {
        // Depth image [H, W], float32
        msg->height = tensor_cpu.size(0);
        msg->width = tensor_cpu.size(1);
        msg->encoding = encoding;
        msg->step = msg->width * sizeof(float);
        
        size_t data_size = msg->height * msg->step;
        msg->data.resize(data_size);
        
        std::memcpy(msg->data.data(), tensor_cpu.data_ptr<float>(), data_size);
    }
    
    return msg;
}

// Note: pointCloudToMsg() is no longer needed - we use PointCloudGenerator::toRosPointCloud2() instead

sensor_msgs::msg::CameraInfo::SharedPtr OmnidirectionalSlamNode::createCameraInfoMsg(
    const std_msgs::msg::Header& header
) {
    auto msg = std::make_shared<sensor_msgs::msg::CameraInfo>();
    msg->header = header;
    
    // Panorama resolution (width x height)
    // Note: config.panorama_resolution is std::pair<int, int> where:
    //   first = width (2048), second = height (1024)
    int width = config_.panorama_resolution.first;  // width
    int height = config_.panorama_resolution.second;  // height
    
    msg->width = width;
    msg->height = height;
    msg->distortion_model = "plumb_bob";
    
    // For equirectangular projection:
    // focal length = width / (2 * pi) ≈ width / 6.28
    // This ensures the horizontal FOV covers 360 degrees
    double fx = width / (2.0 * M_PI);
    double fy = fx;  // Assume square pixels
    double cx = width / 2.0;
    double cy = height / 2.0;
    
    // Camera intrinsic matrix K
    msg->k = {
        fx,  0.0, cx,
        0.0, fy,  cy,
        0.0, 0.0, 1.0
    };
    
    // Distortion coefficients (k1, k2, t1, t2, k3)
    // For panorama, distortion is already corrected
    msg->d = {0.0, 0.0, 0.0, 0.0, 0.0};
    
    // Rectification matrix (identity for monocular)
    msg->r = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    
    // Projection matrix P
    msg->p = {
        fx,  0.0, cx,  0.0,
        0.0, fy,  cy,  0.0,
        0.0, 0.0, 1.0, 0.0
    };
    
    return msg;
}

void OmnidirectionalSlamNode::publishTransforms(const rclcpp::Time& timestamp) {
    // Publish world -> omnidirectional_camera transform
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = timestamp;
    transform.header.frame_id = "world";
    transform.child_frame_id = "camera_link";
    
    // Identity transform (camera at origin)
    transform.transform.translation.x = 0.0;
    transform.transform.translation.y = 0.0;
    transform.transform.translation.z = 0.0;
    transform.transform.rotation.x = 0.0;
    transform.transform.rotation.y = 0.0;
    transform.transform.rotation.z = 0.0;
    transform.transform.rotation.w = 1.0;
    
    tf_broadcaster_->sendTransform(transform);
}

} // namespace my_stereo_pkg


// ========================================
// Main function
// ========================================

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<my_stereo_pkg::OmnidirectionalSlamNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Fatal error: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}
