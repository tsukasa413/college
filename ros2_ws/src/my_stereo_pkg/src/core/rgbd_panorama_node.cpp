/**
 * @file rgbd_panorama_node.cpp
 * @brief ROS2 Node Implementation for Real-Time RGBD Panorama Generation
 * 
 * C++/CUDA implementation equivalent to sphere-stereo/python/main.py
 * Subscribes to quad_cam_system topics and publishes RGBD panorama
 */

#include "my_stereo_pkg/rgbd_panorama_node.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <chrono>
#include <filesystem>

namespace my_stereo_pkg {

RGBDPanoramaNode::RGBDPanoramaNode(const rclcpp::NodeOptions& options)
    : Node("rgbd_panorama_node", options),
      device_(at::kCUDA, 0),
      frame_count_(0),
      running_(true)
{
    // Declare and get parameters
    this->declare_parameter<std::string>("calibration_path", "");
    this->declare_parameter<std::vector<int64_t>>("references_indices", {0, 2});
    this->declare_parameter<double>("min_dist", 0.55);
    this->declare_parameter<double>("max_dist", 100.0);
    this->declare_parameter<int>("candidate_count", 32);
    this->declare_parameter<double>("sigma_i", 10.0);
    this->declare_parameter<double>("sigma_s", 25.0);
    this->declare_parameter<std::vector<int64_t>>("matching_resolution", {1024, 1024});
    this->declare_parameter<std::vector<int64_t>>("rgb_to_stitch_resolution", {1216, 1216});
    this->declare_parameter<std::vector<int64_t>>("panorama_resolution", {2048, 1024});
    this->declare_parameter<std::vector<int64_t>>("original_resolution", {1944, 1096});
    this->declare_parameter<int>("device_id", 0);
    this->declare_parameter<double>("fps", 10.0);  // Lower FPS for CUDA processing
    this->declare_parameter<bool>("use_test_images", false);
    this->declare_parameter<std::string>("test_images_dir", "");

    calibration_path_ = this->get_parameter("calibration_path").as_string();
    references_indices_ = this->get_parameter("references_indices").as_integer_array();
    min_dist_ = this->get_parameter("min_dist").as_double();
    max_dist_ = this->get_parameter("max_dist").as_double();
    candidate_count_ = this->get_parameter("candidate_count").as_int();
    sigma_i_ = this->get_parameter("sigma_i").as_double();
    sigma_s_ = this->get_parameter("sigma_s").as_double();
    matching_resolution_ = this->get_parameter("matching_resolution").as_integer_array();
    rgb_to_stitch_resolution_ = this->get_parameter("rgb_to_stitch_resolution").as_integer_array();
    panorama_resolution_ = this->get_parameter("panorama_resolution").as_integer_array();
    original_resolution_ = this->get_parameter("original_resolution").as_integer_array();
    device_id_ = this->get_parameter("device_id").as_int();
    fps_ = this->get_parameter("fps").as_double();
    use_test_images_ = this->get_parameter("use_test_images").as_bool();
    test_images_dir_ = this->get_parameter("test_images_dir").as_string();

    RCLCPP_INFO(this->get_logger(), "=== RGBD Panorama Node Starting ===");
    RCLCPP_INFO(this->get_logger(), "Calibration: %s", calibration_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "References: [%ld, %ld]", 
                references_indices_[0], references_indices_[1]);
    RCLCPP_INFO(this->get_logger(), "Distance range: [%.2f, %.2f]", min_dist_, max_dist_);
    RCLCPP_INFO(this->get_logger(), "Matching resolution: [%ld, %ld]",
                matching_resolution_[0], matching_resolution_[1]);
    RCLCPP_INFO(this->get_logger(), "Panorama resolution: [%ld, %ld]",
                panorama_resolution_[0], panorama_resolution_[1]);
    RCLCPP_INFO(this->get_logger(), "FPS: %.1f", fps_);
    RCLCPP_INFO(this->get_logger(), "Mode: %s", use_test_images_ ? "Test Images" : "Live Camera");
    if (use_test_images_) {
        RCLCPP_INFO(this->get_logger(), "Test images directory: %s", test_images_dir_.c_str());
    }

    // Load calibrations
    load_calibrations();

    // Initialize RGBD Estimator
    initialize_rgbd_estimator();

    // Initialize input source
    if (use_test_images_) {
        load_test_images();
    } else {
        initialize_cameras();
    }

    // Setup publishers
    pub_rgb_panorama_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/rgbd_panorama/rgb", 10
    );
    pub_depth_panorama_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/rgbd_panorama/depth", 10
    );
    pub_inv_depth_panorama_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/rgbd_panorama/inv_depth", 10
    );

    // Create timer for processing frames
    auto timer_period = std::chrono::duration<double>(1.0 / fps_);
    timer_ = this->create_wall_timer(
        timer_period,
        std::bind(&RGBDPanoramaNode::process_frames, this)
    );

    RCLCPP_INFO(this->get_logger(), "Node initialized successfully. Starting capture...");
}

RGBDPanoramaNode::~RGBDPanoramaNode() {
    RCLCPP_INFO(this->get_logger(), "Shutting down RGBD Panorama Node");
    running_ = false;
    
    // Wait for capture threads
    for (auto& thread : capture_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Release cameras
    for (auto& cam : cameras_) {
        if (cam.isOpened()) {
            cam.release();
        }
    }
}

void RGBDPanoramaNode::load_calibrations() {
    RCLCPP_INFO(this->get_logger(), "Loading calibrations from: %s", calibration_path_.c_str());
    std::cout << "[RGBDPanoramaNode] load_calibrations() starting..." << std::endl;

    std::ifstream file(calibration_path_);
    if (!file.is_open()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open calibration file: %s", 
                     calibration_path_.c_str());
        throw std::runtime_error("Calibration file not found");
    }

    nlohmann::json file_root = nlohmann::json::parse(file);
    nlohmann::json root;

    // Extract value0 if exists (Basalt format)
    if (file_root.contains("value0")) {
        root = file_root["value0"];
    } else {
        root = file_root;
    }

    // Convert parameters to std::vector<int>
    std::vector<int> matching_res = {
        static_cast<int>(matching_resolution_[0]),
        static_cast<int>(matching_resolution_[1])
    };
    std::vector<int> original_res = {
        static_cast<int>(original_resolution_[0]),
        static_cast<int>(original_resolution_[1])
    };

    // Use CalibrationParser from utils.cpp
    namespace ss = sphere_stereo;
    std::cout << "[RGBDPanoramaNode] Calling CalibrationParser::load_json_basalt..." << std::endl;
    std::vector<ss::CameraCalibrationGPU> calib_gpus = 
        ss::CalibrationParser::load_json_basalt(
            calibration_path_, matching_res, original_res
        );
    std::cout << "[RGBDPanoramaNode] Loaded " << calib_gpus.size() << " GPU calibrations" << std::endl;

    // Convert to DoubleSphereCalibration format for RGBD_Estimator
    std::cout << "[RGBDPanoramaNode] Converting to DoubleSphereCalibration..." << std::endl;
    calibrations_.resize(calib_gpus.size());
    for (size_t i = 0; i < calib_gpus.size(); i++) {
        std::cout << "[RGBDPanoramaNode] Processing camera " << i << std::endl;
        std::cout << "  Unified Memory address: " << calib_gpus[i].get_unified_ptr() << std::endl;
        
        const ss::CameraCalibration& h_calib = calib_gpus[i].get_host_ref();
        std::cout << "[RGBDPanoramaNode] Got host reference for camera " << i 
                  << " at address: " << &h_calib << std::endl;
        
        DoubleSphereCalibration& ds = calibrations_[i];
        ds.fx = h_calib.fx;
        ds.fy = h_calib.fy;
        ds.cx = h_calib.cx;
        ds.cy = h_calib.cy;
        ds.xi = h_calib.xi;
        ds.alpha = h_calib.alpha;
        ds.width = static_cast<float>(h_calib.width);
        ds.height = static_cast<float>(h_calib.height);
        ds.matching_scale = h_calib.matching_scale;

        // Copy RT matrix (4x4 row-major) directly from POD structure
        for (int k = 0; k < 16; k++) {
            ds.rt[k] = h_calib.rt[k];
        }
        
        // Extract R (3x3) and t (3x1) from RT matrix
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                ds.R[r * 3 + c] = h_calib.rt[r * 4 + c];
            }
            ds.t[r] = h_calib.rt[r * 4 + 3];
        }
    }

    std::cout << "[RGBDPanoramaNode] All calibrations converted successfully" << std::endl;
    std::cout << "[RGBDPanoramaNode] calib_gpus vector going out of scope..." << std::endl;
    
    RCLCPP_INFO(this->get_logger(), "Loaded %zu camera calibrations", calibrations_.size());
}

void RGBDPanoramaNode::initialize_rgbd_estimator() {
    RCLCPP_INFO(this->get_logger(), "Initializing RGBD_Estimator...");
    std::cout << "[RGBDPanoramaNode] Starting RGBD_Estimator initialization" << std::endl;

    // Convert references_indices to std::vector<int>
    std::vector<int> ref_indices;
    for (auto idx : references_indices_) {
        ref_indices.push_back(static_cast<int>(idx));
    }
    std::cout << "[RGBDPanoramaNode] Reference indices: " << ref_indices.size() << " cameras" << std::endl;

    // Calculate reprojection viewpoint (center of reference cameras)
    std::vector<float> reprojection_viewpoint(3, 0.0f);
    for (int ref_idx : ref_indices) {
        reprojection_viewpoint[0] += calibrations_[ref_idx].rt[0 * 4 + 3];
        reprojection_viewpoint[1] += calibrations_[ref_idx].rt[1 * 4 + 3];
        reprojection_viewpoint[2] += calibrations_[ref_idx].rt[2 * 4 + 3];
    }
    for (int i = 0; i < 3; i++) {
        reprojection_viewpoint[i] /= ref_indices.size();
    }
    std::cout << "[RGBDPanoramaNode] Reprojection viewpoint: [" 
              << reprojection_viewpoint[0] << ", " 
              << reprojection_viewpoint[1] << ", " 
              << reprojection_viewpoint[2] << "]" << std::endl;

    // Flatten calibration data for RGBD_Estimator constructor
    std::cout << "[RGBDPanoramaNode] Flattening calibration data..." << std::endl;
    std::vector<float> calibrations_rt;
    std::vector<float> calibrations_intrinsics;
    std::vector<float> calibrations_sphere;
    std::vector<float> calibrations_resolution;
    std::vector<int> image_widths;
    std::vector<int> image_heights;

    for (const auto& calib : calibrations_) {
        // RT matrix (16 elements)
        for (int i = 0; i < 16; i++) {
            calibrations_rt.push_back(calib.rt[i]);
        }

        // Intrinsics [fx, fy, cx, cy]
        calibrations_intrinsics.push_back(calib.fx);
        calibrations_intrinsics.push_back(calib.fy);
        calibrations_intrinsics.push_back(calib.cx);
        calibrations_intrinsics.push_back(calib.cy);

        // Sphere parameters [xi, alpha]
        calibrations_sphere.push_back(calib.xi);
        calibrations_sphere.push_back(calib.alpha);

        // Resolution [width, height]
        calibrations_resolution.push_back(calib.width);
        calibrations_resolution.push_back(calib.height);

        image_widths.push_back(static_cast<int>(calib.width));
        image_heights.push_back(static_cast<int>(calib.height));
    }
    std::cout << "[RGBDPanoramaNode] Calibration data flattened: " << calibrations_.size() << " cameras" << std::endl;

    // Create RGBD_Estimator
    std::cout << "[RGBDPanoramaNode] Creating RGBD_Estimator object..." << std::endl;
    rgbd_estimator_ = std::make_unique<RGBD_Estimator>(
        calibrations_rt,
        calibrations_intrinsics,
        calibrations_sphere,
        calibrations_resolution,
        min_dist_,
        max_dist_,
        candidate_count_,
        ref_indices,
        reprojection_viewpoint,
        image_widths,
        image_heights,
        static_cast<int>(matching_resolution_[0]),
        static_cast<int>(matching_resolution_[1]),
        static_cast<int>(rgb_to_stitch_resolution_[0]),
        static_cast<int>(rgb_to_stitch_resolution_[1]),
        static_cast<int>(panorama_resolution_[0]),
        static_cast<int>(panorama_resolution_[1]),
        sigma_i_,
        sigma_s_,
        device_id_
    );

    RCLCPP_INFO(this->get_logger(), "RGBD_Estimator initialized successfully");
}

void RGBDPanoramaNode::load_test_images() {
    RCLCPP_INFO(this->get_logger(), "Loading test images from: %s", test_images_dir_.c_str());
    
    test_images_.resize(4);
    
    for (int i = 0; i < 4; i++) {
        // Load first image from cam{i} directory (skip mask.png)
        std::string cam_dir = test_images_dir_ + "/cam" + std::to_string(i);
        
        // Find first non-mask image file in directory
        std::string image_path;
        for (const auto& entry : std::filesystem::directory_iterator(cam_dir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                std::string ext = entry.path().extension().string();
                
                // Skip mask.png and only accept jpg/jpeg/png images
                if (filename != "mask.png" && 
                    (ext == ".png" || ext == ".jpg" || ext == ".jpeg")) {
                    image_path = entry.path().string();
                    break;
                }
            }
        }
        
        if (image_path.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No image found in %s", cam_dir.c_str());
            throw std::runtime_error("Test image not found");
        }
        
        test_images_[i] = cv::imread(image_path, cv::IMREAD_COLOR);
        if (test_images_[i].empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load image: %s", image_path.c_str());
            throw std::runtime_error("Failed to load test image");
        }
        
        RCLCPP_INFO(this->get_logger(), "Loaded camera %d: %s (size: %dx%d)",
                   i, image_path.c_str(), test_images_[i].cols, test_images_[i].rows);
    }
    
    RCLCPP_INFO(this->get_logger(), "All test images loaded successfully");
}

void RGBDPanoramaNode::initialize_cameras() {
    RCLCPP_INFO(this->get_logger(), "Initializing cameras (GStreamer pipelines)...");
    
    // Initialize buffers
    cameras_.resize(4);
    frame_buffers_.resize(4);
    buffer_mutexes_.resize(4);
    for (int i = 0; i < 4; i++) {
        buffer_mutexes_[i] = std::make_unique<std::mutex>();
    }
    
    int target_width = static_cast<int>(original_resolution_[0]);
    int target_height = static_cast<int>(original_resolution_[1]);
    
    for (int i = 0; i < 4; i++) {
        // GStreamer pipeline matching quad_cam_system
        // Using sensor-mode=2 (1944x1096) for full FOV
        std::string pipeline = 
            "nvarguscamerasrc sensor-id=" + std::to_string(i) + " sensor-mode=2 bufapi-version=1 ! "
            "video/x-raw(memory:NVMM), width=(int)1944, height=(int)1096, format=(string)NV12, framerate=(fraction)25/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! "
            "appsink emit-signals=false sync=false drop=true max-buffers=2";
        
        RCLCPP_INFO(this->get_logger(), "Opening Camera %d...", i);
        RCLCPP_INFO(this->get_logger(), "Pipeline: %s", pipeline.c_str());
        
        cameras_[i].open(pipeline, cv::CAP_GSTREAMER);
        
        if (cameras_[i].isOpened()) {
            // Test frame capture
            cv::Mat test_frame;
            if (cameras_[i].read(test_frame)) {
                RCLCPP_INFO(this->get_logger(), "Camera %d opened successfully (frame size: %dx%d)", 
                           i, test_frame.cols, test_frame.rows);
                
                // Start capture thread
                capture_threads_.emplace_back(&RGBDPanoramaNode::capture_worker, this, i);
            } else {
                RCLCPP_ERROR(this->get_logger(), "Camera %d opened but cannot read frames!", i);
                cameras_[i].release();
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to open camera %d!", i);
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "Camera initialization complete");
}

void RGBDPanoramaNode::capture_worker(int camera_id) {
    RCLCPP_INFO(this->get_logger(), "Starting capture worker for camera %d", camera_id);
    
    while (running_ && cameras_[camera_id].isOpened()) {
        cv::Mat frame;
        if (cameras_[camera_id].read(frame)) {
            // Store in buffer (thread-safe)
            std::lock_guard<std::mutex> lock(*buffer_mutexes_[camera_id]);
            frame_buffers_[camera_id].push_back(frame.clone());
            if (frame_buffers_[camera_id].size() > 2) {
                frame_buffers_[camera_id].pop_front();
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "Capture worker for camera %d stopped", camera_id);
}

void RGBDPanoramaNode::process_frames() {
    frame_count_++;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    RCLCPP_INFO(this->get_logger(), "Processing frame %d", frame_count_);
    
    try {
        // Get frames based on mode
        std::vector<cv::Mat> frames(4);
        
        if (use_test_images_) {
            // Use static test images
            RCLCPP_INFO(this->get_logger(), "Using test images");
            for (int i = 0; i < 4; i++) {
                frames[i] = test_images_[i].clone();
                RCLCPP_INFO(this->get_logger(), "  Frame %d: %dx%d, channels=%d, type=%d",
                           i, frames[i].cols, frames[i].rows, frames[i].channels(), frames[i].type());
            }
        } else {
            // Check if all cameras have frames
            for (int i = 0; i < 4; i++) {
                std::lock_guard<std::mutex> lock(*buffer_mutexes_[i]);
                if (frame_buffers_[i].empty()) {
                    return;  // Wait for all cameras to have frames
                }
            }
            
            // Get latest frames from all cameras
            for (int i = 0; i < 4; i++) {
                std::lock_guard<std::mutex> lock(*buffer_mutexes_[i]);
                frames[i] = frame_buffers_[i].back().clone();
            }
        }
        
        // Preprocess images for matching
        RCLCPP_INFO(this->get_logger(), "Preprocessing images for matching...");
        std::vector<std::vector<float>> images_to_match;
        for (int i = 0; i < 4; i++) {
            RCLCPP_INFO(this->get_logger(), "  Processing camera %d for matching", i);
            images_to_match.push_back(preprocess_image_to_match(frames[i]));
        }
        
        // Preprocess images for stitching (only reference cameras)
        RCLCPP_INFO(this->get_logger(), "Preprocessing images for stitching...");
        std::vector<std::vector<float>> images_to_stitch;
        for (auto ref_idx : references_indices_) {
            RCLCPP_INFO(this->get_logger(), "  Processing camera %ld for stitching", ref_idx);
            images_to_stitch.push_back(preprocess_image_to_stitch(frames[ref_idx]));
        }

        // Estimate RGBD panorama
        RCLCPP_INFO(this->get_logger(), "Calling estimate_RGBD_panorama...");
        auto [rgb_panorama, distance_panorama] = 
            rgbd_estimator_->estimate_RGBD_panorama(images_to_match, images_to_stitch);
        
        // Create ROS header with current timestamp
        std_msgs::msg::Header header;
        header.stamp = this->get_clock()->now();
        header.frame_id = "camera_base";

        // Convert to ROS messages and publish
        auto rgb_msg = tensor_to_rgb_msg(
            rgb_panorama,
            static_cast<int>(panorama_resolution_[0]),
            static_cast<int>(panorama_resolution_[1]),
            header
        );
        pub_rgb_panorama_->publish(*rgb_msg);

        auto depth_msg = tensor_to_depth_msg(
            distance_panorama,
            static_cast<int>(panorama_resolution_[0]),
            static_cast<int>(panorama_resolution_[1]),
            header
        );
        pub_depth_panorama_->publish(*depth_msg);

        // Compute inverse depth for visualization
        std::vector<float> inv_depth(distance_panorama.size());
        for (size_t i = 0; i < distance_panorama.size(); i++) {
            inv_depth[i] = 1.0f / distance_panorama[i];
        }
        auto inv_depth_msg = tensor_to_depth_msg(
            inv_depth,
            static_cast<int>(panorama_resolution_[0]),
            static_cast<int>(panorama_resolution_[1]),
            header
        );
        pub_inv_depth_panorama_->publish(*inv_depth_msg);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        ).count();

        RCLCPP_INFO(this->get_logger(), "Frame %d processed in %ld ms (%.2f FPS)",
                    frame_count_, duration, 1000.0 / duration);
        
        // If using test images, process only once and stop timer
        if (use_test_images_) {
            RCLCPP_INFO(this->get_logger(), "Test image processing complete. Node will continue running for topic inspection.");
            timer_->cancel();
        }

    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error processing frame %d: %s",
                     frame_count_, e.what());
    }
}

std::vector<float> RGBDPanoramaNode::preprocess_image_to_match(const cv::Mat& img) {
    RCLCPP_INFO(this->get_logger(), "    Input: %dx%d, type=%d, continuous=%d",
               img.cols, img.rows, img.type(), img.isContinuous());
    
    // Ensure input is continuous
    cv::Mat input = img.isContinuous() ? img : img.clone();
    
    // Resize to matching resolution
    cv::Mat resized;
    RCLCPP_INFO(this->get_logger(), "    Resizing to %ldx%ld...",
               matching_resolution_[0], matching_resolution_[1]);
    cv::resize(input, resized, 
               cv::Size(static_cast<int>(matching_resolution_[0]),
                       static_cast<int>(matching_resolution_[1])),
               0, 0, cv::INTER_AREA);
    RCLCPP_INFO(this->get_logger(), "    Resize complete");

    // Convert BGR to RGB and to float [0, 255]
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Flatten to vector [H*W*3]
    std::vector<float> output(rgb.rows * rgb.cols * 3);
    for (int i = 0; i < rgb.rows * rgb.cols; i++) {
        output[i * 3 + 0] = static_cast<float>(rgb.data[i * 3 + 0]);
        output[i * 3 + 1] = static_cast<float>(rgb.data[i * 3 + 1]);
        output[i * 3 + 2] = static_cast<float>(rgb.data[i * 3 + 2]);
    }

    return output;
}

std::vector<float> RGBDPanoramaNode::preprocess_image_to_stitch(const cv::Mat& img) {
    // Resize to stitching resolution
    cv::Mat resized;
    cv::resize(img, resized,
               cv::Size(static_cast<int>(rgb_to_stitch_resolution_[0]),
                       static_cast<int>(rgb_to_stitch_resolution_[1])),
               0, 0, cv::INTER_AREA);

    // Convert BGR to RGB and to float [0, 255]
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Flatten to vector [H*W*3]
    std::vector<float> output(rgb.rows * rgb.cols * 3);
    for (int i = 0; i < rgb.rows * rgb.cols; i++) {
        output[i * 3 + 0] = static_cast<float>(rgb.data[i * 3 + 0]);
        output[i * 3 + 1] = static_cast<float>(rgb.data[i * 3 + 1]);
        output[i * 3 + 2] = static_cast<float>(rgb.data[i * 3 + 2]);
    }

    return output;
}

sensor_msgs::msg::Image::SharedPtr RGBDPanoramaNode::tensor_to_rgb_msg(
    const std::vector<uint8_t>& rgb_data,
    int width, int height,
    const std_msgs::msg::Header& header
) {
    auto msg = std::make_shared<sensor_msgs::msg::Image>();
    msg->header = header;
    msg->height = height;
    msg->width = width;
    msg->encoding = "rgb8";
    msg->is_bigendian = false;
    msg->step = width * 3;
    msg->data = rgb_data;

    return msg;
}

sensor_msgs::msg::Image::SharedPtr RGBDPanoramaNode::tensor_to_depth_msg(
    const std::vector<float>& depth_data,
    int width, int height,
    const std_msgs::msg::Header& header
) {
    auto msg = std::make_shared<sensor_msgs::msg::Image>();
    msg->header = header;
    msg->height = height;
    msg->width = width;
    msg->encoding = "32FC1";
    msg->is_bigendian = false;
    msg->step = width * sizeof(float);
    
    // Copy float data to uint8 vector
    msg->data.resize(depth_data.size() * sizeof(float));
    std::memcpy(msg->data.data(), depth_data.data(), msg->data.size());

    return msg;
}

} // namespace my_stereo_pkg

// Main function
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<my_stereo_pkg::RGBDPanoramaNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
