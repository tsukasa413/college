/**
 * @file sphere_stereo_node.cpp
 * @brief ROS 2 node for real-time sphere stereo depth estimation
 * 
 * Integrates CameraCapture and DepthEstimator classes to provide
 * real-time RGB-D panorama generation as ROS topics.
 * 
 * @copyright CC BY-NC-SA 3.0
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>

#include "sphere_stereo_ros/CameraCapture.hpp"
#include "sphere_stereo_ros/DepthEstimator.hpp"
#include "sphere_stereo_ros/CalibrationSet.hpp"

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <chrono>

using namespace sphere_stereo_ros;

class SphereStereoNode : public rclcpp::Node
{
public:
    /**
     * @brief Constructor
     */
    SphereStereoNode() : Node("sphere_stereo_node")
    {
        // Declare parameters
        this->declare_parameter("calibration_file", std::string("/home/motoken/sphere-stereo/resources/calibration.json"));
        this->declare_parameter("config_path", std::string("/home/motoken/sphere-stereo/resources/"));
        this->declare_parameter("matching_width", 224);
        this->declare_parameter("matching_height", 224);
        this->declare_parameter("camera_fps", 30.0);
        this->declare_parameter("publish_fps", 30.0);
        this->declare_parameter("min_dist", 0.4);
        this->declare_parameter("max_dist", 100.0);
        this->declare_parameter("num_depth_candidates", 32);
        
        // Get parameters
        std::string calibration_file = this->get_parameter("calibration_file").as_string();
        std::string config_path = this->get_parameter("config_path").as_string();
        int matching_width = this->get_parameter("matching_width").as_int();
        int matching_height = this->get_parameter("matching_height").as_int();
        double camera_fps = this->get_parameter("camera_fps").as_double();
        double publish_fps = this->get_parameter("publish_fps").as_double();
        double min_dist = this->get_parameter("min_dist").as_double();
        double max_dist = this->get_parameter("max_dist").as_double();
        int num_depth_candidates = this->get_parameter("num_depth_candidates").as_int();
        
        RCLCPP_INFO(this->get_logger(), "Initializing Sphere Stereo Node");
        RCLCPP_INFO(this->get_logger(), "Calibration file: %s", calibration_file.c_str());
        RCLCPP_INFO(this->get_logger(), "Matching resolution: %dx%d", matching_width, matching_height);
        
        try {
            // Initialize calibration
            RCLCPP_INFO(this->get_logger(), "Loading calibration...");
            calibration_ = std::make_shared<CalibrationSet>();
            calibration_->loadFromFile(calibration_file, matching_width, matching_height);
            RCLCPP_INFO(this->get_logger(), "Calibration loaded successfully");
            
            // Initialize camera capture
            RCLCPP_INFO(this->get_logger(), "Initializing camera capture...");
            CameraCapture::Config capture_config;
            capture_config.camera_fps = camera_fps;
            capture_config.matching_width = matching_width;
            capture_config.matching_height = matching_height;
            capture_config.stitch_width = 672;
            capture_config.stitch_height = 672;
            capture_config.use_dummy_data = true;  // Set to false when real cameras are available
            
            capture_ = std::make_shared<CameraCapture>(capture_config);
            capture_->initialize();
            RCLCPP_INFO(this->get_logger(), "Camera capture initialized successfully");
            
            // Initialize depth estimator
            RCLCPP_INFO(this->get_logger(), "Initializing depth estimator...");
            DepthEstimatorConfig depth_config;
            depth_config.matching_width = matching_width;
            depth_config.matching_height = matching_height;
            depth_config.stitch_width = 672;
            depth_config.stitch_height = 672;
            depth_config.pano_width = 256;
            depth_config.pano_height = 128;
            depth_config.num_depth_candidates = num_depth_candidates;
            depth_config.min_dist = static_cast<float>(min_dist);
            depth_config.max_dist = static_cast<float>(max_dist);
            depth_config.reference_indices = {0, 2};  // Default reference cameras
            depth_config.sigma_i = 10.0f;
            depth_config.sigma_s = 25.0f;
            depth_config.sigma_i_dist = 5.0f;
            depth_config.sigma_s_dist = 12.5f;
            depth_config.cost_clamp = 500.0f;
            
            depth_estimator_ = std::make_shared<DepthEstimator>(*calibration_, depth_config);
            depth_estimator_->initialize();
            RCLCPP_INFO(this->get_logger(), "Depth estimator initialized successfully");
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize: %s", e.what());
            rclcpp::shutdown();
            return;
        }
        
        // Initialize image transport
        image_transport_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
        
        // Create publishers
        rgb_publisher_ = image_transport_->advertise("image_raw", 1);
        depth_publisher_ = image_transport_->advertise("depth_raw", 1);
        
        RCLCPP_INFO(this->get_logger(), "Publishers created");
        
        // Create timer for main processing loop
        auto timer_period = std::chrono::milliseconds(static_cast<int>(1000.0 / publish_fps));
        timer_ = this->create_wall_timer(timer_period, std::bind(&SphereStereoNode::timer_callback, this));
        
        RCLCPP_INFO(this->get_logger(), "Sphere Stereo Node started successfully (%.1f Hz)", publish_fps);
        
        // Initialize frame counter for performance monitoring
        frame_count_ = 0;
        last_fps_time_ = this->now();
    }
    
    /**
     * @brief Destructor
     */
    ~SphereStereoNode()
    {
        RCLCPP_INFO(this->get_logger(), "Shutting down Sphere Stereo Node");
    }

private:
    /**
     * @brief Timer callback for main processing loop
     */
    void timer_callback()
    {
        try {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // 1. Capture images from cameras
            std::vector<cv::Mat> images_matching, images_stitch;
            if (!capture_->grab(images_matching, images_stitch)) {
                RCLCPP_WARN(this->get_logger(), "Failed to capture images");
                return;
            }
            
            // 2. Update depth estimation
            if (!depth_estimator_->update(images_matching, images_stitch)) {
                RCLCPP_WARN(this->get_logger(), "Failed to update depth estimation");
                return;
            }
            
            // 3. Get results
            cv::Mat rgb_panorama, depth_map;
            if (!depth_estimator_->getRgbPanorama(rgb_panorama) || 
                !depth_estimator_->getDepthMap(depth_map)) {
                RCLCPP_WARN(this->get_logger(), "Failed to get estimation results");
                return;
            }
            
            // 4. Publish RGB image
            publish_rgb_image(rgb_panorama);
            
            // 5. Publish depth image
            publish_depth_image(depth_map);
            
            // Performance monitoring
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            frame_count_++;
            auto current_time = this->now();
            auto time_diff = current_time - last_fps_time_;
            
            if (time_diff.seconds() >= 5.0) {  // Report FPS every 5 seconds
                double fps = frame_count_ / time_diff.seconds();
                RCLCPP_INFO(this->get_logger(), 
                           "Processing: %.1f FPS, Last frame: %ld ms", 
                           fps, duration.count());
                frame_count_ = 0;
                last_fps_time_ = current_time;
            }
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error in timer callback: %s", e.what());
        }
    }
    
    /**
     * @brief Publish RGB panorama image
     */
    void publish_rgb_image(const cv::Mat& rgb_image)
    {
        try {
            // Convert OpenCV Mat to ROS Image message
            std_msgs::msg::Header header;
            header.stamp = this->now();
            header.frame_id = "camera_link";
            
            auto rgb_msg = cv_bridge::CvImage(header, "bgr8", rgb_image).toImageMsg();
            rgb_publisher_.publish(rgb_msg);
            
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception (RGB): %s", e.what());
        }
    }
    
    /**
     * @brief Publish depth map image
     */
    void publish_depth_image(const cv::Mat& depth_image)
    {
        try {
            // Convert OpenCV Mat to ROS Image message
            std_msgs::msg::Header header;
            header.stamp = this->now();
            header.frame_id = "camera_link";
            
            // Depth should be in meters as 32-bit float
            auto depth_msg = cv_bridge::CvImage(header, "32FC1", depth_image).toImageMsg();
            depth_publisher_.publish(depth_msg);
            
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception (Depth): %s", e.what());
        }
    }
    
    // Members
    std::shared_ptr<CalibrationSet> calibration_;
    std::shared_ptr<CameraCapture> capture_;
    std::shared_ptr<DepthEstimator> depth_estimator_;
    
    std::shared_ptr<image_transport::ImageTransport> image_transport_;
    image_transport::Publisher rgb_publisher_;
    image_transport::Publisher depth_publisher_;
    
    rclcpp::TimerBase::SharedPtr timer_;
    
    // Performance monitoring
    int frame_count_;
    rclcpp::Time last_fps_time_;
};

/**
 * @brief Main function
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<SphereStereoNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("sphere_stereo_node"), "Exception in main: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}