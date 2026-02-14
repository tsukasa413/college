/**
 * Omnidirectional RGBD SLAM Node for ROS 2
 * 
 * Integrates:
 * - 4-camera fisheye streaming
 * - Full-sphere stereo depth estimation
 * - 3D point cloud generation
 * - ROS 2 topic publishing for SLAM integration
 * 
 * Published topics:
 * - /omnidirectional/rgb_panorama (sensor_msgs/Image)
 * - /omnidirectional/depth_panorama (sensor_msgs/Image)
 * - /omnidirectional/point_cloud (sensor_msgs/PointCloud2)
 */

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/header.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "my_stereo_pkg/depth_estimation.hpp"
#include "my_stereo_pkg/calibration.hpp"
#include "my_stereo_pkg/point_cloud_generator.hpp"

#include <vector>
#include <string>
#include <memory>
#include <chrono>

namespace my_stereo_pkg {

// Import Calibration from my_stereo namespace
using my_stereo::Calibration;

/**
 * Configuration structure for the SLAM node
 */
struct SlamConfig {
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
    
    // Point cloud filtering
    float pointcloud_min_depth;
    float pointcloud_max_depth;
    
    // Publishing rate
    double publish_rate_hz;
    
    static SlamConfig load(const std::string& config_path);
};


/**
 * Camera Streamer for ROS 2 node
 */
class CameraStreamer {
public:
    CameraStreamer(int num_cameras = 4, int sensor_mode = 2);
    ~CameraStreamer();
    
    bool initialize();
    bool captureFrames(std::vector<cv::Mat>& frames);
    void close();
    
private:
    std::string buildGStreamerPipeline(int camera_id);
    
    int num_cameras_;
    int sensor_mode_;
    std::vector<cv::VideoCapture> caps_;
};


/**
 * Omnidirectional RGBD SLAM Node
 */
class OmnidirectionalSlamNode : public rclcpp::Node {
public:
    explicit OmnidirectionalSlamNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    
    ~OmnidirectionalSlamNode();
    
private:
    /**
     * Initialize the node components
     */
    void initialize();
    
    /**
     * Main processing loop callback
     */
    void processingLoop();
    
    /**
     * Load calibration from JSON
     */
    std::vector<Calibration> loadCalibration();
    
    /**
     * Load camera masks
     */
    std::vector<at::Tensor> loadMasks();
    
    /**
     * Preprocess camera frames to tensors
     */
    std::vector<at::Tensor> preprocessFrames(
        const std::vector<cv::Mat>& frames,
        const std::pair<int, int>& target_resolution
    );
    
    /**
     * Convert torch tensor to ROS Image message
     */
    sensor_msgs::msg::Image::SharedPtr tensorToImageMsg(
        const torch::Tensor& tensor,
        const std_msgs::msg::Header& header,
        const std::string& encoding
    );
    
    /**
     * Create camera info message for the panorama camera
     */
    sensor_msgs::msg::CameraInfo::SharedPtr createCameraInfoMsg(
        const std_msgs::msg::Header& header
    );
    
    /**
     * Publish TF transforms
     */
    void publishTransforms(const rclcpp::Time& timestamp);
    
    // ROS 2 publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
    
    // TF2 broadcaster
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // Timer for processing loop
    rclcpp::TimerBase::SharedPtr timer_;
    
    // Configuration
    SlamConfig config_;
    std::string dataset_path_;
    
    // Processing components
    std::unique_ptr<CameraStreamer> streamer_;
    std::unique_ptr<RGBDEstimator> estimator_;
    std::unique_ptr<PointCloudGenerator> pointcloud_generator_;
    
    // Calibration and masks
    std::vector<Calibration> calibrations_;
    std::vector<at::Tensor> masks_;
    
    // Device
    at::Device device_;
    
    // Frame counter
    int frame_count_;
    
    // Performance monitoring
    std::chrono::high_resolution_clock::time_point last_time_;
};

} // namespace my_stereo_pkg
