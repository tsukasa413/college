/**
 * @file rgbd_panorama_node.hpp
 * @brief ROS2 Node for Real-Time RGBD Panorama Generation from Multi-Fisheye Cameras
 * 
 * C++/CUDA implementation of sphere-stereo/python/main.py
 * Subscribes to 4 synchronized fisheye camera images and publishes RGBD panorama
 */

#ifndef MY_STEREO_PKG__RGBD_PANORAMA_NODE_HPP_
#define MY_STEREO_PKG__RGBD_PANORAMA_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <deque>

#include "my_stereo_pkg/depth_estimation.hpp"
#include "my_stereo_pkg/stitcher.hpp"
#include "my_stereo_pkg/utils.hpp"

namespace my_stereo_pkg {

class RGBDPanoramaNode : public rclcpp::Node {
public:
    /**
     * @brief Constructor
     */
    explicit RGBDPanoramaNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

    /**
     * @brief Destructor
     */
    ~RGBDPanoramaNode();

private:
    /**
     * @brief Load camera calibrations from JSON file
     */
    void load_calibrations();

    /**
     * @brief Initialize RGBD_Estimator with loaded calibrations
     */
    void initialize_rgbd_estimator();

    /**
     * @brief Initialize camera capture (GStreamer pipelines)
     */
    void initialize_cameras();

    /**
     * @brief Load test images from directory
     */
    void load_test_images();

    /**
     * @brief Camera capture worker thread
     */
    void capture_worker(int camera_id);

    /**
     * @brief Timer callback for processing frames
     */
    void process_frames();

    /**
     * @brief Preprocess image for matching (resize + RGB to float)
     */
    std::vector<float> preprocess_image_to_match(const cv::Mat& img);

    /**
     * @brief Preprocess image for stitching (resize + RGB to float)
     */
    std::vector<float> preprocess_image_to_stitch(const cv::Mat& img);

    /**
     * @brief Convert RGB panorama tensor to ROS Image message
     */
    sensor_msgs::msg::Image::SharedPtr tensor_to_rgb_msg(
        const std::vector<uint8_t>& rgb_data,
        int width, int height,
        const std_msgs::msg::Header& header
    );

    /**
     * @brief Convert depth panorama to ROS Image message (32FC1 format)
     */
    sensor_msgs::msg::Image::SharedPtr tensor_to_depth_msg(
        const std::vector<float>& depth_data,
        int width, int height,
        const std_msgs::msg::Header& header
    );

    // ROS2 Parameters
    std::string calibration_path_;
    std::vector<int64_t> references_indices_;
    float min_dist_;
    float max_dist_;
    int candidate_count_;
    float sigma_i_;
    float sigma_s_;
    std::vector<int64_t> matching_resolution_;
    std::vector<int64_t> rgb_to_stitch_resolution_;
    std::vector<int64_t> panorama_resolution_;
    std::vector<int64_t> original_resolution_;
    int device_id_;
    bool use_test_images_;
    std::string test_images_dir_;

    // Camera capture (direct access)
    std::vector<cv::VideoCapture> cameras_;
    std::vector<std::thread> capture_threads_;
    std::vector<std::deque<cv::Mat>> frame_buffers_;
    std::vector<std::unique_ptr<std::mutex>> buffer_mutexes_;
    bool running_;
    double fps_;

    // Test images (static input)
    std::vector<cv::Mat> test_images_;

    // Timer for processing
    rclcpp::TimerBase::SharedPtr timer_;

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_rgb_panorama_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_panorama_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_inv_depth_panorama_;

    // Core processing
    std::unique_ptr<RGBD_Estimator> rgbd_estimator_;
    std::vector<DoubleSphereCalibration> calibrations_;
    at::Device device_;

    // Frame counter for logging
    int frame_count_;
};

} // namespace my_stereo_pkg

#endif // MY_STEREO_PKG__RGBD_PANORAMA_NODE_HPP_
