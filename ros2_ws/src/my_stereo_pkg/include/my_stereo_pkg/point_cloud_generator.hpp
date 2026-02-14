/**
 * Point Cloud Generator for Omnidirectional SLAM
 * 
 * Converts panorama RGB-D data to 3D point cloud.
 * Uses equirectangular projection model to map (u,v) pixels to 3D (x,y,z) points.
 * 
 * Coordinate system:
 * - Panorama width → Longitude θ (0 to 2π)
 * - Panorama height → Latitude φ (-π/2 to π/2)
 * 
 * Conversion formulas:
 *   x = d * cos(φ) * sin(θ)
 *   y = d * sin(φ)
 *   z = d * cos(φ) * cos(θ)
 * 
 * where d is the estimated distance from the depth panorama.
 */

#pragma once

#include <torch/torch.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <vector>
#include <memory>

namespace my_stereo_pkg {

/**
 * Point Cloud Generator Class
 * 
 * Generates 3D point clouds from panoramic RGB-D images.
 * Uses CUDA kernels for efficient parallel processing.
 */
class PointCloudGenerator {
public:
    /**
     * Constructor
     * 
     * @param panorama_width Width of the panorama image
     * @param panorama_height Height of the panorama image
     * @param device Torch device (CUDA or CPU)
     */
    PointCloudGenerator(
        int panorama_width,
        int panorama_height,
        const at::Device& device
    );
    
    /**
     * Generate 3D point cloud from panorama RGB-D
     * 
     * @param rgb_panorama RGB panorama tensor [H, W, 3] (uint8)
     * @param distance_panorama Distance panorama tensor [H, W] (float32)
     * @param min_depth Minimum depth threshold
     * @param max_depth Maximum depth threshold
     * @return Point cloud tensor [N, 6] where each row is [x, y, z, r, g, b]
     */
    at::Tensor generate(
        const at::Tensor& rgb_panorama,
        const at::Tensor& distance_panorama,
        float min_depth = 0.1f,
        float max_depth = 100.0f
    );
    
    /**
     * Convert point cloud tensor to ROS 2 PointCloud2 message
     * 
     * @param points Point cloud tensor [N, 6] where each row is [x, y, z, r, g, b]
     * @param frame_id Frame ID for the point cloud
     * @param timestamp Timestamp for the message
     * @return ROS 2 PointCloud2 message
     */
    static sensor_msgs::msg::PointCloud2 toRosPointCloud2(
        const at::Tensor& points,
        const std::string& frame_id,
        const rclcpp::Time& timestamp
    );
    
private:
    int panorama_width_;
    int panorama_height_;
    at::Device device_;
    
    // Precomputed lookup tables for spherical conversion
    at::Tensor longitude_;  // [W] - θ values
    at::Tensor latitude_;   // [H] - φ values
    at::Tensor cos_lat_;    // [H] - cos(φ) values
    at::Tensor sin_lat_;    // [H] - sin(φ) values
    
    /**
     * Precompute spherical coordinate lookup tables
     */
    void precomputeSphericalLuts();
};

} // namespace my_stereo_pkg
