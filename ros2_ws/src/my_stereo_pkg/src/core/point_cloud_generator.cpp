/**
 * Point Cloud Generator Implementation
 */

#include "my_stereo_pkg/point_cloud_generator.hpp"
#include <rclcpp/rclcpp.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

namespace my_stereo_pkg {

// Forward declaration of CUDA kernel wrapper (defined in pointcloud_kernels.cu)
void launchPanoramaToPointCloudKernel(
    const float* distance,
    const unsigned char* rgb,
    const float* cos_lat,
    const float* sin_lat,
    float* points,
    bool* valid_mask,
    int width,
    int height,
    float min_depth,
    float max_depth,
    cudaStream_t stream
);


// ========================================
// Point Cloud Generator Implementation
// ========================================

PointCloudGenerator::PointCloudGenerator(
    int panorama_width,
    int panorama_height,
    const at::Device& device
)
    : panorama_width_(panorama_width),
      panorama_height_(panorama_height),
      device_(device)
{
    precomputeSphericalLuts();
    std::cout << "PointCloudGenerator initialized: " 
              << panorama_width_ << "x" << panorama_height_ << std::endl;
}


void PointCloudGenerator::precomputeSphericalLuts() {
    // Precompute longitude values on CPU first: θ = 2π * u / width
    auto longitude_cpu = torch::zeros({panorama_width_}, torch::dtype(torch::kFloat32));
    auto longitude_acc = longitude_cpu.accessor<float, 1>();
    for (int u = 0; u < panorama_width_; ++u) {
        longitude_acc[u] = (2.0f * M_PI * u) / panorama_width_;
    }
    longitude_ = longitude_cpu.to(device_);  // Transfer to GPU
    
    // Precompute latitude values on CPU: φ = π * (0.5 - v / height)
    // This maps: v=0 → φ=π/2 (top), v=height → φ=-π/2 (bottom)
    auto latitude_cpu = torch::zeros({panorama_height_}, torch::dtype(torch::kFloat32));
    auto cos_lat_cpu = torch::zeros({panorama_height_}, torch::dtype(torch::kFloat32));
    auto sin_lat_cpu = torch::zeros({panorama_height_}, torch::dtype(torch::kFloat32));
    
    auto latitude_acc = latitude_cpu.accessor<float, 1>();
    auto cos_lat_acc = cos_lat_cpu.accessor<float, 1>();
    auto sin_lat_acc = sin_lat_cpu.accessor<float, 1>();
    
    for (int v = 0; v < panorama_height_; ++v) {
        float phi = M_PI * (0.5f - static_cast<float>(v) / panorama_height_);
        latitude_acc[v] = phi;
        cos_lat_acc[v] = std::cos(phi);
        sin_lat_acc[v] = std::sin(phi);
    }
    
    // Transfer to GPU
    latitude_ = latitude_cpu.to(device_);
    cos_lat_ = cos_lat_cpu.to(device_);
    sin_lat_ = sin_lat_cpu.to(device_);
    
    std::cout << "  Spherical LUTs precomputed" << std::endl;
}


at::Tensor PointCloudGenerator::generate(
    const at::Tensor& rgb_panorama,
    const at::Tensor& distance_panorama,
    float min_depth,
    float max_depth
) {
    // Validate input tensors
    TORCH_CHECK(rgb_panorama.dim() == 3, "RGB panorama must be 3D [H, W, 3]");
    TORCH_CHECK(distance_panorama.dim() == 2, "Distance panorama must be 2D [H, W]");
    TORCH_CHECK(rgb_panorama.size(0) == panorama_height_, "RGB height mismatch");
    TORCH_CHECK(rgb_panorama.size(1) == panorama_width_, "RGB width mismatch");
    TORCH_CHECK(distance_panorama.size(0) == panorama_height_, "Distance height mismatch");
    TORCH_CHECK(distance_panorama.size(1) == panorama_width_, "Distance width mismatch");
    
    // Ensure tensors are on the correct device and contiguous
    auto rgb_device = rgb_panorama.to(device_).contiguous();
    auto distance_device = distance_panorama.to(device_).contiguous();
    
    int total_pixels = panorama_width_ * panorama_height_;
    
    // Allocate output tensors
    auto points = torch::zeros({total_pixels, 6}, 
                               torch::dtype(torch::kFloat32).device(device_));
    auto valid_mask = torch::zeros({total_pixels}, 
                                   torch::dtype(torch::kBool).device(device_));
    
    // Launch CUDA kernel using wrapper function
    launchPanoramaToPointCloudKernel(
        distance_device.data_ptr<float>(),
        rgb_device.data_ptr<unsigned char>(),
        cos_lat_.data_ptr<float>(),
        sin_lat_.data_ptr<float>(),
        points.data_ptr<float>(),
        valid_mask.data_ptr<bool>(),
        panorama_width_,
        panorama_height_,
        min_depth,
        max_depth,
        0  // default stream
    );
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error in panoramaToPointCloudKernel: ") + 
            cudaGetErrorString(err)
        );
    }
    
    // Filter out invalid points
    auto valid_points = points.index_select(0, valid_mask.nonzero().squeeze(1));
    
    return valid_points;
}


sensor_msgs::msg::PointCloud2 PointCloudGenerator::toRosPointCloud2(
    const at::Tensor& points,
    const std::string& frame_id,
    const rclcpp::Time& timestamp
) {
    // points: [N, 6] tensor where each row is [x, y, z, r, g, b]
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 6, 
                "Points must be [N, 6] tensor");
    
    // Move to CPU if needed
    auto points_cpu = points.cpu().contiguous();
    auto points_acc = points_cpu.accessor<float, 2>();
    
    int num_points = points_cpu.size(0);
    
    // Create PointCloud2 message
    sensor_msgs::msg::PointCloud2 cloud_msg;
    cloud_msg.header.stamp = timestamp;
    cloud_msg.header.frame_id = frame_id;
    
    // Set point cloud fields (XYZRGB)
    cloud_msg.height = 1;
    cloud_msg.width = num_points;
    cloud_msg.is_dense = false;
    cloud_msg.is_bigendian = false;
    
    // Define fields
    sensor_msgs::msg::PointField field;
    
    // X, Y, Z
    field.name = "x";
    field.offset = 0;
    field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field.count = 1;
    cloud_msg.fields.push_back(field);
    
    field.name = "y";
    field.offset = 4;
    cloud_msg.fields.push_back(field);
    
    field.name = "z";
    field.offset = 8;
    cloud_msg.fields.push_back(field);
    
    // RGB (packed as single float32)
    field.name = "rgb";
    field.offset = 12;
    field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field.count = 1;
    cloud_msg.fields.push_back(field);
    
    cloud_msg.point_step = 16;  // 4 floats * 4 bytes
    cloud_msg.row_step = cloud_msg.point_step * num_points;
    cloud_msg.data.resize(cloud_msg.row_step);
    
    // Fill data
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(cloud_msg, "rgb");
    
    for (int i = 0; i < num_points; ++i) {
        *iter_x = points_acc[i][0];
        *iter_y = points_acc[i][1];
        *iter_z = points_acc[i][2];
        
        // Pack RGB as uint32 (BGR order for RViz compatibility)
        uint8_t r = static_cast<uint8_t>(points_acc[i][3]);
        uint8_t g = static_cast<uint8_t>(points_acc[i][4]);
        uint8_t b = static_cast<uint8_t>(points_acc[i][5]);
        
        iter_r[0] = b; // Blue (RViz expects BGR)
        iter_r[1] = g; // Green  
        iter_r[2] = r; // Red 
        iter_r[3] = 0;
        
        ++iter_x;
        ++iter_y;
        ++iter_z;
        ++iter_r;
    }
    
    return cloud_msg;
}

} // namespace my_stereo_pkg
