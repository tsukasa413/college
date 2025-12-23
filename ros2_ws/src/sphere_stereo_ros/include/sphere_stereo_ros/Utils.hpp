/**
 * @file Utils.hpp
 * @brief Utility classes and functions for sphere stereo vision
 * 
 * This is a C++ port of the Python implementation from:
 * "Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images"
 * Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
 * CVPR 2021
 * 
 * @copyright CC BY-NC-SA 3.0

// CUDA Error checking macro
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

 */

#pragma once

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>

namespace sphere_stereo_ros {

// =============================================================================
// Type Aliases
// =============================================================================
using Vec2f = Eigen::Vector2f;
using Vec3f = Eigen::Vector3f;
using Vec4f = Eigen::Vector4f;
using Mat3f = Eigen::Matrix3f;
using Mat4f = Eigen::Matrix4f;
using Quatf = Eigen::Quaternionf;

// =============================================================================
// Constants
// =============================================================================
constexpr int kNumCameras = 4;  ///< Default number of fisheye cameras

// =============================================================================
// Calibration Class
// =============================================================================
/**
 * @brief Double Sphere camera model calibration data
 * 
 * Based on "The Double Sphere Camera Model" (https://arxiv.org/abs/1807.08957)
 * Stores intrinsic parameters (fx, fy, cx, cy, xi, alpha) and extrinsic pose (RT matrix).
 */
class Calibration {
public:
    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------
    
    /**
     * @brief Default constructor
     */
    Calibration() = default;

    /**
     * @brief Construct calibration from parameters
     * 
     * @param original_resolution Original image resolution [width, height]
     * @param principal Principal point [cx, cy]
     * @param focal_length Focal length [fx, fy]
     * @param xi Double sphere xi parameter
     * @param alpha Double sphere alpha parameter
     * @param rt 4x4 camera pose matrix (camera to world)
     * @param matching_scale Scale factor for matching resolution [scale_x, scale_y]
     */
    Calibration(const Vec2f& original_resolution,
                const Vec2f& principal,
                const Vec2f& focal_length,
                float xi,
                float alpha,
                const Mat4f& rt,
                const Vec2f& matching_scale);

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------
    
    const Vec2f& originalResolution() const { return original_resolution_; }
    const Vec2f& principal() const { return principal_; }
    const Vec2f& focalLength() const { return focal_length_; }
    float xi() const { return xi_; }
    float alpha() const { return alpha_; }
    const Mat4f& rt() const { return rt_; }
    const Vec2f& matchingScale() const { return matching_scale_; }
    
    /**
     * @brief Get scaled principal point for matching resolution
     */
    Vec2f scaledPrincipal() const { 
        return Vec2f(principal_.x() * matching_scale_.x(), 
                     principal_.y() * matching_scale_.y()); 
    }
    
    /**
     * @brief Get scaled focal length for matching resolution
     */
    Vec2f scaledFocalLength() const { 
        return Vec2f(focal_length_.x() * matching_scale_.x(), 
                     focal_length_.y() * matching_scale_.y()); 
    }

    // -------------------------------------------------------------------------
    // Projection / Unprojection
    // -------------------------------------------------------------------------
    
    /**
     * @brief Unproject a 2D pixel to a 3D point on the unit sphere
     * 
     * Implements the Double Sphere unprojection model.
     * 
     * @param uv Pixel coordinates (in matching resolution)
     * @param[out] valid Whether the unprojection is valid
     * @return 3D point on unit sphere (or invalid if outside valid region)
     */
    Vec3f unproject(const Vec2f& uv, bool& valid) const;
    
    /**
     * @brief Unproject multiple pixels to 3D points on the unit sphere
     * 
     * @param uv_array Array of pixel coordinates (Nx2 matrix)
     * @param[out] valid_array Array of validity flags
     * @return Nx3 matrix of 3D points
     */
    Eigen::MatrixXf unprojectBatch(const Eigen::MatrixXf& uv_array, 
                                    std::vector<bool>& valid_array) const;
    
    /**
     * @brief Project a 3D point to 2D pixel coordinates
     * 
     * Implements the Double Sphere projection model.
     * 
     * @param point 3D point in camera coordinates
     * @param[out] valid Whether the projection is valid (point is in front of camera)
     * @return 2D pixel coordinates (in matching resolution)
     */
    Vec2f project(const Vec3f& point, bool& valid) const;
    
    /**
     * @brief Project multiple 3D points to 2D pixel coordinates
     * 
     * @param points_array Nx3 matrix of 3D points
     * @param[out] valid_array Array of validity flags
     * @return Nx2 matrix of pixel coordinates
     */
    Eigen::MatrixXf projectBatch(const Eigen::MatrixXf& points_array, 
                                  std::vector<bool>& valid_array) const;

private:
    Vec2f original_resolution_{0.0f, 0.0f};
    Vec2f principal_{0.0f, 0.0f};
    Vec2f focal_length_{0.0f, 0.0f};
    float xi_{0.0f};
    float alpha_{0.0f};
    Mat4f rt_{Mat4f::Identity()};
    Vec2f matching_scale_{1.0f, 1.0f};
};

// =============================================================================
// CalibrationSet Class
// =============================================================================
/**
 * @brief Container for multiple camera calibrations
 * 
 * Loads and manages calibration data for multiple fisheye cameras
 * from a JSON configuration file (basalt format).
 */
class CalibrationSet {
public:
    /**
     * @brief Construct calibration set from JSON file
     * 
     * @param json_path Path to calibration JSON file
     * @param matching_resolution Target resolution for depth estimation [width, height]
     * @throws std::runtime_error if file cannot be read or parsed
     */
    explicit CalibrationSet(const std::string& json_path, 
                            const Vec2f& matching_resolution);
    
    /**
     * @brief Get number of cameras
     */
    size_t numCameras() const { return calibrations_.size(); }
    
    /**
     * @brief Get calibration for a specific camera
     * @param index Camera index
     * @throws std::out_of_range if index is invalid
     */
    const Calibration& at(size_t index) const;
    
    /**
     * @brief Get calibration for a specific camera (operator overload)
     */
    const Calibration& operator[](size_t index) const { return calibrations_[index]; }
    
    /**
     * @brief Get all calibrations
     */
    const std::vector<Calibration>& calibrations() const { return calibrations_; }

private:
    std::vector<Calibration> calibrations_;
};

// =============================================================================
// Color Conversion Utilities
// =============================================================================
/**
 * @brief Convert RGB to YCbCr color space
 * 
 * @param rgb Input RGB image (CV_8UC3 or CV_32FC3)
 * @return YCbCr image (same type as input)
 */
cv::Mat rgb2YCbCr(const cv::Mat& rgb);

// Note: GPU version requires OpenCV with CUDA support
// Uncomment when OpenCV CUDA modules are available
// /**
//  * @brief Convert RGB to YCbCr color space (in-place on GPU Mat)
//  * 
//  * @param rgb Input RGB GpuMat
//  * @param ycbcr Output YCbCr GpuMat
//  */
// void rgb2YCbCrGpu(const cv::cuda::GpuMat& rgb, cv::cuda::GpuMat& ycbcr);

// =============================================================================
// Image I/O Utilities
// =============================================================================
/**
 * @brief Read and preprocess input fisheye images
 * 
 * @param filename Image filename
 * @param dataset_path Base path to dataset
 * @param matching_resolution Target resolution for matching
 * @param stitch_resolution Target resolution for stitching (RGB output)
 * @param calibrations Camera calibrations
 * @param reference_indices Indices of reference cameras for stitching
 * @param[out] images_to_match Output images resized for matching
 * @param[out] images_to_stitch Output images resized for stitching
 * @return true if all images were read successfully
 */
bool readInputImages(const std::string& filename,
                     const std::string& dataset_path,
                     const cv::Size& matching_resolution,
                     const cv::Size& stitch_resolution,
                     const CalibrationSet& calibrations,
                     const std::vector<int>& reference_indices,
                     std::vector<cv::Mat>& images_to_match,
                     std::vector<cv::Mat>& images_to_stitch);

// =============================================================================
// Math Utilities
// =============================================================================
/**
 * @brief Create a rotation matrix from quaternion (qx, qy, qz, qw)
 */
Mat3f quaternionToRotationMatrix(float qx, float qy, float qz, float qw);

/**
 * @brief Create a 4x4 transformation matrix from position and quaternion
 */
Mat4f poseToMatrix(float px, float py, float pz, 
                   float qx, float qy, float qz, float qw);

}  // namespace sphere_stereo_ros
