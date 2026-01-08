/**
 * @file Utils.cpp
 * @brief Implementation of utility classes and functions for sphere stereo vision
 */

#include "sphere_stereo_ros/Utils.hpp"

#include <fstream>
#include <cmath>
#include <iostream>
#include <algorithm>  // for std::clamp

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace sphere_stereo_ros {

// =============================================================================
// Calibration Implementation
// =============================================================================

Calibration::Calibration(const Vec2f& original_resolution,
                         const Vec2f& principal,
                         const Vec2f& focal_length,
                         float xi,
                         float alpha,
                         const Mat4f& rt,
                         const Vec2f& matching_scale)
    : original_resolution_(original_resolution)
    , principal_(principal)
    , focal_length_(focal_length)
    , xi_(xi)
    , alpha_(alpha)
    , rt_(rt)
    , matching_scale_(matching_scale)
{
}

Vec3f Calibration::unproject(const Vec2f& uv, bool& valid) const
{
    // Apply matching scale to get normalized coordinates
    const Vec2f scaled_principal = scaledPrincipal();
    const Vec2f scaled_fl = scaledFocalLength();
    
    // m_xy = (uv - principal * scale) / (fl * scale)
    const float mx = (uv.x() - scaled_principal.x()) / scaled_fl.x();
    const float my = (uv.y() - scaled_principal.y()) / scaled_fl.y();
    
    const float r2 = mx * mx + my * my;
    
    // Check validity: 1 - (2*alpha - 1) * r2 >= 0
    const float discriminant = 1.0f - (2.0f * alpha_ - 1.0f) * r2;
    valid = (discriminant >= 0.0f);
    
    if (!valid) {
        return Vec3f(0.0f, 0.0f, 0.0f);
    }
    
    // m_z = (1 - alpha^2 * r2) / (alpha * sqrt(max(discriminant, 0)) + 1 - alpha)
    const float sqrt_discriminant = std::sqrt(std::max(discriminant, 0.0f));
    const float mz = (1.0f - alpha_ * alpha_ * r2) / 
                     (alpha_ * sqrt_discriminant + 1.0f - alpha_);
    
    // Compute the 3D point on the unit sphere
    // scale = (mz * xi + sqrt(mz^2 + (1 - xi^2) * r2)) / (mz^2 + r2)
    const float mz2 = mz * mz;
    const float xi2 = xi_ * xi_;
    const float inner = mz2 + (1.0f - xi2) * r2;
    const float scale = (mz * xi_ + std::sqrt(std::max(inner, 0.0f))) / (mz2 + r2);
    
    Vec3f point(scale * mx, scale * my, scale * mz - xi_);
    
    return point;
}

Eigen::MatrixXf Calibration::unprojectBatch(const Eigen::MatrixXf& uv_array, 
                                             std::vector<bool>& valid_array) const
{
    const int n = static_cast<int>(uv_array.rows());
    Eigen::MatrixXf points(n, 3);
    valid_array.resize(n);
    
    const Vec2f scaled_principal = scaledPrincipal();
    const Vec2f scaled_fl = scaledFocalLength();
    
    for (int i = 0; i < n; ++i) {
        const float u = uv_array(i, 0);
        const float v = uv_array(i, 1);
        
        const float mx = (u - scaled_principal.x()) / scaled_fl.x();
        const float my = (v - scaled_principal.y()) / scaled_fl.y();
        
        const float r2 = mx * mx + my * my;
        const float discriminant = 1.0f - (2.0f * alpha_ - 1.0f) * r2;
        
        valid_array[i] = (discriminant >= 0.0f);
        
        if (!valid_array[i]) {
            points(i, 0) = 0.0f;
            points(i, 1) = 0.0f;
            points(i, 2) = 0.0f;
            continue;
        }
        
        const float sqrt_discriminant = std::sqrt(std::max(discriminant, 0.0f));
        const float mz = (1.0f - alpha_ * alpha_ * r2) / 
                         (alpha_ * sqrt_discriminant + 1.0f - alpha_);
        
        const float mz2 = mz * mz;
        const float xi2 = xi_ * xi_;
        const float inner = mz2 + (1.0f - xi2) * r2;
        const float scale = (mz * xi_ + std::sqrt(std::max(inner, 0.0f))) / (mz2 + r2);
        
        points(i, 0) = scale * mx;
        points(i, 1) = scale * my;
        points(i, 2) = scale * mz - xi_;
    }
    
    return points;
}

Vec2f Calibration::project(const Vec3f& point, bool& valid) const
{
    const float x = point.x();
    const float y = point.y();
    const float z = point.z();
    
    // d1 = ||point||
    const float d1 = point.norm();
    
    // c = xi * d1 + z
    const float c = xi_ * d1 + z;
    
    // d2 = ||(x, y, c)||
    const float d2 = std::sqrt(x * x + y * y + c * c);
    
    // norm = alpha * d2 + (1 - alpha) * c
    const float norm = alpha_ * d2 + (1.0f - alpha_) * c;
    
    // Validity check
    float w1, w2;
    if (alpha_ > 0.5f) {
        w1 = (1.0f - alpha_) / alpha_;
    } else {
        w1 = alpha_ / (1.0f - alpha_);
    }
    w2 = (w1 + xi_) / std::sqrt(2.0f * w1 * xi_ + xi_ * xi_ + 1.0f);
    
    valid = (z > -w2 * d1);
    
    if (!valid || std::abs(norm) < 1e-8f) {
        return Vec2f(0.0f, 0.0f);
    }
    
    // Apply matching scale
    const Vec2f scaled_principal = scaledPrincipal();
    const Vec2f scaled_fl = scaledFocalLength();
    
    // uv = fl * scale * (x, y) / norm + principal * scale
    const float u = scaled_fl.x() * x / norm + scaled_principal.x();
    const float v = scaled_fl.y() * y / norm + scaled_principal.y();
    
    return Vec2f(u, v);
}

Eigen::MatrixXf Calibration::projectBatch(const Eigen::MatrixXf& points_array, 
                                           std::vector<bool>& valid_array) const
{
    const int n = static_cast<int>(points_array.rows());
    Eigen::MatrixXf uv(n, 2);
    valid_array.resize(n);
    
    const Vec2f scaled_principal = scaledPrincipal();
    const Vec2f scaled_fl = scaledFocalLength();
    
    // Precompute w2 for validity check
    float w1, w2;
    if (alpha_ > 0.5f) {
        w1 = (1.0f - alpha_) / alpha_;
    } else {
        w1 = alpha_ / (1.0f - alpha_);
    }
    w2 = (w1 + xi_) / std::sqrt(2.0f * w1 * xi_ + xi_ * xi_ + 1.0f);
    
    for (int i = 0; i < n; ++i) {
        const float x = points_array(i, 0);
        const float y = points_array(i, 1);
        const float z = points_array(i, 2);
        
        const float d1 = std::sqrt(x * x + y * y + z * z);
        const float c = xi_ * d1 + z;
        const float d2 = std::sqrt(x * x + y * y + c * c);
        const float norm = alpha_ * d2 + (1.0f - alpha_) * c;
        
        valid_array[i] = (z > -w2 * d1) && (std::abs(norm) >= 1e-8f);
        
        if (!valid_array[i]) {
            uv(i, 0) = 0.0f;
            uv(i, 1) = 0.0f;
            continue;
        }
        
        uv(i, 0) = scaled_fl.x() * x / norm + scaled_principal.x();
        uv(i, 1) = scaled_fl.y() * y / norm + scaled_principal.y();
    }
    
    return uv;
}

// =============================================================================
// CalibrationSet Implementation
// =============================================================================

CalibrationSet::CalibrationSet(const std::string& json_path, 
                               const Vec2f& matching_resolution)
{
    // Read JSON file
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open calibration file: " + json_path);
    }
    
    nlohmann::json root;
    try {
        file >> root;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("JSON parse error in " + json_path + ": " + e.what());
    }
    
    // Parse basalt format: root["value0"] contains the actual data
    const auto& data = root.contains("value0") ? root["value0"] : root;
    
    if (!data.contains("T_imu_cam") || !data.contains("intrinsics") || !data.contains("resolution")) {
        throw std::runtime_error("Invalid calibration format: missing required fields");
    }
    
    const auto& extrinsics_array = data["T_imu_cam"];
    const auto& intrinsics_array = data["intrinsics"];
    const auto& resolution_array = data["resolution"];
    
    const size_t num_cameras = extrinsics_array.size();
    if (intrinsics_array.size() != num_cameras || resolution_array.size() != num_cameras) {
        throw std::runtime_error("Calibration arrays have inconsistent sizes");
    }
    
    calibrations_.reserve(num_cameras);
    
    for (size_t i = 0; i < num_cameras; ++i) {
        const auto& extrinsics = extrinsics_array[i];
        const auto& intrinsics = intrinsics_array[i];
        const auto& resolution = resolution_array[i];
        
        // Check camera type
        const std::string camera_type = intrinsics["camera_type"];
        if (camera_type != "ds") {
            throw std::runtime_error("Unsupported camera model: " + camera_type + 
                                     ". Only 'ds' (double sphere) is supported.");
        }
        
        // Parse intrinsics
        const auto& cam_params = intrinsics["intrinsics"];
        const float fx = cam_params["fx"].get<float>();
        const float fy = cam_params["fy"].get<float>();
        const float cx = cam_params["cx"].get<float>();
        const float cy = cam_params["cy"].get<float>();
        const float xi = cam_params["xi"].get<float>();
        const float alpha = cam_params["alpha"].get<float>();
        
        // Parse resolution
        const float orig_width = resolution[0].get<float>();
        const float orig_height = resolution[1].get<float>();
        
        // Parse extrinsics (quaternion + position)
        const float qx = extrinsics["qx"].get<float>();
        const float qy = extrinsics["qy"].get<float>();
        const float qz = extrinsics["qz"].get<float>();
        const float qw = extrinsics["qw"].get<float>();
        const float px = extrinsics["px"].get<float>();
        const float py = extrinsics["py"].get<float>();
        const float pz = extrinsics["pz"].get<float>();
        
        // Build transformation matrix
        Mat4f rt = poseToMatrix(px, py, pz, qx, qy, qz, qw);
        
        // Compute matching scale
        Vec2f matching_scale(
            matching_resolution.x() / orig_width,
            matching_resolution.y() / orig_height
        );
        
        calibrations_.emplace_back(
            Vec2f(orig_width, orig_height),
            Vec2f(cx, cy),
            Vec2f(fx, fy),
            xi,
            alpha,
            rt,
            matching_scale
        );
    }
}

const Calibration& CalibrationSet::at(size_t index) const
{
    if (index >= calibrations_.size()) {
        throw std::out_of_range("Camera index out of range: " + std::to_string(index));
    }
    return calibrations_[index];
}

// =============================================================================
// Color Conversion Implementation
// =============================================================================

cv::Mat rgb2YCbCr(const cv::Mat& rgb)
{
    if (rgb.empty()) {
        throw std::runtime_error("rgb2YCbCr: Input image is empty");
    }
    
    cv::Mat result;
    
    if (rgb.type() == CV_8UC3) {
        // Allocate output
        result = cv::Mat(rgb.rows, rgb.cols, CV_8UC3);
        
        for (int y = 0; y < rgb.rows; ++y) {
            const uchar* rgb_row = rgb.ptr<uchar>(y);
            uchar* ycbcr_row = result.ptr<uchar>(y);
            
            for (int x = 0; x < rgb.cols; ++x) {
                // OpenCV uses BGR order
                const float b = static_cast<float>(rgb_row[x * 3 + 0]);
                const float g = static_cast<float>(rgb_row[x * 3 + 1]);
                const float r = static_cast<float>(rgb_row[x * 3 + 2]);
                
                // Y'CbCr conversion (ITU-R BT.601)
                float Y  = std::clamp(16.0f  + 0.1826f * r + 0.6142f * g + 0.062f  * b, 16.0f, 235.0f);
                float Cb = std::clamp(128.0f - 0.1006f * r - 0.3386f * g + 0.4392f * b, 16.0f, 240.0f);
                float Cr = std::clamp(128.0f + 0.4392f * r - 0.3989f * g - 0.0403f * b, 16.0f, 240.0f);
                
                ycbcr_row[x * 3 + 0] = static_cast<uchar>(Y);
                ycbcr_row[x * 3 + 1] = static_cast<uchar>(Cb);
                ycbcr_row[x * 3 + 2] = static_cast<uchar>(Cr);
            }
        }
    }
    else if (rgb.type() == CV_32FC3) {
        result = cv::Mat(rgb.rows, rgb.cols, CV_32FC3);
        
        for (int y = 0; y < rgb.rows; ++y) {
            const float* rgb_row = rgb.ptr<float>(y);
            float* ycbcr_row = result.ptr<float>(y);
            
            for (int x = 0; x < rgb.cols; ++x) {
                const float b = rgb_row[x * 3 + 0];
                const float g = rgb_row[x * 3 + 1];
                const float r = rgb_row[x * 3 + 2];
                
                float Y  = std::clamp(16.0f  + 0.1826f * r + 0.6142f * g + 0.062f  * b, 16.0f, 235.0f);
                float Cb = std::clamp(128.0f - 0.1006f * r - 0.3386f * g + 0.4392f * b, 16.0f, 240.0f);
                float Cr = std::clamp(128.0f + 0.4392f * r - 0.3989f * g - 0.0403f * b, 16.0f, 240.0f);
                
                ycbcr_row[x * 3 + 0] = Y;
                ycbcr_row[x * 3 + 1] = Cb;
                ycbcr_row[x * 3 + 2] = Cr;
            }
        }
    }
    else {
        throw std::runtime_error("rgb2YCbCr: Unsupported image type " + std::to_string(rgb.type()));
    }
    
    return result;
}

// Note: GPU version requires OpenCV with CUDA support
// Uncomment when OpenCV CUDA modules are available
// void rgb2YCbCrGpu(const cv::cuda::GpuMat& rgb, cv::cuda::GpuMat& ycbcr)
// {
//     // For now, download to CPU, convert, and upload
//     // TODO: Implement CUDA kernel for this conversion
//     cv::Mat cpu_rgb;
//     rgb.download(cpu_rgb);
//     
//     cv::Mat cpu_ycbcr = rgb2YCbCr(cpu_rgb);
//     ycbcr.upload(cpu_ycbcr);
// }

// =============================================================================
// Image I/O Implementation
// =============================================================================

bool readInputImages(const std::string& filename,
                     const std::string& dataset_path,
                     const cv::Size& matching_resolution,
                     const cv::Size& stitch_resolution,
                     const CalibrationSet& calibrations,
                     const std::vector<int>& reference_indices,
                     std::vector<cv::Mat>& images_to_match,
                     std::vector<cv::Mat>& images_to_stitch)
{
    images_to_match.clear();
    images_to_stitch.clear();
    
    const size_t num_cameras = calibrations.numCameras();
    images_to_match.reserve(num_cameras);
    
    for (size_t cam_index = 0; cam_index < num_cameras; ++cam_index) {
        const auto& calib = calibrations[cam_index];
        
        // Build file path
        std::string file_path = dataset_path + "/cam" + std::to_string(cam_index) + "/" + filename;
        
        // Read image
        cv::Mat image = cv::imread(file_path, cv::IMREAD_UNCHANGED);
        
        if (image.empty()) {
            std::cerr << "Warning: Cannot read image: " << file_path << std::endl;
            return false;
        }
        
        // Check resolution
        const Vec2f& orig_res = calib.originalResolution();
        if (image.cols != static_cast<int>(orig_res.x()) || 
            image.rows != static_cast<int>(orig_res.y()) ||
            image.channels() != 3) {
            std::cerr << "Warning: Invalid image size/channels for: " << file_path << std::endl;
            return false;
        }
        
        // Convert to float32 [0, 255]
        cv::Mat float_image;
        if (image.depth() == CV_8U) {
            image.convertTo(float_image, CV_32FC3);
        }
        else if (image.depth() == CV_16U) {
            image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);
        }
        else if (image.depth() == CV_32F) {
            double min_val, max_val;
            cv::minMaxLoc(image.reshape(1), &min_val, &max_val);
            if (max_val > 1.0) {
                std::cerr << "Warning: Float image has out-of-range values, clipping: " << file_path << std::endl;
            }
            image.convertTo(float_image, CV_32FC3, 255.0);
            float_image = cv::min(float_image, 255.0f);
            float_image = cv::max(float_image, 0.0f);
        }
        else {
            std::cerr << "Warning: Invalid image type for: " << file_path << std::endl;
            return false;
        }
        
        // Check if this is a reference camera for stitching
        bool is_reference = std::find(reference_indices.begin(), reference_indices.end(), 
                                       static_cast<int>(cam_index)) != reference_indices.end();
        if (is_reference) {
            cv::Mat image_to_stitch;
            cv::resize(float_image, image_to_stitch, stitch_resolution, 0, 0, cv::INTER_AREA);
            images_to_stitch.push_back(image_to_stitch);
        }
        
        // Resize for matching
        cv::Mat image_to_match;
        cv::resize(float_image, image_to_match, matching_resolution, 0, 0, cv::INTER_AREA);
        images_to_match.push_back(image_to_match);
    }
    
    return true;
}

// =============================================================================
// Math Utilities Implementation
// =============================================================================

Mat3f quaternionToRotationMatrix(float qx, float qy, float qz, float qw)
{
    // Use Eigen's quaternion class for robust conversion
    Quatf q(qw, qx, qy, qz);  // Eigen uses (w, x, y, z) order
    q.normalize();
    return q.toRotationMatrix();
}

Mat4f poseToMatrix(float px, float py, float pz, 
                   float qx, float qy, float qz, float qw)
{
    Mat4f rt = Mat4f::Identity();
    rt.block<3, 3>(0, 0) = quaternionToRotationMatrix(qx, qy, qz, qw);
    rt(0, 3) = px;
    rt(1, 3) = py;
    rt(2, 3) = pz;
    return rt;
}

}  // namespace sphere_stereo_ros
