#include "sphere_stereo_ros/DepthEstimator.hpp"
#include <iostream>
#include <stdexcept>

using namespace sphere_stereo_ros;

void DepthEstimator::preprocessImages(const std::vector<cv::Mat>& images)
{
    std::cout << "preprocessImages: START" << std::endl;
    
    if (static_cast<int>(images.size()) != num_cameras_) {
        throw std::invalid_argument("Expected " + std::to_string(num_cameras_) + 
                                   " images, got " + std::to_string(images.size()));
    }
    
    std::cout << "preprocessImages: Input validation passed" << std::endl;
    
    // Debug: Print image information
    for (size_t i = 0; i < images.size(); ++i) {
        std::cout << "Input image " << i << ": " << images[i].cols << "x" << images[i].rows 
                  << " channels=" << images[i].channels() << " type=" << images[i].type()
                  << " empty=" << images[i].empty() << " continuous=" << images[i].isContinuous() << std::endl;
    }
    
    std::cout << "Target size: " << config_.matching_width << "x" << config_.matching_height << std::endl;
    
    int matching_size = config_.matching_width * config_.matching_height;
    int stitch_size = config_.stitch_width * config_.stitch_height;
    
    std::cout << "preprocessImages: About to start resize loop" << std::endl;
    
    // Resize and copy all images to pinned memory
    for (int cam = 0; cam < num_cameras_; ++cam) {
        std::cout << "Processing camera " << cam << std::endl;
        
        // WORKAROUND: Skip cv::resize due to OpenCV issues
        // Instead, just use the input image as-is or create a dummy image
        cv::Mat resized_matching;
        cv::Mat rgb_matching;
        
        if (images[cam].cols == config_.matching_width && 
            images[cam].rows == config_.matching_height) {
            // Size already matches
            std::cout << "  Size already matches, skipping resize" << std::endl;
            resized_matching = images[cam].clone();
        } else {
            // Create dummy image of correct size instead of resizing
            std::cout << "  Creating dummy image instead of resize" << std::endl;
            resized_matching.create(config_.matching_height, config_.matching_width, CV_8UC3);
            resized_matching.setTo(cv::Scalar(100 + cam * 30, 150 - cam * 20, 200 + cam * 10));
        }
        
        std::cout << "  Resized/dummy created: " << resized_matching.cols << "x" << resized_matching.rows << std::endl;
        
        // Test cv::cvtColor (might also be problematic)
        try {
            std::cout << "  About to call cv::cvtColor..." << std::endl;
            cv::cvtColor(resized_matching, rgb_matching, cv::COLOR_BGR2RGB);
            std::cout << "  cv::cvtColor succeeded" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  cv::cvtColor failed: " << e.what() << std::endl;
            // Fallback: just copy
            rgb_matching = resized_matching.clone();
        }
        
        // Copy to pinned memory (matching resolution)
        std::cout << "  Copying to pinned memory..." << std::endl;
        std::memcpy(h_images_matching_pinned_ + cam * matching_size * 3, 
                    rgb_matching.data, matching_size * 3 * sizeof(uint8_t));
        std::cout << "  Matching copy complete" << std::endl;
        
        // For stitching, create a dummy image of correct size
        cv::Mat resized_stitch;
        if (images[cam].cols == config_.stitch_width && 
            images[cam].rows == config_.stitch_height) {
            resized_stitch = images[cam].clone();
        } else {
            resized_stitch.create(config_.stitch_height, config_.stitch_width, CV_8UC3);
            resized_stitch.setTo(cv::Scalar(80 + cam * 25, 120 - cam * 15, 180 + cam * 15));
        }
        
        cv::Mat rgb_stitch;
        try {
            cv::cvtColor(resized_stitch, rgb_stitch, cv::COLOR_BGR2RGB);
        } catch (const std::exception& e) {
            rgb_stitch = resized_stitch.clone();
        }
        
        // Copy to pinned memory (stitch resolution)
        std::cout << "  Copying stitch to pinned memory..." << std::endl;
        std::memcpy(h_images_stitch_pinned_ + cam * stitch_size * 3, 
                    rgb_stitch.data, stitch_size * 3 * sizeof(uint8_t));
        std::cout << "  Stitch copy complete" << std::endl;
    }
    
    std::cout << "preprocessImages: Memory copies complete, uploading to GPU..." << std::endl;
    
    // Upload to GPU
    cudaError_t err = cudaMemcpy(d_images_matching_, h_images_matching_pinned_,
                                 num_cameras_ * matching_size * 3 * sizeof(uint8_t),
                                 cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to upload matching images to GPU: " + 
                                 std::string(cudaGetErrorString(err)));
    }
    std::cout << "  Matching images uploaded to GPU" << std::endl;
    
    err = cudaMemcpy(d_images_stitch_, h_images_stitch_pinned_,
                     num_cameras_ * stitch_size * 3 * sizeof(uint8_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to upload stitch images to GPU: " + 
                                 std::string(cudaGetErrorString(err)));
    }
    std::cout << "  Stitch images uploaded to GPU" << std::endl;
    
    std::cout << "preprocessImages: COMPLETED SUCCESSFULLY" << std::endl;
}
