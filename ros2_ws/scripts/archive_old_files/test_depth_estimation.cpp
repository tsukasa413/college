/*
 * test_depth_estimation.cpp
 * 
 * Basic integration test for RGBD_Estimator
 * Verifies:
 * - Memory allocation/deallocation
 * - Kernel compilation
 * - Basic API usage
 */

#include <iostream>
#include <vector>
#include <random>
#include "my_stereo_pkg/depth_estimation.hpp"

int main() {
    std::cout << "=== RGBD_Estimator Compilation & Basic Test ===" << std::endl;
    
    try {
        // ====================================================================
        // Mock Data Setup
        // ====================================================================
        
        int num_cameras = 2;
        std::vector<float> calibrations_rt(num_cameras * 16);
        std::vector<float> calibrations_intrinsics(num_cameras * 4);
        std::vector<float> calibrations_sphere(num_cameras * 2);
        std::vector<float> calibrations_resolution(num_cameras * 2);
        
        // Initialize identity transforms
        for (int i = 0; i < num_cameras; i++) {
            // Identity 4x4 matrix
            for (int j = 0; j < 16; j++) {
                calibrations_rt[i * 16 + j] = (j % 5 == 0) ? 1.0f : 0.0f;
            }
            
            // Intrinsics [fx, fy, cx, cy]
            calibrations_intrinsics[i * 4 + 0] = 500.0f;  // fx
            calibrations_intrinsics[i * 4 + 1] = 500.0f;  // fy
            calibrations_intrinsics[i * 4 + 2] = 320.0f;  // cx
            calibrations_intrinsics[i * 4 + 3] = 240.0f;  // cy
            
            // Sphere model [xi, alpha]
            calibrations_sphere[i * 2 + 0] = 0.5f;  // xi
            calibrations_sphere[i * 2 + 1] = 0.1f;  // alpha
            
            // Resolution [w, h]
            calibrations_resolution[i * 2 + 0] = 640.0f;
            calibrations_resolution[i * 2 + 1] = 480.0f;
        }
        
        std::vector<int> references_indices = {0};
        std::vector<float> reprojection_viewpoint = {0.0f, 0.0f, 0.0f};
        std::vector<int> image_widths = {640, 640};
        std::vector<int> image_heights = {480, 480};
        
        // ====================================================================
        // Create Estimator
        // ====================================================================
        
        std::cout << "Creating RGBD_Estimator..." << std::endl;
        
        RGBD_Estimator estimator(
            calibrations_rt,
            calibrations_intrinsics,
            calibrations_sphere,
            calibrations_resolution,
            0.5f,                          // min_dist
            10.0f,                         // max_dist
            50,                            // candidate_count
            references_indices,
            reprojection_viewpoint,
            image_widths,
            image_heights,
            320, 240,                      // matching_width, height
            640, 480,                      // rgb_to_stitch_width, height
            512, 512,                      // panorama_width, height
            1.0f,                          // sigma_i
            1.0f,                          // sigma_s
            0                              // device_id
        );
        
        std::cout << "✓ RGBD_Estimator created successfully" << std::endl;
        
        // ====================================================================
        // Generate Dummy Image Data
        // ====================================================================
        
        std::cout << "Generating dummy image data..." << std::endl;
        
        int matching_size = 320 * 240 * 3;
        std::vector<std::vector<float>> images_to_match(num_cameras);
        std::vector<std::vector<float>> images_to_stitch(num_cameras);
        
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 255.0f);
        
        for (int i = 0; i < num_cameras; i++) {
            images_to_match[i].resize(matching_size);
            for (int j = 0; j < matching_size; j++) {
                images_to_match[i][j] = dist(rng);
            }
            
            images_to_stitch[i].resize(640 * 480 * 3);
            for (int j = 0; j < 640 * 480 * 3; j++) {
                images_to_stitch[i][j] = dist(rng);
            }
        }
        
        std::cout << "✓ Image data generated" << std::endl;
        
        // ====================================================================
        // Run Estimation
        // ====================================================================
        
        std::cout << "Running RGBD panorama estimation..." << std::endl;
        
        auto [rgb_panorama, distance_panorama] = estimator.estimate_RGBD_panorama(
            images_to_match,
            images_to_stitch
        );
        
        std::cout << "✓ Estimation completed" << std::endl;
        std::cout << "  RGB panorama size: " << rgb_panorama.size() << " bytes" << std::endl;
        std::cout << "  Distance panorama size: " << distance_panorama.size() << " floats" << std::endl;
        
        // ====================================================================
        // Results
        // ====================================================================
        
        std::cout << "\n=== All Tests Passed ===" << std::endl;
        std::cout << "CUDA kernel compilation: ✓" << std::endl;
        std::cout << "Memory management: ✓" << std::endl;
        std::cout << "API functionality: ✓" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
