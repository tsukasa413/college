/**
 * @file test_utils.cpp
 * @brief Test program for Utils.hpp/cpp functionality
 * 
 * This standalone test verifies:
 * - JSON calibration loading
 * - Camera parameter parsing
 * - Project/unproject round-trip consistency
 */

#include "sphere_stereo_ros/Utils.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <string>

using namespace sphere_stereo_ros;

// ANSI color codes for output
#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define RESET "\033[0m"

void printTestResult(const std::string& test_name, bool passed) {
    if (passed) {
        std::cout << GREEN << "[PASS] " << RESET << test_name << std::endl;
    } else {
        std::cout << RED << "[FAIL] " << RESET << test_name << std::endl;
    }
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "  sphere_stereo_ros Utils Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // Parse command line arguments
    std::string calibration_path;
    if (argc > 1) {
        calibration_path = argv[1];
    } else {
        // Default path (relative to typical ROS 2 workspace structure)
        calibration_path = "install/sphere_stereo_ros/share/sphere_stereo_ros/config/calibration.json";
        std::cout << YELLOW << "[INFO] " << RESET 
                  << "No calibration path provided. Using default: " << calibration_path << std::endl;
    }

    // Matching resolution for testing (same as original Python code default)
    Vec2f matching_resolution(608.0f, 608.0f);
    
    int tests_passed = 0;
    int tests_total = 0;

    // =========================================================================
    // Test 1: Load calibration from JSON
    // =========================================================================
    std::cout << "\n--- Test 1: JSON Calibration Loading ---" << std::endl;
    tests_total++;
    
    std::unique_ptr<CalibrationSet> calib_set;
    try {
        calib_set = std::make_unique<CalibrationSet>(calibration_path, matching_resolution);
        
        std::cout << "  Number of cameras: " << calib_set->numCameras() << std::endl;
        
        if (calib_set->numCameras() > 0) {
            const auto& cam0 = calib_set->at(0);
            std::cout << "  Camera 0 parameters:" << std::endl;
            std::cout << "    - fx: " << cam0.focalLength().x() << std::endl;
            std::cout << "    - fy: " << cam0.focalLength().y() << std::endl;
            std::cout << "    - cx: " << cam0.principal().x() << std::endl;
            std::cout << "    - cy: " << cam0.principal().y() << std::endl;
            std::cout << "    - xi: " << cam0.xi() << std::endl;
            std::cout << "    - alpha: " << cam0.alpha() << std::endl;
            std::cout << "    - matching_scale: (" << cam0.matchingScale().x() 
                      << ", " << cam0.matchingScale().y() << ")" << std::endl;
            
            printTestResult("JSON calibration loading", true);
            tests_passed++;
        } else {
            printTestResult("JSON calibration loading (no cameras found)", false);
        }
    } catch (const std::exception& e) {
        std::cerr << RED << "  Error: " << e.what() << RESET << std::endl;
        printTestResult("JSON calibration loading", false);
        std::cerr << "\nAborting remaining tests due to calibration load failure." << std::endl;
        return 1;
    }

    // =========================================================================
    // Test 2: Project a 3D point
    // =========================================================================
    std::cout << "\n--- Test 2: 3D Point Projection ---" << std::endl;
    tests_total++;
    
    const auto& cam0 = calib_set->at(0);
    Vec3f test_point(0.0f, 0.0f, 1.0f);  // Point directly in front (1m away)
    
    std::cout << "  Input 3D point: (" << test_point.x() << ", " 
              << test_point.y() << ", " << test_point.z() << ")" << std::endl;
    
    bool proj_valid = false;
    Vec2f projected = cam0.project(test_point, proj_valid);
    
    std::cout << "  Projected pixel: (" << projected.x() << ", " << projected.y() << ")" << std::endl;
    std::cout << "  Projection valid: " << (proj_valid ? "yes" : "no") << std::endl;
    
    // Expected: should project near principal point (scaled)
    Vec2f expected_center = cam0.scaledPrincipal();
    float proj_error = (projected - expected_center).norm();
    std::cout << "  Expected (principal point): (" << expected_center.x() << ", " 
              << expected_center.y() << ")" << std::endl;
    std::cout << "  Projection error from center: " << proj_error << " pixels" << std::endl;
    
    bool proj_test_passed = proj_valid && (proj_error < 1.0f);  // Should be very close to center
    printTestResult("3D point projection", proj_test_passed);
    if (proj_test_passed) tests_passed++;

    // =========================================================================
    // Test 3: Unproject back to 3D
    // =========================================================================
    std::cout << "\n--- Test 3: Pixel Unprojection ---" << std::endl;
    tests_total++;
    
    bool unproj_valid = false;
    Vec3f unprojected = cam0.unproject(projected, unproj_valid);
    
    std::cout << "  Unprojected 3D: (" << unprojected.x() << ", " 
              << unprojected.y() << ", " << unprojected.z() << ")" << std::endl;
    std::cout << "  Unprojection valid: " << (unproj_valid ? "yes" : "no") << std::endl;
    
    // Normalize for comparison (unproject gives unit vector direction)
    Vec3f test_point_normalized = test_point.normalized();
    float direction_error = (unprojected.normalized() - test_point_normalized).norm();
    std::cout << "  Original direction: (" << test_point_normalized.x() << ", " 
              << test_point_normalized.y() << ", " << test_point_normalized.z() << ")" << std::endl;
    std::cout << "  Direction error: " << direction_error << std::endl;
    
    bool unproj_test_passed = unproj_valid && (direction_error < 0.01f);
    printTestResult("Pixel unprojection", unproj_test_passed);
    if (unproj_test_passed) tests_passed++;

    // =========================================================================
    // Test 4: Round-trip consistency (project -> unproject)
    // =========================================================================
    std::cout << "\n--- Test 4: Round-trip Consistency ---" << std::endl;
    tests_total++;
    
    // Test with an off-center point
    Vec3f off_center_point(0.3f, 0.2f, 1.0f);
    off_center_point.normalize();  // Make it unit vector
    
    std::cout << "  Test point (normalized): (" << off_center_point.x() << ", " 
              << off_center_point.y() << ", " << off_center_point.z() << ")" << std::endl;
    
    bool rt_proj_valid = false;
    Vec2f rt_projected = cam0.project(off_center_point, rt_proj_valid);
    std::cout << "  Projected: (" << rt_projected.x() << ", " << rt_projected.y() << ")" << std::endl;
    
    bool rt_unproj_valid = false;
    Vec3f rt_unprojected = cam0.unproject(rt_projected, rt_unproj_valid);
    rt_unprojected.normalize();
    
    std::cout << "  Unprojected: (" << rt_unprojected.x() << ", " 
              << rt_unprojected.y() << ", " << rt_unprojected.z() << ")" << std::endl;
    
    float roundtrip_error = (rt_unprojected - off_center_point).norm();
    std::cout << "  Round-trip error: " << roundtrip_error << std::endl;
    
    bool roundtrip_test_passed = rt_proj_valid && rt_unproj_valid && (roundtrip_error < 0.001f);
    printTestResult("Round-trip consistency", roundtrip_test_passed);
    if (roundtrip_test_passed) tests_passed++;

    // =========================================================================
    // Test 5: Quaternion to rotation matrix
    // =========================================================================
    std::cout << "\n--- Test 5: Quaternion to Rotation Matrix ---" << std::endl;
    tests_total++;
    
    // Identity quaternion (qx=0, qy=0, qz=0, qw=1) should give identity matrix
    Mat3f identity_rot = quaternionToRotationMatrix(0.0f, 0.0f, 0.0f, 1.0f);
    float identity_error = (identity_rot - Mat3f::Identity()).norm();
    std::cout << "  Identity quaternion -> rotation matrix error: " << identity_error << std::endl;
    
    bool quat_test_passed = (identity_error < 1e-6f);
    printTestResult("Quaternion to rotation matrix", quat_test_passed);
    if (quat_test_passed) tests_passed++;

    // =========================================================================
    // Test 6: RGB to YCbCr conversion
    // =========================================================================
    std::cout << "\n--- Test 6: RGB to YCbCr Conversion ---" << std::endl;
    tests_total++;
    
    // Create a small test image (2x2 BGR)
    std::cout << "  Creating test image..." << std::endl;
    cv::Mat test_rgb(2, 2, CV_8UC3);
    std::cout << "  Image created: " << test_rgb.cols << "x" << test_rgb.rows 
              << ", type=" << test_rgb.type() << ", empty=" << test_rgb.empty() << std::endl;
    
    test_rgb.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 0, 255);    // Red (BGR)
    test_rgb.at<cv::Vec3b>(0, 1) = cv::Vec3b(0, 255, 0);    // Green
    test_rgb.at<cv::Vec3b>(1, 0) = cv::Vec3b(255, 0, 0);    // Blue
    test_rgb.at<cv::Vec3b>(1, 1) = cv::Vec3b(255, 255, 255); // White
    std::cout << "  Image filled with test colors" << std::endl;
    
    try {
        std::cout << "  Calling rgb2YCbCr..." << std::flush;
        cv::Mat ycbcr = rgb2YCbCr(test_rgb);
        std::cout << " done" << std::endl;
        std::cout << "  RGB to YCbCr conversion completed." << std::endl;
        std::cout << "  Output type: " << ycbcr.type() << " (expected CV_8UC3=" << CV_8UC3 << ")" << std::endl;
        std::cout << "  Output size: " << ycbcr.cols << "x" << ycbcr.rows << std::endl;
        
        bool ycbcr_test_passed = (ycbcr.type() == CV_8UC3 && ycbcr.cols == 2 && ycbcr.rows == 2);
        printTestResult("RGB to YCbCr conversion", ycbcr_test_passed);
        if (ycbcr_test_passed) tests_passed++;
    } catch (const std::exception& e) {
        std::cerr << RED << "  Error: " << e.what() << RESET << std::endl;
        printTestResult("RGB to YCbCr conversion", false);
    }

    // =========================================================================
    // Summary
    // =========================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Test Summary: " << tests_passed << "/" << tests_total << " passed" << std::endl;
    std::cout << "========================================" << std::endl;

    if (tests_passed == tests_total) {
        std::cout << GREEN << "\nAll tests passed! Phase 1 verification complete." << RESET << std::endl;
        return 0;
    } else {
        std::cout << RED << "\nSome tests failed. Please review the output above." << RESET << std::endl;
        return 1;
    }
}
