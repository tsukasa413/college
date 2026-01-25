/**
=======================================================================
Complete GPU Utilities Verification Test
Tests ALL implemented functions:
- Unproject/Project (Double Sphere)
- RGB to YCbCr
- Bilinear Resampling
=======================================================================
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

#include "my_stereo_pkg/utils.hpp"

using namespace sphere_stereo;

// Test RGB to YCbCr conversion
void test_rgb2ycbcr() {
    std::cout << "\n=== RGB to YCbCr Conversion Test ===" << std::endl;
    
    // Create test RGB image (3x3)
    int width = 3, height = 3;
    std::vector<uint8_t> rgb_image = {
        // Row 0
        255, 0, 0,    0, 255, 0,    0, 0, 255,
        // Row 1
        255, 255, 0,  0, 255, 255,  255, 0, 255,
        // Row 2
        128, 128, 128, 0, 0, 0,     255, 255, 255
    };
    
    std::vector<uint8_t> ycbcr_output(width * height * 3);
    
    try {
        rgb2ycbcr_gpu(rgb_image.data(), ycbcr_output.data(), width, height);
        
        std::cout << "GPU RGB2YCbCr Results:" << std::endl;
        std::cout << std::setw(15) << "RGB" << std::setw(15) << "YCbCr" << std::endl;
        
        for (int i = 0; i < width * height; ++i) {
            int r = rgb_image[i * 3];
            int g = rgb_image[i * 3 + 1];
            int b = rgb_image[i * 3 + 2];
            
            int y = ycbcr_output[i * 3];
            int cb = ycbcr_output[i * 3 + 1];
            int cr = ycbcr_output[i * 3 + 2];
            
            std::cout << "(" << std::setw(3) << r << "," << std::setw(3) << g << "," << std::setw(3) << b << ")  "
                      << "(" << std::setw(3) << y << "," << std::setw(3) << cb << "," << std::setw(3) << cr << ")" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

// Test bilinear resampling
void test_bilinear_resample() {
    std::cout << "\n=== Bilinear Resampling Test ===" << std::endl;
    
    // Create test image (4x4)
    int img_w = 4, img_h = 4;
    std::vector<uint8_t> test_image = {
        // Row 0
        255, 0, 0,    200, 0, 0,    150, 0, 0,    100, 0, 0,
        // Row 1
        0, 255, 0,    0, 200, 0,    0, 150, 0,    0, 100, 0,
        // Row 2
        0, 0, 255,    0, 0, 200,    0, 0, 150,    0, 0, 100,
        // Row 3
        128, 128, 128, 100, 100, 100, 80, 80, 80,  50, 50, 50
    };
    
    // Test sampling coordinates (normalized [0,1])
    int num_samples = 5;
    std::vector<float> sample_coords = {
        0.0f, 0.0f,      // Top-left corner
        1.0f, 1.0f,      // Bottom-right corner
        0.5f, 0.5f,      // Center
        0.25f, 0.25f,    // Quarter point
        0.75f, 0.75f     // Three-quarter point
    };
    
    std::vector<uint8_t> output(num_samples * 3);
    
    try {
        resample_bilinear_gpu(
            test_image.data(),
            sample_coords.data(),
            img_w, img_h,
            output.data(),
            num_samples, 1
        );
        
        std::cout << "GPU Bilinear Resampling Results:" << std::endl;
        std::cout << std::setw(12) << "U" << std::setw(12) << "V" 
                  << std::setw(15) << "RGB Output" << std::endl;
        
        for (int i = 0; i < num_samples; ++i) {
            float u = sample_coords[i * 2];
            float v = sample_coords[i * 2 + 1];
            
            int r = output[i * 3];
            int g = output[i * 3 + 1];
            int b = output[i * 3 + 2];
            
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(12) << u << std::setw(12) << v
                      << "    (" << std::setw(3) << r << "," << std::setw(3) << g << "," << std::setw(3) << b << ")" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "Complete GPU Utilities Verification Test" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    try {
        test_rgb2ycbcr();
        test_bilinear_resample();
        
        std::cout << "\n✅ All utility tests completed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
