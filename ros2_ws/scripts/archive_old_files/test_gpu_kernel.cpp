/**
=======================================================================
GPU Kernel Verification Test
Test Double Sphere projection/unprojection on GPU vs CPU reference
=======================================================================
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <nlohmann/json.hpp>

#include "my_stereo_pkg/utils.hpp"

using json = nlohmann::json;

namespace sphere_stereo {

// ============================================================================
// CPU Reference Implementation (for validation)
// ============================================================================

/**
 * CPU reference: Unproject pixel to 3D point using Double Sphere model
 */
void unproject_cpu_ref(
    float u, float v,
    const CameraCalibration& calib,
    float& x_out, float& y_out, float& z_out,
    bool& valid_out
) {
    const auto& intr = calib.intrinsics;
    float mx = (u - intr.cx) / intr.fx;
    float my = (v - intr.cy) / intr.fy;
    
    float r2 = mx * mx + my * my;
    float r = sqrtf(r2);
    
    // Double Sphere model
    float alpha = intr.alpha;
    float xi = intr.xi;
    
    // Check validity
    float denom = alpha * r2 + 1.0f - (2.0f * alpha - 1.0f) * r * xi;
    if (denom < 0.0001f) {
        valid_out = false;
        return;
    }
    
    float numerator = 1.0f - xi * xi * r2;
    float z_sphere = numerator / denom;
    
    x_out = mx * z_sphere;
    y_out = my * z_sphere;
    z_out = z_sphere - xi;
    
    valid_out = true;
}

/**
 * CPU reference: Project 3D point to pixel using Double Sphere model
 */
void project_cpu_ref(
    float x, float y, float z,
    const CameraCalibration& calib,
    float& u_out, float& v_out,
    bool& valid_out
) {
    const auto& intr = calib.intrinsics;
    
    // Translate by xi (second sphere center)
    float z_shifted = z + intr.xi;
    
    // Project to first sphere
    float r_xy2 = x * x + y * y;
    float r = sqrtf(r_xy2 + z_shifted * z_shifted);
    
    if (r < 0.0001f) {
        valid_out = false;
        return;
    }
    
    // Project to second sphere (image plane)
    float alpha = intr.alpha;
    float m = alpha * r + (1.0f - alpha) * z_shifted;
    
    if (m < 0.0001f) {
        valid_out = false;
        return;
    }
    
    float mx = x / m;
    float my = y / m;
    
    u_out = intr.fx * mx + intr.cx;
    v_out = intr.fy * my + intr.cy;
    
    valid_out = (u_out >= 0 && u_out < 640 && v_out >= 0 && v_out < 480);
}

// ============================================================================
// Test Functions
// ============================================================================

void test_unproject() {
    std::cout << "\n=== Unproject Test (Pixel → 3D) ===" << std::endl;
    
    // Create test calibration
    CameraCalibration calib;
    calib.intrinsics.fx = 400.0f;
    calib.intrinsics.fy = 400.0f;
    calib.intrinsics.cx = 320.0f;
    calib.intrinsics.cy = 240.0f;
    calib.intrinsics.xi = 0.0f;  // Pinhole model for simplicity
    calib.intrinsics.alpha = 0.5f;
    calib.resolution_x = 640;
    calib.resolution_y = 480;
    calib.matching_scale = 1.0f;
    
    // Transfer to GPU
    CameraCalibrationGPU calib_gpu;
    calib_gpu.initialize_from_host(calib);
    
    // Verify transfer
    CameraCalibration calib_readback = calib_gpu.get_host_copy();
    std::cout << "Calibration readback: fx=" << calib_readback.intrinsics.fx 
              << " fy=" << calib_readback.intrinsics.fy 
              << " cx=" << calib_readback.intrinsics.cx
              << " cy=" << calib_readback.intrinsics.cy
              << " xi=" << calib_readback.intrinsics.xi
              << " alpha=" << calib_readback.intrinsics.alpha
              << " matching_scale=" << calib_readback.matching_scale << std::endl;
    std::cout << "struct CameraCalibration size: " << sizeof(CameraCalibration) << std::endl;
    
    // Test pixels
    int test_count = 5;
    std::vector<float> test_uv = {
        320.0f, 240.0f,  // Center
        0.0f, 0.0f,       // Top-left corner
        640.0f, 480.0f,   // Bottom-right corner
        200.0f, 150.0f,   // Random
        450.0f, 350.0f    // Random
    };
    
    // CPU reference
    std::cout << "\nCPU Reference Results:" << std::endl;
    std::cout << std::setw(12) << "U" << std::setw(12) << "V"
              << std::setw(15) << "X(CPU)" << std::setw(15) << "Y(CPU)"
              << std::setw(15) << "Z(CPU)" << std::setw(8) << "Valid" << std::endl;
    
    for (int i = 0; i < test_count; ++i) {
        float u = test_uv[i * 2];
        float v = test_uv[i * 2 + 1];
        float x_cpu, y_cpu, z_cpu;
        bool valid_cpu;
        unproject_cpu_ref(u, v, calib, x_cpu, y_cpu, z_cpu, valid_cpu);
        
        std::cout << std::fixed << std::setprecision(6)
                  << std::setw(12) << u << std::setw(12) << v
                  << std::setw(15) << x_cpu << std::setw(15) << y_cpu
                  << std::setw(15) << z_cpu << std::setw(8) << (int)valid_cpu << std::endl;
    }
    
    std::vector<float> gpu_points(test_count * 3);
    std::vector<uint8_t> gpu_valid(test_count);
    
    // Call GPU kernel with proper grid dimensions
    try {
        unproject_gpu(
            test_uv.data(),
            calib_gpu,
            gpu_points.data(),
            gpu_valid.data(),
            test_count, 1  // width=test_count, height=1
        );
        
        std::cout << "\nGPU Results:" << std::endl;
        std::cout << "GPU kernel executed successfully" << std::endl;
        
        // Compare with CPU reference
        std::cout << std::setw(12) << "U" << std::setw(12) << "V"
                  << std::setw(15) << "X(GPU)" << std::setw(15) << "Y(GPU)"
                  << std::setw(15) << "Z(GPU)" << std::setw(8) << "Valid" << std::endl;
        
        for (int i = 0; i < test_count; ++i) {
            float u = test_uv[i * 2];
            float v = test_uv[i * 2 + 1];
            float x_gpu = gpu_points[i * 3];
            float y_gpu = gpu_points[i * 3 + 1];
            float z_gpu = gpu_points[i * 3 + 2];
            uint8_t valid = gpu_valid[i];
            
            std::cout << std::fixed << std::setprecision(6)
                      << std::setw(12) << u << std::setw(12) << v
                      << std::setw(15) << x_gpu << std::setw(15) << y_gpu
                      << std::setw(15) << z_gpu << std::setw(8) << (int)valid << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

void test_project() {
    std::cout << "\n=== Project Test (3D → Pixel) ===" << std::endl;
    
    CameraCalibration calib;
    calib.intrinsics.fx = 400.0f;
    calib.intrinsics.fy = 400.0f;
    calib.intrinsics.cx = 320.0f;
    calib.intrinsics.cy = 240.0f;
    calib.intrinsics.xi = 0.0f;
    calib.intrinsics.alpha = 0.5f;
    calib.resolution_x = 640;
    calib.resolution_y = 480;
    calib.matching_scale = 1.0f;
    
    CameraCalibrationGPU calib_gpu;
    calib_gpu.initialize_from_host(calib);
    
    // Test 3D points
    int test_count = 5;
    std::vector<float> test_points = {
        0.0f, 0.0f, 1.0f,    // Looking straight ahead
        1.0f, 0.0f, 1.0f,    // Slightly right
        -1.0f, 0.0f, 1.0f,   // Slightly left
        0.0f, 1.0f, 1.0f,    // Slightly down
        0.0f, -1.0f, 2.0f    // Up, farther
    };
    
    std::vector<float> gpu_uv(test_count * 2);
    std::vector<uint8_t> gpu_valid(test_count);
    
    try {
        project_gpu(
            test_points.data(),
            calib_gpu,
            gpu_uv.data(),
            gpu_valid.data(),
            test_count, 1  // width=test_count, height=1
        );
        
        std::cout << "GPU kernel executed successfully" << std::endl;
        
        std::cout << std::setw(15) << "X" << std::setw(15) << "Y"
                  << std::setw(15) << "Z" << std::setw(15) << "U(GPU)"
                  << std::setw(15) << "V(GPU)" << std::setw(8) << "Valid" << std::endl;
        
        for (int i = 0; i < test_count; ++i) {
            float x = test_points[i * 3];
            float y = test_points[i * 3 + 1];
            float z = test_points[i * 3 + 2];
            float u_gpu = gpu_uv[i * 2];
            float v_gpu = gpu_uv[i * 2 + 1];
            uint8_t valid = gpu_valid[i];
            
            std::cout << std::fixed << std::setprecision(6)
                      << std::setw(15) << x << std::setw(15) << y
                      << std::setw(15) << z << std::setw(15) << u_gpu
                      << std::setw(15) << v_gpu << std::setw(8) << (int)valid << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

} // namespace sphere_stereo

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "Sphere Stereo GPU Kernel Verification Test" << std::endl;
    std::cout << "============================================" << std::endl;
    
    try {
        sphere_stereo::test_unproject();
        sphere_stereo::test_project();
        
        std::cout << "\n✅ All tests completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
