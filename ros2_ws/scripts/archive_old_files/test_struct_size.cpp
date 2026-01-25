#include <iostream>
#include "ros2_ws/src/my_stereo_pkg/include/my_stereo_pkg/utils.hpp"

int main() {
    std::cout << "Intrinsics size: " << sizeof(sphere_stereo::Intrinsics) << std::endl;
    std::cout << "CameraExtrinsics size: " << sizeof(sphere_stereo::CameraExtrinsics) << std::endl;
    std::cout << "CameraCalibration size: " << sizeof(sphere_stereo::CameraCalibration) << std::endl;
    
    std::cout << "\nOffsets:" << std::endl;
    sphere_stereo::CameraCalibration c;
    std::cout << "intrinsics: " << offsetof(sphere_stereo::CameraCalibration, intrinsics) << std::endl;
    std::cout << "extrinsics: " << offsetof(sphere_stereo::CameraCalibration, extrinsics) << std::endl;
    std::cout << "matching_scale: " << offsetof(sphere_stereo::CameraCalibration, matching_scale) << std::endl;
    std::cout << "resolution_x: " << offsetof(sphere_stereo::CameraCalibration, resolution_x) << std::endl;
    std::cout << "resolution_y: " << offsetof(sphere_stereo::CameraCalibration, resolution_y) << std::endl;
    
    return 0;
}
