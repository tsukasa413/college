cmake_minimum_required(VERSION 3.16)
project(my_stereo_pkg LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Jetson AGX Orin optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(STATUS "Building for ARM64 (Jetson AGX Orin)")
  add_compile_options(-O3 -march=armv8-a+crc+crypto -mtune=cortex-a78)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_ARCHITECTURES 87)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ROS2 packages (without cv_bridge yet - will be added after OpenCV)
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)

# Find required packages
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
# OpenCV configuration for Jetson - use aarch64 libs
include_directories(/usr/include/opencv4)
set(OpenCV_LIBRARIES 
    /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.5.4d
    /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.5.4d
    /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.5.4d
    /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.5.4d
    /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.5.4d
    /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.5.4d
    /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.5.4d
)

# cv_bridge has OpenCV version conflict, skip it for now
# We'll implement manual ROS Image <-> cv::Mat conversion
# find_package(cv_bridge REQUIRED)

# LibTorch for ISB Filter and Stitcher
execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via find_package, trying pip package")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_FOUND_RESULT
    )
    if(PYBIND11_FOUND_RESULT EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
        find_package(pybind11 REQUIRED)
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============================================================================
# Sphere Stereo GPU Kernels (Utils)
# ============================================================================

add_library(cuda_utils_kernels STATIC
    src/cuda/utils_kernel.cu
)

set_target_properties(cuda_utils_kernels PROPERTIES
    CUDA_ARCHITECTURES 87
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
)

target_include_directories(cuda_utils_kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR}
)

target_compile_options(cuda_utils_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=128>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# ============================================================================
# Sphere Stereo Depth Estimation Kernels (Fused)
# ============================================================================

add_library(cuda_depth_estimation_kernels STATIC
    src/cuda/depth_estimation.cu
)

set_target_properties(cuda_depth_estimation_kernels PROPERTIES
    CUDA_ARCHITECTURES 87
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
)

target_include_directories(cuda_depth_estimation_kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR}
)

target_compile_options(cuda_depth_estimation_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=64>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# ============================================================================
# Sphere Stereo Host Utilities & Depth Estimation
# ============================================================================

add_library(sphere_stereo_utils SHARED
    src/core/utils.cpp
    src/core/depth_estimation.cpp
    src/core/isb_filter.cpp
    src/core/stitcher.cpp
    src/cuda/isb_filter.cu
    src/cuda/stitcher.cu
    src/cuda_wrapper.cu
)

set_target_properties(sphere_stereo_utils PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
)

target_include_directories(sphere_stereo_utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CUDA_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(sphere_stereo_utils
    PUBLIC
        -Wl,--whole-archive cuda_utils_kernels cuda_depth_estimation_kernels -Wl,--no-whole-archive
        ${CUDA_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
        ${TORCH_LIBRARIES}
        Eigen3::Eigen
)

target_compile_options(sphere_stereo_utils PRIVATE 
    -Wall -Wextra -Wno-unused-parameter
)

# ============================================================================
# Installation (ROS2 style)
# ============================================================================

install(
    DIRECTORY include/
    DESTINATION include
)

install(
    TARGETS sphere_stereo_utils cuda_utils_kernels cuda_depth_estimation_kernels
    EXPORT export_${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# ============================================================================
# Python Bindings (pybind11)
# ============================================================================

if(pybind11_FOUND)
    message(STATUS "Building Python bindings with pybind11")
    
    pybind11_add_module(sphere_stereo_cuda 
        src/bindings.cpp
    )
    
    target_include_directories(sphere_stereo_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDA_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
    )
    
    target_link_libraries(sphere_stereo_cuda PRIVATE
        sphere_stereo_utils
        ${CUDA_LIBRARIES}
    )
    
    # Install to local install directory (not system-wide)
    install(TARGETS sphere_stereo_cuda
        LIBRARY DESTINATION lib
    )
else()
    message(WARNING "pybind11 not found, skipping Python bindings")
endif()

# ============================================================================
# Test Executable
# ============================================================================

# ============================================================================
# Test: Depth Estimation
# ============================================================================

# Commented out - test file moved to organized test suite
# add_executable(test_depth_estimation
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/test_depth_estimation.cpp
# )
# 
# target_include_directories(test_depth_estimation PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# 
# target_link_libraries(test_depth_estimation PRIVATE
#     sphere_stereo_utils
# )
# 
# install(TARGETS test_depth_estimation
#     RUNTIME DESTINATION lib/${PROJECT_NAME}
# )

# ============================================================================
# Test: GPU Kernel (existing)
# ============================================================================

# Commented out - test file moved to organized test suite
# add_executable(test_gpu_kernel
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/test_gpu_kernel.cpp
# )
# 
# target_include_directories(test_gpu_kernel PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# 
# target_link_libraries(test_gpu_kernel PRIVATE
#     sphere_stereo_utils
# )

# install(TARGETS test_gpu_kernel
#     RUNTIME DESTINATION lib/${PROJECT_NAME}
# )

# ============================================================================
# Test: Minimal Depth Estimation (DISABLED - missing test file)
# ============================================================================

# add_executable(test_minimal_depth
#     test/test_minimal_depth.cpp
# )
# 
# target_include_directories(test_minimal_depth PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# 
# target_link_libraries(test_minimal_depth PRIVATE
#     sphere_stereo_utils
# )
# 
# install(TARGETS test_minimal_depth
#     RUNTIME DESTINATION lib/${PROJECT_NAME}
# )

# ============================================================================
# Additional Test: All Utilities (DISABLED - missing test file)
# ============================================================================

# add_executable(test_all_utils
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/test_all_utils.cpp
# )
# 
# target_include_directories(test_all_utils PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# 
# target_link_libraries(test_all_utils PRIVATE
#     sphere_stereo_utils
# )

# ============================================================================
# StereoEngine Integration Module
# ============================================================================

# ============================================================================
# StereoEngine Integration Module (DISABLED - files not present)
# ============================================================================

# # Try to find jsoncpp
# find_library(JSONCPP_LIBRARY jsoncpp)
# find_path(JSONCPP_INCLUDE_DIR jsoncpp/json/json.h)
# 
# if(JSONCPP_LIBRARY AND JSONCPP_INCLUDE_DIR)
#     message(STATUS "jsoncpp found, building StereoEngine")
#     message(STATUS "  Library: ${JSONCPP_LIBRARY}")
#     message(STATUS "  Include: ${JSONCPP_INCLUDE_DIR}")
#     
#     # StereoEngine library
#     cuda_add_library(stereo_engine SHARED
#         src/core/stereo_engine.cpp
#         src/cuda/stereo_engine_kernels.cu
#     )
#     
#     target_include_directories(stereo_engine PUBLIC
#         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#         $<INSTALL_INTERFACE:include>
#         ${JSONCPP_INCLUDE_DIR}
#         /usr/include/opencv4
#     )
#     
#     target_link_libraries(stereo_engine
#         ${CUDA_LIBRARIES}
#         ${TORCH_LIBRARIES}
#         ${OpenCV_LIBRARIES}
#         sphere_stereo_utils
#         ${JSONCPP_LIBRARY}
#     )
#     
#     # Python bindings for StereoEngine
#     if(pybind11_FOUND)
#         pybind11_add_module(stereo_engine_cuda
#             src/stereo_engine_bindings.cpp
#         )
#         
#         target_link_libraries(stereo_engine_cuda PRIVATE
#             stereo_engine
#         )
#         
#         # Install to Python dist-packages
#         install(TARGETS stereo_engine_cuda
#             LIBRARY DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
#         )
#         
#         message(STATUS "StereoEngine Python bindings will be built")
#     endif()
#     
#     # Install StereoEngine library
#     install(TARGETS stereo_engine
#         EXPORT export_${PROJECT_NAME}
#         LIBRARY DESTINATION lib
#         ARCHIVE DESTINATION lib
#         RUNTIME DESTINATION bin
#         INCLUDES DESTINATION include
#     )
#     
#     # ========================================================================
#     # Standalone C++ Evaluator (Performance Benchmark)
#     # ========================================================================
#     message(STATUS "Building standalone evaluator...")
#     
#     add_executable(standalone_eval
#         src/apps/standalone_eval.cpp
#     )
#     
#     target_include_directories(standalone_eval PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/include
#         /usr/include/opencv4
#         ${JSONCPP_INCLUDE_DIR}
#     )
#     
#     target_link_libraries(standalone_eval
#         stereo_engine
#         ${OpenCV_LIBRARIES}
#         ${JSONCPP_LIBRARY}
#         stdc++fs  # For std::filesystem
#     )
#     
#     # Install standalone evaluator
#     install(TARGETS standalone_eval
#         RUNTIME DESTINATION lib/${PROJECT_NAME}
#     )
#     
#     message(STATUS "  standalone_eval executable will be built")
#     
# else()
#     message(WARNING "jsoncpp not found, skipping StereoEngine module")
#     message(WARNING "  Install with: sudo apt-get install libjsoncpp-dev")
# endif()

# ============================================================================
# ROS2 RGBD Panorama Node
# ============================================================================

message(STATUS "Building RGBD Panorama ROS2 Node...")

add_executable(rgbd_panorama_node
    src/core/rgbd_panorama_node.cpp
)

target_include_directories(rgbd_panorama_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/include/opencv4
)

target_link_libraries(rgbd_panorama_node
    sphere_stereo_utils
    ${OpenCV_LIBRARIES}
    ${TORCH_LIBRARIES}
)

ament_target_dependencies(rgbd_panorama_node
    rclcpp
    sensor_msgs
)

install(TARGETS rgbd_panorama_node
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config
    OPTIONAL
)

message(STATUS "  rgbd_panorama_node will be built")

# ============================================================================
# ROS2 Export
# ============================================================================

ament_export_include_directories(include)
ament_export_libraries(sphere_stereo_utils cuda_utils_kernels cuda_depth_estimation_kernels)
ament_export_targets(export_${PROJECT_NAME})

ament_package()

message(STATUS "=== Sphere Stereo Build ===")
message(STATUS "CUDA: ${CUDA_FOUND}")
message(STATUS "Eigen3: ${EIGEN3_FOUND}")
message(STATUS "pybind11: ${pybind11_FOUND}")
message(STATUS "================================")