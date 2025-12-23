cmake_minimum_required(VERSION 3.16)
project(sphere_stereo_ros LANGUAGES CXX CUDA)

# ==============================================================================
# C++ Standard
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA Standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ==============================================================================
# Build Type (Default: Release for performance)
# ==============================================================================
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# ==============================================================================
# CUDA Configuration for Jetson AGX Orin (sm_87)
# ==============================================================================
set(CMAKE_CUDA_ARCHITECTURES 87)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_87")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")

# ==============================================================================
# Find Dependencies
# ==============================================================================
# ROS 2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)

# External libraries
# Manual OpenCV configuration for Jetson (broken CMake/pkg-config files)
set(OpenCV_INCLUDE_DIRS "/usr/include/opencv4")
set(OpenCV_LIB_DIR "/usr/lib/aarch64-linux-gnu")
set(OpenCV_LIBRARIES
  "${OpenCV_LIB_DIR}/libopencv_core.so.4.5d"
  "${OpenCV_LIB_DIR}/libopencv_imgproc.so.4.5d"
  "${OpenCV_LIB_DIR}/libopencv_imgcodecs.so.4.5d"
  "${OpenCV_LIB_DIR}/libopencv_highgui.so.4.5d"
  "${OpenCV_LIB_DIR}/libopencv_calib3d.so.4.5d"
  "${OpenCV_LIB_DIR}/libopencv_videoio.so.4.5d"
)
set(OpenCV_VERSION "4.5.4")

# cv_bridge manual configuration for ROS 2 Humble
# We'll use runtime linking to avoid CMake conflicts
set(CV_BRIDGE_INCLUDE_DIRS "/opt/ros/humble/include/cv_bridge")
set(CV_BRIDGE_LIBRARIES "/opt/ros/humble/lib/libcv_bridge.so")
message(STATUS "Using manual cv_bridge configuration")
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

# nlohmann/json - try find_package first, fallback to header-only
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  message(STATUS "nlohmann_json not found via find_package, using header-only mode")
  # Assume nlohmann/json.hpp is available in system include or provided locally
endif()

# Print found versions
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBRARIES}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")

# ==============================================================================
# Include Directories
# ==============================================================================
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# ==============================================================================
# CUDA Library (must be defined before main lib)
# ==============================================================================
add_library(sphere_stereo_cuda SHARED
  src/cuda/stitcher.cu
  src/cuda/isb_filter.cu
  src/cuda/cost_volume.cu
)

set_target_properties(sphere_stereo_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(sphere_stereo_cuda PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/cuda>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(sphere_stereo_cuda
  ${CUDA_LIBRARIES}
  ${CUDA_cudart_LIBRARY}
)

# ==============================================================================
# Library: sphere_stereo_ros_lib (Core logic, ROS-independent)
# ==============================================================================
add_library(sphere_stereo_ros_lib SHARED
  src/Utils.cpp
  src/Stitcher.cpp
  src/IsbFilter.cpp
  src/DepthEstimator.cpp
  src/CameraCapture.cpp
)

target_include_directories(sphere_stereo_ros_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(sphere_stereo_ros_lib
  ${OpenCV_LIBRARIES}
  Eigen3::Eigen
  Threads::Threads
  sphere_stereo_cuda
)

# If nlohmann_json was found via find_package, link it
if(nlohmann_json_FOUND)
  target_link_libraries(sphere_stereo_ros_lib nlohmann_json::nlohmann_json)
endif()

# ==============================================================================
# Standalone Executable: sphere_stereo_main (ROS-independent)
# ==============================================================================
add_executable(sphere_stereo_main
  src/main.cpp
)

target_include_directories(sphere_stereo_main PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(sphere_stereo_main
  sphere_stereo_ros_lib
  ${OpenCV_LIBRARIES}
  Threads::Threads
)

# ==============================================================================
# Test Executable: utils_test
# ==============================================================================
add_executable(utils_test
  src/test_utils.cpp
)

target_include_directories(utils_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(utils_test
  sphere_stereo_ros_lib
)

# ==============================================================================
# Installation
# ==============================================================================
# Install headers
install(DIRECTORY include/
  DESTINATION include
)

# Install CUDA library
install(TARGETS sphere_stereo_cuda
  EXPORT sphere_stereo_cudaTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install library
install(TARGETS sphere_stereo_ros_lib
  EXPORT sphere_stereo_ros_libTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# Install executables
install(TARGETS sphere_stereo_main utils_test
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# ==============================================================================
# ROS 2 Node Executable
# ==============================================================================
add_executable(sphere_stereo_node
  src/sphere_stereo_node.cpp
)

target_include_directories(sphere_stereo_node PRIVATE
  ${CV_BRIDGE_INCLUDE_DIRS}
)

target_link_libraries(sphere_stereo_node
  sphere_stereo_ros_lib
  sphere_stereo_cuda
  ${CV_BRIDGE_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

ament_target_dependencies(sphere_stereo_node
  rclcpp
  std_msgs
  sensor_msgs
  image_transport
)

install(TARGETS sphere_stereo_node
  DESTINATION lib/${PROJECT_NAME}
)

# ==============================================================================
# Export for ament
# ==============================================================================
ament_export_targets(sphere_stereo_ros_libTargets sphere_stereo_cudaTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  OpenCV
  Eigen3
  rclcpp
)

# ==============================================================================
# Testing
# ==============================================================================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

# Add minimal test executable

# Add minimal test executable
add_executable(sphere_stereo_minimal src/main_minimal.cpp)
target_link_libraries(sphere_stereo_minimal sphere_stereo_ros_lib ${OpenCV_LIBS})

install(TARGETS sphere_stereo_minimal
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

