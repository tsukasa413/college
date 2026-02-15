cmake_minimum_required(VERSION 3.8)
project(my_stereo_pkg)

# Enable CUDA support
enable_language(CUDA)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Jetson AGX Orin specific optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(STATUS "Building for ARM64 architecture (Jetson AGX Orin)")
  add_compile_options(-march=armv8-a+crc+crypto -mtune=cortex-a78)
  add_compile_options(-O3 -DNDEBUG)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
# find_package(cv_bridge REQUIRED)  # Temporarily disabled due to OpenCV issues

# Find OpenCV (required for standalone estimator)
# Manually find OpenCV libraries - use the actual .so files in aarch64-linux-gnu
set(OpenCV_INCLUDE_DIRS "/usr/include/opencv4")
set(OpenCV_LIBS 
  /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.5d
  /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.5d
  /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.5d
  /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.5d
  /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.5d
  /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.5d
)

message(STATUS "Using OpenCV from: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Find PyTorch 
set(CMAKE_PREFIX_PATH "/home/motoken/.local/lib/python3.10/site-packages/torch/share/cmake/Torch;${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
# Don't apply TORCH_CXX_FLAGS globally, only to specific targets
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find PyTorch Python library for Python bindings
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libtorch_python.so'))"
  OUTPUT_VARIABLE TORCH_PYTHON_LIBRARY
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "PyTorch Python library: ${TORCH_PYTHON_LIBRARY}")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find Eigen3 (for standalone estimator)
find_package(Eigen3 REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Set CUDA architecture for Jetson AGX Orin (Ampere architecture)
set(CMAKE_CUDA_ARCHITECTURES 87)

# Include directories
include_directories(
  include/
  ${TORCH_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

# Create CUDA kernels library
add_library(cuda_kernels_lib STATIC
  src/cuda/stitcher.cu
  src/cuda/isb_filter.cu
  src/cuda/cost_kernels.cu
  src/cuda/pointcloud_kernels.cu
  src/cuda_wrapper.cu
)

# Set properties for CUDA library
set_target_properties(cuda_kernels_lib PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_SEPARABLE_COMPILATION ON
)

# Set CUDA compilation flags
target_compile_options(cuda_kernels_lib PRIVATE 
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS}>
)

# Prevent C++ defines from leaking to CUDA/ptxas
set_source_files_properties(
  src/cuda/stitcher.cu
  src/cuda/isb_filter.cu
  src/cuda/cost_kernels.cu
  src/cuda_wrapper.cu
  PROPERTIES
  COMPILE_OPTIONS "-D_GLIBCXX_USE_CXX11_ABI=1"
)

# Link CUDA libraries to kernels library
target_link_libraries(cuda_kernels_lib PUBLIC
  CUDA::cudart
  CUDA::nvToolsExt
  ${TORCH_LIBRARIES}
)

# Create Stitcher C++ library
add_library(stitcher_lib STATIC
  src/core/stitcher.cpp
  src/core/isb_filter.cpp
  src/core/depth_estimation.cpp
)

# Set properties for Stitcher library
set_target_properties(stitcher_lib PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

# Apply PyTorch C++ flags to Stitcher library
target_compile_options(stitcher_lib PRIVATE
  ${TORCH_CXX_FLAGS}
)

# Link Stitcher library with CUDA kernels and PyTorch
target_link_libraries(stitcher_lib PUBLIC
  cuda_kernels_lib
  CUDA::nvToolsExt
  ${TORCH_LIBRARIES}
)

# Create Point Cloud Generator library (separate from stitcher_lib due to ROS 2 dependency)
add_library(pointcloud_lib STATIC
  src/core/point_cloud_generator.cpp
)

# Set properties for Point Cloud library
set_target_properties(pointcloud_lib PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

# Apply PyTorch C++ flags
target_compile_options(pointcloud_lib PRIVATE
  ${TORCH_CXX_FLAGS}
)

# Link ROS 2 dependencies for Point Cloud library (uses PLAIN signature)
ament_target_dependencies(pointcloud_lib
  rclcpp
  sensor_msgs
)

# Link Point Cloud library with PyTorch and CUDA (must use PLAIN signature after ament_target_dependencies)
target_link_libraries(pointcloud_lib
  cuda_kernels_lib
  ${TORCH_LIBRARIES}
  CUDA::cudart
)

# Create PyTorch extension module (instead of pybind11_add_module)
add_library(_core_cpp MODULE 
  src/bindings.cpp
)

# Set properties for the PyTorch extension module
set_target_properties(_core_cpp PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
  PREFIX ""  # Remove lib prefix for Python modules
)

# Link PyTorch libraries 
target_link_libraries(_core_cpp PRIVATE
  ${TORCH_LIBRARIES}
  ${TORCH_PYTHON_LIBRARY}
  pybind11::module
  cuda_kernels_lib
  stitcher_lib
)

# Compiler-specific options
target_compile_definitions(_core_cpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install the pybind11 module
install(TARGETS _core_cpp
  DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}"
)

# Create standalone C++ executable
add_executable(standalone_estimator
  src/standalone_estimator.cpp
)

# Set properties for standalone executable
set_target_properties(standalone_estimator PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Link standalone executable with all required libraries
target_link_libraries(standalone_estimator PRIVATE
  stitcher_lib
  cuda_kernels_lib
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBS}
  CUDA::nvToolsExt
)

# Apply PyTorch C++ flags
target_compile_options(standalone_estimator PRIVATE
  ${TORCH_CXX_FLAGS}
)

# Install standalone executable
install(TARGETS standalone_estimator
  DESTINATION lib/${PROJECT_NAME}
)

# Create omnidirectional SLAM ROS 2 node
add_executable(omnidirectional_slam_node
  src/omnidirectional_slam_node.cpp
)

# Set properties for SLAM node
set_target_properties(omnidirectional_slam_node PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Apply PyTorch C++ flags
target_compile_options(omnidirectional_slam_node PRIVATE
  ${TORCH_CXX_FLAGS}
)

# Link ROS 2 dependencies FIRST (before target_link_libraries)
ament_target_dependencies(omnidirectional_slam_node
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
)

# Link SLAM node with all required libraries
target_link_libraries(omnidirectional_slam_node
  stitcher_lib
  pointcloud_lib
  cuda_kernels_lib
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBS}
  CUDA::nvToolsExt
)

# Install SLAM node executable
install(TARGETS omnidirectional_slam_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install resources
install(DIRECTORY
  resources
  DESTINATION share/${PROJECT_NAME}/
)

# Install Python scripts
install(PROGRAMS
  scripts/debug_main.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files (if any)
# install(DIRECTORY
#   config
#   DESTINATION share/${PROJECT_NAME}/
# )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()