cmake_minimum_required(VERSION 3.8)
project(my_stereo_pkg)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
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
# find_package(rclcpp REQUIRED)
# find_package(sensor_msgs REQUIRED)
# find_package(cv_bridge REQUIRED)  # Temporarily disabled due to OpenCV issues

# Find PyTorch - Temporarily disabled for basic testing
# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find CUDA - Temporarily disabled for basic testing  
# find_package(CUDAToolkit REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Set CUDA architecture for Jetson AGX Orin (Ampere architecture) - Temporarily disabled
# set(CMAKE_CUDA_ARCHITECTURES 87)

# Include directories - Temporarily disabled for basic testing
# include_directories(
#   ${TORCH_INCLUDE_DIRS}
#   ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
# )

# Create pybind11 module
pybind11_add_module(_core_cpp 
  src/bindings.cpp
)

# Set properties for the pybind11 module
set_target_properties(_core_cpp PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

# Link libraries - Temporarily disabled for basic testing
# target_link_libraries(_core_cpp PRIVATE
#   ${TORCH_LIBRARIES}
#   CUDA::cudart
#   CUDA::cublas
#   CUDA::curand
#   CUDA::cusparse
# )

# Compiler-specific options
target_compile_definitions(_core_cpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install the pybind11 module
install(TARGETS _core_cpp
  DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}"
)

# Install Python scripts
install(PROGRAMS
  scripts/debug_main.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files (if any)
# install(DIRECTORY
#   launch
#   DESTINATION share/${PROJECT_NAME}/
# )

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