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

# Find required packages
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/include
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
    ${CMAKE_CURRENT_SOURCE_DIR}/include
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
)

set_target_properties(sphere_stereo_utils PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
)

target_include_directories(sphere_stereo_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(sphere_stereo_utils
    PUBLIC
        -Wl,--whole-archive cuda_utils_kernels cuda_depth_estimation_kernels -Wl,--no-whole-archive
        ${CUDA_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
        Eigen3::Eigen
)

target_compile_options(sphere_stereo_utils PRIVATE 
    -Wall -Wextra -Wno-unused-parameter
)

# ============================================================================
# Installation
# ============================================================================

install(
    DIRECTORY include/
    DESTINATION include
)

install(
    TARGETS sphere_stereo_utils cuda_utils_kernels
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# ============================================================================
# Test Executable
# ============================================================================

# ============================================================================
# Test: Depth Estimation
# ============================================================================

add_executable(test_depth_estimation
    ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/test_depth_estimation.cpp
)

target_include_directories(test_depth_estimation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test_depth_estimation PRIVATE
    sphere_stereo_utils
)

install(TARGETS test_depth_estimation
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# ============================================================================
# Test: GPU Kernel (existing)
# ============================================================================

add_executable(test_gpu_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/test_gpu_kernel.cpp
)

target_include_directories(test_gpu_kernel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test_gpu_kernel PRIVATE
    sphere_stereo_utils
)

install(TARGETS test_gpu_kernel
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# ============================================================================
# Additional Test: All Utilities
# ============================================================================

add_executable(test_all_utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/test_all_utils.cpp
)

target_include_directories(test_all_utils PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test_all_utils PRIVATE
    sphere_stereo_utils
)

message(STATUS "=== Sphere Stereo Build ===")
message(STATUS "CUDA: ${CUDA_FOUND}")
message(STATUS "Eigen3: ${EIGEN3_FOUND}")
message(STATUS "================================")