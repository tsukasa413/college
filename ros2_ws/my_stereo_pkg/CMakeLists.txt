cmake_minimum_required(VERSION 3.8)
project(my_stereo_pkg)

# C++17標準を使用
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ROS 2依存関係
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Python依存関係
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Pybind11
find_package(pybind11 REQUIRED)

# ========================================================
# ▼ PyTorchの場所をPythonに聞いて設定する（重要！） ▼
# ========================================================
execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_RESULT
)

if(TORCH_RESULT EQUAL 0 AND TORCH_CMAKE_PATH)
    message(STATUS "Found PyTorch CMake path: ${TORCH_CMAKE_PATH}")
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
else()
    message(WARNING "PyTorch not found in Python. Please install: pip3 install torch")
endif()

# LibTorch（PyTorchがインストールされている場合は必ず見つかる）
# CUDA関連のエラーを無視するための設定
set(CAFFE2_USE_CUDNN OFF CACHE BOOL "" FORCE)
set(USE_CUDA OFF CACHE BOOL "" FORCE)
set(Caffe2_DIR "" CACHE PATH "" FORCE)

find_package(Torch QUIET)
if(Torch_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  message(STATUS "LibTorch found: ${TORCH_VERSION}")
  add_definitions(-DUSE_TORCH)
else()
  message(WARNING "LibTorch not found. Building without PyTorch support.")
endif()

# CUDA (オプション)
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  
  # Jetson用CUDAアーキテクチャ設定 (AGX Orin は Ampere: sm_87)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87)
  endif()
  
  message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
  add_definitions(-DUSE_CUDA)
else()
  message(WARNING "CUDA not found. Building without CUDA support.")
endif()

# Pybind11を使用したC++ライブラリ _core_cpp の作成
pybind11_add_module(_core_cpp
  src/bindings/bindings.cpp
)

# C++17の設定を明示的に適用
target_compile_features(_core_cpp PRIVATE cxx_std_17)

# LibTorchが見つかった場合
if(Torch_FOUND)
  target_link_libraries(_core_cpp PRIVATE ${TORCH_LIBRARIES})
  target_include_directories(_core_cpp PRIVATE ${TORCH_INCLUDE_DIRS})
endif()

# CUDA が見つかった場合
if(CUDAToolkit_FOUND)
  target_link_libraries(_core_cpp PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
  )
endif()

# Pythonパッケージのインストール
ament_python_install_package(${PROJECT_NAME})

# _core_cpp ライブラリをPythonパッケージディレクトリにインストール
install(TARGETS _core_cpp
  DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}"
)

# スクリプトのインストール
install(PROGRAMS
  scripts/debug_main.py
  DESTINATION lib/${PROJECT_NAME}
)

# テスト
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
