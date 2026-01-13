cmake_minimum_required(VERSION 3.16)
project(sphere_stereo LANGUAGES CXX)

# C++17を要求
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# コンパイルオプション
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")

# ========================================
# 依存パッケージの検索
# ========================================
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
endif()

# スレッドライブラリ
find_package(Threads REQUIRED)

# ========================================
# インクルードディレクトリ
# ========================================
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# ========================================
# ライブラリ: CameraCapture
# ========================================
add_library(camera_capture SHARED
    src/CameraCapture.cpp
)

target_link_libraries(camera_capture
    ${OpenCV_LIBS}
    Threads::Threads
)

target_include_directories(camera_capture PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ========================================
# 実行ファイル: sphere_stereo_node
# ========================================
add_executable(sphere_stereo_node
    src/main.cpp
)

target_link_libraries(sphere_stereo_node
    camera_capture
    ${OpenCV_LIBS}
    Threads::Threads
)

# ========================================
# インストールターゲット
# ========================================
install(TARGETS camera_capture sphere_stereo_node
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)

# ========================================
# ビルド情報の表示
# ========================================
message(STATUS "========================================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "========================================")
