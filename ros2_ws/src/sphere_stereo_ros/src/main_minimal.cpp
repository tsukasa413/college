/**
 * @file main_minimal.cpp
 * @brief 最小限のテスト版 - OpenCV操作を回避
 */

#include "sphere_stereo_ros/DepthEstimator.hpp"
#include "sphere_stereo_ros/Utils.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace sphere_stereo_ros;

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Sphere Stereo Minimal Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // パラメータ設定
    const std::string calib_path = "/home/motoken/sphere-stereo/resources/calibration.json";
    const int matching_width = 224;
    const int matching_height = 224;
    
    std::cout << "Loading calibration: " << calib_path << std::endl;
    Vec2f matching_resolution(static_cast<float>(matching_width), static_cast<float>(matching_height));
    
    try {
        CalibrationSet calibration(calib_path, matching_resolution);
        std::cout << "Loaded " << calibration.numCameras() << " cameras" << std::endl;
        
        // DepthEstimator 設定
        DepthEstimatorConfig config;
        config.matching_width = matching_width;
        config.matching_height = matching_height;
        config.stitch_width = 672;
        config.stitch_height = 672;
        config.pano_width = 256;
        config.pano_height = 128;
        config.num_depth_candidates = 64;
        config.min_dist = 0.4f;
        config.max_dist = 1000.0f;
        
        std::cout << "Initializing DepthEstimator..." << std::endl;
        DepthEstimator depth_estimator(calibration, config);
        depth_estimator.initialize();
        std::cout << "DepthEstimator initialized successfully" << std::endl;
        
        // ダミー画像を作成（OpenCVのcreate/setToを使わない方法）
        std::cout << "Creating minimal dummy images..." << std::endl;
        std::vector<cv::Mat> images_to_match(4);
        
        const int total_pixels = matching_width * matching_height;
        
        for (int cam = 0; cam < 4; ++cam) {
            std::cout << "Creating image " << cam << std::endl;
            
            // 生データを手動作成
            uint8_t* data = new uint8_t[total_pixels * 3];
            const uint8_t base_color[3] = {
                static_cast<uint8_t>(100 + cam * 30),
                static_cast<uint8_t>(150 - cam * 20), 
                static_cast<uint8_t>(200 + cam * 10)
            };
            
            // メモリに色データを設定
            for (int i = 0; i < total_pixels; ++i) {
                data[i * 3 + 0] = base_color[0];
                data[i * 3 + 1] = base_color[1];
                data[i * 3 + 2] = base_color[2];
            }
            
            // cv::Matコンストラクタで直接作成
            images_to_match[cam] = cv::Mat(matching_height, matching_width, CV_8UC3, data);
            
            std::cout << "  Image " << cam << ": " << images_to_match[cam].cols << "x" 
                      << images_to_match[cam].rows << " channels=" << images_to_match[cam].channels()
                      << " empty=" << images_to_match[cam].empty() << std::endl;
        }
        
        std::cout << "Dummy images created successfully" << std::endl;
        
        // DepthEstimator::update() をテスト
        cv::Mat rgb_panorama, distance_map;
        
        std::cout << "Calling depth_estimator.update()..." << std::endl;
        depth_estimator.update(images_to_match, rgb_panorama, distance_map);
        std::cout << "depth_estimator.update() completed!" << std::endl;
        
        // 結果検証
        std::cout << "RGB panorama: " << rgb_panorama.cols << "x" << rgb_panorama.rows 
                  << " (empty=" << rgb_panorama.empty() << ")" << std::endl;
        std::cout << "Distance map: " << distance_map.cols << "x" << distance_map.rows 
                  << " (empty=" << distance_map.empty() << ")" << std::endl;
        
        std::cout << "SUCCESS: First frame test completed!" << std::endl;
        
        // メモリクリーンアップ
        for (int cam = 0; cam < 4; ++cam) {
            delete[] images_to_match[cam].data;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "Test completed successfully" << std::endl;
    return 0;
}
