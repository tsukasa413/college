/**
 * @file main.cpp
 * @brief Sphere Stereo リアルタイム実行のメインエントリーポイント
 * 
 * Python版 realtime_capture.py を C++ に移植。
 * 4台のフィッシュアイカメラからリアルタイムでRGB-Dパノラマを生成。
 */

#include "sphere_stereo_ros/CameraCapture.hpp"
#include "sphere_stereo_ros/DepthEstimator.hpp"
#include "sphere_stereo_ros/Utils.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace sphere_stereo_ros;

// FPS計測用のヘルパークラス
class FPSCounter {
public:
    explicit FPSCounter(int window_size = 30) 
        : window_size_(window_size), frame_count_(0), fps_(0.0) {}

    void tick() {
        auto now = std::chrono::steady_clock::now();
        
        if (frame_count_ == 0) {
            start_time_ = now;
        } else if (frame_count_ % window_size_ == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time_).count();
            fps_ = window_size_ / elapsed;
            start_time_ = now;
        }
        
        ++frame_count_;
    }

    double getFPS() const { return fps_; }
    int getFrameCount() const { return frame_count_; }

private:
    int window_size_;
    int frame_count_;
    double fps_;
    std::chrono::steady_clock::time_point start_time_;
};

int main(int argc, char** argv) {
    // ========================================
    // コマンドライン引数の解析
    // ========================================
    const cv::String keys =
        "{help h usage ?   |       | このヘルプメッセージを表示 }"
        "{calib            | resources/calibration.json | カメラキャリブレーションファイル }"
        "{camera_indices   | 0 2 4 6 | カメラデバイスIDリスト (スペース区切り) }"
        "{capture_width    | 1216   | カメラキャプチャ解像度（幅） }"
        "{capture_height   | 1216   | カメラキャプチャ解像度（高さ） }"
        "{matching_width   | 224    | マッチング用解像度（幅） }"
        "{matching_height  | 224    | マッチング用解像度（高さ） }"
        "{stitch_width     | 672    | スティッチング用解像度（幅） }"
        "{stitch_height    | 672    | スティッチング用解像度（高さ） }"
        "{pano_width       | 256    | パノラマ出力幅 }"
        "{pano_height      | 128    | パノラマ出力高さ }"
        "{min_dist         | 0.4    | 最小距離 [m] }"
        "{max_dist         | 1000.0 | 最大距離 [m] }"
        "{num_hypotheses   | 64     | 距離仮説数 }"
        "{fps              | 30.0   | カメラフレームレート }"
        "{no_camera        |        | カメラ無しモード（テスト用） }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Sphere Stereo Real-time RGB-D Panorama Estimation");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    // パラメータ取得
    std::string calib_path = parser.get<std::string>("calib");
    std::vector<int> camera_indices;
    {
        std::string indices_str = parser.get<std::string>("camera_indices");
        std::istringstream iss(indices_str);
        int idx;
        while (iss >> idx) {
            camera_indices.push_back(idx);
        }
    }
    
    const int capture_width = parser.get<int>("capture_width");
    const int capture_height = parser.get<int>("capture_height");
    const int matching_width = parser.get<int>("matching_width");
    const int matching_height = parser.get<int>("matching_height");
    const int stitch_width = parser.get<int>("stitch_width");
    const int stitch_height = parser.get<int>("stitch_height");
    const int pano_width = parser.get<int>("pano_width");
    const int pano_height = parser.get<int>("pano_height");
    const float min_dist = parser.get<float>("min_dist");
    const float max_dist = parser.get<float>("max_dist");
    const int num_hypotheses = parser.get<int>("num_hypotheses");
    const double fps_param = parser.get<double>("fps");
    const bool no_camera = parser.has("no_camera");

    if (!parser.check()) {
        parser.printErrors();
        return -1;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "Sphere Stereo Real-time Capture" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Calibration file: " << calib_path << std::endl;
    std::cout << "Camera indices: ";
    for (int idx : camera_indices) std::cout << idx << " ";
    std::cout << std::endl;
    std::cout << "Capture resolution: " << capture_width << "x" << capture_height << std::endl;
    std::cout << "Matching resolution: " << matching_width << "x" << matching_height << std::endl;
    std::cout << "Stitching resolution: " << stitch_width << "x" << stitch_height << std::endl;
    std::cout << "Panorama resolution: " << pano_width << "x" << pano_height << std::endl;
    std::cout << "Distance range: [" << min_dist << ", " << max_dist << "] m" << std::endl;
    std::cout << "Number of hypotheses: " << num_hypotheses << std::endl;
    std::cout << "========================================" << std::endl;

    // ========================================
    // キャリブレーションの読み込み
    // ========================================
    std::cout << "\nLoading calibration..." << std::endl;
    Vec2f matching_resolution(static_cast<float>(matching_width), static_cast<float>(matching_height));
    CalibrationSet calibration(calib_path, matching_resolution);
    std::cout << "Loaded " << calibration.numCameras() << " cameras" << std::endl;

    // ========================================
    // DepthEstimator 設定
    // ========================================
    DepthEstimatorConfig config;
    config.matching_width = matching_width;
    config.matching_height = matching_height;
    config.stitch_width = stitch_width;
    config.stitch_height = stitch_height;
    config.pano_width = pano_width;
    config.pano_height = pano_height;
    config.num_depth_candidates = num_hypotheses;
    config.min_dist = min_dist;
    config.max_dist = max_dist;

    // ========================================
    // DepthEstimator の初期化
    // ========================================
    std::cout << "\nInitializing DepthEstimator..." << std::endl;
    DepthEstimator depth_estimator(calibration, config);
    depth_estimator.initialize();
    std::cout << "DepthEstimator initialized successfully" << std::endl;

    // ========================================
    // カメラの初期化（オプション）
    // ========================================
    std::unique_ptr<CameraCapture> camera;
    if (!no_camera) {
        std::cout << "\nInitializing cameras..." << std::endl;
        try {
            camera = std::make_unique<CameraCapture>(camera_indices, capture_width, capture_height, fps_param);
            if (!camera->isOpened()) {
                std::cerr << "ERROR: Failed to initialize cameras" << std::endl;
                return -1;
            }
            std::cout << "Cameras initialized successfully (" << camera->getNumCameras() << " cameras)" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << std::endl;
            return -1;
        }
    } else {
        std::cout << "\nRunning in no-camera mode (for testing)" << std::endl;
    }

    // ========================================
    // メインループ
    // ========================================
    std::cout << "\nStarting capture loop..." << std::endl;
    std::cout << "Press 'q' to quit, 's' to save current frame" << std::endl;

    std::vector<cv::Mat> images_to_match;
    std::vector<cv::Mat> images_to_stitch;
    cv::Mat rgb_panorama, distance_map;
    FPSCounter fps_counter;

    const cv::Size matching_size(matching_width, matching_height);
    const cv::Size stitch_size(stitch_width, stitch_height);
    const size_t num_cams = camera_indices.size();

    // カメラ無しモード用のテスト画像
    if (no_camera) {
        std::cout << "Creating dummy images for testing..." << std::endl;
        // ダミー画像を生成
        images_to_match.resize(num_cams);
        images_to_stitch.resize(num_cams);
        
        for (size_t i = 0; i < num_cams; ++i) {
            // cv::Mat::create を使用して明示的にメモリを確保
            images_to_match[i].create(matching_height, matching_width, CV_8UC3);
            
            // 代わりに色で区別
            cv::Scalar cam_color(50 + i * 50, 100 + i * 30, 150 - i * 20);
            images_to_match[i].setTo(cam_color);
            
            images_to_stitch[i].create(stitch_height, stitch_width, CV_8UC3);
            images_to_stitch[i].setTo(cam_color);
            
            std::cout << "  Camera " << i << ": " << images_to_match[i].cols << "x" 
                      << images_to_match[i].rows << " channels=" << images_to_match[i].channels()
                      << " type=" << images_to_match[i].type() << " (empty=" << images_to_match[i].empty()
                      << " continuous=" << images_to_match[i].isContinuous() << ")" << std::endl;
            
            // テスト: OpenCV 関数が正常に動作するかチェック
            cv::Mat test_resize;
            try {
                cv::resize(images_to_match[i], test_resize, cv::Size(100, 100));
                std::cout << "    resize test: OK (" << test_resize.cols << "x" << test_resize.rows << ")" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "    resize test: FAILED - " << e.what() << std::endl;
            }
        }
        std::cout << "Dummy images created: " << num_cams << " cameras" << std::endl;
    }

    // 単一フレームのみ実行してテスト
    bool first_frame_only = true;

    while (true) {
        // カメラからフレームを取得
        if (camera) {
            bool success = camera->captureImages(images_to_match, images_to_stitch,
                                                matching_size, stitch_size);
            
            if (!success) {
                std::cerr << "Warning: Frame capture failed, continuing..." << std::endl;
                continue;
            }
        }

        // ========================================
        // DepthEstimator::update() の実行
        // ========================================
        std::cout << "Calling depth_estimator.update()..." << std::endl;
        try {
            depth_estimator.update(images_to_match, rgb_panorama, distance_map);
        } catch (const std::exception& e) {
            std::cerr << "ERROR in depth_estimator.update(): " << e.what() << std::endl;
            break;
        }
        std::cout << "depth_estimator.update() completed" << std::endl;

        // ========================================
        // 結果の検証
        // ========================================
        std::cout << "RGB panorama: " << rgb_panorama.cols << "x" << rgb_panorama.rows 
                  << " (empty=" << rgb_panorama.empty() << ")" << std::endl;
        std::cout << "Distance map: " << distance_map.cols << "x" << distance_map.rows 
                  << " (empty=" << distance_map.empty() << ")" << std::endl;

        if (rgb_panorama.empty() || distance_map.empty()) {
            std::cerr << "Warning: Empty output from DepthEstimator" << std::endl;
            break;
        }

        // ========================================
        // 深度マップの可視化
        // ========================================
        // 距離を逆数に変換して正規化（近いものを明るく）
        cv::Mat inv_distance;
        inv_distance.create(distance_map.rows, distance_map.cols, CV_32FC1);
        
        const float* dist_ptr = distance_map.ptr<float>();
        float* inv_ptr = inv_distance.ptr<float>();
        const int total_pixels = distance_map.rows * distance_map.cols;
        
        const float inv_min = 1.0f / max_dist;
        const float inv_max = 1.0f / min_dist;
        const float inv_range = inv_max - inv_min;
        
        for (int i = 0; i < total_pixels; ++i) {
            const float inv_d = 1.0f / (dist_ptr[i] + 1e-6f);
            // [1/max_dist, 1/min_dist] の範囲を [0, 1] にマッピング
            inv_ptr[i] = std::clamp((inv_d - inv_min) / inv_range, 0.0f, 1.0f);
        }
        
        // 0-255 に変換してカラーマップ適用（MAGMA）
        cv::Mat distance_vis;
        inv_distance.convertTo(distance_vis, CV_8UC1, 255.0);
        cv::applyColorMap(distance_vis, distance_vis, cv::COLORMAP_MAGMA);
        
        // RGB と距離マップを縦に並べて表示
        // Note: Stitcher::downloadRGBPanorama() returns BGR already
        cv::Mat display;
        cv::vconcat(rgb_panorama, distance_vis, display);
        
        // 情報オーバーレイ（cv::putText がセグフォルトするため、コンソール出力のみ）
        // std::stringstream ss;
        // ss << "FPS: " << std::fixed << std::setprecision(1) << fps_counter.getFPS()
        //    << " | Frame: " << fps_counter.getFrameCount();
        // cv::putText(display, ss.str(), cv::Point(10, 30),
        //            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        
        cv::imshow("Sphere Stereo - RGB-D Panorama", display);

        // FPS計測
        fps_counter.tick();
        std::cout << "FPS: " << std::fixed << std::setprecision(2) 
                  << fps_counter.getFPS() << " | Frame: " << fps_counter.getFrameCount() << std::endl;

        // テストモードでは1フレームのみ実行
        if (first_frame_only) {
            std::cout << "First frame test completed successfully!" << std::endl;
            break;
        }

        // キー入力チェック
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) { // 'q' または ESC
            std::cout << "\n\nExiting..." << std::endl;
            break;
        } else if (key == 's' || key == 'S') {
            // 現在のフレームを保存
            std::string filename_rgb = "output_rgb_" + std::to_string(fps_counter.getFrameCount()) + ".png";
            std::string filename_depth = "output_depth_" + std::to_string(fps_counter.getFrameCount()) + ".exr";
            cv::imwrite(filename_rgb, rgb_panorama);
            cv::imwrite(filename_depth, distance_map);
            std::cout << "\nSaved: " << filename_rgb << ", " << filename_depth << std::endl;
        }
    }

    // クリーンアップ
    cv::destroyAllWindows();
    std::cout << "Total frames captured: " << fps_counter.getFrameCount() << std::endl;
    std::cout << "Done." << std::endl;

    return 0;
}
