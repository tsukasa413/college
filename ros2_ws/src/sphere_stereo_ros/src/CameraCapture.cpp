#include "sphere_stereo_ros/CameraCapture.hpp"
#include <iostream>

namespace sphere_stereo_ros {

CameraCapture::CameraCapture(const std::vector<int>& camera_indices,
                             int capture_width,
                             int capture_height,
                             double fps)
    : capture_width_(capture_width), capture_height_(capture_height)
{
    const size_t num_cameras = camera_indices.size();
    caps_.reserve(num_cameras);
    frame_buffers_.resize(num_cameras);
    grab_success_.resize(num_cameras, false);

    for (size_t i = 0; i < num_cameras; ++i) {
        const int idx = camera_indices[i];
        cv::VideoCapture cap(idx);
        
        if (!cap.isOpened()) {
            // 既に開いたカメラをクリーンアップ
            caps_.clear();
            throw std::runtime_error("Failed to open camera " + std::to_string(idx));
        }

        // 解像度とFPSの設定
        cap.set(cv::CAP_PROP_FRAME_WIDTH, capture_width_);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, capture_height_);
        cap.set(cv::CAP_PROP_FPS, fps);

        // MJPEG形式を試す（帯域幅削減）
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

        caps_.push_back(std::move(cap));

        std::cout << "[CameraCapture] Camera " << idx << " opened successfully: "
                  << caps_.back().get(cv::CAP_PROP_FRAME_WIDTH) << "x"
                  << caps_.back().get(cv::CAP_PROP_FRAME_HEIGHT) << " @ "
                  << caps_.back().get(cv::CAP_PROP_FPS) << " FPS" << std::endl;
    }
}

CameraCapture::~CameraCapture() {
    for (auto& cap : caps_) {
        if (cap.isOpened()) {
            cap.release();
        }
    }
}

bool CameraCapture::captureImages(std::vector<cv::Mat>& images_to_match,
                                  std::vector<cv::Mat>& images_to_stitch,
                                  const cv::Size& matching_size,
                                  const cv::Size& stitch_size)
{
    const size_t num_cameras = caps_.size();
    
    // 出力先のリサイズ
    images_to_match.resize(num_cameras);
    images_to_stitch.resize(num_cameras);

    // ========================================
    // Phase 1: 並列 grab()
    // ========================================
    // すべてのカメラで同時にフレームをキャプチャ（バッファに格納のみ）
    threads_.clear();
    threads_.reserve(num_cameras);

    for (size_t i = 0; i < num_cameras; ++i) {
        threads_.emplace_back([this, i]() {
            grab_success_[i] = caps_[i].grab();
        });
    }

    // すべてのスレッドが完了するまで待機
    for (auto& t : threads_) {
        t.join();
    }

    // ========================================
    // Phase 2: retrieve() と リサイズ
    // ========================================
    bool all_success = true;
    for (size_t i = 0; i < num_cameras; ++i) {
        if (!grab_success_[i]) {
            std::cerr << "[CameraCapture] Warning: Camera " << i << " grab failed" << std::endl;
            all_success = false;
            continue;
        }

        // バッファから画像を取得
        if (!caps_[i].retrieve(frame_buffers_[i])) {
            std::cerr << "[CameraCapture] Warning: Camera " << i << " retrieve failed" << std::endl;
            all_success = false;
            continue;
        }

        // マッチング用の低解像度画像
        cv::resize(frame_buffers_[i], images_to_match[i], matching_size, 0, 0, cv::INTER_AREA);

        // スティッチング用の高解像度画像
        if (stitch_size.width == capture_width_ && stitch_size.height == capture_height_) {
            // リサイズ不要の場合はコピー
            images_to_stitch[i] = frame_buffers_[i].clone();
        } else {
            cv::resize(frame_buffers_[i], images_to_stitch[i], stitch_size, 0, 0, cv::INTER_LINEAR);
        }
    }

    return all_success;
}

bool CameraCapture::isOpened() const {
    for (const auto& cap : caps_) {
        if (!cap.isOpened()) {
            return false;
        }
    }
    return true;
}

} // namespace sphere_stereo_ros
