#ifndef SPHERE_STEREO_ROS_CAMERA_CAPTURE_HPP
#define SPHERE_STEREO_ROS_CAMERA_CAPTURE_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <thread>
#include <stdexcept>

namespace sphere_stereo_ros {

/**
 * @brief 複数のUSBカメラから並列キャプチャを行うクラス
 * 
 * Python版のMultiCameraCaptureをC++に移植。
 * 4台のカメラに対してgrab()を並列実行し、フレーム同期を確保する。
 */
class CameraCapture {
public:
    /**
     * @brief コンストラクタ
     * @param camera_indices カメラデバイスIDのリスト (例: {0, 2, 4, 6})
     * @param capture_width キャプチャ時の画像幅
     * @param capture_height キャプチャ時の画像高さ
     * @param fps フレームレート (デフォルト30)
     * @throws std::runtime_error カメラが開けなかった場合
     */
    CameraCapture(const std::vector<int>& camera_indices,
                  int capture_width = 1216,
                  int capture_height = 1216,
                  double fps = 30.0);

    /**
     * @brief デストラクタ
     */
    ~CameraCapture();

    // Non-copyable, non-movable
    CameraCapture(const CameraCapture&) = delete;
    CameraCapture& operator=(const CameraCapture&) = delete;
    CameraCapture(CameraCapture&&) = delete;
    CameraCapture& operator=(CameraCapture&&) = delete;

    /**
     * @brief 全カメラから並列キャプチャして2種類の解像度で画像を取得
     * 
     * grab()を全カメラに対して並列実行し、フレーム同期を確保。
     * その後retrieve()で画像を取得し、2種類の解像度にリサイズ。
     * 
     * @param images_to_match マッチング用低解像度画像の出力先 (例: 1024x1024)
     * @param images_to_stitch スティッチング用高解像度画像の出力先 (例: 1216x1216)
     * @param matching_size マッチング用解像度
     * @param stitch_size スティッチング用解像度
     * @return true: 成功, false: フレーム取得失敗
     */
    bool captureImages(std::vector<cv::Mat>& images_to_match,
                      std::vector<cv::Mat>& images_to_stitch,
                      const cv::Size& matching_size = cv::Size(1024, 1024),
                      const cv::Size& stitch_size = cv::Size(1216, 1216));

    /**
     * @brief カメラが正常に開かれているか確認
     * @return すべてのカメラが有効ならtrue
     */
    bool isOpened() const;

    /**
     * @brief カメラ台数を取得
     */
    int getNumCameras() const { return static_cast<int>(caps_.size()); }

private:
    std::vector<cv::VideoCapture> caps_;         ///< カメラキャプチャオブジェクト
    std::vector<cv::Mat> frame_buffers_;         ///< フレームバッファ (再利用)
    std::vector<bool> grab_success_;             ///< grab成功フラグ
    std::vector<std::thread> threads_;           ///< 並列grab用スレッド
    
    int capture_width_;
    int capture_height_;
};

} // namespace sphere_stereo_ros

#endif // SPHERE_STEREO_ROS_CAMERA_CAPTURE_HPP
