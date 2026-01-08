import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
from collections import deque
import time

class SynchronizedQuadCamera(Node):
    def __init__(self):
        super().__init__('synchronized_quad_camera')

        # デフォルトFPSを24に下げて負荷削減（32→24で25%削減）
        self.declare_parameter('fps', 24)
        self.declare_parameter('simple_mode', False)  # シンプルモード追加
        self.declare_parameter('full_fov_mode', True)  # フル画角モード（4K→リサイズ）
        self.declare_parameter('target_width', 1920)   # リサイズ後の幅（高画質）
        self.declare_parameter('target_height', 1080)  # リサイズ後の高さ（高画質）
        self.declare_parameter('quality_priority', False)  # 画質優先モード
        
        self.fps = self.get_parameter('fps').value
        self.simple_mode = self.get_parameter('simple_mode').value
        self.full_fov_mode = self.get_parameter('full_fov_mode').value
        self.target_width = self.get_parameter('target_width').value
        self.target_height = self.get_parameter('target_height').value
        self.quality_priority = self.get_parameter('quality_priority').value

        # Initialize attributes first
        self.bridge = CvBridge()
        self.caps = []
        self.camera_publishers = []
        
        # 非同期フレームバッファ（負荷分散用）
        self.frame_buffers = [deque(maxlen=2) for _ in range(4)]
        self.capture_threads = []
        self.running = True

        # Publishers for each camera
        for i in range(4):
            topic_name = f'camera_{i}/image_raw'
            publisher = self.create_publisher(Image, topic_name, 10)
            self.camera_publishers.append(publisher)
        
        # Initialize all cameras with optimized but stable pipeline
        for i in range(4):
            if self.simple_mode:
                # 最もシンプルで確実なパイプライン（デバッグ用）
                pipeline = (
                    f"nvarguscamerasrc sensor-id={i} ! "
                    "video/x-raw(memory:NVMM), width=(int)1296, height=(int)732, format=(string)NV12, framerate=(fraction)24/1 ! "
                    "nvvidconv ! "
                    "video/x-raw, format=(string)BGRx ! "
                    "videoconvert ! "
                    "appsink"
                )
                self.get_logger().info(f'Using SIMPLE pipeline for camera {i}')
            elif self.full_fov_mode:
                # 1080p取得→画質劣化を最小限に抑制（リサイズなし or 最小リサイズ）
                # sensor-mode=2: 1944x1096 フル画角（720pクロップより広い）
                # 画質優先：リサイズを最小限に抑える
                if self.target_width == 1944 and self.target_height == 1096:
                    # リサイズなし：元解像度そのまま（最高画質）
                    pipeline = (
                        f"nvarguscamerasrc sensor-id={i} sensor-mode=2 bufapi-version=1 ! "
                        "video/x-raw(memory:NVMM), width=(int)1944, height=(int)1096, format=(string)NV12, framerate=(fraction)32/1 ! "
                        "nvvidconv ! "
                        "video/x-raw, format=(string)BGRx ! "
                        "videoconvert ! "
                        "video/x-raw, format=(string)BGR ! "
                        "appsink emit-signals=false sync=false drop=true max-buffers=1"
                    )
                    self.get_logger().info(f'Using FULL-RES 1080p (no resize) pipeline for camera {i}')
                elif self.target_width == 3840 and self.target_height == 2160:
                    # 4K出力：sensor-mode=0で真の最高画質
                    pipeline = (
                        f"nvarguscamerasrc sensor-id={i} sensor-mode=0 bufapi-version=1 ! "
                        "video/x-raw(memory:NVMM), width=(int)3840, height=(int)2160, format=(string)NV12, framerate=(fraction)16/1 ! "
                        "nvvidconv ! "
                        "video/x-raw, format=(string)BGRx ! "
                        "videoconvert ! "
                        "video/x-raw, format=(string)BGR ! "
                        "appsink emit-signals=false sync=false drop=true max-buffers=1"
                    )
                    self.get_logger().info(f'Using ULTRA-QUALITY 4K (no resize) pipeline for camera {i}')
                elif self.target_width == 1920 and self.target_height == 1080:
                    # 録画最適化：1080p軽量モード（バッファロスト防止）
                    pipeline = (
                        f"nvarguscamerasrc sensor-id={i} sensor-mode=2 bufapi-version=1 ! "
                        "video/x-raw(memory:NVMM), width=(int)1944, height=(int)1096, format=(string)NV12, framerate=(fraction)25/1 ! "
                        f"nvvidconv ! "
                        f"video/x-raw, width=(int){self.target_width}, height=(int){self.target_height}, format=(string)BGRx ! "
                        "videoconvert ! "
                        "video/x-raw, format=(string)BGR ! "
                        "appsink emit-signals=false sync=false drop=true max-buffers=2"
                    )
                    self.get_logger().info(f'Using RECORDING-OPTIMIZED 1080p pipeline for camera {i}')
                else:
                    # 最小リサイズ：画質劣化を抑制
                    pipeline = (
                        f"nvarguscamerasrc sensor-id={i} sensor-mode=2 bufapi-version=1 ! "
                        "video/x-raw(memory:NVMM), width=(int)1944, height=(int)1096, format=(string)NV12, framerate=(fraction)32/1 ! "
                        f"nvvidconv ! "
                        f"video/x-raw, width=(int){self.target_width}, height=(int){self.target_height}, format=(string)BGRx ! "
                        "videoconvert ! "
                        "video/x-raw, format=(string)BGR ! "
                        "appsink emit-signals=false sync=false drop=true max-buffers=1"
                    )
                    self.get_logger().info(f'Using HIGH-QUALITY {self.target_width}x{self.target_height} pipeline for camera {i}')
            else:
                # 安定性を重視した最適化パイプライン（センサークロップあり）
                # sensor-mode=3で720p相当、録画用に最適化
                pipeline = (
                    f"nvarguscamerasrc sensor-id={i} sensor-mode=3 bufapi-version=1 ! "
                    "video/x-raw(memory:NVMM), width=(int)1296, height=(int)732, format=(string)NV12, framerate=(fraction)20/1 ! "
                    "nvvidconv ! "
                    f"video/x-raw, width=(int){self.target_width}, height=(int){self.target_height}, format=(string)BGRx ! "
                    "videoconvert ! "
                    "video/x-raw, format=(string)BGR ! "
                    "appsink emit-signals=false sync=false drop=true max-buffers=3"
                )
                self.get_logger().info(f'Using CROP-MODE 720p pipeline for camera {i}')
            
            self.get_logger().info(f'Opening Camera {i} (720p Synchronized)...')
            self.get_logger().info(f'Pipeline: {pipeline}')
            
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            # 少し待機してから再試行
            import time
            time.sleep(0.5)
            
            if cap.isOpened():
                # 実際にフレームが取得できるか確認
                ret, frame = cap.read()
                if ret:
                    self.caps.append(cap)
                    self.get_logger().info(f'Camera {i} opened successfully (frame size: {frame.shape})')
                    # 非同期キャプチャスレッドを開始（負荷分散）
                    thread = threading.Thread(target=self.capture_worker, args=(i, cap))
                    thread.daemon = True
                    thread.start()
                    self.capture_threads.append(thread)
                else:
                    self.get_logger().error(f'Camera {i} opened but cannot read frames!')
                    cap.release()
                    self.caps.append(None)
            else:
                self.get_logger().error(f'Failed to open camera {i}!')
                self.caps.append(None)

        # 軽量パブリッシュタイマー（バッファされたフレームを送信するだけ）
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.publish_frames)

    def capture_worker(self, camera_id, cap):
        """各カメラ用の非同期キャプチャワーカー（別スレッドで動作、負荷分散）"""
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # バッファに最新フレームを保存（古いものは自動削除）
                self.frame_buffers[camera_id].append((time.time(), frame))
            else:
                # キャプチャ失敗時は少し待機
                time.sleep(0.01)

    def publish_frames(self):
        """軽量フレームパブリッシャー（バッファされたフレームを送信するだけ）"""
        current_time = self.get_clock().now().to_msg()
        
        # 各カメラの最新フレームを取得してパブリッシュ
        for camera_id in range(4):
            if self.frame_buffers[camera_id]:
                timestamp, frame = self.frame_buffers[camera_id][-1]  # 最新フレーム
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = current_time  # 同じタイムスタンプで同期
                msg.header.frame_id = f"camera_link_{camera_id}"
                self.camera_publishers[camera_id].publish(msg)

    def destroy_node(self):
        """クリーンアップ（非同期スレッドも適切に終了）"""
        self.running = False  # スレッド終了シグナル
        
        # スレッド終了待機
        for thread in self.capture_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        # カメラリソース解放
        if hasattr(self, 'caps'):
            for cap in self.caps:
                if cap and cap.isOpened():
                    cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SynchronizedQuadCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 適切なクリーンアップ
        node.running = False  # スレッド停止
        for thread in node.capture_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        if hasattr(node, 'caps'):
            for cap in node.caps:
                if cap and cap.isOpened():
                    cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()