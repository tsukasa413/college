import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class MonoQuadCamera(Node):
    def __init__(self):
        super().__init__('mono_quad_camera')

        self.declare_parameter('fps', 24)
        self.declare_parameter('target_width', 1944)
        self.declare_parameter('target_height', 1096)
        
        self.fps = self.get_parameter('fps').value
        self.target_width = self.get_parameter('target_width').value
        self.target_height = self.get_parameter('target_height').value

        # Initialize attributes
        self.bridge = CvBridge()
        self.caps = []
        self.camera_publishers = []

        # Publishers for each camera (mono images)
        for i in range(4):
            topic_name = f'camera_{i}/image_mono'
            publisher = self.create_publisher(Image, topic_name, 10)
            self.camera_publishers.append(publisher)
        
        # Initialize all cameras with mono-optimized pipeline
        for i in range(4):
            # モノクロ最大解像度パイプライン
            # sensor-mode=2: 最大解像度モード（1944x1096）
            # nvvidconv: GRAY8形式で出力（モノクロ）、リサイズなし
            pipeline = (
                f"nvarguscamerasrc sensor-id={i} sensor-mode=2 bufapi-version=1 ! "
                "video/x-raw(memory:NVMM), width=(int)1944, height=(int)1096, format=(string)NV12, framerate=(fraction)24/1 ! "
                "queue max-size-buffers=1 leaky=downstream ! "
                "nvvidconv ! "
                f"video/x-raw, width=(int){self.target_width}, height=(int){self.target_height}, format=(string)GRAY8 ! "
                "queue max-size-buffers=1 leaky=downstream ! "
                "appsink emit-signals=false sync=false drop=true max-buffers=1"
            )
            
            self.get_logger().info(f'Opening Mono Camera {i} ({self.target_width}x{self.target_height})...')
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if cap.isOpened():
                self.caps.append(cap)
                self.get_logger().info(f'Mono Camera {i} opened successfully')
            else:
                self.get_logger().error(f'Failed to open mono camera {i}!')
                self.caps.append(None)

        # Start synchronized capture
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.sync_capture)

    def sync_capture(self):
        """Synchronized capture from all cameras (monochrome)"""
        # Get current timestamp for synchronization
        current_time = self.get_clock().now().to_msg()
        
        frames = []
        # Capture frames from all cameras simultaneously
        for i, cap in enumerate(self.caps):
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # フレームがすでにグレースケールかチェック
                    if len(frame.shape) == 3:
                        # カラーの場合はグレースケールに変換
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        # すでにグレースケールの場合
                        gray_frame = frame
                    frames.append((i, gray_frame))
                else:
                    self.get_logger().warn(f'Failed to capture from mono camera {i}')
            else:
                frames.append((i, None))

        # Publish all mono frames with the same timestamp
        for camera_id, frame in frames:
            if frame is not None:
                # モノクロ画像として送信
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="mono8")
                msg.header.stamp = current_time  # Same timestamp for all cameras
                msg.header.frame_id = f"camera_link_{camera_id}"
                self.camera_publishers[camera_id].publish(msg)

    def destroy_node(self):
        """Cleanup when node is destroyed"""
        if hasattr(self, 'caps'):
            for cap in self.caps:
                if cap and cap.isOpened():
                    cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MonoQuadCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        if hasattr(node, 'caps'):
            for cap in node.caps:
                if cap and cap.isOpened():
                    cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()