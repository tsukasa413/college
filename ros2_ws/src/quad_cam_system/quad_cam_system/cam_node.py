import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class GStreamerCamera(Node):
    def __init__(self):
        super().__init__('gstreamer_camera')

        self.declare_parameter('sensor_id', 0)
        self.declare_parameter('fps', 32)

        self.sensor_id = self.get_parameter('sensor_id').value
        self.fps = self.get_parameter('fps').value

        topic_name = f'camera_{self.sensor_id}/image_raw'
        self.publisher_ = self.create_publisher(Image, topic_name, 10)
        self.bridge = CvBridge()

        # --- 安定化版 GStreamer Pipeline ---
        # 1. sensor-mode=2: 1944x1096モードを明示的に指定し、初期化の迷いをなくす
        # 2. bufapi-version=1: 新しいJetPackでのメモリ確保エラーを防ぐおまじない
        # 3. queue: 各処理の間にバッファ(queue)を挟み、一方が詰まっても他方を止めないようにする
        self.pipeline = (
            f"nvarguscamerasrc sensor-id={self.sensor_id} sensor-mode=2 bufapi-version=1 ! "
            "video/x-raw(memory:NVMM), width=(int)1944, height=(int)1096, format=(string)NV12, framerate=(fraction)32/1 ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "nvvidconv left=8 right=1928 top=8 bottom=1088 ! "
            "video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "appsink sync=false drop=true"
        )
        
        self.get_logger().info(f'Opening Camera {self.sensor_id} (Stable Pipeline)...')
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera {self.sensor_id}!')
            return

        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = f"camera_link_{self.sensor_id}"
                self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = GStreamerCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'cap') and node.cap.isOpened():
            node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()