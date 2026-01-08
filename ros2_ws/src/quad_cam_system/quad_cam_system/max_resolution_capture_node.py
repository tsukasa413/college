import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import time
from datetime import datetime
import os

class MaxResolutionImageCapture(Node):
    def __init__(self):
        super().__init__('max_resolution_image_capture')
        
        # Parameters
        self.declare_parameter('output_dir', '/tmp/captured_images')
        self.declare_parameter('num_cameras', 4)
        self.declare_parameter('capture_filename_prefix', 'max_res')
        
        self.output_dir = self.get_parameter('output_dir').value
        self.num_cameras = self.get_parameter('num_cameras').value
        self.filename_prefix = self.get_parameter('capture_filename_prefix').value
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.bridge = CvBridge()
        self.caps = []
        
        # Initialize cameras with maximum resolution (1944x1096) pipeline
        self.get_logger().info('Initializing cameras with maximum resolution (1944x1096)...')
        
        for i in range(self.num_cameras):
            # Maximum resolution pipeline using sensor-mode=2
            pipeline = (
                f"nvarguscamerasrc sensor-id={i} sensor-mode=2 bufapi-version=1 ! "
                "video/x-raw(memory:NVMM), width=(int)1944, height=(int)1096, format=(string)NV12, framerate=(fraction)30/1 ! "
                "nvvidconv ! "
                "video/x-raw, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! "
                "appsink emit-signals=false sync=false drop=true max-buffers=1"
            )
            
            self.get_logger().info(f'Opening Camera {i} (Max Resolution 1944x1096)...')
            
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            # Wait and verify camera initialization
            time.sleep(0.5)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.caps.append(cap)
                    self.get_logger().info(f'Camera {i} initialized successfully (frame size: {frame.shape})')
                else:
                    self.get_logger().error(f'Camera {i} opened but cannot read frames!')
                    cap.release()
                    self.caps.append(None)
            else:
                self.get_logger().error(f'Failed to open camera {i}!')
                self.caps.append(None)
        
        # Count successfully initialized cameras
        active_cameras = sum(1 for cap in self.caps if cap is not None)
        self.get_logger().info(f'Successfully initialized {active_cameras} out of {self.num_cameras} cameras')
        
        if active_cameras == 0:
            self.get_logger().error('No cameras were successfully initialized!')
            return
        
        # Service for capturing images (will be triggered externally)
        self.create_timer(0.1, self.status_check_callback)  # Just for keeping the node alive
        
        self.get_logger().info('Ready to capture maximum resolution images. Call capture_images() method.')
        
    def capture_images(self):
        """
        Capture one image from each camera simultaneously at maximum resolution
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
        captured_count = 0
        
        # Prepare threading for simultaneous capture
        capture_threads = []
        captured_frames = [None] * self.num_cameras
        
        def capture_single_camera(cam_id, cap):
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    captured_frames[cam_id] = frame
                    self.get_logger().info(f'Camera {cam_id} captured frame: {frame.shape}')
                else:
                    self.get_logger().warning(f'Camera {cam_id} failed to capture frame')
        
        # Start capture threads for all cameras simultaneously
        for i, cap in enumerate(self.caps):
            if cap is not None:
                thread = threading.Thread(target=capture_single_camera, args=(i, cap))
                capture_threads.append(thread)
                thread.start()
        
        # Wait for all captures to complete
        for thread in capture_threads:
            thread.join()
        
        # Save captured frames
        for i, frame in enumerate(captured_frames):
            if frame is not None:
                filename = f"{self.filename_prefix}_camera{i}_{timestamp}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                
                # Save with maximum quality
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                captured_count += 1
                self.get_logger().info(f'Saved: {filepath} (size: {frame.shape})')
        
        self.get_logger().info(f'Capture complete! Saved {captured_count} images at maximum resolution (1944x1096)')
        return captured_count
    
    def capture_images_and_publish(self):
        """
        Capture images and also publish them as ROS messages
        """
        timestamp_ros = self.get_clock().now().to_msg()
        captured_count = 0
        
        # Publishers for each camera (create on demand)
        publishers = []
        for i in range(self.num_cameras):
            topic_name = f'camera_{i}/captured_image'
            publisher = self.create_publisher(Image, topic_name, 1)
            publishers.append(publisher)
        
        capture_threads = []
        captured_frames = [None] * self.num_cameras
        
        def capture_single_camera(cam_id, cap):
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    captured_frames[cam_id] = frame
        
        # Simultaneous capture
        for i, cap in enumerate(self.caps):
            if cap is not None:
                thread = threading.Thread(target=capture_single_camera, args=(i, cap))
                capture_threads.append(thread)
                thread.start()
        
        # Wait for completion
        for thread in capture_threads:
            thread.join()
        
        # Save and publish
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        for i, frame in enumerate(captured_frames):
            if frame is not None:
                # Save to file
                filename = f"{self.filename_prefix}_camera{i}_{timestamp}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                
                # Publish as ROS message
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = timestamp_ros
                msg.header.frame_id = f"camera_link_{i}"
                publishers[i].publish(msg)
                
                captured_count += 1
                self.get_logger().info(f'Saved and published: camera {i}, {filepath}')
        
        return captured_count
    
    def status_check_callback(self):
        """Periodic status check"""
        pass
    
    def cleanup(self):
        """Clean up camera resources"""
        for i, cap in enumerate(self.caps):
            if cap is not None and cap.isOpened():
                cap.release()
                self.get_logger().info(f'Released camera {i}')

def main(args=None):
    rclpy.init(args=args)
    
    node = MaxResolutionImageCapture()
    
    try:
        # Example: Capture images after 3 seconds
        time.sleep(3.0)
        node.get_logger().info('Starting image capture...')
        captured = node.capture_images()
        
        if captured > 0:
            # Wait a bit and capture again as example
            time.sleep(2.0)
            node.get_logger().info('Capturing again with ROS publish...')
            node.capture_images_and_publish()
        
        # Keep node alive for manual testing
        node.get_logger().info('Node ready for external triggers. Press Ctrl+C to exit.')
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()