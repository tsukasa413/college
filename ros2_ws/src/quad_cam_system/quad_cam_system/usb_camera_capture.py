import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import threading
import time
from datetime import datetime
import os

class USBCameraMaxResolutionCapture(Node):
    def __init__(self):
        super().__init__('usb_camera_max_resolution_capture')
        
        # Parameters
        self.declare_parameter('output_dir', '/tmp/captured_images')
        self.declare_parameter('num_cameras', 4)
        self.declare_parameter('capture_filename_prefix', 'usb_max_res')
        
        self.output_dir = self.get_parameter('output_dir').value
        self.num_cameras = self.get_parameter('num_cameras').value
        self.filename_prefix = self.get_parameter('capture_filename_prefix').value
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.bridge = CvBridge()
        self.caps = []
        
        # Service for capturing images
        self.capture_service = self.create_service(
            Trigger, 
            'capture_usb_max_resolution_images', 
            self.capture_service_callback
        )
        
        # Publishers for captured images
        self.image_publishers = []
        for i in range(self.num_cameras):
            topic_name = f'camera_{i}/captured_image'
            publisher = self.create_publisher(Image, topic_name, 1)
            self.image_publishers.append(publisher)
        
        # Initialize USB cameras with maximum resolution
        self.get_logger().info('Initializing USB cameras with maximum resolution...')
        
        for i in range(self.num_cameras):
            self.get_logger().info(f'Opening USB Camera {i}...')
            
            cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                # Try to set maximum resolution
                # Set common high resolutions and see what's supported
                resolutions_to_try = [
                    (1920, 1080),  # 1080p
                    (1280, 720),   # 720p
                    (1944, 1096),  # Custom max resolution
                    (3840, 2160),  # 4K (if supported)
                ]
                
                supported_resolution = None
                for width, height in resolutions_to_try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    
                    # Verify the resolution was set
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        supported_resolution = (actual_width, actual_height)
                        self.get_logger().info(f'Camera {i} supports resolution: {actual_width}x{actual_height}')
                        break
                
                if supported_resolution:
                    self.caps.append(cap)
                    self.get_logger().info(f'USB Camera {i} initialized successfully (resolution: {supported_resolution})')
                else:
                    self.get_logger().error(f'USB Camera {i} opened but cannot read frames!')
                    cap.release()
                    self.caps.append(None)
            else:
                self.get_logger().error(f'Failed to open USB camera {i}!')
                self.caps.append(None)
        
        # Count successfully initialized cameras
        active_cameras = sum(1 for cap in self.caps if cap is not None)
        self.get_logger().info(f'Successfully initialized {active_cameras} out of {self.num_cameras} USB cameras')
        
        if active_cameras == 0:
            self.get_logger().error('No USB cameras were successfully initialized!')
            return
        
        self.get_logger().info('USB Max resolution image capture service ready!')
        self.get_logger().info('Call service: ros2 service call /capture_usb_max_resolution_images std_srvs/srv/Trigger')
        
    def capture_service_callback(self, request, response):
        """
        Service callback for capturing images
        """
        try:
            captured_count = self.capture_images()
            
            if captured_count > 0:
                response.success = True
                response.message = f"Successfully captured {captured_count} images from USB cameras"
                self.get_logger().info(response.message)
            else:
                response.success = False
                response.message = "Failed to capture any images from USB cameras"
                self.get_logger().error(response.message)
                
        except Exception as e:
            response.success = False
            response.message = f"Error during capture: {str(e)}"
            self.get_logger().error(response.message)
            
        return response
    
    def capture_images(self):
        """
        Capture one image from each USB camera simultaneously
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
        timestamp_ros = self.get_clock().now().to_msg()
        captured_count = 0
        
        # Prepare threading for simultaneous capture
        capture_threads = []
        captured_frames = [None] * self.num_cameras
        captured_resolutions = [None] * self.num_cameras
        
        def capture_single_camera(cam_id, cap):
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    captured_frames[cam_id] = frame
                    captured_resolutions[cam_id] = frame.shape[:2]  # (height, width)
                    self.get_logger().info(f'USB Camera {cam_id} captured frame: {frame.shape}')
                else:
                    self.get_logger().warning(f'USB Camera {cam_id} failed to capture frame')
        
        # Start capture threads for all cameras simultaneously
        self.get_logger().info('Starting simultaneous capture from all USB cameras...')
        for i, cap in enumerate(self.caps):
            if cap is not None:
                thread = threading.Thread(target=capture_single_camera, args=(i, cap))
                capture_threads.append(thread)
                thread.start()
        
        # Wait for all captures to complete
        for thread in capture_threads:
            thread.join()
        
        # Save captured frames and publish
        for i, frame in enumerate(captured_frames):
            if frame is not None:
                # Save to file with maximum quality
                filename = f"{self.filename_prefix}_camera{i}_{timestamp}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                
                # Save with maximum JPEG quality
                success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                
                if success:
                    # Publish as ROS message
                    msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                    msg.header.stamp = timestamp_ros
                    msg.header.frame_id = f"usb_camera_link_{i}"
                    self.image_publishers[i].publish(msg)
                    
                    captured_count += 1
                    height, width = captured_resolutions[i]
                    self.get_logger().info(f'✓ Saved and published: USB camera {i}, {filepath} (resolution: {width}x{height})')
                else:
                    self.get_logger().error(f'✗ Failed to save image for USB camera {i}')
        
        if captured_count > 0:
            self.get_logger().info(f'📸 Capture complete! Successfully saved {captured_count} images to: {self.output_dir}')
        else:
            self.get_logger().error('❌ No images were captured!')
            
        return captured_count
    
    def cleanup(self):
        """Clean up camera resources"""
        for i, cap in enumerate(self.caps):
            if cap is not None and cap.isOpened():
                cap.release()
                self.get_logger().info(f'Released USB camera {i}')

def main(args=None):
    rclpy.init(args=args)
    
    node = USBCameraMaxResolutionCapture()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()