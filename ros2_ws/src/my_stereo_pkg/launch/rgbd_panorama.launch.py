"""
Launch file for RGBD Panorama Node with quad camera system
Subscribes to synchronized camera images and publishes RGBD panorama
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Path to calibration file (modify as needed)
    pkg_share = get_package_share_directory('my_stereo_pkg')
    default_calib_path = os.path.join(pkg_share, 'config', 'calibration.json')
    
    return LaunchDescription([
        Node(
            package='my_stereo_pkg',
            executable='rgbd_panorama_node',
            name='rgbd_panorama_node',
            output='screen',
            parameters=[{
                # Calibration file path (Basalt format JSON)
                'calibration_path': default_calib_path,
                
                # Reference camera indices for panorama viewpoint
                'references_indices': [0, 2],
                
                # Distance range [meters]
                'min_dist': 0.55,
                'max_dist': 100.0,
                
                # Sphere sweeping parameters
                'candidate_count': 32,
                
                # ISB filter parameters
                'sigma_i': 10.0,  # Intensity sigma
                'sigma_s': 25.0,  # Spatial sigma
                
                # Image resolutions [width, height]
                'matching_resolution': [1024, 1024],
                'rgb_to_stitch_resolution': [1216, 1216],
                'panorama_resolution': [2048, 1024],
                'original_resolution': [1944, 1096],
                
                # CUDA device ID
                'device_id': 0,
            }],
            # Remap topics to match quad_cam_system
            remappings=[
                ('/camera_0/image_raw', '/camera_0/image_raw'),
                ('/camera_1/image_raw', '/camera_1/image_raw'),
                ('/camera_2/image_raw', '/camera_2/image_raw'),
                ('/camera_3/image_raw', '/camera_3/image_raw'),
            ]
        )
    ])
