"""
Complete launch file for RGBD Panorama System
Launches quad camera system + RGBD panorama node simultaneously
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directories
    quad_cam_pkg = get_package_share_directory('quad_cam_system')
    stereo_pkg = get_package_share_directory('my_stereo_pkg')
    
    # Path to calibration file
    calib_path = os.path.join(stereo_pkg, 'config', 'calibration.json')
    
    return LaunchDescription([
        # Launch quad camera system (max quality mode)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(quad_cam_pkg, 'launch', 'max_quality_mode.launch.py')
            ),
        ),
        
        # Launch RGBD panorama node
        Node(
            package='my_stereo_pkg',
            executable='rgbd_panorama_node',
            name='rgbd_panorama_node',
            output='screen',
            parameters=[{
                'calibration_path': calib_path,
                'references_indices': [0, 2],
                'min_dist': 0.55,
                'max_dist': 100.0,
                'candidate_count': 32,
                'sigma_i': 10.0,
                'sigma_s': 25.0,
                'matching_resolution': [1024, 1024],
                'rgb_to_stitch_resolution': [1216, 1216],
                'panorama_resolution': [2048, 1024],
                'original_resolution': [1944, 1096],
                'device_id': 0,
            }],
        )
    ])
