"""
Omnidirectional RGBD SLAM Launch File

Launches the omnidirectional SLAM node with 4-camera fisheye streaming,
full-sphere depth estimation, and 3D point cloud generation.

Published topics:
- /omnidirectional/rgb_panorama: Panoramic RGB image (2048x1024)
- /omnidirectional/depth_panorama: Depth map (2048x1024, 32FC1)
- /omnidirectional/point_cloud: 3D point cloud (sensor_msgs/PointCloud2)
- /rgb/camera_info: Camera info for the panorama
- /odom: Visual odometry (if enabled)

Usage:
  ros2 launch my_stereo_pkg omnidirectional_slam.launch.py
  
  # With custom dataset path:
  ros2 launch my_stereo_pkg omnidirectional_slam.launch.py dataset_path:=/path/to/resources
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    dataset_path_arg = DeclareLaunchArgument(
        'dataset_path',
        default_value='/home/motoken/college/ros2_ws/src/my_stereo_pkg/resources',
        description='Path to calibration and config files'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    # Omnidirectional SLAM node
    slam_node = Node(
        package='my_stereo_pkg',
        executable='omnidirectional_slam_node',
        name='omnidirectional_slam_node',
        output='screen',
        parameters=[{
            'dataset_path': LaunchConfiguration('dataset_path'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        # Increase buffer for large point clouds
        arguments=['--ros-args', '--log-level', 'info']
    )
    
    return LaunchDescription([
        dataset_path_arg,
        use_sim_time_arg,
        slam_node,
    ])
