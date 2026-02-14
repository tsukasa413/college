"""
Omnidirectional RGBD SLAM Launch File with RViz2

Launches the omnidirectional SLAM node and RViz2 with pre-configured visualization.

Usage:
  # Launch with RViz2
  ros2 launch my_stereo_pkg omnidirectional_slam_with_rviz.launch.py
  
  # Launch without RViz2
  ros2 launch my_stereo_pkg omnidirectional_slam_with_rviz.launch.py start_rviz:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Get package share directory
    pkg_share = FindPackageShare('my_stereo_pkg')
    
    # RViz config file path
    rviz_config_file = PathJoinSubstitution([
        pkg_share,
        'rviz',
        'omnidirectional_slam.rviz'
    ])
    
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
    
    start_rviz_arg = DeclareLaunchArgument(
        'start_rviz',
        default_value='true',
        description='Start RViz2 for visualization'
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
        arguments=['--ros-args', '--log-level', 'info']
    )
    
    # RViz2 node (conditional)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(LaunchConfiguration('start_rviz'))
    )
    
    return LaunchDescription([
        dataset_path_arg,
        use_sim_time_arg,
        start_rviz_arg,
        slam_node,
        rviz_node,
    ])
