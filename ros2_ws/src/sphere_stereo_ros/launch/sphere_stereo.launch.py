#!/usr/bin/env python3
"""
Launch file for sphere stereo node
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # Default paths (adjust as needed)
    default_calibration_file = os.path.join(
        os.path.expanduser("~"), 
        "sphere-stereo", 
        "resources", 
        "calibration.json"
    )
    default_config_path = os.path.join(
        os.path.expanduser("~"), 
        "sphere-stereo", 
        "resources"
    )
    
    # Launch arguments
    calibration_file_arg = DeclareLaunchArgument(
        'calibration_file',
        default_value=default_calibration_file,
        description='Path to calibration JSON file'
    )
    
    config_path_arg = DeclareLaunchArgument(
        'config_path',
        default_value=default_config_path,
        description='Path to configuration directory'
    )
    
    matching_width_arg = DeclareLaunchArgument(
        'matching_width',
        default_value='224',
        description='Width for depth matching resolution'
    )
    
    matching_height_arg = DeclareLaunchArgument(
        'matching_height',
        default_value='224',
        description='Height for depth matching resolution'
    )
    
    camera_fps_arg = DeclareLaunchArgument(
        'camera_fps',
        default_value='30.0',
        description='Camera capture frame rate'
    )
    
    publish_fps_arg = DeclareLaunchArgument(
        'publish_fps',
        default_value='10.0',
        description='ROS topic publish frame rate'
    )
    
    min_dist_arg = DeclareLaunchArgument(
        'min_dist',
        default_value='0.4',
        description='Minimum distance for depth estimation (meters)'
    )
    
    max_dist_arg = DeclareLaunchArgument(
        'max_dist',
        default_value='100.0',
        description='Maximum distance for depth estimation (meters)'
    )
    
    num_depth_candidates_arg = DeclareLaunchArgument(
        'num_depth_candidates',
        default_value='32',
        description='Number of depth candidates for sphere sweeping'
    )
    
    # Node definition
    sphere_stereo_node = Node(
        package='sphere_stereo_ros',
        executable='sphere_stereo_node',
        name='sphere_stereo_node',
        output='screen',
        parameters=[{
            'calibration_file': LaunchConfiguration('calibration_file'),
            'config_path': LaunchConfiguration('config_path'),
            'matching_width': LaunchConfiguration('matching_width'),
            'matching_height': LaunchConfiguration('matching_height'),
            'camera_fps': LaunchConfiguration('camera_fps'),
            'publish_fps': LaunchConfiguration('publish_fps'),
            'min_dist': LaunchConfiguration('min_dist'),
            'max_dist': LaunchConfiguration('max_dist'),
            'num_depth_candidates': LaunchConfiguration('num_depth_candidates'),
        }],
        remappings=[
            ('image_raw', 'sphere_stereo/image_raw'),
            ('depth_raw', 'sphere_stereo/depth_raw'),
        ]
    )
    
    return LaunchDescription([
        calibration_file_arg,
        config_path_arg,
        matching_width_arg,
        matching_height_arg,
        camera_fps_arg,
        publish_fps_arg,
        min_dist_arg,
        max_dist_arg,
        num_depth_candidates_arg,
        sphere_stereo_node,
    ])