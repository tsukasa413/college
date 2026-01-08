from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'output_dir',
            default_value='/tmp/captured_images',
            description='Directory to save captured images'
        ),
        DeclareLaunchArgument(
            'num_cameras',
            default_value='4',
            description='Number of cameras to capture from'
        ),
        DeclareLaunchArgument(
            'filename_prefix',
            default_value='max_res',
            description='Prefix for captured image filenames'
        ),
        
        Node(
            package='quad_cam_system',
            executable='max_resolution_capture_service',
            name='max_resolution_capture_service',
            parameters=[
                {
                    'output_dir': LaunchConfiguration('output_dir'),
                    'num_cameras': LaunchConfiguration('num_cameras'),
                    'capture_filename_prefix': LaunchConfiguration('filename_prefix')
                }
            ],
            output='screen'
        )
    ])