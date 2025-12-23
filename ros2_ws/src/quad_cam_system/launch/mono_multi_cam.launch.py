from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Single node that handles all 4 cameras with monochrome output
    mono_node = Node(
        package='quad_cam_system',
        executable='mono_cam_node',
        name='mono_quad_camera',
        parameters=[
            {'fps': 24},
            {'target_width': 1944},
            {'target_height': 1096}
        ],
        output='screen'
    )

    return LaunchDescription([mono_node])