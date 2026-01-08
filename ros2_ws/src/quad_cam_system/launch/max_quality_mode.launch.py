from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 最高画質モード（リサイズなし）
    # 1944x1096 そのまま出力で画質劣化ゼロ
    sync_node = Node(
        package='quad_cam_system',
        executable='sync_cam_node',
        name='synchronized_quad_camera',
        parameters=[{
            'fps': 24,               
            'simple_mode': False,    
            'full_fov_mode': True,   # フル画角維持
            'target_width': 1944,    # 元解像度そのまま（リサイズなし）
            'target_height': 1096    # 元解像度そのまま（リサイズなし）
        }],
        output='screen'
    )

    return LaunchDescription([sync_node])