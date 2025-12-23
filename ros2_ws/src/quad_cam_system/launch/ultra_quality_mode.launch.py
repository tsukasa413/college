from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 真の最高画質モード（4K出力、リサイズなし）
    # 3840x2160 4K解像度そのままで画質劣化ゼロ
    sync_node = Node(
        package='quad_cam_system',
        executable='sync_cam_node',
        name='synchronized_quad_camera',
        parameters=[{
            'fps': 16,               # 4Kモードで確実に動作する16fps
            'simple_mode': False,    
            'full_fov_mode': True,   # フル画角維持
            'target_width': 3840,    # 4K解像度そのまま（リサイズなし）
            'target_height': 2160    # 4K解像度そのまま（リサイズなし）
        }],
        output='screen'
    )

    return LaunchDescription([sync_node])