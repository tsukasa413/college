from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 単一カメラでの4Kテスト（カメラ0のみ）
    sync_node = Node(
        package='quad_cam_system',
        executable='sync_cam_node',
        name='synchronized_quad_camera',
        parameters=[{
            'fps': 16,               # 4Kモードで確実に動作する16fps
            'simple_mode': False,    
            'full_fov_mode': True,   # フル画角維持
            'target_width': 3840,    # 4K解像度そのまま（リサイズなし）
            'target_height': 2160,   # 4K解像度そのまま（リサイズなし）
            'num_cameras': 1         # 1台のカメラのみでテスト
        }],
        output='screen'
    )

    return LaunchDescription([sync_node])