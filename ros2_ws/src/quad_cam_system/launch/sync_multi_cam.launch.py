from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Single node that handles all 4 cameras with synchronization
    # フル画角維持（4K→リサイズ）モードで動作
    sync_node = Node(
        package='quad_cam_system',
        executable='sync_cam_node',
        name='synchronized_quad_camera',
        parameters=[{
            'fps': 24,               # 4Kモードのため24fps（30fps上限）
            'simple_mode': False,    # フル機能使用
            'full_fov_mode': True,   # 4K取得→リサイズで画角維持
            'target_width': 1280,    # リサイズ後の幅
            'target_height': 720     # リサイズ後の高さ
        }],
        output='screen'
    )

    return LaunchDescription([sync_node])