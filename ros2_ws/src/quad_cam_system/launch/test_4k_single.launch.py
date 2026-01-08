from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 4Kテスト用（1台のみ）
    # 負荷を確認するため1台だけで4K出力テスト
    sync_node = Node(
        package='quad_cam_system',
        executable='sync_cam_node',
        name='synchronized_quad_camera',
        parameters=[{
            'fps': 16,               # 4Kモードでは16fps上限
            'simple_mode': False,    
            'full_fov_mode': True,   # フル画角維持
            'target_width': 3840,    # 4K解像度（最高画質）
            'target_height': 2160,   # 4K解像度（最高画質）
            'test_single_camera': True  # 1台のみテスト
        }],
        output='screen'
    )

    return LaunchDescription([sync_node])