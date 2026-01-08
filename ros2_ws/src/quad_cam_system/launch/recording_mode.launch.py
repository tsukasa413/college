from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 軽量録画モード（バッファロスト防止）
    # 解像度とフレームレートを録画に最適化
    sync_node = Node(
        package='quad_cam_system',
        executable='sync_cam_node',
        name='synchronized_quad_camera',
        parameters=[{
            'fps': 25,               # 録画に最適な25fps（安定性向上）
            'simple_mode': False,    
            'full_fov_mode': True,   # フル画角維持
            'target_width': 1920,    # Full HD（4Kより軽量）
            'target_height': 1080,   # Full HD（4Kより軽量）
            'quality_priority': True # 画質優先モード
        }],
        output='screen'
    )

    return LaunchDescription([sync_node])