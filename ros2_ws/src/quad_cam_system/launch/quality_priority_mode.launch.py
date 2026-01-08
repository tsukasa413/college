from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 画質優先モード（1080p + 高品質設定）
    # nvvidconvの品質設定を最高レベルに調整
    sync_node = Node(
        package='quad_cam_system',
        executable='sync_cam_node',
        name='synchronized_quad_camera',
        parameters=[{
            'fps': 24,               
            'simple_mode': False,    
            'full_fov_mode': True,   # フル画角維持
            'target_width': 1944,    # 元解像度そのまま（リサイズなし）
            'target_height': 1096,   # 元解像度そのまま（リサイズなし）
            'quality_priority': True # 画質優先フラグ
        }],
        output='screen'
    )

    return LaunchDescription([sync_node])