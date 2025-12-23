from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # フル画角維持モード（1080p高画質出力）
    # 画角を狭くせず、画質も劣化させない
    sync_node = Node(
        package='quad_cam_system',
        executable='sync_cam_node',
        name='synchronized_quad_camera',
        parameters=[{
            'fps': 24,               # 1080pモード上限に合わせて24fps
            'simple_mode': False,    
            'full_fov_mode': True,   # フル画角維持
            'target_width': 1920,    # 出力解像度：1080p高画質
            'target_height': 1080    # 出力解像度：1080p高画質
        }],
        output='screen'
    )

    return LaunchDescription([sync_node])