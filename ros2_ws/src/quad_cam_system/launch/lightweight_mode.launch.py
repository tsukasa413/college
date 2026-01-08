from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 超軽量モード（最小負荷・録画安定性重視）
    # 720p解像度で最高の安定性
    sync_node = Node(
        package='quad_cam_system',
        executable='sync_cam_node',
        name='synchronized_quad_camera',
        parameters=[{
            'fps': 20,               # 軽量な20fps
            'simple_mode': True,     # 最軽量パイプライン
            'full_fov_mode': False,  # クロップモードで負荷軽減
            'target_width': 1280,    # HD解像度
            'target_height': 720     # HD解像度
        }],
        output='screen'
    )

    return LaunchDescription([sync_node])