from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 前回動作したシンプルモードを再現
    sync_node = Node(
        package='quad_cam_system',
        executable='sync_cam_node',
        name='synchronized_quad_camera',
        parameters=[{
            'fps': 15,               
            'simple_mode': True,     # 最もシンプルなパイプライン
            'full_fov_mode': False,
            'target_width': 1296,    
            'target_height': 732
        }],
        output='screen'
    )

    return LaunchDescription([sync_node])