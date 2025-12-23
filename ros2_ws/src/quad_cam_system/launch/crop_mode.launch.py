from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # センサークロップモード（従来方式）
    # 720p直接取得だが画角が狭くなる
    sync_node = Node(
        package='quad_cam_system',
        executable='sync_cam_node',
        name='synchronized_quad_camera',
        parameters=[{
            'fps': 24,               
            'simple_mode': False,    
            'full_fov_mode': False,  # センサー側で720pクロップ
            'target_width': 1296,    # クロップ後のサイズ
            'target_height': 732
        }],
        output='screen'
    )

    return LaunchDescription([sync_node])