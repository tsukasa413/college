from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    launch_actions = []

    for i in range(4):
        node = Node(
            package='quad_cam_system',
            executable='cam_node',
            name=f'camera_node_{i}',
            parameters=[{'sensor_id': i}, {'fps': 32}],
            output='screen'
        )

        # 確実に衝突を避けるため 10秒間隔 に設定
        delayed_node = TimerAction(
            period=float(i) * 10.0,
            actions=[node]
        )
        launch_actions.append(delayed_node)

    return LaunchDescription(launch_actions)