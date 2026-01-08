from launch import LaunchDescription
from launch_ros.actions import Node

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
        launch_actions.append(node)

    return LaunchDescription(launch_actions)