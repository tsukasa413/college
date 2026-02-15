#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # パッケージパスを取得
    pkg_share = FindPackageShare('my_stereo_pkg')
    
    # 設定ファイルのパス
    imu_odometry_config = PathJoinSubstitution([
        pkg_share, 'config', 'imu_odometry.yaml'
    ])
    
    # 環境変数設定（強いライブラリを使用）
    set_realsense_env = SetEnvironmentVariable(
        'LD_LIBRARY_PATH', 
        '/home/motoken/college/librealsense/build/Release:${LD_LIBRARY_PATH}'
    )
    
    set_rsusb_env = SetEnvironmentVariable(
        'RS2_USB_BACKEND', '1'
    )
    
    # RealSenseカメラノード
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense2_camera_node',
        namespace='camera',
        parameters=[{
            'enable_color': False,
            'enable_depth': False,
            'enable_infra1': True,
            'enable_infra2': True,
            'enable_gyro': True,
            'enable_accel': True,
            'unite_imu_method': 1,
            'gyro_fps': 200.0,
            'accel_fps': 250.0,
            'enable_sync': True
        }],
        output='screen',
        emulate_tty=True
    )
    
    # robot_localization EKF ノード
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        parameters=[imu_odometry_config],
        remappings=[
            ('/odometry/filtered', '/imu_odom')
        ],
        output='screen',
        emulate_tty=True
    )
    
    # Static TF Publisher (必要に応じてカメラ座標系を設定)
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_base_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'camera_link'],
        output='screen'
    )

    return LaunchDescription([
        set_realsense_env,
        set_rsusb_env,
        realsense_node,
        ekf_node,
        static_tf_node
    ])