#!/usr/bin/env python3

"""
RealSense D435i Visual-Inertial Odometry Launch File

Purpose:
- RealSense D435i から IR画像 + IMU データのみを取得
- rtabmap_odom でステレオビジュアル・IMUオドメトリを計算
- 魚眼カメラとの帯域競合を避けて軽量に動作

Usage:
    ros2 launch my_stereo_pkg realsense_odom.launch.py

Output Topics:
    /odom (nav_msgs/Odometry): 自己位置推定
    /tf: odom->base_link変換
    
Author: motoken
Date: 2026-02-14
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    
    # Launch引数の定義
    declare_device_type_arg = DeclareLaunchArgument(
        'device_type',
        default_value='d435i',
        description='RealSense device type (d435i, d455, etc)'
    )
    
    declare_enable_debug_arg = DeclareLaunchArgument(
        'enable_debug',
        default_value='false',
        description='Enable debug visualization and info topics'
    )
    
    declare_camera_name_arg = DeclareLaunchArgument(
        'camera_name',
        default_value='camera',
        description='Camera namespace'
    )
    
    declare_infra_fps_arg = DeclareLaunchArgument(
        'infra_fps',
        default_value='30',
        description='IR camera frame rate (15, 30, 60)'
    )
    
    declare_infra_width_arg = DeclareLaunchArgument(
        'infra_width',
        default_value='640',
        description='IR camera width'
    )
    
    declare_infra_height_arg = DeclareLaunchArgument(
        'infra_height', 
        default_value='480',
        description='IR camera height'
    )
    
    # Launch設定値を変数化
    device_type = LaunchConfiguration('device_type')
    enable_debug = LaunchConfiguration('enable_debug')
    camera_name = LaunchConfiguration('camera_name')
    infra_fps = LaunchConfiguration('infra_fps')
    infra_width = LaunchConfiguration('infra_width')
    infra_height = LaunchConfiguration('infra_height')
    
    # ==============================================
    # 1. Static Transform Publisher (TF設定)
    # ==============================================
    # base_link -> camera_link の位置関係を定義
    # 注意: 実際のロボット/カメラマウント位置に合わせて調整してください
    
    static_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_transform',
        arguments=[
            # x, y, z, qx, qy, qz, qw 
            '0.1', '0.0', '0.05',  # RealSenseの位置: 前方10cm, 高さ5cm 
            '0.0', '0.0', '0.0', '1.0',  # 回転なし
            'base_link',  # Parent frame
            'camera_link'  # Child frame  
        ],
        output='screen'
    )
    
    # ==============================================
    # 2. RealSense Camera Driver
    # ==============================================
    # IR画像 + IMU のみ有効、RGB/Depthは無効（帯域節約）
    
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense2_camera_node',
        namespace=camera_name,
        parameters=[{
            # ========== Stream Enable/Disable ==========
            'enable_color': False,           # RGBストリーム無効（帯域節約）
            'enable_depth': False,           # Depthストリーム無効（帯域節約） 
            'enable_infra1': True,           # 左IR画像有効（ステレオVIO用）
            'enable_infra2': True,           # 右IR画像有効（ステレオVIO用）
            'enable_fisheye': False,         # 魚眼無効
            'enable_gyro': True,             # ジャイロ有効（VIO必須）
            'enable_accel': True,            # 加速度センサ有効（VIO必須）
            
            # ========== IR Camera Settings ==========
            'infra_width': infra_width,      # IR解像度 幅
            'infra_height': infra_height,    # IR解像度 高さ
            'infra_fps': infra_fps,          # IRフレームレート
            
            # ========== IMU Settings ==========
            'gyro_fps': 400,                 # ジャイロサンプリング頻度 
            'accel_fps': 250,                # 加速度センササンプリング頻度
            'unite_imu_method': 'linear_interpolation',  # IMU統合方法
            
            # ========== Frame Settings ==========
            'base_frame_id': 'camera_link',
            'depth_frame_id': 'camera_depth_frame',
            'infra_frame_id': 'camera_infra_frame',
            'color_frame_id': 'camera_color_frame',
            'gyro_frame_id': 'camera_gyro_frame',
            'accel_frame_id': 'camera_accel_frame',
            'pose_frame_id': 'camera_pose_frame',
            
            # ========== Performance Settings ==========
            'publish_tf': True,              # camera_link以下のTF発行
            'tf_publish_rate': 30.0,         # TF発行頻度
            
            # ========== Debug Settings ==========  
            'enable_sync': True,             # 同期有効
            'align_depth': False,            # Depth位置合わせ（使わないので無効）
            
            # ========== Auto Exposure ==========
            'enable_auto_exposure': True,    # 自動露出有効
        }],
        output='screen',
        emulate_tty=True,
        respawn=True,
        respawn_delay=2
    )
    
    # ==============================================  
    # 3. RTAB-Map Visual Odometry Node
    # ==============================================
    # ステレオIR + IMU による軽量VIOを計算
    
    rtabmap_odom_node = Node(
        package='rtabmap_odom',
        executable='stereo_odometry',
        name='stereo_odometry',
        output='screen',
        parameters=[{
            # ========== Frame IDs ==========
            'frame_id': 'base_link',         # ロボット中心座標系 
            'odom_frame_id': 'odom',         # オドメトリ座標系
            
            # ========== Stereo Settings ==========
            'stereo': True,                  # ステレオカメラモード
            'subscribe_rgbd': False,         # RGBD無効
            'subscribe_rgb': False,          # RGB無効
            'subscribe_depth': False,        # Depth画像無効
            'subscribe_scan': False,         # LaserScan無効
            'subscribe_scan_cloud': False,   # 点群無効（今回は魚眼用）
            
            # ========== IMU Integration ========== 
            # IMU disabled since not available on this device
            'wait_imu_to_init': False,       # IMU無効のため待機しない
            # 'imu_topic': '/camera/camera/imu',  # IMU無効
            
            # ========== Synchronization ==========
            'approx_sync': True,             # 近似同期（時刻ズレ許容）
            'queue_size': 10,                # メッセージキューサイズ
            
            # ========== Visual Odometry Parameters ==========
            'Odom/Strategy': 0,              # 0=Frame-to-Map, 1=Frame-to-Frame
            'Odom/EstimationType': 1,        # 0=2D, 1=3D (IMU無効のため3D+IMUは使用不可)
            'Odom/ResetCountdown': 1,        # リセット前のフレーム数
            'Odom/Holonomic': False,         # 全方位移動ではない
            
            # ========== Feature Detection ==========
            'Vis/EstimationType': 1,         # 0=PnP, 1=PnPRansac, 2=Epipolar
            'Vis/FeatureType': 0,            # 0=SURF, 1=SIFT, 2=ORB, 3=FAST, etc
            'Vis/MaxFeatures': 500,          # 特徴点最大数（計算負荷調整）
            'Vis/MinInliers': 15,            # 最小インライア数
            
            # ========== Stereo Parameters ==========
            'Stereo/MaxDisparity': 128.0,    # 最大視差
            'Stereo/MinIou': 0.1,           # マッチング閾値
            
            # ========== IMU Parameters (DISABLED) ==========
            # 'Imu/FilterMadgwick': True,      # IMU無効のため無効化
            # 'Imu/GravityNorm': 9.81,         # IMU無効のため無効化
            
            # ========== Debug/Info Settings ==========
            'publish_null_when_lost': True,   # ロスト時nullパブリッシュ
            'guess_from_tf': False,           # TFからの推測無効
            'tf_delay': 0.05,                 # TF遅延(秒)
        }],
        remappings=[
            # RealSenseのIRトピックを接続 (実際のトピック名に修正)
            ('left/image_rect', '/camera/camera/infra1/image_rect_raw'),
            ('right/image_rect', '/camera/camera/infra2/image_rect_raw'),  
            ('left/camera_info', '/camera/camera/infra1/camera_info'),
            ('right/camera_info', '/camera/camera/infra2/camera_info'),
            # IMU is disabled since it's not available
            # ('imu', '/camera/camera/imu'),
        ],
        respawn=True,
        respawn_delay=2
    )
    
    # ==============================================
    # 4. Debug Nodes (Optional)
    # ==============================================
    # enable_debug=true の場合のみ起動
    
    debug_info_node = Node(
        package='rtabmap_odom', 
        executable='stereo_odometry',
        name='stereo_odometry_info',
        condition=IfCondition(enable_debug),
        parameters=[{
            'subscribe_odom_info': True,
        }],
        remappings=[
            ('odom_info', '/stereo_odometry/odom_info'),
        ],
        output='screen'
    )
    
    # ==============================================
    # Launch Description
    # ==============================================
    
    return LaunchDescription([
        # Launch引数
        declare_device_type_arg,
        declare_enable_debug_arg,
        declare_camera_name_arg,
        declare_infra_fps_arg,
        declare_infra_width_arg,
        declare_infra_height_arg,
        
        # ノード群
        static_tf_publisher,
        realsense_node,
        rtabmap_odom_node,
        
        # デバッグノード（条件付き）
        debug_info_node,
    ])