# RealSense D435i Visual-Inertial Odometry

RealSense D435iを使用した軽量な視覚慣性オドメトリシステム

## システム構成

```
RealSense D435i (IR+IMU) → realsense2_camera → rtabmap_odom → /odom + TF
```

## 特徴

- **軽量**: RGB/Depthストリームを無効化し、IR画像+IMUのみ使用
- **高速**: ステレオビジュアルオドメトリで安定した自己位置推定
- **帯域節約**: 魚眼カメラとの併用時に帯域競合を回避

## 使用方法

### 1. パッケージビルド

```bash
cd ~/college/ros2_ws
colcon build --packages-select my_stereo_pkg
source install/setup.bash
```

### 2. オドメトリ起動

```bash
# 基本起動
ros2 launch my_stereo_pkg realsense_odom.launch.py

# デバッグモード有効
ros2 launch my_stereo_pkg realsense_odom.launch.py enable_debug:=true

# IRフレームレート変更（60fps）
ros2 launch my_stereo_pkg realsense_odom.launch.py infra_fps:=60
```

### 3. 可視化

```bash
# RViz2でオドメトリ可視化
ros2 run rviz2 rviz2 -d /path/to/my_stereo_pkg/rviz/realsense_odom.rviz

# オドメトリデータ確認
ros2 topic echo /odom

# TF確認
ros2 run tf2_tools view_frames
```

## Launch引数

| 引数 | デフォルト | 説明 |
|------|------------|------|
| `device_type` | `d435i` | RealSenseデバイス型番 |
| `camera_name` | `camera` | カメラ名前空間 |
| `infra_fps` | `30` | IRカメラフレームレート (15/30/60) |
| `infra_width` | `640` | IR画像幅 |
| `infra_height` | `480` | IR画像高さ |
| `enable_debug` | `false` | デバッグ情報有効 |

## 出力トピック

| トピック | 型 | 説明 |
|----------|-----|-----|
| `/odom` | `nav_msgs/Odometry` | 自己位置推定 |
| `/tf` | `tf2_msgs/TFMessage` | odom→base_link変換 |
| `/camera/infra1/image_rect_raw` | `sensor_msgs/Image` | 左IR画像 |
| `/camera/infra2/image_rect_raw` | `sensor_msgs/Image` | 右IR画像 |
| `/camera/imu` | `sensor_msgs/Imu` | IMUデータ |

## TF Tree

```
odom
└── base_link
    └── camera_link
        ├── camera_infra_frame
        ├── camera_gyro_frame
        ├── camera_accel_frame
        └── ...
```

## 設定調整

### カメラ位置調整

`realsense_odom.launch.py` の `static_tf_publisher` 部分を編集:

```python
# base_link からRealSenseへの変位 (x, y, z, qx, qy, qz, qw)
arguments=[
    '0.1', '0.0', '0.05',  # 前方10cm, 右0cm, 上5cm
    '0.0', '0.0', '0.0', '1.0',  # 回転なし
    'base_link', 'camera_link'
]
```

### オドメトリパラメータ調整

特徴点数やアルゴリズム設定を調整する場合は、`rtabmap_odom_node` の `parameters` を編集。

## トラブルシューティング

### 1. オドメトリが出力されない

```bash
# IMU起動確認
ros2 topic echo /camera/imu --once

# IR画像確認
ros2 topic echo /camera/infra1/image_rect_raw --once
```

### 2. TF変換エラー

- `static_transform_publisher` の座標値を確認
- ロボット中心とカメラの物理的位置関係を正確に測定

### 3. 特徴点不足でロスト

- IRプロジェクタ有効化: `enable_emitter: true`
- 特徴点数増加: `Vis/MaxFeatures: 1000`

## 次のステップ

1. ✅ オドメトリ実装完了
2. 🔄 魚眼カメラ点群との統合  
3. 🔄 rtabmap_slam によるSLAM完成

## パフォーマンス

- **CPU使用率**: ~15-25%（Jetson AGX Orin）
- **メモリ使用**: ~200MB  
- **更新頻度**: 30Hz（IRフレームレート依存）
- **遅延**: ~30-50ms