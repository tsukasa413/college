# 最大解像度画像撮影プログラム

このプログラムは、複数のカメラから同時に最大解像度(1944x1096)の画像を撮影します。

## ファイル構成

- `max_resolution_capture_node.py`: 基本の画像撮影ノード
- `max_resolution_capture_service.py`: サービスとして呼び出し可能な画像撮影ノード
- `max_resolution_capture.launch.py`: 基本ノード用のlaunchファイル
- `max_resolution_capture_service.launch.py`: サービスノード用のlaunchファイル

## 特徴

- **最大解像度**: 1944x1096ピクセル（各カメラのセンサーの最大解像度）
- **同時撮影**: 複数のカメラから同じタイミングで画像を撮影
- **高画質保存**: JPEG品質100%で保存
- **ROS2対応**: 撮影した画像をROS2メッセージとしてもパブリッシュ

## ビルド方法

```bash
cd /home/motoken/college/ros2_ws
colcon build --packages-select quad_cam_system
source install/setup.bash
```

## 使用方法

### 1. 基本的な使用方法（自動撮影）

```bash
# Launch the basic capture node
ros2 launch quad_cam_system max_resolution_capture.launch.py

# カスタムパラメータで実行
ros2 launch quad_cam_system max_resolution_capture.launch.py \
    output_dir:=/home/motoken/college/captured_images \
    num_cameras:=4 \
    filename_prefix:=test_capture
```

### 2. サービスとして使用（手動トリガー）

```bash
# Launch the service node
ros2 launch quad_cam_system max_resolution_capture_service.launch.py

# 別のターミナルから撮影をトリガー
ros2 service call /capture_max_resolution_images std_srvs/srv/Trigger
```

### 3. パラメータ説明

- `output_dir`: 画像保存先ディレクトリ (デフォルト: `/tmp/captured_images`)
- `num_cameras`: 使用するカメラ数 (デフォルト: `4`)
- `filename_prefix`: ファイル名のプレフィックス (デフォルト: `max_res`)

### 4. 撮影される画像

- ファイル名形式: `{prefix}_camera{id}_{timestamp}.jpg`
- 例: `max_res_camera0_20260108_143022_123.jpg`
- 解像度: 1944x1096ピクセル
- 品質: JPEG 100%（最高品質）

### 5. ROS2トピック

撮影時に以下のトピックに画像がパブリッシュされます：

- `/camera_0/captured_image`
- `/camera_1/captured_image`
- `/camera_2/captured_image`
- `/camera_3/captured_image`

### 6. ログの確認

```bash
# ノードの状況を確認
ros2 node list
ros2 node info /max_resolution_capture_service

# サービスの確認
ros2 service list | grep capture
ros2 service type /capture_max_resolution_images
```

## トラブルシューティング

### カメラが認識されない場合

```bash
# カメラデバイスの確認
ls /dev/video*

# GStreamerのテスト
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! nvvidconv ! autovideosink
```

### 権限エラーが発生する場合

```bash
# カメラデバイスへのアクセス権限を確認
sudo usermod -a -G video $USER
# 再ログインが必要
```

### メモリ不足エラー

- `num_cameras` パラメータを減らして使用カメラ数を調整
- 不要なプロセスを停止してメモリを確保

## 注意事項

- 最大解像度での撮影はメモリ使用量が大きくなります
- 同時撮影時は一時的にCPU使用率が上昇します  
- 保存先ディレクトリの容量を十分に確保してください（1枚約2-5MB）