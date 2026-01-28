# RGBD Panorama Node - 動作テスト手順

## ✅ ビルド完了

```bash
cd ~/college/ros2_ws
colcon build --packages-select my_stereo_pkg
```

**結果**: ビルド成功 ✓

## 📋 動作テスト手順

### 1. 環境セットアップ
```bash
cd ~/college/ros2_ws
source install/setup.bash
```

### 2. カメラシステムの起動（オプション）
実際のカメラを使う場合:
```bash
# ターミナル1
ros2 launch quad_cam_system max_quality_mode.launch.py
```

### 3. RGBDパノラマノードの起動

#### 方法A: Launch ファイル使用
```bash
# ターミナル2（カメラが起動している場合）
ros2 launch my_stereo_pkg rgbd_panorama.launch.py
```

#### 方法B: ノード直接起動
```bash
ros2 run my_stereo_pkg rgbd_panorama_node \
  --ros-args \
  -p calibration_path:=/home/motoken/college/ros2_ws/src/my_stereo_pkg/config/calibration.json \
  -p references_indices:="[0, 2]" \
  -p min_dist:=0.55 \
  -p max_dist:=100.0
```

### 4. ノードの確認
```bash
# 別ターミナル
ros2 node list | grep rgbd
# -> /rgbd_panorama_node

ros2 topic list | grep rgbd
# -> /rgbd_panorama/rgb
# -> /rgbd_panorama/depth
# -> /rgbd_panorama/inv_depth
```

### 5. トピックの購読確認
```bash
# RGB パノラマを確認
ros2 topic echo /rgbd_panorama/rgb --once

# 深度マップを確認
ros2 topic echo /rgbd_panorama/depth --once
```

### 6. 画像の可視化
```bash
# RGBパノラマ表示
ros2 run rqt_image_view rqt_image_view /rgbd_panorama/rgb

# 逆深度マップ表示
ros2 run rqt_image_view rqt_image_view /rgbd_panorama/inv_depth
```

## 🧪 テストデータでの動作確認

実際のカメラが無い場合、rosbagで再生:
```bash
# bagファイルがある場合
ros2 bag play my_slam_dataset

# 別ターミナルでノード起動
ros2 run my_stereo_pkg rgbd_panorama_node \
  --ros-args \
  -p calibration_path:=/path/to/calibration.json
```

## 📊 期待される動作

### 正常起動時のログ
```
[INFO] [rgbd_panorama_node]: === RGBD Panorama Node Starting ===
[INFO] [rgbd_panorama_node]: Calibration: /path/to/calibration.json
[INFO] [rgbd_panorama_node]: References: [0, 2]
[INFO] [rgbd_panorama_node]: Distance range: [0.55, 100.00]
[INFO] [rgbd_panorama_node]: Matching resolution: [1024, 1024]
[INFO] [rgbd_panorama_node]: Panorama resolution: [2048, 1024]
[INFO] [rgbd_panorama_node]: Loading calibrations from: ...
[INFO] [rgbd_panorama_node]: Loaded 4 camera calibrations
[INFO] [rgbd_panorama_node]: Initializing RGBD_Estimator...
[INFO] [rgbd_panorama_node]: RGBD_Estimator initialized successfully
[INFO] [rgbd_panorama_node]: Node initialized successfully. Waiting for images...
```

### フレーム処理時のログ
```
[INFO] [rgbd_panorama_node]: Processing frame 1 (timestamp: xxx.yyy)
[INFO] [rgbd_panorama_node]: Frame 1 processed in 250 ms (4.00 FPS)
```

## ⚠️ トラブルシューティング

### 1. キャリブレーションファイルが見つからない
```bash
# パスを確認
ls /home/motoken/college/ros2_ws/src/my_stereo_pkg/config/calibration.json

# パラメータで明示的に指定
ros2 run my_stereo_pkg rgbd_panorama_node \
  --ros-args \
  -p calibration_path:=/full/path/to/calibration.json
```

### 2. カメラ画像が来ない
```bash
# カメラトピックを確認
ros2 topic list | grep camera
ros2 topic hz /camera_0/image_raw

# ノードが購読しているトピックを確認
ros2 node info /rgbd_panorama_node
```

### 3. CUDAメモリエラー
```bash
# GPUメモリ確認
nvidia-smi

# 解像度を下げて再起動
ros2 run my_stereo_pkg rgbd_panorama_node \
  --ros-args \
  -p matching_resolution:="[512, 512]" \
  -p panorama_resolution:="[1024, 512]"
```

### 4. 同期エラー（画像が揃わない）
```bash
# message_filtersの許容時間を増やす
# rgbd_panorama_node.cpp の SyncPolicy(100) を SyncPolicy(200) に変更
```

## 📈 パフォーマンスモニタリング

```bash
# CPU使用率
htop

# GPU使用率
watch -n 1 nvidia-smi

# トピック周波数
ros2 topic hz /rgbd_panorama/rgb

# メッセージ遅延
ros2 topic delay /rgbd_panorama/rgb
```

## 🎯 次のステップ

1. **実カメラでテスト**: quad_cam_systemと連携
2. **パラメータ調整**: 解像度、距離範囲、候補数
3. **録画**: rosbag2でRGBDデータを保存
4. **可視化**: RViz2でパノラマ表示
5. **SLAM統合**: Basalt等のSLAMシステムに接続

---

**動作確認日**: 2026-01-27
**ビルド環境**: Jetson AGX Orin, ROS2 Humble, CUDA 11.4
