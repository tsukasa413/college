# Basalt SLAM System

## 📋 **目次**
1. [Docker環境設定](#docker環境設定)
2. [カメラキャリブレーション](#カメラキャリブレーション)
3. [SLAM実行](#slam実行)
4. [結果確認](#結果確認)
5. [トラブルシューティング](#トラブルシューティング)

---

## Docker環境設定

### 🐳 **Basaltコンテナ起動**
```bash
docker run -it --rm \
    --runtime nvidia \
    --network host \
    --privileged \
    --env DISPLAY=$DISPLAY \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume /dev:/dev \
    --volume ~/docker_sync:/data \
    basalt:jetson /bin/bash
```

### 📁 **データファイル準備**
```bash
# ホスト側で実行（コンテナ起動前）
cd ~/college/ros2_ws/
# モノクロバッグをdocker_syncに移動
mv ./mono_slam_data ~/docker_sync/
# またはROS1形式に変換
rosbags-convert --src ./mono_slam_data/ --dst ~/docker_sync/mono_camera.bag
```

---

## カメラキャリブレーション

### 🎯 **AprilGridキャリブレーション**
```bash
# コンテナ内で実行
/opt/basalt/build/basalt_calibrate \
    --dataset-path /data/mono_camera.bag \
    --dataset-type bag \
    --aprilgrid /opt/basalt/data/aprilgrid_6x6.json \
    --result-path /data/calib_results/ \
    --cam-types ds ds ds ds
```

### 📊 **キャリブレーション設定**
| パラメータ | 値 | 説明 |
|------------|-----|------|
| `--dataset-path` | `/data/mono_camera.bag` | バッグファイルパス |
| `--dataset-type` | `bag` | ROS1バッグ形式 |
| `--aprilgrid` | `/opt/basalt/data/aprilgrid_6x6.json` | グリッド設定 |
| `--result-path` | `/data/calib_results/` | 結果保存先 |
| `--cam-types` | `ds ds ds ds` | 4台のカメラ（Double Sphere Model） |

### 🔧 **カメラモデル種類**
- `pinhole`: ピンホールモデル
- `ds`: Double Sphere（魚眼レンズ推奨）
- `eucm`: Extended Unified Camera Model
- `kb4`: Kannala-Brandt Model

---

## SLAM実行

### 🚀 **Visual-Inertial SLAM**
```bash
# IMU付きSLAM（推奨）
/opt/basalt/build/basalt_vio \
    --dataset-path /data/mono_camera.bag \
    --cam-calib /data/calib_results/calibration.json \
    --dataset-type bag \
    --config-path /opt/basalt/data/euroc_config.json \
    --result-path /data/vio_results/ \
    --save-trajectory
```

### 📹 **Visual-Only SLAM**
```bash
# カメラのみSLAM
/opt/basalt/build/basalt_vo \
    --dataset-path /data/mono_camera.bag \
    --cam-calib /data/calib_results/calibration.json \
    --dataset-type bag \
    --config-path /opt/basalt/data/euroc_config.json \
    --result-path /data/vo_results/ \
    --save-trajectory
```

### 📊 **SLAM設定パラメータ**
| パラメータ | 説明 | 例 |
|------------|------|-----|
| `--cam-calib` | キャリブレーション結果 | `/data/calib_results/calibration.json` |
| `--config-path` | SLAM設定ファイル | `/opt/basalt/data/euroc_config.json` |
| `--save-trajectory` | 軌跡保存フラグ | - |
| `--show-gui` | GUI表示（オプション） | - |

---

## 結果確認

### 📁 **出力ファイル構造**
```
/data/calib_results/
├── calibration.json      # カメラキャリブレーション
├── calib_result.txt      # キャリブレーション詳細
└── images/               # キャリブレーション画像

/data/vio_results/
├── trajectory.txt        # SLAM軌跡
├── poses.txt             # カメラポーズ
└── landmarks.txt         # ランドマーク座標
```

### 📊 **結果可視化**
```bash
# Python可視化スクリプト例
python3 /opt/basalt/scripts/plot_trajectory.py \
    --trajectory /data/vio_results/trajectory.txt \
    --output /data/trajectory_plot.png
```

### 📈 **性能評価**
```bash
# 軌跡精度評価
/opt/basalt/build/basalt_mapper \
    --vocabulary /opt/basalt/data/orbvoc.dbow3 \
    --cam-calib /data/calib_results/calibration.json \
    --dataset-path /data/mono_camera.bag \
    --result-path /data/mapping_results/
```

---

## 設定ファイル

### 🎛️ **カスタム設定作成**
```bash
# コンテナ内でカスタム設定をコピー
cp /opt/basalt/data/euroc_config.json /data/custom_config.json

# 設定編集（ホスト側）
nano ~/docker_sync/custom_config.json
```

### ⚙️ **主要設定項目**
```json
{
  "vio_config": {
    "optical_flow_type": "frame_to_frame",
    "optical_flow_detection_grid_size": 50,
    "optical_flow_max_recovered_dist2": 0.09,
    "vio_max_states": 3,
    "vio_max_kfs": 7,
    "vio_min_frames_after_kf": 5,
    "vio_new_kf_keypoints_thresh": 0.7,
    "vio_debug": false,
    "vio_extended_logging": false,
    "vio_no_motion_regalarization": true
  }
}
```

---

## トラブルシューティング

### ❌ **よくある問題**

#### 1. **キャリブレーションエラー**
```bash
# 問題: AprilGrid検出失敗
# 解決: グリッドサイズ確認・照明改善
--aprilgrid /opt/basalt/data/aprilgrid_4x3.json  # より小さいグリッド
```

#### 2. **バッグファイル読み込みエラー**
```bash
# 問題: rosbag形式不一致
# 解決: rosbags-convert で再変換
rosbags-convert --src /data/mono_slam_data/ --dst /data/mono_camera_new.bag
```

#### 3. **GPU/CUDA エラー**
```bash
# 問題: NVIDIA Runtime未対応
# 解決: CPUモードで実行
/opt/basalt/build/basalt_calibrate --use-cpu
```

#### 4. **メモリ不足**
```bash
# 問題: 大容量バッグファイル
# 解決: バッグファイル分割
rosbag filter input.bag output.bag "t.to_sec() <= 1640000000.0"
```

### 🔧 **デバッグオプション**
```bash
# 詳細ログ出力
export GLOG_v=1
export GLOG_logtostderr=1

# GUI表示でリアルタイム確認
/opt/basalt/build/basalt_vio --show-gui --dataset-path /data/mono_camera.bag
```

---

## ワークフロー例

### 🎯 **完全なSLAMパイプライン**
```bash
# 1. データ準備（ホスト側）
cd ~/college/ros2_ws/
ros2 launch quad_cam_system mono_multi_cam.launch.py &
ros2 bag record -o mono_slam_raw /camera_*/image_mono --storage mcap
rosbags-convert --src ./mono_slam_raw/ --dst ~/docker_sync/mono_data.bag

# 2. Basaltコンテナ起動
docker run -it --rm --runtime nvidia --network host --privileged \
    --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume /dev:/dev --volume ~/docker_sync:/data basalt:jetson /bin/bash

# 3. キャリブレーション（コンテナ内）
/opt/basalt/build/basalt_calibrate \
    --dataset-path /data/mono_data.bag \
    --dataset-type bag \
    --aprilgrid /opt/basalt/data/aprilgrid_6x6.json \
    --result-path /data/calib/ \
    --cam-types ds ds ds ds

# 4. SLAM実行（コンテナ内）
/opt/basalt/build/basalt_vio \
    --dataset-path /data/mono_data.bag \
    --cam-calib /data/calib/calibration.json \
    --dataset-type bag \
    --result-path /data/slam_results/ \
    --save-trajectory

# 5. 結果確認（ホスト側）
ls ~/docker_sync/slam_results/
```

### 📊 **性能比較**
| データ形式 | 処理速度 | メモリ使用量 | 精度 |
|-----------|----------|--------------|------|
| **カラー（BGR8）** | 遅い | 大 | 良 |
| **モノクロ（GRAY8）** | **高速** | **小** | **優秀** |
| **圧縮済み** | 中程度 | 中 | 良 |