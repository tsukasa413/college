# RGBD Panorama Node

## 概要
4つの魚眼カメラからRGBDパノラマをリアルタイムで生成するROS2ノード。
sphere-stereoのPython実装をC++/CUDAで再実装し、ROS2に統合。

## システム構成

```
quad_cam_system (max_quality_mode)
  ├─ /camera_0/image_raw
  ├─ /camera_1/image_raw
  ├─ /camera_2/image_raw
  └─ /camera_3/image_raw
              ↓
    rgbd_panorama_node
      (C++/CUDA処理)
              ↓
  ├─ /rgbd_panorama/rgb         (RGB パノラマ画像)
  ├─ /rgbd_panorama/depth       (深度マップ)
  └─ /rgbd_panorama/inv_depth   (逆深度マップ)
```

## セットアップ

### 1. キャリブレーションファイルの準備
Basalt形式のキャリブレーションJSONファイルを配置:
```bash
cd ~/college/ros2_ws/src/my_stereo_pkg/
mkdir -p config
# calibration.jsonを config/ フォルダに配置
```

### 2. ビルド
```bash
cd ~/college/ros2_ws/
colcon build --packages-select my_stereo_pkg
source install/setup.bash
```

## 起動方法

### カメラシステムの起動
```bash
# ターミナル1: カメラノードを起動
cd ~/college/ros2_ws/
source install/setup.bash
ros2 launch quad_cam_system max_quality_mode.launch.py
```

### RGBDパノラマノードの起動
```bash
# ターミナル2: パノラマ生成ノードを起動
cd ~/college/ros2_ws/
source install/setup.bash
ros2 launch my_stereo_pkg rgbd_panorama.launch.py
```

### 結果の表示
```bash
# ターミナル3: RGBパノラマを表示
ros2 run rqt_image_view rqt_image_view /rgbd_panorama/rgb

# 深度マップを表示
ros2 run rqt_image_view rqt_image_view /rgbd_panorama/inv_depth
```

## パラメータ

launchファイルで以下のパラメータを調整可能:

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `calibration_path` | `config/calibration.json` | キャリブレーションファイルのパス |
| `references_indices` | `[0, 2]` | リファレンスカメラのインデックス |
| `min_dist` | `0.55` | 最小距離 [m] |
| `max_dist` | `100.0` | 最大距離 [m] |
| `candidate_count` | `32` | 深度候補数 |
| `sigma_i` | `10.0` | ISBフィルタの輝度シグマ |
| `sigma_s` | `25.0` | ISBフィルタの空間シグマ |
| `matching_resolution` | `[1024, 1024]` | マッチング解像度 |
| `panorama_resolution` | `[2048, 1024]` | パノラマ解像度 |

## アルゴリズム

### 処理フロー
1. **画像取得**: 4カメラから同期画像を取得
2. **前処理**: リサイズ + BGR→RGB変換
3. **深度推定**: Sphere Sweeping Stereo (CUDA)
   - コストボリューム計算
   - ISBフィルタ (多解像度bilateral filter)
4. **ステッチング**: マルチビューパノラマ合成
   - 距離マップ再投影
   - インペイント
   - ブレンディング
5. **パブリッシュ**: RGBパノラマ + 深度マップ

### 主要モジュール
- `RGBD_Estimator`: 深度推定エンジン
- `ISBFilter`: 階層的bilateral filter
- `Stitcher`: パノラマステッチング
- `CalibrationParser`: Basalt形式読み込み

## トラブルシューティング

### カメラ画像が来ない
```bash
# トピックの確認
ros2 topic list | grep camera

# 画像の確認
ros2 topic echo /camera_0/image_raw --no-arr
```

### CUDAエラー
```bash
# GPUメモリの確認
nvidia-smi

# プロセスの再起動
sudo systemctl restart nvargus-daemon
```

### フレームレートが低い
- `matching_resolution` を下げる (e.g., `[512, 512]`)
- `candidate_count` を減らす (e.g., `16`)

## パフォーマンス

Jetson AGX Orin (CUDA 87):
- 解像度: 1024x1024 → 2048x1024パノラマ
- 処理時間: ~200-500ms/frame
- GPU使用率: ~60-80%

## 参考文献

```bibtex
@InProceedings{Meuleman_2021_CVPR,
    author = {Andreas Meuleman and Hyeonjoong Jang and Daniel S. Jeon and Min H. Kim},
    title = {Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images},
    booktitle = {CVPR},
    year = {2021}
}
```
