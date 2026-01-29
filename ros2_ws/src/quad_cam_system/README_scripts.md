# Quad Camera System - バッグ変換スクリプト

## 📋 概要

このディレクトリには、ROS2バッグファイルの変換スクリプトが含まれています。

## 📁 スクリプト一覧

### 🔄 **convert_bag_simple.py**
ROS2バッグファイルの構造を単純コピー

**使用方法:**
```bash
cd /home/motoken/college/ros2_ws
python3 src/quad_cam_system/scripts/convert_bag_simple.py <input_bag_dir> <output_bag_dir>
```

**例:**
```bash
python3 src/quad_cam_system/scripts/convert_bag_simple.py my_slam_dataset my_slam_mono
```

---

### 🎨 **convert_bag_to_mono.py**
ROS2バッグファイルをカラーからモノクロ（mono8）に変換

**使用方法:**
```bash
cd /home/motoken/college/ros2_ws
python3 src/quad_cam_system/scripts/convert_bag_to_mono.py <input_bag> <output_bag>
```

**例:**
```bash
python3 src/quad_cam_system/scripts/convert_bag_to_mono.py ./my_slam_dataset ./my_slam_mono_dataset
```

---

### 📦 **convert_ros1_bag_to_mono.py**
ROS1バッグファイルをモノクロに変換

**使用方法:**
```bash
cd /home/motoken/college/ros2_ws
python3 src/quad_cam_system/scripts/convert_ros1_bag_to_mono.py <input.bag> <output.bag>
```

**例:**
```bash
python3 src/quad_cam_system/scripts/convert_ros1_bag_to_mono.py my_slam_ros1.bag my_slam_mono.bag
```

---

### 🔧 **simple_convert_ros1.py**
簡易的なROS1バッグ変換ツール

**使用方法:**
```bash
cd /home/motoken/college/ros2_ws
python3 src/quad_cam_system/scripts/simple_convert_ros1.py <input_bag> <output_bag>
```

---

## 🎯 推奨ワークフロー

### カラーバッグをモノクロに変換してBasaltで使用

1. **ROS2バッグを録画:**
```bash
cd ~/college/ros2_ws/
ros2 bag record -o my_slam_dataset \
  /camera_0/image_raw \
  /camera_1/image_raw \
  /camera_2/image_raw \
  /camera_3/image_raw \
  --storage mcap
```

2. **モノクロに変換:**
```bash
python3 src/quad_cam_system/scripts/convert_bag_to_mono.py \
  ./my_slam_dataset \
  ./my_slam_mono_dataset
```

3. **ROS1形式に変換:**
```bash
rosbags-convert --src ./my_slam_mono_dataset/ --dst ./my_camera_dataset.bag
```

4. **Dockerフォルダに移動:**
```bash
mv ./my_camera_dataset.bag ~/docker_sync/
```

---

## ⚠️ 注意事項

### rosbags APIの変更
- `rosbags`ライブラリはバージョンによってAPIが変更される可能性があります
- エラーが発生する場合は、最初からモノクロカメラで録画することを推奨します

### 最初からモノクロで録画（推奨）
```bash
# モノクロ版カメラを起動
cd ~/college/ros2_ws/
ros2 launch quad_cam_system mono_multi_cam.launch.py

# 別ターミナルでモノクロデータを録画
ros2 bag record -o mono_slam_data \
  /camera_0/image_mono \
  /camera_1/image_mono \
  /camera_2/image_mono \
  /camera_3/image_mono \
  --storage mcap
```

---

## 📊 データ削減効果

| 項目 | カラー版 | モノクロ版 | 削減率 |
|------|----------|------------|--------|
| **エンコーディング** | BGR8 | GRAY8 | - |
| **データ量** | 100% | **33%** | **67%削減** |
| **SLAM適用性** | 良 | **最適** | **処理高速化** |

---

## 🔗 関連ドキュメント

- **カメラ操作全般**: `/home/motoken/college/README_camera.md`
- **Basalt使用方法**: `/home/motoken/college/README_basalt.md`
- **Quad Camera System**: 親ディレクトリのREADME

---

## 💡 トラブルシューティング

### ImportError: rosbags
```bash
pip3 install rosbags
```

### API変更エラー
最新のrosbags APIに合わせてスクリプトを修正するか、直接モノクロ録画を使用してください。

### メモリ不足
大容量バッグファイルの場合、処理中にメモリ不足になる可能性があります。その場合は分割して処理してください。

---

**作成日**: 2026年1月29日  
**パッケージ**: quad_cam_system  
**ライセンス**: MIT
