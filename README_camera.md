# カメラについて

## 📋 **目次**
1. [基本操作](#基本操作)
2. [マルチカメラシステム](#camera-multi-launch)
3. [データ録画](#bag)
4. [バッグファイル変換（カラー→モノクロ）](#バッグファイル変換カラーモノクロ)

---

## 基本操作
### カメラ映像を見たいとき
```
sudo nvpmodel -m 0
sudo jetson_clocks
eCAM_argus_camera 
```
### カメラを増やしたいとき
```bash
cd ~/Documents/e-CAM80_CUOAGX/e-CAM80_CUOAGX/e-CAM80_CUOAGX_JETSON_AGX_ORIN_L4T36.3.0_04-JUNE-2024_R03/
sudo chmod +x ./install_binaries.sh
sudo -E ./install_binaries.sh
```
⚠️ **注意**: 自動でRebootします

### カメラを確認する
#### リストで確認
```bash
sudo dmesg | grep -i "Detected eimx415 sensor"
```
#### どこに出力されているかで確認
```bash
ls /dev/video*
```

---

# camera multi launch

## 📊 **利用可能なモード**
| モード | 解像度 | 特徴 | 用途 |
|--------|--------|------|------|
| **通常版** | 720p | 各カメラ独立 | テスト・デバッグ |
| **同期版** | 1080p+ | フレーム同期 | 高品質SLAM |
| **モノクロ版** | 1944x1096 | グレースケール | 軽量SLAM |

## service reset
```bash
# プロセスをクリーンアップ
sudo pkill -9 gst-launch-1.0
sudo pkill -9 python3

# カメラサービス再起動
sudo systemctl restart nvargus-daemon

# サービスの立ち上がりを少し待つ
sleep 3
```

## launch
### 🔧 **事前準備**
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
ls /dev/video*
```

### 🎬 **起動方法**
#### 🔸 **通常版**（各カメラ独立）
```bash
cd ~/college/ros2_ws/
colcon build --symlink-install
source install/setup.bash
ros2 launch quad_cam_system multi_cam.launch.py
```

#### 🔸 **同期版**（フレーム同期）
```bash
cd ~/college/ros2_ws/
colcon build --symlink-install
source install/setup.bash
ros2 launch quad_cam_system max_quality_mode.launch.py
```

#### 🔸 **モノクロ版**（グレースケール同期・最大解像度）
```bash
cd ~/college/ros2_ws/
colcon build --symlink-install
source install/setup.bash
ros2 launch quad_cam_system mono_multi_cam.launch.py
```

---
## bag

### 📹 **データ録画**

#### 🎨 **カラー版**
```bash
cd ~/college/ros2_ws/
rm -rf my_slam_dataset/
ros2 bag record -o my_slam_dataset \
  /camera_0/image_raw \
  /camera_1/image_raw \
  /camera_2/image_raw \
  /camera_3/image_raw \
  --storage mcap
```

#### ⚫ **モノクロ版**
```bash
cd ~/college/ros2_ws/
rm -rf my_slam_dataset/
ros2 bag record -o my_slam_dataset \
  /camera_0/image_mono \
  /camera_1/image_mono \
  /camera_2/image_mono \
  /camera_3/image_mono \
  --storage mcap
```

### 📦 **ROS1形式への変換**
```bash
cd ~/college/ros2_ws/
rm -rf my_camera_dataset.bag 
rosbags-convert --src ./my_slam_dataset/ --dst ./my_camera_dataset.bag
rm -rf ~/docker_sync/my_camera_dataset.bag
mv ./my_camera_dataset.bag ~/docker_sync/
```

### 📦 **ROS1形式への変換**
```bash
cd ~/college/ros2_ws/
rm -rf my_camera_dataset.bag 
rosbags-convert --src ./my_slam_dataset/ --dst ./my_camera_dataset.bag
rm -rf ~/docker_sync/my_camera_dataset.bag
mv ./my_camera_dataset.bag ~/docker_sync/
```

---

# バッグファイル変換（カラー→モノクロ）

## 📋 **概要**
カラーバッグファイルをモノクロに変換してSLAM精度向上とデータ量削減を実現。

## ⭐ **推奨方法: 直接モノクロ録画**
最も簡単で確実な方法は、最初からモノクロカメラで録画することです：

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

## 🛠️ **既存バッグの変換（上級者向け）**

⚠️ **注意**: rosbags ライブラリのAPI変更により、画像変換スクリプトは環境によって動作しない場合があります。

### 代替手段
```bash
# 1. カラーバッグをROS2で再生
ros2 bag play my_slam_dataset

# 2. 別ターミナルでモノクロカメラの変換ノードを起動
# （別途image_transportパッケージを使用）
ros2 run image_transport republish compressed in:=/camera_0/image_raw out:=/camera_0/image_mono

# 3. モノクロ版を録画
ros2 bag record -o mono_converted \
  /camera_0/image_mono \
  /camera_1/image_mono \
  /camera_2/image_mono \
  /camera_3/image_mono \
  --storage mcap
```

## 📁 **ファイル移動**
```bash
# モノクロバッグをDocker syncフォルダに移動
mv ./mono_slam_data ~/docker_sync/
# またはROS1形式に変換してBasaltで使用
rosbags-convert --src ./mono_slam_data/ --dst ~/docker_sync/mono_camera.bag
```

💡 **Basaltでの使用方法**: 詳細は [`README_basalt.md`](./README_basalt.md) を参照

## 📊 **データ削減効果**
| 項目 | カラー版 | モノクロ版 | 削減率 |
|------|----------|------------|--------|
| **エンコーディング** | BGR8 | GRAY8 | - |
| **データ量** | 100% | **33%** | **67%削減** |
| **ファイルサイズ** | 大 | **小** | **約1/3** |
| **SLAM適用性** | 良 | **最適** | **処理高速化** |

## 🎯 **推奨ワークフロー**
1. **🎬 データ収集**: `mono_multi_cam.launch.py` でモノクロ録画
2. **📊 品質確認**: `ros2 bag info` でデータ確認  
3. **🚀 SLAM実行**: Basaltなどでモノクロバッグを使用
4. **📈 結果比較**: カラー版と精度・速度を比較

## 💡 **トラブルシューティング**
- **rosbags エラー**: 直接モノクロ録画を推奨
- **API変更**: rosbags ライブラリのバージョン依存
- **大容量ファイル**: モノクロ化で約67%削減効果あり