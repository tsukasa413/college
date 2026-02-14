# Omnidirectional RGBD SLAM - RViz2 Configuration Guide

## 🎨 RViz2での可視化

### 1. RViz2の起動

```bash
# ターミナル1: SLAMノード起動
cd ~/college/ros2_ws
source install/setup.bash
ros2 launch my_stereo_pkg omnidirectional_slam.launch.py

# ターミナル2: RViz2起動
source install/setup.bash
ros2 run rviz2 rviz2
```

### 2. RViz2の設定

#### 2.1 Fixed Frameの設定
- **Fixed Frame**: `omnidirectional_camera`

#### 2.2 点群の表示

1. **Add** → **PointCloud2**
   - Topic: `/omnidirectional/point_cloud`
   - Size (m): `0.01` (点のサイズ)
   - Style: `Flat Squares` または `Points`
   - Color Transformer: `RGB8` （カラー点群）
   - Decay Time: `0` （最新のデータのみ表示）

#### 2.3 RGBパノラマの表示

1. **Add** → **Image**
   - Topic: `/omnidirectional/rgb_panorama`
   - Transport: `raw`

#### 2.4 深度パノラマの表示

1. **Add** → **Image**
   - Topic: `/omnidirectional/depth_panorama`
   - Transport: `raw`
   - Normalize Range: チェック

#### 2.5 カメラ情報の表示

1. **Add** → **Camera**
   - Image Topic: `/omnidirectional/rgb_panorama`
   - Camera Info Topic: `/rgb/camera_info`

### 3. 推奨表示設定

#### パノラマビュー（2D）
- RGBパノラマ（左）
- 深度パノラマ（右）

#### 3Dビュー
- 点群表示
- グリッド表示（10m x 10m）
- カメラ矢印表示

### 4. パフォーマンスモニタリング

```bash
# トピック周波数確認
ros2 topic hz /omnidirectional/point_cloud

# 点群サイズ確認
ros2 topic echo /omnidirectional/point_cloud --once | grep "width:"

# メッセージ遅延確認
ros2 topic delay /omnidirectional/point_cloud
```

### 5. トピック一覧

| トピック名 | メッセージ型 | 説明 |
|---|---|---|
| `/omnidirectional/rgb_panorama` | `sensor_msgs/Image` | RGB パノラマ (2048x1024, RGB8) |
| `/omnidirectional/depth_panorama` | `sensor_msgs/Image` | 深度マップ (2048x1024, 32FC1) |
| `/omnidirectional/point_cloud` | `sensor_msgs/PointCloud2` | 3D点群 (XYZRGB, ~143万点) |
| `/rgb/camera_info` | `sensor_msgs/CameraInfo` | カメラ情報 (equirectangular) |

### 6. RViz2設定の保存

設定を保存するには：
1. **File** → **Save Config As...**
2. ファイル名: `omnidirectional_slam.rviz`
3. 保存場所: `~/college/ros2_ws/src/my_stereo_pkg/rviz/`

### 7. 保存した設定で起動

```bash
ros2 run rviz2 rviz2 -d ~/college/ros2_ws/src/my_stereo_pkg/rviz/omnidirectional_slam.rviz
```

## 🔧 トラブルシューティング

### 点群が表示されない場合

1. Fixed Frameが正しく設定されているか確認
   ```bash
   ros2 topic echo /omnidirectional/point_cloud --once | grep "frame_id:"
   ```

2. 点群データが来ているか確認
   ```bash
   ros2 topic hz /omnidirectional/point_cloud
   ```

3. RViz2のPointCloud2設定を確認
   - Topic名が正しいか
   - Size (m)を大きくしてみる（0.05など）

### 画像が表示されない場合

1. トピックが正しくパブリッシュされているか確認
   ```bash
   ros2 topic list | grep panorama
   ```

2. エンコーディングを確認
   ```bash
   ros2 topic echo /omnidirectional/rgb_panorama --once | grep "encoding:"
   ```

### パフォーマンスが悪い場合

1. 点群のダウンサンプリング
   - Size (m)を大きくする
   - Decay Timeを0に設定

2. 画像解像度の変更
   - `config.json`で`panorama_resolution`を調整

3. 処理周波数の調整
   - `config.json`で`publish_rate_hz`を下げる

## 📊 期待される性能

- **処理時間**: 200-250ms/frame
- **点群サイズ**: 約143万点
- **パブリッシュ周波数**: 1-2 Hz（処理時間による）
- **パノラマ解像度**: 2048x1024

## 🚀 次のステップ：RTAB-Map統合

詳細は [README_SLAM.md](README_SLAM.md) を参照してください。
