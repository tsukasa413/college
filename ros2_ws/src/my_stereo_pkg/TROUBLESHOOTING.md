# 🔧 トラブルシューティング：点群・深度画像の表示問題

## ❌ 問題：点群が表示されない & 深度画像が真っ黒

### 原因の特定

#### 1. バックグラウンドプロセスの確認

複数のSLAMノードが起動していないか確認：

```bash
ps aux | grep omnidirectional_slam_node | grep -v grep
```

**解決策**：重複プロセスを停止
```bash
# プロセスIDを確認して停止
killall omnidirectional_slam_node

# または特定のPIDを停止
kill <PID>
```

#### 2. カメラの競合エラー

エラーメッセージ例：
```
Error generated. Failed to create CaptureSession
```

**原因**：別のプロセスがカメラを使用中

**解決策**：
```bash
# すべてのArgusカメラプロセスを停止
sudo systemctl restart nvargus-daemon

# 1-2秒待ってから再起動
sleep 2
```

#### 3. 深度値の範囲確認

SLAMノードのログで深度統計を確認： 

```
Frame 30 | Time: 231 ms | Points: 1429339 | Depth: min=0.50 max=15.23 mean=3.45 valid=1800000/2097152 (85.8%)
```

**判定基準**：
- `valid` が 0% → 深度推定が完全に失敗
- `valid` が < 10% → 距離範囲設定が不適切
- `valid` が > 60% → 正常

### 設定の最適化

#### config.jsonの調整

[resources/config.json](resources/config.json)を編集：

```json
{
  "distance": {
    "min": 0.4,        // 最小距離（近すぎると推定失敗）
    "max": 25.0,       // 最大距離
    "candidates": 128  // 距離候補数（多いほど精度高いが遅い）
  },
  "pointcloud": {
    "min_depth": 0.3,  // 点群フィルタリング最小深度
    "max_depth": 50.0  // 点群フィルタリング最大深度
  },
  "publish_rate_hz": 5.0  // パブリッシュ周波数（低いほど安定）
}
```

**推奨値**：
- 室内環境: `min=0.4, max=10.0`
- 屋外環境: `min=0.8, max=25.0`
- 狭い部屋: `min=0.3, max=5.0`

#### RViz2の設定確認

1. **Fixed Frame**：`omnidirectional_camera`に設定

2. **PointCloud2の設定**：
   - Topic: `/omnidirectional/point_cloud` ✓
   - Color Transformer: `RGB8` ✓
   - Size (m): `0.01` （見えない場合は0.05に）
   - Style: `Flat Squares`
   - Decay Time: `0`

3. **深度画像（DepthImage）の設定**：
   - Topic: `/omnidirectional/depth_panorama` ✓
   - **Normalize Range: ☑ チェック必須**
   - Min Value: `0.3`
   - Max Value: `25.0`

## ✅ 完全な起動手順（推奨）

### ステップ1: クリーンスタート

```bash
# すべてのROS 2プロセスを停止
killall -9 omnidirectional_slam_node
killall -9 rviz2

# カメラデーモンを再起動
sudo systemctl restart nvargus-daemon
sleep 2
```

### ステップ2: ビルド（設定変更後）

```bash
cd ~/college/ros2_ws
colcon build --packages-select my_stereo_pkg --symlink-install
source install/setup.bash
```

### ステップ3: 起動（RViz2付き）

```bash
ros2 launch my_stereo_pkg omnidirectional_slam_with_rviz.launch.py
```

または個別に起動：

```bash
# ターミナル1: SLAMノード
ros2 launch my_stereo_pkg omnidirectional_slam.launch.py

# ターミナル2（別ターミナル）: RViz2
source install/setup.bash
ros2 run rviz2 rviz2 -d ~/college/ros2_ws/src/my_stereo_pkg/rviz/omnidirectional_slam.rviz
```

### ステップ4: トピック確認

```bash
# 新しいターミナルで
source install/setup.bash

# トピックリスト
ros2 topic list

# 点群の周波数確認
ros2 topic hz /omnidirectional/point_cloud

# 深度画像のデータ確認
ros2 topic echo /omnidirectional/depth_panorama --once | head -20
```

## 🎯 診断コマンド集

### 深度値の統計確認

SLAMノードのログ（30フレームごと）：
```
Frame 30 | Time: 231 ms | Points: 1429339 | Depth: min=0.50 max=15.23 mean=3.45 valid=1800000/2097152 (85.8%)
```

**読み方**：
- `Points: 1429339` → 生成された点群の数（多いほど良い）
- `min=0.50` → 最小深度値（メートル）
- `max=15.23` → 最大深度値（メートル）
- `valid=1800000/2097152 (85.8%)` → 有効ピクセル率

### カメラストリームの確認

```bash
# カメラデバイスの確認
ls -l /dev/video*

# Argusカメラサービスの状態
sudo systemctl status nvargus-daemon

# GStreamerテスト（カメラ0のみ）
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),width=1944,height=1096,format=NV12' ! \
  nvvidconv ! autovideosink
```

### メモリ使用量の確認

```bash
# GPU メモリ
sudo tegrastats

# システムメモリ
free -h
```

## 🔍 よくある問題と解決策

### 問題1: 点群が表示されるが色がない

**原因**：Color Transformerの設定ミス

**解決策**：
1. RViz2のPointCloud2パネルを選択
2. Color Transformer: `RGB8` に変更
3. `Auto Size` のチェックを外す

### 問題2: 深度画像のノイズが多い

**原因**：フィルタパラメータが弱い

**解決策**：[config.json](resources/config.json)を編集
```json
{
  "filter": {
    "sigma_i": 20.0,  // 強度の重み（大きいほど強くフィルタ）
    "sigma_s": 35.0   // 空間的な重み
  }
}
```

### 問題3: 処理が遅い（< 1 FPS）

**原因1**：距離候補数が多すぎる

**解決策**：
```json
{
  "distance": {
    "candidates": 64  // 128 → 64に削減
  }
}
```

**原因2**：解像度が高すぎる

**解決策**：
```json
{
  "resolution": {
    "matching": [384, 384],    // 512→384に削減
    "panorama": [1536, 768]    // 2048→1536に削減
  }
}
```

### 問題4: カメラ初期化エラー

**エラー**：
```
Failed to create CaptureSession
```

**解決策1**：
```bash
# Argusデーモン再起動
sudo systemctl restart nvargus-daemon
sleep 3
```

**解決策2**：カメラの順次初期化待機時間を延長

omnidirectional_slam_node.cppの修正は不要（デフォルトで2秒待機）

### 問題5: RViz2がクラッシュする

**原因**：点群データが大きすぎる

**解決策**：
1. PointCloud2の`Size (m)`を増やす → 0.02
2. `Decay Time`を0に設定
3. `publish_rate_hz`を下げる → 3.0

## 📊 正常動作の目安

### 期待される性能

| 項目 | 目標値 | 許容範囲 |
|---|---|---|
| 処理時間 | 200-250ms | 150-350ms |
| 点群サイズ | 130-150万点 | 100万点以上 |
| 有効深度率 | 70-90% | 60%以上 |
| パブリッシュ周波数 | 4-5 Hz | 2 Hz以上 |
| GPU使用率 | 60-80% | - |
| メモリ使用量 | 1.5-2.0 GB | 2.5 GB未満 |

### 正常なログ出力例

```
[INFO] [omnidirectional_slam_node]: Initialization complete!
[INFO] [omnidirectional_slam_node]: Starting real-time processing at 5.0 Hz
[INFO] [omnidirectional_slam_node]: Frame 30 | Time: 221 ms | Points: 1430986 | Depth: min=0.52 max=18.34 mean=4.12 valid=1650000/2097152 (78.7%)
[INFO] [omnidirectional_slam_node]: Frame 60 | Time: 219 ms | Points: 1429210 | Depth: min=0.48 max=19.01 mean=4.05 valid=1680000/2097152 (80.1%)
```

## 🆘 それでも解決しない場合

### デバッグモードで起動

```bash
ros2 launch my_stereo_pkg omnidirectional_slam.launch.py --ros-args --log-level DEBUG
```

### トピックの詳細確認

```bash
# 深度画像のエンコーディング確認
ros2 topic echo /omnidirectional/depth_panorama --once | grep encoding

# 点群のフィールド確認
ros2 topic echo /omnidirectional/point_cloud --once | grep -A 10 fields
```

### ログファイルの確認

```bash
ls -lh ~/.ros/log/latest/
cat ~/.ros/log/latest/omnidirectional_slam_node-*.log
```

## 📖 関連ドキュメント

- [RVIZ_GUIDE.md](RVIZ_GUIDE.md) - RViz2の詳細設定
- [README_SLAM.md](README_SLAM.md) - SLAM統合ガイド
- [VERIFICATION.md](VERIFICATION.md) - 動作確認記録
