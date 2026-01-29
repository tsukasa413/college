# Standalone C++ RGBD Estimator

完全にC++のみで動作する全天球ステレオ深度推定プログラムです。Pythonへの依存は一切ありません。

## 概要

このプログラムは、複数の魚眼カメラから取得した画像を使用して、全方位RGBDパノラマ（RGB画像 + 深度マップ）を生成します。

## 特徴

- **完全C++実装**: Pythonへの依存なし
- **高速処理**: CUDA加速により308ms/frame (最適化後272ms)
- **シンプルなインターフェース**: コマンドライン引数でデータセットパスを指定するだけ
- **標準ライブラリ使用**: OpenCV, Eigen3, LibTorch, nlohmann/json

## ビルド方法

```bash
cd /home/motoken/college/ros2_ws
colcon build --packages-select my_stereo_pkg
```

## 実行方法

### 🚀 **推奨方法: 実行スクリプト使用**

```bash
# パッケージディレクトリから実行
cd /home/motoken/college/ros2_ws/src/my_stereo_pkg
./run_standalone_estimator.sh

# またはワークスペースルートから
cd /home/motoken/college/ros2_ws
./src/my_stereo_pkg/run_standalone_estimator.sh
```

**カスタムパスを指定:**
```bash
./src/my_stereo_pkg/run_standalone_estimator.sh <dataset_path> <output_dir>
```

### 📝 **直接実行（手動）**

基本的な実行:
```bash
cd /home/motoken/college/ros2_ws
LD_LIBRARY_PATH=/home/motoken/.local/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH \
./install/my_stereo_pkg/lib/my_stereo_pkg/standalone_estimator
```

カスタムパスを指定:
```bash
LD_LIBRARY_PATH=/home/motoken/.local/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH \
./install/my_stereo_pkg/lib/my_stereo_pkg/standalone_estimator \
<dataset_path> <output_dir>
```

**引数:**
- `dataset_path` (オプション): データセットディレクトリ (デフォルト: `/home/motoken/college/sphere-stereo/resources`)
- `output_dir` (オプション): 出力ディレクトリ (デフォルト: `/home/motoken/college/ros2_ws/output/standalone`)

## 入力データ構造

```
dataset_path/
├── calibration.json          # カメラキャリブレーションファイル
├── cam0/
│   ├── 0.jpg                # フレーム0の画像
│   └── mask.png             # マスク画像 (オプション)
├── cam1/
│   ├── 0.jpg
│   └── mask.png
├── cam2/
│   ├── 0.jpg
│   └── mask.png
└── cam3/
    ├── 0.jpg
    └── mask.png
```

### calibration.jsonの形式

Basalt形式のキャリブレーションファイルを使用:
- Double Sphere カメラモデル (ds)
- 各カメラの内部パラメータ (fx, fy, cx, cy, xi, alpha)
- 各カメラの外部パラメータ (quaternion + translation)

## 出力

プログラムは以下のファイルを生成します:

1. **rgb_panorama.png** - RGBパノラマ画像 (2048x1024 pixels)
2. **distance_panorama.exr** - 距離パノラマ (float32, メートル単位)
3. **distance_panorama_colored.png** - 可視化用の距離マップ (MAGMA colormap)

## パイプライン構成

1. **キャリブレーション読み込み** - JSON形式からカメラパラメータを取得
2. **画像読み込み** - 各カメラの画像をロードしてリサイズ
3. **マスク読み込み** - 有効領域のマスクを適用
4. **視点計算** - 参照カメラの中心位置を計算
5. **RGBD推定** - `my_stereo_pkg::RGBDEstimator` を使用して深度推定

## パフォーマンス

- **実行時間**: ~272ms/frame (Jetson AGX Orin)
- **スループット**: ~3.7 FPS
- **Pythonからの高速化**: 約37.3倍

## 依存関係

- **OpenCV 4.5+**: 画像I/O
- **Eigen3**: 線形代数演算
- **LibTorch (PyTorch C++ API)**: テンソル演算とCUDAサポート
- **nlohmann/json**: JSON解析 (ヘッダーオンリー)
- **CUDA 12.x**: GPU加速

## トラブルシューティング

### LibTorchライブラリが見つからない

```bash
export LD_LIBRARY_PATH=/home/motoken/.local/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```

### セグメンテーションフォルト

- OpenCVのバージョンを確認してください
- CMakeLists.txtのOpenCVライブラリパスが正しいか確認してください

### メモリ不足エラー

- 解像度を下げるか、`candidate_count`を減らしてください
- プログラム内の定数を調整できます

## 実装の詳細

### 主要ファイル

- `src/standalone_estimator.cpp` - メインプログラム
- `include/my_stereo_pkg/depth_estimation.hpp` - RGBDEstimator定義
- `src/core/depth_estimation.cpp` - 深度推定の実装
- `src/cuda/*.cu` - CUDAカーネル

### アルゴリズム

本実装は以下の論文に基づいています:

> Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images  
> Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim  
> Proc. IEEE CVPR 2021 (Oral)

## ライセンス

MIT License
