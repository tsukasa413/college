# C++ vs Python Evaluation for Sphere Sweeping Stereo

## 概要

このディレクトリには、C++実装とPython実装の正確性とパフォーマンスを比較評価するプログラムが含まれています。

## 構成要素

### 1. `main_eval.cpp`
C++で実装された評価プログラムです。以下の機能を提供します：

- **calibration.json の完全互換パース**: Python版の `utils.parse_json_calib` と同じ数学手順で Calibration 構造体を初期化
- **画像ロードとテンソル変換**: Python版の `read_input_images` に準拠した処理
- **正確性検証**: Python版の出力と比較し、以下の指標を算出
  - MAE (Mean Absolute Error): RGB/深度の平均絶対誤差
  - Max Error: 最大ピクセル誤差とその座標
  - Match Rate: しきい値内に収まるピクセルの割合
- **パフォーマンス計測**: CUDA Event を使用したGPU実行時間の正確な測定
- **可視化**: 差分のヒートマップ画像を生成

### 2. `run_evaluation.py`
評価を自動化するPythonスクリプトです：

1. Python版を実行してリファレンス出力を生成
2. C++版を実行してリファレンスと比較
3. 結果をまとめて表示

## ビルド方法

```bash
cd /home/motoken/college/ros2_ws
colcon build --packages-select my_stereo_pkg --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## 使用方法

### 方法1: 自動評価スクリプトを使用（推奨）

```bash
cd /home/motoken/college/ros2_ws
python3 scripts/run_evaluation.py --dataset_path /path/to/evaluation_dataset
```

オプション:
- `--dataset_path`: データセットのパス（デフォルト: `evaluation_dataset`）
- `--min_dist`: 最小距離（デフォルト: 0.55）
- `--max_dist`: 最大距離（デフォルト: 100.0）
- `--skip_python`: Python版の実行をスキップ（既存の出力を使用）
- `--visualize`: C++版で可視化を有効化

### 方法2: 手動で実行

#### Step 1: Python版でリファレンス出力を生成

```bash
cd /home/motoken/college/sphere-stereo/python
python3 main.py \
  --dataset_path /path/to/evaluation_dataset \
  --references_indices 0 2 \
  --min_dist 0.55 \
  --max_dist 100 \
  --candidate_count 32 \
  --sigma_i 10 \
  --sigma_s 25 \
  --matching_resolution 1024 1024 \
  --rgb_to_stitch_resolution 1216 1216 \
  --panorama_resolution 2048 1024 \
  --device cuda:0 \
  --saving True \
  --visualize False \
  --evaluate False
```

出力: `evaluation_dataset/output/rgb_*.png` と `inv_distance_*.tif`

#### Step 2: C++版で評価を実行

```bash
cd /home/motoken/college/ros2_ws
source install/setup.bash
./install/my_stereo_pkg/lib/my_stereo_pkg/main_eval \
  --dataset_path /path/to/evaluation_dataset \
  --min_dist 0.55 \
  --max_dist 100
```

出力: 
- コンソールに精度とパフォーマンスの統計情報
- `evaluation_dataset/eval_output/diff_rgb_*.png`: RGB差分ヒートマップ
- `evaluation_dataset/eval_output/diff_depth_*.png`: 深度差分ヒートマップ

## データセット構造

評価用データセットは以下の構造が必要です：

```
evaluation_dataset/
├── calibration.json          # カメラキャリブレーション
├── cam0/
│   ├── 0.jpg                # 入力画像
│   ├── mask.png             # マスク（オプション）
│   └── ...
├── cam1/
│   ├── 0.jpg
│   └── ...
├── cam2/
│   ├── 0.jpg
│   └── ...
└── output/                   # Python版の出力（自動生成）
    ├── rgb_0.png
    ├── inv_distance_0.tif
    └── ...
```

## 出力の解釈

### コンソール出力例

```
========================================
Sphere Sweeping Stereo - C++ Evaluation
========================================
Dataset: evaluation_dataset
Distance range: [0.55, 100]
Candidates: 32
Matching resolution: 1024x1024
Panorama resolution: 2048x1024

Processing: 0.jpg
  C++ execution time: 45.3 ms
  RGB Comparison:
    MAE:        0.523
    Max Error:  5.0 at (1024, 512)
    Match Rate: 99.8% (threshold=±1)
  Depth Comparison:
    MAE:        0.0012
    Max Error:  0.025 at (856, 324)
    Match Rate: 98.5% (threshold=0.01)

========================================
Summary Statistics (1 frames)
========================================
Performance:
  Average GPU time:    45.3 ms/frame
  Average FPS:         22.1

RGB Accuracy (vs Python):
  Average MAE:         0.523
  Average Max Error:   5.0
  Average Match Rate:  99.8%

Depth Accuracy (vs Python):
  Average MAE:         0.0012
  Average Max Error:   0.025
  Average Match Rate:  98.5%
```

### 指標の意味

- **GPU Time**: カーネル実行の純粋な時間（cudaEventで計測）
- **FPS**: 1秒あたりの処理フレーム数
- **MAE**: 平均絶対誤差（低いほど良い）
- **Max Error**: 最大誤差（位置情報付き）
- **Match Rate**: しきい値内のピクセル割合（高いほど良い）
  - RGB: ±1 の範囲内（0-255スケール）
  - Depth: 0.01 の範囲内（逆距離単位）

## 実装の詳細

### calibration.json のパース精度

C++版は以下を保証します：

1. **四元数→回転行列変換**: scipy.spatial.transform.Rotation と同じアルゴリズム
2. **matching_scale 計算**: 浮動小数点演算の順序とキャストを完全一致
3. **JSON解析**: nlohmann/json を使用して Python の json.load と同じ精度

### 画像ロードの互換性

C++版は以下を実装：

1. **型変換**: uint8/uint16/float32 を Python と同じルールで [0-255] に正規化
2. **BGR→RGB変換**: OpenCV のデフォルト BGR を RGB に変換
3. **リサイズ**: cv::INTER_AREA（Python の cv2.INTER_AREA と等価）
4. **テンソル形状**: [H, W, 3] (HWC形式) で一貫性を保つ

### GPU時間計測の正確性

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// ... GPU処理 ...
cudaEventRecord(stop);
cudaEventSynchronize(stop);  // 重要: カーネル完了を待機

float gpu_time_ms;
cudaEventElapsedTime(&gpu_time_ms, start, stop);
```

CPU側の `std::chrono` ではなく、CUDA Event を使用することで、カーネル起動のオーバーヘッドを除外した純粋なGPU実行時間を計測します。

## トラブルシューティング

### エラー: "CUDA is not available!"

LibTorch が CUDA サポート付きでビルドされているか確認してください：

```python
python3 -c "import torch; print(torch.cuda.is_available())"
```

### エラー: "Failed to open calibration file"

`calibration.json` がデータセットディレクトリ直下に存在するか確認してください。

### エラー: "Python reference output not found"

先に Python版を実行してリファレンス出力を生成してください：

```bash
python3 scripts/run_evaluation.py --dataset_path /path/to/dataset
```

または `--skip_python` オプションを外してください。

### 警告: "Cannot read image for file ..."

- 画像ファイルが存在するか確認
- ファイル名が `calibration.json` 内のカメラ数と一致するか確認
- 画像形式が OpenCV でサポートされているか確認（jpg, png, tif など）

## 注意事項

1. **テンソル形状**: Python版は [H, W, 3]（HWC形式）を使用。PyTorch の一部操作は [C, H, W]（CHW形式）を期待するため、`permute` の前後でコメントを確認してください。

2. **浮動小数点精度**: GPU演算の順序やアトミック操作により、わずかな誤差（< 1e-5）が発生する可能性があります。これは正常です。

3. **メモリ使用量**: 高解像度画像や多数のカメラでは GPU メモリが不足する可能性があります。`--matching_resolution` を調整してください。

## ライセンス

このコードは以下の論文に基づいています：

```
Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images
Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
CVPR 2021
```

CC BY-NC-SA 3.0 ライセンスに従います。
