# Sphere Sweeping Stereo - C++ Evaluation Quick Start

## セットアップが完了しました！

評価プログラム `main_eval` が正常にビルドされました。

## 実行方法

### 1. 最も簡単な方法 - 自動評価スクリプト

```bash
cd /home/motoken/college/ros2_ws
python3 scripts/run_evaluation.py --dataset_path /path/to/evaluation_dataset
```

このスクリプトは以下を自動実行します：
1. Python版を実行してリファレンス出力を生成
2. C++版を実行してリファレンスと比較
3. 精度とパフォーマンスの統計を表示
4. 差分ヒートマップを保存

### 2. C++版のみを実行（Python出力が既にある場合）

```bash
cd /home/motoken/college/ros2_ws
scripts/run_main_eval.sh --dataset_path /path/to/evaluation_dataset
```

または

```bash
cd /home/motoken/college/ros2_ws
python3 scripts/run_evaluation.py --dataset_path /path/to/evaluation_dataset --skip_python
```

### 3. 必要な環境変数を手動設定して実行

```bash
cd /home/motoken/college/ros2_ws
source install/setup.bash
export LD_LIBRARY_PATH=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
./install/my_stereo_pkg/lib/my_stereo_pkg/main_eval --dataset_path /path/to/evaluation_dataset
```

## コマンドラインオプション

- `--dataset_path <path>`: データセットディレクトリのパス（デフォルト: `evaluation_dataset`）
- `--min_dist <float>`: 最小距離（デフォルト: 0.55）
- `--max_dist <float>`: 最大距離（デフォルト: 100.0）
- `--visualize`: 可視化を有効化（実行中に画像を表示）

## データセット構造

データセットは以下の構造が必要です：

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

## 出力

### コンソール出力

```
========================================
Sphere Sweeping Stereo - C++ Evaluation
========================================
Dataset: evaluation_dataset
...

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

### ファイル出力

- `evaluation_dataset/eval_output/diff_rgb_*.png`: RGB差分ヒートマップ
- `evaluation_dataset/eval_output/diff_depth_*.png`: 深度差分ヒートマップ

## トラブルシューティング

### エラー: "libtorch.so: cannot open shared object file"

ラッパースクリプトを使用してください：

```bash
scripts/run_main_eval.sh --dataset_path /path/to/dataset
```

または手動でLD_LIBRARY_PATHを設定：

```bash
export LD_LIBRARY_PATH=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
```

### エラー: "CUDA is not available!"

```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

でCUDAが利用可能か確認してください。

### エラー: "Failed to open calibration file"

`calibration.json` がデータセットディレクトリ直下に存在するか確認してください。

## 詳細情報

完全なドキュメントは以下を参照してください：
- `/home/motoken/college/ros2_ws/EVALUATION_README.md`

## 実装の特徴

### 1. Python完全互換の calibration.json パース
- 四元数→回転行列変換がscipy.spatial.transformと完全一致
- matching_scale計算が1ビット単位で一致
- 浮動小数点演算の順序を厳密に再現

### 2. 画像ロードの完全互換
- uint8/uint16/float32の型変換がPythonと同じルール
- BGR→RGB変換を正確に実施
- cv::INTER_AREAリサイズでPythonのcv2.INTER_AREAと等価

### 3. 正確なGPU時間計測
- cudaEvent_tを使用した純粋なカーネル実行時間
- CPU起動オーバーヘッドを除外
- cudaEventSynchronizeで確実に完了を待機

### 4. 詳細な精度検証
- ピクセル単位のRGB/深度比較
- MAE, Max Error, Match Rateを算出
- 差分のヒートマップ可視化

## ライセンス

このコードは以下の論文に基づいています：

```
Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images
Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
CVPR 2021
```

CC BY-NC-SA 3.0 ライセンス
