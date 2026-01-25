# 数値等価性検証 - 使用ガイド

## 概要

Python（PyTorch）実装とC++/CUDA実装のDepth Estimationアルゴリズムが数学的・数値的に等価であることを検証するためのテストスイートです。

## 成果物

### 1. アルゴリズム比較レポート
**ファイル**: [ALGORITHM_COMPARISON.md](ALGORITHM_COMPARISON.md)

Python版とCUDA版の実装を詳細に解析し、以下を提供：
- 関数/カーネルの対応表
- 数学的な式の比較
- 実装差異の特定
- 未実装箇所のリスト

**主な発見**:
- ✅ Double Sphere Model (Unproject/Project) は実装済
- ✅ Adaptive Camera Selection は実装済
- ⚠️ RT行列の方向が逆（要修正）
- ❌ Sphere Sweeping, Cost Computation, ISB Filterが未実装（モック状態）

### 2. 数値等価性検証スクリプト
**ファイル**: [scripts/equivalence_test.py](scripts/equivalence_test.py)

同一の入力データを両実装に投入し、出力の数値差を計算：
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Max Absolute Error
- 標準偏差

**テストフェーズ**:
- **Phase 1**: 幾何変換（Unproject/Project）単体テスト（要CUDA API）
- **Phase 2**: Camera Selection比較（要CUDA API）
- **Phase 3**: フルパイプライン比較（CUDA実装完了後）

### 3. 幾何変換精度テスト
**ファイル**: [scripts/test_geometry.py](scripts/test_geometry.py)

Python実装のDouble Sphere Modelの数値精度を検証：
- 往復変換（Unproject → Project）の誤差測定
- 距離依存性テスト（0.5m～10m）

**結果**: ✅ Max Error < 1e-5（高精度で一致）

## 使用方法

### 環境準備

```bash
cd /home/motoken/college/ros2_ws

# ROS2環境をソース
source install/setup.bash

# Pythonパスを設定
export PYTHONPATH="$PWD/install/my_stereo_pkg/lib:$PYTHONPATH"
```

### テスト実行

#### 1. 幾何変換テスト（Python版のみ）

```bash
python3 scripts/test_geometry.py
```

**期待出力**:
```
======================================================================
  幾何変換 往復テスト (Unproject → Project)
======================================================================
[4] 誤差:
    - Max Error:  7.629395e-06 pixels
    - Mean Error: 7.629395e-07 pixels
    ✓ PASS: 誤差 < 1e-5
```

#### 2. 等価性検証（モック実装確認）

```bash
python3 scripts/equivalence_test.py
```

**現在の出力**:
```
Phase 3: フルパイプラインテスト
[2] CUDA版実行中...
    ✓ CUDA推定完了 (0.023秒)
    [MOCK] 出力: ダミーデータ

[3] PyTorch版実行...
    ⚠️ スキップ（CUDA版がモック実装のため）

[4] 現状サマリー:
    - CUDA版: モック実装
    - PyTorch版: 未実行
    - 比較: 不可能
```

## 検証結果サマリー

### 実装状態

| コンポーネント | Python | CUDA | 数値検証 |
|-------------|--------|------|---------|
| Double Sphere Unproject | ✅ | ✅ | ⏳ CUDA API待ち |
| Double Sphere Project | ✅ | ✅ | ⏳ CUDA API待ち |
| Camera Selection | ✅ | ⚠️ RT逆 | ⏳ CUDA API待ち |
| Sphere Sweeping | ✅ | ❌ 未実装 | ❌ 不可能 |
| Grid Sampling | ✅ | ❌ 未実装 | ❌ 不可能 |
| Cost (SAD) | ✅ | ❌ 未実装 | ❌ 不可能 |
| ISB Filter | ✅ | ❌ 未実装 | ❌ 不可能 |
| Quadratic Fitting | ✅ | ❌ Placeholder | ❌ 不可能 |
| Stitching | ✅ | ❌ 未実装 | ❌ 不可能 |

### Python実装の精度（自己検証）

✅ **Double Sphere Model**: 往復変換誤差 < 1e-5 pixels  
✅ **距離依存性**: 0.5m～10mで誤差なし

### CUDA実装の状態

⚠️ **幾何変換**: 実装済（未検証）  
❌ **コアパイプライン**: モック実装（ダミーデータ返却）

## 次のステップ

完全な数値等価性検証には以下が必要：

### 1. CUDA実装の完成

#### 優先度：高
- [ ] **Texture Object Sampling** (L309-331の実装)
  - `tex2D<uchar4>()` によるBilinear補間
  - 正規化座標の変換
  
- [ ] **Cost Computation** (L320-330の実装)
  - SAD（差分絶対値和）の計算
  - 最小コストの探索

#### 優先度：中
- [ ] **ISB Filter の移植** (最難関)
  - Python実装: `isb_filter.py`
  - CUDA実装: 新規カーネル必要
  - エッジ保存型平滑化

- [ ] **Quadratic Fitting** (L401-407の実装)
  - サブピクセル精度向上
  - 左右コストからの補間

#### 優先度：低
- [ ] **Distance Filtering**
  - ISB Filterの再適用
  
- [ ] **Stitching**
  - 複数魚眼画像のパノラマ合成

### 2. テストAPIの追加

現在不足している単体テスト用API：

```cpp
// depth_estimation.hpp に追加が必要

// Phase 1用
std::pair<std::vector<float>, std::vector<bool>>
test_unproject(const std::vector<float>& uv, int camera_index);

std::pair<std::vector<float>, std::vector<bool>>
test_project(const std::vector<float>& pt_3d, int camera_index);

// Phase 2用
std::pair<std::vector<int>, std::vector<float>>
get_selected_cameras(int reference_index);
```

これらを実装後、`equivalence_test.py` のPhase 1, 2が有効化されます。

### 3. 完全実装後のテスト手順

```bash
# 1. CUDAコードの完全実装
vim ros2_ws/src/my_stereo_pkg/src/cuda/depth_estimation.cu

# 2. 再ビルド
cd ros2_ws
colcon build --packages-select my_stereo_pkg

# 3. 等価性検証
python3 scripts/equivalence_test.py

# 4. 期待される出力
# ======================================================================
#   Distance Map - 数値比較結果
# ======================================================================
#   Mean Squared Error (MSE):    1.234567e-05
#   Mean Absolute Error (MAE):   3.456789e-03
#   Max Absolute Error:          1.234567e-02
#   ✓ PASS: MSE < 1e-4 (高精度で一致)
```

## トラブルシューティング

### Q: "No module named 'sphere_stereo_cuda'"

**A**: CUDAモジュールがビルドされていません。

```bash
cd ros2_ws
colcon build --packages-select my_stereo_pkg
source install/setup.bash
export PYTHONPATH="$PWD/install/my_stereo_pkg/lib:$PYTHONPATH"
```

### Q: Python実装のインポートエラー

**A**: sphere-stereoのパスを確認してください。

```bash
ls /home/motoken/college/sphere-stereo/python/depth_estimation.py
# ファイルが存在することを確認
```

### Q: CUDA out of memory

**A**: 画像解像度を下げるか、バッチサイズを調整してください。

```python
# equivalence_test.py の設定
matching_width, matching_height = 160, 120  # 320x240から縮小
```

## 参考資料

- **論文**: Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images (CVPR 2021 Oral)
- **元実装**: `/home/motoken/college/sphere-stereo/`
- **比較レポート**: [ALGORITHM_COMPARISON.md](ALGORITHM_COMPARISON.md)
- **ROS2統合**: [README_depth_estimation.md](README_depth_estimation.md)

## ライセンス

本検証スクリプトは sphere-stereo プロジェクトと同じライセンス（CC BY-NC-SA 3.0）に従います。

---

**最終更新**: 2025年1月  
**ステータス**: 🚧 幾何変換のみ検証完了、コアパイプライン未実装
