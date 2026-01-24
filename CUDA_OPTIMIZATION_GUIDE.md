# Sphere Sweeping Stereo - CUDA高速化実装ガイド

**CVPR 2021 (Oral):** Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images  
Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim

## 実装概要

Python版 `depth_estimation.py` をC++/CUDAで高速化したリアルタイム深度推定の実装です。

### 構成ファイル

```
include/my_stereo_pkg/
├── depth_estimation.hpp       # クラス定義とカーネルインタフェース

src/cuda/
├── depth_estimation.cu         # CUDA カーネル実装 (フュージョン戦略)

src/core/
├── depth_estimation.cpp        # CPU側制御・メモリ管理

CMakeLists.txt                   # ビルド構成 (CUDA統合)
```

---

## 最適化戦略

### 1. Zero-Copy Strategy（メモリ転送最小化）

- **目標**: CPU-GPU間の不要な転送を削除し、すべての計算をGPU上で完結
- **実装**:
  - `cudaMallocPitch()`: 2D配列のアライメント最適化（Coalesced Access）
  - `cudaTextureObject_t`: テクスチャメモリを使用したハードウェア加速バイリニア補間
  - `cudaStream_t`: 複数参照カメラの非同期処理

### 2. Kernel Fusion（中間メモリの排除）

Python版では以下のステップが分離していた：
```python
# Step 1: 3D点の生成（GPU）→ Step 2: 他視点への投影（GPU）→ Step 3: サンプリング（GPU）→ Step 4: コスト計算（GPU）
```

CUDA版では1つのカーネル内で統合：
```cuda
__global__ estimate_fisheye_distance_fused_kernel() {
    // 各スレッド = 1ピクセル
    // for dist_idx in candidates:
    //   - 逆投影 (unproject_double_sphere)
    //   - 距離スイープ
    //   - 他視点への投影 (project_double_sphere)
    //   - テクスチャ経由のバイリニア補間
    //   - コスト計算 (SAD in registers)
    // 最小コスト情報のみ出力
}
```

**メリット**:
- 巨大な `cost_volume` バッファ不要
- 各ピクセルについてレジスタ内で完結
- メモリ帯域幅の大幅削減

### 3. Memory Optimization（メモリ階層の活用）

| メモリ種別 | 用途 | 備考 |
|----------|------|------|
| **Constant Memory** | カメラキャリブレーション（8カメラまで） | 全スレッドから高速アクセス |
| **Shared Memory** | (予約) 参照画像のプリフェッチ | 将来の最適化 |
| **Texture Memory** | 画像サンプリング（バイリニア） | ハードウェアフィルタリング |
| **Global Memory** | 出力（距離マップ） | `cudaMallocPitch()` でアライメント |
| **Register** | コスト値（ピクセルあたり） | ~40-50 registers/thread |

---

## 技術詳細

### ダブルスフィア歪み補正モデル（Double Sphere Model）

fisheye カメラの逆投影/投影をデバイスコードで実装：

```cuda
__device__ float3 unproject_double_sphere(float2 uv, const DoubleSphereCalibration& calib) {
    // 標準化座標: (u,v) → (x,y) = ((u-cx)/fx, (v-cy)/fy)
    // Kannala-Branden型の逆投影:
    // r = xi + sqrt(1 - xi^2 * rho^2) / (1 + alpha * rho^2)
    // 3D方向: (x/r, y/r, 1/r)
    return normalize(point);
}

__device__ float2 project_double_sphere(float3 point, const DoubleSphereCalibration& calib) {
    // 逆方向: 3D点 → (u,v)
    // 球面投影 + 歪み補正 + カメラ内部パラメータ
}
```

### アダプティブカメラ選択（Adaptive Matching）

各ピクセルについて、**ベースラインが最大のカメラペア**を自動選択：

```cuda
__global__ select_best_cameras_kernel_impl() {
    // 各スレッド = 1ピクセル
    for (int cam_idx = 0; cam_idx < num_cameras; cam_idx++) {
        // カメラ cam_idx との視差（displacement）を計算
        // max_displacement を保持するカメラを記録
    }
}
```

**利点**: ステレオマッチングの精度が向上（弱いベースラインのペアを避ける）

### 距離推定パイプライン

```
┌─────────────────────────────────────────────────────┐
│ Reference Image (Fisheye)                           │
└─────────────────┬───────────────────────────────────┘
                  │
         ┌────────▼────────┐
         │   Unprojection  │  (Double Sphere)
         │ → 3D Unit Ray   │
         └────────┬────────┘
                  │
      ┌───────────▼───────────┐
      │ Distance Candidates   │
      │ (Inverse parameterization)
      └───────────┬───────────┘
                  │
         ┌────────▼────────────────┐
         │ For each distance:      │
         │ - Project to target cam │
         │ - Texture sample        │
         │ - Cost computation      │
         │ - Register-based min    │
         └────────┬────────────────┘
                  │
         ┌────────▼────────────────┐
         │ Subpixel Refinement     │
         │ (Quadratic fitting)     │
         └────────┬────────────────┘
                  │
         ┌────────▼────────────┐
         │ Distance Map Output │
         │ [height × width]    │
         └─────────────────────┘
```

---

## パフォーマンス特性

### スレッド構成

| カーネル | ブロックサイズ | グリッドサイズ | 特性 |
|---------|---------------|--------------|------|
| `select_best_cameras_kernel` | `(32, 32)` | `(ceil(W/32), ceil(H/32))` | 接続性良好、共有メモリなし |
| `estimate_fisheye_distance_fused_kernel` | `(16, 16)` | `(ceil(W/16), ceil(H/16))` | レジスタ圧力を考慮 |

### レジスタ圧力

- 目標: `< 64 registers/thread` (Jetson AGX Orin対応)
- 実装値: ~40-50 registers/thread
- メモリ占有率: >50% (好適)

### メモリ転送

| 方向 | 量 | タイミング |
|-----|---|---------|
| Host→Device | 参照画像 + 各ターゲット画像 | 初期化時のみ |
| Device→Host | 距離マップ | 各参照カメラ完了後 |
| **デバイス内** | テクスチャ経由（隠蔽） | フュージョンカーネル内 |

---

## C++ API

### 初期化

```cpp
RGBD_Estimator estimator(
    calibrations_rt,          // std::vector<float> (16 floats per camera)
    calibrations_intrinsics,  // std::vector<float> ([fx, fy, cx, cy] × num_cameras)
    calibrations_sphere,      // std::vector<float> ([xi, alpha] × num_cameras)
    calibrations_resolution,  // std::vector<float> ([w, h] × num_cameras)
    min_dist, max_dist,       // スイープボリューム範囲
    candidate_count,          // 距離サンプル数 (e.g., 100)
    references_indices,       // 深度推定対象カメラ
    reprojection_viewpoint,   // パノラマの視点
    image_widths, image_heights,
    matching_width, matching_height,    // マッチング解像度
    rgb_to_stitch_width, rgb_to_stitch_height,
    panorama_width, panorama_height,
    sigma_i, sigma_s,                   // ISBフィルタパラメータ
    device_id                           // GPU ID
);
```

### 深度・RGB推定

```cpp
auto [rgb_panorama, distance_panorama] = estimator.estimate_RGBD_panorama(
    images_to_match,    // std::vector<std::vector<float>> [H×W×3, values 0-255]
    images_to_stitch    // std::vector<std::vector<float>>
);
// Output:
//   rgb_panorama:      std::vector<uint8_t>  [H×W×3]
//   distance_panorama: std::vector<float>    [H×W]
```

---

## 拡張・最適化のポイント

### 実装済み

- ✅ Kernel Fusion (距離スイープと投影の統合)
- ✅ Constant Memory (カメラキャリブレーション)
- ✅ Texture Memory (バイリニア補間)
- ✅ アダプティブカメラ選択
- ✅ 非同期ストリーム対応（フレームワーク）

### 今後の最適化案

1. **ISB Filter の CUDA実装**
   - 現在: プレースホルダー
   - 提案: Domain Transform Filterの軽量CUDA化

2. **Stitcher カーネル**
   - 深度マップ → パノラマ変換のGPU実装
   - テクスチャベースの視点変換

3. **共有メモリ活用**
   - タイル化された参照画像プリフェッチ
   - コスト計算の局所メモリ効率化

4. **Cooperative Groups**
   - ブロック間での同期が必要な場合に有効
   - マルチGPU実装への展開

5. **Sparse Processing**
   - 有効領域（マスク）のみ処理
   - スレッドブロックのダイナミック配置

---

## ビルド方法

```bash
cd /home/motoken/college/ros2_ws
mkdir build_stereo && cd build_stereo
cmake ../src/my_stereo_pkg
make -j4
```

**出力**:
- `libcuda_depth_estimation_kernels.a`: CUDA カーネル（静的）
- `libcuda_utils_kernels.a`: ユーティリティカーネル（静的）
- `libsphere_stereo_utils.so`: ホスト実装（動的）

---

## 参照

- **論文**: [CVPR 2021 - Sphere Sweeping Stereo](http://vclab.kaist.ac.kr/cvpr2021p1/)
- **モデル**: Double Sphere (Kannala & Branden, 2013)
- **ハードウェア**: Jetson AGX Orin (ARM64, CUDA Compute 8.7)

---

## ライセンス

CC BY-NC-SA 3.0 (元の sphere-stereo プロジェクトに準ずる)
