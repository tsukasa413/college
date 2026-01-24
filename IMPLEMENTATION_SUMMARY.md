# CUDA高速化実装 - 球形ステレオ深度推定

## 概要

CVPR 2021 (Oral) 論文「Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images」のPython実装をC++/CUDAで高速化しました。

**実装規模**: 1,186行コード
- `depth_estimation.cu`: 428行（CUDAカーネル）
- `depth_estimation.cpp`: 491行（C++ラッパー・メモリ管理）
- `depth_estimation.hpp`: 267行（クラス定義）

## 主な最適化

### ✅ Kernel Fusion（カーネル融合）

PyTorchでは複数ステップに分かれていた処理を1つのCUDAカーネルに統合：

```
Python版:  逆投影 → 投影 → グリッドサンプル → SAD計算 (各ステップでGPUメモリ往来)
CUDA版:    1つのカーネル内ですべて実行（レジスタ内で完結）
```

**メリット**: 
- 中間バッファ（cost_volume）不要 → メモリ節約
- メモリ帯域幅削減 → 高速化

### ✅ Zero-Copy Strategy（ゼロコピー戦略）

- GPU上ですべての計算を完結
- 必要な転送は初期化時の画像データのみ
- `cudaMallocPitch()` による2D配列アライメント最適化

### ✅ Memory Optimization（メモリ最適化）

| メモリ種別 | 用途 |
|----------|------|
| **Constant Memory** | 8カメラのキャリブレーション (全スレッド共有) |
| **Texture Memory** | 画像サンプリング（ハードウェアバイリニア補間）|
| **Register** | コスト計算値（ピクセルあたり ~40-50 reg） |
| **Global Memory** | 出力距離マップ（cudaMallocPitch で最適化） |

## ファイル構成

```
include/my_stereo_pkg/
└── depth_estimation.hpp        # クラス定義・カーネルインタフェース

src/cuda/
└── depth_estimation.cu          # CUDA カーネル実装

src/core/
└── depth_estimation.cpp         # CPU側制御・メモリ管理

CMakeLists.txt                    # CUDA統合ビルド
```

## ビルド

```bash
cd /home/motoken/college/ros2_ws
mkdir -p build_stereo && cd build_stereo
cmake ../src/my_stereo_pkg
make -j4
```

**出力ライブラリ**:
- `libcuda_depth_estimation_kernels.a` (26 KB)
- `libcuda_utils_kernels.a` (19 KB)
- `libsphere_stereo_utils.so` (1.2 MB)

## 使用方法

```cpp
#include "my_stereo_pkg/depth_estimation.hpp"

// 初期化
RGBD_Estimator estimator(
    calibrations_rt,          // std::vector<float> (16 floats/camera)
    calibrations_intrinsics,  // std::vector<float> (4 floats/camera)
    calibrations_sphere,      // std::vector<float> (2 floats/camera)
    calibrations_resolution,  // std::vector<float> (2 floats/camera)
    min_dist, max_dist, candidate_count,
    references_indices, reprojection_viewpoint,
    image_widths, image_heights,
    matching_width, matching_height,
    rgb_stitch_width, rgb_stitch_height,
    panorama_width, panorama_height,
    sigma_i, sigma_s, device_id
);

// 深度推定
auto [rgb_panorama, distance_panorama] = estimator.estimate_RGBD_panorama(
    images_to_match,    // std::vector<std::vector<float>> [H×W×3, 0-255]
    images_to_stitch    // std::vector<std::vector<float>>
);
```

## パフォーマンス特性

- **ターゲットハードウェア**: Jetson AGX Orin (ARM64, CUDA 8.7)
- **レジスタ圧力**: < 64 registers/thread
- **メモリ占有率**: > 50%
- **スレッドブロック**: (16, 16)

## 技術詳細

### ダブルスフィア歪み補正 (Double Sphere Model)

fisheyeカメラの逆投影・投影をデバイスコードで実装：

```cuda
__device__ float3 unproject_double_sphere(float2 uv, const DoubleSphereCalibration& calib);
__device__ float2 project_double_sphere(float3 point, const DoubleSphereCalibration& calib);
```

### アダプティブカメラ選択

各ピクセルについて、ベースラインが最大のカメラペアを自動選択：
```cuda
__global__ select_best_cameras_kernel_impl();
```

### フュージョンカーネル

統合カーネルで距離スイープと投影を同時実行：
```cuda
__global__ estimate_fisheye_distance_fused_kernel_impl();
```

## 実装のポイント

1. **Constant Memory活用**: キャリブレーションデータは全スレッドから読み込み専用なので、constant memoryに配置して高速アクセス

2. **Texture Memory**: `grid_sample`の代わりにテクスチャメモリを使用してハードウェア加速バイリニア補間

3. **レジスタ内コスト計算**: 巨大な中間バッファ代わりにレジスタで最小コストを保持

4. **非同期処理**: 複数の参照カメラについて `cudaStream_t` で並列処理

## 今後の拡張

- [ ] ISB Filter (Guided Filter) の CUDA実装
- [ ] Stitcher の GPU化（パノラマ合成）
- [ ] 共有メモリによるプリフェッチ
- [ ] Cooperative Groups による最適化
- [ ] マルチGPU対応

## 参照

- **論文**: [CVPR 2021 Oral](http://vclab.kaist.ac.kr/cvpr2021p1/)
- **Model**: Double Sphere (Kannala & Branden, 2013)
- **License**: CC BY-NC-SA 3.0

---

詳細な最適化ガイドは [CUDA_OPTIMIZATION_GUIDE.md](CUDA_OPTIMIZATION_GUIDE.md) を参照してください。
