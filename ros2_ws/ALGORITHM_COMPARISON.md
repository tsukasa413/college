# Depth Estimation アルゴリズム比較レポート

## 1. 実装対応表

| 処理ステップ | Python (depth_estimation.py) | C++/CUDA (depth_estimation.cu) | 実装状態 |
|-------------|------------------------------|--------------------------------|---------|
| **Double Sphere Unprojection** | `utils.unproject()` | `unproject_double_sphere()` (L33-66) | ✅ 実装済 |
| **Double Sphere Projection** | `utils.project()` | `project_double_sphere()` (L69-106) | ✅ 実装済 |
| **3D変換** | `torch.matmul(pt, rt.T)` | `transform_point()` (L114-124) | ✅ 実装済 |
| **Adaptive Camera Selection** | `select_camera()` (L81-133) | `select_best_cameras_kernel_impl()` (L139-199) | ✅ 実装済 |
| **Sphere Sweeping Volume** | `estimate_fisheye_distance()` (L135-206) | `estimate_fisheye_distance_fused_kernel_impl()` (L234-349) | ⚠️ 部分実装 |
| **Grid Sampling** | `torch.nn.functional.grid_sample()` | Texture Object (L309-331省略) | ⚠️ 未実装 |
| **Cost Computation (SAD)** | `torch.sum(torch.abs(...), dim=1)` (L176) | L320-330あたり（省略部分） | ⚠️ 未実装 |
| **Cost Filtering (ISB)** | `cost_filter.apply()` (L180) | なし | ❌ 未実装 |
| **Distance Selection** | `torch.min(cost_volume, dim=0)` (L183) | L333あたり（省略部分） | ⚠️ 未実装 |
| **Quadratic Fitting** | L186-195 | `refine_distance_quadratic_kernel()` (L401-407) | ❌ Placeholder |
| **Distance Filtering** | `distance_filter.apply()` (L203) | なし | ❌ 未実装 |
| **Stitching** | `fishey_stitcher.stitch()` (L231) | なし | ❌ 未実装 |

## 2. 数学的整合性の詳細解析

### 2.1 Double Sphere Model

#### Unprojection (2D → 3D)

**Python版** (`utils.py:unproject`):
```python
# Normalized image coordinates
mx = (uv[..., 0] - calibration.fl[0]) / calibration.fl[0]
my = (uv[..., 1] - calibration.fl[1]) / calibration.fl[1]

# Inverse projection
r2 = mx**2 + my**2
mz = (1 - alpha**2 * r2) / (alpha * sqrt(1 - (2*alpha - 1) * r2) + 1 - alpha)
```

**CUDA版** (`depth_estimation.cu:L44-66`):
```cuda
// Normalized image coordinates
float x = (uv.x - calib.cx) / calib.fx;
float y = (uv.y - calib.cy) / calib.fy;

// Fisheye coordinate
float rho_sq = x * x + y * y;
float rho = sqrtf(rho_sq);

// Double sphere inverse
float numer = 1.0f - xi * xi * rho_sq;
if (numer < 0.0f) return invalid;

float r = xi + sqrtf(numer) / (1.0f + alpha * rho_sq);
```

**⚠️ 数学的差異**:
1. **Principal Point**: Python版は`calibration.fl`のみ使用、CUDA版は`cx, cy`を使用
2. **変数名の対応**: Python `mx, my` ↔ CUDA `x, y`
3. **式の形式**: Python版とCUDA版で式の展開形が異なる可能性（要検証）

#### Projection (3D → 2D)

**Python版**:
```python
d1 = torch.norm(point, dim=-1, keepdim=True)
d2 = torch.norm(point[..., :2], dim=-1, keepdim=True)
w1 = alpha / (1 - alpha)
w2 = (w1 + xi) / sqrt(2 * w1 * xi + xi**2 + 1)
```

**CUDA版** (`depth_estimation.cu:L82-106`):
```cuda
float norm = sqrtf(point.x*point.x + point.y*point.y + point.z*point.z);
point = normalize(point);

// Double sphere projection
float z1 = (xi * norm + sqrtf(1.0f - xi*xi + xi*xi*norm*norm));
float x1 = point.x / z1;
float y1 = point.y / z1;
float rho_sq = x1*x1 + y1*y1;
float r = alpha + sqrtf(1.0f - alpha*alpha * (1.0f + rho_sq));
```

**⚠️ 数学的差異**:
- 式の展開形が異なる（等価性要検証）
- 正規化のタイミングが異なる

### 2.2 Adaptive Camera Selection

**Python版** (`select_camera()` L81-133):
```python
for cam_index, (calibration, mask) in enumerate(zip(self.calibrations, masks)):
    pt_near = pt_unit * self.min_dist
    pt_far = pt_unit * self.max_dist
    
    # Transform to matched camera
    rt = torch.matmul(torch.inverse(calibration.rt), reference_calibration.rt)
    pt_near = torch.matmul(..., rt.T)
    
    # Project
    uv_near, valid_near = project(pt_near, calibration)
    uv_far, valid_far = project(pt_far, calibration)
    
    # Displacement
    displacement = torch.norm(uv_near - uv_far, dim=-1)
    
    # Update best
    current_best = ((displacement > max_displacement) * valid_near * valid_far * masks...)
    selected_camera[current_best] = cam_index
    max_displacement[current_best] = displacement[current_best]
```

**CUDA版** (`select_best_cameras_kernel_impl` L139-199):
```cuda
for (int cam_idx = 0; cam_idx < num_cameras; cam_idx++) {
    if (cam_idx == 0) continue;  // ⚠️ Skip reference
    
    // Transform points
    float3 pt_near_cam = transform_point(pt_near, cam_calib.rt);
    float3 pt_far_cam = transform_point(pt_far, cam_calib.rt);
    
    // Project
    float2 uv_near = project_double_sphere(pt_near_cam, cam_calib);
    float2 uv_far = project_double_sphere(pt_far_cam, cam_calib);
    
    // Displacement
    float disp = sqrtf((uv_near.x - uv_far.x)^2 + (uv_near.y - uv_far.y)^2);
    
    if (disp > max_disp) {
        max_disp = disp;
        best_camera = cam_idx;
    }
}
```

**❌ 重大な差異**:
1. **RT行列の方向**: Python版は`torch.inverse(calibration.rt)`を使用、CUDA版は直接`cam_calib.rt`を使用
2. **Reference Cameraの扱い**: CUDA版は`cam_idx == 0`をスキップ（Python版は全カメラを考慮）
3. **マスク処理**: Python版はマスクを使用、CUDA版はマスク未使用

### 2.3 Sphere Sweeping & Cost Computation

**Python版** (`estimate_fisheye_distance()` L147-176):
```python
# Inverse distance parameterization
distance_candidates = 1 / torch.linspace(1/min_dist, 1/max_dist, candidate_count)

# Create sweeping volume [B, 3, C, H, W]
point_volume = distance_candidates.view(C,1,1,1) * pt_unit.view(1,H,W,3)

# Per-camera sampling
for cam_index, calibration in enumerate(self.calibrations):
    rt = torch.matmul(torch.inverse(calibration.rt), reference_calibration.rt)
    point_volume_in_cam = torch.matmul(..., rt.T)
    uv, _ = project(point_volume_in_cam[..., :3], calibration)
    
    # Normalize to [-1, 1]
    uv = ((uv + 0.5) / resolution) * 2 - 1
    
    # Sample with bilinear interpolation
    sweeping_volume_for_cam = torch.nn.functional.grid_sample(
        image, uv, align_corners=False
    )
    
    # Adaptive selection
    selected_mask = (selected_camera == cam_index)
    sweeping_volume[selected_mask] = sweeping_volume_for_cam[selected_mask]

# Cost computation (SAD)
cost_volume = torch.sum(torch.abs(sweeping_volume - reference_image), dim=1)
```

**CUDA版** (`estimate_fisheye_distance_fused_kernel_impl` L234-349):
```cuda
for (int dist_idx = 0; dist_idx < candidate_count; dist_idx++) {
    // Inverse distance
    float inv_dist_min = 1.0f / min_dist;
    float inv_dist_max = 1.0f / max_dist;
    float inv_dist = inv_dist_min - (inv_dist_min - inv_dist_max) * 
                     (dist_idx / (candidate_count - 1));
    float distance = 1.0f / inv_dist;
    
    // 3D point
    float3 pt_3d = pt_unit * distance;
    
    // Transform to selected camera
    float3 pt_cam = transform_point(pt_3d, selected_calib.rt);
    
    // Project
    float2 uv_proj = project_double_sphere(pt_cam, selected_calib);
    
    // Bounds check
    if (uv_proj.x < 0 || uv_proj.x >= width || ...) continue;
    
    // Bilinear sampling via Texture Object (省略部分 L309-331)
    // uchar4 sampled_color = tex2D<uchar4>(d_texobjs[selected_cam], uv_proj.x, uv_proj.y);
    
    // SAD computation (省略部分)
    // float cost = abs(sampled_color.x - ref_r) + abs(sampled_color.y - ref_g) + ...;
    
    // Update minimum
    // if (cost < min_cost) { min_cost = cost; min_cost_index = dist_idx; }
}
```

**⚠️ 実装差異**:
1. **Sweeping Volume構築**: Python版は全候補×全カメラのテンソル生成、CUDA版はピクセル単位でループ
2. **Grid Sample正規化**: Python版は`[-1, 1]`範囲、CUDA版はピクセル座標のまま
3. **Texture Object**: CUDA版の補間実装が省略されている（モック状態）
4. **Cost計算**: 省略部分のため未検証

### 2.4 Subpixel Refinement (Quadratic Fitting)

**Python版** (`estimate_fisheye_distance()` L186-195):
```python
left_cost = torch.gather(cost_volume, 0, clamp(selected_index - 1, 0, C-1))
right_cost = torch.gather(cost_volume, 0, clamp(selected_index + 1, 0, C-1))

variation = 0.5 * (left_cost - right_cost) / ((left_cost + right_cost) - 2*min_cost + 1e-8)
variation = torch.clamp(variation, min=-0.5, max=0.5)

# Boundary handling
variation[selected_index == C-1] = 0
variation[selected_index == 0] = 0

selected_index_map = selected_index.float() + variation
```

**CUDA版** (`refine_distance_quadratic_kernel` L401-407):
```cuda
// Placeholder for subpixel refinement
// In full implementation, perform quadratic fitting...
cudaStreamSynchronize(stream);
```

**❌ 差異**: CUDA版は未実装（Placeholder）

## 3. 重大な未実装箇所

### 3.1 Cost Volume Filtering (ISB Filter)

**Python版**: `ISB_Filter.apply()` を使用してコストボリュームを平滑化

**CUDA版**: **完全に未実装**

これは出力に**最大の影響**を与える処理です。フィルタなしでは：
- ノイズが多い
- エッジが保存されない
- 深度マップの品質が大幅に低下

### 3.2 Distance Map Post-Filtering

**Python版**: `distance_filter.apply()` で最終的な深度マップを平滑化

**CUDA版**: **未実装**

### 3.3 Stitching

**Python版**: `Stitcher.stitch()` で複数の魚眼画像をパノラマに合成

**CUDA版**: **未実装**（`estimate_RGBD_panorama`がモック実装）

## 4. 数値的等価性の期待値

現在の実装状態を考慮すると：

| 処理 | 一致度予測 | 理由 |
|-----|----------|------|
| Unprojection/Projection | **95-99%** | 式の形式が異なるが数学的には等価の可能性 |
| Camera Selection | **70-80%** | RT行列の方向が逆、マスク処理なし |
| Sweeping Volume (Raw) | **0%** | モック実装のため未実行 |
| Cost Computation | **0%** | 未実装 |
| Filtered Cost | **0%** | ISB Filter未実装 |
| Final Distance Map | **0%** | パイプライン全体が未実装 |

## 5. 検証すべき数値精度

### 5.1 幾何変換の精度

- **Float32 vs Float64**: CUDA版はfloat32、PyTorch版はデフォルトfloat32（要確認）
- **三角関数**: `sqrtf()` vs `torch.sqrt()` の精度差（約1e-7程度）

### 5.2 Bilinear Interpolation

- **PyTorch `grid_sample`**: 高精度な補間
- **CUDA Texture Object**: ハードウェア補間（微小な誤差の可能性）

### 5.3 累積誤差

- **Sphere Sweeping**: 50候補×76800ピクセルの累積で誤差が増幅
- **許容範囲**: MSE < 1e-4 は**幾何変換のみ**で達成可能、フルパイプラインでは困難

## 6. テスト戦略

### Phase 1: 幾何変換のみ
- `unproject_double_sphere()` と `project_double_sphere()` の単体テスト
- 目標: Max Error < 1e-5

### Phase 2: Camera Selection
- `select_best_cameras_kernel()` の出力を比較
- 目標: 選択されたカメラIDの一致率 > 95%

### Phase 3: フルパイプライン（ISB Filter実装後）
- 完全な `estimate_fisheye_distance()` の比較
- 目標: Distance Map MSE < 0.01m

## 7. 結論

**現状**: C++/CUDA実装は**基礎的な幾何変換のみ完成**しており、コア部分（Sweeping, Cost, Filter, Stitch）が未実装またはモック状態です。

**数値的等価性**: 現時点では検証不可能。以下を実装後に再検証が必要：
1. ✅ Texture Object Sampling
2. ✅ Cost Volume Computation
3. ✅ ISB Filter (最重要)
4. ✅ Distance Map Post-Filter
5. ✅ Stitcher

**推奨アクション**:
1. 幾何変換（Unproject/Project）の単体テストを先行実施
2. Camera Selectionの整合性確認（RT行列の方向を修正）
3. Sweeping/Cost計算を完全実装
4. ISB Filterの移植（Python→CUDA、最も困難）
