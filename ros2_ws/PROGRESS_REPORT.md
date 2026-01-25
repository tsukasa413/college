## 🎯 Summary of Improvements

### Phase 1: Geometric Fixes (Before)
- Valid Pixels: **15%**
- MAE: **2.74m**
- Problem: Constant memory uninitialized → all-zero calibration

### Phase 2: Cost Volume Pipeline (Current)
- Valid Pixels: **86.0%** (↑571%)
- Quantized MAE: **0.996m** (↓86%)
- Candidate Match: **13.28%** (↑166x)
- Problem: ISB Filter not applied → noisy cost volume

### Phase 3: ISB Filter (To Do)
- Expected Valid Pixels: **>90%**
- Expected MAE: **<0.5m**
- Implementation: Guided filter on cost volume planes

## Key Insight

**The cost volume must be filtered BEFORE distance selection**, not after!

Python pipeline:
```python
cost_volume = compute_costs()  # [64, 480, 640]
cost_volume = clamp(cost_volume, max=500)
cost_volume = isb_filter.apply(cost_volume)  # ← Filter HERE
distance = argmin(cost_volume, dim=0)  # Then select
```

Current C++ pipeline:
```cpp
cost_volume = compute_costs();
// ISB filter skipped (TODO)
distance = argmin(cost_volume);  // Select from noisy volume
```

This explains why:
- ✅ Candidate matching improved (correct cost volume structure)
- ❌ Raw MAE still high (no smoothing → picks wrong local minima)
