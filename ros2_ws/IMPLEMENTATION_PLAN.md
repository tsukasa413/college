# Cost Volume Filtering Implementation Plan

## Problem
Current C++ implementation:
1. Computes costs in kernel
2. Selects minimum distance **immediately**
3. (Optional) Filters distance map

Python implementation:
1. Computes **cost volume** [candidate_count, height, width]
2. **Clamps** cost volume (max=500)
3. **Filters** cost volume with ISB Filter
4. Selects minimum distance

## Memory Requirements
- Cost Volume: 64 candidates × 640 × 480 × 4 bytes = **78.6 MB**
- Current peak: ~200 MB
- Total with cost volume: ~280 MB (acceptable on 32GB Jetson)

## Implementation Steps

### 1. Add cost volume generation kernel
```cuda
__global__ void compute_cost_volume_kernel(
    float* d_cost_volume,           // [candidate_count, height, width]
    const uchar4* d_reference_image,
    const uchar4* const* d_images,
    const int* d_selected_cameras,
    const DoubleSphereCalibration* d_calibrations,
    const DoubleSphereCalibration& reference_calib,
    const CameraConfig& config,
    const cudaTextureObject_t* d_texobjs
);
```

### 2. Clamp cost volume
```cuda
__global__ void clamp_cost_volume_kernel(
    float* d_cost_volume,
    int total_elements,
    float max_value = 500.0f
);
```

### 3. Apply ISB Filter to each depth plane
```cpp
for (int d = 0; d < config.candidate_count; d++) {
    float* cost_plane = d_cost_volume + d * (width * height);
    isb_filter_->apply(d_guide, cost_plane, width, height);
}
```

### 4. Select minimum from filtered cost volume
```cuda
__global__ void select_distance_from_cost_volume_kernel(
    float* d_distance_map,
    const float* d_cost_volume,
    const CameraConfig& config
);
```

## Expected Impact
- **Memory**: +78.6 MB (13% increase)
- **Speed**: ~20% slower (2 kernel passes instead of fused)
- **Accuracy**: MAE should drop from 2.74m to <0.5m

## Alternative: Separable ISB Filter
If full cost volume is too slow:
1. Apply 1D horizontal filter to cost volume
2. Apply 1D vertical filter
This reduces complexity from O(r²) to O(2r) where r = filter radius.
