# Real-Time RGBD Estimator - C++ Implementation

## Overview

High-performance real-time depth estimation system for 4-camera fisheye setup running on NVIDIA Jetson AGX Orin. Achieves **59ms per frame (~16.9 FPS)** through aggressive CUDA optimization and GStreamer-based camera streaming.

## Performance Summary

| Configuration | Processing Time | Speedup |
|--------------|----------------|---------|
| **Initial (Python)** | 10,200 ms | - |
| **Initial C++** (1024x1024, 64 candidates) | 282 ms | 36x |
| **After Resolution Reduction** (512x512) | 71.1 ms | 143x |
| **After Candidate Reduction** (32 candidates) | 70.2 ms | 145x |
| **After Stream Propagation** | 66.9 ms | 152x |
| **After RGB→YCbCr CUDA** | 62.5 ms | 163x |
| **🎯 Real-Time Streaming** | **59.2 ms** | **172x** |

**Current Performance:**
- Average: 59.2 ms (16.9 FPS)
- Best: 50.4 ms (19.8 FPS)
- Worst: 83.3 ms (initialization overhead)

## Key Features

### 1. GStreamer Camera Streaming
- Direct integration with `nvarguscamerasrc` (Jetson camera API)
- 4-camera simultaneous capture at 1944x1096 @ 30fps
- Zero-copy NVMM memory pipeline
- Sensor mode 2: Full FOV without cropping

### 2. CUDA Optimizations
- ✅ Stream-based async execution (4 parallel camera streams)
- ✅ Per-camera buffer allocation (eliminates contention)
- ✅ Custom RGB→YCbCr CUDA kernel (eliminates LibTorch overhead)
- ✅ Constant memory for distance candidates
- ✅ Optimized kernel launch parameters

### 3. Algorithm Optimizations
- Resolution: 1024→512 (3.97x speedup)
- Candidates: 64→32 (depth sampling reduction)
- ISB Filter: Tuned sigma parameters

## System Requirements

### Hardware
- **NVIDIA Jetson AGX Orin** (or compatible)
- 4x CSI cameras (IMX219 or similar)
- 8GB+ RAM
- 32GB+ storage

### Software
- Ubuntu 20.04 (L4T)
- CUDA 11.4+
- GStreamer 1.16+
- OpenCV 4.5+ with GStreamer support
- LibTorch 2.0+
- ROS 2 Foxy (optional)

## Installation

### 1. Install Dependencies

```bash
# OpenCV with GStreamer support (should be pre-installed on Jetson)
sudo apt-get install libopencv-dev libopencv-videoio-dev

# PyTorch for C++ (LibTorch)
# Download from: https://pytorch.org/get-started/locally/
# Or use system Python torch:
pip3 install torch torchvision
```

### 2. Build the Package

```bash
cd ~/college/ros2_ws
colcon build --packages-select my_stereo_pkg --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

Build time: ~2-3 minutes on Jetson AGX Orin

### 3. Verify Installation

```bash
# Check if cameras are detected
ls /dev/video*

# Test GStreamer pipeline
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! nvvidconv ! 'video/x-raw(memory:NVMM)' ! fakesink
```

## Usage

### Basic Usage (Headless Mode)

```bash
export LD_LIBRARY_PATH=/home/motoken/.local/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
source install/setup.bash

# Run with default settings
./install/my_stereo_pkg/lib/my_stereo_pkg/standalone_estimator

# Specify calibration path and output directory
./install/my_stereo_pkg/lib/my_stereo_pkg/standalone_estimator \
    /path/to/calibration \
    /path/to/output
```

**Expected Output:**
```
========================================
Real-Time C++ RGBD Estimator
========================================

Configuration:
  Calibration path: /home/motoken/college/sphere-stereo/resources
  Output dir: /home/motoken/college/ros2_ws/output/standalone
  Save every frame: No
  Show display: No (headless)
  Distance range: [0.6, 10]
  Candidates: 32
  Original resolution: [1944, 1096]
  Matching resolution: [512, 512]
  Panorama resolution: [2048, 1024]

[1/6] Initializing camera streams...
Camera 0 initialized: 1944x1096
Camera 1 initialized: 1944x1096
Camera 2 initialized: 1944x1096
Camera 3 initialized: 1944x1096

[6/6] Warming up with live camera frames...

========================================
STARTING REAL-TIME INFERENCE
Running in headless mode (Ctrl+C to quit)
========================================

Frame 30 | Avg: 59.7 ms (16.7 FPS) | Min: 50.4 ms | Max: 83.3 ms
Frame 60 | Avg: 59.2 ms (16.9 FPS) | Min: 50.4 ms | Max: 83.3 ms
...
```

### Display Mode (with GUI)

```bash
# Requires X11/Wayland display
./install/my_stereo_pkg/lib/my_stereo_pkg/standalone_estimator \
    /path/to/calibration \
    /path/to/output \
    display
```
ex
```
./install/my_stereo_pkg/lib/my_stereo_pkg/standalone_estimator /home/motoken/college/ros2_ws/src/my_stereo_pkg/resources /home/motoken/college/ros2_ws/output/standalone display
```

**Controls:**
- `q` or `ESC`: Quit
- `s`: Save snapshot

### Save Every Frame Mode

```bash
# WARNING: Very I/O intensive!
./install/my_stereo_pkg/lib/my_stereo_pkg/standalone_estimator \
    /path/to/calibration \
    /path/to/output \
    save
```

Saves:
- `frame_N_rgb.png`: RGB panorama
- `frame_N_distance.exr`: Raw distance map (32-bit float)

## Configuration

### Pipeline Parameters

Edit `src/standalone_estimator.cpp`:

```cpp
// Distance estimation range
const float min_dist = 0.6f;
const float max_dist = 10.0f;

// Depth candidates (lower = faster, less accurate)
const int candidate_count = 32;  // 16, 32, 48, 64

// Processing resolutions
const std::pair<int, int> matching_resolution = {512, 512};  // 384, 512, 768
const std::pair<int, int> rgb_to_stitch_resolution = {1216, 1216};
const std::pair<int, int> panorama_resolution = {2048, 1024};

// ISB Filter parameters
const float sigma_i = 10.0f;  // Intensity sigma
const float sigma_s = 25.0f;  // Spatial sigma
```

### Camera Pipeline

Edit `CameraStreamer::buildGStreamerPipeline()`:

```cpp
// Sensor modes (IMX219):
// 0: 3840x2160 @ 16fps (4K)
// 1: 3840x2160 @ 30fps (4K, binned)
// 2: 1944x1096 @ 32fps (1080p, full FOV) <- Default
// 3: 1296x732 @ 48fps (720p, cropped)

int sensor_mode = 2;  // Change to 0 for 4K, 3 for 720p
```

## Architecture

### System Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   Camera Streaming Layer                     │
├─────────────────────────────────────────────────────────────┤
│  GStreamer nvarguscamerasrc (4 cameras, 1944x1096 @ 30fps) │
│           ↓ NVMM → BGR conversion (nvvidconv)               │
│           ↓ OpenCV cv::VideoCapture                         │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  Preprocessing Layer                         │
├─────────────────────────────────────────────────────────────┤
│  • BGR → RGB conversion                                      │
│  • uint8 → float32 [0-255]                                   │
│  • Resize to matching resolution (512x512)                   │
│  • LibTorch tensor creation (HWC → CHW)                      │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Depth Estimation Pipeline (CUDA)                │
├─────────────────────────────────────────────────────────────┤
│  [Per-Camera Async Processing - 4 Streams]                  │
│                                                              │
│  Stage 1: RGB → YCbCr (CUDA kernel)                         │
│           ↓                                                  │
│  Stage 2: Cost Volume Computation (CUDA)                    │
│           • Camera selection                                 │
│           • Sphere sweep stereo                              │
│           • Double sphere projection                         │
│           ↓                                                  │
│  Stage 3: ISB Filter (CUDA)                                 │
│           • Edge-preserving smoothing                        │
│           ↓                                                  │
│  Stage 4: Winner-Take-All + Quadratic Fitting (CUDA)        │
│           • Subpixel depth refinement                        │
│           ↓                                                  │
│  Stage 5: Distance Map Post-Filter (CUDA)                   │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   Stitching Layer (CUDA)                     │
├─────────────────────────────────────────────────────────────┤
│  • Z-buffer reprojection to equirectangular                  │
│  • Inpainting for hole filling                               │
│  • Multi-camera blending                                     │
│  • Output: 2048x1024 RGB + Distance panorama                │
└─────────────────────────────────────────────────────────────┘
```

### Memory Management

- **Zero-copy architecture**: Direct GPU memory allocation
- **Pre-allocated buffers**: Per-camera cost volumes and distance maps
- **Async execution**: 4 CUDA streams for parallel processing
- **No CPU-GPU transfers** during inference loop

### CUDA Kernel Details

| Kernel | Grid Size | Block Size | Memory |
|--------|-----------|------------|--------|
| RGB→YCbCr | (H/16, W/16) | (16, 16) | Global |
| Cost Volume | (W/16, H/16) | (16, 16) | Constant + Texture |
| ISB Filter | Custom | Custom | Shared |
| Final Depth | (W/16, H/16) | (16, 16) | Global |

## Troubleshooting

### Camera Not Detected

```bash
# Check camera connections
ls /dev/video*

# Test individual camera
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! nvvidconv ! xvimagesink

# Check Argus daemon
sudo systemctl status nvargus-daemon
sudo systemctl restart nvargus-daemon
```

### LibTorch Not Found

```bash
# Set library path
export LD_LIBRARY_PATH=/path/to/torch/lib:$LD_LIBRARY_PATH

# Or add to ~/.bashrc
echo 'export LD_LIBRARY_PATH=/home/motoken/.local/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Out of Memory

```bash
# Reduce resolution
matching_resolution = {384, 384}  # Instead of {512, 512}

# Reduce candidates
candidate_count = 16  # Instead of 32

# Monitor GPU memory
sudo tegrastats
```

### Low FPS

```bash
# Check GPU frequency
sudo jetson_clocks --show

# Enable maximum performance
sudo jetson_clocks

# Monitor performance
sudo tegrastats
```

## Performance Profiling

### NVTX Markers

The code includes NVTX annotations for Nsight Systems profiling:

```bash
# Capture profile
nsys profile -o rgbd_profile \
    --trace=cuda,nvtx,osrt \
    --duration=10 \
    ./install/my_stereo_pkg/lib/my_stereo_pkg/standalone_estimator

# View in Nsight Systems GUI
nsys-ui rgbd_profile.qdrep
```

### Key Metrics

- **Stage 1 (RGB→YCbCr)**: ~1-2 ms
- **Stage 2 (Cost Volume)**: ~15-20 ms
- **Stage 3 (ISB Filter)**: ~20-25 ms
- **Stage 4 (Final Depth)**: ~3-5 ms
- **Stage 5 (Post-Filter)**: ~10-15 ms
- **Stitching**: ~5-10 ms

## Future Optimizations

Target: **33ms (30 FPS)**

Remaining speedup needed: **1.79x**

### Proposed Improvements

1. **ISB Filter Optimization** (~10ms gain)
   - Reduce iteration count
   - Simplify bilateral weights
   - Use separable approximation

2. **Resolution Reduction** (~5ms gain)
   - 512→384 matching resolution
   - Adaptive resolution based on scene complexity

3. **Stitcher Batch Processing** (~5ms gain)
   - Process 4 cameras simultaneously in single kernel
   - Eliminate loop overhead

4. **FP16 Precision** (~5ms gain)
   - Unified FP16 throughout ISB filter
   - Avoid FP16↔FP32 conversions

5. **Candidate Reduction** (~5ms gain)
   - 32→24 candidates
   - Adaptive candidate selection

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{rgbd_realtime_2026,
  title={Real-Time RGBD Estimation for Multi-Camera Fisheye Systems},
  author={Your Name},
  year={2026},
  howpublished={GitHub},
  note={Optimized C++/CUDA implementation achieving 16.9 FPS on Jetson AGX Orin}
}
```

## License

[Your License Here]

## Related Documentation

- [Python Bindings](docs/PYTHON_BINDINGS.md) - Python interface for C++ components
- [Calibration Guide](docs/CALIBRATION.md) - Camera calibration procedures
- [Basalt Integration](README_basalt.md) - Visual-inertial odometry

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Last Updated:** February 5, 2026  
**Version:** 1.0.0  
**Status:** Production-ready for Jetson AGX Orin





# 1. ICP Odometry停止
pkill -f icp_odometry

# 2. ICP Odometry再起動（新しい設定で）
cd ~/college/ros2_ws && source install/setup.bash && \
ros2 run rtabmap_odom icp_odometry --ros-args \
  --params-file src/my_stereo_pkg/config/icp_odometry_config.yaml \
  -r scan_cloud:=/omnidirectional/point_cloud &

# 3. RViz起動（新しい設定で）
cd ~/college/ros2_ws && source install/setup.bash && \
rviz2 -d src/my_stereo_pkg/rviz/lightweight_slam.rviz