# Stitcher C++ Class - Python Bindings

## Overview

The `Stitcher` class provides GPU-accelerated RGB-D panorama creation from fisheye camera images using CUDA and LibTorch. It is implemented in C++ for performance and exposed to Python via pybind11.

## Features

- **GPU Acceleration**: All computations run on CUDA-enabled GPUs
- **Double Sphere Camera Model**: Accurate fisheye camera projection/unprojection
- **Z-Buffer Reprojection**: Handles occlusion and multi-camera geometry
- **Inpainting**: Fills holes in distance maps using directional weights
- **Smooth Blending**: Creates seamless panoramas with weighted blending

## Installation

The package is built as part of the ROS 2 workspace:

```bash
cd ~/college/ros2_ws
colcon build --packages-select my_stereo_pkg
source install/setup.bash
```

## Python Usage

### Basic Example

```python
import torch
from my_stereo_pkg import _core_cpp

# Create calibrations
calibrations = []
for i in range(num_cameras):
    calib = _core_cpp.Calibration()
    calib.fl = (500.0, 500.0)  # Focal length (fx, fy)
    calib.principal = (320.0, 240.0)  # Principal point
    calib.xi = 1.2  # Distortion parameter 1
    calib.alpha = 0.5  # Distortion parameter 2
    calib.matching_scale = 1.0
    calib.rt = torch.eye(4, dtype=torch.float32, device='cuda:0')
    calibrations.append(calib)

# Create reprojection viewpoint (world origin)
reprojection_viewpoint = torch.zeros(3, dtype=torch.float32, device='cuda:0')

# Create masks (all valid pixels)
masks = torch.ones(num_cameras, height, width, dtype=torch.float32, device='cuda:0')

# Initialize stitcher
stitcher = _core_cpp.Stitcher(
    calibrations,
    reprojection_viewpoint,
    masks,
    min_dist=0.1,
    max_dist=10.0,
    matching_cols=width,
    matching_rows=height,
    rgb_to_stitch_cols=width,
    rgb_to_stitch_rows=height,
    panorama_cols=pano_width,
    panorama_rows=pano_height,
    device=torch.device('cuda:0'),
    smoothing_radius=15,
    inpainting_iterations=32
)

# Stitch images
images = [...]  # List of [H, W, 3] uint8 tensors
distance_maps = [...]  # List of [H, W] float32 tensors

rgb_panorama, depth_panorama = stitcher.stitch(images, distance_maps)
```

### Complete Example

See `examples/python_stitcher_example.py` for a complete working example:

```bash
python3 src/my_stereo_pkg/examples/python_stitcher_example.py
```

## API Reference

### Calibration Class

Represents camera calibration parameters for the Double Sphere Camera Model.

**Attributes:**
- `fl` (tuple[float, float]): Focal length (fx, fy)
- `principal` (tuple[float, float]): Principal point (cx, cy)
- `xi` (float): First distortion parameter
- `alpha` (float): Second distortion parameter
- `matching_scale` (float): Scale factor for matching resolution
- `rt` (torch.Tensor): [4, 4] extrinsic matrix (rotation + translation)

### Stitcher Class

Creates RGB-D panoramas from multiple fisheye camera views.

**Constructor:**

```python
Stitcher(
    calibrations: List[Calibration],
    reprojection_viewpoint: torch.Tensor,  # [3] float32
    masks: torch.Tensor,  # [N, H, W] float32
    min_dist: float,
    max_dist: float,
    matching_cols: int,
    matching_rows: int,
    rgb_to_stitch_cols: int,
    rgb_to_stitch_rows: int,
    panorama_cols: int,
    panorama_rows: int,
    device: torch.device,
    smoothing_radius: int = 15,
    inpainting_iterations: int = 32
)
```

**Parameters:**
- `calibrations`: List of camera calibration parameters
- `reprojection_viewpoint`: Reference point for panorama creation (world coordinates)
- `masks`: Valid pixel masks for each camera [num_cameras, height, width]
- `min_dist`, `max_dist`: Expected distance range in meters
- `matching_cols`, `matching_rows`: Resolution for depth matching
- `rgb_to_stitch_cols`, `rgb_to_stitch_rows`: Resolution of RGB images to stitch
- `panorama_cols`, `panorama_rows`: Output panorama resolution
- `device`: PyTorch device (e.g., `torch.device('cuda:0')`)
- `smoothing_radius`: Radius for blending weight smoothing (default: 15)
- `inpainting_iterations`: Number of inpainting passes (default: 32)

**Methods:**

```python
def stitch(
    images: List[torch.Tensor],  # List of [H, W, 3] uint8
    distance_maps: List[torch.Tensor]  # List of [H, W] float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stitch fisheye images and distance maps into RGB-D panorama
    
    Returns:
        RGB panorama [panorama_rows, panorama_cols, 3] uint8
        Depth panorama [panorama_rows, panorama_cols] float32
    """
```

## Technical Details

### Architecture

1. **Initialization Phase:**
   - Vectorize camera calibrations
   - Create inpainting weight lookup tables (CUDA)
   - Create blending weight lookup tables (CUDA)
   - Smooth blending masks using box filter convolution

2. **Stitching Phase:**
   - Reproject distance maps to reference viewpoint (CUDA, z-buffering)
   - Inpaint holes in distance maps (CUDA, iterative)
   - Merge RGB-D data into panorama (CUDA, weighted blending)

### Performance

- **GPU Execution**: All heavy computations run on CUDA
- **Memory Efficiency**: Pre-allocated tensors minimize allocations
- **Parallel Processing**: CUDA kernels process pixels in parallel

### Camera Model

Uses the **Double Sphere Camera Model** for fisheye lenses:

```
Project: 3D point → 2D pixel
Unproject: 2D pixel + depth → 3D point
```

Parameters:
- `xi`: Controls distortion shape
- `alpha`: Controls field of view

## Requirements

- **Hardware**: NVIDIA GPU with CUDA support (tested on Jetson AGX Orin)
- **Software**:
  - CUDA 12.2+
  - PyTorch 2.4+ with GPU support
  - Python 3.10+
  - ROS 2 Humble

## Build Configuration

The package uses CMake with:
- C++17 standard
- CUDA separate compilation
- LibTorch C++ API
- pybind11 for Python bindings

Key libraries:
- `cuda_kernels_lib`: CUDA kernel implementations
- `stitcher_lib`: C++ Stitcher class
- `_core_cpp`: Python binding module

## Troubleshooting

### ImportError: undefined symbol

If you see `undefined symbol: THPDeviceType`:
- Ensure PyTorch Python library is linked correctly
- Check CMakeLists.txt links `${TORCH_PYTHON_LIBRARY}`

### CUDA out of memory

- Reduce panorama resolution
- Reduce number of cameras
- Decrease inpainting iterations

### Slow performance

- Ensure tensors are on GPU device
- Check that CUDA is available: `torch.cuda.is_available()`
- Monitor GPU usage: `nvidia-smi`

## License

See LICENSE file in the package root.

## References

- Double Sphere Camera Model: [Usenko et al., 2018]
- sphere-stereo: Original Python implementation
- LibTorch: PyTorch C++ API
- pybind11: Python/C++ bindings
