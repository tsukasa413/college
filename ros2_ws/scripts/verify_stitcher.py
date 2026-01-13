#!/usr/bin/env python3
"""
Verification script to compare Python and C++ Stitcher implementations

This script loads the same calibration and input data for both
the original Python stitcher and the new C++ stitcher, then
compares their outputs pixel-by-pixel.

NOTE: This script must be run with sphere-stereo's virtual environment activated!
      Use: ./scripts/run_verification.sh
"""
import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
import cv2
from collections import namedtuple

# Verify we're in the correct environment
print("🔍 Checking Python environment...")
print(f"   Python: {sys.executable}")
print(f"   PyTorch: {torch.__version__}")

try:
    import cupy
    print(f"   CuPy: {cupy.__version__}")
except ImportError:
    print("   ❌ CuPy not found! Please run with sphere-stereo's .venv activated")
    sys.exit(1)

# Add paths for both implementations
sys.path.insert(0, '/home/motoken/college/sphere-stereo')

# Import Python version (from sphere-stereo .venv)
try:
    from python.stitcher import Stitcher as PythonStitcher
    from python.utils import Calibration as PythonCalibration
    print("   ✅ Python Stitcher imported from sphere-stereo")
except ImportError as e:
    print(f"   ❌ Failed to import Python Stitcher: {e}")
    print("   Make sure sphere-stereo's .venv is activated")
    sys.exit(1)

# Import C++ version (from ROS 2 workspace)
try:
    sys.path.insert(0, '/home/motoken/college/ros2_ws/install/my_stereo_pkg/lib/python3.10/site-packages')
    from my_stereo_pkg import _core_cpp
    print("   ✅ C++ Stitcher imported from my_stereo_pkg")
except ImportError as e:
    print(f"   ❌ Failed to import C++ Stitcher: {e}")
    print("   Make sure ROS 2 workspace is sourced")
    sys.exit(1)

print("")

# Resource paths
RESOURCES_DIR = Path('/home/motoken/college/sphere-stereo/resources')
CALIB_FILE = RESOURCES_DIR / 'calibration.json'
OUTPUT_DIR = Path('/tmp/stitcher_verification')
OUTPUT_DIR.mkdir(exist_ok=True)

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to 3x3 rotation matrix"""
    # Normalize quaternion
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

def load_calibrations(calib_file, device):
    """
    Load calibration from JSON file and create both Python and C++ calibrations
    
    Returns:
        python_calibs: List of Python Calibration objects
        cpp_calibs: List of C++ Calibration objects
        resolution: Original image resolution
    """
    with open(calib_file, 'r') as f:
        calib_data = json.load(f)
    
    value0 = calib_data['value0']
    intrinsics_list = value0['intrinsics']
    T_imu_cam_list = value0['T_imu_cam']
    resolution = value0['resolution'][0]  # Assume all cameras have same resolution
    
    python_calibs = []
    cpp_calibs = []
    
    for i, (intrinsics, T_imu_cam) in enumerate(zip(intrinsics_list, T_imu_cam_list)):
        # Extract intrinsics
        intr = intrinsics['intrinsics']
        fx, fy = intr['fx'], intr['fy']
        cx, cy = intr['cx'], intr['cy']
        xi, alpha = intr['xi'], intr['alpha']
        
        # Extract pose (quaternion + translation)
        px, py, pz = T_imu_cam['px'], T_imu_cam['py'], T_imu_cam['pz']
        qx, qy, qz, qw = T_imu_cam['qx'], T_imu_cam['qy'], T_imu_cam['qz'], T_imu_cam['qw']
        
        # Build 4x4 transformation matrix
        R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        rt = np.eye(4, dtype=np.float32)
        rt[:3, :3] = R
        rt[:3, 3] = [px, py, pz]
        rt_tensor = torch.from_numpy(rt).to(device)
        
        # Python Calibration
        py_calib = PythonCalibration(
            original_resolution=torch.tensor(resolution, dtype=torch.int32, device=device),
            principal=torch.tensor([cx, cy], dtype=torch.float32, device=device),
            fl=torch.tensor([fx, fy], dtype=torch.float32, device=device),
            xi=xi,
            alpha=alpha,
            rt=rt_tensor,
            matching_scale=1.0  # Will be updated later
        )
        python_calibs.append(py_calib)
        
        # C++ Calibration
        cpp_calib = _core_cpp.Calibration()
        cpp_calib.fl = (fx, fy)
        cpp_calib.principal = (cx, cy)
        cpp_calib.xi = xi
        cpp_calib.alpha = alpha
        cpp_calib.matching_scale = 1.0
        cpp_calib.rt = rt_tensor
        cpp_calibs.append(cpp_calib)
    
    return python_calibs, cpp_calibs, resolution

def load_images_and_masks(num_cameras, device):
    """
    Load input images and masks
    
    Returns:
        images: List of [H, W, 3] uint8 tensors (BGR format from OpenCV)
        masks: [num_cameras, H, W] float32 tensor
    """
    images = []
    masks_list = []
    
    for i in range(num_cameras):
        # Load image
        img_path = RESOURCES_DIR / f'cam{i}' / '0.jpg'
        img = cv2.imread(str(img_path))  # BGR format
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Convert to RGB and to tensor
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).to(device)
        images.append(img_tensor)
        
        # Load mask
        mask_path = RESOURCES_DIR / f'cam{i}' / 'mask.png'
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask_tensor = torch.from_numpy(mask).float().to(device) / 255.0
        else:
            # Use all-ones mask if not available
            h, w = img.shape[:2]
            mask_tensor = torch.ones(h, w, dtype=torch.float32, device=device)
        
        masks_list.append(mask_tensor)
    
    masks = torch.stack(masks_list, dim=0)
    
    return images, masks

def create_dummy_distance_maps(images, min_dist, max_dist, device):
    """
    Create dummy distance maps for testing
    (In real scenario, these would come from stereo matching)
    """
    distance_maps = []
    for img in images:
        h, w = img.shape[:2]
        # Create gradient distance map
        dist = torch.ones(h, w, dtype=torch.float32, device=device) * (min_dist + max_dist) / 2
        # Add some variation
        y_grad = torch.arange(h, device=device).unsqueeze(1).float() / h
        dist = dist + y_grad * (max_dist - min_dist) * 0.3
        distance_maps.append(dist)
    
    return distance_maps

def compute_metrics(py_rgb, py_depth, cpp_rgb, cpp_depth):
    """
    Compute comparison metrics between Python and C++ outputs
    
    Returns:
        dict with MAE, MSE, max error, etc.
    """
    # Move to CPU and convert to numpy
    py_rgb_np = py_rgb.cpu().numpy().astype(np.float32)
    cpp_rgb_np = cpp_rgb.cpu().numpy().astype(np.float32)
    py_depth_np = py_depth.cpu().numpy()
    cpp_depth_np = cpp_depth.cpu().numpy()
    
    # RGB metrics (normalize to [0, 1])
    py_rgb_norm = py_rgb_np / 255.0
    cpp_rgb_norm = cpp_rgb_np / 255.0
    
    rgb_mae = np.mean(np.abs(py_rgb_norm - cpp_rgb_norm))
    rgb_mse = np.mean((py_rgb_norm - cpp_rgb_norm) ** 2)
    rgb_max_error = np.max(np.abs(py_rgb_norm - cpp_rgb_norm))
    
    # Depth metrics
    # Mask out invalid depths (e.g., zero or very large values)
    valid_mask = (py_depth_np > 0) & (cpp_depth_np > 0) & \
                 (py_depth_np < 1000) & (cpp_depth_np < 1000)
    
    if valid_mask.sum() > 0:
        py_depth_valid = py_depth_np[valid_mask]
        cpp_depth_valid = cpp_depth_np[valid_mask]
        
        depth_mae = np.mean(np.abs(py_depth_valid - cpp_depth_valid))
        depth_mse = np.mean((py_depth_valid - cpp_depth_valid) ** 2)
        depth_max_error = np.max(np.abs(py_depth_valid - cpp_depth_valid))
        depth_valid_ratio = valid_mask.sum() / valid_mask.size
    else:
        depth_mae = depth_mse = depth_max_error = float('inf')
        depth_valid_ratio = 0.0
    
    return {
        'rgb_mae': rgb_mae,
        'rgb_mse': rgb_mse,
        'rgb_max_error': rgb_max_error,
        'depth_mae': depth_mae,
        'depth_mse': depth_mse,
        'depth_max_error': depth_max_error,
        'depth_valid_ratio': depth_valid_ratio
    }

def save_comparison_images(py_rgb, py_depth, cpp_rgb, cpp_depth, output_dir):
    """Save output images for visual comparison"""
    # Move to CPU and convert to numpy
    py_rgb_np = py_rgb.cpu().numpy()
    cpp_rgb_np = cpp_rgb.cpu().numpy()
    py_depth_np = py_depth.cpu().numpy()
    cpp_depth_np = cpp_depth.cpu().numpy()
    
    # Save RGB images (convert RGB to BGR for OpenCV)
    cv2.imwrite(str(output_dir / 'python_rgb.png'), 
                cv2.cvtColor(py_rgb_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / 'cpp_rgb.png'), 
                cv2.cvtColor(cpp_rgb_np, cv2.COLOR_RGB2BGR))
    
    # RGB difference (amplified for visualization)
    rgb_diff = np.abs(py_rgb_np.astype(np.float32) - cpp_rgb_np.astype(np.float32))
    rgb_diff_vis = np.clip(rgb_diff * 10, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / 'rgb_diff.png'), 
                cv2.cvtColor(rgb_diff_vis, cv2.COLOR_RGB2BGR))
    
    # Save depth maps (normalized to [0, 255])
    def normalize_depth(depth):
        valid_mask = (depth > 0) & (depth < 1000)
        if valid_mask.sum() > 0:
            vmin, vmax = depth[valid_mask].min(), depth[valid_mask].max()
            depth_norm = np.zeros_like(depth)
            depth_norm[valid_mask] = (depth[valid_mask] - vmin) / (vmax - vmin) * 255
            return depth_norm.astype(np.uint8)
        return np.zeros_like(depth, dtype=np.uint8)
    
    cv2.imwrite(str(output_dir / 'python_depth.png'), normalize_depth(py_depth_np))
    cv2.imwrite(str(output_dir / 'cpp_depth.png'), normalize_depth(cpp_depth_np))
    
    # Depth difference
    valid_mask = (py_depth_np > 0) & (cpp_depth_np > 0) & \
                 (py_depth_np < 1000) & (cpp_depth_np < 1000)
    depth_diff = np.zeros_like(py_depth_np)
    if valid_mask.sum() > 0:
        depth_diff[valid_mask] = np.abs(py_depth_np[valid_mask] - cpp_depth_np[valid_mask])
        depth_diff_norm = (depth_diff / (depth_diff[valid_mask].max() + 1e-6) * 255).astype(np.uint8)
    else:
        depth_diff_norm = np.zeros_like(py_depth_np, dtype=np.uint8)
    
    cv2.imwrite(str(output_dir / 'depth_diff.png'), depth_diff_norm)

def main():
    """
    Main verification function
    """
    print("=" * 70)
    print("Stitcher Verification: Python vs C++ Implementation")
    print("=" * 70)
    
    # Configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Load calibrations
    print(f"\n🔧 Loading calibrations from {CALIB_FILE}...")
    python_calibs, cpp_calibs, original_resolution = load_calibrations(CALIB_FILE, device)
    num_cameras = len(python_calibs)
    print(f"   ✅ Loaded {num_cameras} camera calibrations")
    print(f"   Original resolution: {original_resolution[0]}x{original_resolution[1]}")
    
    # Load images and masks
    print(f"\n📸 Loading images and masks...")
    images, masks = load_images_and_masks(num_cameras, device)
    img_h, img_w = images[0].shape[:2]
    print(f"   ✅ Loaded {num_cameras} images ({img_w}x{img_h})")
    print(f"   ✅ Loaded {num_cameras} masks")
    
    # Configuration parameters
    min_dist = 0.1
    max_dist = 10.0
    matching_scale = 0.5  # Downsample for faster matching
    matching_cols = int(img_w * matching_scale)
    matching_rows = int(img_h * matching_scale)
    rgb_to_stitch_cols = img_w
    rgb_to_stitch_rows = img_h
    panorama_cols = int(img_w * 1.5)
    panorama_rows = int(img_h * 1.0)
    smoothing_radius = 15
    inpainting_iterations = 32
    
    # Update matching scale in calibrations
    for py_calib, cpp_calib in zip(python_calibs, cpp_calibs):
        py_calib.matching_scale = matching_scale
        cpp_calib.matching_scale = matching_scale
    
    # Downsample masks for matching resolution
    masks_matching = torch.nn.functional.interpolate(
        masks.unsqueeze(1), 
        size=(matching_rows, matching_cols), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(1)
    
    print(f"\n⚙️  Configuration:")
    print(f"   Matching resolution: {matching_cols}x{matching_rows}")
    print(f"   Panorama resolution: {panorama_cols}x{panorama_rows}")
    print(f"   Distance range: [{min_dist}, {max_dist}]")
    print(f"   Smoothing radius: {smoothing_radius}")
    print(f"   Inpainting iterations: {inpainting_iterations}")
    
    # Create dummy distance maps
    print(f"\n📊 Creating dummy distance maps...")
    distance_maps_full = create_dummy_distance_maps(images, min_dist, max_dist, device)
    
    # Python version expects distance maps at matching resolution
    distance_maps_matching = []
    for dist in distance_maps_full:
        dist_matching = torch.nn.functional.interpolate(
            dist.unsqueeze(0).unsqueeze(0),
            size=(matching_rows, matching_cols),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        distance_maps_matching.append(dist_matching)
    
    print(f"   ✅ Created {num_cameras} distance maps (full res: {img_w}x{img_h}, matching: {matching_cols}x{matching_rows})")
    
    # Reprojection viewpoint (origin)
    reprojection_viewpoint = torch.zeros(3, dtype=torch.float32, device=device)
    
    # Initialize Python Stitcher
    print(f"\n🐍 Initializing Python Stitcher...")
    try:
        # Change to sphere-stereo directory for Python version (needs CUDA source files)
        original_cwd = os.getcwd()
        os.chdir('/home/motoken/college/sphere-stereo')
        
        # Python version expects masks as a list of [H, W] tensors, but torch.cat
        # concatenates them along dim=0, so we need to stack them instead
        # Pass the already stacked tensor directly, not as a list
        python_stitcher = PythonStitcher(
            python_calibs,
            reprojection_viewpoint,
            [masks_matching[i].unsqueeze(0) for i in range(num_cameras)],  # Each mask needs batch dimension
            min_dist,
            max_dist,
            (matching_cols, matching_rows),
            (rgb_to_stitch_cols, rgb_to_stitch_rows),
            (panorama_cols, panorama_rows),
            device,
            smoothing_radius,
            inpainting_iterations
        )
        print(f"   ✅ Python Stitcher initialized")
        
        os.chdir(original_cwd)
    except Exception as e:
        print(f"   ❌ Failed to initialize Python Stitcher: {e}")
        import traceback
        traceback.print_exc()
        os.chdir(original_cwd)
        return
    
    # Initialize C++ Stitcher
    print(f"\n⚡ Initializing C++ Stitcher...")
    try:
        cpp_stitcher = _core_cpp.Stitcher(
            cpp_calibs,
            reprojection_viewpoint,
            masks_matching,
            min_dist,
            max_dist,
            matching_cols,
            matching_rows,
            rgb_to_stitch_cols,
            rgb_to_stitch_rows,
            panorama_cols,
            panorama_rows,
            device,
            smoothing_radius,
            inpainting_iterations
        )
        print(f"   ✅ C++ Stitcher initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize C++ Stitcher: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run Python stitching
    print(f"\n🔄 Running Python stitcher.stitch()...")
    try:
        # Python version expects distance maps at matching resolution
        py_rgb, py_depth = python_stitcher.stitch(images, distance_maps_matching)
        print(f"   ✅ Python stitching complete")
        print(f"      RGB: {py_rgb.shape}, Depth: {py_depth.shape}")
    except Exception as e:
        print(f"   ❌ Python stitching failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run C++ stitching
    print(f"\n⚡ Running C++ stitcher.stitch()...")
    try:
        # C++ version also expects distance maps at matching resolution
        cpp_rgb, cpp_depth = cpp_stitcher.stitch(images, distance_maps_matching)
        print(f"   ✅ C++ stitching complete")
        print(f"      RGB: {cpp_rgb.shape}, Depth: {cpp_depth.shape}")
    except Exception as e:
        print(f"   ❌ C++ stitching failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compute metrics
    print(f"\n📊 Computing comparison metrics...")
    metrics = compute_metrics(py_rgb, py_depth, cpp_rgb, cpp_depth)
    
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n📷 RGB Panorama Comparison:")
    print(f"   MAE (Mean Absolute Error):    {metrics['rgb_mae']:.6f} (normalized [0,1])")
    print(f"   MSE (Mean Squared Error):     {metrics['rgb_mse']:.6f}")
    print(f"   Max Error:                    {metrics['rgb_max_error']:.6f}")
    print(f"   MAE in [0-255]:               {metrics['rgb_mae'] * 255:.3f} / 255")
    
    print(f"\n🗺️  Depth Panorama Comparison:")
    print(f"   MAE (Mean Absolute Error):    {metrics['depth_mae']:.6f} meters")
    print(f"   MSE (Mean Squared Error):     {metrics['depth_mse']:.6f}")
    print(f"   Max Error:                    {metrics['depth_max_error']:.6f} meters")
    print(f"   Valid pixel ratio:            {metrics['depth_valid_ratio']:.2%}")
    
    #判定基準
    rgb_threshold = 5.0 / 255.0  # 5/255 in normalized scale
    depth_threshold = 0.01  # 1cm
    
    rgb_pass = metrics['rgb_mae'] < rgb_threshold
    depth_pass = metrics['depth_mae'] < depth_threshold
    
    print(f"\n✅ PASS/FAIL Assessment:")
    print(f"   RGB MAE < {rgb_threshold*255:.1f}/255:  {'✅ PASS' if rgb_pass else '❌ FAIL'}")
    print(f"   Depth MAE < {depth_threshold}m:     {'✅ PASS' if depth_pass else '❌ FAIL'}")
    
    if rgb_pass and depth_pass:
        print(f"\n🎉 OVERALL: ✅ PASS - Implementations match within tolerance!")
    else:
        print(f"\n⚠️  OVERALL: ❌ FAIL - Implementations differ significantly")
    
    # Save comparison images
    print(f"\n💾 Saving comparison images to {OUTPUT_DIR}...")
    save_comparison_images(py_rgb, py_depth, cpp_rgb, cpp_depth, OUTPUT_DIR)
    print(f"   ✅ Images saved:")
    print(f"      - python_rgb.png, cpp_rgb.png")
    print(f"      - python_depth.png, cpp_depth.png")
    print(f"      - rgb_diff.png, depth_diff.png")
    
    print(f"\n" + "=" * 70)
    print("Verification Complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()
