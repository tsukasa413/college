#!/usr/bin/env python3
"""
Full Pipeline Verification Script
Compare Python RGBD_Estimator vs C++ RGBDEstimator

Follows the exact logic from sphere-stereo/python/main.py:
- Uses parse_json_calib for calibration loading
- Uses read_input_images for image preprocessing
- Transfers calibration parameters correctly to C++ version
- Compares results and saves output images
"""

import torch
import json
import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add sphere-stereo Python implementation to path
sys.path.insert(0, '/home/motoken/college/sphere-stereo/python')
from depth_estimation import RGBD_Estimator as PyRGBDEstimator
from utils import parse_json_calib, read_input_images

# Add ROS 2 workspace C++ bindings to path
sys.path.insert(0, '/home/motoken/college/ros2_ws/install/my_stereo_pkg/lib/python3.10/site-packages')
from my_stereo_pkg import Calibration as CppCalibration
from my_stereo_pkg import RGBDEstimator as CppRGBDEstimator


def create_cpp_calibrations(py_calibrations, device):
    """
    Convert Python Calibration objects to C++ Calibration objects
    Ensures matching_scale is transferred correctly
    """
    cpp_calibrations = []
    
    for py_calib in py_calibrations:
        cpp_calib = CppCalibration()
        
        # Transfer intrinsics
        cpp_calib.fl = (py_calib.fl[0].item(), py_calib.fl[1].item())
        cpp_calib.principal = (py_calib.principal[0].item(), py_calib.principal[1].item())
        cpp_calib.xi = py_calib.xi
        cpp_calib.alpha = py_calib.alpha
        
        # Transfer extrinsics (RT matrix)
        cpp_calib.rt = py_calib.rt.clone()
        
        # Transfer matching_scale (CRITICAL for correct projection) - both x and y scales
        cpp_calib.matching_scale = (py_calib.matching_scale[0].item(), py_calib.matching_scale[1].item())
        
        cpp_calibrations.append(cpp_calib)
    
    return cpp_calibrations


def save_comparison_images(py_rgb, cpp_rgb, py_distance, cpp_distance, output_dir, 
                          min_dist, max_dist):
    """
    Save RGB and distance comparison images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert tensors to numpy (CPU)
    py_rgb_np = py_rgb.cpu().numpy().astype(np.uint8)
    cpp_rgb_np = cpp_rgb.cpu().numpy().astype(np.uint8)
    py_distance_np = py_distance.cpu().numpy()
    cpp_distance_np = cpp_distance.cpu().numpy()
    
    # Save RGB images (convert RGB to BGR for OpenCV)
    cv2.imwrite(str(output_dir / "python_rgb.png"), cv2.cvtColor(py_rgb_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "cpp_rgb.png"), cv2.cvtColor(cpp_rgb_np, cv2.COLOR_RGB2BGR))
    
    # Save RGB difference (amplified for visibility)
    rgb_diff = np.abs(py_rgb_np.astype(np.float32) - cpp_rgb_np.astype(np.float32))
    rgb_diff_amplified = np.clip(rgb_diff * 10, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "rgb_diff.png"), cv2.cvtColor(rgb_diff_amplified, cv2.COLOR_RGB2BGR))
    
    # Save distance maps (colorized)
    def colorize_distance(distance, min_d, max_d):
        """Convert distance to colormap (inverse distance for better visualization)"""
        inv_distance = 1.0 / distance
        normalized = (inv_distance - 1.0/max_d) / (1.0/min_d - 1.0/max_d)
        normalized = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA)
    
    py_dist_color = colorize_distance(py_distance_np, min_dist, max_dist)
    cpp_dist_color = colorize_distance(cpp_distance_np, min_dist, max_dist)
    
    cv2.imwrite(str(output_dir / "python_distance.png"), py_dist_color)
    cv2.imwrite(str(output_dir / "cpp_distance.png"), cpp_dist_color)
    
    # Save distance difference (absolute error in meters, colorized)
    dist_diff = np.abs(py_distance_np - cpp_distance_np)
    dist_diff_normalized = np.clip(dist_diff / 0.5 * 255, 0, 255).astype(np.uint8)  # 0.5m -> white
    dist_diff_color = cv2.applyColorMap(dist_diff_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / "distance_diff.png"), dist_diff_color)
    
    print(f"\n  Saved comparison images to: {output_dir}")


def verify_full_pipeline():
    """Main verification function following main.py logic"""
    device = torch.device("cuda:0")
    print("=" * 80)
    print("Full Pipeline Verification: Python vs C++ RGBD_Estimator")
    print("Using sphere-stereo main.py logic")
    print("=" * 80)
    
    # ========== Configuration (matching main.py defaults) ==========
    dataset_path = '/home/motoken/college/sphere-stereo/resources'
    calib_path = os.path.join(dataset_path, 'calibration.json')
    output_dir = '/home/motoken/college/ros2_ws/output'
    
    # Pipeline parameters (from main.py defaults)
    min_dist = 0.6  # Changed from 0.55 to match typical usage
    max_dist = 10.0  # Changed from 100 for indoor scene
    candidate_count = 64  # Increased from 32 for better quality
    references_indices = [0,1, 2,3]  # Matching main.py default
    
    # Resolution parameters (from main.py defaults) - use lists to match main.py
    matching_resolution = [1024, 1024]  # [width, height]
    rgb_to_stitch_resolution = [1216, 1216]
    panorama_resolution = [2048, 1024]
    
    # Filter parameters (from main.py defaults)
    sigma_i = 10.0
    sigma_s = 25.0
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Calibration: {calib_path}")
    print(f"  Output: {output_dir}")
    print(f"  Distance range: [{min_dist}, {max_dist}]")
    print(f"  Candidates: {candidate_count}")
    print(f"  References: {references_indices}")
    print(f"  Matching resolution: {matching_resolution}")
    print(f"  Stitch resolution: {rgb_to_stitch_resolution}")
    print(f"  Panorama resolution: {panorama_resolution}")
    print(f"  Sigma_i: {sigma_i}, Sigma_s: {sigma_s}")
    
    # ========== Load Calibration using parse_json_calib ==========
    print(f"\n[1/5] Loading calibration with parse_json_calib...")
    with open(calib_path, 'r') as f:
        calib_data = json.load(f)
    
    raw_calibration = calib_data['value0']
    
    # Use parse_json_calib from utils.py (本家の関数)
    # Explicitly specify original resolution based on actual images
    original_resolution = [1944, 1096]  # [width, height] for our test images
    py_calibrations = parse_json_calib(
        raw_calibration, 
        matching_resolution, 
        device, 
        original_resolution
    )
    
    num_cameras = len(py_calibrations)
    print(f"  Loaded {num_cameras} cameras")
    print(f"  Original resolution: {original_resolution}")
    print(f"  Matching scale example (cam0): {py_calibrations[0].matching_scale.tolist()}")
    
    # ========== Load Images and Masks using read_input_images ==========
    print(f"\n[2/5] Loading images with read_input_images...")
    filename = "0.jpg"  # First frame
    
    # Use read_input_images from utils.py (本家の関数)
    image_data = read_input_images(
        filename, 
        dataset_path, 
        matching_resolution, 
        rgb_to_stitch_resolution, 
        py_calibrations, 
        references_indices
    )
    
    if not image_data["is_valid"]:
        raise RuntimeError("Failed to load images")
    
    images_to_match_np = image_data["images_to_match"]
    images_to_stitch_np = image_data["images_to_stitch"]
    
    print(f"  Loaded {len(images_to_match_np)} images for matching")
    print(f"  Loaded {len(images_to_stitch_np)} images for stitching")
    
    # Convert numpy arrays to torch tensors
    images_to_match = [torch.tensor(img, device=device, dtype=torch.float32) 
                       for img in images_to_match_np]
    images_to_stitch = [torch.tensor(img, device=device, dtype=torch.float32) 
                        for img in images_to_stitch_np]
    
    # ========== Load Masks ==========
    print(f"\n[3/5] Loading masks...")
    masks = []
    for cam_index in range(num_cameras):
        mask_path = os.path.join(dataset_path, f"cam{cam_index}", "mask.png")
        
        if os.path.isfile(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            # cv2.resize output shape: (height, width) when given (width, height)
            mask = cv2.resize(mask, tuple(matching_resolution), cv2.INTER_AREA)
            # mask shape is now (matching_resolution[1], matching_resolution[0]) = (height, width)
            masks.append(torch.tensor(mask, device=device, dtype=torch.float32).unsqueeze(0) / 255)
        else:
            # matching_resolution is [width, height], create mask as [height, width]
            # cv2.resize returns (height, width), so we must match that
            mask_shape = (matching_resolution[1], matching_resolution[0])  # (height, width)
            masks.append(torch.ones([1, mask_shape[0], mask_shape[1]], 
                                   device=device, dtype=torch.float32))
    
    print(f"  Loaded {len(masks)} masks")
    print(f"  Mask shape: {masks[0].shape}")
    
    # ========== Calculate reprojection viewpoint (center of references) ==========
    reprojection_viewpoint = torch.zeros([3], device=device, dtype=torch.float32)
    for ref_idx in references_indices:
        reprojection_viewpoint += py_calibrations[ref_idx].rt[:3, 3] / len(references_indices)
    
    print(f"  Reprojection viewpoint: {reprojection_viewpoint.tolist()}")
    
    # ========== Create C++ Calibrations from Python Calibrations ==========
    print(f"\n[4/5] Creating C++ calibrations from Python calibrations...")
    cpp_calibrations = create_cpp_calibrations(py_calibrations, device)
    print(f"  Created {len(cpp_calibrations)} C++ calibrations")
    print(f"  Matching scale transferred (cam0): ({cpp_calibrations[0].matching_scale[0]}, {cpp_calibrations[0].matching_scale[1]})")
    
    # ========== Initialize Estimators ==========
    print(f"\n[5/5] Initializing and running pipelines...")
    
    # Python RGBD_Estimator
    print("  Initializing Python RGBD_Estimator...")
    py_estimator = PyRGBDEstimator(
        calibrations=py_calibrations,
        min_dist=min_dist,
        max_dist=max_dist,
        candidate_count=candidate_count,
        references_indices=references_indices,
        reprojection_viewpoint=reprojection_viewpoint,
        masks=masks,
        matching_resolution=matching_resolution,
        rgb_to_stitch_resolution=rgb_to_stitch_resolution,
        panorama_resolution=panorama_resolution,
        sigma_i=sigma_i,
        sigma_s=sigma_s,
        device=device
    )
    print("  ✓ Python estimator initialized")
    
    # C++ RGBDEstimator
    print("  Initializing C++ RGBDEstimator...")
    cpp_estimator = CppRGBDEstimator(
        calibrations=cpp_calibrations,
        min_dist=min_dist,
        max_dist=max_dist,
        candidate_count=candidate_count,
        references_indices=references_indices,
        reprojection_viewpoint=reprojection_viewpoint,
        masks=masks,
        matching_resolution=matching_resolution,
        rgb_to_stitch_resolution=rgb_to_stitch_resolution,
        panorama_resolution=panorama_resolution,
        sigma_i=sigma_i,
        sigma_s=sigma_s,
        device=device
    )
    print("  ✓ C++ estimator initialized")
    
    # ========== Run Pipelines with Timing ==========
    print("\n  Running pipelines...")
    
    # Warmup C++
    print("  Warming up C++...")
    for _ in range(2):
        cpp_estimator.estimate_RGBD_panorama(images_to_match, images_to_stitch)
    torch.cuda.synchronize()
    
    # Python version
    print("\n  Running Python RGBD_Estimator...")
    torch.cuda.synchronize()
    py_start = torch.cuda.Event(enable_timing=True)
    py_end = torch.cuda.Event(enable_timing=True)
    
    py_start.record()
    py_rgb, py_distance = py_estimator.estimate_RGBD_panorama(images_to_match, images_to_stitch)
    py_end.record()
    torch.cuda.synchronize()
    py_time = py_start.elapsed_time(py_end)
    
    print(f"    Time: {py_time:.2f} ms")
    print(f"    RGB shape: {py_rgb.shape}, Distance shape: {py_distance.shape}")
    
    # DIAGNOSTIC: Probe Python intermediate values at (512, 512)
    probe_x, probe_y = 512, 512
    print(f"\n  === PYTHON DIAGNOSTIC PROBE at ({probe_x}, {probe_y}) ===")
    
    # Test unproject/project for first camera
    test_uv = torch.tensor([[probe_x, probe_y]], dtype=torch.float32, device=device)
    from utils import unproject, project
    pt_unit, valid = unproject(test_uv, py_calibrations[0])
    print(f"  [UNPROJECT] Point: {pt_unit[0].cpu().numpy()}, Valid: {valid[0].item()}")
    
    # Reproject back
    uv_back, valid_proj = project(pt_unit, py_calibrations[0])
    print(f"  [PROJECT] UV back: {uv_back[0].cpu().numpy()}, Valid: {valid_proj[0].item()}")
    
    # Note: Cost volume intermediate values would require modifying the Python code
    # to expose them, which we'll skip for now
    print(f"  [FINAL DISTANCE] Python: {py_distance[probe_y, probe_x].item():.4f} m")
    print("  === END PYTHON PROBE ===\n")
    
    # C++ version
    print("\n  Running C++ RGBDEstimator...")
    torch.cuda.synchronize()
    cpp_start = torch.cuda.Event(enable_timing=True)
    cpp_end = torch.cuda.Event(enable_timing=True)
    
    cpp_start.record()
    cpp_rgb, cpp_distance = cpp_estimator.estimate_RGBD_panorama(images_to_match, images_to_stitch)
    cpp_end.record()
    torch.cuda.synchronize()
    cpp_time = cpp_start.elapsed_time(cpp_end)
    
    print(f"    Time: {cpp_time:.2f} ms")
    print(f"    RGB shape: {cpp_rgb.shape}, Distance shape: {cpp_distance.shape}")
    
    # DIAGNOSTIC: C++ probe output is printed during execution
    print(f"\n  === C++ DIAGNOSTIC (see output above) ===")
    print(f"  [FINAL DISTANCE] C++: {cpp_distance[probe_y, probe_x].item():.4f} m")
    print("  === END C++ PROBE ===\n")
    
    # ========== Compare Results ==========
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    # RGB comparison
    py_rgb_float = py_rgb.float()
    cpp_rgb_float = cpp_rgb.float()
    rgb_mae = (py_rgb_float - cpp_rgb_float).abs().mean().item()
    rgb_max_err = (py_rgb_float - cpp_rgb_float).abs().max().item()
    
    print(f"\nRGB Panorama:")
    print(f"  MAE: {rgb_mae:.6f} / 255 ({(rgb_mae/255.0)*100:.4f}%)")
    print(f"  Max Error: {rgb_max_err:.6f} / 255")
    
    # Distance comparison
    # Filter out invalid distances (> max_dist) for better comparison
    py_valid_mask = (py_distance >= min_dist) & (py_distance <= max_dist)
    cpp_valid_mask = (cpp_distance >= min_dist) & (cpp_distance <= max_dist)
    both_valid_mask = py_valid_mask & cpp_valid_mask
    
    if both_valid_mask.sum() > 0:
        distance_mae = (py_distance[both_valid_mask] - cpp_distance[both_valid_mask]).abs().mean().item()
        distance_max_err = (py_distance[both_valid_mask] - cpp_distance[both_valid_mask]).abs().max().item()
        distance_mean = py_distance[both_valid_mask].mean().item()
        
        print(f"\nDistance Panorama (valid pixels only):")
        print(f"  Valid pixels: {both_valid_mask.sum().item()} / {py_distance.numel()} ({100*both_valid_mask.sum().item()/py_distance.numel():.1f}%)")
        print(f"  Python valid: {py_valid_mask.sum().item()}, C++ valid: {cpp_valid_mask.sum().item()}")
        print(f"  MAE: {distance_mae:.6f} m ({(distance_mae/distance_mean)*100:.4f}%)")
        print(f"  Max Error: {distance_max_err:.6f} m")
        print(f"  Mean distance: {distance_mean:.3f} m")
        print(f"  Python range: [{py_distance[py_valid_mask].min().item():.3f}, {py_distance[py_valid_mask].max().item():.3f}]")
        print(f"  C++ range: [{cpp_distance[cpp_valid_mask].min().item():.3f}, {cpp_distance[cpp_valid_mask].max().item():.3f}]")
    else:
        print(f"\nDistance Panorama:")
        print(f"  ERROR: No valid pixels found!")
        print(f"  Python valid pixels: {py_valid_mask.sum().item()}")
        print(f"  C++ valid pixels: {cpp_valid_mask.sum().item()}")
        distance_mae = float('inf')
        distance_mean = 0
    
    # Performance
    speedup = py_time / cpp_time
    print(f"\nPerformance:")
    print(f"  Python: {py_time:.2f} ms")
    print(f"  C++:    {cpp_time:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    # ========== Save Comparison Images ==========
    print(f"\nSaving comparison images...")
    save_comparison_images(py_rgb, cpp_rgb, py_distance, cpp_distance, 
                          output_dir, min_dist, max_dist)
    
    # ========== Summary ==========
    print("\n" + "=" * 80)
    if rgb_mae < 1.0 and distance_mae < 0.2:
        print("✓ VERIFICATION PASSED: Results match within acceptable tolerance")
    else:
        print("✗ VERIFICATION FAILED: Results differ significantly")
    print("=" * 80)
    
    return {
        'rgb_mae': rgb_mae,
        'distance_mae': distance_mae,
        'py_time': py_time,
        'cpp_time': cpp_time,
        'speedup': speedup
    }


if __name__ == "__main__":
    # Change to sphere-stereo directory for Python version to find CUDA files
    original_dir = os.getcwd()
    os.chdir('/home/motoken/college/sphere-stereo')
    
    try:
        results = verify_full_pipeline()
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        os.chdir(original_dir)
