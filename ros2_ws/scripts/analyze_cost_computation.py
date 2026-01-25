#!/usr/bin/env python3
"""
analyze_cost_computation.py

Deep analysis of cost computation differences between Python and C++/CUDA implementations.
Focuses on identifying the root cause of systematic bias (3.46m).
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Set up library paths
torch_lib_path = os.path.join(torch.utils.cmake_prefix_path, "..", "lib")
if os.path.exists(torch_lib_path):
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if torch_lib_path not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}"

# Add ROS2 library path
ros2_lib_path = "/home/motoken/college/ros2_ws/install/my_stereo_pkg/lib"
if os.path.exists(ros2_lib_path):
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if ros2_lib_path not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{ros2_lib_path}:{current_ld_path}"

# Change to sphere-stereo directory for Python imports
original_cwd = os.getcwd()
sphere_stereo_path = "/home/motoken/college/sphere-stereo"
if os.path.exists(sphere_stereo_path):
    os.chdir(sphere_stereo_path)
    sys.path.insert(0, os.path.join(sphere_stereo_path, "python"))

try:
    import sys
    sys.path.append("/home/motoken/college/sphere-stereo/python")
    from depth_estimation import RGBD_Estimator as PythonRGBD_Estimator
    print("✓ Python implementation loaded")
except ImportError as e:
    print(f"✗ Failed to import Python implementation: {e}")
    # Try alternative import path
    try:
        sys.path.append("/home/motoken/college/sphere-stereo")
        from python.depth_estimation import RGBD_Estimator as PythonRGBD_Estimator
        print("✓ Python implementation loaded (alternative path)")
    except ImportError as e2:
        print(f"✗ Failed to import Python implementation (alternative): {e2}")
        sys.exit(1)

# Load C++/CUDA implementation
sys.path.append("/home/motoken/college/ros2_ws/install/my_stereo_pkg/lib/python3.8/site-packages")
try:
    import my_stereo_pkg
    print("✓ C++/CUDA implementation loaded")
except ImportError as e:
    print(f"✗ Failed to import C++/CUDA implementation: {e}")
    sys.exit(1)

def analyze_camera_selection_differences():
    """
    Analyze camera selection algorithm differences between Python and C++.
    """
    print("=" * 80)
    print("CAMERA SELECTION ANALYSIS")
    print("=" * 80)
    
    # Generate test data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Simple test configuration
    config = {
        'min_dist': 1.0,
        'max_dist': 10.0,
        'candidate_count': 64,
        'references_indices': [0, 1],
        'matching_width': 640,
        'matching_height': 480,
        'pano_width': 1280,
        'pano_height': 640,
        'sigma_i': 30.0,
        'sigma_s': 30.0
    }
    
    # Generate synthetic calibrations (simplified)
    num_cameras = 4
    calibrations = []
    
    # Import calibration class
    sys.path.append("/home/motoken/college/sphere-stereo/python")
    from calibration import DoubleSphereCalibration
    
    for i in range(num_cameras):
        calib = DoubleSphereCalibration()
        calib.width = config['matching_width']
        calib.height = config['matching_height']
        calib.fl = torch.tensor([300.0, 300.0], device=device)
        calib.principal = torch.tensor([config['matching_width']/2, config['matching_height']/2], device=device)
        calib.xi = torch.tensor(-0.2, device=device) 
        calib.alpha = torch.tensor(0.6, device=device)
        
        # Different camera positions
        angle = i * np.pi / 2  # 90 degree spacing
        calib.rt = torch.eye(4, device=device)
        calib.rt[:3, 3] = torch.tensor([np.cos(angle), np.sin(angle), 0.0], device=device)
        
        calibrations.append(calib)
    
    # Generate synthetic images
    images = []
    for i in range(num_cameras):
        # Create gradient pattern for testing
        h, w = config['matching_height'], config['matching_width']
        x, y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='ij')
        
        # Create pattern based on camera position
        pattern = ((x + i * 100) % 255).float()
        img = torch.stack([pattern, pattern * 0.8, pattern * 0.6], dim=-1)
        img = img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        images.append(img.to(device))
    
    # Generate masks
    masks = []
    for i in range(num_cameras):
        mask = torch.ones(1, 1, config['matching_height'], config['matching_width'], device=device)
        masks.append(mask)
    
    print(f"Generated {num_cameras} synthetic cameras with {config['matching_width']}x{config['matching_height']} images")
    
    # Initialize Python RGBD_Estimator
    print("\n[Python] Initializing and analyzing camera selection...")
    py_estimator = PythonRGBD_Estimator(
        min_dist=config['min_dist'],
        max_dist=config['max_dist'],
        candidate_count=config['candidate_count'],
        matching_resolution=(config['matching_width'], config['matching_height']),
        device=device
    )
    
    # Set reference indices
    py_estimator.references_indices = config['references_indices']
    
    # Run camera selection
    py_estimator.select_best_cameras(calibrations, images, masks)
    
    print(f"  ✓ Python camera selection complete")
    print(f"    Selected cameras shape: {len(py_estimator.selected_cameras)}")
    
    if len(py_estimator.selected_cameras) > 0:
        selected_cam_0 = py_estimator.selected_cameras[0]
        unique_selections = torch.unique(selected_cam_0)
        print(f"    Unique camera selections: {unique_selections.cpu().numpy()}")
        
        # Analyze selection distribution
        for cam_id in unique_selections:
            if cam_id >= 0:
                count = (selected_cam_0 == cam_id).sum().item()
                percentage = count / selected_cam_0.numel() * 100
                print(f"    Camera {cam_id}: {count} pixels ({percentage:.1f}%)")
    
    # Initialize C++/CUDA RGBD_Estimator
    print("\n[C++/CUDA] Initializing and analyzing camera selection...")
    
    # Convert calibrations to flat format for C++
    calibrations_flat = []
    for calib in calibrations:
        calib_dict = {
            'width': int(calib.width),
            'height': int(calib.height), 
            'fl': calib.fl.cpu().numpy().astype(np.float32),
            'principal': calib.principal.cpu().numpy().astype(np.float32),
            'xi': float(calib.xi),
            'alpha': float(calib.alpha),
            'rt': calib.rt.cpu().numpy().astype(np.float32)
        }
        calibrations_flat.append(calib_dict)
    
    # Convert images to uint8 RGBA format
    images_uint8 = []
    for img in images:
        img_np = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_rgba = np.zeros((img_np.shape[0], img_np.shape[1], 4), dtype=np.uint8)
        img_rgba[..., :3] = img_np
        img_rgba[..., 3] = 255  # Alpha channel
        images_uint8.append(img_rgba)
    
    try:
        cuda_estimator = my_stereo_pkg.RGBD_Estimator(
            config['min_dist'], config['max_dist'], config['candidate_count'],
            config['references_indices'], config['matching_width'], config['matching_height'],
            config['pano_width'], config['pano_height'], config['sigma_i'], config['sigma_s'],
            0  # device_id
        )
        
        result = cuda_estimator.estimate_RGBD_panorama(calibrations_flat, images_uint8)
        
        print(f"  ✓ C++/CUDA estimation complete")
        print(f"    Result keys: {result.keys()}")
        
    except Exception as e:
        print(f"  ✗ C++/CUDA estimation failed: {e}")
        return
    
    # Compare camera selection strategies
    print("\n[Comparison] Camera Selection Strategy Analysis")
    print("  Python method: Displacement-based selection")
    print("  C++/CUDA method: Best camera selection kernel")
    print("  → Potential difference in displacement calculation or thresholds")
    
    return True

def analyze_distance_parameterization():
    """
    Analyze differences in distance parameterization between implementations.
    """
    print("\n" + "=" * 80)
    print("DISTANCE PARAMETERIZATION ANALYSIS")
    print("=" * 80)
    
    min_dist, max_dist, candidate_count = 1.0, 10.0, 64
    
    # Python method
    py_candidates = 1 / torch.linspace(1 / min_dist, 1 / max_dist, candidate_count)
    
    # C++ method (from CUDA kernel)
    def cpp_distance_candidates(min_d, max_d, count):
        inv_min = 1.0 / max_d  # Note: inverted order
        inv_max = 1.0 / min_d
        candidates = []
        for i in range(count):
            inv_dist = inv_min + (inv_max - inv_min) * (i / (count - 1))
            candidates.append(1.0 / inv_dist)
        return np.array(candidates)
    
    cpp_candidates = cpp_distance_candidates(min_dist, max_dist, candidate_count)
    
    print(f"Distance range: {min_dist}m to {max_dist}m, {candidate_count} candidates")
    print(f"\nPython candidates:")
    print(f"  First 5: {py_candidates[:5].numpy()}")
    print(f"  Last 5: {py_candidates[-5:].numpy()}")
    print(f"  Total range: {py_candidates[0]:.3f} - {py_candidates[-1]:.3f}")
    
    print(f"\nC++ candidates:")
    print(f"  First 5: {cpp_candidates[:5]}")
    print(f"  Last 5: {cpp_candidates[-5:]}")
    print(f"  Total range: {cpp_candidates[0]:.3f} - {cpp_candidates[-1]:.3f}")
    
    # Compare differences
    py_np = py_candidates.cpu().numpy()
    diff = np.abs(py_np - cpp_candidates)
    
    print(f"\nDifferences:")
    print(f"  Max difference: {diff.max():.6f}m")
    print(f"  Mean difference: {diff.mean():.6f}m")
    print(f"  RMS difference: {np.sqrt((diff**2).mean()):.6f}m")
    
    if diff.max() > 1e-6:
        print("  ⚠️ Significant differences detected in distance parameterization!")
        
        # Find order difference
        py_order = np.argsort(py_np)
        cpp_order = np.argsort(cpp_candidates)
        
        if not np.array_equal(py_order, cpp_order):
            print("  → Distance ordering is different between implementations")
        else:
            print("  → Distance ordering is consistent")
    else:
        print("  ✓ Distance parameterization is consistent")
    
    return True

def main():
    """
    Main analysis function for cost computation differences.
    """
    print("Deep Analysis of Cost Computation Differences")
    print("=" * 80)
    
    try:
        # Analyze distance parameterization first
        if not analyze_distance_parameterization():
            print("Distance parameterization analysis failed")
            return False
        
        # Analyze camera selection differences
        if not analyze_camera_selection_differences():
            print("Camera selection analysis failed")
            return False
        
        print("\n" + "=" * 80)
        print("COST COMPUTATION ANALYSIS COMPLETE")
        print("=" * 80)
        print("\nKey Findings:")
        print("1. Distance parameterization differences may contribute to systematic bias")
        print("2. Camera selection strategy differences likely affect cost computation")
        print("3. Texture sampling and coordinate system differences require investigation")
        print("\nRecommendations:")
        print("1. Unify distance candidate generation between Python and C++")
        print("2. Verify camera selection displacement calculation consistency")
        print("3. Check texture coordinate normalization and sampling methods")
        
        return True
        
    except Exception as e:
        print(f"Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)