#!/usr/bin/env python3
"""
C++版とPython版の詳細比較
"""
import torch
import numpy as np
import sys
import ctypes
import os

# Add paths
sys.path.append('/home/motoken/college/ros2_ws/install/my_stereo_pkg/lib/python3.10/site-packages')
sys.path.append('/home/motoken/college/sphere-stereo/python')

# Import implementations
from my_stereo_pkg.my_stereo_pkg import RGBD_Estimator as RGBD_Estimator_CPP
from utils import Calibration, parse_json_calib
from depth_estimation import RGBD_Estimator as RGBD_Estimator_Python

def main():
    # Load calibration
    with open('/home/motoken/college/sphere-stereo/resources/calibration.json', 'r') as f:
        import json
        calibration_data = json.load(f)['value0']
    
    device = torch.device('cuda:0')
    matching_resolution = (640, 480)
    original_resolution = (1280, 1024)
    
    calibrations = parse_json_calib(
        calibration_data, matching_resolution, device, original_resolution
    )
    
    # Create synthetic images
    images = [torch.randn(480, 640, 3, device=device) * 50 + 128 for _ in range(4)]
    images_to_match = [img.clone() for img in images[:2]]
    images_to_stitch = [images[0].clone()]
    
    config = {
        'min_dist': 1.0,
        'max_dist': 10.0,
        'candidate_count': 64,
        'references_indices': [0, 1],
        'matching_width': 640,
        'matching_height': 480,
        'rgb_width': 640,
        'rgb_height': 480,
        'pano_width': 1280,
        'pano_height': 640,
        'sigma_i': 30.0,
        'sigma_s': 30.0
    }
    
    print("=" * 80)
    print("DETAILED COMPARISON TEST")
    print("=" * 80)
    
    # Python version
    print("\n[Python Path] Running...")
    estimator_py = RGBD_Estimator_Python(
        calibrations, config['min_dist'], config['max_dist'],
        config['candidate_count'], config['references_indices'],
        (config['matching_width'], config['matching_height']),
        (config['rgb_width'], config['rgb_height']),
        (config['pano_width'], config['pano_height']),
        device, config['sigma_i'], config['sigma_s']
    )
    
    rgb_py, distance_py = estimator_py.estimate_RGBD_panorama(images_to_match, images_to_stitch)
    
    # C++ version
    print("[C++/CUDA Path] Running...")
    
    # Prepare calibration arrays
    num_cameras = len(calibrations)
    rt_arrays = []
    intrinsics_arrays = []
    sphere_params_arrays = []
    resolutions_arrays = []
    
    for calib in calibrations:
        rt_arrays.extend(calib.rt.cpu().flatten().tolist())
        intrinsics_arrays.extend([calib.fl[0].item(), calib.fl[1].item(),
                                 calib.principal[0].item(), calib.principal[1].item()])
        sphere_params_arrays.extend([calib.xi, calib.alpha])
        resolutions_arrays.extend([calib.original_resolution[0], calib.original_resolution[1]])
    
    rt_arrays_flat = (ctypes.c_float * len(rt_arrays))(*rt_arrays)
    intrinsics_flat = (ctypes.c_float * len(intrinsics_arrays))(*intrinsics_arrays)
    sphere_params_flat = (ctypes.c_float * len(sphere_params_arrays))(*sphere_params_arrays)
    resolutions_flat = (ctypes.c_int * len(resolutions_arrays))(*resolutions_arrays)
    
    estimator_cpp = RGBD_Estimator_CPP(
        num_cameras,
        rt_arrays_flat, len(rt_arrays),
        intrinsics_flat, len(intrinsics_arrays),
        sphere_params_flat, len(sphere_params_arrays),
        resolutions_flat, len(resolutions_arrays),
        config['min_dist'], config['max_dist'],
        config['candidate_count'],
        config['references_indices'], len(config['references_indices']),
        config['matching_width'], config['matching_height'],
        config['rgb_width'], config['rgb_height'],
        config['pano_width'], config['pano_height'],
        config['sigma_i'], config['sigma_s']
    )
    
    # Convert images
    images_match_cpp = [(img.cpu().numpy().astype(np.uint8).flatten()).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
                        for img in images_to_match]
    images_stitch_cpp = [(img.cpu().numpy().astype(np.uint8).flatten()).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
                         for img in images_to_stitch]
    
    images_match_ptrs = (ctypes.c_void_p * len(images_match_cpp))(*[ctypes.cast(ptr, ctypes.c_void_p) for ptr in images_match_cpp])
    images_stitch_ptrs = (ctypes.c_void_p * len(images_stitch_cpp))(*[ctypes.cast(ptr, ctypes.c_void_p) for ptr in images_stitch_cpp])
    
    num_to_match = len(images_to_match)
    num_to_stitch = len(images_to_stitch)
    
    size_per_image_match = config['matching_height'] * config['matching_width'] * 3
    size_per_image_stitch = config['rgb_height'] * config['rgb_width'] * 3
    
    rgb_buffer = np.zeros(config['pano_height'] * config['pano_width'] * 3, dtype=np.uint8)
    distance_buffer = np.zeros(config['pano_height'] * config['pano_width'], dtype=np.float32)
    
    rgb_ptr = rgb_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    distance_ptr = distance_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    estimator_cpp.estimate_RGBD_panorama(
        images_match_ptrs, num_to_match, size_per_image_match,
        images_stitch_ptrs, num_to_stitch, size_per_image_stitch,
        rgb_ptr, distance_ptr
    )
    
    rgb_cpp = torch.from_numpy(rgb_buffer.reshape(config['pano_height'], config['pano_width'], 3))
    distance_cpp = torch.from_numpy(distance_buffer.reshape(config['pano_height'], config['pano_width']))
    
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    
    # Compare distances
    mask_py = (distance_py > config['min_dist']) & (distance_py < config['max_dist'])
    mask_cpp = (distance_cpp > config['min_dist']) & (distance_cpp < config['max_dist'])
    
    print(f"\nPython:")
    print(f"  Valid pixels: {mask_py.sum()}/{mask_py.numel()} ({100*mask_py.float().mean():.2f}%)")
    print(f"  Distance range: [{distance_py[mask_py].min():.4f}, {distance_py[mask_py].max():.4f}]")
    print(f"  Distance mean: {distance_py[mask_py].mean():.4f}")
    
    print(f"\nC++/CUDA:")
    print(f"  Valid pixels: {mask_cpp.sum()}/{mask_cpp.numel()} ({100*mask_cpp.float().mean():.2f}%)")
    print(f"  Distance range: [{distance_cpp[mask_cpp].min():.4f}, {distance_cpp[mask_cpp].max():.4f}]")
    print(f"  Distance mean: {distance_cpp[mask_cpp].mean():.4f}")
    
    # Compare where both valid
    mask_both = mask_py & mask_cpp
    if mask_both.sum() > 0:
        diff = torch.abs(distance_py[mask_both] - distance_cpp[mask_both])
        print(f"\nDifference (where both valid):")
        print(f"  MAE: {diff.mean():.4f} m")
        print(f"  Max error: {diff.max():.4f} m")
        print(f"  Median error: {diff.median():.4f} m")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
