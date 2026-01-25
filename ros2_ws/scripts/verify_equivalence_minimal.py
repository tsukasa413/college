#!/usr/bin/env python3
"""
verify_equivalence_minimal.py

Minimal equivalence test to debug CUDA memory issues.
Tests with smaller resolution and verbose output.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add C++/CUDA module to path
cuda_module_path = Path(__file__).parent.parent / "install" / "my_stereo_pkg" / "lib"
sys.path.insert(0, str(cuda_module_path))

try:
    import sphere_stereo_cuda
    print("✓ C++/CUDA module loaded")
except ImportError as e:
    print(f"✗ Failed to load C++/CUDA module: {e}")
    sys.exit(1)

def test_minimal():
    """
    Minimal test with small resolution to isolate memory issue.
    """
    print("\n" + "="*80)
    print("MINIMAL CUDA TEST")
    print("="*80)
    
    # Very small configuration
    width, height = 320, 240
    pano_width, pano_height = 640, 320
    num_cameras = 2  # Only 2 cameras
    
    config = {
        'min_dist': 1.0,
        'max_dist': 10.0,
        'candidate_count': 32,  # Reduced
        'references_indices': [0],  # Only 1 reference
        'matching_width': width,
        'matching_height': height,
        'rgb_width': width,
        'rgb_height': height,
        'pano_width': pano_width,
        'pano_height': pano_height,
        'sigma_i': 30.0,
        'sigma_s': 30.0,
    }
    
    print(f"\nTest Configuration:")
    print(f"  Image size: {width}x{height}")
    print(f"  Cameras: {num_cameras}")
    print(f"  References: {config['references_indices']}")
    print(f"  Candidates: {config['candidate_count']}")
    
    # Generate simple calibrations
    fx, fy = 200.0, 200.0
    cx, cy = width / 2.0, height / 2.0
    xi, alpha = 0.5, 0.6
    
    camera_positions = [
        [0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0],
    ]
    
    rt_matrices = []
    intrinsics_list = []
    sphere_params_list = []
    resolution_list = []
    
    for pos in camera_positions:
        rt = np.eye(4, dtype=np.float32)
        rt[0, 3] = pos[0]
        rt[1, 3] = pos[1]
        rt[2, 3] = pos[2]
        
        rt_matrices.extend(rt.flatten().tolist())
        intrinsics_list.extend([fx, fy, cx, cy])
        sphere_params_list.extend([xi, alpha])
        resolution_list.extend([width, height])
    
    print("\nCalibration data sizes:")
    print(f"  RT matrices: {len(rt_matrices)} floats ({num_cameras} * 16)")
    print(f"  Intrinsics: {len(intrinsics_list)} floats ({num_cameras} * 4)")
    print(f"  Sphere params: {len(sphere_params_list)} floats ({num_cameras} * 2)")
    print(f"  Resolutions: {len(resolution_list)} ints ({num_cameras} * 2)")
    
    # Generate simple test images (constant gradient)
    images_np = []
    for i in range(num_cameras):
        img = np.zeros((height, width, 3), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                img[y, x, 0] = (x / width) * 255
                img[y, x, 1] = (y / height) * 255
                img[y, x, 2] = 128 + i * 50
        
        print(f"  Image {i} shape: {img.shape}, range: [{img.min():.1f}, {img.max():.1f}]")
        images_np.append(img.flatten().tolist())
    
    images_to_stitch = [images_np[i] for i in config['references_indices']]
    
    print(f"\nImage data sizes:")
    print(f"  Images to match: {len(images_np)} x {len(images_np[0])} floats")
    print(f"  Images to stitch: {len(images_to_stitch)} x {len(images_to_stitch[0])} floats")
    
    # Create estimator
    print("\nInitializing RGBD_Estimator...")
    try:
        estimator = sphere_stereo_cuda.RGBD_Estimator(
            calibrations_rt=rt_matrices,
            calibrations_intrinsics=intrinsics_list,
            calibrations_sphere=sphere_params_list,
            calibrations_resolution=resolution_list,
            min_dist=config['min_dist'],
            max_dist=config['max_dist'],
            candidate_count=config['candidate_count'],
            references_indices=config['references_indices'],
            reprojection_viewpoint=[0.0, 0.0, 0.0],
            image_widths=[width] * num_cameras,
            image_heights=[height] * num_cameras,
            matching_width=config['matching_width'],
            matching_height=config['matching_height'],
            rgb_to_stitch_width=config['rgb_width'],
            rgb_to_stitch_height=config['rgb_height'],
            panorama_width=config['pano_width'],
            panorama_height=config['pano_height'],
            sigma_i=config['sigma_i'],
            sigma_s=config['sigma_s'],
            device=0
        )
        print("✓ Estimator created successfully")
    except Exception as e:
        print(f"✗ Estimator creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run estimation
    print("\nRunning estimate_RGBD_panorama...")
    try:
        rgb_pano_flat, distance_pano_flat = estimator.estimate_RGBD_panorama(
            images_np, images_to_stitch
        )
        print("✓ Estimation complete")
        
        # Check output sizes
        expected_rgb_size = pano_height * pano_width * 3
        expected_dist_size = pano_height * pano_width
        
        print(f"\nOutput sizes:")
        print(f"  RGB: {len(rgb_pano_flat)} (expected: {expected_rgb_size})")
        print(f"  Distance: {len(distance_pano_flat)} (expected: {expected_dist_size})")
        
        if len(rgb_pano_flat) == expected_rgb_size and len(distance_pano_flat) == expected_dist_size:
            print("\n✓ TEST PASSED: Output sizes correct")
            
            # Reshape and show statistics
            rgb_pano = np.array(rgb_pano_flat, dtype=np.uint8).reshape(pano_height, pano_width, 3)
            distance_pano = np.array(distance_pano_flat, dtype=np.float32).reshape(pano_height, pano_width)
            
            print(f"\nRGB panorama statistics:")
            print(f"  Shape: {rgb_pano.shape}")
            print(f"  Range: [{rgb_pano.min()}, {rgb_pano.max()}]")
            print(f"  Mean: {rgb_pano.mean():.1f}")
            
            print(f"\nDistance panorama statistics:")
            print(f"  Shape: {distance_pano.shape}")
            print(f"  Range: [{distance_pano.min():.3f}, {distance_pano.max():.3f}]")
            print(f"  Mean: {distance_pano.mean():.3f}")
            print(f"  Valid (min<d<max): {np.sum((distance_pano > config['min_dist']) & (distance_pano < config['max_dist']))}")
        else:
            print("\n✗ TEST FAILED: Output sizes mismatch")
            
    except Exception as e:
        print(f"✗ Estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    print("MINIMAL TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_minimal()
