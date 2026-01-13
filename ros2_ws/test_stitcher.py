#!/usr/bin/env python3
"""
Test script for Stitcher C++ class via Python bindings
"""
import sys
import torch
import numpy as np

# Source the workspace
sys.path.insert(0, '/home/motoken/college/ros2_ws/install/my_stereo_pkg/lib/python3.10/site-packages')

try:
    from my_stereo_pkg import _core_cpp
    print("✅ Successfully imported _core_cpp module")
    
    # Check available classes and functions
    print("\n📦 Available classes and functions:")
    for name in dir(_core_cpp):
        if not name.startswith('_'):
            print(f"   - {name}")
    
    # Test Calibration struct
    print("\n🔧 Testing Calibration struct:")
    calib = _core_cpp.Calibration()
    calib.fl = (500.0, 500.0)
    calib.principal = (320.0, 240.0)
    calib.xi = 1.2
    calib.alpha = 0.5
    calib.matching_scale = 0.5
    calib.rt = torch.eye(4, dtype=torch.float32)
    print(f"   Created Calibration: fl={calib.fl}, principal={calib.principal}")
    print(f"   xi={calib.xi}, alpha={calib.alpha}, scale={calib.matching_scale}")
    print(f"   rt shape: {calib.rt.shape}")
    
    # Test Stitcher instantiation
    print("\n🎨 Testing Stitcher class:")
    print("   Creating test parameters...")
    
    # Create minimal test setup (2 cameras)
    num_cameras = 2
    matching_cols = 640
    matching_rows = 480
    rgb_cols = 640  # Must match matching resolution for stitching
    rgb_rows = 480  # Must match matching resolution for stitching
    pano_cols = 1280
    pano_rows = 640
    
    # Create calibrations
    calibrations = []
    for i in range(num_cameras):
        cal = _core_cpp.Calibration()
        cal.fl = (500.0, 500.0)
        cal.principal = (matching_cols / 2.0, matching_rows / 2.0)
        cal.xi = 1.2
        cal.alpha = 0.5
        cal.matching_scale = 1.0
        cal.rt = torch.eye(4, dtype=torch.float32)
        # Add small offset for second camera
        if i == 1:
            cal.rt[0, 3] = 0.1  # 10cm baseline
        calibrations.append(cal)
    
    # Create reprojection viewpoint (origin)
    reprojection_viewpoint = torch.zeros(3, dtype=torch.float32)
    
    # Create masks (all valid)
    masks = torch.ones(num_cameras, matching_rows, matching_cols, dtype=torch.float32)
    
    # CUDA device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    # Move tensors to device
    reprojection_viewpoint = reprojection_viewpoint.to(device)
    masks = masks.to(device)
    for cal in calibrations:
        cal.rt = cal.rt.to(device)
    
    print(f"   Creating Stitcher with {num_cameras} cameras...")
    try:
        stitcher = _core_cpp.Stitcher(
            calibrations,
            reprojection_viewpoint,
            masks,
            0.1,  # min_dist
            10.0,  # max_dist
            matching_cols,
            matching_rows,
            rgb_cols,
            rgb_rows,
            pano_cols,
            pano_rows,
            device,
            15,  # smoothing_radius
            32   # inpainting_iterations
        )
        print("   ✅ Stitcher created successfully!")
        
        # Test stitch method with dummy data
        print("\n🔄 Testing stitch method with dummy data...")
        images = []
        distance_maps = []
        for i in range(num_cameras):
            # Create dummy RGB image [H, W, 3] uint8
            img = torch.randint(0, 255, (rgb_rows, rgb_cols, 3), dtype=torch.uint8, device=device)
            images.append(img)
            
            # Create dummy distance map [H, W] float32
            dist = torch.rand(rgb_rows, rgb_cols, dtype=torch.float32, device=device) * 5.0 + 0.5
            distance_maps.append(dist)
        
        print(f"   Input: {num_cameras} images of shape {images[0].shape}")
        print(f"   Input: {num_cameras} distance maps of shape {distance_maps[0].shape}")
        
        # Call stitch
        rgb_pano, depth_pano = stitcher.stitch(images, distance_maps)
        
        print(f"   ✅ Stitch completed!")
        print(f"   Output RGB panorama: {rgb_pano.shape} {rgb_pano.dtype}")
        print(f"   Output depth panorama: {depth_pano.shape} {depth_pano.dtype}")
        
    except Exception as e:
        print(f"   ❌ Error creating/using Stitcher: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ All tests completed!")
    
except ImportError as e:
    print(f"❌ Failed to import module: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
