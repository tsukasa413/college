#!/usr/bin/env python3
"""
Test script for CUDA-accelerated Depth Estimation
"""

import sys
import numpy as np

def test_basic_import():
    """Test that the module can be imported"""
    print("=" * 70)
    print("Test 1: Basic Module Import")
    print("=" * 70)
    
    try:
        import sphere_stereo_cuda
        print("✓ sphere_stereo_cuda imported successfully")
        
        # Check available classes
        print(f"✓ RGBD_Estimator available: {hasattr(sphere_stereo_cuda, 'RGBD_Estimator')}")
        print(f"✓ DoubleSphereCalibration available: {hasattr(sphere_stereo_cuda, 'DoubleSphereCalibration')}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        return False

def test_estimator_creation():
    """Test creating an RGBD_Estimator instance"""
    print("\n" + "=" * 70)
    print("Test 2: RGBD_Estimator Creation")
    print("=" * 70)
    
    try:
        import sphere_stereo_cuda
        
        # Mock calibration data
        num_cameras = 2
        
        # RT matrices (identity for simplicity)
        calibrations_rt = []
        for i in range(num_cameras):
            rt = [1.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.0, 1.0]
            calibrations_rt.extend(rt)
        
        # Intrinsics [fx, fy, cx, cy]
        calibrations_intrinsics = [500.0, 500.0, 320.0, 240.0] * num_cameras
        
        # Sphere model [xi, alpha]
        calibrations_sphere = [0.5, 0.1] * num_cameras
        
        # Resolution [width, height]
        calibrations_resolution = [640.0, 480.0] * num_cameras
        
        # Other parameters
        min_dist = 0.5
        max_dist = 10.0
        candidate_count = 50
        references_indices = [0]
        reprojection_viewpoint = [0.0, 0.0, 0.0]
        image_widths = [640, 640]
        image_heights = [480, 480]
        matching_width = 320
        matching_height = 240
        rgb_to_stitch_width = 640
        rgb_to_stitch_height = 480
        panorama_width = 512
        panorama_height = 512
        sigma_i = 1.0
        sigma_s = 1.0
        device = 0
        
        print("Creating RGBD_Estimator...")
        estimator = sphere_stereo_cuda.RGBD_Estimator(
            calibrations_rt,
            calibrations_intrinsics,
            calibrations_sphere,
            calibrations_resolution,
            min_dist,
            max_dist,
            candidate_count,
            references_indices,
            reprojection_viewpoint,
            image_widths,
            image_heights,
            matching_width,
            matching_height,
            rgb_to_stitch_width,
            rgb_to_stitch_height,
            panorama_width,
            panorama_height,
            sigma_i,
            sigma_s,
            device
        )
        
        print("✓ RGBD_Estimator created successfully")
        print(f"  - Number of cameras: {num_cameras}")
        print(f"  - Matching resolution: {matching_width}x{matching_height}")
        print(f"  - Distance range: {min_dist} - {max_dist}")
        print(f"  - Candidate count: {candidate_count}")
        
        return True, estimator
    except Exception as e:
        print(f"✗ Failed to create estimator: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_estimation_pipeline(estimator):
    """Test the full estimation pipeline with dummy data"""
    print("\n" + "=" * 70)
    print("Test 3: Estimation Pipeline")
    print("=" * 70)
    
    try:
        # Generate dummy images
        num_cameras = 2
        matching_width = 320
        matching_height = 240
        
        print("Generating dummy image data...")
        images_to_match = []
        images_to_stitch = []
        
        for i in range(num_cameras):
            # Matching images (320x240x3)
            img_match = np.random.rand(matching_height * matching_width * 3).astype(np.float32) * 255.0
            images_to_match.append(img_match.tolist())
            
            # Stitching images (640x480x3)
            img_stitch = np.random.rand(480 * 640 * 3).astype(np.float32) * 255.0
            images_to_stitch.append(img_stitch.tolist())
        
        print("✓ Dummy images generated")
        print(f"  - Matching images: {len(images_to_match)} @ {matching_width}x{matching_height}")
        print(f"  - Stitching images: {len(images_to_stitch)} @ 640x480")
        
        print("\nRunning depth estimation pipeline...")
        rgb_panorama, distance_panorama = estimator.estimate_RGBD_panorama(
            images_to_match,
            images_to_stitch
        )
        
        print("✓ Estimation completed")
        print(f"  - RGB panorama shape: {rgb_panorama.shape if hasattr(rgb_panorama, 'shape') else len(rgb_panorama)}")
        print(f"  - Distance panorama shape: {distance_panorama.shape if hasattr(distance_panorama, 'shape') else len(distance_panorama)}")
        
        return True
    except Exception as e:
        print(f"✗ Estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("CUDA Depth Estimation - Python Binding Tests")
    print("=" * 70)
    
    # Test 1: Import
    if not test_basic_import():
        print("\n✗ Import test failed. Exiting.")
        return 1
    
    # Test 2: Creation
    success, estimator = test_estimator_creation()
    if not success:
        print("\n✗ Creation test failed. Exiting.")
        return 1
    
    # Test 3: Pipeline
    if not test_estimation_pipeline(estimator):
        print("\n✗ Pipeline test failed. Exiting.")
        return 1
    
    # Summary
    print("\n" + "=" * 70)
    print("✓ All Tests Passed")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
