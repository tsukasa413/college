#!/usr/bin/env python3
"""
test_python_units.py

Unified Python Unit Test Suite
===========================================================================
Combines and consolidates:
- test_depth_estimation_python.py: CUDA module import and basic functionality
- test_geometry.py: Double sphere roundtrip geometry test
- Parts of verify_utils.py: Python reference implementations
===========================================================================
"""

import sys
import os
import numpy as np
import torch
import unittest
import traceback
from pathlib import Path

# Setup paths
sys.path.insert(0, '/home/motoken/college/sphere-stereo/python')
sys.path.insert(0, '/home/motoken/college/ros2_ws/install/my_stereo_pkg/lib/python3.8/site-packages')

class TestBasicImports(unittest.TestCase):
    """Test 1: Basic module imports and availability"""
    
    def test_python_sphere_stereo_import(self):
        """Test Python sphere-stereo module imports"""
        try:
            from utils import project, unproject
            from depth_estimation import RGBD_Estimator
            self.assertTrue(True, "Python sphere-stereo imports successful")
        except ImportError as e:
            self.fail(f"Failed to import Python sphere-stereo: {e}")
    
    def test_cuda_module_import(self):
        """Test C++/CUDA module import"""
        try:
            import my_stereo_pkg
            self.assertTrue(hasattr(my_stereo_pkg, 'RGBD_Estimator'), 
                          "RGBD_Estimator class available")
            print("✓ C++/CUDA RGBD_Estimator available")
        except ImportError as e:
            self.skipTest(f"C++/CUDA module not available: {e}")


class TestGeometry(unittest.TestCase):
    """Test 2: Double sphere geometry functions"""
    
    def setUp(self):
        """Setup test calibration"""
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Create test calibration
        class Calibration:
            def __init__(self, device):
                self.device = device
                self.fl = torch.tensor([300.0, 300.0], device=self.device)
                self.principal = torch.tensor([320.0, 240.0], device=self.device)
                self.xi = 0.1
                self.alpha = 0.5
                self.matching_scale = 1.0
                self.rt = torch.eye(4, device=self.device)
        
        self.calib = Calibration(self.device)
    
    def test_geometry_roundtrip(self):
        """Test unproject -> project roundtrip"""
        try:
            from utils import project, unproject
            
            # Test UV coordinates
            uv_test = torch.tensor([
                [320.0, 240.0],  # center
                [200.0, 150.0],  # offset
                [440.0, 330.0],  # another offset
            ], device=self.device).unsqueeze(0)
            
            print(f"Testing geometry roundtrip with {uv_test.shape[1]} points")
            
            # Unproject to 3D
            pt_3d, valid = unproject(uv_test, self.calib)
            self.assertTrue(valid.all().item(), "All unproject operations should be valid")
            
            # Project back to 2D
            uv_reprojected, valid_reproj = project(pt_3d, self.calib)
            self.assertTrue(valid_reproj.all().item(), "All reproject operations should be valid")
            
            # Check roundtrip accuracy
            error = torch.abs(uv_test - uv_reprojected).max()
            max_allowed_error = 1e-3  # 0.001 pixel
            
            self.assertLess(error.item(), max_allowed_error, 
                          f"Roundtrip error {error.item():.6f} exceeds threshold {max_allowed_error}")
            
            print(f"✓ Geometry roundtrip passed with max error: {error.item():.6f} pixels")
            
        except ImportError as e:
            self.skipTest(f"Geometry functions not available: {e}")


class TestRGBDEstimator(unittest.TestCase):
    """Test 3: RGBD_Estimator basic functionality"""
    
    def test_python_estimator_creation(self):
        """Test Python RGBD_Estimator creation"""
        try:
            from depth_estimation import RGBD_Estimator
            from utils import Calibration as DoubleSphereCalibration
            
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
            # Create minimal calibration with proper parameters
            calib = DoubleSphereCalibration(
                original_resolution=(320, 240),
                principal=torch.tensor([160.0, 120.0], device=device),
                fl=torch.tensor([250.0, 250.0], device=device),
                xi=-0.2,
                alpha=0.6,
                rt=torch.eye(4, device=device),
                matching_scale=torch.tensor([1.0, 1.0], device=device)
            )
            
            calibrations = [calib]
            masks = [torch.ones(1, 240, 320, device=device)]
            
            estimator = RGBD_Estimator(
                calibrations=calibrations,
                min_dist=1.0,
                max_dist=10.0,
                candidate_count=32,
                references_indices=[0],
                reprojection_viewpoint=torch.tensor([0.0, 0.0, 0.0], device=device),
                masks=masks,
                matching_resolution=(320, 240),
                rgb_to_stitch_resolution=(320, 240),
                panorama_resolution=(640, 320),
                sigma_i=30.0,
                sigma_s=30.0,
                device=device
            )
            
            self.assertIsNotNone(estimator, "Python RGBD_Estimator should be created")
            self.assertEqual(estimator.min_dist, 1.0, "Min distance should be set correctly")
            self.assertEqual(estimator.max_dist, 10.0, "Max distance should be set correctly")
            self.assertEqual(estimator.candidate_count, 32, "Candidate count should be set correctly")
            
            print(f"✓ Python RGBD_Estimator created successfully")
            
        except ImportError as e:
            self.skipTest(f"Python RGBD_Estimator not available: {e}")
            
            print("✓ Python RGBD_Estimator created successfully")
            
        except ImportError as e:
            self.skipTest(f"Python RGBD_Estimator not available: {e}")
    
    def test_cuda_estimator_creation(self):
        """Test C++/CUDA RGBD_Estimator creation"""
        try:
            import my_stereo_pkg
            
            # Test configuration
            config = {
                'min_dist': 1.0,
                'max_dist': 10.0,
                'candidate_count': 32,
                'references_indices': [0],
                'matching_width': 320,
                'matching_height': 240,
                'rgb_width': 320,
                'rgb_height': 240,
                'pano_width': 640,
                'pano_height': 320,
                'sigma_i': 30.0,
                'sigma_s': 30.0
            }
            
            estimator = my_stereo_pkg.RGBD_Estimator(
                config['min_dist'], config['max_dist'], config['candidate_count'],
                config['references_indices'], config['matching_width'], config['matching_height'],
                config['pano_width'], config['pano_height'], 
                config['sigma_i'], config['sigma_s'], 0  # device_id
            )
            
            self.assertIsNotNone(estimator, "C++/CUDA RGBD_Estimator should be created")
            print("✓ C++/CUDA RGBD_Estimator created successfully")
            
        except ImportError as e:
            self.skipTest(f"C++/CUDA RGBD_Estimator not available: {e}")
        except Exception as e:
            self.fail(f"C++/CUDA RGBD_Estimator creation failed: {e}")


class TestUtilityFunctions(unittest.TestCase):
    """Test 4: Utility function implementations"""
    
    def test_double_sphere_reference_implementation(self):
        """Test CPU reference implementations of double sphere model"""
        
        def unproject_reference(u, v, fx, fy, cx, cy, xi, alpha):
            """Reference implementation for unprojection"""
            mx = (u - cx) / fx
            my = (v - cy) / fy
            r2 = mx * mx + my * my
            
            denom = alpha * r2 + 1.0 - (2.0 * alpha - 1.0) * r2 * xi
            if denom < 0.0001:
                return None, None, None, False
            
            numerator = 1.0 - xi * xi * r2
            z_sphere = numerator / denom
            
            x_3d = mx * z_sphere
            y_3d = my * z_sphere
            z_3d = z_sphere - xi
            
            return x_3d, y_3d, z_3d, True
        
        # Test with known values
        fx, fy = 300.0, 300.0
        cx, cy = 320.0, 240.0
        xi, alpha = 0.1, 0.5
        
        # Test center pixel
        x, y, z, valid = unproject_reference(cx, cy, fx, fy, cx, cy, xi, alpha)
        
        self.assertTrue(valid, "Center pixel should unproject successfully")
        self.assertIsNotNone(x, "X coordinate should be computed")
        self.assertAlmostEqual(x, 0.0, places=6, msg="Center pixel X should be 0")
        self.assertAlmostEqual(y, 0.0, places=6, msg="Center pixel Y should be 0")
        
        print(f"✓ Reference implementation test passed: ({x:.6f}, {y:.6f}, {z:.6f})")


def run_comprehensive_tests():
    """Run all unit tests with detailed reporting"""
    
    print("=" * 80)
    print("Unified Python Unit Test Suite")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestBasicImports,
        TestGeometry, 
        TestRGBDEstimator,
        TestUtilityFunctions
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Results Summary")
    print("=" * 80)
    print(f"Total Tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.splitlines()[-1]}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("="*80)
    print("Unified Python Unit Test Suite")
    print("="*80)
    print(f"\nThis suite consolidates:")
    print(f"  • test_geometry.py - Geometry function tests")
    print(f"  • test_depth_estimation_python.py - RGBD estimator tests")
    print(f"  • verify_utils.py - Utility function tests")
    print(f"\nTotal Test Classes: 4")
    print(f"  1. TestBasicImports - Module import verification")
    print(f"  2. TestGeometry - Double sphere geometry")
    print(f"  3. TestRGBDEstimator - RGBD estimator creation")
    print(f"  4. TestUtilityFunctions - Utility functions")
    print()
    
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)