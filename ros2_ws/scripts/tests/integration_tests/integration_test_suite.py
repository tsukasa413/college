#!/usr/bin/env python3
"""
integration_test_suite.py

Integration Test Suite
===========================================================================
Combines and consolidates:
- equivalence_test.py: Equivalence testing with real data
- compare_detailed.py: Detailed comparison between implementations
===========================================================================
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import time
import traceback
import argparse

# Setup library paths
torch_lib_path = os.path.join(torch.utils.cmake_prefix_path, "..", "lib")
if os.path.exists(torch_lib_path):
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if torch_lib_path not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}"

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
    from depth_estimation import RGBD_Estimator as PythonRGBD_Estimator
    from isb_filter import ISB_Filter as PythonISBFilter
    from utils import Calibration as DoubleSphereCalibration
    print("✓ Python implementations loaded")
except ImportError as e:
    print(f"✗ Failed to import Python implementations: {e}")
    sys.exit(1)

os.chdir(original_cwd)

try:
    sys.path.append("/home/motoken/college/ros2_ws/install/my_stereo_pkg/lib/python3.8/site-packages")
    import my_stereo_pkg
    print("✓ C++/CUDA implementations loaded")
except ImportError as e:
    print(f"✗ Failed to import C++/CUDA implementations: {e}")
    sys.exit(1)


class IntegrationTestSuite:
    """Integration test suite for end-to-end verification"""
    
    def __init__(self, device=None):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
        # Configuration
        self.config = {
            'min_dist': 1.0,
            'max_dist': 10.0,
            'candidate_count': 64,
            'references_indices': [0, 1],
            'matching_width': 320,
            'matching_height': 240,
            'rgb_width': 320,
            'rgb_height': 240,
            'pano_width': 640,
            'pano_height': 320,
            'sigma_i': 30.0,
            'sigma_s': 30.0
        }
    
    def load_test_dataset(self):
        """Load or generate test dataset"""
        print("Loading test dataset...")
        
        # For now, use synthetic data
        # In a real scenario, this would load actual camera data
        calibrations = self._generate_realistic_calibrations()
        images = self._generate_realistic_images()
        masks = self._generate_realistic_masks()
        
        return calibrations, images, masks
    
    def _generate_realistic_calibrations(self):
        """Generate realistic camera calibrations"""
        calibrations = []
        num_cameras = 6  # More cameras for better coverage
        
        # Use already imported calibration class
        # from utils import Calibration as DoubleSphereCalibration
        
        # Camera positions in a circular arrangement
        for i in range(num_cameras):
            width, height = self.config['matching_width'], self.config['matching_height']
            
            # Realistic fisheye parameters
            principal = torch.tensor([width/2 + i*2, height/2 + i*2], device=self.device)
            fl = torch.tensor([250.0 + i * 10, 250.0 + i * 10], device=self.device)
            
            # Circular camera arrangement
            angle = i * 2 * np.pi / num_cameras
            radius = 2.0
            rt = torch.eye(4, device=self.device)
            rt[:3, 3] = torch.tensor([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0.0
            ], device=self.device)
            
            # Camera looks toward center
            # Simplified rotation matrix (looking toward origin)
            look_angle = angle + np.pi
            rt[0, 0] = np.cos(look_angle)
            rt[0, 1] = -np.sin(look_angle)
            rt[1, 0] = np.sin(look_angle)
            rt[1, 1] = np.cos(look_angle)
            
            calib = DoubleSphereCalibration(
                original_resolution=(width, height),
                principal=principal,
                fl=fl,
                xi=-0.2 + i * 0.05,
                alpha=0.5 + i * 0.02,
                rt=rt,
                matching_scale=torch.tensor([1.0, 1.0], device=self.device)
            )
            
            calibrations.append(calib)
        
        print(f"Generated {len(calibrations)} realistic camera calibrations")
        return calibrations
    
    def _generate_realistic_images(self):
        """Generate realistic synthetic images"""
        num_cameras = 6
        images = []
        
        h, w = self.config['matching_height'], self.config['matching_width']
        
        for i in range(num_cameras):
            # Create more complex pattern simulating real-world scene
            x, y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='ij')
            x_norm = x.float() / w - 0.5
            y_norm = y.float() / h - 0.5
            
            # Simulate different textures and lighting per camera
            base_pattern = 100 + 50 * torch.sin(x_norm * 10 + i) * torch.cos(y_norm * 8 + i)
            
            # Add radial pattern (fisheye-like distortion effect)
            radius = torch.sqrt(x_norm**2 + y_norm**2)
            radial_effect = 1.0 + 0.3 * torch.sin(radius * 5)
            
            # Add noise and lighting variation
            noise = 20 * torch.randn_like(base_pattern)
            lighting = 0.8 + 0.4 * torch.sin(x_norm * 2) * torch.cos(y_norm * 2)
            
            pattern = (base_pattern * radial_effect * lighting + noise).clamp(0, 255)
            
            # Create RGB image with color variation
            r_channel = pattern
            g_channel = pattern * (0.9 + 0.2 * torch.sin(x_norm * 3))
            b_channel = pattern * (0.8 + 0.3 * torch.cos(y_norm * 3))
            
            img = torch.stack([r_channel, g_channel, b_channel], dim=0).permute(1, 2, 0)
            img = img.clamp(0, 255) / 255.0  # Normalize to [0, 1]
            
            # Convert to proper format [C, H, W]
            img = img.permute(2, 0, 1).to(self.device)
            images.append(img)
        
        print(f"Generated {len(images)} realistic synthetic images")
        return images
    
    def _generate_realistic_masks(self):
        """Generate realistic masks with some occlusions"""
        num_cameras = 6
        masks = []
        
        for i in range(num_cameras):
            h, w = self.config['matching_height'], self.config['matching_width']
            mask = torch.ones(1, h, w, device=self.device)
            
            # Add some realistic occlusions (simulating blind spots)
            center_x, center_y = w // 2, h // 2
            
            # Random circular occlusions
            for _ in range(2 + i % 3):  # Different number per camera
                occ_x = torch.randint(w // 4, 3 * w // 4, (1,)).item()
                occ_y = torch.randint(h // 4, 3 * h // 4, (1,)).item()
                radius = torch.randint(5, 15, (1,)).item()
                
                y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
                dist_from_center = torch.sqrt((x_coords - occ_x)**2 + (y_coords - occ_y)**2)
                occlusion_mask = dist_from_center < radius
                mask[0, occlusion_mask] = 0
            
            masks.append(mask)
        
        print(f"Generated {len(masks)} realistic masks with occlusions")
        return masks
    
    def test_end_to_end_equivalence(self):
        """Test end-to-end equivalence with realistic data"""
        print("\n" + "="*60)
        print("End-to-End Equivalence Test")
        print("="*60)
        
        try:
            # Load test data
            calibrations, images, masks = self.load_test_dataset()
            
            # Initialize estimators
            py_estimator = PythonRGBD_Estimator(
                min_dist=self.config['min_dist'],
                max_dist=self.config['max_dist'],
                candidate_count=self.config['candidate_count'],
                matching_resolution=(self.config['matching_width'], self.config['matching_height']),
                device=self.device
            )
            py_estimator.references_indices = self.config['references_indices']
            
            # Prepare calibrations for C++
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
            
            # Convert images for C++
            images_uint8 = []
            for img in images:
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_rgba = np.zeros((img_np.shape[0], img_np.shape[1], 4), dtype=np.uint8)
                img_rgba[..., :3] = img_np
                img_rgba[..., 3] = 255
                images_uint8.append(img_rgba)
            
            cpp_estimator = my_stereo_pkg.RGBD_Estimator(
                self.config['min_dist'], self.config['max_dist'], self.config['candidate_count'],
                self.config['references_indices'], self.config['matching_width'], self.config['matching_height'],
                self.config['pano_width'], self.config['pano_height'],
                self.config['sigma_i'], self.config['sigma_s'], 0
            )
            
            print("Running Python implementation...")
            py_start = time.time()
            py_rgb, py_dist = py_estimator.estimate_RGBD_panorama(calibrations, images, masks)
            py_time = time.time() - py_start
            
            print("Running C++/CUDA implementation...")
            cpp_start = time.time()
            cpp_result = cpp_estimator.estimate_RGBD_panorama(calibrations_flat, images_uint8)
            cpp_time = time.time() - cpp_start
            
            # Extract results
            py_rgb_np = py_rgb.cpu().numpy()
            py_dist_np = py_dist.cpu().numpy()
            
            cpp_rgb_np = np.frombuffer(cpp_result['rgb'], dtype=np.uint8).reshape(
                (self.config['pano_height'], self.config['pano_width'], 3))
            cpp_dist_np = np.frombuffer(cpp_result['distance'], dtype=np.float32).reshape(
                (self.config['pano_height'], self.config['pano_width']))
            
            # Detailed comparison
            rgb_mae = np.mean(np.abs(py_rgb_np.astype(float) - cpp_rgb_np.astype(float)))
            dist_mae = np.mean(np.abs(py_dist_np - cpp_dist_np))
            
            # Valid pixel analysis
            py_valid = py_dist_np > 0
            cpp_valid = cpp_dist_np > 0
            both_valid = py_valid & cpp_valid
            
            valid_ratio = both_valid.mean()
            valid_dist_mae = np.mean(np.abs(py_dist_np[both_valid] - cpp_dist_np[both_valid])) if both_valid.any() else float('inf')
            
            print(f"Performance:")
            print(f"  Python time: {py_time:.2f}s")
            print(f"  C++/CUDA time: {cpp_time:.2f}s")
            print(f"  Speedup: {py_time/cpp_time:.2f}x")
            
            print(f"Quality metrics:")
            print(f"  RGB MAE: {rgb_mae:.2f}")
            print(f"  Distance MAE (all): {dist_mae:.3f}m")
            print(f"  Distance MAE (valid): {valid_dist_mae:.3f}m")
            print(f"  Valid pixel ratio: {valid_ratio*100:.1f}%")
            
            # Success criteria
            success = (
                rgb_mae < 100 and
                valid_dist_mae < 5.0 and
                valid_ratio > 0.3
            )
            
            self.results['end_to_end'] = {
                'success': success,
                'rgb_mae': rgb_mae,
                'dist_mae': dist_mae,
                'valid_dist_mae': valid_dist_mae,
                'valid_ratio': valid_ratio,
                'speedup': py_time/cpp_time
            }
            
            # Visualization
            self._visualize_comparison(py_rgb_np, py_dist_np, cpp_rgb_np, cpp_dist_np)
            
            if success:
                print("✓ End-to-end equivalence test PASSED")
            else:
                print("❌ End-to-end equivalence test FAILED")
            
            return success
            
        except Exception as e:
            print(f"❌ End-to-end test failed: {e}")
            traceback.print_exc()
            return False
    
    def _visualize_comparison(self, py_rgb, py_dist, cpp_rgb, cpp_dist):
        """Visualize comparison results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Python results
        axes[0, 0].imshow(py_rgb.astype(np.uint8))
        axes[0, 0].set_title('Python RGB')
        axes[0, 0].axis('off')
        
        py_dist_vis = py_dist.copy()
        py_dist_vis[py_dist <= 0] = np.nan
        im1 = axes[0, 1].imshow(py_dist_vis, cmap='jet', vmin=1, vmax=10)
        axes[0, 1].set_title('Python Distance')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # C++ results
        axes[1, 0].imshow(cpp_rgb)
        axes[1, 0].set_title('C++/CUDA RGB')
        axes[1, 0].axis('off')
        
        cpp_dist_vis = cpp_dist.copy()
        cpp_dist_vis[cpp_dist <= 0] = np.nan
        im2 = axes[1, 1].imshow(cpp_dist_vis, cmap='jet', vmin=1, vmax=10)
        axes[1, 1].set_title('C++/CUDA Distance')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1])
        
        # Differences
        rgb_diff = np.abs(py_rgb.astype(float) - cpp_rgb.astype(float))
        axes[0, 2].imshow(rgb_diff.astype(np.uint8))
        axes[0, 2].set_title('RGB Difference')
        axes[0, 2].axis('off')
        
        valid_mask = (py_dist > 0) & (cpp_dist > 0)
        dist_diff = np.abs(py_dist - cpp_dist)
        dist_diff[~valid_mask] = np.nan
        im3 = axes[1, 2].imshow(dist_diff, cmap='hot', vmin=0, vmax=2)
        axes[1, 2].set_title('Distance Difference')
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2])
        
        plt.tight_layout()
        output_path = f"{original_cwd}/integration_test_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison visualization saved to: {output_path}")
    
    def test_performance_scaling(self):
        """Test performance with different input sizes"""
        print("\n" + "="*60)
        print("Performance Scaling Test")
        print("="*60)
        
        test_sizes = [
            (160, 120),   # Small
            (320, 240),   # Medium
            (640, 480),   # Large
        ]
        
        performance_results = []
        
        for width, height in test_sizes:
            print(f"\nTesting with resolution {width}x{height}")
            
            # Update config for this test
            old_width, old_height = self.config['matching_width'], self.config['matching_height']
            self.config['matching_width'], self.config['matching_height'] = width, height
            self.config['pano_width'], self.config['pano_height'] = width * 2, height
            
            try:
                # Generate test data for this resolution
                calibrations, images, masks = self.load_test_dataset()
                
                # Quick test with Python only (C++ setup is more complex for different sizes)
                py_estimator = PythonRGBD_Estimator(
                    min_dist=self.config['min_dist'],
                    max_dist=self.config['max_dist'],
                    candidate_count=32,  # Reduce for speed
                    matching_resolution=(width, height),
                    device=self.device
                )
                py_estimator.references_indices = [0]  # Single reference for speed
                
                # Time the operation
                start_time = time.time()
                py_rgb, py_dist = py_estimator.estimate_RGBD_panorama(calibrations, images, masks)
                elapsed_time = time.time() - start_time
                
                # Calculate metrics
                pixels = width * height
                pixels_per_second = pixels / elapsed_time
                
                print(f"  Time: {elapsed_time:.2f}s")
                print(f"  Pixels/second: {pixels_per_second:.0f}")
                
                performance_results.append({
                    'resolution': f"{width}x{height}",
                    'pixels': pixels,
                    'time': elapsed_time,
                    'pixels_per_second': pixels_per_second
                })
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                performance_results.append({
                    'resolution': f"{width}x{height}",
                    'pixels': width * height,
                    'time': float('inf'),
                    'pixels_per_second': 0
                })
            
            # Restore original config
            self.config['matching_width'], self.config['matching_height'] = old_width, old_height
            self.config['pano_width'], self.config['pano_height'] = old_width * 2, old_height
        
        # Analyze scaling
        print(f"\nPerformance Scaling Analysis:")
        for result in performance_results:
            print(f"  {result['resolution']}: {result['time']:.2f}s ({result['pixels_per_second']:.0f} pixels/s)")
        
        # Expected scaling should be roughly linear with pixel count
        scaling_efficient = True
        if len(performance_results) >= 2:
            for i in range(1, len(performance_results)):
                expected_ratio = performance_results[i]['pixels'] / performance_results[0]['pixels']
                actual_ratio = performance_results[i]['time'] / performance_results[0]['time']
                efficiency = expected_ratio / actual_ratio
                
                print(f"  Scaling efficiency {performance_results[0]['resolution']} -> {performance_results[i]['resolution']}: {efficiency:.2f}")
                
                if efficiency < 0.5:  # More than 2x slower than expected
                    scaling_efficient = False
        
        self.results['performance_scaling'] = {
            'results': performance_results,
            'efficient': scaling_efficient
        }
        
        return scaling_efficient
    
    def run_all_integration_tests(self):
        """Run all integration tests"""
        print("="*80)
        print("Integration Test Suite")
        print(f"Device: {self.device}")
        print("="*80)
        print(f"\nThis suite consolidates:")
        print(f"  • equivalence_test.py - End-to-end equivalence verification")
        print(f"  • compare_detailed.py - Detailed comparison and visualization")
        print(f"\nTotal Tests: 2")
        print()
        
        tests = [
            ("End-to-End Equivalence", self.test_end_to_end_equivalence),
            ("Performance Scaling", self.test_performance_scaling)
        ]
        
        passed = 0
        start_time = time.time()
        
        for name, test_func in tests:
            print(f"\nRunning {name} test...")
            try:
                if test_func():
                    passed += 1
                    print(f"✓ {name} test PASSED")
                else:
                    print(f"❌ {name} test FAILED")
            except Exception as e:
                print(f"❌ {name} test crashed: {e}")
                traceback.print_exc()
        
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("Integration Test Results Summary")
        print("="*80)
        print(f"Total Tests: {len(tests)}")
        print(f"Passed: {passed}")
        print(f"Failed: {len(tests) - passed}")
        print(f"Success Rate: {100.0 * passed / len(tests):.1f}%")
        print(f"Total Time: {total_time:.2f}s")
        
        return passed == len(tests)


def main():
    """Main integration test function"""
    parser = argparse.ArgumentParser(description="Integration Test Suite")
    parser.add_argument("--device", default="cuda:0", help="Device to use (default: cuda:0)")
    
    args = parser.parse_args()
    
    try:
        device = torch.device(args.device)
        suite = IntegrationTestSuite(device=device)
        
        success = suite.run_all_integration_tests()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Integration test suite failed: {e}")
        traceback.print_exc()
        return 1
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)