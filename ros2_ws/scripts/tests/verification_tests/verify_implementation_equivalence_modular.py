#!/usr/bin/env python3
"""
verify_implementation_equivalence.py

Comprehensive Equivalence Verification Suite (Modular Version)
===========================================================================
Orchestrates verification tests from individual module files:
- modules/verify_calibration.py: Calibration data verification
- modules/verify_geometry.py: Double sphere geometry functions
- modules/verify_stitcher.py: Stitcher component verification
- modules/verify_isb.py: ISB filter verification
- modules/verify_rgbd.py: RGBD estimator verification

Original consolidated files:
- verify_equivalence.py, verify_equivalence_minimal.py
- verify_isb_filter.py, verify_stitcher.py, verify_utils.py
===========================================================================
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
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

# Import implementations
try:
    from utils import Calibration as DoubleSphereCalibration
    print("✓ Python implementations loaded")
    has_python_impl = True
except ImportError as e:
    print(f"✗ Failed to import Python implementations: {e}")
    has_python_impl = False

try:
    import my_stereo_pkg
    print("✓ C++/CUDA implementations loaded")
    has_cuda_impl = True
except ImportError as e:
    print(f"✗ Failed to import C++/CUDA implementations: {e}")
    has_cuda_impl = False

# Add module path
module_dir = Path(__file__).parent / "modules"
sys.path.insert(0, str(module_dir))

# Import verification modules
from verify_calibration import verify_calibration_loading
from verify_geometry import verify_geometry_functions
from verify_stitcher import verify_stitcher
from verify_isb import verify_isb_filter
from verify_rgbd import verify_rgbd_estimator


class ImplementationVerifier:
    """Orchestrates all verification tests"""
    
    def __init__(self, device='cuda:0', minimal_mode=False):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.minimal_mode = minimal_mode
        self.results = {}
        
        # Configuration
        if minimal_mode:
            self.config = {
                'min_dist': 3.0,
                'max_dist': 10.0,
                'candidate_count': 32,
                'references_indices': [0, 1],
                'matching_width': 160,
                'matching_height': 80,
                'pano_width': 640,
                'pano_height': 320,
                'sigma_i': 30.0,
                'sigma_s': 30.0,
                'num_cameras': 2
            }
        else:
            self.config = {
                'min_dist': 3.0,
                'max_dist': 10.0,
                'candidate_count': 128,
                'references_indices': [0, 1, 2, 3],
                'matching_width': 320,
                'matching_height': 160,
                'pano_width': 1280,
                'pano_height': 640,
                'sigma_i': 30.0,
                'sigma_s': 30.0,
                'num_cameras': 4
            }
    
    def generate_synthetic_calibration(self):
        """Generate synthetic calibration for testing"""
        calibrations = []
        for i in range(self.config['num_cameras']):
            angle = i * (2 * np.pi / self.config['num_cameras'])
            R = torch.tensor([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ], dtype=torch.float32, device=self.device)
            
            t = torch.zeros(3, 1, device=self.device)
            rt = torch.cat([R, t], dim=1)
            rt = torch.cat([rt, torch.tensor([[0, 0, 0, 1]], device=self.device)], dim=0)
            
            calib = DoubleSphereCalibration(
                original_resolution=(self.config['matching_width'], self.config['matching_height']),
                principal=torch.tensor([self.config['matching_width']/2, self.config['matching_height']/2], device=self.device),
                fl=torch.tensor([250.0, 250.0], device=self.device),
                xi=-0.2,
                alpha=0.6,
                rt=rt,
                matching_scale=torch.tensor([1.0, 1.0], device=self.device)
            )
            calibrations.append(calib)
        
        return calibrations
    
    def generate_masks(self):
        """Generate synthetic masks for testing"""
        masks = []
        for _ in range(self.config['num_cameras']):
            mask = torch.ones(
                (self.config['matching_height'], self.config['matching_width']),
                dtype=torch.bool,
                device=self.device
            )
            masks.append(mask)
        return masks
    
    def run_all_verifications(self):
        """Run all verification tests"""
        print("="*80)
        print("Comprehensive Equivalence Verification Suite (Modular)")
        print(f"Mode: {'Minimal' if self.minimal_mode else 'Full'}")
        print(f"Device: {self.device}")
        print("="*80)
        print(f"\nThis suite runs 5 verification modules:")
        print(f"  • verify_calibration.py - Calibration structure")
        print(f"  • verify_geometry.py - Geometry functions")
        print(f"  • verify_stitcher.py - Stitcher component")
        print(f"  • verify_isb.py - ISB filter")
        print(f"  • verify_rgbd.py - RGBD estimator")
        print()
        
        all_results = {}
        passed_count = 0
        failed_count = 0
        
        # Run each verification module
        print("[1/5] Running calibration verification...")
        success, results = verify_calibration_loading(
            self.config, self.device, self.generate_synthetic_calibration
        )
        all_results['calibration'] = results
        if success:
            passed_count += 1
        else:
            failed_count += 1
        
        print("\n[2/5] Running geometry verification...")
        success, results = verify_geometry_functions(self.device)
        all_results['geometry'] = results
        if success:
            passed_count += 1
        else:
            failed_count += 1
        
        print("\n[3/5] Running stitcher verification...")
        success, results = verify_stitcher(
            self.config, self.device,
            self.generate_synthetic_calibration,
            self.generate_masks
        )
        all_results['stitcher'] = results
        if success:
            passed_count += 1
        else:
            failed_count += 1
        
        print("\n[4/5] Running ISB filter verification...")
        success, results = verify_isb_filter(
            self.config, self.device, has_cuda_impl
        )
        all_results['isb_filter'] = results
        if success:
            passed_count += 1
        else:
            failed_count += 1
        
        print("\n[5/5] Running RGBD estimator verification...")
        success, results = verify_rgbd_estimator(
            self.config, self.device, has_cuda_impl
        )
        all_results['rgbd_estimator'] = results
        if success:
            passed_count += 1
        else:
            failed_count += 1
        
        # Print summary
        print("\n" + "="*80)
        print("Verification Summary")
        print("="*80)
        print(f"Total tests: 5")
        print(f"Passed: {passed_count}")
        print(f"Failed: {failed_count}")
        print(f"Success rate: {passed_count/5*100:.1f}%")
        
        return all_results, failed_count == 0


def main():
    parser = argparse.ArgumentParser(description='Implementation Equivalence Verification Suite')
    parser.add_argument('--device', default='cuda:0', help='Device to use (default: cuda:0)')
    parser.add_argument('--minimal', action='store_true', help='Use minimal mode (smaller resolution)')
    
    args = parser.parse_args()
    
    verifier = ImplementationVerifier(device=args.device, minimal_mode=args.minimal)
    results, success = verifier.run_all_verifications()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
