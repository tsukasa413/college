#!/usr/bin/env python3
"""
analyze_depth_estimation_suite.py

Comprehensive Depth Estimation Analysis Suite (Modular Version)
===========================================================================
Orchestrates analysis tools from individual module files:
- modules/analyze_distance.py: Distance parameterization analysis
- modules/analyze_cost.py: Cost computation analysis
- modules/debug_rt.py: RT matrix debugging
- modules/debug_isb.py: ISB filter debugging

Original consolidated files:
- analyze_distance_parameterization.py
- analyze_cost_computation.py
- debug_rt_matrix.py
- debug_isb_difference.py
===========================================================================
"""

import sys
import os
import torch
from pathlib import Path
import argparse

# Setup library paths
torch_lib_path = os.path.join(torch.utils.cmake_prefix_path, "..", "lib")
if os.path.exists(torch_lib_path):
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if torch_lib_path not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}"

# Change to sphere-stereo directory
sphere_stereo_path = "/home/motoken/college/sphere-stereo"
if os.path.exists(sphere_stereo_path):
    os.chdir(sphere_stereo_path)
    sys.path.insert(0, os.path.join(sphere_stereo_path, "python"))

# Add module path
module_dir = Path(__file__).parent / "modules"
sys.path.insert(0, str(module_dir))

# Import analysis modules
from analyze_distance import analyze_distance_parameterization
from analyze_cost import analyze_cost_computation
from debug_rt import debug_rt_matrix
from debug_isb import debug_isb_filter


class DepthEstimationAnalyzer:
    """Orchestrates all depth estimation analysis tools"""
    
    def __init__(self, device='cuda:0', output_dir=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        
        # Standard configuration
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
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def generate_synthetic_calibration(self):
        """Generate synthetic calibration for RT matrix testing"""
        import numpy as np
        from utils import Calibration as DoubleSphereCalibration
        
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
    
    def run_comprehensive_analysis(self):
        """Run all analysis components"""
        print("="*80)
        print("Comprehensive Depth Estimation Analysis Suite (Modular)")
        print(f"Device: {self.device}")
        print("="*80)
        print(f"\nThis suite runs 4 analysis modules:")
        print(f"  • analyze_distance.py - Distance parameterization")
        print(f"  • analyze_cost.py - Cost computation")
        print(f"  • debug_rt.py - RT matrix debugging")
        print(f"  • debug_isb.py - ISB filter debugging")
        print()
        
        all_results = {}
        
        # Run each analysis module
        print("[1/4] Distance Parameterization Analysis")
        results = analyze_distance_parameterization(
            self.config, self.device, self.output_dir
        )
        all_results['distance_param'] = results
        
        print("\n[2/4] Cost Computation Analysis")
        results = analyze_cost_computation(self.config, self.device)
        all_results['cost_computation'] = results
        
        print("\n[3/4] RT Matrix Debug")
        results = debug_rt_matrix(
            self.config, self.device, self.generate_synthetic_calibration
        )
        all_results['rt_matrix'] = results
        
        print("\n[4/4] ISB Filter Debug")
        results = debug_isb_filter(self.config, self.device)
        all_results['isb_filter'] = results
        
        # Print summary
        print("\n" + "="*80)
        print("Analysis Summary")
        print("="*80)
        print(f"Completed 4 analysis modules")
        if self.output_dir:
            print(f"Results saved to: {self.output_dir}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description='Depth Estimation Analysis Suite')
    parser.add_argument('--device', default='cuda:0', help='Device to use (default: cuda:0)')
    parser.add_argument('--output-dir', help='Output directory for plots and results')
    
    args = parser.parse_args()
    
    analyzer = DepthEstimationAnalyzer(
        device=args.device,
        output_dir=args.output_dir
    )
    results = analyzer.run_comprehensive_analysis()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
