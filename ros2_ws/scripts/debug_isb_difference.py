#!/usr/bin/env python3
"""
Debug ISB Filter Differences
"""

import torch
import sys
import os
import numpy as np

# Change to sphere-stereo directory
original_dir = os.getcwd()
os.chdir('/home/motoken/college/sphere-stereo')

sys.path.insert(0, '/home/motoken/college/sphere-stereo/python')
from isb_filter import ISB_Filter as PyFilter

os.chdir(original_dir)

sys.path.insert(0, '/home/motoken/college/ros2_ws/install/my_stereo_pkg/lib/python3.10/site-packages')
from my_stereo_pkg import ISBFilter as CppFilter

def debug():
    device = torch.device("cuda:0")
    D, W, H = 32, 128, 128
    
    print("=" * 80)
    print("ISB Filter Detailed Difference Analysis")
    print("=" * 80)
    
    # Use fixed seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Create fixed input
    guide = torch.randint(0, 255, (H, W, 3), dtype=torch.uint8, device=device)
    cost = torch.randn((D, H, W), dtype=torch.float32, device=device)
    
    print(f"\nInput:")
    print(f"  Guide: {guide.shape}, dtype={guide.dtype}, device={guide.device}")
    print(f"  Cost: {cost.shape}, dtype={cost.dtype}, device={cost.device}")
    print(f"  Guide range: [{guide.min()}, {guide.max()}]")
    print(f"  Cost range: [{cost.min():.4f}, {cost.max():.4f}]")
    
    sigma_i, sigma_s = 10.0, 15.0
    print(f"\nParameters:")
    print(f"  sigma_i: {sigma_i}")
    print(f"  sigma_s: {sigma_s}")
    
    # Initialize filters
    os.chdir('/home/motoken/college/sphere-stereo')
    py_filter = PyFilter(D, (W, H), device)
    print(f"\nPython filter:")
    print(f"  Scale count: {py_filter.scale_count}")
    print(f"  Guide shapes: {[g.shape for g in py_filter.guides]}")
    print(f"  Cost shapes: {[c.shape for c in py_filter.costs]}")
    os.chdir(original_dir)
    
    cpp_filter = CppFilter(D, (W, H), device)
    print(f"\nC++ filter initialized")
    
    # Run filters
    print("\n" + "-" * 80)
    print("Running filters...")
    print("-" * 80)
    
    os.chdir('/home/motoken/college/sphere-stereo')
    py_result, py_guide = py_filter.apply(guide.clone(), cost.clone(), sigma_i, sigma_s)
    os.chdir(original_dir)
    
    cpp_result, cpp_guide = cpp_filter.apply(guide.clone(), cost.clone(), sigma_i, sigma_s)
    
    # Analyze results
    print("\nOutput:")
    print(f"  Python result: {py_result.shape}")
    print(f"    Range: [{py_result.min():.4f}, {py_result.max():.4f}]")
    print(f"    Mean: {py_result.mean():.4f}")
    print(f"    Std: {py_result.std():.4f}")
    
    print(f"\n  C++ result: {cpp_result.shape}")
    print(f"    Range: [{cpp_result.min():.4f}, {cpp_result.max():.4f}]")
    print(f"    Mean: {cpp_result.mean():.4f}")
    print(f"    Std: {cpp_result.std():.4f}")
    
    # Detailed difference analysis
    diff = py_result - cpp_result
    abs_diff = diff.abs()
    
    print("\n" + "=" * 80)
    print("DIFFERENCE ANALYSIS")
    print("=" * 80)
    
    print(f"\nAbsolute Difference:")
    print(f"  MAE: {abs_diff.mean():.6f}")
    print(f"  Max: {abs_diff.max():.6f}")
    print(f"  Min: {abs_diff.min():.6f}")
    print(f"  Std: {abs_diff.std():.6f}")
    
    # Find where largest differences occur
    max_diff_idx = abs_diff.argmax()
    max_diff_d = max_diff_idx // (H * W)
    max_diff_h = (max_diff_idx % (H * W)) // W
    max_diff_w = (max_diff_idx % (H * W)) % W
    
    print(f"\nMax difference location:")
    print(f"  Position: (d={max_diff_d}, h={max_diff_h}, w={max_diff_w})")
    print(f"  Python value: {py_result[max_diff_d, max_diff_h, max_diff_w]:.6f}")
    print(f"  C++ value: {cpp_result[max_diff_d, max_diff_h, max_diff_w]:.6f}")
    print(f"  Difference: {diff[max_diff_d, max_diff_h, max_diff_w]:.6f}")
    
    # Histogram of differences
    print(f"\nDifference distribution:")
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    percentile_values = torch.quantile(abs_diff.flatten(), torch.tensor(percentiles, device=device) / 100.0)
    for p, v in zip(percentiles, percentile_values):
        print(f"  {p:3d}th percentile: {v:.6f}")
    
    # Check if difference is uniform or concentrated
    high_error_mask = abs_diff > 0.1
    print(f"\nHigh error pixels (>0.1):")
    print(f"  Count: {high_error_mask.sum().item()} / {abs_diff.numel()} ({100*high_error_mask.sum().item()/abs_diff.numel():.2f}%)")
    
    # Analyze per-depth-layer
    print(f"\nPer-depth-layer analysis:")
    for d in range(min(5, D)):  # First 5 layers
        layer_mae = abs_diff[d].mean().item()
        layer_max = abs_diff[d].max().item()
        print(f"  Layer {d}: MAE={layer_mae:.6f}, Max={layer_max:.6f}")
    
    # Check guide differences
    guide_diff = (py_guide.float() - cpp_guide.float()).abs()
    print(f"\nGuide differences:")
    print(f"  MAE: {guide_diff.mean():.6f}")
    print(f"  Max: {guide_diff.max():.6f}")
    
    print("\n" + "=" * 80)
    if abs_diff.mean() > 1e-5:
        print("❌ SIGNIFICANT DIFFERENCES DETECTED")
        print("This indicates a potential implementation mismatch.")
    else:
        print("✓ Minimal differences - within numerical precision")
    print("=" * 80)

if __name__ == "__main__":
    debug()
