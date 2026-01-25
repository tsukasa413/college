#!/usr/bin/env python3
"""
analyze_distance_parameterization.py
Focus on distance parameterization differences as the main source of systematic bias.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

def analyze_distance_parameterization():
    """
    Analyze differences in distance parameterization between implementations.
    """
    print("=" * 80)
    print("DISTANCE PARAMETERIZATION ANALYSIS")
    print("=" * 80)
    
    min_dist, max_dist, candidate_count = 1.0, 10.0, 64
    
    # Python method
    py_candidates = 1 / torch.linspace(1 / min_dist, 1 / max_dist, candidate_count)
    
    # C++ method  
    def cpp_distance_candidates(min_d, max_d, count):
        inv_dist_min = 1.0 / min_d
        inv_dist_max = 1.0 / max_d
        candidates = []
        for i in range(count):
            inv_dist = inv_dist_min - (inv_dist_min - inv_dist_max) * (i / (count - 1))
            candidates.append(1.0 / inv_dist)
        return np.array(candidates)
    
    cpp_candidates = cpp_distance_candidates(min_dist, max_dist, candidate_count)
    
    print(f"Distance range: {min_dist}m to {max_dist}m, {candidate_count} candidates")
    print(f"\nPython method:")
    print(f"  First 5: {py_candidates[:5].numpy()}")
    print(f"  Last 5: {py_candidates[-5:].numpy()}")
    
    print(f"\nC++ method:")  
    print(f"  First 5: {cpp_candidates[:5]}")
    print(f"  Last 5: {cpp_candidates[-5:]}")
    
    # Compare differences
    py_np = py_candidates.cpu().numpy()
    diff = np.abs(py_np - cpp_candidates)
    
    print(f"\nDifferences:")
    print(f"  Max difference: {diff.max():.6f}m")
    print(f"  Mean difference: {diff.mean():.6f}m")
    
    # Check ordering
    py_increasing = np.all(np.diff(py_np) >= 0)
    cpp_increasing = np.all(np.diff(cpp_candidates) >= 0)
    
    print(f"\nOrdering:")
    print(f"  Python increasing: {py_increasing}")
    print(f"  C++ increasing: {cpp_increasing}")
    
    if py_increasing != cpp_increasing:
        print("  ⚠️ CRITICAL: Distance ordering is opposite!")
        print("     This explains the systematic bias!")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    indices = np.arange(candidate_count)
    axes[0].plot(indices, py_np, 'b-o', markersize=3, label='Python')
    axes[0].plot(indices, cpp_candidates, 'r-x', markersize=3, label='C++')
    axes[0].set_title('Distance Candidates Comparison')
    axes[0].set_xlabel('Candidate Index')
    axes[0].set_ylabel('Distance (m)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(indices, diff, 'g-o', markersize=3)
    axes[1].set_title('Absolute Difference')
    axes[1].set_xlabel('Candidate Index') 
    axes[1].set_ylabel('|Python - C++| (m)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = '/home/motoken/college/ros2_ws/scripts/distance_parameterization_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Analysis visualization saved to: {save_path}")
    
    return diff.mean(), diff.max()

def main():
    print("Distance Parameterization Analysis")
    print("=" * 50)
    
    param_mean_diff, param_max_diff = analyze_distance_parameterization()
    
    print(f"\nKEY FINDINGS:")
    print(f"- Mean difference: {param_mean_diff:.6f}m")
    print(f"- Max difference: {param_max_diff:.6f}m")
    
    if param_mean_diff > 1.0:
        print(f"\n⚠️ Significant distance parameterization differences found!")
        print(f"This contributes to systematic bias in MAE.")
    else:
        print(f"\n✓ Distance parameterization differences are manageable.")
    
    return True

if __name__ == "__main__":
    main()