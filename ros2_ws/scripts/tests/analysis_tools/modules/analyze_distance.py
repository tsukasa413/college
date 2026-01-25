"""
Distance Parameterization Analysis Module

Analyzes the inverse distance parameterization and its derivative.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_distance_parameterization(config, device, output_dir=None):
    """Analyze distance parameterization details"""
    print("\\n" + "="*60)
    print("Distance Parameterization Analysis")
    print("="*60)
    
    try:
        print("[1/5] Computing inverse distance range...")
        min_inv_dist = 1.0 / config['max_dist']
        max_inv_dist = 1.0 / config['min_dist']
        print(f"  Min distance: {config['min_dist']:.2f}m -> inv: {max_inv_dist:.6f}")
        print(f"  Max distance: {config['max_dist']:.2f}m -> inv: {min_inv_dist:.6f}")
        
        print(f"[2/5] Generating {config['candidate_count']} candidates...")
        candidates_idx = torch.arange(config['candidate_count'], device=device)
        t = candidates_idx / (config['candidate_count'] - 1)
        inv_dist = min_inv_dist + t * (max_inv_dist - min_inv_dist)
        distances = 1.0 / inv_dist
        
        print(f"  First candidate: {distances[0].item():.4f}m")
        print(f"  Last candidate: {distances[-1].item():.4f}m")
        print(f"  Median candidate: {distances[config['candidate_count']//2].item():.4f}m")
        
        print("[3/5] Analyzing distance distribution...")
        dist_diffs = distances[1:] - distances[:-1]
        print(f"  Min step: {dist_diffs.min().item():.6f}m")
        print(f"  Max step: {dist_diffs.max().item():.6f}m")
        print(f"  Mean step: {dist_diffs.mean().item():.6f}m")
        
        print("[4/5] Computing derivatives...")
        d_inv_dist = (max_inv_dist - min_inv_dist) / (config['candidate_count'] - 1)
        d_dist_d_inv_dist = -1.0 / (inv_dist ** 2)
        d_dist_d_idx = d_dist_d_inv_dist * d_inv_dist
        
        print(f"  d(1/d)/didx: {d_inv_dist:.8f}")
        print(f"  d(dist)/d(1/d) range: [{d_dist_d_inv_dist.min().item():.4f}, {d_dist_d_inv_dist.max().item():.4f}]")
        
        print("[5/5] Creating visualization...")
        if output_dir:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Distance vs index
            axes[0,0].plot(candidates_idx.cpu().numpy(), distances.cpu().numpy())
            axes[0,0].set_xlabel('Candidate Index')
            axes[0,0].set_ylabel('Distance (m)')
            axes[0,0].set_title('Distance Parameterization')
            axes[0,0].grid(True)
            
            # Inverse distance vs index
            axes[0,1].plot(candidates_idx.cpu().numpy(), inv_dist.cpu().numpy())
            axes[0,1].set_xlabel('Candidate Index')
            axes[0,1].set_ylabel('Inverse Distance (1/m)')
            axes[0,1].set_title('Inverse Distance (Linear)')
            axes[0,1].grid(True)
            
            # Step size
            axes[1,0].plot(candidates_idx[:-1].cpu().numpy(), dist_diffs.cpu().numpy())
            axes[1,0].set_xlabel('Candidate Index')
            axes[1,0].set_ylabel('Step Size (m)')
            axes[1,0].set_title('Distance Step Size')
            axes[1,0].grid(True)
            
            # Derivative
            axes[1,1].plot(candidates_idx.cpu().numpy(), d_dist_d_idx.cpu().numpy())
            axes[1,1].set_xlabel('Candidate Index')
            axes[1,1].set_ylabel('d(dist)/d(idx)')
            axes[1,1].set_title('Distance Derivative')
            axes[1,1].grid(True)
            
            plt.tight_layout()
            output_path = Path(output_dir) / 'distance_parameterization.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved plot to {output_path}")
        
        print("✓ Distance parameterization analysis complete")
        
        return {
            'min_inv_dist': min_inv_dist,
            'max_inv_dist': max_inv_dist,
            'min_step': dist_diffs.min().item(),
            'max_step': dist_diffs.max().item(),
            'mean_step': dist_diffs.mean().item()
        }
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {}
