"""
Cost Computation Analysis Module

Analyzes cost volume computation and aggregation.
"""

import torch
import numpy as np


def analyze_cost_computation(config, device):
    """Analyze cost computation details"""
    print("\\n" + "="*60)
    print("Cost Computation Analysis")
    print("="*60)
    
    try:
        print("[1/4] Creating synthetic cost volume...")
        batch_size = 1
        cost_volume = torch.randn(
            batch_size,
            config['candidate_count'],
            config['matching_height'],
            config['matching_width'],
            device=device
        )
        print(f"  ✓ Cost volume shape: {tuple(cost_volume.shape)}")
        print(f"  ✓ Cost range: [{cost_volume.min().item():.4f}, {cost_volume.max().item():.4f}]")
        
        print("[2/4] Analyzing cost statistics per candidate...")
        cost_mean = cost_volume.mean(dim=(2, 3))
        cost_std = cost_volume.std(dim=(2, 3))
        print(f"  Mean cost range: [{cost_mean.min().item():.4f}, {cost_mean.max().item():.4f}]")
        print(f"  Std dev range: [{cost_std.min().item():.4f}, {cost_std.max().item():.4f}]")
        
        print("[3/4] Computing winner-take-all depth...")
        min_cost_indices = cost_volume.argmin(dim=1)
        unique_winners = torch.unique(min_cost_indices)
        print(f"  ✓ Depth map shape: {tuple(min_cost_indices.shape)}")
        print(f"  ✓ Unique depth candidates used: {len(unique_winners)}/{config['candidate_count']}")
        
        print("[4/4] Cost aggregation analysis...")
        # Simulate simple box filter aggregation
        kernel_size = 5
        padding = kernel_size // 2
        aggregated = torch.nn.functional.avg_pool2d(
            cost_volume,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        
        diff = torch.abs(cost_volume - aggregated)
        print(f"  Aggregation difference: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")
        
        print("✓ Cost computation analysis complete")
        
        return {
            'cost_shape': tuple(cost_volume.shape),
            'cost_range': (cost_volume.min().item(), cost_volume.max().item()),
            'unique_winners': len(unique_winners),
            'total_candidates': config['candidate_count']
        }
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {}
