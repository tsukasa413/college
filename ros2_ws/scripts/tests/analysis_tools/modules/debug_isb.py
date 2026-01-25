"""
ISB Filter Debug Module

Debug ISB (Iterative Spatial Bilateral) filter behavior.
"""

import torch


def debug_isb_filter(config, device):
    """Debug ISB filter implementation"""
    print("\\n" + "="*60)
    print("ISB Filter Debug Analysis")
    print("="*60)
    
    try:
        print("[1/4] Testing ISB filter initialization...")
        try:
            from isb_filter import ISB_Filter
            isb_filter = ISB_Filter(
                config['candidate_count'],
                (config['matching_width'], config['matching_height']),
                device
            )
            print("  ✓ ISB filter initialized")
        except Exception as e:
            print(f"  ✗ ISB filter initialization failed: {e}")
            return {}
        
        print("[2/4] Creating test cost volume...")
        batch_size = 1
        cost_volume = torch.randn(
            batch_size,
            config['candidate_count'],
            config['matching_height'],
            config['matching_width'],
            device=device
        )
        print(f"  ✓ Input shape: {tuple(cost_volume.shape)}")
        
        print("[3/4] Running filter...")
        # Create dummy guide image (YUV format)
        guide = torch.randint(0, 256, (config['matching_height'], config['matching_width'], 3), 
                             dtype=torch.uint8, device=device)
        # apply() returns (filtered_cost, filtered_guide)
        filtered_cost, _ = isb_filter.apply(guide, cost_volume[0], config['sigma_i'], config['sigma_s'])
        print(f"  ✓ Output shape: {tuple(filtered_cost.shape)}")
        
        print("[4/4] Comparing before/after filtering...")
        # Both are [candidates, height, width]
        diff = torch.abs(cost_volume[0] - filtered_cost)
        print(f"  Max difference: {diff.max().item():.6f}")
        print(f"  Mean difference: {diff.mean().item():.6f}")
        print(f"  Median difference: {diff.median().item():.6f}")
        
        # Analyze smoothness - compute horizontal gradients
        cost_grad = torch.abs(cost_volume[0, :, :, 1:] - cost_volume[0, :, :, :-1])
        filtered_grad = torch.abs(filtered_cost[:, :, 1:] - filtered_cost[:, :, :-1])
        
        print(f"  Before filtering - mean gradient: {cost_grad.mean().item():.6f}")
        print(f"  After filtering - mean gradient: {filtered_grad.mean().item():.6f}")
        print(f"  Smoothness improvement: {(1 - filtered_grad.mean()/cost_grad.mean()).item()*100:.2f}%")
        
        print("\\n✓ ISB filter debug complete")
        
        return {
            'max_diff': diff.max().item(),
            'mean_diff': diff.mean().item(),
            'smoothness_improvement': (1 - filtered_grad.mean()/cost_grad.mean()).item()
        }
        
    except Exception as e:
        print(f"✗ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return {}
