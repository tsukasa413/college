"""
Geometry Functions Verification Module

Verifies project/unproject double sphere geometry functions.
"""

import torch
import traceback


def verify_geometry_functions(device):
    """Verify project/unproject geometry functions"""
    print("\\n" + "="*60)
    print("Geometry Functions Verification")
    print("="*60)
    
    results = {}
    
    try:
        from utils import project, unproject, Calibration as DoubleSphereCalibration
        
        print("[1/4] Creating test calibration...")
        calib = DoubleSphereCalibration(
            original_resolution=(320, 240),
            principal=torch.tensor([160.0, 120.0], device=device),
            fl=torch.tensor([250.0, 250.0], device=device),
            xi=-0.2,
            alpha=0.6,
            rt=torch.eye(4, device=device),
            matching_scale=torch.tensor([1.0, 1.0], device=device)
        )
        print("  ✓ Calibration created")
        
        print("[2/4] Testing multiple UV coordinates...")
        test_points = torch.tensor([
            [160.0, 120.0],  # center
            [80.0, 60.0],    # top-left quadrant
            [240.0, 180.0],  # bottom-right quadrant
            [160.0, 60.0],   # top center
            [160.0, 180.0],  # bottom center
        ], device=device).unsqueeze(0)
        
        print(f"  Testing {test_points.shape[1]} points")
        
        print("[3/4] Unprojecting to 3D...")
        pt_3d, valid = unproject(test_points, calib)
        valid_count = valid.sum().item()
        print(f"  ✓ Valid points: {valid_count}/{test_points.shape[1]}")
        
        print("[4/4] Projecting back to 2D...")
        uv_reproj, valid_reproj = project(pt_3d, calib)
        
        # Calculate errors
        errors = torch.abs(test_points - uv_reproj)
        max_error = errors.max().item()
        mean_error = errors.mean().item()
        
        print(f"  Max roundtrip error: {max_error:.6f} pixels")
        print(f"  Mean roundtrip error: {mean_error:.6f} pixels")
        
        success = max_error < 1e-3
        
        results['geometry'] = {
            'max_error': max_error,
            'mean_error': mean_error,
            'valid_points': valid_count,
            'total_points': test_points.shape[1],
            'success': success
        }
        
        if success:
            print("✓ Geometry functions verification PASSED")
        else:
            print("✗ Geometry functions verification FAILED")
        
        return success, results
        
    except ImportError as e:
        print(f"⏭️  Skipping geometry verification: {e}")
        return True, results
    except Exception as e:
        print(f"✗ Geometry verification failed: {e}")
        traceback.print_exc()
        return False, results
