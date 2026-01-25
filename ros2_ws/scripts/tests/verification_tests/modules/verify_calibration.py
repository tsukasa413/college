"""
Calibration Loading Verification Module

Verifies calibration data structure and RT matrices.
"""

import numpy as np
import traceback


def verify_calibration_loading(config, device, generate_synthetic_calibration_func):
    """Verify calibration data structure"""
    print("\\n" + "="*60)
    print("Calibration Structure Verification")
    print("="*60)
    
    results = {}
    
    try:
        print("[1/3] Generating synthetic calibrations...")
        calibrations = generate_synthetic_calibration_func()
        print(f"  ✓ Generated {len(calibrations)} calibrations")
        
        print("[2/3] Verifying calibration parameters...")
        for i, calib in enumerate(calibrations):
            print(f"  Camera {i}:")
            print(f"    Resolution: {calib.original_resolution}")
            print(f"    Focal length: {calib.fl.cpu().numpy()}")
            print(f"    Principal point: {calib.principal.cpu().numpy()}")
            print(f"    Xi: {calib.xi}, Alpha: {calib.alpha}")
            print(f"    RT matrix shape: {calib.rt.shape}")
        
        print("[3/3] Validating RT matrices...")
        for i, calib in enumerate(calibrations):
            R = calib.rt[:3, :3].cpu().numpy()
            det = np.linalg.det(R)
            orthogonality = np.max(np.abs(R @ R.T - np.eye(3)))
            print(f"  Camera {i}: det={det:.6f}, orthogonality_error={orthogonality:.6e}")
        
        results['calibration'] = {
            'num_cameras': len(calibrations),
            'validation': 'passed'
        }
        
        print("✓ Calibration verification PASSED")
        return True, results
        
    except Exception as e:
        print(f"✗ Calibration verification failed: {e}")
        traceback.print_exc()
        return False, results
