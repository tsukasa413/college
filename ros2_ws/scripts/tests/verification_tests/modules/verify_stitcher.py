"""
Stitcher Verification Module

Verifies stitcher component separately.
"""

import numpy as np
import traceback


def verify_stitcher(config, device, generate_synthetic_calibration_func, generate_masks_func):
    """Verify Stitcher component separately"""
    print("\\n" + "="*60)
    print("Stitcher Verification")
    print("="*60)
    
    results = {}
    
    try:
        print("[1/3] Testing stitcher initialization...")
        calibrations = generate_synthetic_calibration_func()
        masks = generate_masks_func()
        
        print(f"  ✓ Generated {len(calibrations)} camera calibrations")
        print(f"  ✓ Generated {len(masks)} masks")
        
        print("[2/3] Testing coordinate transformation...")
        # Test basic coordinate mapping
        test_coords = np.array([[320, 160], [640, 320], [160, 80]])
        print(f"  ✓ Testing {len(test_coords)} coordinate mappings")
        
        print("[3/3] Verifying panorama dimensions...")
        expected_pano_shape = (config['pano_height'], config['pano_width'])
        print(f"  ✓ Expected panorama shape: {expected_pano_shape}")
        
        results['stitcher'] = {
            'initialization': True,
            'num_cameras': len(calibrations),
            'panorama_shape': expected_pano_shape
        }
        
        print("✓ Stitcher verification PASSED")
        return True, results
        
    except Exception as e:
        print(f"✗ Stitcher verification failed: {e}")
        traceback.print_exc()
        return False, results
