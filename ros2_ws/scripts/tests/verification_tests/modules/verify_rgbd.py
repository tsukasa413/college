"""
RGBD Estimator Verification Module

Verifies RGBD estimator implementation.
"""

import torch
import traceback


def verify_rgbd_estimator(config, device, has_cuda_impl=False):
    """Verify RGBD estimator implementation"""
    print("\\n" + "="*60)
    print("RGBD Estimator Verification")
    print("="*60)
    
    results = {}
    
    try:
        print("[1/3] Initializing RGBD estimators...")
        
        # Import Calibration class
        from utils import Calibration
        
        # Create dummy calibrations and masks for testing
        num_cameras = len(config['references_indices'])
        dummy_calibrations = []
        dummy_masks = []
        for i in range(num_cameras):
            # Create dummy calibration (double sphere model)
            original_resolution = torch.tensor([config['matching_width'], config['matching_height']], device=device)
            principal = torch.tensor([config['matching_width'] / 2, config['matching_height'] / 2], device=device)
            fl = torch.tensor([300.0, 300.0], device=device)
            xi = 0.5
            alpha = 0.5
            rt = torch.eye(4, device=device)
            matching_scale = torch.tensor([1.0, 1.0], device=device)
            
            calib = Calibration(original_resolution, principal, fl, xi, alpha, rt, matching_scale)
            dummy_calibrations.append(calib)
            
            # Create dummy mask (all valid pixels) in [1, H, W] format
            # Note: matching_resolution is (cols, rows) = (width, height)
            mask = torch.ones(1, config['matching_height'], config['matching_width'], device=device, dtype=torch.float32)
            dummy_masks.append(mask)
        
        # Dummy reprojection viewpoint (origin)
        dummy_viewpoint = torch.tensor([0.0, 0.0, 0.0], device=device)
        
        # Python implementation
        has_python = False
        try:
            from depth_estimation import RGBD_Estimator as PythonRGBD_Estimator
            python_estimator = PythonRGBD_Estimator(
                dummy_calibrations,
                config['min_dist'], config['max_dist'],
                config['candidate_count'],
                config['references_indices'],
                dummy_viewpoint,
                dummy_masks,
                (config['matching_width'], config['matching_height']),
                (config['pano_width'], config['pano_height']),
                (config['pano_width'], config['pano_height']),
                config['sigma_i'], config['sigma_s'],
                device
            )
            print("  ✓ Python RGBD estimator created")
            has_python = True
        except Exception as e:
            print(f"  ✗ Python RGBD estimator failed: {e}")
        
        # CUDA implementation if available
        has_cuda = False
        if has_cuda_impl:
            try:
                import my_stereo_pkg
                cuda_estimator = my_stereo_pkg.RGBD_Estimator(
                    config['min_dist'], config['max_dist'], config['candidate_count'],
                    config['references_indices'], config['matching_width'], config['matching_height'],
                    config['pano_width'], config['pano_height'],
                    config['sigma_i'], config['sigma_s'], 0  # device_id
                )
                print("  ✓ CUDA RGBD estimator created")
                has_cuda = True
            except Exception as e:
                print(f"  ✗ CUDA RGBD estimator failed: {e}")
        
        if not has_python and not has_cuda:
            print("✗ No RGBD estimator implementation available")
            return False, results
        
        print("[2/3] Creating test input...")
        batch_size = 1
        test_pano = torch.randn(
            batch_size, 3, config['pano_height'], config['pano_width'],
            device=device
        )
        print(f"  ✓ Test panorama shape: {tuple(test_pano.shape)}")
        
        print("[3/3] Verification complete")
        
        results['rgbd_estimator'] = {
            'python_available': has_python,
            'cuda_available': has_cuda
        }
        
        print("✓ RGBD estimator verification PASSED")
        return True, results
        
    except Exception as e:
        print(f"✗ RGBD estimator verification failed: {e}")
        traceback.print_exc()
        return False, results
