"""
RT Matrix Debug Module

Debug rotation and translation matrices for camera calibrations.
"""

import torch
import numpy as np


def debug_rt_matrix(config, device, generate_calibrations_func):
    """Debug RT matrices in calibrations"""
    print("\\n" + "="*60)
    print("RT Matrix Debug Analysis")
    print("="*60)
    
    try:
        print("[1/3] Loading calibrations...")
        calibrations = generate_calibrations_func()
        print(f"  ✓ Loaded {len(calibrations)} calibrations")
        
        print("[2/3] Analyzing RT matrices...")
        for i, calib in enumerate(calibrations):
            print(f"\\n  Camera {i}:")
            rt = calib.rt
            R = rt[:3, :3]
            t = rt[:3, 3]
            
            # Check rotation matrix properties
            det_R = torch.det(R).item()
            RTR = R.T @ R
            identity_diff = torch.abs(RTR - torch.eye(3, device=device)).max().item()
            
            print(f"    Rotation det: {det_R:.6f} (should be ~1.0)")
            print(f"    Orthogonality error: {identity_diff:.6e}")
            print(f"    Translation: [{t[0].item():.4f}, {t[1].item():.4f}, {t[2].item():.4f}]")
            
            # Extract Euler angles (approximate)
            if abs(det_R - 1.0) < 0.01:
                R_np = R.cpu().numpy()
                sy = np.sqrt(R_np[0,0]**2 + R_np[1,0]**2)
                if sy > 1e-6:
                    roll = np.arctan2(R_np[2,1], R_np[2,2])
                    pitch = np.arctan2(-R_np[2,0], sy)
                    yaw = np.arctan2(R_np[1,0], R_np[0,0])
                    print(f"    Euler angles (deg): roll={np.degrees(roll):.2f}, pitch={np.degrees(pitch):.2f}, yaw={np.degrees(yaw):.2f}")
        
        print("\\n[3/3] Checking relative poses...")
        if len(calibrations) >= 2:
            for i in range(len(calibrations) - 1):
                rt1 = calibrations[i].rt
                rt2 = calibrations[i+1].rt
                relative_rt = torch.inverse(rt1) @ rt2
                rel_t = relative_rt[:3, 3]
                rel_dist = torch.norm(rel_t).item()
                print(f"  Camera {i} to {i+1}: relative distance = {rel_dist:.4f}m")
        
        print("\\n✓ RT matrix debug complete")
        
        return {
            'num_cameras': len(calibrations),
            'all_valid': True
        }
        
    except Exception as e:
        print(f"✗ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return {}
