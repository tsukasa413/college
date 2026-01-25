#!/usr/bin/env python3
"""
相対RT行列とunprojection結果のデバッグ
"""
import torch
import numpy as np
import sys
import os

# Add sphere-stereo to path
sys.path.append('/home/motoken/college/sphere-stereo/python')

# Import Python implementation
from utils import Calibration, unproject, project, parse_json_calib

def main():
    # Load calibration
    with open('/home/motoken/college/sphere-stereo/resources/calibration.json', 'r') as f:
        import json
        calibration_data = json.load(f)
    
    # Extract value0 (basalt format)
    raw_calibration = calibration_data['value0']
    
    device = torch.device('cuda:0')
    matching_resolution = (640, 480)
    original_resolution = (1280, 1024)
    
    # Parse calibrations
    calibrations = parse_json_calib(
        raw_calibration, matching_resolution, device, original_resolution
    )
    
    print("=" * 80)
    print("PYTHON CALIBRATION DEBUG")
    print("=" * 80)
    
    # Check camera 0 and camera 1
    cam0 = calibrations[0]
    cam1 = calibrations[1]
    
    print("\nCamera 0:")
    print(f"  fl: {cam0.fl}")
    print(f"  principal: {cam0.principal}")
    print(f"  xi: {cam0.xi}, alpha: {cam0.alpha}")
    print(f"  matching_scale: {cam0.matching_scale}")
    print(f"  rt:\n{cam0.rt}")
    
    print("\nCamera 1:")
    print(f"  fl: {cam1.fl}")
    print(f"  principal: {cam1.principal}")
    print(f"  xi: {cam1.xi}, alpha: {cam1.alpha}")
    print(f"  matching_scale: {cam1.matching_scale}")
    print(f"  rt:\n{cam1.rt}")
    
    # Compute relative RT: inv(cam1.rt) @ cam0.rt
    print("\n" + "=" * 80)
    print("RELATIVE RT MATRIX")
    print("=" * 80)
    
    rt_rel = torch.matmul(torch.inverse(cam1.rt), cam0.rt)
    print(f"\ninv(cam1.rt) @ cam0.rt:\n{rt_rel}")
    
    # Extract R and t
    R = rt_rel[:3, :3].cpu().numpy()
    t = rt_rel[:3, 3].cpu().numpy()
    print(f"\nR (rotation):\n{R}")
    print(f"t (translation):\n{t}")
    
    # Test unprojection at center pixel
    print("\n" + "=" * 80)
    print("UNPROJECTION TEST")
    print("=" * 80)
    
    uv = torch.tensor([[320.0, 240.0]], device=device)
    print(f"\nTest pixel: u={uv[0, 0]}, v={uv[0, 1]}")
    
    pt_cam0, valid_cam0 = unproject(uv, cam0)
    print(f"\nCamera 0 unproject:")
    print(f"  Point: {pt_cam0[0]}")
    print(f"  Valid: {valid_cam0[0]}")
    print(f"  Norm: {torch.norm(pt_cam0[0])}")
    
    pt_cam1, valid_cam1 = unproject(uv, cam1)
    print(f"\nCamera 1 unproject:")
    print(f"  Point: {pt_cam1[0]}")
    print(f"  Valid: {valid_cam1[0]}")
    print(f"  Norm: {torch.norm(pt_cam1[0])}")
    
    # Transform point from cam0 to cam1
    print("\n" + "=" * 80)
    print("COORDINATE TRANSFORMATION")
    print("=" * 80)
    
    pt_cam0_homo = torch.cat([pt_cam0[0], torch.ones(1, device=device)])
    pt_cam1_from_cam0 = torch.matmul(pt_cam0_homo, rt_rel.T)[:3]
    
    print(f"\nPoint in cam0 frame: {pt_cam0[0]}")
    print(f"Transformed to cam1 frame: {pt_cam1_from_cam0}")
    print(f"Norm after transform: {torch.norm(pt_cam1_from_cam0)}")
    
    # Normalize and project
    pt_cam1_from_cam0_normalized = pt_cam1_from_cam0 / torch.norm(pt_cam1_from_cam0)
    uv_proj, valid_proj = project(pt_cam1_from_cam0_normalized.unsqueeze(0), cam1)
    
    print(f"\nNormalized: {pt_cam1_from_cam0_normalized}")
    print(f"Projected to cam1 image: u={uv_proj[0, 0]}, v={uv_proj[0, 1]}")
    print(f"Valid: {valid_proj[0]}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
