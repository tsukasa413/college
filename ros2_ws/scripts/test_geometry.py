#!/usr/bin/env python3
"""
幾何変換 単体テスト (Unproject/Project)

Double Sphere Model の数値精度を検証します。
このテストは CUDA カーネルの単体テスト用 API が実装されたら有効化されます。
"""

import numpy as np
import torch
import sys

sys.path.insert(0, '/home/motoken/college/sphere-stereo/python')
from utils import project, unproject


def create_test_calibration():
    """テスト用キャリブレーション"""
    class Calibration:
        def __init__(self):
            self.fl = torch.tensor([200.0, 200.0], device='cuda:0')
            self.principal = torch.tensor([160.0, 120.0], device='cuda:0')
            self.xi = 0.1
            self.alpha = 0.5
            self.matching_scale = 1.0  # 追加
            self.rt = torch.eye(4, device='cuda:0')
    
    return Calibration()


def test_roundtrip():
    """
    Unproject → Project の往復変換テスト
    
    期待値: 元のUV座標に戻ること（誤差 < 1e-5）
    """
    print("\n" + "="*70)
    print("  幾何変換 往復テスト (Unproject → Project)")
    print("="*70)
    
    calib = create_test_calibration()
    
    # テスト用UV座標（画像中心付近）
    uv_test = torch.tensor([
        [160.0, 120.0],  # 中心
        [100.0, 80.0],   # 左上
        [220.0, 160.0],  # 右下
        [160.0, 200.0],  # 下部
        [50.0, 50.0],    # 角
    ], device='cuda:0').unsqueeze(0)
    
    print(f"\n[1] 入力UV座標: {uv_test.shape}")
    print(f"    {uv_test[0].cpu().numpy()}")
    
    # Unproject: UV → 3D unit vector
    pt_3d, valid = unproject(uv_test, calib)
    print(f"\n[2] Unproject結果:")
    print(f"    - 3D points shape: {pt_3d.shape}")
    print(f"    - Valid mask: {valid[0].cpu().numpy()}")
    print(f"    - Sample point: {pt_3d[0, 0].cpu().numpy()}")
    
    # Project: 3D → UV
    uv_reconstructed, valid_proj = project(pt_3d, calib)
    print(f"\n[3] Project結果:")
    print(f"    - UV shape: {uv_reconstructed.shape}")
    print(f"    {uv_reconstructed[0].cpu().numpy()}")
    
    # 誤差計算
    error = torch.abs(uv_test - uv_reconstructed)
    max_error = torch.max(error).item()
    mean_error = torch.mean(error).item()
    
    print(f"\n[4] 誤差:")
    print(f"    - Max Error:  {max_error:.6e} pixels")
    print(f"    - Mean Error: {mean_error:.6e} pixels")
    
    if max_error < 1e-5:
        print(f"    ✓ PASS: 誤差 < 1e-5")
    elif max_error < 1e-3:
        print(f"    ⚠ WARNING: 1e-5 <= 誤差 < 1e-3")
    else:
        print(f"    ✗ FAIL: 誤差 >= 1e-3")
    
    print("="*70)


def test_distance_variation():
    """
    異なる距離での投影精度テスト
    
    深度推定では 0.5m～10m の範囲を使用するため、
    この範囲での精度を確認
    """
    print("\n" + "="*70)
    print("  距離依存性テスト")
    print("="*70)
    
    calib = create_test_calibration()
    
    # 中心ピクセル
    uv = torch.tensor([[160.0, 120.0]], device='cuda:0').unsqueeze(0)
    
    # Unproject to unit vector
    pt_unit, _ = unproject(uv, calib)
    
    # 異なる距離でテスト
    distances = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    print(f"\n距離ごとの投影精度:")
    print(f"{'距離 (m)':>10} | {'Max Error (px)':>15} | {'判定':>6}")
    print("-" * 70)
    
    for dist in distances:
        # 3D point at distance
        pt_3d = pt_unit * dist
        
        # Project back
        uv_proj, valid = project(pt_3d, calib)
        
        # Error
        error = torch.abs(uv - uv_proj)
        max_error = torch.max(error).item()
        
        status = "✓" if max_error < 1e-3 else "✗"
        print(f"{dist:>10.1f} | {max_error:>15.6e} | {status:>6}")
    
    print("="*70)


if __name__ == "__main__":
    print("="*70)
    print("  Double Sphere Model 幾何変換テスト (Python版)")
    print("="*70)
    print("\n⚠️ 注意: これはPython実装の精度確認です")
    print("   CUDA版との比較には、CUDA側の単体テストAPIが必要です")
    
    try:
        test_roundtrip()
        test_distance_variation()
        
        print("\n" + "="*70)
        print("  全テスト完了")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ エラー: {e}")
        import traceback
        traceback.print_exc()
