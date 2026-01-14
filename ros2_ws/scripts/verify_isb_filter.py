import torch
import time
import sys
import os

# オリジナルのPython版をインポートできるようにパスを追加
sys.path.insert(0, '/home/motoken/college/sphere-stereo/python')
from isb_filter import ISB_Filter as PyFilter

# ROS 2ワークスペースのC++モジュールをインポート
sys.path.insert(0, '/home/motoken/college/ros2_ws/install/my_stereo_pkg/lib/python3.10/site-packages')
from my_stereo_pkg import ISBFilter as CppFilter

def verify():
    device = torch.device("cuda:0")
    D, W, H = 64, 640, 480
    sigma_i, sigma_s = 0.1, 15.0

    print(f"--- ISB Filter Verification ({W}x{H}, D={D}) ---")

    # 1. データの準備
    guide = torch.randint(0, 255, (H, W, 3), dtype=torch.uint8, device=device)
    cost = torch.randn((D, H, W), dtype=torch.float32, device=device)

    # 2. インスタンス化
    py_filter = PyFilter(D, (W, H), device)
    cpp_filter = CppFilter(D, (W, H), device)

    # ウォームアップ (CUDAの初期化ラグを除く)
    for _ in range(3):
        cpp_filter.apply(guide, cost, sigma_i, sigma_s)

    # 3. 数値の正確性テスト
    py_res, _ = py_filter.apply(guide, cost, sigma_i, sigma_s)
    cpp_res, _ = cpp_filter.apply(guide, cost, sigma_i, sigma_s)

    mae = (py_res - cpp_res).abs().mean().item()
    max_err = (py_res - cpp_res).abs().max().item()
    print(f"Numerical Accuracy:")
    print(f"  MAE: {mae:.8f}")
    print(f"  Max Error: {max_err:.8f}")

    # 4. パフォーマンステスト
    iters = 20
    torch.cuda.synchronize()
    
    # Python (CuPy) 計測
    start = time.time()
    for _ in range(iters):
        py_filter.apply(guide, cost, sigma_i, sigma_s)
    torch.cuda.synchronize()
    py_time = (time.time() - start) / iters

    # C++ (LibTorch) 計測
    start = time.time()
    for _ in range(iters):
        cpp_filter.apply(guide, cost, sigma_i, sigma_s)
    torch.cuda.synchronize()
    cpp_time = (time.time() - start) / iters

    print(f"\nPerformance:")
    print(f"  Python version: {py_time*1000:.2f} ms")
    print(f"  C++ version:    {cpp_time*1000:.2f} ms")
    print(f"  Speedup:        {py_time/cpp_time:.2f}x")

if __name__ == "__main__":
    verify()