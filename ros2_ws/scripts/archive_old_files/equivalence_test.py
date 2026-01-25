#!/usr/bin/env python3
"""
数値等価性検証スクリプト (Equivalence Test)

Python (PyTorch) 実装と C++/CUDA 実装の出力を比較し、
数値的な等価性を検証します。

テストフェーズ:
- Phase 1: 幾何変換 (Unproject/Project) の単体テスト
- Phase 2: Adaptive Camera Selection の比較
- Phase 3: フルパイプライン (ISB Filter実装後)
"""

import sys
import os
import numpy as np
import torch
import time
from typing import Tuple, List

# ROS2ワークスペースのパスを追加
sys.path.insert(0, '/home/motoken/college/ros2_ws/install/my_stereo_pkg/lib')
sys.path.insert(0, '/home/motoken/college/sphere-stereo/python')

try:
    import sphere_stereo_cuda
    CUDA_AVAILABLE = True
    print("✓ sphere_stereo_cuda モジュールのインポート成功")
except ImportError as e:
    CUDA_AVAILABLE = False
    print(f"✗ sphere_stereo_cuda モジュールのインポート失敗: {e}")
    print("  ヒント: colcon build --packages-select my_stereo_pkg を実行してください")

try:
    # sphere-stereo の Python実装をインポート
    import sys
    sphere_stereo_path = '/home/motoken/college/sphere-stereo/python'
    if sphere_stereo_path not in sys.path:
        sys.path.insert(0, sphere_stereo_path)
    
    from depth_estimation import RGBD_Estimator as PyTorch_RGBD_Estimator
    from utils import project, unproject
    
    # Calibrationクラスの簡易定義（importエラー回避用）
    class Calibration:
        def __init__(self):
            self.fl = None
            self.principal = None
            self.xi = 0.0
            self.alpha = 0.0
            self.matching_scale = 1.0
            self.rt = None
    
    PYTORCH_AVAILABLE = True
    print("✓ Python実装のインポート成功")
except ImportError as e:
    PYTORCH_AVAILABLE = False
    print(f"✗ Python実装のインポート失敗: {e}")
    print(f"  ヒント: sphere-stereo/python ディレクトリが存在することを確認してください")


# ============================================================================
# ユーティリティ関数
# ============================================================================

def generate_dummy_calibration(num_cameras: int = 2, 
                               device: str = 'cuda:0') -> List:
    """
    ダミーのカメラキャリブレーションを生成
    
    Args:
        num_cameras: カメラ数
        device: PyTorchデバイス
        
    Returns:
        calibrations: キャリブレーションのリスト
    """
    calibrations = []
    
    for i in range(num_cameras):
        calib = Calibration()
        
        # 内部パラメータ
        calib.fl = torch.tensor([200.0, 200.0], device=device)  # focal length
        calib.principal = torch.tensor([160.0, 120.0], device=device)  # principal point
        calib.xi = 0.1  # Double sphere parameter 1
        calib.alpha = 0.5  # Double sphere parameter 2
        calib.matching_scale = 1.0
        
        # 外部パラメータ (4x4 同次変換行列)
        rt = torch.eye(4, device=device)
        rt[0, 3] = i * 0.1  # X方向に0.1mずつオフセット
        calib.rt = rt
        
        calibrations.append(calib)
    
    return calibrations


def generate_random_images(num_cameras: int, 
                          width: int, 
                          height: int,
                          device: str = 'cuda:0') -> List[torch.Tensor]:
    """
    ランダムノイズ画像を生成（テスト用）
    
    Args:
        num_cameras: カメラ数
        width: 画像幅
        height: 画像高さ
        device: PyTorchデバイス
        
    Returns:
        images: [num_cameras] の画像リスト [H, W, 3] float32 [0, 255]
    """
    images = []
    
    for _ in range(num_cameras):
        # ランダムノイズ + グラデーション（構造を持たせる）
        base = torch.rand(height, width, 3, device=device) * 100.0
        
        # グラデーション追加
        y_grad = torch.linspace(0, 155, height, device=device).view(-1, 1, 1).expand(-1, width, 3)
        x_grad = torch.linspace(0, 155, width, device=device).view(1, -1, 1).expand(height, -1, 3)
        
        image = base + (y_grad + x_grad) / 2.0
        image = torch.clamp(image, 0, 255)
        
        images.append(image)
    
    return images


def calibration_to_cuda_format(calibrations: List) -> Tuple:
    """
    PyTorchのCalibrationオブジェクトをCUDA用の配列に変換
    
    Returns:
        (calibrations_rt, calibrations_intrinsics, calibrations_sphere, calibrations_resolution)
    """
    num_cameras = len(calibrations)
    
    calibrations_rt = []
    calibrations_intrinsics = []
    calibrations_sphere = []
    calibrations_resolution = []
    
    for calib in calibrations:
        # RT行列 (4x4 → 16要素、row-major)
        rt = calib.rt.cpu().numpy().flatten().tolist()
        calibrations_rt.extend(rt)
        
        # 内部パラメータ [fx, fy, cx, cy]
        fx = calib.fl[0].item()
        fy = calib.fl[1].item()
        cx = calib.principal[0].item()
        cy = calib.principal[1].item()
        calibrations_intrinsics.extend([fx, fy, cx, cy])
        
        # Double sphere パラメータ [xi, alpha]
        calibrations_sphere.extend([calib.xi, calib.alpha])
        
        # 解像度 [width, height]
        calibrations_resolution.extend([320.0, 240.0])  # 固定値
    
    return (calibrations_rt, calibrations_intrinsics, 
            calibrations_sphere, calibrations_resolution)


def compute_metrics(pytorch_output: np.ndarray, 
                   cuda_output: np.ndarray) -> dict:
    """
    2つの出力の数値的差異を計算
    
    Args:
        pytorch_output: Python版の出力
        cuda_output: CUDA版の出力
        
    Returns:
        metrics: 各種メトリクス
    """
    diff = pytorch_output - cuda_output
    
    metrics = {
        'mse': np.mean(diff ** 2),
        'mae': np.mean(np.abs(diff)),
        'max_error': np.max(np.abs(diff)),
        'min_error': np.min(np.abs(diff)),
        'std_error': np.std(diff),
        'pytorch_mean': np.mean(pytorch_output),
        'cuda_mean': np.mean(cuda_output),
        'pytorch_std': np.std(pytorch_output),
        'cuda_std': np.std(cuda_output),
    }
    
    return metrics


def print_metrics(metrics: dict, test_name: str):
    """メトリクスを見やすく出力"""
    print(f"\n{'='*70}")
    print(f"  {test_name} - 数値比較結果")
    print(f"{'='*70}")
    print(f"  Mean Squared Error (MSE):    {metrics['mse']:.6e}")
    print(f"  Mean Absolute Error (MAE):   {metrics['mae']:.6e}")
    print(f"  Max Absolute Error:          {metrics['max_error']:.6e}")
    print(f"  Min Absolute Error:          {metrics['min_error']:.6e}")
    print(f"  Standard Deviation (Error):  {metrics['std_error']:.6e}")
    print(f"{'-'*70}")
    print(f"  PyTorch Output Mean:         {metrics['pytorch_mean']:.6f}")
    print(f"  CUDA Output Mean:            {metrics['cuda_mean']:.6f}")
    print(f"  PyTorch Output Std:          {metrics['pytorch_std']:.6f}")
    print(f"  CUDA Output Std:             {metrics['cuda_std']:.6f}")
    print(f"{'='*70}")
    
    # 合否判定
    if metrics['mse'] < 1e-4:
        print(f"  ✓ PASS: MSE < 1e-4 (高精度で一致)")
    elif metrics['mse'] < 1e-2:
        print(f"  ⚠ WARNING: 1e-4 <= MSE < 1e-2 (許容範囲内)")
    else:
        print(f"  ✗ FAIL: MSE >= 1e-2 (大きな差異あり)")


# ============================================================================
# Phase 1: 幾何変換テスト (TODO: 実装予定)
# ============================================================================

def test_unproject_project():
    """
    Unproject/Projectの単体テスト
    
    NOTE: 現在未実装。CUDA側で幾何変換の単体テスト用APIが必要
    """
    print("\n" + "="*70)
    print("Phase 1: 幾何変換テスト (Unproject/Project)")
    print("="*70)
    print("⚠️ CUDA側の単体テストAPIが未実装のためスキップ")
    print("   実装後に以下をテスト:")
    print("   - unproject_double_sphere() の精度")
    print("   - project_double_sphere() の精度")
    print("   - 往復変換 (project → unproject) の誤差")


# ============================================================================
# Phase 2: Camera Selection テスト (TODO: 実装予定)
# ============================================================================

def test_camera_selection():
    """
    Adaptive Camera Selection の比較
    
    NOTE: 現在未実装。select_best_cameras_kernel の出力を取得するAPIが必要
    """
    print("\n" + "="*70)
    print("Phase 2: Camera Selection テスト")
    print("="*70)
    print("⚠️ CUDA側のカメラ選択結果を取得するAPIが未実装のためスキップ")
    print("   実装後に以下をテスト:")
    print("   - selected_cameras の一致率")
    print("   - max_displacement の数値差")


# ============================================================================
# Phase 3: フルパイプラインテスト (現在モック実装)
# ============================================================================

def test_full_pipeline():
    """
    フルパイプライン (estimate_RGBD_panorama) の比較
    
    NOTE: 現在CUDA側がモック実装のため、ダミーデータを返すのみ
    """
    print("\n" + "="*70)
    print("Phase 3: フルパイプラインテスト")
    print("="*70)
    
    if not CUDA_AVAILABLE:
        print("✗ CUDA実装が利用不可のためスキップ")
        return
    
    if not PYTORCH_AVAILABLE:
        print("✗ PyTorch実装が利用不可のためスキップ")
        return
    
    print("\n[1] テストデータ生成中...")
    
    # パラメータ
    num_cameras = 2
    matching_width, matching_height = 320, 240
    stitch_width, stitch_height = 640, 480
    panorama_width, panorama_height = 800, 400
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"    デバイス: {device}")
    
    # キャリブレーション生成
    calibrations = generate_dummy_calibration(num_cameras, device)
    print(f"    ✓ カメラキャリブレーション生成完了 ({num_cameras}台)")
    
    # 画像生成
    images_to_match = generate_random_images(num_cameras, matching_width, matching_height, device)
    images_to_stitch = generate_random_images(num_cameras, stitch_width, stitch_height, device)
    print(f"    ✓ ランダム画像生成完了")
    print(f"      - Matching: {num_cameras} @ {matching_width}x{matching_height}")
    print(f"      - Stitching: {num_cameras} @ {stitch_width}x{stitch_height}")
    
    # マスク生成（全て有効）
    masks = [torch.ones(matching_height, matching_width, device=device).unsqueeze(0) 
             for _ in range(num_cameras)]
    
    # ============================================================================
    # CUDA版実行
    # ============================================================================
    
    print("\n[2] CUDA版実行中...")
    
    try:
        # CUDA用にデータ変換
        calib_rt, calib_intrinsics, calib_sphere, calib_resolution = \
            calibration_to_cuda_format(calibrations)
        
        # CUDA Estimator作成
        cuda_estimator = sphere_stereo_cuda.RGBD_Estimator(
            calib_rt,
            calib_intrinsics,
            calib_sphere,
            calib_resolution,
            0.5,  # min_dist
            10.0,  # max_dist
            50,  # candidate_count
            [0],  # references_indices
            [0.0, 0.0, 0.0],  # reprojection_viewpoint
            [matching_width] * num_cameras,  # image_widths
            [matching_height] * num_cameras,  # image_heights
            matching_width,
            matching_height,
            stitch_width,
            stitch_height,
            panorama_width,
            panorama_height,
            1.0,  # sigma_i
            1.0,  # sigma_s
            0  # device
        )
        print("    ✓ CUDA Estimator作成完了")
        
        # 画像をCPUに転送してfloat32配列化
        cuda_images_to_match = [img.cpu().numpy().flatten().tolist() 
                                for img in images_to_match]
        cuda_images_to_stitch = [img.cpu().numpy().flatten().tolist() 
                                 for img in images_to_stitch]
        
        # 推定実行
        start_time = time.time()
        cuda_rgb, cuda_distance = cuda_estimator.estimate_RGBD_panorama(
            cuda_images_to_match,
            cuda_images_to_stitch
        )
        cuda_time = time.time() - start_time
        
        print(f"    ✓ CUDA推定完了 ({cuda_time:.3f}秒)")
        print(f"      - RGB shape: {cuda_rgb.shape}")
        print(f"      - Distance shape: {cuda_distance.shape}")
        
        # NumPy配列に変換
        cuda_rgb_np = np.array(cuda_rgb)
        cuda_distance_np = np.array(cuda_distance)
        
    except Exception as e:
        print(f"    ✗ CUDA版実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================================
    # PyTorch版実行 (TODO: 完全実装後に有効化)
    # ============================================================================
    
    print("\n[3] PyTorch版実行...")
    print("    ⚠️ PyTorch版の実行は現在スキップ（完全実装後に有効化予定）")
    print("    理由:")
    print("      - CUDA版がモック実装のため有意義な比較ができない")
    print("      - ISB Filter等の依存ライブラリの初期化が必要")
    
    # 将来の実装用コメント:
    # try:
    #     pytorch_estimator = PyTorch_RGBD_Estimator(
    #         calibrations, 0.5, 10.0, 50, [0], 
    #         torch.tensor([0.0, 0.0, 0.0], device=device),
    #         masks, (matching_width, matching_height),
    #         (stitch_width, stitch_height), (panorama_width, panorama_height),
    #         1.0, 1.0, device
    #     )
    #     
    #     start_time = time.time()
    #     pytorch_rgb, pytorch_distance = pytorch_estimator.estimate_RGBD_panorama(
    #         images_to_match, images_to_stitch
    #     )
    #     pytorch_time = time.time() - start_time
    #     
    #     pytorch_rgb_np = pytorch_rgb.cpu().numpy()
    #     pytorch_distance_np = pytorch_distance.cpu().numpy()
    #     
    #     # メトリクス計算
    #     distance_metrics = compute_metrics(pytorch_distance_np, cuda_distance_np)
    #     print_metrics(distance_metrics, "Distance Map")
    #     
    # except Exception as e:
    #     print(f"    ✗ PyTorch版実行エラー: {e}")
    
    # ============================================================================
    # 現状の確認
    # ============================================================================
    
    print("\n[4] 現状サマリー:")
    print("    - CUDA版: モック実装（ダミーデータを返却）")
    print("    - PyTorch版: 未実行")
    print("    - 比較: 不可能")
    print("\n    次のステップ:")
    print("    1. CUDA版のSweeping/Cost計算を実装")
    print("    2. ISB Filterを実装")
    print("    3. このスクリプトを再実行して数値比較")


# ============================================================================
# メイン
# ============================================================================

def main():
    """メインエントリーポイント"""
    
    print("="*70)
    print("  Depth Estimation 数値等価性検証スクリプト")
    print("="*70)
    print(f"  Python実装: {'✓ 利用可能' if PYTORCH_AVAILABLE else '✗ 利用不可'}")
    print(f"  CUDA実装:   {'✓ 利用可能' if CUDA_AVAILABLE else '✗ 利用不可'}")
    print(f"  PyTorch CUDA: {'✓ 利用可能' if torch.cuda.is_available() else '✗ 利用不可'}")
    print("="*70)
    
    # 各フェーズのテスト実行
    test_unproject_project()  # Phase 1 (未実装)
    test_camera_selection()   # Phase 2 (未実装)
    test_full_pipeline()      # Phase 3 (モック実装)
    
    print("\n" + "="*70)
    print("  テスト完了")
    print("="*70)
    print("\n詳細な比較レポートは ALGORITHM_COMPARISON.md を参照してください")


if __name__ == "__main__":
    main()
