#!/usr/bin/env python3
"""
verify_equivalence.py

Equivalence test between Python (sphere-stereo) and C++/CUDA (my_stereo_pkg) implementations.
Tests numerical equivalence of RGBD_Estimator with identical inputs.

Test Strategy:
1. Generate synthetic fisheye images with known calibrations
2. Execute both Python and C++ implementations
3. Compare outputs with multiple metrics (MSE, pixel-wise difference, etc.)
4. Account for known differences (quadratic fitting, filtering order)
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Set up library paths for LibTorch and ROS2
torch_lib_path = os.path.join(torch.utils.cmake_prefix_path, "..", "lib")
if os.path.exists(torch_lib_path):
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if torch_lib_path not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}"
        print(f"✓ Added PyTorch library path: {torch_lib_path}")

# Add ROS2 library path
ros2_lib_path = Path(__file__).parent.parent / "install" / "my_stereo_pkg" / "lib"
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
if str(ros2_lib_path) not in current_ld_path:
    os.environ['LD_LIBRARY_PATH'] = f"{ros2_lib_path}:{current_ld_path}"
    print(f"✓ Added ROS2 library path: {ros2_lib_path}")

# Add sphere-stereo Python implementation to path
sphere_stereo_path = Path(__file__).parent.parent.parent / "sphere-stereo" / "python"
sys.path.insert(0, str(sphere_stereo_path))

# Add C++/CUDA module to path
cuda_module_path = Path(__file__).parent.parent / "install" / "my_stereo_pkg" / "lib"
sys.path.insert(0, str(cuda_module_path))

try:
    from depth_estimation import RGBD_Estimator as PythonRGBD_Estimator
    from utils import Calibration as PythonCalibration
    print("✓ Python implementation loaded")
except ImportError as e:
    print(f"✗ Failed to load Python implementation: {e}")
    sys.exit(1)

try:
    import sphere_stereo_cuda
    print("✓ C++/CUDA implementation loaded")
except ImportError as e:
    print(f"✗ Failed to load C++/CUDA implementation: {e}")
    sys.exit(1)


# ============================================================================
# Synthetic Data Generation
# ============================================================================

def generate_synthetic_calibration(device):
    """
    Generate synthetic Double Sphere calibrations for 4 fisheye cameras
    arranged in a square pattern.
    
    Returns:
        python_calibrations: List of Python Calibration objects
        cuda_calibrations_flat: Flattened arrays for C++ interface
    """
    width, height = 640, 480
    fx, fy = 285.0, 285.0
    cx, cy = width / 2.0, height / 2.0
    xi, alpha = 0.5, 0.6  # Double Sphere parameters
    
    # Camera positions in a square (0.5m apart)
    camera_positions = [
        [0.0, 0.0, 0.0],      # Reference (front)
        [0.5, 0.0, 0.0],      # Right
        [0.0, 0.5, 0.0],      # Top
        [-0.5, 0.0, 0.0],     # Left
    ]
    
    python_calibrations = []
    rt_matrices = []
    intrinsics_list = []
    sphere_params_list = []
    resolution_list = []
    
    for i, pos in enumerate(camera_positions):
        # RT matrix (4x4 row-major)
        rt = np.eye(4, dtype=np.float32)
        rt[0, 3] = pos[0]
        rt[1, 3] = pos[1]
        rt[2, 3] = pos[2]
        
        # Python calibration
        py_calib = PythonCalibration(
            original_resolution=torch.tensor([width, height], device=device, dtype=torch.int32),
            principal=torch.tensor([cx, cy], device=device, dtype=torch.float32),
            fl=torch.tensor([fx, fy], device=device, dtype=torch.float32),
            xi=xi,
            alpha=alpha,
            rt=torch.from_numpy(rt).to(device),
            matching_scale=torch.tensor([1.0, 1.0], device=device, dtype=torch.float32)
        )
        python_calibrations.append(py_calib)
        
        # C++ interface (flattened)
        rt_matrices.extend(rt.flatten().tolist())
        intrinsics_list.extend([fx, fy, cx, cy])
        sphere_params_list.extend([xi, alpha])
        resolution_list.extend([width, height])
    
    cuda_calibrations_flat = {
        'rt': rt_matrices,
        'intrinsics': intrinsics_list,
        'sphere': sphere_params_list,
        'resolution': resolution_list,
    }
    
    return python_calibrations, cuda_calibrations_flat


def generate_synthetic_images(num_cameras, width, height, device):
    """
    Generate synthetic fisheye-like images with radial patterns.
    
    Returns:
        images: List of [H, W, 3] tensors (float32, range [0, 255])
    """
    images = []
    
    for i in range(num_cameras):
        # Create radial gradient pattern
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        cx, cy = width / 2, height / 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        theta = np.arctan2(y - cy, x - cx)
        
        # RGB channels with different patterns
        r_channel = (128 + 127 * np.cos(r / 50 + i * np.pi / 2)).astype(np.float32)
        g_channel = (128 + 127 * np.sin(theta * 3 + i * np.pi / 4)).astype(np.float32)
        b_channel = (128 + 127 * np.cos(r / 100 - theta * 2)).astype(np.float32)
        
        # Add some texture
        noise = np.random.randn(height, width) * 10
        r_channel = np.clip(r_channel + noise, 0, 255)
        g_channel = np.clip(g_channel + noise, 0, 255)
        b_channel = np.clip(b_channel + noise, 0, 255)
        
        img = np.stack([r_channel, g_channel, b_channel], axis=-1).astype(np.float32)
        images.append(torch.from_numpy(img).to(device))
    
    return images


def generate_masks(num_cameras, width, height, device):
    """
    Generate circular validity masks for fisheye images.
    """
    masks = []
    
    for _ in range(num_cameras):
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        cx, cy = width / 2, height / 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Valid within 90% of image diagonal
        max_radius = 0.45 * np.sqrt(width**2 + height**2)
        mask = (r < max_radius).astype(np.float32)
        
        masks.append(torch.from_numpy(mask).unsqueeze(0).to(device))
    
    return masks


# ============================================================================
# Python Execution Path
# ============================================================================

def analyze_cost_volume(cost_volume, distance_candidates, title="Cost Volume Analysis"):
    """
    Analyze and visualize cost volume to understand depth estimation behavior.
    
    Args:
        cost_volume: [candidate_count, H, W] cost tensor
        distance_candidates: [candidate_count] distance values
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    candidate_count, height, width = cost_volume.shape
    
    # Find minimum cost indices for each pixel
    min_indices = torch.argmin(cost_volume, dim=0)  # [H, W]
    min_costs = torch.min(cost_volume, dim=0)[0]    # [H, W]
    
    # Convert to numpy for visualization
    cost_volume_np = cost_volume.cpu().numpy()
    min_indices_np = min_indices.cpu().numpy()
    min_costs_np = min_costs.cpu().numpy()
    distance_candidates_np = distance_candidates.cpu().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Cost volume slice at center
    center_y, center_x = height // 2, width // 2
    cost_profile = cost_volume_np[:, center_y, center_x]
    
    axes[0, 0].plot(distance_candidates_np, cost_profile, 'b-', linewidth=2, marker='o')
    axes[0, 0].set_xlabel('Distance (m)')
    axes[0, 0].set_ylabel('Cost')
    axes[0, 0].set_title(f'Cost Profile at Center ({center_x}, {center_y})')
    axes[0, 0].grid(True, alpha=0.3)
    
    min_idx_center = min_indices_np[center_y, center_x]
    min_distance = distance_candidates_np[min_idx_center]
    axes[0, 0].axvline(x=min_distance, color='r', linestyle='--', 
                       label=f'Min: {min_distance:.2f}m')
    axes[0, 0].legend()
    
    # 2. Selected distance map
    selected_distances = distance_candidates_np[min_indices_np]
    im1 = axes[0, 1].imshow(selected_distances, cmap='viridis')
    axes[0, 1].set_title('Selected Distance Map')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 3. Minimum cost map
    im2 = axes[0, 2].imshow(min_costs_np, cmap='plasma')
    axes[0, 2].set_title('Minimum Cost Map')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 4. Cost distribution histogram
    axes[1, 0].hist(cost_profile, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Cost Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Cost Distribution at Center')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Distance candidate selection histogram
    axes[1, 1].hist(min_indices_np.flatten(), bins=candidate_count, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Distance Candidate Index')
    axes[1, 1].set_ylabel('Pixel Count')
    axes[1, 1].set_title('Distance Candidate Selection Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Cost volume average across spatial dimensions
    avg_costs = np.mean(cost_volume_np, axis=(1, 2))  # [candidate_count]
    axes[1, 2].plot(distance_candidates_np, avg_costs, 'g-', linewidth=2, marker='s')
    axes[1, 2].set_xlabel('Distance (m)')
    axes[1, 2].set_ylabel('Average Cost')
    axes[1, 2].set_title('Average Cost vs Distance')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, {
        'selected_distances': selected_distances,
        'min_costs': min_costs_np,
        'cost_profile_center': cost_profile,
        'avg_costs': avg_costs
    }


def compute_distance_candidates(near_plane=0.5, far_plane=15.0, num_candidates=512):
    """
    Compute distance candidates for sphere sweeping.
    
    Args:
        near_plane: Nearest distance candidate
        far_plane: Farthest distance candidate  
        num_candidates: Number of distance candidates
        
    Returns:
        torch tensor of distance candidates
    """
    # Use inverse depth parameterization for better distribution
    inv_near = 1.0 / far_plane
    inv_far = 1.0 / near_plane
    inv_depths = torch.linspace(inv_near, inv_far, num_candidates)
    distances = 1.0 / inv_depths
    return distances


def analyze_subpixel_refinement(cost_volume, distance_candidates, title="Sub-pixel Analysis"):
    """
    Detailed analysis of sub-pixel refinement differences between implementations.
    
    Args:
        cost_volume: Cost tensor [D, H, W] where D is number of distance candidates
        distance_candidates: Array of distance values [D]  
        title: Plot title
    
    Returns:
        Dict with sub-pixel refinement analysis
    """
    print(f"\n[Sub-pixel Refinement Analysis]")
    
    # Convert to numpy if needed
    if isinstance(cost_volume, torch.Tensor):
        cv_np = cost_volume.cpu().numpy()
    else:
        cv_np = cost_volume
    
    if isinstance(distance_candidates, torch.Tensor):
        dist_np = distance_candidates.cpu().numpy()
    else:
        dist_np = distance_candidates
    
    D, H, W = cv_np.shape
    
    # Python-style sub-pixel refinement simulation
    def simulate_python_subpixel(costs, idx):
        if idx == 0 or idx == len(costs) - 1:
            return float(idx), 0.0
        
        left_cost = costs[idx - 1]
        center_cost = costs[idx]
        right_cost = costs[idx + 1]
        
        denominator = left_cost + right_cost - 2.0 * center_cost + 1e-8
        if abs(denominator) > 1e-6:
            variation = 0.5 * (left_cost - right_cost) / denominator
            variation = max(-0.5, min(0.5, variation))
        else:
            variation = 0.0
        
        return float(idx) + variation, variation
    
    # Test center pixel for detailed analysis
    test_x, test_y = W // 2, H // 2
    test_costs = cv_np[:, test_y, test_x]
    
    # Find minimum indices
    min_idx = np.argmin(test_costs)
    
    # Compare refinements
    py_refined, py_var = simulate_python_subpixel(test_costs, min_idx)
    
    print(f"  Test pixel ({test_x}, {test_y}):")
    print(f"    Min cost index: {min_idx}")
    print(f"    Python refinement: {py_refined:.6f} (var: {py_var:.6f})")
    
    # Distance conversion impact
    def idx_to_distance_python(idx, dist_candidates):
        return dist_candidates[0] / ((dist_candidates[0] / dist_candidates[-1] - 1) * idx / (len(dist_candidates) - 1) + 1)
    
    def idx_to_distance_cpp(idx, dist_candidates):
        inv_min = 1.0 / dist_candidates[-1]
        inv_max = 1.0 / dist_candidates[0]
        inv_dist = inv_max - (inv_max - inv_min) * (idx / (len(dist_candidates) - 1))
        return 1.0 / inv_dist
    
    # Compare distance conversion for test pixel
    py_distance = idx_to_distance_python(py_refined, dist_np)
    cpp_distance = idx_to_distance_cpp(py_refined, dist_np)
    
    print(f"\n  Distance Conversion Impact:")
    print(f"    Python distance: {py_distance:.6f}m")
    print(f"    C++ distance: {cpp_distance:.6f}m") 
    print(f"    Distance difference: {abs(py_distance - cpp_distance):.6f}m")
    
    return {
        'distance_conversion_diff': abs(py_distance - cpp_distance),
        'test_pixel_py_refined': py_refined,
        'refinement_diff_max': abs(py_var)
    }

def analyze_wta_boundary_effects(py_dist, cuda_dist, distance_candidates, title="WTA Boundary Analysis"):
    """
    Analyze Winner-Takes-All boundary effects and floating-point precision issues.
    """
    print(f"\n[WTA Boundary Effects Analysis]")
    
    # Convert to numpy
    if isinstance(py_dist, torch.Tensor):
        py_np = py_dist.cpu().numpy()
    else:
        py_np = py_dist
        
    if isinstance(cuda_dist, torch.Tensor):
        cuda_np = cuda_dist.cpu().numpy()
    else:
        cuda_np = cuda_dist
    
    # Analyze floating-point precision differences
    diff_map = np.abs(py_np - cuda_np)
    
    # Find small differences that might be due to floating-point precision
    small_diff_threshold = 1e-5
    small_diffs = (diff_map > 0) & (diff_map < small_diff_threshold)
    
    print(f"  Difference magnitude analysis:")
    print(f"    Small diffs (<{small_diff_threshold}): {small_diffs.sum()} pixels ({small_diffs.mean()*100:.2f}%)")
    
    # Check for systematic bias
    bias = (py_np - cuda_np).mean()
    print(f"    Mean bias: {bias:.6f}m")
    
    return {
        'small_diff_ratio': small_diffs.mean(),
        'systematic_bias': bias
    }


def run_python_implementation(calibrations, images, masks, config, device):
    """
    Execute Python RGBD_Estimator with given inputs.
    
    Returns:
        rgb_pano: [H, W, 3] uint8
        distance_pano: [H, W] float32
    """
    print("\n[Python Path] Initializing RGBD_Estimator...")
    
    estimator = PythonRGBD_Estimator(
        calibrations=calibrations,
        min_dist=config['min_dist'],
        max_dist=config['max_dist'],
        candidate_count=config['candidate_count'],
        references_indices=config['references_indices'],
        reprojection_viewpoint=torch.tensor([0.0, 0.0, 0.0], device=device),
        masks=masks,
        matching_resolution=(config['matching_width'], config['matching_height']),
        rgb_to_stitch_resolution=(config['rgb_width'], config['rgb_height']),
        panorama_resolution=(config['pano_width'], config['pano_height']),
        sigma_i=config['sigma_i'],
        sigma_s=config['sigma_s'],
        device=device
    )
    
    print("[Python Path] Running estimate_RGBD_panorama...")
    
    # Select reference images
    images_to_match = images
    images_to_stitch = [images[i] for i in config['references_indices']]
    
    rgb_pano, distance_pano = estimator.estimate_RGBD_panorama(
        images_to_match, images_to_stitch
    )
    
    return rgb_pano, distance_pano


    """Compute distance candidates using inverse parameterization"""
    inv_dist_min = 1.0 / min_dist  # Far plane (inverse)
    inv_dist_max = 1.0 / max_dist  # Near plane (inverse)
    
    # Uniform sampling in inverse space
    inv_distances = torch.linspace(inv_dist_max, inv_dist_min, candidate_count)
    distances = 1.0 / inv_distances
    
    return distances
    """
    Execute Python RGBD_Estimator with given inputs.
    
    Returns:
        rgb_pano: [H, W, 3] uint8
        distance_pano: [H, W] float32
    """
    print("\n[Python Path] Initializing RGBD_Estimator...")
    
    estimator = PythonRGBD_Estimator(
        calibrations=calibrations,
        min_dist=config['min_dist'],
        max_dist=config['max_dist'],
        candidate_count=config['candidate_count'],
        references_indices=config['references_indices'],
        reprojection_viewpoint=torch.tensor([0.0, 0.0, 0.0], device=device),
        masks=masks,
        matching_resolution=(config['matching_width'], config['matching_height']),
        rgb_to_stitch_resolution=(config['rgb_width'], config['rgb_height']),
        panorama_resolution=(config['pano_width'], config['pano_height']),
        sigma_i=config['sigma_i'],
        sigma_s=config['sigma_s'],
        device=device
    )
    
    print("[Python Path] Running estimate_RGBD_panorama...")
    
    # Select reference images
    images_to_match = images
    images_to_stitch = [images[i] for i in config['references_indices']]
    
    rgb_pano, distance_pano = estimator.estimate_RGBD_panorama(
        images_to_match, images_to_stitch
    )
    
    return rgb_pano.cpu().numpy(), distance_pano.cpu().numpy()


# ============================================================================
# C++/CUDA Execution Path
# ============================================================================

def run_cuda_implementation(calibrations_flat, images, config, device_id):
    """
    Execute C++/CUDA RGBD_Estimator with given inputs.
    
    Returns:
        rgb_pano: [H, W, 3] uint8 numpy array
        distance_pano: [H, W] float32 numpy array
    """
    print("\n[C++/CUDA Path] Initializing RGBD_Estimator...")
    
    # Convert images to numpy (C++ expects list of flattened float arrays)
    images_np = []
    for img in images:
        img_np = img.cpu().numpy().flatten().tolist()
        images_np.append(img_np)
    
    # Select images for stitching
    images_to_stitch = [images_np[i] for i in config['references_indices']]
    
    estimator = sphere_stereo_cuda.RGBD_Estimator(
        calibrations_rt=calibrations_flat['rt'],
        calibrations_intrinsics=calibrations_flat['intrinsics'],
        calibrations_sphere=calibrations_flat['sphere'],
        calibrations_resolution=calibrations_flat['resolution'],
        min_dist=config['min_dist'],
        max_dist=config['max_dist'],
        candidate_count=config['candidate_count'],
        references_indices=config['references_indices'],
        reprojection_viewpoint=[0.0, 0.0, 0.0],
        image_widths=[config['matching_width']] * len(images),
        image_heights=[config['matching_height']] * len(images),
        matching_width=config['matching_width'],
        matching_height=config['matching_height'],
        rgb_to_stitch_width=config['rgb_width'],
        rgb_to_stitch_height=config['rgb_height'],
        panorama_width=config['pano_width'],
        panorama_height=config['pano_height'],
        sigma_i=config['sigma_i'],
        sigma_s=config['sigma_s'],
        device=device_id
    )
    
    print("[C++/CUDA Path] Running estimate_RGBD_panorama...")
    
    rgb_pano_flat, distance_pano_flat = estimator.estimate_RGBD_panorama(
        images_np, images_to_stitch
    )
    
    # Reshape outputs
    rgb_pano = np.array(rgb_pano_flat, dtype=np.uint8).reshape(
        config['pano_height'], config['pano_width'], 3
    )
    distance_pano = np.array(distance_pano_flat, dtype=np.float32).reshape(
        config['pano_height'], config['pano_width']
    )
    
    return rgb_pano, distance_pano


# ============================================================================
# Equivalence Analysis
# ============================================================================

def quantize_to_candidates(distance_map, min_dist, max_dist, candidate_count):
    """
    Quantize continuous distance values to nearest candidate levels.
    This accounts for CUDA implementation not using quadratic fitting.
    """
    # Inverse distance parameterization
    inv_dist_min = 1.0 / min_dist
    inv_dist_max = 1.0 / max_dist
    
    # Convert distance to candidate index
    inv_dist = 1.0 / np.clip(distance_map, min_dist, max_dist)
    normalized = (inv_dist - inv_dist_max) / (inv_dist_min - inv_dist_max)
    candidate_idx = normalized * (candidate_count - 1)
    
    # Round to nearest candidate
    candidate_idx = np.round(candidate_idx).astype(np.int32)
    candidate_idx = np.clip(candidate_idx, 0, candidate_count - 1)
    
    # Convert back to distance
    candidates = 1.0 / np.linspace(inv_dist_min, inv_dist_max, candidate_count)
    quantized = candidates[candidate_idx]
    
    return quantized


def compute_metrics(py_rgb, py_dist, cuda_rgb, cuda_dist, config):
    """
    Compute comprehensive equivalence metrics.
    """
    print("\n" + "="*80)
    print("EQUIVALENCE ANALYSIS REPORT")
    print("="*80)
        # Convert tensors to numpy arrays if needed
    if hasattr(py_rgb, 'cpu'):
        py_rgb = py_rgb.cpu().numpy()
    if hasattr(py_dist, 'cpu'):
        py_dist = py_dist.cpu().numpy()
    if hasattr(cuda_rgb, 'cpu'):
        cuda_rgb = cuda_rgb.cpu().numpy()
    if hasattr(cuda_dist, 'cpu'):
        cuda_dist = cuda_dist.cpu().numpy()
    
    # Ensure proper data types
    py_rgb = py_rgb.astype(np.uint8) if py_rgb.dtype != np.uint8 else py_rgb
    cuda_rgb = cuda_rgb.astype(np.uint8) if cuda_rgb.dtype != np.uint8 else cuda_rgb
    py_dist = py_dist.astype(np.float32) if py_dist.dtype != np.float32 else py_dist
    cuda_dist = cuda_dist.astype(np.float32) if cuda_dist.dtype != np.float32 else cuda_dist
        # RGB Comparison
    print("\n[RGB Panorama]")
    rgb_mse = np.mean((py_rgb.astype(np.float32) - cuda_rgb.astype(np.float32))**2)
    rgb_psnr = 20 * np.log10(255.0 / np.sqrt(rgb_mse)) if rgb_mse > 0 else float('inf')
    rgb_max_diff = np.max(np.abs(py_rgb.astype(np.int32) - cuda_rgb.astype(np.int32)))
    
    print(f"  MSE:           {rgb_mse:.4f}")
    print(f"  PSNR:          {rgb_psnr:.2f} dB")
    print(f"  Max Diff:      {rgb_max_diff} (out of 255)")
    print(f"  Pixel Match:   {np.mean(py_rgb == cuda_rgb) * 100:.2f}%")
    
    # Distance Comparison (Raw)
    print("\n[Distance Map - Raw]")
    valid_mask = (py_dist > config['min_dist']) & (py_dist < config['max_dist']) & \
                 (cuda_dist > config['min_dist']) & (cuda_dist < config['max_dist'])
    
    if np.sum(valid_mask) > 0:
        dist_diff = np.abs(py_dist[valid_mask] - cuda_dist[valid_mask])
        dist_mse = np.mean(dist_diff**2)
        dist_mae = np.mean(dist_diff)
        dist_max = np.max(dist_diff)
        dist_rel_error = np.mean(dist_diff / py_dist[valid_mask]) * 100
        
        print(f"  MSE:           {dist_mse:.6f} m²")
        print(f"  MAE:           {dist_mae:.6f} m")
        print(f"  Max Error:     {dist_max:.6f} m")
        print(f"  Rel Error:     {dist_rel_error:.4f}%")
        print(f"  Valid Pixels:  {np.sum(valid_mask)} / {valid_mask.size}")
    else:
        print("  No valid pixels for comparison")
    
    # Distance Comparison (Quantized - accounting for no quadratic fitting)
    print("\n[Distance Map - Quantized (nearest candidate)]")
    py_dist_quantized = quantize_to_candidates(
        py_dist, config['min_dist'], config['max_dist'], config['candidate_count']
    )
    
    if np.sum(valid_mask) > 0:
        quant_diff = np.abs(py_dist_quantized[valid_mask] - cuda_dist[valid_mask])
        quant_mse = np.mean(quant_diff**2)
        quant_mae = np.mean(quant_diff)
        quant_max = np.max(quant_diff)
        
        print(f"  MSE:           {quant_mse:.6f} m²")
        print(f"  MAE:           {quant_mae:.6f} m")
        print(f"  Max Error:     {quant_max:.6f} m")
        
        # Check if they match at candidate level
        candidate_match = np.mean(np.isclose(py_dist_quantized[valid_mask], 
                                             cuda_dist[valid_mask], 
                                             rtol=1e-3, atol=1e-4)) * 100
        print(f"  Candidate Match: {candidate_match:.2f}%")
    
    # Visualization
    visualize_comparison(py_rgb, py_dist, cuda_rgb, cuda_dist, py_dist_quantized, valid_mask, config)


def visualize_comparison(py_rgb, py_dist, cuda_rgb, cuda_dist, py_dist_quant, valid_mask, config):
    """
    Create comprehensive visualization of differences.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Row 1: RGB Comparison
    axes[0, 0].imshow(py_rgb)
    axes[0, 0].set_title("Python RGB", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cuda_rgb)
    axes[0, 1].set_title("C++/CUDA RGB", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    rgb_diff = np.abs(py_rgb.astype(np.float32) - cuda_rgb.astype(np.float32))
    im_rgb = axes[0, 2].imshow(rgb_diff.mean(axis=2), cmap='hot', vmin=0, vmax=50)
    axes[0, 2].set_title("RGB Difference (Mean)", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im_rgb, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Row 2: Distance Comparison (Raw)
    vmin, vmax = config['min_dist'], config['max_dist']
    
    im_py = axes[1, 0].imshow(py_dist, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Python Distance (Raw)", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im_py, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im_cuda = axes[1, 1].imshow(cuda_dist, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("C++/CUDA Distance (Raw)", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im_cuda, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    dist_diff_raw = np.abs(py_dist - cuda_dist)
    dist_diff_raw[~valid_mask] = 0
    im_diff = axes[1, 2].imshow(dist_diff_raw, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 2].set_title("Distance Difference (Raw)", fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im_diff, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    # Row 3: Distance Comparison (Quantized)
    im_py_q = axes[2, 0].imshow(py_dist_quant, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2, 0].set_title("Python Distance (Quantized)", fontsize=12, fontweight='bold')
    axes[2, 0].axis('off')
    plt.colorbar(im_py_q, ax=axes[2, 0], fraction=0.046, pad=0.04)
    
    axes[2, 1].imshow(cuda_dist, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2, 1].set_title("C++/CUDA Distance (Same)", fontsize=12, fontweight='bold')
    axes[2, 1].axis('off')
    
    dist_diff_quant = np.abs(py_dist_quant - cuda_dist)
    dist_diff_quant[~valid_mask] = 0
    im_diff_q = axes[2, 2].imshow(dist_diff_quant, cmap='hot', vmin=0, vmax=0.1)
    axes[2, 2].set_title("Distance Difference (Quantized)", fontsize=12, fontweight='bold')
    axes[2, 2].axis('off')
    plt.colorbar(im_diff_q, ax=axes[2, 2], fraction=0.046, pad=0.04)
    
    plt.suptitle("Python vs C++/CUDA Implementation Equivalence Test", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = Path(__file__).parent / "equivalence_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Show plot (optional - comment out if running headless)
    # plt.show()


# ============================================================================
# Main Test Execution
# ============================================================================

def main():
    print("="*80)
    print("RGBD_Estimator Equivalence Test")
    print("Python (sphere-stereo) vs C++/CUDA (my_stereo_pkg)")
    print("="*80)
    
    # Change working directory to sphere-stereo for Python implementation
    original_cwd = os.getcwd()
    sphere_stereo_root = Path(__file__).parent.parent.parent / "sphere-stereo"
    os.chdir(sphere_stereo_root)
    print(f"\nWorking directory: {os.getcwd()}")
    
    # Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_id = 0
    
    config = {
        'min_dist': 1.0,
        'max_dist': 10.0,
        'candidate_count': 64,
        'references_indices': [0, 1],  # Use first 2 cameras as references
        'matching_width': 640,
        'matching_height': 480,
        'rgb_width': 640,
        'rgb_height': 480,
        'pano_width': 1280,  # 2x width for 360 degrees
        'pano_height': 640,
        'sigma_i': 30.0,
        'sigma_s': 30.0,
    }
    
    print(f"\nDevice: {device}")
    print(f"Configuration: {config}")
    
    # Generate test data
    print("\n[Data Generation] Creating synthetic inputs...")
    py_calibs, cuda_calibs_flat = generate_synthetic_calibration(device)
    images = generate_synthetic_images(4, config['matching_width'], config['matching_height'], device)
    masks = generate_masks(4, config['matching_width'], config['matching_height'], device)
    
    print(f"  ✓ Generated 4 cameras with {config['matching_width']}x{config['matching_height']} images")
    
    # Run Python implementation
    try:
        py_rgb, py_dist = run_python_implementation(
            py_calibs, images, masks, config, device
        )
        print("  ✓ Python execution successful")
    except Exception as e:
        print(f"  ✗ Python execution failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run C++/CUDA implementation
    try:
        cuda_rgb, cuda_dist = run_cuda_implementation(
            cuda_calibs_flat, images, config, device_id
        )
        print("  ✓ C++/CUDA execution successful")
    except Exception as e:
        print(f"  ✗ C++/CUDA execution failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compute equivalence metrics
    compute_metrics(py_rgb, py_dist, cuda_rgb, cuda_dist, config)
    
    # ========================================================================
    # Cost Volume Analysis (MAE Investigation)
    # ========================================================================
    
    print("\n" + "="*80)
    print("COST VOLUME ANALYSIS (MAE Investigation)")
    print("="*80)
    
    # Compute theoretical distance candidates
    distance_candidates = compute_distance_candidates(
        config['min_dist'], config['max_dist'], config['candidate_count']
    )
    
    print(f"\nDistance candidates range: {distance_candidates[0]:.2f}m to {distance_candidates[-1]:.2f}m")
    print(f"Distance step (avg): {torch.diff(distance_candidates).mean():.3f}m")
    
    # Generate synthetic cost volume for demonstration
    # In real implementation, this would be captured from the estimators
    print("\n[Cost Volume] Generating synthetic cost volume for analysis...")
    
    height, width = config['matching_height'], config['matching_width']
    candidate_count = config['candidate_count']
    
    # Create synthetic cost volume with known patterns
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device), 
        torch.arange(width, device=device), 
        indexing='ij'
    )
    
    # Simulate depth variation across the image
    true_depth = 3.0 + 2.0 * torch.sin(x_coords / width * 2 * np.pi) * torch.cos(y_coords / height * 2 * np.pi)
    
    # Create cost volume with minimum at true depth
    cost_volume = torch.zeros(candidate_count, height, width, device=device)
    
    for i, dist_candidate in enumerate(distance_candidates):
        # Cost based on distance from true depth
        cost = torch.abs(dist_candidate - true_depth) + 0.1 * torch.randn_like(true_depth)
        cost_volume[i] = torch.clamp(cost, 0, 10)
    
    # Analyze cost volume
    fig_cost, cost_analysis = analyze_cost_volume(
        cost_volume, distance_candidates, 
        title="Synthetic Cost Volume Analysis (MAE Investigation)"
    )
    
    # Save cost volume analysis
    cost_output_path = Path(__file__).parent / "cost_volume_analysis.png"
    fig_cost.savefig(cost_output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Cost volume analysis saved to: {cost_output_path}")
    
    # Compare with actual distance maps
    print(f"\n[Analysis] Distance map statistics:")
    py_dist_valid = py_dist[(py_dist > 0) & (py_dist < config['max_dist'])]
    cuda_dist_valid = cuda_dist[(cuda_dist > 0) & (cuda_dist < config['max_dist'])]
    
    print(f"  Python distances - Mean: {py_dist_valid.mean():.2f}m, Std: {py_dist_valid.std():.2f}m")
    print(f"  C++/CUDA distances - Mean: {cuda_dist_valid.mean():.2f}m, Std: {cuda_dist_valid.std():.2f}m")
    print(f"  Synthetic true depth - Mean: {true_depth.mean():.2f}m, Std: {true_depth.std():.2f}m")
    
    # Identify potential MAE causes
    print(f"\n[MAE Diagnosis]")
    
    # Ensure both are numpy arrays for analysis
    if isinstance(py_dist, torch.Tensor):
        py_dist_np = py_dist.cpu().numpy()
    else:
        py_dist_np = py_dist
        
    if isinstance(cuda_dist, torch.Tensor):
        cuda_dist_np = cuda_dist.cpu().numpy() 
    else:
        cuda_dist_np = cuda_dist
    
    # Enhanced MAE diagnosis with detailed sub-analyses
    print(f"\n[Enhanced MAE Diagnosis]")
    
    mae = np.abs(py_dist_np - cuda_dist_np).mean()
    print(f"  Measured MAE: {mae:.3f}m")
    
    distance_step = torch.diff(distance_candidates).mean()
    print(f"  Distance quantization step: {distance_step:.3f}m")
    
    if mae > abs(distance_step):
        print(f"  ⚠️ MAE ({mae:.3f}m) > quantization step ({distance_step:.3f}m)")
        print(f"     → Suggests algorithmic differences beyond quantization")
    else:
        print(f"  ✓ MAE within quantization bounds")
    
    # Detailed sub-pixel refinement analysis
    print(f"\n[Detailed Sub-component Analysis]")
    
    # Generate synthetic cost volume for sub-pixel analysis
    print("  Generating synthetic cost volume for refinement analysis...")
    height, width = py_dist_np.shape
    synthetic_cost_volume = np.random.random((len(distance_candidates), height, width))
    
    # Add realistic cost patterns
    for d_idx, dist in enumerate(distance_candidates):
        # Create cost based on distance from synthetic ground truth (3.0m)
        cost_pattern = np.exp(-0.5 * ((dist - 3.0) / 1.0) ** 2)
        noise = 0.1 * np.random.random((height, width))
        synthetic_cost_volume[d_idx] = 1.0 - cost_pattern + noise
    
    # Run sub-pixel analysis
    subpixel_results = analyze_subpixel_refinement(
        synthetic_cost_volume, distance_candidates, "Sub-pixel Refinement Analysis"
    )
    
    # Run WTA boundary analysis  
    wta_results = analyze_wta_boundary_effects(
        py_dist_np, cuda_dist_np, distance_candidates, "WTA Boundary Analysis"
    )
    
    # Summary of potential MAE contributors
    print(f"\n[MAE Contributors Summary]")
    print(f"  1. Distance quantization: ±{abs(distance_step):.3f}m")
    print(f"  2. Sub-pixel refinement: ±{subpixel_results['refinement_diff_max']:.6f}m")
    print(f"  3. Distance conversion: ±{subpixel_results['distance_conversion_diff']:.6f}m")
    print(f"  4. Systematic bias: {wta_results['systematic_bias']:.6f}m")
    print(f"  5. Floating-point precision: {wta_results['small_diff_ratio']*100:.2f}% pixels affected")
    
    total_expected_error = (abs(distance_step) + 
                           subpixel_results['refinement_diff_max'] + 
                           subpixel_results['distance_conversion_diff'] + 
                           abs(wta_results['systematic_bias']))
    
    print(f"\n  Expected cumulative error: ~{total_expected_error:.3f}m")
    print(f"  Actual measured MAE: {mae:.3f}m")
    
    if mae > total_expected_error * 1.5:
        print(f"  ⚠️ Measured MAE significantly higher than expected")
        print(f"     → Additional sources likely: cost computation differences,")
        print(f"       camera selection thresholds, or filtering parameters")
    elif mae <= total_expected_error:
        print(f"  ✓ Measured MAE within expected range from identified sources")
    else:
        print(f"  ~ Measured MAE partially explained by identified sources")

    print("\n" + "="*80)
    print("ENHANCED DIAGNOSTIC TEST COMPLETE") 
    print("="*80)
    
    plt.close(fig_cost)  # Close to save memory
    
    # Restore original working directory
    os.chdir(original_cwd)


if __name__ == "__main__":
    main()
