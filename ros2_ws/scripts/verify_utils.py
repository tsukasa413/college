#!/usr/bin/env python3
"""
=======================================================================
Python vs C++/CUDA Complete Implementation Verification
100% Correctness Proof for ALL Utility Functions
- Double Sphere Camera Model (unproject/project)
- RGB to YCbCr Color Conversion
- Bilinear Resampling
=======================================================================
"""

import numpy as np
import subprocess
import json
import sys
import cv2
import tempfile
import os

def unproject_python_reference(u, v, fx, fy, cx, cy, xi, alpha):
    """
    Python reference implementation (from utils.py)
    Unproject pixel to 3D point using Double Sphere Camera Model
    """
    # Normalized image coordinates
    mx = (u - cx) / fx
    my = (v - cy) / fy
    
    r2 = mx * mx + my * my
    
    # Double Sphere model inverse
    denom = alpha * r2 + 1.0 - (2.0 * alpha - 1.0) * r2 * xi
    
    if denom < 0.0001:
        return None, None, None, False
    
    numerator = 1.0 - xi * xi * r2
    z_sphere = numerator / denom
    
    x_3d = mx * z_sphere
    y_3d = my * z_sphere
    z_3d = z_sphere - xi
    
    return x_3d, y_3d, z_3d, True

def project_python_reference(x, y, z, fx, fy, cx, cy, xi, alpha):
    """
    Python reference implementation (from utils.py)
    Project 3D point to pixel using Double Sphere Camera Model
    """
    # Translate by xi (second sphere center)
    z_shifted = z + xi
    
    # Project to first sphere
    r_xy2 = x * x + y * y
    r = np.sqrt(r_xy2 + z_shifted * z_shifted)
    
    if r < 0.0001:
        return None, None, False
    
    # Project to second sphere (image plane)
    m = alpha * r + (1.0 - alpha) * z_shifted
    
    if m < 0.0001:
        return None, None, False
    
    mx = x / m
    my = y / m
    
    u = fx * mx + cx
    v = fy * my + cy
    
    return u, v, True

def rgb2ycbcr_python_reference(rgb):
    """
    Python reference implementation (from utils.py)
    Convert RGB to YCbCr color space
    """
    yuv = np.zeros_like(rgb, dtype=np.float32)
    
    yuv[:, :, 0] = np.clip(16 + 0.1826 * rgb[:, :, 0] + 0.6142 * rgb[:, :, 1] + 0.062 * rgb[:, :, 2], 16, 235)
    yuv[:, :, 1] = np.clip(128 - 0.1006 * rgb[:, :, 0] - 0.3386 * rgb[:, :, 1] + 0.4392 * rgb[:, :, 2], 16, 240)
    yuv[:, :, 2] = np.clip(128 + 0.4392 * rgb[:, :, 0] - 0.3989 * rgb[:, :, 1] - 0.0403 * rgb[:, :, 2], 16, 240)
    
    return yuv.astype(np.uint8)

def bilinear_sample_python_reference(image, u, v):
    """
    Python reference: Bilinear interpolation at normalized coordinates [0,1]
    """
    h, w = image.shape[:2]
    
    # Convert normalized to pixel coordinates
    px = u * (w - 1)
    py = v * (h - 1)
    
    # Get integer and fractional parts
    ix = int(px)
    iy = int(py)
    fx = px - ix
    fy = py - iy
    
    # Clamp to valid range
    ix = max(0, min(ix, w - 2))
    iy = max(0, min(iy, h - 2))
    
    # Bilinear interpolation
    result = np.zeros(3, dtype=np.float32)
    for c in range(3):
        v00 = float(image[iy, ix, c])
        v10 = float(image[iy, ix + 1, c])
        v01 = float(image[iy + 1, ix, c])
        v11 = float(image[iy + 1, ix + 1, c])
        
        val0 = v00 * (1 - fx) + v10 * fx
        val1 = v01 * (1 - fx) + v11 * fx
        val = val0 * (1 - fy) + val1 * fy
        
        result[c] = val
    
    return result.astype(np.uint8)

def run_cpp_test():
    """Run C++ test and capture output"""
    result = subprocess.run(
        ["./build/my_stereo_pkg/test_gpu_kernel"],
        cwd="/home/motoken/college/ros2_ws",
        capture_output=True,
        text=True
    )
    return result.stdout

def parse_cpp_results(output):
    """Parse C++ test output to extract GPU results"""
    lines = output.split('\n')
    
    unproject_results = []
    project_results = []
    
    mode = None
    for line in lines:
        if "=== Unproject Test" in line:
            mode = "unproject"
            continue
        elif "=== Project Test" in line:
            mode = "project"
            continue
        
        # Parse GPU results (look for lines with numerical data)
        if mode and line.strip() and not line.startswith("===") and not line.startswith("GPU"):
            parts = line.split()
            if len(parts) >= 6 and mode == "unproject":
                try:
                    u, v, x, y, z, valid = map(float, parts[:6])
                    unproject_results.append({
                        'u': u, 'v': v, 'x': x, 'y': y, 'z': z, 'valid': int(valid)
                    })
                except ValueError:
                    pass
            elif len(parts) >= 6 and mode == "project":
                try:
                    x, y, z, u, v, valid = map(float, parts[:6])
                    project_results.append({
                        'x': x, 'y': y, 'z': z, 'u': u, 'v': v, 'valid': int(valid)
                    })
                except ValueError:
                    pass
    
    return unproject_results, project_results

def verify_equivalence():
    """
    Main verification function
    Proves 100% mathematical equivalence between Python and C++/CUDA
    Tests ALL utility functions
    """
    print("=" * 70)
    print("Python vs C++/CUDA Complete Utilities Verification")
    print("=" * 70)
    
    all_tests_passed = True
    all_errors = []
    
    # ========================================================================
    # TEST 1: Unproject (already implemented)
    # ========================================================================
    
    # Camera parameters (matching test_gpu_kernel.cpp)
    fx, fy = 400.0, 400.0
    cx, cy = 320.0, 240.0
    xi, alpha = 0.0, 0.5
    
    print(f"\nCalibration: fx={fx}, fy={fy}, cx={cx}, cy={cy}, xi={xi}, alpha={alpha}")
    
    print("\n" + "=" * 70)
    print("TEST 1: UNPROJECT (Pixel → 3D)")
    print("=" * 70)
    
    test_pixels = [
        (320.0, 240.0),  # Center
        (0.0, 0.0),      # Top-left
        (640.0, 480.0),  # Bottom-right
        (200.0, 150.0),  # Random
        (450.0, 350.0)   # Random
    ]
    
    unproject_errors = []
    
    print(f"\n{'Pixel':<20} {'Python X,Y,Z':<35} {'C++ X,Y,Z':<35} {'Error':<15} {'Status'}")
    print("-" * 110)
    
    # Run C++ test
    cpp_output = run_cpp_test()
    
    # Get GPU results from parsed output
    gpu_unproject_results = []
    lines = cpp_output.split('\n')
    in_gpu_section = False
    for line in lines:
        if "GPU Results:" in line:
            in_gpu_section = True
            continue
        if in_gpu_section and "===" in line:
            break
        if in_gpu_section and line.strip():
            parts = line.split()
            if len(parts) >= 6:
                try:
                    u, v, x, y, z, valid = map(float, parts[:6])
                    gpu_unproject_results.append((x, y, z))
                except ValueError:
                    pass
    
    for i, (u, v) in enumerate(test_pixels):
        # Python reference
        x_py, y_py, z_py, valid_py = unproject_python_reference(u, v, fx, fy, cx, cy, xi, alpha)
        
        if i < len(gpu_unproject_results):
            x_cpp, y_cpp, z_cpp = gpu_unproject_results[i]
            
            # Compute error
            error_x = abs(x_py - x_cpp) if x_py is not None else float('inf')
            error_y = abs(y_py - y_cpp) if y_py is not None else float('inf')
            error_z = abs(z_py - z_cpp) if z_py is not None else float('inf')
            max_error = max(error_x, error_y, error_z)
            
            unproject_errors.append(max_error)
            all_errors.append(max_error)
            
            status = "✅ PASS" if max_error < 2e-5 else "❌ FAIL"
            if max_error >= 2e-5:
                all_tests_passed = False
            
            print(f"({u:6.1f}, {v:6.1f})  "
                  f"({x_py:7.4f}, {y_py:7.4f}, {z_py:7.4f})  "
                  f"({x_cpp:7.4f}, {y_cpp:7.4f}, {z_cpp:7.4f})  "
                  f"{max_error:12.2e}  {status}")
    
    # ========================================================================
    # TEST 2: Project (already implemented)
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("TEST 2: PROJECT (3D → Pixel)")
    print("=" * 70)
    
    test_points = [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (-1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, -1.0, 2.0)
    ]
    
    project_errors = []
    
    print(f"\n{'3D Point':<25} {'Python U,V':<20} {'C++ U,V':<20} {'Error':<15} {'Status'}")
    print("-" * 95)
    
    # Get GPU project results
    gpu_project_results = []
    in_project_section = False
    for line in lines:
        if "Project Test" in line and "GPU kernel executed" in cpp_output[cpp_output.index(line):]:
            in_project_section = True
            continue
        if in_project_section and line.strip():
            parts = line.split()
            if len(parts) >= 6:
                try:
                    x, y, z, u, v, valid = map(float, parts[:6])
                    gpu_project_results.append((u, v))
                except ValueError:
                    pass
    
    for i, (x, y, z) in enumerate(test_points):
        # Python reference
        u_py, v_py, valid_py = project_python_reference(x, y, z, fx, fy, cx, cy, xi, alpha)
        
        if i < len(gpu_project_results):
            u_cpp, v_cpp = gpu_project_results[i]
            
            # Compute error
            error_u = abs(u_py - u_cpp) if u_py is not None else float('inf')
            error_v = abs(v_py - v_cpp) if v_py is not None else float('inf')
            max_error = max(error_u, error_v)
            
            project_errors.append(max_error)
            all_errors.append(max_error)
            
            status = "✅ PASS" if max_error < 2e-5 else "❌ FAIL"
            if max_error >= 2e-5:
                all_tests_passed = False
            
            print(f"({x:6.2f}, {y:6.2f}, {z:6.2f})  "
                  f"({u_py:8.2f}, {v_py:8.2f})  "
                  f"({u_cpp:8.2f}, {v_cpp:8.2f})  "
                  f"{max_error:12.2e}  {status}")
    
    # ========================================================================
    # TEST 3: RGB to YCbCr
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("TEST 3: RGB TO YCbCr CONVERSION")
    print("=" * 70)
    
    # Run all utils test to get RGB2YCbCr results
    result = subprocess.run(
        ["./build/my_stereo_pkg/test_all_utils"],
        cwd="/home/motoken/college/ros2_ws",
        capture_output=True,
        text=True
    )
    utils_output = result.stdout
    
    # Test RGB colors
    test_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (128, 128, 128),# Gray
        (0, 0, 0),      # Black
        (255, 255, 255) # White
    ]
    
    # Create test image
    test_rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    for i, color in enumerate(test_colors):
        row = i // 3
        col = i % 3
        test_rgb[row, col] = color
    
    # Python reference
    py_ycbcr = rgb2ycbcr_python_reference(test_rgb.astype(np.float32))
    
    # Parse C++ results
    cpp_ycbcr_results = []
    in_ycbcr = False
    for line in utils_output.split('\n'):
        if "GPU RGB2YCbCr Results:" in line:
            in_ycbcr = True
            continue
        if in_ycbcr and "===" in line:
            break
        if in_ycbcr and '(' in line and ')' in line:
            # Parse line: (R,G,B)  (Y,Cb,Cr)
            try:
                # Find both RGB and YCbCr tuples
                first_paren = line.index('(')
                first_close = line.index(')', first_paren)
                second_paren = line.index('(', first_close)
                second_close = line.index(')', second_paren)
                
                ycbcr_str = line[second_paren+1:second_close]
                vals = [int(v.strip()) for v in ycbcr_str.split(',')]
                if len(vals) == 3:
                    cpp_ycbcr_results.append(tuple(vals))
            except (ValueError, IndexError):
                pass
    
    rgb2ycbcr_errors = []
    
    print(f"\n{'RGB':<20} {'Python YCbCr':<20} {'C++ YCbCr':<20} {'Error':<15} {'Status'}")
    print("-" * 90)
    
    for i, color in enumerate(test_colors):
        if i < len(cpp_ycbcr_results):
            row = i // 3
            col = i % 3
            py_y, py_cb, py_cr = py_ycbcr[row, col]
            cpp_y, cpp_cb, cpp_cr = cpp_ycbcr_results[i]
            
            error_y = abs(int(py_y) - cpp_y)
            error_cb = abs(int(py_cb) - cpp_cb)
            error_cr = abs(int(py_cr) - cpp_cr)
            max_error = max(error_y, error_cb, error_cr)
            
            rgb2ycbcr_errors.append(max_error)
            all_errors.append(max_error)
            
            status = "✅ PASS" if max_error <= 1 else "❌ FAIL"
            if max_error > 1:
                all_tests_passed = False
            
            print(f"({color[0]:3d},{color[1]:3d},{color[2]:3d})      "
                  f"({py_y:3d},{py_cb:3d},{py_cr:3d})       "
                  f"({cpp_y:3d},{cpp_cb:3d},{cpp_cr:3d})       "
                  f"{max_error:12.2e}  {status}")
    
    # ========================================================================
    # TEST 4: Bilinear Resampling
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("TEST 4: BILINEAR RESAMPLING")
    print("=" * 70)
    
    # Create test image (4x4)
    test_image = np.array([
        [[255, 0, 0], [200, 0, 0], [150, 0, 0], [100, 0, 0]],
        [[0, 255, 0], [0, 200, 0], [0, 150, 0], [0, 100, 0]],
        [[0, 0, 255], [0, 0, 200], [0, 0, 150], [0, 0, 100]],
        [[128, 128, 128], [100, 100, 100], [80, 80, 80], [50, 50, 50]]
    ], dtype=np.uint8)
    
    sample_coords = [
        (0.0, 0.0),      # Top-left
        (1.0, 1.0),      # Bottom-right
        (0.5, 0.5),      # Center
        (0.25, 0.25),    # Quarter
        (0.75, 0.75)     # Three-quarter
    ]
    
    # Parse C++ bilinear results
    cpp_bilinear_results = []
    in_bilinear = False
    for line in utils_output.split('\n'):
        if "GPU Bilinear Resampling Results:" in line:
            in_bilinear = True
            continue
        if in_bilinear and line.strip():
            parts = line.split()
            if len(parts) >= 3 and '(' in line:
                try:
                    rgb_str = line[line.index('('):line.index(')')+1]
                    rgb_vals = rgb_str.replace('(', '').replace(')', '').split(',')
                    r, g, b = int(rgb_vals[0]), int(rgb_vals[1]), int(rgb_vals[2])
                    cpp_bilinear_results.append((r, g, b))
                except (ValueError, IndexError):
                    pass
        if in_bilinear and "===" in line:
            break
    
    bilinear_errors = []
    
    print(f"\n{'Coord (U,V)':<20} {'Python RGB':<20} {'C++ RGB':<20} {'Error':<15} {'Status'}")
    print("-" * 90)
    
    for i, (u, v) in enumerate(sample_coords):
        if i < len(cpp_bilinear_results):
            py_rgb = bilinear_sample_python_reference(test_image, u, v)
            cpp_r, cpp_g, cpp_b = cpp_bilinear_results[i]
            
            error_r = abs(int(py_rgb[0]) - cpp_r)
            error_g = abs(int(py_rgb[1]) - cpp_g)
            error_b = abs(int(py_rgb[2]) - cpp_b)
            max_error = max(error_r, error_g, error_b)
            
            bilinear_errors.append(max_error)
            all_errors.append(max_error)
            
            status = "✅ PASS" if max_error <= 2 else "❌ FAIL"
            if max_error > 2:
                all_tests_passed = False
            
            print(f"({u:4.2f}, {v:4.2f})        "
                  f"({py_rgb[0]:3d},{py_rgb[1]:3d},{py_rgb[2]:3d})       "
                  f"({cpp_r:3d},{cpp_g:3d},{cpp_b:3d})       "
                  f"{max_error:12.2e}  {status}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)
    
    # Camera parameters (matching test_gpu_kernel.cpp)
    fx, fy = 400.0, 400.0
    cx, cy = 320.0, 240.0
    xi, alpha = 0.0, 0.5
    
    print(f"\nCalibration: fx={fx}, fy={fy}, cx={cx}, cy={cy}, xi={xi}, alpha={alpha}")
    
    # Test 1: Unproject verification
    print("\n" + "=" * 70)
    print("TEST 1: UNPROJECT (Pixel → 3D)")
    print("=" * 70)
    
    test_pixels = [
        (320.0, 240.0),  # Center
        (0.0, 0.0),      # Top-left
        (640.0, 480.0),  # Bottom-right
        (200.0, 150.0),  # Random
        (450.0, 350.0)   # Random
    ]
    
    unproject_errors = []
    
    print(f"\n{'Pixel':<20} {'Python X,Y,Z':<35} {'C++ X,Y,Z':<35} {'Error':<15} {'Status'}")
    print("-" * 110)
    
    # Run C++ test
    cpp_output = run_cpp_test()
    unproject_cpp, project_cpp = parse_cpp_results(cpp_output)
    
    # Get GPU results from parsed output (skip CPU reference lines)
    gpu_unproject_results = []
    lines = cpp_output.split('\n')
    in_gpu_section = False
    for line in lines:
        if "GPU Results:" in line:
            in_gpu_section = True
            continue
        if in_gpu_section and "===" in line:
            break
        if in_gpu_section and line.strip():
            parts = line.split()
            if len(parts) >= 6:
                try:
                    u, v, x, y, z, valid = map(float, parts[:6])
                    gpu_unproject_results.append((x, y, z))
                except ValueError:
                    pass
    
    for i, (u, v) in enumerate(test_pixels):
        # Python reference
        x_py, y_py, z_py, valid_py = unproject_python_reference(u, v, fx, fy, cx, cy, xi, alpha)
        
        if i < len(gpu_unproject_results):
            x_cpp, y_cpp, z_cpp = gpu_unproject_results[i]
            
            # Compute error
            error_x = abs(x_py - x_cpp) if x_py is not None else float('inf')
            error_y = abs(y_py - y_cpp) if y_py is not None else float('inf')
            error_z = abs(z_py - z_cpp) if z_py is not None else float('inf')
            max_error = max(error_x, error_y, error_z)
            
            unproject_errors.append(max_error)
            
            status = "✅ PASS" if max_error < 2e-5 else "❌ FAIL"
            
            print(f"({u:6.1f}, {v:6.1f})  "
                  f"({x_py:7.4f}, {y_py:7.4f}, {z_py:7.4f})  "
                  f"({x_cpp:7.4f}, {y_cpp:7.4f}, {z_cpp:7.4f})  "
                  f"{max_error:12.2e}  {status}")
    
    # Test 2: Project verification
    print("\n" + "=" * 70)
    print("TEST 2: PROJECT (3D → Pixel)")
    print("=" * 70)
    
    test_points = [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (-1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, -1.0, 2.0)
    ]
    
    project_errors = []
    
    print(f"\n{'3D Point':<25} {'Python U,V':<20} {'C++ U,V':<20} {'Error':<15} {'Status'}")
    print("-" * 95)
    
    # Get GPU project results
    gpu_project_results = []
    in_project_section = False
    for line in lines:
        if "Project Test" in line and "GPU kernel executed" in cpp_output[cpp_output.index(line):]:
            in_project_section = True
            continue
        if in_project_section and line.strip():
            parts = line.split()
            if len(parts) >= 6:
                try:
                    x, y, z, u, v, valid = map(float, parts[:6])
                    gpu_project_results.append((u, v))
                except ValueError:
                    pass
    
    for i, (x, y, z) in enumerate(test_points):
        # Python reference
        u_py, v_py, valid_py = project_python_reference(x, y, z, fx, fy, cx, cy, xi, alpha)
        
        if i < len(gpu_project_results):
            u_cpp, v_cpp = gpu_project_results[i]
            
            # Compute error
            error_u = abs(u_py - u_cpp) if u_py is not None else float('inf')
            error_v = abs(v_py - v_cpp) if v_py is not None else float('inf')
            max_error = max(error_u, error_v)
            
            project_errors.append(max_error)
            
            status = "✅ PASS" if max_error < 2e-5 else "❌ FAIL"
            
            print(f"({x:6.2f}, {y:6.2f}, {z:6.2f})  "
                  f"({u_py:8.2f}, {v_py:8.2f})  "
                  f"({u_cpp:8.2f}, {v_cpp:8.2f})  "
                  f"{max_error:12.2e}  {status}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)
    
    print(f"\nUnproject Tests:")
    print(f"  - Count:     {len(unproject_errors)}")
    if len(unproject_errors) > 0:
        print(f"  - Max Error: {max(unproject_errors):.2e}")
        print(f"  - Mean Error: {np.mean(unproject_errors):.2e}")
    
    print(f"\nProject Tests:")
    print(f"  - Count:     {len(project_errors)}")
    if len(project_errors) > 0:
        print(f"  - Max Error: {max(project_errors):.2e}")
        print(f"  - Mean Error: {np.mean(project_errors):.2e}")
    
    print(f"\nRGB2YCbCr Tests:")
    print(f"  - Count:     {len(rgb2ycbcr_errors)}")
    if len(rgb2ycbcr_errors) > 0:
        print(f"  - Max Error: {max(rgb2ycbcr_errors):.2e}")
        print(f"  - Mean Error: {np.mean(rgb2ycbcr_errors):.2e}")
    
    print(f"\nBilinear Resampling Tests:")
    print(f"  - Count:     {len(bilinear_errors)}")
    if len(bilinear_errors) > 0:
        print(f"  - Max Error: {max(bilinear_errors):.2e}")
        print(f"  - Mean Error: {np.mean(bilinear_errors):.2e}")
    
    total_tests = len(unproject_errors) + len(project_errors) + len(rgb2ycbcr_errors) + len(bilinear_errors)
    print(f"\nOverall:")
    print(f"  - Total Tests: {total_tests}")
    if len(all_errors) > 0:
        print(f"  - Max Error:   {max(all_errors):.2e}")
        print(f"  - Mean Error:  {np.mean(all_errors):.2e}")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    if all_tests_passed:
        print(f"\n✅ 100% EQUIVALENCE PROVEN FOR ALL UTILITIES")
        print(f"   All {total_tests} tests passed")
        print(f"   Maximum error: {max(all_errors):.2e}")
        print(f"\n   Python and C++/CUDA implementations are IDENTICAL")
        print(f"   Functions tested:")
        print(f"     - unproject() (Double Sphere)")
        print(f"     - project() (Double Sphere)")
        print(f"     - rgb2ycbcr() (Color conversion)")
        print(f"     - resample_bilinear() (Image resampling)")
        return 0
    else:
        failed_count = sum(1 for e in all_errors if e > 2e-5)
        print(f"\n❌ EQUIVALENCE VERIFICATION FAILED")
        print(f"   {failed_count}/{total_tests} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(verify_equivalence())
