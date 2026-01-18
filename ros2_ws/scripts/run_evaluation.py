#!/usr/bin/env python3
"""
Evaluation script to run both Python and C++ implementations
and compare their results.

This script:
1. Runs the Python version to generate reference outputs
2. Runs the C++ version (main_eval) to validate against Python
3. Compares accuracy and performance metrics
"""

import sys
import os
import subprocess
import argparse
import time
import re
from pathlib import Path

def run_python_reference(dataset_path, args):
    """Run the Python version to generate reference outputs"""
    print("=" * 60)
    print("Step 1: Running Python reference implementation")
    print("=" * 60)
    
    # Path to original Python main.py
    sphere_stereo_dir = Path(__file__).parent.parent.parent / "sphere-stereo"
    python_script = sphere_stereo_dir / "python" / "main.py"
    
    if not python_script.exists():
        print(f"Error: Python script not found: {python_script}")
        return False, 0.0
    
    # Make dataset_path absolute
    dataset_path = str(Path(dataset_path).resolve())
    
    cmd = [
        "python3",
        str(python_script),
        "--dataset_path", dataset_path,
        "--references_indices", "0", "2",
        "--min_dist", str(args.min_dist),
        "--max_dist", str(args.max_dist),
        "--candidate_count", str(args.candidate_count),
        "--sigma_i", str(args.sigma_i),
        "--sigma_s", str(args.sigma_s),
        "--matching_resolution", str(args.matching_resolution[0]), str(args.matching_resolution[1]),
        "--rgb_to_stitch_resolution", str(args.rgb_to_stitch_resolution[0]), str(args.rgb_to_stitch_resolution[1]),
        "--panorama_resolution", str(args.panorama_resolution[0]), str(args.panorama_resolution[1]),
        "--device", "cuda:0",
        "--saving", "True",
        "--visualize", "False",
        "--evaluate", "False"
    ]
    
    print("Command:", " ".join(cmd))
    print("Working directory:", str(sphere_stereo_dir))
    print()
    
    # Measure execution time
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(sphere_stereo_dir))
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000.0
    
    if result.returncode != 0:
        print("Error: Python reference implementation failed")
        print(result.stderr)
        return False, 0.0
    
    print(f"\n✓ Python execution completed in {elapsed_ms:.1f} ms")
    print("Python reference outputs saved to:", os.path.join(dataset_path, "output"))
    return True, elapsed_ms

def run_cpp_evaluation(dataset_path, args):
    """Run the C++ evaluation program"""
    print("\n" + "=" * 60)
    print("Step 2: Running C++ evaluation")
    print("=" * 60)
    
    # Path to C++ executable (in install directory after colcon build)
    workspace_path = Path(__file__).parent.parent
    cpp_executable = workspace_path / "install" / "my_stereo_pkg" / "lib" / "my_stereo_pkg" / "main_eval"
    
    if not cpp_executable.exists():
        print(f"Error: C++ executable not found: {cpp_executable}")
        print("Did you build the package with: colcon build --packages-select my_stereo_pkg")
        return False, 0.0, {}
    
    # Set up environment (LibTorch library path and memory limits)
    env = os.environ.copy()
    
    # Add LibTorch library path
    try:
        import torch
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if 'LD_LIBRARY_PATH' in env:
            env['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{env['LD_LIBRARY_PATH']}"
        else:
            env['LD_LIBRARY_PATH'] = torch_lib_path
    except ImportError:
        print("Warning: Could not find torch library path")
    
    # Set memory management environment variables for Jetson
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    env['OPENCV_CUDA_FORCE_CPU_PATH'] = '1'
    
    cmd = [
        str(cpp_executable),
        "--dataset_path", dataset_path,
        "--min_dist", str(args.min_dist),
        "--max_dist", str(args.max_dist)
    ]
    
    if args.visualize:
        cmd.append("--visualize")
    
    print("Command:", " ".join(cmd))
    print()
    
    # Measure execution time
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000.0
    
    if result.returncode != 0:
        print("Error: C++ evaluation failed")
        print(result.stderr)
        return False, 0.0, {}
    
    # Print C++ output
    print(result.stdout)
    
    # Parse C++ output to extract metrics
    metrics = {}
    try:
        # Extract C++ execution time (GPU kernel time)
        cpp_gpu_match = re.search(r'C\+\+ Execution Time:\s+([\d.]+)\s+ms', result.stdout)
        if cpp_gpu_match:
            metrics['cpp_gpu_time'] = float(cpp_gpu_match.group(1))
        
        # Extract accuracy metrics
        rgb_mae_match = re.search(r'RGB Mean Absolute Error:\s+([\d.]+)', result.stdout)
        if rgb_mae_match:
            metrics['rgb_mae'] = float(rgb_mae_match.group(1))
        
        depth_mae_match = re.search(r'Depth Mean Absolute Error:\s+([\d.]+)', result.stdout)
        if depth_mae_match:
            metrics['depth_mae'] = float(depth_mae_match.group(1))
        
        match_rate_match = re.search(r'Match Rate \(Tolerance\):\s+([\d.]+)\s+%', result.stdout)
        if match_rate_match:
            metrics['match_rate'] = float(match_rate_match.group(1))
    except Exception as e:
        print(f"Warning: Could not parse C++ metrics: {e}")
    
    print(f"\n✓ C++ evaluation completed in {elapsed_ms:.1f} ms (total process time)")
    return True, elapsed_ms, metrics

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate C++ implementation against Python reference"
    )
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        default="evaluation_dataset",
        help="Path to dataset directory"
    )
    parser.add_argument(
        '--min_dist',
        type=float,
        default=0.55,
        help="Minimum distance for sphere sweep"
    )
    parser.add_argument(
        '--max_dist',
        type=float,
        default=100.0,
        help="Maximum distance for sphere sweep"
    )
    parser.add_argument(
        '--candidate_count',
        type=int,
        default=32,
        help="Number of distance candidates"
    )
    parser.add_argument(
        '--sigma_i',
        type=float,
        default=10.0,
        help="ISB filter sigma_i parameter"
    )
    parser.add_argument(
        '--sigma_s',
        type=float,
        default=25.0,
        help="ISB filter sigma_s parameter"
    )
    parser.add_argument(
        '--matching_resolution',
        nargs=2,
        type=int,
        default=[1024, 1024],
        help="Resolution for matching (width height)"
    )
    parser.add_argument(
        '--rgb_to_stitch_resolution',
        nargs=2,
        type=int,
        default=[1216, 1216],
        help="Resolution for RGB stitching (width height)"
    )
    parser.add_argument(
        '--panorama_resolution',
        nargs=2,
        type=int,
        default=[2048, 1024],
        help="Output panorama resolution (width height)"
    )
    parser.add_argument(
        '--skip_python',
        action='store_true',
        help="Skip Python reference generation (use existing outputs)"
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help="Enable visualization in C++ evaluation"
    )
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.isdir(args.dataset_path):
        print(f"Error: Dataset directory not found: {args.dataset_path}")
        return 1
    
    # Check if calibration.json exists
    calib_path = os.path.join(args.dataset_path, "calibration.json")
    if not os.path.isfile(calib_path):
        print(f"Error: calibration.json not found in {args.dataset_path}")
        return 1
    
    print("=" * 60)
    print("C++ vs Python Evaluation Pipeline")
    print("=" * 60)
    print(f"Dataset: {args.dataset_path}")
    print(f"Distance range: [{args.min_dist}, {args.max_dist}]")
    print(f"Candidates: {args.candidate_count}")
    print(f"Matching resolution: {args.matching_resolution}")
    print(f"Panorama resolution: {args.panorama_resolution}")
    print()
    
    # Step 1: Run Python reference (unless skipped)
    python_time_ms = 0.0
    if not args.skip_python:
        success, python_time_ms = run_python_reference(args.dataset_path, args)
        if not success:
            return 1
    else:
        print("Skipping Python reference generation (using existing outputs)")
        output_dir = os.path.join(args.dataset_path, "output")
        if not os.path.isdir(output_dir):
            print(f"Error: Output directory not found: {output_dir}")
            print("Run without --skip_python to generate reference outputs")
            return 1
    
    # Step 2: Run C++ evaluation
    success, cpp_time_ms, metrics = run_cpp_evaluation(args.dataset_path, args)
    if not success:
        return 1
    
    # ====================================================================
    # Print Final Comparison Report
    # ====================================================================
    print("\n" + "=" * 70)
    print(" " * 20 + "EVALUATION SUMMARY")
    print("=" * 70)
    print()
    
    print("--------------------------------------------------")
    print("[Performance Comparison]")
    print("--------------------------------------------------")
    if python_time_ms > 0.0:
        speedup = python_time_ms / cpp_time_ms if cpp_time_ms > 0 else 0
        print(f"  Python Execution Time:  {python_time_ms:8.1f} ms  ({python_time_ms/1000:.2f} s)")
        print(f"  C++ Execution Time:     {cpp_time_ms:8.1f} ms  ({cpp_time_ms/1000:.2f} s)")
        if metrics.get('cpp_gpu_time'):
            print(f"  C++ GPU Kernel Time:    {metrics['cpp_gpu_time']:8.1f} ms  ({metrics['cpp_gpu_time']/1000:.2f} s)")
        print()
        print(f"  Speedup:                   {speedup:5.1f} x  (C++ is {speedup:.1f} times faster)")
    else:
        print(f"  C++ Execution Time:     {cpp_time_ms:8.1f} ms  ({cpp_time_ms/1000:.2f} s)")
        if metrics.get('cpp_gpu_time'):
            print(f"  C++ GPU Kernel Time:    {metrics['cpp_gpu_time']:8.1f} ms  ({metrics['cpp_gpu_time']/1000:.2f} s)")
        print("  Python time: Not measured (--skip_python was used)")
    print()
    
    print("--------------------------------------------------")
    print("[Accuracy Comparison]")
    print("--------------------------------------------------")
    if metrics:
        if 'rgb_mae' in metrics:
            print(f"  RGB Mean Absolute Error:   {metrics['rgb_mae']:6.1f}")
        if 'depth_mae' in metrics:
            print(f"  Depth Mean Absolute Error: {metrics['depth_mae']:6.4f}")
        if 'match_rate' in metrics:
            print(f"  Match Rate (Tolerance):    {metrics['match_rate']:6.1f} %")
    else:
        print("  Metrics not available")
    print()
    
    print("--------------------------------------------------")
    print("[Output Files]")
    print("--------------------------------------------------")
    output_dir = os.path.join(args.dataset_path, "output") if os.path.isdir(os.path.join(args.dataset_path, "output")) else "ros2_ws/output"
    print(f"  All outputs saved to: {output_dir}")
    print()
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
