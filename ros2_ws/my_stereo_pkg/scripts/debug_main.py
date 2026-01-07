#!/usr/bin/env python3
"""
Debug main script for my_stereo_pkg
Tests C++ bindings with PyTorch tensors
"""

import sys

def main():
    print("=" * 60)
    print("my_stereo_pkg - C++ Binding Test")
    print("=" * 60)
    
    # Step 1: Import C++ bindings first
    try:
        import my_stereo_pkg._core_cpp as core_cpp
        print("✓ C++ bindings (_core_cpp) imported successfully")
        print(f"  LibTorch support: {core_cpp.HAS_TORCH}")
    except ImportError as e:
        print(f"✗ Failed to import C++ bindings: {e}")
        print("  Make sure the package is built and installed:")
        print("  cd ~/Documents/college/ros2_ws")
        print("  colcon build --packages-select my_stereo_pkg")
        print("  source install/setup.bash")
        return 1
    
    # Step 2: If LibTorch is available in C++, test with PyTorch tensor
    if core_cpp.HAS_TORCH:
        # Import PyTorch
        try:
            import torch
            print(f"✓ PyTorch imported successfully (version: {torch.__version__})")
        except ImportError as e:
            print(f"✗ Failed to import PyTorch: {e}")
            print("  Please install PyTorch first:")
            print("  pip3 install torch")
            return 1
        
        # Check CUDA availability
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✓ CUDA is available (device: {torch.cuda.get_device_name(0)})")
        else:
            device = torch.device("cpu")
            print("⚠ CUDA is not available, using CPU instead")
        
        # Create a random tensor
        try:
            tensor = torch.rand(100, 100, device=device)
            print(f"✓ Created random tensor with shape {tensor.shape} on {tensor.device}")
        except Exception as e:
            print(f"✗ Failed to create tensor: {e}")
            return 1
        
        # Step 3: Test the connection with tensor
        print("\nCalling C++ function test_connection(tensor)...")
        print("-" * 60)
        try:
            core_cpp.test_connection(tensor)
            print("-" * 60)
            print("✓ C++ function executed successfully!")
        except Exception as e:
            print(f"✗ Failed to call C++ function: {e}")
            return 1
    else:
        # LibTorch not available, call dummy function
        print("\n⚠ LibTorch is not compiled in C++ bindings")
        print("Calling dummy test_connection()...")
        print("-" * 60)
        try:
            core_cpp.test_connection()
            print("-" * 60)
            print("✓ C++ function executed (without tensor support)")
        except Exception as e:
            print(f"✗ Failed to call C++ function: {e}")
            return 1
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    return 0

if __name__ == '__main__':
    sys.exit(main())
