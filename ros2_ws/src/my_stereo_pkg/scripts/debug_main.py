#!/usr/bin/env python3
"""
Debug main script for my_stereo_pkg
This script demonstrates the usage of the C++ bindings with PyTorch
"""

import sys
import torch

def main():
    print("=== My Stereo Package Debug Script ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        device = 'cuda'
    else:
        print("Using CPU device")
        device = 'cpu'
    
    # Create random tensor
    print(f"\nCreating random tensor (100x100) on {device} device...")
    tensor = torch.randn(100, 100, device=device)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor device: {tensor.device}")
    print(f"Tensor dtype: {tensor.dtype}")
    
    # Import C++ module with error handling
    try:
        from my_stereo_pkg import _core_cpp
        print("\nSuccessfully imported _core_cpp module")
    except ImportError as e:
        print(f"ERROR: Failed to import _core_cpp module: {e}")
        print("Make sure the package is built and installed correctly")
        return 1
    except Exception as e:
        print(f"ERROR: Unexpected error while importing _core_cpp: {e}")
        return 1
    
    # Test connection with C++ function
    try:
        print("\nCalling _core_cpp.test_connection(tensor.cpu().numpy())...")
        _core_cpp.test_connection(tensor.cpu().numpy())
        print("✓ test_connection call successful!")
        
        print("\n=== Debug script completed successfully ===")
        
    except Exception as e:
        print(f"ERROR: Failed to call test_connection: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
    sys.exit(main())