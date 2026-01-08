#!/usr/bin/env python3
"""
Debug main script for my_stereo_pkg
This script demonstrates the usage of the C++ bindings (basic test)
"""

import sys

def main():
    print("=== My Stereo Package Debug Script (Basic Test) ===")
    print(f"Python version: {sys.version}")
    
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
        print("\nCalling _core_cpp.test_connection()...")
        _core_cpp.test_connection()
        print("✓ test_connection call successful!")
        
        print("\n=== Debug script completed successfully ===")
        
    except Exception as e:
        print(f"ERROR: Failed to call test_connection: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
    sys.exit(main())