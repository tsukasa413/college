"""my_stereo_pkg - Stereo vision package with PyTorch integration"""

__version__ = "0.0.0"

# C++バインディングはサブモジュールとしてアクセス
# from my_stereo_pkg._core_cpp import test_connection

try:
    from . import _core_cpp
    __all__ = ['_core_cpp']
except ImportError as e:
    print(f"Warning: Could not import C++ bindings (_core_cpp): {e}")
    print("The package will work with limited functionality.")
    __all__ = []
