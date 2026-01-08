# my_stereo_pkg Python package
from . import _core_cpp

__version__ = "0.0.0"
__author__ = "motoken"
__email__ = "tttsukasa0413@gmail.com"

# Expose C++ functions to Python
__all__ = ['_core_cpp']