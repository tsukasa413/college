#!/bin/bash
# Wrapper script to run main_eval with proper library paths

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source ROS2 setup
if [ -f "$WORKSPACE_ROOT/install/setup.bash" ]; then
    source "$WORKSPACE_ROOT/install/setup.bash"
fi

# Add LibTorch library path
TORCH_LIB_PATH=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
if [ -n "$TORCH_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB_PATH:$LD_LIBRARY_PATH"
fi

# ========================================================================
# Critical memory management for Jetson unified memory architecture
# ========================================================================
# LibTorch (CUDA) aggressively allocates ~90% of available memory on startup.
# This causes OpenCV operations (like cv::resize) to fail with segfaults
# when they cannot allocate internal buffers.
#
# Solution: Limit LibTorch's CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Force OpenCV to use CPU path if CUDA conflicts occur
export OPENCV_CUDA_FORCE_CPU_PATH=1

# Run main_eval with all arguments
exec "$WORKSPACE_ROOT/install/my_stereo_pkg/lib/my_stereo_pkg/main_eval" "$@"
