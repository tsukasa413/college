#!/bin/bash
# Wrapper script to run verification with sphere-stereo virtual environment

set -e

echo "========================================================================"
echo "Stitcher Verification: Python (.venv) vs C++ Implementation"
echo "========================================================================"
echo ""

# Activate sphere-stereo virtual environment
SPHERE_STEREO_DIR="/home/motoken/college/sphere-stereo"
VENV_ACTIVATE="${SPHERE_STEREO_DIR}/.venv/bin/activate"

if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "❌ Error: Virtual environment not found at ${VENV_ACTIVATE}"
    exit 1
fi

echo "🐍 Activating sphere-stereo virtual environment..."
source "$VENV_ACTIVATE"
echo "   ✅ Virtual environment activated"
echo "   Python: $(which python3)"
echo "   PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "   CuPy: $(python3 -c 'import cupy; print(cupy.__version__)')"
echo ""

# Source ROS 2 workspace (for C++ bindings)
ROS2_WS="/home/motoken/college/ros2_ws"
if [ -f "${ROS2_WS}/install/setup.bash" ]; then
    echo "🤖 Sourcing ROS 2 workspace..."
    source "${ROS2_WS}/install/setup.bash"
    echo "   ✅ ROS 2 workspace sourced"
fi
echo ""

# Run verification script
SCRIPT_PATH="${ROS2_WS}/scripts/verify_stitcher.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Error: Verification script not found at ${SCRIPT_PATH}"
    exit 1
fi

echo "▶️  Running verification script..."
echo ""
python3 "$SCRIPT_PATH" "$@"

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ Script completed successfully"
else
    echo ""
    echo "❌ Script failed with exit code $exit_code"
fi

exit $exit_code
