#!/bin/bash
# Run standalone C++ RGBD Estimator
#
# Usage:
#   ./run_standalone_estimator.sh [dataset_path] [output_dir]
#
# Arguments:
#   dataset_path (optional): Path to dataset directory 
#                           (default: /home/motoken/college/sphere-stereo/resources)
#   output_dir (optional):   Path to output directory
#                           (default: /home/motoken/college/ros2_ws/output/standalone)

# Set LibTorch library path
export LD_LIBRARY_PATH=/home/motoken/.local/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to workspace root (2 levels up from src/my_stereo_pkg/)
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Path to the standalone estimator executable
ESTIMATOR_BIN="$WORKSPACE_ROOT/install/my_stereo_pkg/lib/my_stereo_pkg/standalone_estimator"

# Check if executable exists
if [ ! -f "$ESTIMATOR_BIN" ]; then
    echo "Error: Executable not found at $ESTIMATOR_BIN"
    echo "Please build the package first:"
    echo "  cd $WORKSPACE_ROOT && colcon build --packages-select my_stereo_pkg"
    exit 1
fi

# Run the standalone estimator with provided arguments
exec "$ESTIMATOR_BIN" "$@"
