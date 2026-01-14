#!/bin/bash
set -e

# Source ROS 2 workspace for C++ bindings
source /home/motoken/college/ros2_ws/install/setup.bash

# Run ISB Filter verification
cd /home/motoken/college/sphere-stereo
python3 /home/motoken/college/ros2_ws/scripts/verify_isb_filter.py
