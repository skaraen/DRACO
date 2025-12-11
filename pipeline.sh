#!/bin/bash
# This script is used to:
# 1. draw the picture on the canvas using the canvas/canvas.py
# 2. save the poses in the canvas/canvas.py
# 3. scale the poses to the real world coordinates using the pose/pose/point_scaler.py
# 4. sample joint angles from poses using flow_matching_model/sample_joint_from_pose.py
# 5. publish the poses to the robot using the pose/pose/push_the_angles.py

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE_FILE="$SCRIPT_DIR/traces/.last_trace"

# 1. draw the picture on the canvas using the canvas/canvas.py
echo "Step 1: Drawing on canvas..."
python3 "$SCRIPT_DIR/canvas/canvas.py"

# Check if trace file exists
if [ ! -f "$TRACE_FILE" ]; then
    echo "Error: Trace file not found. Please save your drawing first."
    exit 1
fi

# Read trace name from file
TRACE_NAME=$(cat "$TRACE_FILE" | tr -d '\n\r ')
if [ -z "$TRACE_NAME" ]; then
    echo "Error: Trace name is empty."
    exit 1
fi

echo "Using trace name: $TRACE_NAME"

# 2. scale the poses to the real world coordinates using the pose/pose/point_scaler.py
echo "Step 2: Scaling poses to real world coordinates..."
ros2 run pose point_scaler -- --trace "$TRACE_NAME"

# 3. sample joint angles from poses using flow_matching_model/sample_joint_from_pose.py
echo "Step 3: Sampling joint angles from poses..."
TRACE_DIR="$SCRIPT_DIR/traces/$TRACE_NAME"
python3 "$SCRIPT_DIR/flow_matching_model/sample_joint_from_pose.py" --pose_npy "$TRACE_DIR"

# 4. publish the poses to the robot using the pose/pose/push_the_angles.py
echo "Step 4: Publishing joint angles to robot..."
ros2 run pose push_the_angles -- --trace "$TRACE_NAME"

# 5. reset the arm
echo "Step 5: Resetting arm..."
ros2 run pose reset_arm