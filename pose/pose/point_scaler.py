#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from omx_cpp_interface.msg import ArmJointAngles
import numpy as np
import os
import argparse
from pathlib import Path
import time, math
import tkinter as tk
from tkinter import filedialog


class PointScaler(Node):

    def __init__(self, canvas_output):
        super().__init__('point_scaler')

        # Get the ROS_DOMAIN_ID aka robot number
        ros_domain_id = os.getenv("ROS_DOMAIN_ID", "0")
        try:
            if int(ros_domain_id) < 10:
                ros_domain_id = "0" + str(int(ros_domain_id))
            else:
                ros_domain_id = str(int(ros_domain_id))
        except Exception:
            ros_domain_id = "00"
        self.get_logger().info(f'ROS_DOMAIN_ID: {ros_domain_id}')

        # Arm joint angles publisher
        self.angles_pub = self.create_publisher(
            ArmJointAngles, 
            f'/tb{ros_domain_id}/target_joint_angles', 
            10
        )

        # Load angles from file
        canvas_path = Path(canvas_output)
        if not canvas_path.is_file():
            raise FileNotFoundError(f"Angles file not found: {canvas_output}")
    
        # Current index for publishing
        self.current_index = 0
        
        self.canvas_points = []
        self.real_world_points = []
        self.read_canvas_points(canvas_output)
    
    def read_canvas_points(self, canvas_output):
        data = np.load(canvas_output, allow_pickle=False)

        self.canvas_points = data["poses"]
        self.get_logger().info(f"Pose array shape: {self.canvas_points.shape}")

        self.compute_real_coords()
    
    def compute_real_coords(self):
        center_y = 0.0
        center_z = 0.2

        x = 0.15

        real_length = 0.085
        canvas_length = 200

        ratio = real_length / canvas_length

        for p in self.canvas_points:
            y = center_y + (ratio * p[0])
            z = center_z + (ratio * p[1])

            theta = math.atan(y / x) / 2

            self.real_world_points.append([x, y, z, 0.0, 0.0, math.sin(theta), math.cos(theta)])

        self.write_points()

    def write_points(self):
        """Save all poses in drawing order as npz file"""
        if len(self.real_world_points) == 0:
            print("No poses to save. Please draw something first.")
            return
        
        # Convert to numpy array: shape (N, 7)
        poses_array = np.array(self.real_world_points, dtype=np.float32)
        
        # Default save directory: current directory
        default_dir = Path.cwd()
        default_filename = "real_poses.npz"
        default_path = default_dir / default_filename
        
        # Ask user for save location, with default path
        file_path = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")],
            title="Save poses as NPZ",
            initialdir=str(default_dir),
            initialfile=default_filename
        )
        
        if file_path:
            # Ensure parent directory exists
            save_path = Path(file_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as npz file
            np.savez(save_path, poses=poses_array)
            abs_path = save_path.resolve()
            print(f"Saved {len(self.real_world_points)} poses to: {abs_path}")
            print(f"Pose array shape: {poses_array.shape}")
        

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Scale canvas points to real world coordinates"
    )
    parser.add_argument(
        '--canvas_output',
        type=str,
        required=True,
        help='Path to .npz file containing canvas points (shape: (N, 2))'
    )
    
    # Parse known args to avoid conflicts with rclpy
    parsed_args, unknown = parser.parse_known_args()
    
    rclpy.init(args=args)
    
    try:
        point_scaler = PointScaler(parsed_args.canvas_output)
        
        time.sleep(2)  # Wait for publisher to set up
        rclpy.spin(point_scaler)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'point_scaler' in locals():
            point_scaler.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()