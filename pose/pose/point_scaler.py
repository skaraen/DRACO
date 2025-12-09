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
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

def deg_to_rad(deg):
        return deg * math.pi / 180.0

# standard ZYX yaw from quaternion
def quat_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

# wrap to [-pi, pi]
def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

# Convert angle based on forward vector to original reference based on left vector
def reorient_lidar_angle(angle):
    # Lidar angle 0 is forward, but we want 0 to be left
    l_angle = angle - 90
    if abs(l_angle) > 180:
        l_angle += 360
    return l_angle

def distance_at_angle(msg, angle):
    lidar_angle = reorient_lidar_angle(angle)
    idx_float = (deg_to_rad(lidar_angle) - msg.angle_min) // msg.angle_increment
    idx_int = int(round(idx_float))

    return msg.ranges[idx_int]

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
        self.canvas_output = canvas_output
        canvas_path = Path(canvas_output)
        if not canvas_path.is_file():
            raise FileNotFoundError(f"Angles file not found: {canvas_output}")
    
        # Current index for publishing
        self.current_index = 0
        
        self.canvas_points = []
        self.real_world_points = []

        #LiDAR reorient
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # self.odom_sub = self.create_subscription(
        #     Odometry, '/odom', self.odom_callback, 10
        # )
        # self.current_yaw = None
        self.lidar_sub = self.create_subscription(LaserScan, f'/tb{ros_domain_id}/scan', self.lidar_callback, 10)

        self.board_found = False
        self.points_read = False
        self.board_distance = -1
        self.marker_length = 0.01

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        self.current_yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

    def find_board(self, msg):
        # Rotate to face min_idx later
        self.board_found = True
        self.board_distance = distance_at_angle(msg, 0)

        return
    
        # min_dist = float('inf')
        # min_idx = -1

        # for idx, d in enumerate(msg.ranges):
        #     if d < msg.range_min or d > msg.range_max or math.isnan(d):
        #         continue
            
        #     if d < min_dist:
        #         min_dist = d
        #         min_idx = idx
    
    def lidar_callback(self, msg):
        self.get_logger().info("Hello")
        # Initialize beam indices if not set
        if not self.board_found:
            self.find_board(msg)
        elif not self.points_read:
            self.points_read = True
            self.read_canvas_points(self.canvas_output)
    
    def read_canvas_points(self, canvas_output):
        data = np.load(canvas_output, allow_pickle=False)

        self.canvas_points = data["poses"]
        self.get_logger().info(f"Pose array shape: {self.canvas_points.shape}")

        self.compute_real_coords()
    
    def compute_real_coords(self):
        cy = 0.0
        cz = 0.2
        # h = self.board_distance
        h = 0.15
        # m = self.marker_length
        m = 0.0

        self.get_logger().info(f"Board origin: {h}, {cy}, {cz}")

        real_length = 0.085
        canvas_length = 200
        scale = real_length / canvas_length

        for p in self.canvas_points:
            z = cz + (scale * p[1])
            w = cy + (scale * p[0])

            hyp = math.sqrt((w * w) + (h * h))
            ratio = (hyp - m) / hyp

            x = ratio * h
            y = ratio * w

            theta = math.atan(y / x) / 2

            self.get_logger().info(f"x: {x}, y: {y}, z: {z}")
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