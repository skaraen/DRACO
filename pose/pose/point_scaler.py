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
from pathlib import Path

def deg_to_rad(deg):
    """Convert degrees to radians."""
    return deg * math.pi / 180.0

# Convert angle based on forward vector to original reference based on left vector
def reorient_lidar_angle(angle):
    """Convert LiDAR angle reference from forward-facing (0 deg) to left-facing (0 deg)."""
    l_angle = angle - 90
    if abs(l_angle) > 180:
        l_angle += 360
    return l_angle

def distance_at_angle(msg, angle):
    """Get LiDAR distance reading at a specific angle."""
    lidar_angle = reorient_lidar_angle(angle)
    idx_float = (deg_to_rad(lidar_angle) - msg.angle_min) // msg.angle_increment
    idx_int = int(round(idx_float))
    return msg.ranges[idx_int]

class PointScaler(Node):
    """Node that loads canvas-space poses, detects board distance, and computes real-world coordinates."""

    def __init__(self, trace):
        super().__init__('point_scaler')

        # Determine robot ID through ROS_DOMAIN_ID
        ros_domain_id = os.getenv("ROS_DOMAIN_ID", "0")
        try:
            if int(ros_domain_id) < 10:
                ros_domain_id = "0" + str(int(ros_domain_id))
            else:
                ros_domain_id = str(int(ros_domain_id))
        except Exception:
            ros_domain_id = "00"
        self.get_logger().info(f'ROS_DOMAIN_ID: {ros_domain_id}')

        # Load the directory of saved canvas poses
        self.trace_dir = Path("src/DRACO/traces") / trace
        if not self.trace_dir.is_dir():
            raise FileNotFoundError(f"Trace directory not found: {self.trace_dir}")
    
        # Index and storage
        self.current_index = 0
        self.canvas_points = []
        self.real_world_points = []
        self.curr_file_name = ""

        # LiDAR subscription for board detection
        self.lidar_sub = self.create_subscription(
            LaserScan,
            f'/tb{ros_domain_id}/scan',
            self.lidar_callback,
            10
        )

        # Board detection state
        self.board_found = False
        self.points_read = False
        self.board_distance = -1
        self.marker_length = 0.065 
        self.forward_offset = 0.104

    def find_board(self, msg):
        """Determine board distance using LiDAR beam at 0 degrees."""
        self.board_found = True
        self.board_distance = distance_at_angle(msg, 0) - self.forward_offset
        return
    
    def lidar_callback(self, msg):
        """Process LiDAR readings and, once the board is found, load canvas points."""
        if not self.board_found:
            self.find_board(msg)
        elif not self.points_read:
            self.points_read = True
            self.read_canvas_points()
            rclpy.shutdown()
    
    def read_canvas_points(self):
        """Load canvas-space stroke segments from saved NPZ files."""
        for file in sorted(self.trace_dir.glob("*.npz")):
            self.get_logger().info(f"Loading trace: {file}")

            data = np.load(file, allow_pickle=False)

            if "poses" not in data:
                self.get_logger().warning(f"Skipping {file}: 'poses' key not found")
                continue
            
            self.curr_file_name = file.name
            self.canvas_points = data["poses"]
            self.get_logger().info(
                f"{file.name} pose array shape: {self.canvas_points.shape}"
            )

            self.compute_real_coords()

    def compute_real_coords(self):
        """Convert canvas coordinates to real-world claw-frame target positions."""
        self.real_world_points = []
        cy = 0.0
        cz = 0.15

        M = self.marker_length
        h = 0.17  # forward distance (x-axis)

        # Pixel-to-meter scale factor
        real_length = 0.07
        canvas_length = 200
        scale = real_length / canvas_length

        # Convert each canvas point into a world-frame pose
        for p in self.canvas_points:
            v = cz + (scale * p[1])          # z direction
            w = -1 * (cy + (scale * p[0]))   # y direction (sign flip)

            # Vector from reference point (0, cy, cz) to board point (h, w, v)
            dx = h
            dy = w - cy
            dz = v - cz

            L = math.sqrt(dx*dx + dy*dy + dz*dz)
            if L < 1e-8:
                continue  # Avoid division by zero

            # Move claw back along the line by marker length
            offset_ratio = (L - M) / L

            x = offset_ratio * dx       
            y = cy + offset_ratio * dy
            z = cz + offset_ratio * dz

            # Compute quaternion that orients x-axis along direction vector
            qx = 0.0
            qy = -dz / math.sqrt(2.0 * L * (L + dx))
            qz =  dy / math.sqrt(2.0 * L * (L + dx))
            qw = math.sqrt((L + dx) / (2.0 * L))

            self.real_world_points.append([x, y, z, qx, qy, qz, qw])

        self.write_points()

    def write_points(self):
        """Save the resulting world-frame poses to an NPZ file."""
        if len(self.real_world_points) == 0:
            print("No poses to save. Please draw something first.")
            return
        
        # Convert to array
        poses_array = np.array(self.real_world_points, dtype=np.float32)
        
        # Save into matching rposes file
        cfile = Path(self.curr_file_name) 
        rposes_file_name = cfile.stem.replace("_cposes", "_rposes") + cfile.suffix
        rposes_path = self.trace_dir / rposes_file_name
        
        np.savez(rposes_path, poses=poses_array)

        abs_path = rposes_path.resolve()
        self.get_logger().info(f"Saved {len(self.real_world_points)} poses to: {abs_path}")
        self.get_logger().info(f"Pose array shape: {poses_array.shape}")


def main(args=None):
    """Entry point for the point scaler node."""
    parser = argparse.ArgumentParser(
        description="Scale canvas points to real world coordinates"
    )
    parser.add_argument(
        '--trace',
        type=str,
        required=True,
        help='Trace directory of .npz files containing canvas poses'
    )
    
    parsed_args, unknown = parser.parse_known_args()
    
    rclpy.init(args=args)
    
    try:
        point_scaler = PointScaler(parsed_args.trace)
        
        time.sleep(2)  # Allow node to initialize before spinning
        rclpy.spin(point_scaler)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'point_scaler' in locals():
            point_scaler.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
