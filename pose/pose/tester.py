import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, TransformStamped
from custom_interfaces.msg import Traffic
from omx_cpp_interface.msg import ArmJointAngles
from control_msgs.msg import JointTrajectoryControllerState
import math, time, os, random, torch

from tf2_ros import Buffer, TransformListener

class Tester(Node):
    def __init__(self):
        super().__init__('arm_pose_reader')

        # get the ROS_DOMAIN_ID aka robot number
        ros_domain_id = os.getenv("ROS_DOMAIN_ID", "0")
        try:
            if int(ros_domain_id) < 10:
                ros_domain_id = "0" + str(int(ros_domain_id))
            else:
                ros_domain_id = str(int(ros_domain_id))
        except Exception:
            ros_domain_id = "00"
        self.get_logger().info(f'ROS_DOMAIN_ID: {ros_domain_id}')

        self.ranges = [[-80, 80], 
                       [-90, 75],
                       [-53, 79],
                       [-100, 117]]

        self.phi = [math.radians(-80), math.radians(80)]
        self.theta = [math.radians(0), math.radians(80)]
        self.radius = [0.05, 0.39]

        # Arm joint angles publisher
        self.angles_pub = self.create_publisher(ArmJointAngles, 
                                         f'/tb{ros_domain_id}/target_joint_angles', 
                                         10)
        
        self.pose_pub = self.create_publisher(Pose, 
                                         f'/tb{ros_domain_id}/target_pose', 
                                         10)
        
        self.timer = self.create_timer(4, self.sample_movement)

        self.joint_sub = self.create_subscription(
            JointTrajectoryControllerState,
            '/arm_controller/state',
            self.joint_state_callback,
            10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.dataset = []
        self.count = 0
        self.limit = 200000
        self.velocity_threshold = 0.001
        self.base_link = "world"
        self.ee_link = "end_effector_link"
        
    def sample_angles_radians(self):
        angles_rad = []
        for low_deg, high_deg in self.ranges:
            angle_deg = random.uniform(low_deg, high_deg)
            angles_rad.append(math.radians(angle_deg))
        return angles_rad
    
    def sample_in_spherical_section(self):
        u = random.random()
        v = random.random()
        w = random.random()
        
        r = (self.radius[0]**3 + (self.radius[1]**3 - self.radius[0]**3) * u)**(1/3)

        cos_tmin = math.cos(self.theta[0])
        cos_tmax = math.cos(self.theta[1])
        cos_t = cos_tmin + (cos_tmax - cos_tmin) * v
        theta = math.acos(cos_t)

        phi = self.phi[0] + (self.phi[1] - self.phi[0]) * w

        s = math.sin(theta)
        x = r * s * math.cos(phi)
        y = r * s * math.sin(phi)
        z = r * cos_t

        return x, y, z

    def sample_movement(self):
        angles = self.sample_angles_radians()
        self.get_logger().info(f"Sampled angles: {angles}")

        joints_msg = ArmJointAngles()
        joints_msg.joint1 = float(angles[0])
        joints_msg.joint2 = float(-1)
        joints_msg.joint3 = float(angles[2])
        joints_msg.joint4 = float(angles[3])
        self.angles_pub.publish(joints_msg)

    def sample_movement_pt(self):
        points = self.sample_in_spherical_section()
        self.get_logger().info(f"Sampled points: {points}")

        pose_msg = Pose()
        pose_msg.position.x = points[0]
        pose_msg.position.y = points[1]
        pose_msg.position.z = points[2]
        self.pose_pub.publish(pose_msg)

    def joint_state_callback(self, msg):
        velocities = msg.actual.velocities

        moving = any(abs(v) > self.velocity_threshold for v in velocities)
        if not moving:
            return

        joints = list(msg.actual.positions)
        try:
            t = self.tf_buffer.lookup_transform(
                self.base_link,
                self.ee_link,
                rclpy.time.Time()
            )

            x = t.transform.translation.x
            y = t.transform.translation.y
            z = t.transform.translation.z

            qx = t.transform.rotation.x
            qy = t.transform.rotation.y
            qz = t.transform.rotation.z
            qw = t.transform.rotation.w

        except Exception as e:
            self.get_logger().warn(f"FK unavailable: {e}")
            return

        input_tensor = torch.tensor(joints, dtype=torch.float32)          # 4 joints
        output_tensor = torch.tensor([x, y, z, qx, qy, qz, qw],
                                    dtype=torch.float32)                 # 7-D pose

        self.dataset.append((input_tensor, output_tensor))
        self.count += 1

        if (self.count > self.limit):
            self.save_dataset_and_exit()

        self.get_logger().info(
            f"[Saved] entry {self.count}"
        )
    
    def save_dataset_and_exit(self):
        torch.save(self.dataset, "joint_to_pose_dataset_200k.pt")
        self.get_logger().info(
            f"Saved {len(self.dataset)} samples to joint_to_pose_dataset.pt"
        )

        # Clean shutdown
        self.get_logger().info("Exiting program now.")
        rclpy.shutdown()
        os._exit(0)

def main(args=None):
    rclpy.init(args=args)
    robot_node = Tester()
    time.sleep(2)
    rclpy.spin(robot_node)
    robot_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()