import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from custom_interfaces.msg import Traffic
from omx_cpp_interface.msg import ArmJointAngles, ArmGripperPosition
import math
import time
import os

class ResetRobot(Node):

    def __init__(self):
        super().__init__('bot_reset_arm')

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

        # Publisher for arm joint angles
        self.angles_pub = self.create_publisher(
            ArmJointAngles, 
            f'/tb{ros_domain_id}/target_joint_angles', 
            10
        )
        time.sleep(2)

        # Immediately reset arm on startup
        self.reset_arm()

    def reset_arm(self):
        """Publish a fixed set of joint angles to move the arm into its home pose."""
        joints_msg = ArmJointAngles()
        joints_msg.joint1 = float(0)
        joints_msg.joint2 = float(-1.494)
        joints_msg.joint3 = float(0.508)
        joints_msg.joint4 = float(1.033)
        self.angles_pub.publish(joints_msg)


def main(args=None):
    rclpy.init(args=args)
    robot_node = ResetRobot()
    time.sleep(2)
    rclpy.spin(robot_node)
    robot_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
