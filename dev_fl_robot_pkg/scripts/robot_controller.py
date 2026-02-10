#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import random

class RobotController:
    def __init__(self):
        # ---- Namespace handling
        self.ns = rospy.get_namespace().strip("/")
        self.robot_name = rospy.get_param("~robot_name", None)
        if self.robot_name is None:
            self.robot_name = self.ns

        rospy.loginfo(f"[{self.ns}] Initializing robot controller")

        # ---- Publishers & Subscribers
        self.cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)

        rospy.Subscriber("scan", LaserScan, self.scan_callback)
        rospy.Subscriber("odom", Odometry, self.odom_callback)

        # ---- Robot state
        self.lidar_ranges = None
        self.position = None
        self.yaw = None

        # ---- Safety parameters
        self.collision_distance = 0.25  # meters
        self.linear_speed = 0.20
        self.angular_speed = 0.60

        self.rate = rospy.Rate(10)

    # =============================
    # Callbacks
    # =============================

    def scan_callback(self, msg: LaserScan):
        # Replace inf with max range
        self.lidar_ranges = [
            r if not math.isinf(r) else msg.range_max
            for r in msg.ranges
        ]

    def odom_callback(self, msg: Odometry):
        self.position = msg.pose.pose.position

        orientation = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ])

    # =============================
    # Movement primitives
    # =============================

    def stop(self):
        self.cmd_pub.publish(Twist())

    def move_forward(self):

        twist = Twist()
        twist.linear.x = self.linear_speed
        self.cmd_pub.publish(twist)

    def turn_left(self):
        twist = Twist()
        twist.angular.z = self.angular_speed
        self.cmd_pub.publish(twist)

    def turn_right(self):
        twist = Twist()
        twist.angular.z = -self.angular_speed
        self.cmd_pub.publish(twist)

    # =============================
    # Safety & State
    # =============================

    def is_collision_ahead(self) -> bool:
        if self.lidar_ranges is None:
            return False

        # Check ±15 degrees in front
        center = len(self.lidar_ranges) // 2
        window = 15

        front_ranges = self.lidar_ranges[
            center - window : center + window
        ]

        return min(front_ranges) < self.collision_distance

    def has_collision(self):
        """
        Returns True if the robot is too close to an obstacle
        based on LiDAR readings.
        """
        if self.lidar_ranges is None:
            return False

        return min(self.lidar_ranges) < self.collision_distance

    def get_state(self):
        """
        RL-ready state:
        - Min distance front
        - Min distance left
        - Min distance right
        - Robot yaw
        """
        if self.lidar_ranges is None or self.yaw is None:
            return None

        n = len(self.lidar_ranges)

        front = min(self.lidar_ranges[n//2 - 10 : n//2 + 10])
        left  = min(self.lidar_ranges[n*3//4 - 10 : n*3//4 + 10])
        right = min(self.lidar_ranges[n//4 - 10 : n//4 + 10])

        return {
            "front": front,
            "left": left,
            "right": right,
            "yaw": self.yaw
        }

    # =============================
    # Action interface (for RL)
    # =============================

    def step(self, action: int):
        twist = Twist()

        if action == 0:
            twist.linear.x = self.linear_speed
        elif action == 1:
            twist.angular.z = self.angular_speed
        elif action == 2:
            twist.angular.z = -self.angular_speed
        else:
            pass  # stop

        self.cmd_pub.publish(twist)

    # =============================
    # Manual test loop
    # =============================

    # def spin(self):
    #     rospy.loginfo(f"[{self.ns}] Controller running")
    #     while not rospy.is_shutdown():
    #         # Default behavior: slow forward + obstacle avoidance
    #         self.move_forward()
    #         self.rate.sleep()

    def reset_robot_pose(robot_name, maze_spawn, local_grid_centers):
        rospy.wait_for_service("/gazebo/set_model_state")
        set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        mx, my, _ = maze_spawn
        lx, ly = random.choice(local_grid_centers)

        state = ModelState()
        state.model_name = robot_name
        state.pose.position.x = mx + lx
        state.pose.position.y = my + ly
        state.pose.position.z = 0.01

        yaw = random.uniform(-math.pi, math.pi)
        state.pose.orientation.z = math.sin(yaw / 2.0)
        state.pose.orientation.w = math.cos(yaw / 2.0)

        state.reference_frame = "world"
        set_state(state)

if __name__ == "__main__":
    rospy.init_node("robot_controller")
    controller = RobotController()
    # controller.spin()
