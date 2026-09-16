#!/usr/bin/env python3

import math

import rospy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from tf.transformations import euler_from_quaternion


class RobotInterface:

    def __init__(self, robot_name):

        self.robot_name = robot_name
        self.ns = f"/{robot_name}"

        rospy.loginfo(
            f"[{self.robot_name}] Robot controller initialized."
        )

        self.cmd_pub = rospy.Publisher(
            f"{self.ns}/cmd_vel",
            Twist,
            queue_size=10
        )

        rospy.Subscriber(
            f"{self.ns}/scan",
            LaserScan,
            self.scan_callback,
            queue_size=1
        )

        rospy.Subscriber(
            f"{self.ns}/odom",
            Odometry,
            self.odom_callback,
            queue_size=1
        )

        self.position = None
        self.yaw = None
        self.scan = None
        self.lidar_scan = None
        self.linear_speed = 0.25
        self.angular_speed = 1.0

        self.collision_distance = 0.25

    ####################################################
    # ROS INIT
    ####################################################

    def wait_until_ready(self, timeout=10):

        rate = rospy.Rate(20)
        start = rospy.Time.now()

        while not rospy.is_shutdown():

            if (
                self.position is not None and
                self.yaw is not None and
                self.scan is not None and
                self.lidar_scan is not None
            ):
                rospy.loginfo(f"[{self.robot_name}] Ready.")
                return True

            if (rospy.Time.now() - start).to_sec() > timeout:
                rospy.logerr(f"[{self.robot_name}] Timed out waiting for topics.")
                return False

            rate.sleep()

    def clear_sensor_buffer(self):

        self.position = None
        self.yaw = None
        self.scan = None
        self.lidar_scan = None
    ####################################################
    # ROS CALLBACKS
    ####################################################

    def scan_callback(self, msg):

        self.lidar_scan = msg

        self.scan = [
            r if not math.isinf(r) else msg.range_max
            for r in msg.ranges
        ]

    def odom_callback(self, msg):

        self.position = msg.pose.pose.position

        q = msg.pose.pose.orientation

        _, _, self.yaw = euler_from_quaternion([
            q.x,
            q.y,
            q.z,
            q.w
        ])

    ####################################################
    # ACTIONS
    ####################################################
    def execute_action(self, action):

        twist = Twist()

        if isinstance(action, (list, tuple)):

            linear = float(action[0])
            angular = float(action[1])

            twist.linear.x = max(
                0.0,
                min(self.linear_speed, linear)
            )

            twist.angular.z = max(
                -self.angular_speed,
                min(self.angular_speed, angular)
            )

        else:

            if action == 0:
                twist.linear.x = self.linear_speed

            elif action == 1:
                twist.angular.z = self.angular_speed

            elif action == 2:
                twist.angular.z = -self.angular_speed

        self.cmd_pub.publish(twist)

        return twist.linear.x, twist.angular.z

    # def send_action(self, action):

    #     """
    #     Supports

    #     DQN:
    #         action = 0,1,2

    #     PPO/SAC:
    #         action = [linear, angular]
    #     """

    #     twist = Twist()

    #     if isinstance(action, (list, tuple)):

    #         linear = float(action[0])
    #         angular = float(action[1])

    #         twist.linear.x = max(
    #             0.0,
    #             min(self.linear_speed, linear)
    #         )

    #         twist.angular.z = max(
    #             -self.angular_speed,
    #             min(self.angular_speed, angular)
    #         )

    #     else:

    #         if action == 0:

    #             twist.linear.x = self.linear_speed

    #         elif action == 1:

    #             twist.angular.z = self.angular_speed

    #         elif action == 2:

    #             twist.angular.z = -self.angular_speed

    #     self.cmd_pub.publish(twist)

    ####################################################
    # HELPERS
    ####################################################

    def stop(self):

        self.cmd_pub.publish(Twist())

    def has_collision(self):

        if self.scan is None:

            return False

        return min(self.scan) < self.collision_distance

    def get_pose_state(self):

        if self.position is None or self.yaw is None:
            return None

        return (
            self.position.x,
            self.position.y,
            self.yaw
        )

    def get_pose(self):

        return self.position

    def get_yaw(self):

        return self.yaw

    def get_scan(self):

        return self.scan
