import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math


class RobotControllerClient:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.ns = f"/{robot_name}"

        # Publishers
        self.cmd_pub = rospy.Publisher(f"{self.ns}/cmd_vel", Twist, queue_size=10)

        # Sensor cache
        self.lidar_ranges = None
        self.lidar_scan = None
        self.last_scan = None
        self.last_odom = None

        self.position = None
        self.yaw = None

        # Subscribers
        rospy.Subscriber(f"{self.ns}/scan", LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber(f"{self.ns}/odom", Odometry, self.odom_callback, queue_size=1)

        self.linear_speed = 0.2
        self.angular_speed = 0.5
        self.collision_distance = 0.18

    # ------------------ Callbacks ------------------

    def scan_callback(self, msg):
        self.lidar_ranges = msg.ranges
        self.lidar_scan = msg
        self.last_scan = msg

    def odom_callback(self, msg):
        self.last_odom = msg
        self.position = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2*(q.w*q.z), 1 - 2*(q.z*q.z))

    # ------------------ Actions ------------------

    def step(self, action):
        twist = Twist()
        if action == 0:
            twist.linear.x = self.linear_speed
        elif action == 1:
            twist.angular.z = self.angular_speed
        elif action == 2:
            twist.angular.z = -self.angular_speed
        self.cmd_pub.publish(twist)

    # ------------------ Sensors ------------------

    def has_collision(self):
        if self.lidar_ranges is None:
            return False
        return min(self.lidar_ranges) < self.collision_distance

    def get_pose_state(self):
        if self.last_odom is None:
            return None
        return (self.position.x, self.position.y, self.yaw)

    def get_lidar_ranges(self):
        return self.lidar_ranges

    def get_state(self):
        if self.last_scan is None or self.last_odom is None:
            return None
        return True

    # ------------------ Reset helper ------------------

    def clear_sensor_buffer(self):
        self.lidar_ranges = None
        self.lidar_scan = None
        self.last_scan = None
        self.last_odom = None
        self.position = None
        self.yaw = None