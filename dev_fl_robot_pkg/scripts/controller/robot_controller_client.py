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
        self.cmd_pub = rospy.Publisher(
            f"{self.ns}/cmd_vel", Twist, queue_size=10
        )

        # Subscribers
        self.lidar_ranges = None
        self.lidar_scan = None
        self.position = None
        self.yaw = None
        self.last_scan = None
        self.scan_sub = rospy.Subscriber(
            f"/{robot_name}/scan",
            LaserScan,
            self.scan_callback,
            queue_size=1
        )

        rospy.Subscriber(f"{self.ns}/scan", LaserScan, self.scan_callback)
        rospy.Subscriber(f"{self.ns}/odom", Odometry, self.odom_callback)

        self.linear_speed = 0.2
        self.angular_speed = 0.5
        self.collision_distance = 0.18  

    def scan_callback(self, msg):
        self.lidar_ranges = msg.ranges
        self.lidar_scan = msg

    def get_lidar_scan(self):
        return self.lidar_scan
    
    def odom_callback(self, msg):
        self.position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2*(orientation.w*orientation.z),
            1 - 2*(orientation.z**2)
        )

    def step(self, action):
        twist = Twist()
        if action == 0:
            twist.linear.x = self.linear_speed
        elif action == 1:
            twist.angular.z = self.angular_speed
        elif action == 2:
            twist.angular.z = -self.angular_speed
        self.cmd_pub.publish(twist)

    def has_collision(self):
        if self.lidar_ranges is None:
            return False
        return min(self.lidar_ranges) < self.collision_distance

    def get_state(self):
        return self.lidar_ranges

    def get_pose_state(self):
        if self.position is None or self.yaw is None:
            return None
        return (self.position.x, self.position.y, self.yaw)
    
    def get_lidar_ranges(self):
        return self.lidar_ranges
    
    
    