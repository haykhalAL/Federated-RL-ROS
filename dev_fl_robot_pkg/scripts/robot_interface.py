import rospy
import math

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
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
    # INIT SERVICE
    ####################################################
    def clear_sensor_buffer(self):

        self.position = None
        self.yaw = None
        self.scan = None
        self.lidar_scan = None

    def wait_until_ready(self, timeout=10.0):

        start_time = rospy.Time.now()

        rate = rospy.Rate(20)

        while not rospy.is_shutdown():

            if (
                self.position is not None
                and self.yaw is not None
                and self.lidar_scan is not None
            ):
                rospy.loginfo(
                    f"[{self.robot_name}] Sensors ready."
                )
                return True

            elapsed = (
                rospy.Time.now() - start_time
            ).to_sec()

            if elapsed >= timeout:

                rospy.logwarn(
                    f"[{self.robot_name}] "
                    f"Sensor readiness timeout."
                )

                return False

            rate.sleep()

        return False

    
    ####################################################
    # ROS CALLBACKS
    ####################################################

    def scan_callback(self, msg):

        self.lidar_scan = msg

        self.scan = []

        for r in msg.ranges:

            if math.isnan(r) or math.isinf(r):
                r = msg.range_max

            r = max(
                msg.range_min,
                min(r, msg.range_max)
            )

            self.scan.append(r)

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

        if action == 0:
            # Forward
            twist.linear.x = self.linear_speed
            twist.angular.z = 0.0

        elif action == 1:
            # Forward + left
            twist.linear.x = self.linear_speed
            twist.angular.z = self.angular_speed

        elif action == 2:
            # Forward + right
            twist.linear.x = self.linear_speed
            twist.angular.z = -self.angular_speed

        elif action == 3:
            # Rotate left
            twist.linear.x = 0.0
            twist.angular.z = self.angular_speed

        elif action == 4:
            # Rotate right
            twist.linear.x = 0.0
            twist.angular.z = -self.angular_speed

        else:
            raise ValueError(f"Invalid action: {action}")

        self.cmd_pub.publish(twist)
        return twist.linear.x, twist.angular.z

    def stop(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0

        self.cmd_pub.publish(twist)
    ####################################################
    # COLLISION / SENSOR HELPERS
    ####################################################

    def get_min_lidar_distance(self):

        if self.scan is None:
            return None

        valid_ranges = [
            r for r in self.scan
            if not math.isnan(r) and not math.isinf(r)
        ]

        if not valid_ranges:
            return None

        return min(valid_ranges)

    def has_collision(self):

        if self.scan is None:
            return False

        valid_ranges = [
            r for r in self.scan
            if not math.isnan(r)
            and not math.isinf(r)
        ]

        if not valid_ranges:
            return False

        # Count how many LiDAR rays are extremely close
        close_count = sum(
            1 for r in valid_ranges
            if r < 0.15
        )

        # Require multiple rays to be close before declaring collision
        return close_count >= 3

    ####################################################
    # POSE
    ####################################################

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