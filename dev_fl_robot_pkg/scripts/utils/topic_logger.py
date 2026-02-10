#!/usr/bin/env python3
import rospy
import sys
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

def callback(msg):
    rospy.loginfo(f"[{rospy.get_name()}] {msg}")

def main():
    if len(sys.argv) < 2:
        print("Usage: rosrun dev_fl_robot_pkg topic_logger.py <topic_name>")
        sys.exit(1)

    topic_name = sys.argv[1]

    rospy.init_node(f"logger_{topic_name.replace('/', '_')}", anonymous=True)

    # Type detection (extend this if needed)
    if "odom" in topic_name:
        msg_type = Odometry
    elif "scan" in topic_name:
        msg_type = LaserScan
    elif "cmd_vel" in topic_name:
        msg_type = Twist
    else:
        msg_type = String  # fallback

    rospy.Subscriber(topic_name, msg_type, callback)
    rospy.loginfo(f"🔎 Logging topic: {topic_name}")
    rospy.spin()

if __name__ == "__main__":
    main()
