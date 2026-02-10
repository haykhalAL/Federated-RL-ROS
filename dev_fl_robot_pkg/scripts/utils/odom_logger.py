#!/usr/bin/env python3
import rospy
import sys
from nav_msgs.msg import Odometry

def odom_callback(msg):
    pos = msg.pose.pose.position
    rospy.loginfo("Robot Position -> x: %.2f, y: %.2f, z: %.2f", pos.x, pos.y, pos.z)

def main():
    # Pick robot name from CLI args (ignore ROS internal args)
    robot_name = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("__") else "robot1"

    # Use robot-specific node name
    rospy.init_node(f"{robot_name}_odom_logger", anonymous=True)

    # Subscribe to robot-specific odom topic
    topic = f"/{robot_name}/odom"
    rospy.loginfo(f"Listening to {topic}")
    rospy.Subscriber(topic, Odometry, odom_callback)

    rospy.spin()

if __name__ == "__main__":
    main()
