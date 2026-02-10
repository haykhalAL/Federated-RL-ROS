#!/usr/bin/env python3
import rospy
import sys
from sensor_msgs.msg import LaserScan

def lidar_callback(msg):
    # Example: just show front distance (center ray)
    ranges = msg.ranges
    if ranges:  # avoid empty scans
        front = ranges[len(ranges)//2]
        rospy.loginfo("Front distance: %.2f m", front)

def main():
    # Take robot name from CLI args (skip ROS hidden args like __name:=...)
    robot_name = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("__") else "robot1"

    # Node name is unique per robot
    rospy.init_node(f"{robot_name}_lidar_logger", anonymous=True)

    # Topic is namespaced per robot
    topic = f"/{robot_name}/scan"
    rospy.loginfo(f"Listening to {topic}")
    rospy.Subscriber(topic, LaserScan, lidar_callback)

    rospy.spin()

if __name__ == "__main__":
    main()
