#!/usr/bin/env python3
import yaml
import os
import rospy
import roslaunch
import socket
import time
import subprocess
import xacro
import random
import math
import rospkg

MAZE_SIZE = 6        # meters
CELL_SIZE = 1.0
HALF = MAZE_SIZE / 2
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(
    SCRIPT_DIR,
    "config",
    "robots_config.yaml"
)
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("dev_fl_robot_pkg")
URDF_XACRO_PATH = os.path.join(
    pkg_path,
    "models",
    "urdf",
    "turtlebot3_burger.urdf.xacro"
)

# Mapping between logger names in YAML and relative ROS topics
LOGGER_TOPICS = {
    "odom": "odom",
    "lidar": "scan",
    "velocity": "cmd_vel",
    "location": "amcl_pose",
    "battery": "battery_state",
    "camera": "camera/rgb/image_raw"
}

def set_robot_description(ns, xacro_path):
    doc = xacro.process_file(xacro_path, mappings={"robot_namespace": ns})
    urdf_xml = doc.toxml()
    rospy.set_param(f"/{ns}/robot_description", urdf_xml)

def wait_for_master(host="localhost", port=11311, timeout=20):
    """Wait until roscore is available at host:port."""
    for i in range(timeout):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect((host, port))
            sock.close()
            print("✅ ROS master is up.")
            return True
        except socket.error:
            print(f"⏳ Waiting for roscore... ({i+1}/{timeout})")
            time.sleep(1)
    return False

def list_robot_topics(ns):
    """List topics for a given robot namespace."""
    try:
        from rosgraph.masterapi import Master
        master = Master('/rostopic')
        topics = master.getTopicTypes()
        robot_topics = [t[0] for t in topics if t[0].startswith(f"/{ns}/")]
        if robot_topics:
            rospy.loginfo(f"📡 Topics for {ns}:\n  " + "\n  ".join(robot_topics))
        else:
            rospy.logwarn(f"⚠️ No topics found for {ns} yet.")
    except Exception as e:
        rospy.logerr(f"Error listing topics for {ns}: {e}")

def generate_local_grid_centers():
    """
    Grid centers in LOCAL maze frame (centered at 0,0)
    """
    centers = []
    cells = int(MAZE_SIZE / CELL_SIZE)

    for i in range(cells):
        for j in range(cells):
            x = -HALF + CELL_SIZE / 2 + i * CELL_SIZE
            y = -HALF + CELL_SIZE / 2 + j * CELL_SIZE
            centers.append((x, y))

    return centers


LOCAL_GRID_CENTERS = generate_local_grid_centers()

def sample_grid_spawn_with_offset(maze_spawn):
    """
    maze_spawn = [mx, my, myaw]
    """
    mx, my, _ = maze_spawn

    lx, ly = random.choice(LOCAL_GRID_CENTERS)

    # Global coordinates = maze center + local cell center
    x = mx + lx
    y = my + ly
    yaw = random.uniform(-math.pi, math.pi)

    return x, y, yaw

def main():
    # Ensure roscore is running before continuing
    if not wait_for_master():
        print("❌ ROS master not available, exiting...")
        return

    rospy.init_node("multi_robot_launcher", anonymous=True)

    # --- Regenerate Terminator layouts from YAML before spawning robots
    try:
        subprocess.call(["rosrun", "dev_fl_robot_pkg", "generate_terminal_robots.py"])
        rospy.loginfo("🛠 Regenerated Terminator layouts from robots_config.yaml")
    except Exception as e:
        rospy.logwarn(f"⚠️ Could not regenerate Terminator layouts: {e}")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()

    for robot in config["robots"]:
        if not robot.get("deploy", True):
            rospy.loginfo(f"Skipping {robot['name']} (deploy = false)")
            continue

        name = robot["name"]
        # x, y, z = robot["spawn"]
        x, y, yaw = sample_grid_spawn_with_offset(robot["maze_spawn"])
        z = 0.01

        # --- Load URDF into namespaced robot_description
        set_robot_description(name, URDF_XACRO_PATH)

        # --- Spawn robot model in Gazebo under its namespace
        spawn = roslaunch.core.Node(
            package="gazebo_ros",
            node_type="spawn_model",
            name=f"spawn_{name}",
            namespace=f"/{name}",
            args=f"-urdf -param /{name}/robot_description "
                f"-model {name} -x {x} -y {y} -z {z} -Y {yaw}"
        )
        launch.launch(spawn)
        rospy.loginfo(f"🚀 Deployed {name} at ({x},{y},{z}) in /{name} namespace")

        # --- Start robot movement controller (one per robot)
        controller = roslaunch.core.Node(
            package="dev_fl_robot_pkg",
            node_type="robot_controller.py",
            name="robot_controller",
            namespace=f"/{name}",
            output="screen"
        )
        rospy.set_param(f"/{name}/robot_controller/robot_name", name)
        launch.launch(controller)

        rospy.loginfo(f"🕹 Robot movement controller started for {name}")

        rsp = roslaunch.core.Node(
            package="robot_state_publisher",
            node_type="robot_state_publisher",
            name="robot_state_publisher",
            namespace=f"/{name}",
            output="screen"
        )
        launch.launch(rsp)

        rospy.loginfo(f"🤖 robot_state_publisher started for {name}")
        # --- Spawn maze model for this robot
        mx, my, myaw = robot.get("maze_spawn", [0.0, 0.0, 0.0])
        maze = roslaunch.core.Node(
            package="gazebo_ros",
            node_type="spawn_model",
            name=f"maze_{name}",
            args=f"-sdf -file $(find dev_fl_robot_pkg)/models/maze_6x6/model.sdf "
                 f"-model maze_{name} -x {mx} -y {my} -Y {myaw}",
        )
        launch.launch(maze)
        rospy.loginfo(f"🧩 Spawned maze for {name} at ({mx}, {my}, yaw={myaw})")

        # --- List topics for the spawned robot
        time.sleep(2)  # small delay to let Gazebo advertise topics
        list_robot_topics(name)

        # --- Open Terminator with layout for loggers
        layout_name = f"{name}_layout"
        rospy.loginfo(f"📟 Opening Terminator for {name} with layout {layout_name}")
        subprocess.Popen(["terminator", "-l", layout_name])

    rospy.loginfo("✅ All deployable robots + mazes launched successfully.")
    rospy.spin()

if __name__ == "__main__":
    main()
