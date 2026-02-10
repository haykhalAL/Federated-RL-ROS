import rospy
import math
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import LaserScan
import numpy as np
from std_srvs.srv import Empty
import time


# ----------------------------
# LOW-LEVEL RESET (NO RANDOM)
# ----------------------------
def reset_robot_pose(robot_name, pose):
    rospy.wait_for_service("/gazebo/set_model_state")
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

    x, y, yaw = pose

    state = ModelState()
    state.model_name = robot_name
    state.pose.position.x = x
    state.pose.position.y = y
    state.pose.position.z = 0.01

    state.pose.orientation.z = math.sin(yaw / 2.0)
    state.pose.orientation.w = math.cos(yaw / 2.0)

    state.reference_frame = "world"
    set_state(state)


# ============================
# ENVIRONMENT
# ============================
class RobotEnv:

    def __init__(
        self,
        controller,
        start_pose,
        goal_pose,
        goal_radius,
        max_steps=500
    ):
        self.controller = controller

        # navigation task
        self.start_pose = start_pose      # (x, y, yaw)
        self.goal_pose = goal_pose        # (x, y)
        self.goal_radius = goal_radius

        # episode control
        self.step_count = 0
        self.max_steps = max_steps
        self.prev_dist = None
        self.lidar_ranges = None

        rospy.loginfo("⏳ Waiting for Gazebo physics services...")
        rospy.wait_for_service("/gazebo/pause_physics")
        rospy.wait_for_service("/gazebo/unpause_physics")

        self.pause_physics = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.unpause_physics = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)

        rospy.loginfo("✅ Gazebo physics services connected")

        rospy.Subscriber(
            f"/{self.controller.robot_name}/scan",
            LaserScan,
            self._lidar_callback
        )

    def _lidar_callback(self, msg):
        self.lidar_ranges = np.array(msg.ranges)

    
    # ----------------------------
    # RESET = NEW EPISODE
    # ----------------------------
    def reset(self):
        reset_robot_pose(
            self.controller.robot_name,
            self.start_pose
        )

        rospy.sleep(0.2)
        self.step_count = 0
        self.prev_dist = None

        rospy.loginfo(
            f"[RESET] robot={self.controller.robot_name} "
            f"start={self.start_pose} goal={self.goal_pose}"
        )

        return self.controller.get_state()

    # ----------------------------
    # STEP
    # ----------------------------
    def step(self, action):
        STEP_DT = 0.2  # seconds of simulation per RL step

        reward = 0.0
        done = False
        print("step press")
        # --- unpause physics ---
        self.unpause_physics()
        print("unpaused")
        # --- apply action ONCE ---
        self.controller.step(action)

        # --- let simulation advance deterministically ---
        time.sleep(STEP_DT)

        # --- pause physics ---
        self.pause_physics()

        # --- read pose ---
        pose = self.controller.get_pose_state()
        if pose is None:
            return None, 0.0, False

        px, py, yaw = pose

        # --- read lidar ---
        lidar = self.get_lidar_sectors(num_sectors=24)
        if lidar is None:
            rospy.logwarn("LiDAR not ready, skipping step")
            return None, 0.0, False

        min_lidar = float(np.min(lidar))

        # --- distance to goal ---
        dx = self.goal_pose[0] - px
        dy = self.goal_pose[1] - py
        curr_dist = math.sqrt(dx*dx + dy*dy)

        if not math.isfinite(curr_dist):
            self.prev_dist = None
            return None, -1.0, False

        # --- distance shaping ---
        if self.prev_dist is not None:
            reward += 2.0 * (self.prev_dist - curr_dist)

        self.prev_dist = curr_dist

        # --- wall proximity penalty ---
        if min_lidar < 0.4:
            reward -= (0.4 - min_lidar) * 20.0

        # --- time penalty ---
        reward -= 0.001

        # --- termination ---
        if curr_dist <= self.goal_radius:
            reward += 100.0
            done = True
            rospy.loginfo("🏁 GOAL REACHED")

        elif self.controller.has_collision():
            reward -= 100.0
            done = True
            rospy.logwarn("💥 COLLISION")

        self.step_count += 1
        if self.step_count >= self.max_steps:
            reward -= 20.0
            done = True
            rospy.logwarn("⏱ STEP LIMIT")

        return None, reward, done

    # ----------------------------
    # GOAL CHECK
    # ----------------------------
    def _check_goal(self, state):
        x = state[0]
        y = state[1]

        dx = x - self.goal_pose[0]
        dy = y - self.goal_pose[1]
        dist = math.sqrt(dx*dx + dy*dy)

        return dist <= self.goal_radius
    
    def get_lidar_sectors(self, num_sectors=24):
        scan = self.controller.lidar_scan
        if scan is None:
            return None

        ranges = np.array(scan.ranges, dtype=np.float32)
        ranges = np.clip(ranges, 0.0, scan.range_max)

        angles = np.linspace(
            scan.angle_min,
            scan.angle_max,
            len(ranges),
            endpoint=False
        )

        # normalize to [-pi, pi]
        angles = (angles + math.pi) % (2 * math.pi) - math.pi

        sector_width = 2 * math.pi / num_sectors
        sectors = np.full(num_sectors, scan.range_max, dtype=np.float32)

        for r, a in zip(ranges, angles):
            idx = int((a + math.pi) / sector_width)
            idx = min(idx, num_sectors - 1)
            sectors[idx] = min(sectors[idx], r)

        return sectors
