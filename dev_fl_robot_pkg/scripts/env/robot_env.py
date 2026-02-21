import rospy
import math
import random
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import LaserScan
import numpy as np
from std_srvs.srv import Empty
import time




# ============================
# ENVIRONMENT
# ============================
class RobotEnv:

    def __init__(
        self,
        controller,
        goal_radius,
        paradigm,
        max_steps=500
    ):
        self.controller = controller
        self.paradigm = paradigm
        self.goal_radius = goal_radius
        self.start_pose = None
        self.goal_pose = None
        
        
        # episode control
        self.step_count = 0
        self.max_steps = max_steps
        self.prev_dist = None
        self.lidar_ranges = None

        rospy.loginfo("Waiting for Gazebo physics services...")
        rospy.wait_for_service("/gazebo/pause_physics")
        rospy.wait_for_service("/gazebo/unpause_physics")

        self.pause_physics = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.unpause_physics = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)

        rospy.loginfo("Gazebo physics services connected")

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

        self.pause_physics()
        
        self.reset_robot_pose(
            self.controller.robot_name,
            self.paradigm
        )

        self.controller.clear_sensor_buffer()

        self.unpause_physics()

        state = None
        start = rospy.Time.now()

        while state is None and not rospy.is_shutdown():
            state = self.controller.get_state()

            # safety timeout (prevents infinite hang)
            if (rospy.Time.now() - start).to_sec() > 2.0:
                rospy.logwarn("Reset sensor timeout — retrying...")
                return self.reset()

            rospy.sleep(0.05)

        self.step_count = 0
        self.prev_dist = None

        rospy.loginfo(
            f"[RESET] robot={self.controller.robot_name} "
            f"start={self.start_pose} goal={self.goal_pose}"
        )

        return state

    # ----------------------------
    # LOW-LEVEL RESET (NO RANDOM)
    # ----------------------------
    def reset_robot_pose(self,robot_name,paradigm):
        rospy.wait_for_service("/gazebo/set_model_state")
        set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        self.start_pose, self.goal_pose = self.set_start_and_goals(paradigm)
        x, y, yaw = self.start_pose
        state = ModelState()
        state.model_name = robot_name
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0.01

        state.pose.orientation.z = math.sin(yaw / 2.0)
        state.pose.orientation.w = math.cos(yaw / 2.0)

        state.reference_frame = "world"
        set_state(state)

    # ----------------------------
    # STEP
    # ----------------------------
    def step(self, action):

        STEP_DT = 0.25  # duration of one RL step

        reward = 0.0
        done = False

        # --- apply action ---
        self.controller.step(action)

        # --- let simulation run naturally ---
        rospy.sleep(STEP_DT)

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
        
        goal_angle = math.atan2(dy, dx)
        angle_error = math.atan2(math.sin(goal_angle - yaw), math.cos(goal_angle - yaw))

        heading_reward = math.cos(angle_error)
        reward += 0.1 * heading_reward

        v = action[0]   # raw network action

        forward_bonus = max(0.0, v) * max(0.0, heading_reward)
        reward += 0.4 * forward_bonus

        w = action[1]
        reward -= 0.05 * abs(w)
        if not math.isfinite(curr_dist):
            self.prev_dist = None
            return None, -1.0, False

        # --- reward: progress ---
        if self.prev_dist is not None:
            progress = self.prev_dist - curr_dist

            # only count if actually moving toward goal
            if progress > 0 and heading_reward > 0.3:
                reward += 4.0 * progress

        self.prev_dist = curr_dist

        # --- wall penalty ---
        # if min_lidar < 0.4:
        #     reward -= (0.4 - min_lidar) * 20.0

        # --- time penalty ---
        reward -= 0.001

        # --- success ---
        if curr_dist <= self.goal_radius:
            reward += 100.0
            done = True
            rospy.loginfo("🏁 GOAL REACHED")

        # --- collision ---
        elif self.controller.has_collision():
            reward -= 100.0
            done = True
            rospy.logwarn("💥 COLLISION")

        # --- max steps ---
        self.step_count += 1
        if self.step_count >= self.max_steps:
            reward -= 20.0
            done = True
            rospy.logwarn("⏱ STEP LIMIT")
        print ("lidar :",lidar)
        print ("current location :",pose)
        print ("min dist :", min_lidar, "dist to go :", curr_dist)
        print ("reward :",reward, "start :", self.start_pose, "goals :", self.goal_pose)
        return pose, reward, done

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

    def set_start_and_goals(self, paradigm):
        grid_centers = [
        (-2.5, -2.5), (-1.5, -2.5), (-0.5, -2.5), (0.5, -2.5), (1.5, -2.5), (2.5, -2.5),
        (-2.5, -1.5), (-1.5, -1.5), (-0.5, -1.5), (0.5, -1.5), (1.5, -1.5), (2.5, -1.5),
        (-2.5, -0.5), (-1.5, -0.5), (-0.5, -0.5), (0.5, -0.5), (1.5, -0.5), (2.5, -0.5),
        (-2.5,  0.5), (-1.5,  0.5), (-0.5,  0.5), (0.5,  0.5), (1.5,  0.5), (2.5,  0.5),
        (-2.5,  1.5), (-1.5,  1.5), (-0.5,  1.5), (0.5,  1.5), (1.5,  1.5), (2.5,  1.5),
        (-2.5,  2.5), (-1.5,  2.5), (-0.5,  2.5), (0.5,  2.5), (1.5,  2.5), (2.5,  2.5),
        ]

        #0 - same start&goal per reset, 1 - random start same goal per reset, 2 - random start and goal per reset
        if (paradigm == 0):
            start_pose = self.start_pose
            goal_pose = self.goal_pose
            if (self.start_pose is None):
                x, y = random.choice(grid_centers)
                start_pose = (x, y, 0.0)
            if (self.goal_pose is None):
                x, y = random.choice(grid_centers)
                goal_pose = (x, y, 0.0)
            return start_pose,goal_pose
        elif(paradigm == 1):
            goal_pose = self.goal_pose
            x, y = random.choice(grid_centers)
            start_pose = (x, y, 0.0)
            return start_pose,goal_pose
        elif(paradigm == 2):
            x, y = random.choice(grid_centers)
            start_pose = (x, y, 0.0)
            x, y = random.choice(grid_centers)
            goal_pose = (x, y, 0.0)
            return start_pose,goal_pose

