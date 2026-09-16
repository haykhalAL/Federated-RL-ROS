import rospy
import math
import random
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import numpy as np
from std_srvs.srv import Empty
from utils.maze_utils import generate_grid_centers



# ============================
# ENVIRONMENT
# ============================
class RobotEnv:

    def __init__(
        self,
        robot,
        config
    ):
        self.robot = robot
        self.start_pose = None
        self.goal_pose = None
        self.step_dt = 0.25

        exp = config["experiment"]
        training = exp["training"]

        maze_cfg = config["environment"]["maze"]

        self.maze_size = maze_cfg["size"]
        self.cell_size = maze_cfg["cell_size"]

        self.grid_centers = generate_grid_centers(
            self.maze_size,
            self.cell_size
        )

        self.max_goal_distance = math.sqrt(
            self.maze_size**2 +
            self.maze_size**2
        )
        self.paradigm = exp["paradigm"]
        self.goal_radius = exp["goal"]["radius"]

        self.step_dt = training["step_dt"]
        self.max_steps = training["max_steps"]

        self.num_lidar_sectors = exp["lidar"]["sectors"]
        terminal = self.reward_cfg["terminal"]
        shaping = self.reward_cfg["shaping"]
        
        # episode control
        self.step_count = 0
        self.prev_dist = None

        rospy.loginfo("Waiting for Gazebo physics services...")
        rospy.wait_for_service("/gazebo/pause_physics")
        rospy.wait_for_service("/gazebo/unpause_physics")

        self.pause_physics = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.unpause_physics = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)

        rospy.loginfo("Gazebo physics services connected")
    
    # ----------------------------
    # RESET = NEW EPISODE
    # ----------------------------
        
    def reset(self):

        self.pause_physics()
        
        self.reset_robot_pose(
            self.robot.robot_name,
            self.paradigm
        )

        self.robot.clear_sensor_buffer()

        self.unpause_physics()

        state = None
        start = rospy.Time.now()

        while state is None and not rospy.is_shutdown():
            state = self.build_state()

            # safety timeout (prevents infinite hang)
            if (rospy.Time.now() - start).to_sec() > 2.0:
                rospy.logwarn("Reset sensor timeout — retrying...")
                return self.reset()

            rospy.sleep(0.05)

        self.step_count = 0
        self.prev_dist = None

        rospy.loginfo(
            f"[RESET] robot={self.robot.robot_name} "
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

        self.step_count += 1

        # Execute action
        linear, angular = self.robot.execute_action(action)

        # Let the robot move for one RL timestep
        rospy.sleep(self.step_dt)

        # Stop robot before observing next state
        self.robot.stop()

        next_state = self.build_state()

        if next_state is None:
            return None, 0.0, False, {}

        curr_dist = self.distance_to_goal()

        if curr_dist is None:
            return None, 0.0, False, {}

        lidar = self.get_lidar_sectors()

        if lidar is None:
            return None, 0.0, False, {}

        min_lidar = np.min(lidar)

        reward, reward_info = self.compute_reward(
            linear,
            angular,
            curr_dist,
            min_lidar
        )

        terminated = (
            self.goal_reached(curr_dist)
            or self.is_collision()
        )

        truncated = (
            self.step_count >= self.max_steps
        )

        done = terminated or truncated

        info = self.get_info(
            curr_dist,
            reward,
            min_lidar
        )

        info["reward"] = reward_info
        info["terminated"] = terminated
        info["truncated"] = truncated
        return next_state, reward, done, info

    
    def get_lidar_sectors(self):

        scan = self.robot.lidar_scan

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

        angles = (angles + math.pi) % (2 * math.pi) - math.pi

        sector_width = 2 * math.pi / self.num_lidar_sectors

        sectors = np.full(
            self.num_lidar_sectors,
            scan.range_max,
            dtype=np.float32
        )

        for r, a in zip(ranges, angles):

            idx = int((a + math.pi) / sector_width)

            idx = min(idx, self.num_lidar_sectors - 1)

            sectors[idx] = min(sectors[idx], r)

        return sectors


    def set_start_and_goals(self, paradigm):
        grid_centers = self.grid_centers
        PARADIGM_FIXED = 0
        PARADIGM_RANDOM_START = 1
        PARADIGM_RANDOM_START_GOAL = 2
        #0 - same start&goal per reset, 1 - random start same goal per reset, 2 - random start and goal per reset
        if paradigm == PARADIGM_FIXED::
            start_pose = self.start_pose
            goal_pose = self.goal_pose
            if (self.start_pose is None):
                x, y = random.choice(grid_centers)
                start_pose = (x, y, 0.0)
            if (self.goal_pose is None):
                x, y = random.choice(grid_centers)
                goal_pose = (x, y, 0.0)
            return start_pose,goal_pose
        elif paradigm == PARADIGM_RANDOM_START:
            goal_pose = self.goal_pose
            x, y = random.choice(grid_centers)
            start_pose = (x, y, 0.0)
            return start_pose,goal_pose
        elif paradigm == PARADIGM_RANDOM_START_GOAL:
            x, y = random.choice(grid_centers)
            start_pose = (x, y, 0.0)
            x, y = random.choice(grid_centers)
            goal_pose = (x, y, 0.0)
            return start_pose,goal_pose

    def build_state(self):

        pose = self.robot.get_pose_state()

        if pose is None:
            return None

        px, py, yaw = pose

        lidar = self.get_lidar_sectors()

        if lidar is None:
            return None

        dx = self.goal_pose[0] - px
        dy = self.goal_pose[1] - py

        distance = math.sqrt(dx**2 + dy**2)

        goal_angle = math.atan2(dy, dx)

        angle_error = math.atan2(
            math.sin(goal_angle - yaw),
            math.cos(goal_angle - yaw)
        )

        lidar = lidar / self.robot.lidar_scan.range_max

        goal_distance = min(
            distance / self.max_goal_distance,
            1.0
        )

        state = np.concatenate([
            lidar,
            np.array([
                math.cos(angle_error),
                math.sin(angle_error),
                goal_distance
            ], dtype=np.float32)
        ])

        return state.astype(np.float32)

    def get_info(
        self,
        curr_dist,
        reward,
        min_lidar
    ):

        return {

            "distance": curr_dist,

            "min_lidar": min_lidar,

            "pose": self.robot.get_pose_state(),

            "reward": reward,

            "goal": self.goal_reached(curr_dist),

            "collision": self.robot.has_collision(),

            "step": self.step_count,

            "goal_pose": self.goal_pose,

            "start_pose": self.start_pose

        }

    def goal_reached(self, distance):

        return distance <= self.goal_radius

    def is_collision(self):

        return self.robot.has_collision()

    def distance_to_goal(self):

        pose = self.robot.get_pose_state()

        if pose is None:
            return None

        px, py, _ = pose

        dx = self.goal_pose[0] - px
        dy = self.goal_pose[1] - py

        return math.sqrt(dx**2 + dy**2)


    def compute_reward(
        self,
        linear,
        angular,
        curr_dist,
        min_lidar
    ):

        reward = 0.0

        pose = self.robot.get_pose_state()

        if pose is None:
            return 0.0, {}

        px, py, yaw = pose

        # -------------------------
        # Terminal rewards
        # -------------------------

        if self.goal_reached(curr_dist):
            return self.reward_cfg["goal"], {
                "goal": self.reward_cfg["goal"]
            }

        if self.is_collision():
            return self.reward_cfg["collision"], {
                "collision": self.reward_cfg["collision"]
            }

        reward_info = {}

        # -------------------------
        # Progress reward
        # -------------------------

        progress_reward = 0.0

        if self.prev_dist is not None:

            progress = self.prev_dist - curr_dist

            progress_reward = (
                self.reward_cfg["progress"] * progress
            )

            reward += progress_reward

        self.prev_dist = curr_dist

        reward_info["progress"] = progress_reward

        # -------------------------
        # Heading reward
        # -------------------------

        dx = self.goal_pose[0] - px
        dy = self.goal_pose[1] - py

        goal_angle = math.atan2(dy, dx)

        angle_error = math.atan2(
            math.sin(goal_angle - yaw),
            math.cos(goal_angle - yaw)
        )

        heading = math.cos(angle_error)

        heading_reward = (
            self.reward_cfg["heading"] * heading
        )

        reward += heading_reward

        reward_info["heading"] = heading_reward

        # -------------------------
        # Forward reward
        # -------------------------


        forward_reward = (
            self.reward_cfg["forward"]
            * max(0.0, linear)
            * max(0.0, heading)
        )

        reward += forward_reward

        reward_info["forward"] = forward_reward

        # -------------------------
        # Turning penalty
        # -------------------------


        turn_penalty = (
            self.reward_cfg["turn_penalty"]
            * abs(angular)
        )

        reward -= turn_penalty

        reward_info["turn"] = -turn_penalty

        # -------------------------
        # Wall penalty
        # -------------------------

        wall_penalty = 0.0

        if min_lidar < self.reward_cfg["wall_distance"]:

            wall_penalty = (
                self.reward_cfg["wall_distance"] - min_lidar
            ) * self.reward_cfg["wall_penalty"]

            reward -= wall_penalty

        reward_info["wall"] = -wall_penalty

        # -------------------------
        # Time penalty
        # -------------------------

        reward -= self.reward_cfg["time"]

        reward_info["time"] = -self.reward_cfg["time"]

        reward = float(np.clip(
            reward,
            -1000,
            1000
        ))

        reward_info["total"] = reward

        return reward, reward_info

    @property
    def observation_size(self):
        return self.num_lidar_sectors + 3
