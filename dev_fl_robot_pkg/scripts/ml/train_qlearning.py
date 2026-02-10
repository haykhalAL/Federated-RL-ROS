#!/usr/bin/env python3
import os
import sys
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPTS_ROOT)

import rospy
import random
import math
import numpy as np
from env.robot_env import RobotEnv
from controller.robot_controller_client import RobotControllerClient
from ml.q_learning_agent import QLearningAgent
from ml.dqn_agent import DQNAgent
from ml.ppo_agent import PPOAgent
from utils.navigation_logger import CSVLogger

episode_logger = CSVLogger(
    "logs/episodes.csv",
    header=[
        "episode",
        "total_reward",
        "final_dist",
        "steps",
        "collided",
        "success"
    ]
)

CONFIG_PATH = os.path.join(
    SCRIPTS_ROOT,
    "config",
    "robots_config.yaml"
)

ACTION_DIM = 3
NUM_LIDAR_SECTORS = 24
STATE_DIM = NUM_LIDAR_SECTORS + 3 
def main():
    rospy.init_node("qlearning_trainer", anonymous=True)

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    experiment = config["experiment"]
    robots = config["robots"]

    paradigm = experiment["paradigm"]
    goal_cfg = experiment["goal"]

    robot = robots[0]  # start with robot1
    name = robot["name"]
    maze_spawn = robot["maze_spawn"]
    # Create controller interface (already running as ROS node)
    controller = RobotControllerClient(name)
    # Maze info (must match launcher)
    maze_spawn = [0.0, 0.0, 0.0]

    # Grid centers (same logic you used in launcher)
    grid_centers = [
        (-2.5, -2.5), (-1.5, -2.5), (-0.5, -2.5), (0.5, -2.5), (1.5, -2.5), (2.5, -2.5),
        (-2.5, -1.5), (-1.5, -1.5), (-0.5, -1.5), (0.5, -1.5), (1.5, -1.5), (2.5, -1.5),
        (-2.5, -0.5), (-1.5, -0.5), (-0.5, -0.5), (0.5, -0.5), (1.5, -0.5), (2.5, -0.5),
        (-2.5,  0.5), (-1.5,  0.5), (-0.5,  0.5), (0.5,  0.5), (1.5,  0.5), (2.5,  0.5),
        (-2.5,  1.5), (-1.5,  1.5), (-0.5,  1.5), (0.5,  1.5), (1.5,  1.5), (2.5,  1.5),
        (-2.5,  2.5), (-1.5,  2.5), (-0.5,  2.5), (0.5,  2.5), (1.5,  2.5), (2.5,  2.5),
    ]

    if goal_cfg["mode"] == "random_per_run":
        goal_pose = random.choice(grid_centers)
    else:
        raise ValueError("Unsupported goal mode")

    goal_radius = goal_cfg["radius"]

    rospy.loginfo(f"[GLOBAL GOAL] {goal_pose}, radius={goal_radius}")

    x, y = random.choice(grid_centers)

    start_pose = (x, y, 0.0)

    env = RobotEnv(
        controller=controller,
        start_pose=start_pose,
        goal_pose=goal_pose,
        goal_radius=goal_radius
    )


    rospy.loginfo("🚀 Q-learning training started")
    ACTIONS = [0, 1, 2]  # forward, left, right

    

    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM
    )

    NUM_EPISODES = 5000 

    # traj_logger = CSVLogger(
    #     "logs/trajectory.csv",
    #     header=[
    #         "episode",
    #         "step",
    #         "x",
    #         "y",
    #         "action",
    #         "goal_dist",
    #         "min_lidar",
    #         "reward"
    #     ]
    # )
    
    for episode in range(NUM_EPISODES):
        rospy.loginfo(f"🎬 Episode {episode}")

        env.unpause_physics()
        env.reset()
        env.pause_physics()
        done = False
        total_reward = 0.0

        state = build_ppo_state(env, goal_pose)
        if state is None:
            continue

        while not done and not rospy.is_shutdown():
            action, logp, value = agent.select_action(state)
            print("action:", action, "logp:", logp, "value:", value.item())
            obs, reward, done = env.step(action)
            if obs is None and not done:
                rospy.logwarn("Step failed, retrying")
                continue

            agent.store_transition(
                state=state,
                action=action,
                log_prob=logp,
                value=value,          # 👈 THIS LINE WAS MISSING
                reward=reward,
                done=done
            )
            
            next_state = build_ppo_state(env, goal_pose)

            if next_state is None:
                rospy.logwarn("Next state None, ending episode")
                break

            total_reward += reward
            state = next_state

            # ppo_state = build_dqn_state(env, state, goal_pose)
            # if ppo_state is None:
            #     continue

            # action = agent.select_action(ppo_state)
            # next_state, reward, done = env.step(action)
            # rospy.loginfo(
            #     "\n"
            #     f"  action : {action}\n"
            #     f"  reward : {reward:.3f}\n"
            #     f"  next_state : {next_state}\n"
            #     f"  sensor : {ppo_state[0:5]}\n"
            #     f"  goals  : {ppo_state[5:7]}\n"
            # )
            # if env.step_count % 5 == 0:
            #     traj_logger.log([
            #         episode,
            #         env.step_count,
            #         x,
            #         y,
            #         action,
            #         ppo_state[5:7],
            #         ppo_state[0:5],
            #         reward
            #     ])

            # agent.store_reward(reward, done)
            # total_reward += reward
            # state = next_state

        if len(agent.rewards) > 0:
            agent.update()
        else:
            rospy.logwarn("Skipping PPO update (no transitions collected)")
        
        rospy.loginfo(
            f"[EP {episode}] reward={total_reward:.2f}"
        )

def build_dqn_state(env, state, goal):

    pose = env.controller.get_pose_state()
    if pose is None:
        return None
    x, y, yaw = pose

    if not math.isfinite(yaw):
        yaw = 0.0

    lidar = env.get_lidar_sectors()
    if lidar is None:
        return None

    front, fl, fr, left, right, back, bl, br = lidar

    MAX_RANGE = 3.5  # must match your laser max range

    lidar_vals = np.array([front, fl, fr, left, right], dtype=np.float32)
    lidar_vals = np.clip(lidar_vals, 0.0, MAX_RANGE) / MAX_RANGE

    dx = goal[0] - x
    dy = goal[1] - y

    goal_dist = math.sqrt(dx*dx + dy*dy)
    goal_angle = math.atan2(dy, dx) - yaw

    # normalize angle to [-pi, pi]
    goal_angle = (goal_angle + math.pi) % (2*math.pi) - math.pi

    
    return np.array([
        front / 3.5,
        fl / 3.5,
        fr / 3.5,
        left / 3.5,
        right / 3.5,
        math.cos(goal_angle),     
        math.sin(goal_angle),     
        goal_dist / 5.0           
    ], dtype=np.float32)

def build_ppo_state(env, goal):
    pose = env.controller.get_pose_state()
    if pose is None:
        return None

    x, y, yaw = pose
    if not math.isfinite(yaw):
        yaw = 0.0

    lidar = env.get_lidar_sectors(num_sectors=NUM_LIDAR_SECTORS)
    if lidar is None or len(lidar) != NUM_LIDAR_SECTORS:
        return None

    MAX_RANGE = 3.5
    lidar = np.clip(lidar, 0.0, MAX_RANGE) / MAX_RANGE

    dx = goal[0] - x
    dy = goal[1] - y
    goal_dist = math.sqrt(dx*dx + dy*dy)
    goal_dist = min(goal_dist / 5.0, 1.0)

    goal_angle = math.atan2(dy, dx) - yaw
    goal_angle = math.atan2(math.sin(goal_angle), math.cos(goal_angle))

    return np.concatenate([
        lidar,
        np.array([
            math.cos(goal_angle),
            math.sin(goal_angle),
            goal_dist
        ], dtype=np.float32)
    ])

if __name__ == "__main__":
    main()
