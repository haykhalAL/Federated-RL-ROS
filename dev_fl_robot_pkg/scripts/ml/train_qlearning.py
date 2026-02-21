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
from ml.ppo_agent import PPOAgent,build_ppo_state
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

ACTION_DIM = 2
NUM_LIDAR_SECTORS = 24
STATE_DIM = NUM_LIDAR_SECTORS + 3 

def main():
    rospy.init_node("qlearning_trainer", anonymous=True)

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    experiment = config["experiment"]
    robots = config["robots"]

    robot = robots[0]  # start with robot1
    name = robot["name"]
    maze_spawn = robot["maze_spawn"]
    paradigm = experiment["paradigm"]
    goal_radius= experiment["goal"]["radius"]
    # Create controller interface (already running as ROS node)
    controller = RobotControllerClient(name)

    env = RobotEnv(
        controller=controller,
        goal_radius=goal_radius,
        paradigm = paradigm
    )


    rospy.loginfo("🚀 Q-learning training started")

    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM
    )

    NUM_EPISODES = 500 
    
    for episode in range(NUM_EPISODES):
        rospy.loginfo(f"🎬 Episode {episode}")

        env.reset()
        done = False
        total_reward = 0.0

        state = None
        while state is None and not rospy.is_shutdown():
            state = build_ppo_state(env, env.goal_pose, NUM_LIDAR_SECTORS)
            rospy.sleep(0.05)

        while not done and not rospy.is_shutdown():
            action, logp, value = agent.select_action(state)
            # print("action:", action, "logp:", logp, "value:", value.item())
            _, reward, done = env.step(action)
            # print("state",state)

            next_state = None
            while next_state is None and not rospy.is_shutdown():
                next_state = build_ppo_state(env, env.goal_pose, NUM_LIDAR_SECTORS)
                rospy.sleep(0.02)

            agent.store_transition(
                state=state,
                action=action,
                log_prob=logp,
                value=value,
                reward=reward,
                done=done
            )
            
            total_reward += reward
            state = next_state

        if len(agent.rewards) > 0:
            agent.update()
        else:
            rospy.logwarn("Skipping PPO update (no transitions collected)")
        
        rospy.loginfo(
            f"[EP {episode}] reward={total_reward:.2f}"
        )




if __name__ == "__main__":
    main()
