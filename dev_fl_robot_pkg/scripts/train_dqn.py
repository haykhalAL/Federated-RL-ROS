#!/usr/bin/env python3

import os
import random
import yaml
import math

import numpy as np
import torch

import rospy
import rospkg


from robot_interface import RobotInterface
from env.robot_env import RobotEnv
from ml.dqn_agent import DQNAgent
from dataLogger.logger import DQNLogger


def load_config():

    rospack = rospkg.RosPack()

    config_path = os.path.join(
        rospack.get_path("dev_fl_robot_pkg"),
        "scripts",
        "config",
        "experiment_config.yaml"
    )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_robot(config):

    for robot in config["robots"]:

        if robot["deploy"]:
            return robot

    raise RuntimeError("No deployed robot found in experiment_config.yaml")


def create_agent(env, config):

    algo = config["experiment"]["algorithms"]["dqn"]

    return DQNAgent(

        state_dim=env.observation_size,

        action_dim=3,

        lr=algo["learning_rate"],

        gamma=algo["gamma"],

        epsilon=algo["epsilon"],

        epsilon_min=algo["epsilon_min"],

        epsilon_decay=algo["epsilon_decay"],

        buffer_size=algo["replay_buffer"],

        batch_size=algo["batch_size"]
    )


def train():

    config = load_config()

    exp = config["experiment"]

    training = exp["training"]
    algo = exp["algorithms"]["dqn"]

    set_seed(exp["seed"])

    rospy.init_node("train_dqn")

    robot_cfg = get_robot(config)

    rospy.loginfo(
        f"Using robot: {robot_cfg['name']}"
    )

    robot = RobotInterface(
        robot_cfg["name"]
    )

    if not robot.wait_until_ready():

        rospy.logerr("Robot not ready.")

        return

    env = RobotEnv(
        robot,
        config
    )

    agent = create_agent(
        env,
        config
    )
    
    logger = DQNLogger()
    rospy.loginfo("========== TRAINING START ==========")

    for episode in range(training["episodes"]):

        state = env.reset()

        done = False

        episode_reward = 0.0

        episode_steps = 0

        path_length = 0.0
        previous_pose = None

        collision_count = 0
        success = False
        final_distance = 0.0

        while (
                not rospy.is_shutdown()
                and not done
                and episode_steps < training["max_steps"]
            ):

            action = agent.select_action(
                state
            )

            next_state, reward, done, info = env.step(
                action
            )
            pose = info.get("pose")

            if pose is not None:

                x, y, yaw = pose

                if previous_pose is not None:

                    previous_x, previous_y, _ = previous_pose

                    dx = x - previous_x
                    dy = y - previous_y

                    path_length += math.sqrt(
                        dx * dx + dy * dy
                    )

                previous_pose = pose


            # Track collisions
            if info.get("collision", False):
                collision_count += 1


            # Track goal
            success = info.get(
                "goal",
                success
            )

            final_distance = info.get(
                "distance",
                final_distance
            )
            agent.remember(

                state,

                action,

                reward,

                next_state,

                done
            )

            agent.learn()
            logger.log_step(
                episode=episode,
                step=episode_steps,
                info=info,
                action=action,
                reward=reward
            )
            state = next_state

            episode_reward += reward

            episode_steps += 1

        agent.end_episode()

        if (episode + 1) % algo["target_update"] == 0:
            agent.update_target()

        logger.log_episode(
            episode=episode,
            total_reward=episode_reward,
            steps=episode_steps,
            success=success,
            collision_count=collision_count,
            final_distance=final_distance,
            path_length=path_length,
            epsilon=agent.epsilon
        )
        rospy.loginfo(

            "Episode %d | "
            "Reward %.2f | "
            "Steps %d | "
            "Success %s | "
            "Collisions %d | "
            "Distance %.3f | "
            "Path %.3f | "
            "Epsilon %.3f",

            episode,

            episode_reward,

            episode_steps,

            success,

            collision_count,

            final_distance,

            path_length,

            agent.epsilon
        )

    rospy.loginfo("Training Finished")
    logger.close()


if __name__ == "__main__":

    train()