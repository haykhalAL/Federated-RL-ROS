#!/usr/bin/env python3

import os
import random
import csv
from datetime import datetime

import yaml
import numpy as np
import torch

import rospy
import rospkg

from robot_interface import RobotInterface
from env.robot_env import RobotEnv
from ml.dqn_agent import DQNAgent


# ============================================================
# CONFIG
# ============================================================

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


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# ROBOT CONFIGURATION
# ============================================================

def get_deployed_robots(config):

    robots = [
        robot
        for robot in config["robots"]
        if robot["deploy"]
    ]

    if not robots:
        raise RuntimeError(
            "No deployed robot found in experiment_config.yaml"
        )

    return robots


# ============================================================
# DQN
# ============================================================

def create_agent(env, config):

    algo = config["experiment"]["algorithms"]["dqn"]

    return DQNAgent(
        state_dim=env.observation_size,
        action_dim=5,
        lr=algo["learning_rate"],
        gamma=algo["gamma"],
        epsilon=algo["epsilon"],
        epsilon_min=algo["epsilon_min"],
        epsilon_decay=algo["epsilon_decay"],
        buffer_size=algo["replay_buffer"],
        batch_size=algo["batch_size"],
        target_update=algo["target_update"]
    )


# ============================================================
# SYNCHRONIZED MULTI-ROBOT STEP
# ============================================================

def synchronized_step(robot_instances, actions, step_dt):

    motion_commands = []

    for item, action in zip(robot_instances, actions):

        if item["done"]:
            item["robot"].stop()

            motion_commands.append(
                (
                    item,
                    0.0,
                    0.0
                )
            )

            continue

        linear, angular = item["env"].apply_action(action)

        motion_commands.append(
            (
                item,
                linear,
                angular
            )
        )

    rospy.sleep(step_dt)

    for item in robot_instances:
        item["robot"].stop()

    results = []

    for item, linear, angular in motion_commands:

        env = item["env"]

        if item["done"]:
            results.append(
                (
                    item["state"],
                    0.0,
                    True,
                    {}
                )
            )
            continue

        env.step_count += 1

        result = env.observe_step(
            linear,
            angular
        )

        if result[3] is not None:
            result[3]["linear_velocity"] = linear
            result[3]["angular_velocity"] = angular

        results.append(result)

    return results

# ============================================================
# TRAINING LOGGING
# ============================================================

def create_training_loggers(config, robot_instances):

    log_dir = config["logging"]["log_dir"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(
        log_dir,
        f"dqn_{timestamp}"
    )

    os.makedirs(run_dir, exist_ok=True)

    loggers = {}

    # --------------------------------------------------------
    # General training log
    # --------------------------------------------------------

    training_log_path = os.path.join(
        run_dir,
        "training.log"
    )

    training_log = open(
        training_log_path,
        "w",
        buffering=1
    )

    # --------------------------------------------------------
    # One CSV per robot
    # --------------------------------------------------------

    for item in robot_instances:

        robot_name = item["name"]

        csv_path = os.path.join(
            run_dir,
            f"{robot_name}.csv"
        )

        csv_file = open(
            csv_path,
            "w",
            newline="",
            buffering=1
        )

        writer = csv.writer(csv_file)

        writer.writerow([
            "episode",
            "step",
            "action",
            "reward",
            "progress_reward",
            "heading_reward",
            "forward_reward",
            "turn_reward",
            "wall_reward",
            "time_reward",
            "distance",
            "min_lidar",
            "done",
            "terminated",
            "truncated",
            "collision",
            "goal",
            "x",
            "y",
            "yaw",
            "goal_x",
            "goal_y",
            "goal_yaw",
            "linear_velocity",
            "angular_velocity"
        ])

        loggers[robot_name] = {
            "file": csv_file,
            "writer": writer
        }

    rospy.loginfo(
        "Training logs saved to: %s",
        run_dir
    )

    return run_dir, training_log, loggers


def log_training(training_log, message):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    training_log.write(
        f"[{timestamp}] {message}\n"
    )


def log_robot_step(
    logger,
    episode,
    step,
    action,
    reward,
    info,
    linear,
    angular
):

    pose = info.get("pose", {})
    goal_pose = info.get("goal_pose", {})

    if isinstance(pose, (list, tuple)):

        x = pose[0]
        y = pose[1]
        yaw = pose[2] if len(pose) > 2 else 0.0

    else:

        x = pose.get("x", 0.0)
        y = pose.get("y", 0.0)
        yaw = pose.get("yaw", 0.0)

    if isinstance(goal_pose, (list, tuple)):

        goal_x = goal_pose[0]
        goal_y = goal_pose[1]
        goal_yaw = (
            goal_pose[2]
            if len(goal_pose) > 2
            else 0.0
        )

    else:

        goal_x = goal_pose.get("x", 0.0)
        goal_y = goal_pose.get("y", 0.0)
        goal_yaw = goal_pose.get("yaw", 0.0)

    reward_info = info.get("reward", {})

    logger["writer"].writerow([
        episode,
        step,
        action,
        reward,

        reward_info.get("progress", 0.0),
        reward_info.get("heading", 0.0),
        reward_info.get("forward", 0.0),
        reward_info.get("turn", 0.0),
        reward_info.get("wall", 0.0),
        reward_info.get("time", 0.0),

        info.get("distance", 0.0),
        info.get("min_lidar", 0.0),

        info.get("done", False),
        info.get("terminated", False),
        info.get("truncated", False),
        info.get("collision", False),
        info.get("goal", False),

        x,
        y,
        yaw,

        goal_x,
        goal_y,
        goal_yaw,

        linear,
        angular
    ])


# ============================================================
# RESET ONE ROBOT
# ============================================================

def reset_robot_episode(item, episode):

    state = item["env"].reset()

    if state is None:

        rospy.logerr(
            "%s: reset failed",
            item["name"]
        )

        return None

    item["state"] = state
    item["done"] = False
    item["episode_reward"] = 0.0
    item["episode_steps"] = 0
    item["local_episode"] = episode

    rospy.loginfo(
        "[RESET] %s | episode=%d | state_size=%d | start=%s | goal=%s",
        item["name"],
        episode,
        len(state),
        item["env"].start_pose,
        item["env"].goal_pose
    )

    return state


# ============================================================
# TRAINING
# ============================================================

def train():

    config = load_config()

    exp = config["experiment"]
    training = exp["training"]

    set_seed(exp["seed"])

    rospy.init_node(
        "train_dqn"
    )

    # --------------------------------------------------------
    # Find deployed robots
    # --------------------------------------------------------

    robot_cfgs = get_deployed_robots(
        config
    )

    robot_instances = []

    rospy.loginfo(
        "========== INITIALIZING DEPLOYED ROBOTS =========="
    )

    # --------------------------------------------------------
    # Initialize each robot
    # --------------------------------------------------------

    for robot_cfg in robot_cfgs:

        robot_name = robot_cfg["name"]

        rospy.loginfo(
            "Initializing robot: %s",
            robot_name
        )

        robot = RobotInterface(
            robot_name
        )

        if not robot.wait_until_ready():

            rospy.logerr(
                "Robot not ready: %s",
                robot_name
            )

            continue

        env = RobotEnv(
            robot,
            config,
            robot_cfg
        )

        agent = create_agent(
            env,
            config
        )

        robot_instances.append({
            "name": robot_name,
            "config": robot_cfg,
            "robot": robot,
            "env": env,
            "agent": agent,

            # Independent episode state
            "state": None,
            "done": True,
            "local_episode": 0,
            "episode_reward": 0.0,
            "episode_steps": 0,

            # Total completed episodes
            "completed_episodes": 0
        })

    # --------------------------------------------------------
    # Verify initialization
    # --------------------------------------------------------

    if not robot_instances:

        rospy.logerr(
            "No robots were successfully initialized."
        )

        return

    rospy.loginfo(
        "========== %d ROBOTS INITIALIZED ==========",
        len(robot_instances)
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    run_dir, training_log, robot_loggers = (
        create_training_loggers(
            config,
            robot_instances
        )
    )

    log_training(
        training_log,
        "========== DQN MULTI-ROBOT TRAINING =========="
    )

    log_training(
        training_log,
        f"Robots: {len(robot_instances)}"
    )

    log_training(
        training_log,
        f"Target episodes per robot: "
        f"{training['episodes']}"
    )

    log_training(
        training_log,
        f"Max steps per episode: "
        f"{training['max_steps']}"
    )

    log_training(
        training_log,
        f"Step dt: {training['step_dt']}"
    )

    # ========================================================
    # INITIAL RESET
    # ========================================================

    for item in robot_instances:

        next_episode = (
            item["completed_episodes"] + 1
        )

        state = reset_robot_episode(
            item,
            next_episode
        )

        if state is None:

            log_training(
                training_log,
                f"{item['name']} initial reset FAILED"
            )

            return

        log_training(
            training_log,
            f"[RESET] {item['name']} "
            f"episode={next_episode} "
            f"start={item['env'].start_pose} "
            f"goal={item['env'].goal_pose}"
        )

    # ========================================================
    # GLOBAL SIMULATION LOOP
    #
    # The timestep is synchronized.
    # The robot episodes are NOT synchronized.
    # ========================================================

    global_step = 0

    target_episodes = training["episodes"]

    while not rospy.is_shutdown():

        # ----------------------------------------------------
        # Check whether every robot completed its target
        # number of episodes.
        # ----------------------------------------------------

        all_finished = all(
            item["completed_episodes"]
            >= target_episodes
            for item in robot_instances
        )

        if all_finished:
            break

        global_step += 1

        # ----------------------------------------------------
        # 1. SELECT ACTIONS
        #
        # Finished robots receive action 0.
        # Active robots select their own action.
        # ----------------------------------------------------

        actions = []

        for item in robot_instances:

            if item["completed_episodes"] >= target_episodes:

                actions.append(0)
                continue

            if item["done"]:

                actions.append(0)
                continue

            action = item["agent"].select_action(
                item["state"]
            )

            actions.append(action)

        # ----------------------------------------------------
        # 2. SYNCHRONIZED PHYSICAL STEP
        # ----------------------------------------------------

        results = synchronized_step(
            robot_instances,
            actions,
            training["step_dt"]
        )

        # ----------------------------------------------------
        # 3. PROCESS EACH ROBOT INDEPENDENTLY
        # ----------------------------------------------------

        for index, item in enumerate(
            robot_instances
        ):

            # Robot already completed all requested episodes
            if item["completed_episodes"] >= target_episodes:
                continue

            # Robot was done at the beginning of this
            # simulation step and should not be processed.
            if item["done"]:
                continue

            next_state, reward, done, info = (
                results[index]
            )

            # ------------------------------------------------
            # DQN replay
            # ------------------------------------------------

            if next_state is not None:

                item["agent"].remember(
                    item["state"],
                    actions[index],
                    reward,
                    next_state,
                    done
                )

                item["agent"].learn()

                item["state"] = next_state

            # ------------------------------------------------
            # Episode statistics
            # ------------------------------------------------

            item["episode_reward"] += reward
            item["episode_steps"] += 1

            # ------------------------------------------------
            # Persistent per-robot step log
            # ------------------------------------------------

            log_robot_step(
                robot_loggers[item["name"]],
                item["local_episode"],
                item["episode_steps"],
                actions[index],
                reward,
                info,
                info.get(
                    "linear_velocity",
                    0.0
                ),
                info.get(
                    "angular_velocity",
                    0.0
                )
            )

            # ------------------------------------------------
            # Terminal state
            # ------------------------------------------------

            item["done"] = done

            rospy.loginfo(
                "[GLOBAL %d] %s | "
                "episode=%d | "
                "step=%d | "
                "action=%d | "
                "reward=%.4f | "
                "distance=%.4f | "
                "done=%s",
                global_step,
                item["name"],
                item["local_episode"],
                item["episode_steps"],
                actions[index],
                reward,
                info.get(
                    "distance",
                    -1.0
                ),
                done
            )

            # ------------------------------------------------
            # ROBOT FINISHED ITS OWN EPISODE
            # ------------------------------------------------

            if done:

                item["agent"].end_episode()

                item["completed_episodes"] += 1

                completed = (
                    item["completed_episodes"]
                )

                rospy.loginfo(
                    "[EPISODE COMPLETE] %s | "
                    "episode=%d/%d | "
                    "reward=%.4f | "
                    "steps=%d | "
                    "goal=%s | "
                    "collision=%s",
                    item["name"],
                    completed,
                    target_episodes,
                    item["episode_reward"],
                    item["episode_steps"],
                    info.get("goal", False),
                    info.get("collision", False)
                )

                log_training(
                    training_log,
                    f"[EPISODE COMPLETE] "
                    f"{item['name']} "
                    f"episode={completed}/{target_episodes} "
                    f"reward={item['episode_reward']:.4f} "
                    f"steps={item['episode_steps']} "
                    f"goal={info.get('goal', False)} "
                    f"collision={info.get('collision', False)}"
                )

                # ------------------------------------------------
                # Immediately reset THIS robot.
                #
                # Other robots do NOT need to be done.
                # ------------------------------------------------

                if completed < target_episodes:

                    next_episode = (
                        completed + 1
                    )

                    state = reset_robot_episode(
                        item,
                        next_episode
                    )

                    if state is None:

                        rospy.logerr(
                            "%s reset failed "
                            "after episode %d",
                            item["name"],
                            completed
                        )

                        log_training(
                            training_log,
                            f"[RESET FAILED] "
                            f"{item['name']} "
                            f"episode={next_episode}"
                        )

                        rospy.signal_shutdown(
                            "Robot reset failed"
                        )

                        break

                    log_training(
                        training_log,
                        f"[RESET] "
                        f"{item['name']} "
                        f"episode={next_episode}"
                    )

    # ========================================================
    # CLEANUP
    # ========================================================

    for item in robot_instances:

        item["robot"].stop()

    log_training(
        training_log,
        "========== TRAINING FINISHED =========="
    )

    for item in robot_instances:

        log_training(
            training_log,
            f"{item['name']}: "
            f"{item['completed_episodes']} "
            f"episodes completed"
        )

    training_log.close()

    for logger in robot_loggers.values():
        logger["file"].close()

    rospy.loginfo(
        "========== DQN MULTI-ROBOT TRAINING FINISHED =========="
    )

    rospy.loginfo(
        "Logs: %s",
        run_dir
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()