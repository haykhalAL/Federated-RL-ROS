#!/usr/bin/env python3

import csv
import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


class DQNLogger:

    def __init__(self, base_dir="logs/dqn"):

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.log_dir = os.path.join(
            base_dir,
            timestamp
        )

        os.makedirs(
            self.log_dir,
            exist_ok=True
        )

        self.episodes_path = os.path.join(
            self.log_dir,
            "episodes.csv"
        )

        self.steps_path = os.path.join(
            self.log_dir,
            "steps.csv"
        )

        self.writer = SummaryWriter(
            log_dir=os.path.join(
                self.log_dir,
                "tensorboard"
            )
        )

        self.episode_file = open(
            self.episodes_path,
            "w",
            newline=""
        )

        self.step_file = open(
            self.steps_path,
            "w",
            newline=""
        )

        self.episode_writer = csv.writer(
            self.episode_file
        )

        self.step_writer = csv.writer(
            self.step_file
        )

        self.episode_writer.writerow([
            "episode",
            "total_reward",
            "steps",
            "success",
            "collision_count",
            "final_distance",
            "path_length",
            "epsilon"
        ])

        self.step_writer.writerow([
            "episode",
            "step",
            "x",
            "y",
            "yaw",
            "goal_x",
            "goal_y",
            "distance_to_goal",
            "action",
            "reward",
            "collision"
        ])

        self.episode_file.flush()
        self.step_file.flush()

    def log_step(
        self,
        episode,
        step,
        info,
        action,
        reward
    ):

        pose = info.get("pose")
        goal_pose = info.get("goal_pose")

        x = None
        y = None
        yaw = None

        goal_x = None
        goal_y = None

        if pose is not None:
            x, y, yaw = pose

        if goal_pose is not None:
            goal_x, goal_y, _ = goal_pose

        self.step_writer.writerow([
            episode,
            step,
            x,
            y,
            yaw,
            goal_x,
            goal_y,
            info.get("distance"),
            action,
            reward,
            int(info.get("collision", False))
        ])

    def log_episode(
        self,
        episode,
        total_reward,
        steps,
        success,
        collision_count,
        final_distance,
        path_length,
        epsilon
    ):

        self.episode_writer.writerow([
            episode,
            total_reward,
            steps,
            int(success),
            collision_count,
            final_distance,
            path_length,
            epsilon
        ])

        self.episode_file.flush()
        self.step_file.flush()

        self.writer.add_scalar(
            "Training/EpisodeReward",
            total_reward,
            episode
        )

        self.writer.add_scalar(
            "Training/EpisodeSteps",
            steps,
            episode
        )

        self.writer.add_scalar(
            "Training/Success",
            int(success),
            episode
        )

        self.writer.add_scalar(
            "Training/Collisions",
            collision_count,
            episode
        )

        self.writer.add_scalar(
            "Training/FinalDistance",
            final_distance,
            episode
        )

        self.writer.add_scalar(
            "Training/PathLength",
            path_length,
            episode
        )

        self.writer.add_scalar(
            "Training/Epsilon",
            epsilon,
            episode
        )

    def close(self):

        self.episode_file.close()
        self.step_file.close()

        self.writer.close()