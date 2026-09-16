#!/usr/bin/env python3

import os
import yaml
import rospkg

import train_dqn


def load_yaml():

    rospack = rospkg.RosPack()

    pkg_path = rospack.get_path("dev_fl_robot_pkg")

    config_path = os.path.join(
        pkg_path,
        "config",
        "experiment_config.yaml"
    )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():

    config = load_yaml()

    algorithm = config["experiment"]["algorithm"]

    if algorithm == "dqn":

        train_dqn.train(config)

    elif algorithm == "ppo":

        raise NotImplementedError("PPO trainer not implemented.")

    elif algorithm == "qlearning":

        raise NotImplementedError("Q-Learning trainer not implemented.")

    else:

        raise RuntimeError(
            f"Unknown algorithm: {algorithm}"
        )


if __name__ == "__main__":

    main()