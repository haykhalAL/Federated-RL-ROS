import yaml
import os

def load_robot_config():
    config_path = os.path.join(os.path.dirname(__file__), "../config/robots_loader.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["robots"]

if __name__ == "__main__":
    robots = load_robot_config()
    for r in robots:
        print(f"Robot: {r['name']} | Spawn: {r['spawn']} | Loggers: {r['loggers']}")
