#!/usr/bin/env python3
import os
import yaml
import math
import uuid

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../config/robots_config.yaml"
)
CONFIG_PATH = os.path.abspath(CONFIG_PATH)

# Terminator config
TERMINATOR_CONFIG = os.path.expanduser("~/.config/terminator/config")

terminal_counter = 0
child_counter = 0

# Mapping between logger names and topic names
LOGGER_TOPICS = {
    "odom": "odom",
    "lidar": "scan",
    "velocity": "cmd_vel",
    "location": "amcl_pose",
    "battery": "battery_state",
    "camera": "camera/rgb/image_raw"
}

def make_layout(loggers, parent, depth, robot_name, order_in_parent):
    """
    Recursively build a balanced split.
    Returns (section_lines, top_node_name) where top_node_name is the name
    of the section that is this subtree's immediate child of `parent`.
    """
    global terminal_counter, child_counter

    # Base: single terminal
    if len(loggers) == 1:
        term_name = f"terminal_{terminal_counter}"
        terminal_counter += 1

        lines = []
        lines.append(f"    [[[{term_name}]]]")
        lines.append("      type = Terminal")
        lines.append(f"      parent = {parent}")
        lines.append(f"      order = {order_in_parent}")
        lines.append("      profile = default")
        lines.append(f"      uuid = {uuid.uuid4()}")

        logger = loggers[0]
        topic = LOGGER_TOPICS.get(logger, logger)
        topic_full = f"/{robot_name}/{topic}"

        cmd = (
            f'bash -c "'
            f'source /opt/ros/noetic/setup.bash; '
            f'source ~/federated-learning-ROS/devel/setup.bash; '
            f'rosrun dev_fl_robot_pkg topic_logger.py {topic_full} '
            f'__name:={robot_name}_{logger}_logger; exec bash"'
        )
        lines.append(f"      command = {cmd}\n")
        return lines, term_name

    # Split: more than one logger
    half = math.ceil(len(loggers) / 2)
    left_list, right_list = loggers[:half], loggers[half:]

    paned_name = f"child{child_counter}"
    child_counter += 1
    split_type = "VPaned" if depth % 2 == 0 else "HPaned"

    # Recurse left/right; they will create sections whose parent is `paned_name`
    left_lines, left_top = make_layout(left_list, paned_name, depth + 1, robot_name, 0)
    right_lines, right_top = make_layout(right_list, paned_name, depth + 1, robot_name, 1)

    # Now we can write the Paned section WITH its children list
    paned_lines = []
    paned_lines.append(f"    [[[{paned_name}]]]")
    paned_lines.append(f"      type = {split_type}")
    paned_lines.append(f"      parent = {parent}")
    paned_lines.append(f"      order = {order_in_parent}")
    paned_lines.append("      position = 364")
    paned_lines.append("      ratio = 0.5")
    paned_lines.append(f"      children = [{left_top}, {right_top}]\n")

    # Return: Paned section first, then its children sections
    return paned_lines + left_lines + right_lines, paned_name


def create_robot_layout(robot_name, loggers):
    """Produce one [[<robot>_layout]] block with a Window that owns the subtree."""
    global terminal_counter, child_counter
    terminal_counter = 0
    child_counter = 1  # reserve child0 for Window

    # Build the subtree first to know the immediate child name
    subtree_lines, top_node = make_layout(loggers, "child0", 0, robot_name, 0)

    layout = []
    layout.append(f"  [[{robot_name}_layout]]")

    # Window section (include its children list here)
    layout.append("    [[[child0]]]")
    layout.append("      type = Window")
    layout.append('      parent = ""')
    layout.append("      order = 0")
    layout.append("      position = 200:200")
    layout.append("      size = 800, 600")
    layout.append(f"      children = [{top_node}]\n")

    # Then append the subtree sections
    layout.extend(subtree_lines)

    return "\n".join(layout)


def update_terminator_config(layouts):
    """Replace the [layouts] section with our generated layouts."""
    if os.path.exists(TERMINATOR_CONFIG):
        with open(TERMINATOR_CONFIG, "r") as f:
            content = f.read()
    else:
        content = "[global_config]\n[keybindings]\n[profiles]\n[layouts]\n[plugins]\n"

    # Split out head/tail so we can drop-in our layouts
    parts = content.split("[layouts]")
    head = parts[0]
    tail = ""
    if len(parts) > 1 and "[plugins]" in parts[1]:
        tail = parts[1].split("[plugins]")[1]

    new_content = head + "[layouts]\n" + "\n".join(layouts) + "\n[plugins]\n" + tail

    with open(TERMINATOR_CONFIG, "w") as f:
        f.write(new_content)


if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    layouts = []
    for robot in cfg["robots"]:
        if not robot.get("deploy", True):
            continue
        loggers = robot.get("loggers", [])
        # guard: at least 1 logger
        if not loggers:
            continue
        layouts.append(create_robot_layout(robot["name"], loggers))

    update_terminator_config(layouts)
    print(f"✅ Applied {len(layouts)} layout(s) to {TERMINATOR_CONFIG}")
