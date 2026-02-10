#!/bin/bash

ROBOT_NAME=$1
LOGGER_COUNT=$2

source /opt/ros/noetic/setup.bash
source ~/federated-learning-ROS/devel/setup.bash

for i in $(seq 1 $LOGGER_COUNT); do
    terminator -l "${ROBOT_NAME}_layout" &
done
