#!/usr/bin/env bash

cp rllib/envs/gazebo/assets/launch/reacher_env.launch ~/catkin_ws/src/rr-robot-plugin/launch/
cp rllib/envs/gazebo/assets/worlds/reacher_env.world ~/catkin_ws/src/rr-robot-plugin/worlds/
cp -r rllib/envs/gazebo/assets/models/ball ~/.gazebo/models/

