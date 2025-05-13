#!/usr/bin/env python3

"""
This script is used to test the custom robot and environment without learning.

Run with: python demo-fixed-actions.py
"""

import gymnasium as gym
import numpy as np

# Disable import warnings since gymnasium imports are based on strings
from twsim.envs import terrain  # noqa: F401
from twsim.robots import transwheel  # noqa: F401
from twsim.utils import RobotRecorder

env = gym.make("Terrain-env")
env.unwrapped.print_sim_details()  # type: ignore

recorder = RobotRecorder(output_dir="./output_images", fps=30, overwrite=True)

# 24 rpm = 62.825 rad/s,    Sim freq = 100          Maybe 25.13
# TODO: set more explicit values
action_sequence = np.array(
    [
        [62.825, 62.825, 62.825, 62.825, 0.0, 0.0, 0.0, 0.0],
        [62.825, -62.825, 62.825, -62.825, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        [62.825, 62.825, 62.825, 62.825, 0.0, 1.0, 2.0, 3.0],
        [62.825, 62.825, 62.825, 62.825, 0.0, 1.0, 2.0, 3.0],
        [62.825, 62.825, 62.825, 62.825, 0.0, 1.0, 2.0, 3.0],
        [62.825, 62.825, 62.825, 62.825, 0.0, 1.0, 2.0, 3.0],
        [62.825, 62.825, 62.825, 62.825, 0.0, 1.0, 2.0, 3.0],
        [62.825, 62.825, 62.825, 62.825, 2.0, 2.0, 2.0, 2.0],
        [62.825, 62.825, 62.825, 62.825, 2.0, 2.0, 2.0, 2.0],
        [62.825, 62.825, 62.825, 62.825, 0.0, 1.0, 2.0, 3.0],
        [62.825, 62.825, 62.825, 62.825, 0.0, 1.0, 2.0, 3.0],
        [62.825, 62.825, 62.825, 62.825, 2.0, 2.0, 2.0, 2.0],
        [62.825, 62.825, 62.825, 62.825, 2.0, 2.0, 2.0, 2.0],
        [62.825, 62.825, 62.825, 24, 2.0, 2.0, 2.0, 2.0],
    ]
)

obs, _ = env.reset(seed=0)

num_steps_per_action = 25
max_steps = 100

for stepi in range(max_steps):
    action = action_sequence[int(stepi / num_steps_per_action)]

    obs, reward, terminated, truncated, info = env.step(action)
    # recorder.capture_image(env.render())

    done = terminated or truncated
    if done:
        break


env.close()
# recorder.create_video("output_video.mp4")
