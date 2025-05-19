#!/usr/bin/env python3

"""
This script is used to test the custom robot and environment without learning.

Run with: python demo-fixed-actions.py
"""

from argparse import ArgumentParser

import gymnasium as gym
import torch

# Disable import warnings since gymnasium imports are based on strings
from twsim.envs import plane  # noqa: F401
from twsim.robots import transwheel  # noqa: F401
from twsim.utils import RobotRecorder

parser = ArgumentParser(description="Fixed action sequence demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--video", type=str, help="Output a video.")
args = parser.parse_args()

env = gym.make("Plane-v1", render_mode="rgb_array", num_envs=args.num_envs)

env.unwrapped.print_sim_details()  # type: ignore
print(f"{env.unwrapped.reward_mode=}")  # type: ignore


if args.video:
    # TODO: recorder should take into account the number of envs
    recorder = RobotRecorder(output_dir="./output_images", fps=30, overwrite=True)

normalized_speed = 0.2
forward = torch.ones(4) * normalized_speed

normalized_extension = 0.0
extensions = torch.ones(4) * normalized_extension

# TODO: set more explicit values
action_sequence = [
    # torch.zeros_like(env.action_space.sample())
    torch.cat((forward, extensions)),
    torch.cat((forward, extensions)),
    torch.cat((forward, extensions)),
    torch.cat((forward, extensions)),
    # [62.825, 62.825, 62.825, 62.825, 0.0, 0.0, 0.0, 0.0],
    # [62.825, -62.825, 62.825, -62.825, 0.0, 0.0, 0.0, 0.0],
    # [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
    # [62.825, 62.825, 62.825, 62.825, 0.0, 1.0, 2.0, 3.0],
    # [62.825, 62.825, 62.825, 62.825, 0.0, 1.0, 2.0, 3.0],
    # [62.825, 62.825, 62.825, 62.825, 0.0, 1.0, 2.0, 3.0],
    # [62.825, 62.825, 62.825, 62.825, 0.0, 1.0, 2.0, 3.0],
    # [62.825, 62.825, 62.825, 62.825, 0.0, 1.0, 2.0, 3.0],
    # [62.825, 62.825, 62.825, 62.825, 2.0, 2.0, 2.0, 2.0],
    # [62.825, 62.825, 62.825, 62.825, 2.0, 2.0, 2.0, 2.0],
    # [62.825, 62.825, 62.825, 62.825, 0.0, 1.0, 2.0, 3.0],
    # [62.825, 62.825, 62.825, 62.825, 0.0, 1.0, 2.0, 3.0],
    # [62.825, 62.825, 62.825, 62.825, 2.0, 2.0, 2.0, 2.0],
    # [62.825, 62.825, 62.825, 62.825, 2.0, 2.0, 2.0, 2.0],
    # [62.825, 62.825, 62.825, 24, 2.0, 2.0, 2.0, 2.0],
]

obs, _ = env.reset(seed=0)

num_steps_per_action = 25
max_steps = 500

for stepi in range(max_steps):
    action_index = int(stepi / num_steps_per_action)
    if action_index >= len(action_sequence):
        break

    action = action_sequence[action_index]

    obs, reward, terminated, truncated, info = env.step(action)

    if args.video:
        recorder.capture_image(env.render())

    done = terminated or truncated
    if done:
        break


env.close()

if args.video:
    recorder.save_as_video(args.video, overwrite=True)
