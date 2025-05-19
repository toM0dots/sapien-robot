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
still = torch.zeros(4)
forward = torch.ones(4) * normalized_speed
rotate_left = torch.tensor([1, 1, -1, -1]) * normalized_speed

extensions_0p = torch.ones(4) * -1
extensions_10p = torch.ones(4) * -0.8
extensions_50p = torch.ones(4) * 0.0

# TODO: set more explicit values
action_sequence = [
    torch.cat((still, extensions_10p)),
    torch.cat((forward, extensions_10p)),
    torch.cat((still, extensions_0p)),
    torch.cat((rotate_left, extensions_0p)),
    torch.cat((still, extensions_10p)),
    torch.cat((forward, extensions_10p)),
    torch.cat((still, extensions_50p)),
    torch.cat((forward, extensions_50p)),
]

obs, _ = env.reset(seed=0)

num_steps_per_action = 50
max_steps = 500

old_action_index = -1

for stepi in range(max_steps):
    action_index = int(stepi / num_steps_per_action)
    if action_index >= len(action_sequence):
        break

    if action_index != old_action_index:
        print(f"Step {stepi}: action {action_index + 1} of {len(action_sequence)}")
        old_action_index = action_index

    action = action_sequence[action_index]

    obs, reward, terminated, truncated, info = env.step(action)

    if args.video:
        recorder.capture_image(env.render())

    # done = terminated or truncated
    # if done:
    #     break


env.close()

if args.video:
    recorder.save_as_video(args.video, overwrite=True)
