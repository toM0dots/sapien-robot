#!/usr/bin/env python3

"""
This script is used to test the custom robot and environment without learning.

Run with: python demo-fixed-actions.py
"""

# TODO: unify demos so that they have the same arguments and setup
# TODO: look into reply_trajectory

from dataclasses import dataclass

import gymnasium as gym
import torch
import tyro
from mani_skill.utils.wrappers.record import RecordEpisode

from twsim import envs  # noqa: F401
from twsim.robots import transwheel  # noqa: F401


@dataclass
class Args:
    """Experiment configuration and arguments."""

    # fmt: off

    env_id: str = "Plane-v1"  # Environment ID
    num_envs: int = 1         # Number of environments
    video: bool = False       # Output a video

    # fmt: on


args = tyro.cli(Args)
num_envs = args.num_envs

env = gym.make(args.env_id, render_mode="rgb_array", num_envs=num_envs)

if args.video:
    env = RecordEpisode(
        env,  # type: ignore
        output_dir="./",
        save_trajectory=False,
        info_on_video=True,
        max_steps_per_video=500,
    )

env.unwrapped.print_sim_details()  # type: ignore
print(f"{env.unwrapped.reward_mode=}")  # type: ignore
print("# -------------------------------------------------------------------------- #")

normalized_speed = 0.2
still = torch.zeros(num_envs, 4)
forward = torch.ones(num_envs, 4) * normalized_speed
rotate_left = torch.tensor([[1, 1, -1, -1]] * num_envs) * normalized_speed

extensions_0p = torch.ones(num_envs, 4) * -1
extensions_10p = torch.ones(num_envs, 4) * -0.8
extensions_50p = torch.ones(num_envs, 4) * 0.0

action_sequence = [
    torch.cat((still, extensions_10p), dim=-1),
    torch.cat((forward, extensions_10p), dim=-1),
    torch.cat((still, extensions_0p), dim=-1),
    torch.cat((rotate_left, extensions_0p), dim=-1),
    torch.cat((still, extensions_10p), dim=-1),
    torch.cat((forward, extensions_10p), dim=-1),
    torch.cat((still, extensions_50p), dim=-1),
    torch.cat((forward, extensions_50p), dim=-1),
]

obs, _ = env.reset(seed=0)

num_steps_per_action = 50
max_steps = 500

old_action_index = -1

print("Starting demo with fixed actions...")

for stepi in range(max_steps):
    action_index = int(stepi / num_steps_per_action)
    if action_index >= len(action_sequence):
        break

    if action_index != old_action_index:
        # print(f"Step {stepi}: action {action_index + 1} of {len(action_sequence)}")
        old_action_index = action_index

    action = action_sequence[action_index]

    obs, reward, terminated, truncated, info = env.step(action)

    # No need to check this condition when running a sequence
    # done = terminated or truncated
    # if done: break


env.close()

print("Demo finished.")
