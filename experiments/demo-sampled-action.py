#!/usr/bin/env python

"""
This script is used to test the custom robot and environment installation.

Run with: python demo-sampled-action.py
"""

# TODO: add command line arguments (e.g., num_envs)

import gymnasium as gym

# Disable import warnings since gymnasium imports are based on strings
# TODO: change to plane env
from twsim.envs import terrain  # noqa: F401
from twsim.robots import transwheel  # noqa: F401

env = gym.make("Terrain-env")
env.unwrapped.print_sim_details()  # type: ignore


# Reset with a seed for determinism
obs, _ = env.reset(seed=0)

max_steps = 100
for _ in range(max_steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated
    if done:
        break

env.close()
