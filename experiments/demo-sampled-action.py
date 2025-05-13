#!/usr/bin/env python

"""
This script is used to test the custom robot and environment installation.

Run with: python demo-sampled-action.py
"""

import gymnasium as gym

# Disable import warnings since gymnasium imports are based on strings
from twsim.envs import terrain  # noqa: F401
from twsim.robots import transwheel  # noqa: F401

env = gym.make("Terrain-env")
env.unwrapped.print_sim_details()  # type: ignore

# Reset with a seed for determinism
obs, _ = env.reset(seed=0)

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
