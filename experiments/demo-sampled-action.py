#!/usr/bin/env python

"""
This script is used to test the custom robot and environment installation.

Run with: python demo-sampled-action.py
"""

import gymnasium as gym

# Disable import warnings since gymnasium imports are based on strings
from twsim.envs import terrain  # noqa: F401
from twsim.robots import transwheel  # noqa: F401

env = gym.make(
    "Terrain-env",
    # num_envs=1,
    # obs_mode="state",
    # control_mode="pd_ee_delta_pose",  # there is also "pd_joint_delta_pos", ...
    # render_mode="human",
)

print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0)  # reset with a seed for determinism
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    # env.render()  # a display is required to render

env.close()
