# Primary imports
import gymnasium as gym
import numpy as np

import sys
import os
# Add parent directory to import robot and terrain
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tw_robot
import terrain_env
from robot_recorder import RobotRecorder

# Setup the environment and robot
env = gym.make("Terrain-env", 
               robot_uids="tw_robot", 
               render_mode="rgb_array", # When rendering the robot, the camera is facing the front of the robot (so it may appear reversed)
               control_mode="pd_joint_delta_pos",
               human_render_camera_configs=dict(shader_pack="rt"),
               )

# env.print_sim_details()

recorder = RobotRecorder()

custom_actions = np.array([ [50.0, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0],
                    [50.0, -50.0, 50.0, -50.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
                    [50.0, 50.0, 50.0, 50.0, 0.0, 1.0, 2.0, 3.0],
                    [50.0, 50.0, 50.0, 50.0, 0.0, 1.0, 2.0, 3.0],
                    [50.0, 50.0, 50.0, 50.0, 0.0, 1.0, 2.0, 3.0],
                    [50.0, 50.0, 50.0, 50.0, 0.0, 1.0, 2.0, 3.0],
                    [50.0, 50.0, 50.0, 50.0, 0.0, 1.0, 2.0, 3.0],
                    [50.0, 50.0, 50.0, 50.0, 2.0, 2.0, 2.0, 2.0],
                    [50.0, 50.0, 50.0, 50.0, 2.0, 2.0, 2.0, 2.0],
                    [50.0, 50.0, 50.0, 50.0, 0.0, 1.0, 2.0, 3.0],
                    [50.0, 50.0, 50.0, 50.0, 0.0, 1.0, 2.0, 3.0],
                    [50.0, 50.0, 50.0, 50.0, 2.0, 2.0, 2.0, 2.0],
                    [50.0, 50.0, 50.0, 50.0, 2.0, 2.0, 2.0, 2.0],
                    [50.0, 50.0, 50.0, 50.0, 2.0, 2.0, 2.0, 2.0], ])


capture_i = 0
simulation_steps = 300
obs, _ = env.reset(seed=0)
for i in range(simulation_steps):
    
    # action = env.action_space.sample()
    action = custom_actions[int(i / 25)]
    if(i == 25):
        print("force reset")
        env.reset()

    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated | truncated
    print(f"Done: {done}, Term: {terminated}, Trunc: {truncated}")
    print(f"Step: {i}, Obs shape: {obs.shape}, Reward shape {reward.shape}, Done shape {done.shape}")

    if(terminated):
        print("Resetting sim...")
        env.reset()

    recorder.capture_image(env.render())

env.close()

recorder.create_video("output_video.mp4")
