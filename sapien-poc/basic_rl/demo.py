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

# gym.make to gym.make_vec?
# Setup the environment and robot
env = gym.make("Terrain-env", 
               robot_uids="tw_robot", 
               render_mode="rgb_array", # When rendering the robot, the camera is facing the front of the robot (so it may appear reversed)
               control_mode="pd_joint_delta_pos",
               human_render_camera_configs=dict(shader_pack="rt"),
               )


from stable_baselines3 import A2C
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=6_000)
# model.learn(total_timesteps=3)

# Prepare snapshots/recording
recorder = RobotRecorder()


capture_i = 0

# custom_actions = np.array([ [50.0, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0],
#                     # [50.0, -50.0, 50.0, -50.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
#                     [0.0, 0.0, 0.0, 0.0, 3.0, 2.0, 1.0, 0.0],
#                     [20.0, 80.0, 50.0, 100.0, 0.0, 1.0, 2.0, 3.0],
#                     [75.0, 75.0, 75.0, 75.0, 0.0, 1.0, 2.0, 3.0],
#                     [70.0, 80.0, 90.0, 90.0, 0.0, 1.0, 2.0, 3.0],
#                     [50.0, 50.0, 50.0, 50.0, 0.0, 1.0, 2.0, 3.0],
#                     [50.0, 50.0, 50.0, 50.0, 0.0, 1.0, 2.0, 3.0],
#                     [50.0, 50.0, 50.0, 50.0, 2.0, 2.0, 2.0, 2.0],
#                     [50.0, 50.0, 50.0, 50.0, 2.0, 2.0, 2.0, 2.0],
#                     [50.0, 50.0, 50.0, 50.0, 0.0, 1.0, 2.0, 3.0],
#                     [50.0, 50.0, 50.0, 50.0, 0.0, 1.0, 2.0, 3.0],
#                     [50.0, 50.0, 50.0, 50.0, 2.0, 2.0, 2.0, 2.0],
#                     [50.0, 50.0, 50.0, 50.0, 2.0, 2.0, 2.0, 2.0],
#                     [50.0, 50.0, 50.0, 50.0, 2.0, 2.0, 2.0, 2.0], ])

vec_env = model.get_env()
obs, _ = env.reset()
for i in range(150):
    action, _state = model.predict(obs, deterministic=True)
    # action = custom_actions[int(i / 25)]
    # if(i == 25 or i == 75):
    #     print("force reset")
    #     env.reset()

    
    # print(str(i)+": ",action)
    obs, reward, terminated, truncated, info = env.step(action)
    # obs, reward, done, info = vec_env.step(action)
    # obs, reward, terminated, truncated = env.step(action)
    # print(str(i)+": ",action,"\n",obs)
    # vec_env.render("rgb_array")
    
    recorder.capture_image(env.render())
    
    # VecEnv resets automatically
    if terminated: #done:
    #   obs = vec_env.reset()
        print("Terminated")
        break

# vec_env.close()
env.close()

recorder.create_video("rl_output.mp4")


raise SystemExit("Done")

'''
Agent {          
'qpos': tensor([[   -3.2177,  0.8464,  2.8541,  4.5857,  2.4664,  2.4662,  2.4660,  1.2347,
                    1.2283,  1.2214,  0.4652,  0.4650,  0.4646,  0.9906,  0.9943,  0.9988]]), 
'qvel': tensor([[   3.9682e+01,  5.9672e+01, -1.4642e+01, -5.8066e+01, -1.5829e-02,
                    1.2169e-02,  5.2137e-02, -1.6601e+00, -1.0892e+00, -4.3318e-01,
                    -3.0137e-01, -2.9174e-01, 
                    
                    # Maybe velocities of wheels? 
                    -2.7325e-01, -5.1538e-01, -9.5596e-01,-1.5200e+00]]), 
'controller': {
    'extension_joint': {
        'target_qpos': # Def wheel extensions
        tensor([[   2.4665, 2.4665, 2.4665, 1.2171, 1.2171, 1.2171, 0.4619, 0.4619, 0.4619,
                    0.9836, 0.9836, 0.9836]])}}}
'''