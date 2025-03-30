# Primary imports
import gymnasium as gym
import numpy as np

# Robot and Env
import tw_robot
import terrain_env

# Image/recording processing
from PIL import Image
from shutil import rmtree
from pathlib import Path
import cv2
import os



def tensor_to_image(tensor):
    tensor = np.array(tensor.squeeze().cpu(), dtype=np.uint8)
    return Image.fromarray(tensor)





# Setup the environment and robot
env = gym.make("Terrain-env", 
               robot_uids="tw_robot", 
               render_mode="rgb_array", # When rendering the robot, the camera is facing the front of the robot (so it may appear reversed)
               control_mode="pd_joint_delta_pos",
               human_render_camera_configs=dict(shader_pack="rt"),
               )

'''
Agent {'qpos': tensor([[-3.2177,  0.8464,  2.8541,  4.5857,  2.4664,  2.4662,  2.4660,  1.2347,
          1.2283,  1.2214,  0.4652,  0.4650,  0.4646,  0.9906,  0.9943,  0.9988]]), 'qvel': tensor([[ 3.9682e+01,  5.9672e+01, -1.4642e+01, -5.8066e+01, -1.5829e-02,
          1.2169e-02,  5.2137e-02, -1.6601e+00, -1.0892e+00, -4.3318e-01,
         -3.0137e-01, -2.9174e-01, -2.7325e-01, -5.1538e-01, -9.5596e-01,
         -1.5200e+00]]), 'controller': {'extension_joint': {'target_qpos': tensor([[2.4665, 2.4665, 2.4665, 1.2171, 1.2171, 1.2171, 0.4619, 0.4619, 0.4619,
         0.9836, 0.9836, 0.9836]])}}}
'''

# env.print_sim_details()

# from stable_baselines3 import A2C
# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10_000)

# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(100):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     # vec_env.render("rgb_array")
#     # VecEnv resets automatically
#     if done:
#       obs = vec_env.reset()

# vec_env.close()
# env.close()
# raise SystemExit("Done")

# Prepare snapshots/recording
image_folder = './image_output'
image_dir = Path(image_folder)
if image_dir.exists():
    rmtree(image_dir)
image_dir.mkdir()


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
    # print(obs)
    done = terminated | truncated
    print(f"Done: {done}, Term: {terminated}, Trunc: {truncated}")
    print(f"Step: {i}, Obs shape: {obs.shape}, Reward shape {reward.shape}, Done shape {done.shape}")

    if(terminated):
        print("Resetting sim...")
        env.reset()

    # Process and save snapshot
    image = tensor_to_image(env.render())
    image.save(f"image_output/cam{capture_i:05}.png")

    capture_i += 1

env.close()


# Compile simulation snapshots into a video
capture_fps = 25
video_name = 'output_video.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith('.png')]
images.sort() # Images get loaded out of order, need to organize them by name
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), capture_fps, (width, height))
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)
cv2.destroyAllWindows()
video.release()
