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
    tensor = np.array(tensor.squeeze(), dtype=np.uint8)
    return Image.fromarray(tensor)

# Setup the environment and robot
env = gym.make("Terrain-env", 
               robot_uids="tw_robot", 
               render_mode="rgb_array", 
               human_render_camera_configs=dict(shader_pack="rt"),)

env.print_sim_details()

# Prepare snapshots/recording
image_folder = './image_output'
image_dir = Path(image_folder)
if image_dir.exists():
    rmtree(image_dir)
image_dir.mkdir()

print("being all")
print(env.action_space)
print("-1")
print(env.action_space.sample())
print("-2")
print(env.action_space.sample())
print("end samples")

# raise SystemExit
# Simulation loop
capture_i = 0
simulation_steps = 100
obs, _ = env.reset(seed=0)
for i in range(simulation_steps):
    
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated | truncated
    print(f"Step: {i}, Obs shape: {obs.shape}, Reward shape {reward.shape}, Done shape {done.shape}")

    # Process and save snapshot
    image = tensor_to_image(env.render_rgb_array())
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




# ['extension_joint_front_left_0', 'extension_joint_front_left_1', 'extension_joint_front_left_2', 'extension_joint_front_right_0', 'extension_joint_front_right_1', 'extension_joint_front_right_2', 'extension_joint_rear_left_0', 'extension_joint_rear_left_1', 'extension_joint_rear_left_2', 'extension_joint_rear_right_0', 'extension_joint_rear_right_1', 'extension_joint_rear_right_2']
# ['wheel_joint_front_left', 'wheel_joint_front_right', 'wheel_joint_rear_left', 'wheel_joint_rear_right']
# ['extension_joint_front_left_0', 'extension_joint_front_left_1', 'extension_joint_front_left_2', 'extension_joint_front_right_0', 'extension_joint_front_right_1', 'extension_joint_front_right_2', 'extension_joint_rear_left_0', 'extension_joint_rear_left_1', 'extension_joint_rear_left_2', 'extension_joint_rear_right_0', 'extension_joint_rear_right_1', 'extension_joint_rear_right_2']
# ['wheel_joint_front_left', 'wheel_joint_front_right', 'wheel_joint_rear_left', 'wheel_joint_rear_right']
# Box(0.0, 3.1415927, (12,), float32)
# [0.47070292 2.6439304  2.453756   2.9150815  1.4896942  2.898555
#  2.8408165  2.9836972  2.309478   1.8424743  0.9898338  2.7874093 ]
# [0.2609273  3.02554    2.1249518  2.459501   1.2093369  2.8597262
#  1.0109758  1.6923621  0.76296705 1.824136   0.5914434  0.7782967 ]