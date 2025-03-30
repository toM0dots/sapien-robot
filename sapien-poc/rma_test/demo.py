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

import ppo as PPO
import module as ppo_module
import torch.nn as nn
import torch
import math
from ruamel.yaml import YAML, dump, RoundTripDumper
import time
from raisim_gym_helper import ConfigurationSaver
try:
    import wandb
except:
    wandb = None


def tensor_to_image(tensor):
    tensor = np.array(tensor.squeeze().cpu(), dtype=np.uint8)
    return Image.fromarray(tensor)

def unify_extensions(actions, num_wheel_extensions, display=False):
    # First four actions are for the wheels, ignore
    num_wheels = 4

    for wheel_num in range(num_wheels):
        first_extension_idx = num_wheels + wheel_num * num_wheel_extensions
        uniform_action = actions[first_extension_idx] # Get the sample for the first extension

        for extension_num in range(num_wheel_extensions):
            actions[first_extension_idx + extension_num] = uniform_action # Set all extensions of that wheel to the first's sample

    if display:
        print("--------- Actions --------")
        print(f"FLW Ext: {actions[num_wheels+ 0*num_wheel_extensions]:.2f}\tFRW Ext: {actions[num_wheels+ 1*num_wheel_extensions]:.2f}")
        print(f"{actions[0]:.2f}\t []-----[] {actions[1]:.2f}\n" +
                               "\t   |   |   \n" +
                               "\t   |   |   \n" +
              f"{actions[2]:.2f}\t []-----[] {actions[3]:.2f}")
        print(f"RLW Ext: {actions[num_wheels+ 2*num_wheel_extensions]:.2f}\tRRW Ext: {actions[num_wheels+ 3*num_wheel_extensions]:.2f}")
        print("---------------------------")
    return actions


# Setup the environment and robot
env = gym.make("Terrain-env", 
               robot_uids="tw_robot", 
               render_mode="rgb_array", # When rendering the robot, the camera is facing the front of the robot (so it may appear reversed)
               control_mode="pd_joint_delta_pos",
               human_render_camera_configs=dict(shader_pack="rt"),)

raise SystemExit

'''
# create environment from the configuration file
env = VecEnv(rsg_a1_task.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
'''

cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

init_var = 0.3
module_type = ppo_module.MLPEncode_wrap
ob_dim = 1
act_dim = 1
activation_fn_map = {'none': None, 'tanh': nn.Tanh}
output_activation_fn = activation_fn_map[cfg['architecture']['activation']]
small_init_flag = cfg['architecture']['small_init']
baseDim = cfg['environment']['baseDim']
geomDim = int(cfg['environment']['geomDim'])*int(cfg['environment']['use_slope_dots'])
n_futures = int(cfg['environment']['n_futures'])
device_type = 'cpu'#'cuda' #'cuda:{}'.format(args.gpu)

n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
saver = ConfigurationSaver(log_dir="./rsg_a1_task/something",
                           save_items=["./Environment.hpp", "./demo.py"], ) #config = cfg, overwrite = True
loaded_graph_flat = torch.jit.load("./policy_22000.pt", map_location=torch.device(device_type))
flat_expert = ppo_module.Steps_Expert(loaded_graph_flat, device=device_type, baseDim=42,
                                      geomDim=2, n_futures=1, num_g1=n_futures)
# env = VecEnv(rsg_a1_task.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

total_steps = n_steps * 1 #n_steps * env.num_envs
penalty_scale = np.array([cfg['environment']['lateralVelRewardCoeff'], cfg['environment']['angularVelRewardCoeff'], cfg['environment']['deltaTorqueRewardCoeff'], cfg['environment']['actionRewardCoeff'], cfg['environment']['sidewaysRewardCoeff'], cfg['environment']['jointSpeedRewardCoeff'], cfg['environment']['deltaContactRewardCoeff'], cfg['environment']['deltaReleaseRewardCoeff'], cfg['environment']['footSlipRewardCoeff'], cfg['environment']['upwardRewardCoeff'], cfg['environment']['workRewardCoeff'], cfg['environment']['yAccRewardCoeff'], 1., 1., 1.])
# penalty_scale = np.array([1, 1, 1,1,1, 1, 1, 1,1,1,1,1, 1., 1., 1.])

actor = ppo_module.Actor(module_type(cfg['architecture']['policy_net'],
                            nn.LeakyReLU,
                            ob_dim//2,
                            act_dim,
                            output_activation_fn,
                            small_init_flag,
                            base_obdim = baseDim,
                            geom_dim = geomDim,
                            n_futures = n_futures),
                            ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, init_var),
                            device_type)

critic = ppo_module.Critic(module_type(cfg['architecture']['value_net'],
                            nn.LeakyReLU,
                            ob_dim//2,
                            1,
                            base_obdim = baseDim,
                            geom_dim = geomDim,
                            n_futures = n_futures),
                            device_type)

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.997,
              lam=0.95,
              num_mini_batches=4,
              device=device_type,
              log_dir=saver.data_dir,
              mini_batch_sampling='in_order',
              learning_rate=5e-4,
              flat_expert=flat_expert
              )


avg_rewards = []
# '''
for update in range(500001) if args.loadid is None else range(args.loadid + 1, 500001):

    start = time.time()
    env.reset()
    reward_ll_sum = 0
    forwardX_sum = 0
    penalty_sum = 0
    done_sum = 0
    average_dones = 0.

    # actual training
    for step in range(n_steps):
        obs = env.observe()#not freeze_encoder)
        action = ppo.observe(obs)
        reward, dones = env.step(action)
        unscaled_reward_info = env.get_reward_info()
        forwardX = unscaled_reward_info[:, 0]
        penalty = unscaled_reward_info[:, 1:]
        ppo.step(value_obs=obs, rews=reward, dones=dones, infos=[])
        done_sum = done_sum + sum(dones)
        reward_ll_sum = reward_ll_sum + sum(reward)
        forwardX_sum += np.sum(forwardX)
        penalty_sum += np.sum(penalty, axis=0)

    env.curriculum_callback()

    # take st step to get value obs
    obs = env.observe()#not freeze_encoder)
    ppo.update(actor_obs=obs,
               value_obs=obs,
               log_this_iteration=update % 10 == 0,
               update=update)
    
    end = time.time()
    
    forwardX = forwardX_sum / total_steps
    forwardXReward = forwardX_sum * cfg['environment']['forwardVelRewardCoeff'] / total_steps

    forwardY, forwardZ, deltaTorque, action, sideways, jointSpeed, deltaContact, deltaRelease, footSlip, upward, work, yAcc, torqueSquare, stepHeight, walkedDist = penalty_sum / total_steps
    forwardYReward, forwardZReward, deltaTorqueReward, actionReward, sidewaysReward, jointSpeedReward, deltaContactReward, deltaReleaseReward, footSlipReward, upwardReward, workReward, yAccReward, torq, stepHeight, walkedDist = scaled_penalty = penalty_sum * penalty_scale / total_steps

    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    actor.distribution.enforce_minimum_std((torch.ones(12)*0.2).to(device_type))
    if wandb:
        wandb.log({'forwardX': forwardX, 
        'forwardX_reward': forwardXReward, 
        'forwardY': forwardY, 
        'forwardY_reward': forwardYReward, 
        'forwardZ': forwardZ, 
        'forwardZ_reward': forwardZReward, 
        'deltaTorque': deltaTorque, 
        'deltaTorque_reward': deltaTorqueReward, 
        'action': action, 
        'stepHeight': stepHeight,
        'action_reward': actionReward, 
        'sideways': sideways, 
        'sideways_reward': sidewaysReward, 
        'jointSpeed': jointSpeed, 
        'jointSpeed_reward': jointSpeedReward, 
        'deltaContact': deltaContact, 
        'deltaContact_reward': deltaContactReward, 
        'deltaRelease': deltaRelease, 
        'deltaRelease_reward': deltaReleaseReward, 
        'footSlip': footSlip, 
        'footSlip_reward': footSlipReward, 
        'upward': upward, 
        'upward_reward': upwardReward, 
        'work': work, 
        'work_reward': workReward, 
        'yAcc': yAcc, 
        'yAcc_reward': yAccReward,
        'torqueSquare': torqueSquare,
        'dones': average_dones,
        'walkedDist': walkedDist})

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("average forward reward: ", '{:0.10f}'.format(forwardXReward)))
    print('{:<40} {:>6}'.format("average penalty reward: ", ', '.join(['{:0.4f}'.format(r) for r in scaled_penalty])))
    print('{:<40} {:>6}'.format("average walked dist: ", '{:0.10f}'.format(scaled_penalty[-1])))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("lr: ", '{:.4e}'.format(ppo.optimizer.param_groups[0]["lr"])))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('std: ')
    print(np.exp(actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')
    # '''



env.print_sim_details()

# Prepare snapshots/recording
image_folder = './image_output'
image_dir = Path(image_folder)
if image_dir.exists():
    rmtree(image_dir)
image_dir.mkdir()


# raise SystemExit
# Simulation loop
capture_i = 0
simulation_steps = 25
obs, _ = env.reset(seed=0)
for i in range(simulation_steps):
    
    action = env.action_space.sample()
    action = unify_extensions(action, env.agent.num_wheel_extensions, display=True)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated | truncated
    print(f"Step: {i}, Obs shape: {obs.shape}, Reward shape {reward.shape}, Done shape {done.shape}")

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




# ['extension_joint_front_left_0', 'extension_joint_front_left_1', 'extension_joint_front_left_2', 'extension_joint_front_right_0', 'extension_joint_front_right_1', 'extension_joint_front_right_2', 'extension_joint_rear_left_0', 'extension_joint_rear_left_1', 'extension_joint_rear_left_2', 'extension_joint_rear_right_0', 'extension_joint_rear_right_1', 'extension_joint_rear_right_2']
# ['wheel_joint_front_left', 'wheel_joint_front_right', 'wheel_joint_rear_left', 'wheel_joint_rear_right']
# ['extension_joint_front_left_0', 'extension_joint_front_left_1', 'extension_joint_front_left_2', 'extension_joint_front_right_0', 'extension_joint_front_right_1', 'extension_joint_front_right_2', 'extension_joint_rear_left_0', 'extension_joint_rear_left_1', 'extension_joint_rear_left_2', 'extension_joint_rear_right_0', 'extension_joint_rear_right_1', 'extension_joint_rear_right_2']
# ['wheel_joint_front_left', 'wheel_joint_front_right', 'wheel_joint_rear_left', 'wheel_joint_rear_right']
# Box(0.0, 3.1415927, (12,), float32)
# [0.47070292 2.6439304  2.453756   2.9150815  1.4896942  2.898555
#  2.8408165  2.9836972  2.309478   1.8424743  0.9898338  2.7874093 ]
# [0.2609273  3.02554    2.1249518  2.459501   1.2093369  2.8597262
#  1.0109758  1.6923621  0.76296705 1.824136   0.5914434  0.7782967 ]