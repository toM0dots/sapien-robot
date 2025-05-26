#!/usr/bin/env bash

mamba activate gymtest
python demo-ppo.py --env-id PlaneVel-v1 --control-mode wheel_vel_ext_pos
# --total-timesteps 1000 --num-envs 32
python demo-ppo.py --env-id Plane --control-mode wheel_vel_ext_pos
# tensorboard --logdir runs --port 8438
python demo-ppo.py --env-id Explore-v1 --control-mode wheel_vel_ext_pos


