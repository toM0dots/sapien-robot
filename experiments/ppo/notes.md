# Notes for PPO experiments

## Training Notes

```bash
# PlaneVel
python train-ppo.py --exp-name PlaneEnv --env-id PlaneVel-v1 --control-mode wheel_vel_ext_pos --num-steps 80
python train-ppo.py --env-id PlaneVel-v1 --control-mode wheel_vel_ext_pos --evaluate --checkpoint runs/PlaneEnv/final_ckpt.pt --num_eval_envs 1 --num_eval_steps 80
python train-ppo.py --env-id StepVel-v1 --control-mode wheel_vel_ext_pos --evaluate --checkpoint runs/PlaneEnv/final_ckpt.pt --num_eval_envs 1 --num_eval_steps 80

# StepVel
python train-ppo.py --exp-name StepEnv --env-id StepVel-v1 --control-mode wheel_vel_ext_pos --num-steps 80
python train-ppo.py --env-id PlaneVel-v1 --control-mode wheel_vel_ext_pos --evaluate --checkpoint runs/StepEnv/final_ckpt.pt --num_eval_envs 1 --num_eval_steps 80
python train-ppo.py --env-id StepVel-v1 --control-mode wheel_vel_ext_pos --evaluate --checkpoint runs/StepEnv/final_ckpt.pt --num_eval_envs 1 --num_eval_steps 80

# TODO: StepVelSensor
python train-ppo.py --exp-name SensorEnv --env-id StepVelSensor-v1 --control-mode wheel_vel_ext_pos --num-steps 80
python train-ppo.py --env-id PlaneVelSensor-v1 --control-mode wheel_vel_ext_pos --evaluate --checkpoint runs/SensorEnv/final_ckpt.pt --num_eval_envs 1 --num_eval_steps 80
python train-ppo.py --env-id StepVelSensor-v1 --control-mode wheel_vel_ext_pos --evaluate --checkpoint runs/SensorEnv/final_ckpt.pt --num_eval_envs 1 --num_eval_steps 80
```



## Evaluation Notes

TODO: I need to make the step and sensor optional? Just one configurable environment instead of multiple
TODO: make video aspect ratio configurable

```bash
# Evaluate the model trained in PlaneVel in PlaneVel
python evaluate.py --env-id PlaneVel-v1 --checkpoint runs/Stiffer/final_ckpt.pt
mv runs/Stiffer/test_videos runs/Stiffer/planevel
# No problems

# Evaluate the model trained in PlaneVel in StepVel
python evaluate.py --env-id StepVel-v1 --checkpoint runs/Stiffer/final_ckpt.pt
mv runs/Stiffer/test_videos runs/Stiffer/stepvel
# Gets stuck on the first step

# Evaluate the model trained in StepVel in PlaneVel
python evaluate.py --env-id PlaneVel-v1 --checkpoint runs/Step/final_ckpt.pt
mv runs/Step/test_videos runs/Step/planevel
# Unnecessary extension

# Evaluate the model trained in StepVel in StepVel
python evaluate.py --env-id StepVel-v1 --checkpoint runs/Step/final_ckpt.pt
mv runs/Step/test_videos runs/Step/stepvel
# Unnecessary extension

# Evaluate the model trained in StepVelSensor in StepVelSensor (with and without step?)
```
