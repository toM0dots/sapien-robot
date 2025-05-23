# Evaluation Notes

TODO: I need to make the step and sensor optional? Just one configurable environment instead of multiple

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
