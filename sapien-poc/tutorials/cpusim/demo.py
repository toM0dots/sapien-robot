# Import required packages
import gymnasium as gym
import mani_skill.envs
import time
import matplotlib.pyplot as plt
env = gym.make("PegInsertionSide-v1",
               render_mode="rgb_array"
               )
obs, _ = env.reset(seed=0)

# Save the first frame as output.png
plt.imshow(env.render()[0].numpy()) # we take [0].numpy() as everything is a batched tensor
plt.savefig("output.png")
plt.close()

env.unwrapped.print_sim_details() # print verbose details about the configuration
done = False
start_time = time.time()
while not done:
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
N = info["elapsed_steps"].item()
dt = time.time() - start_time
FPS = N / (dt)
print(f"Frames Per Second = {N} / {dt} = {FPS}")