from dataclasses import dataclass

import tyro


@dataclass
class Args:
    "Evaluate a trained policy."

    # fmt: off

    checkpoint: str                          # Path to the checkpoint file
    env_id: str = "PlaneVel-v1"              # Environment ID
    control_mode: str = "wheel_vel_ext_pos"  # Control mode
    capture_video: bool = True               # Save videos to ./runs/{run_name}/test_videos
    num_eval_envs: int = 8                   # Number of parallel evaluation environments
    num_eval_steps: int = 500                # Number of steps to run in each evaluation environment
    eval_reconfiguration_freq: int = 1       # Reconfigure the environment each reset to ensure objects are randomized
    eval_partial_reset: bool = False         # Let parallel evaluation environments reset upon termination instead of truncation
    cuda: bool = True                        # Use GPU for evaluation

    # fmt: on


if __name__ == "__main__":
    #
    #  Parse arguments
    #

    args = tyro.cli(Args)

    from collections import defaultdict
    from pathlib import Path

    import gymnasium as gym
    import torch
    from agent import Agent
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    from twsim.envs import plane  # noqa: F401
    from twsim.robots import transwheel  # noqa: F401

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        control_mode=args.control_mode,
    )

    # Create the evaluation environment
    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **env_kwargs,  # type: ignore
    )

    # Flatten action spaces if needed
    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    eval_output_dir = str(Path(args.checkpoint).parent / "test_videos")

    print(f"Saving eval videos to {eval_output_dir}")

    eval_envs = RecordEpisode(
        eval_envs,  # type: ignore
        output_dir=eval_output_dir,
        save_trajectory=True,
        trajectory_name="trajectory",
        max_steps_per_video=args.num_eval_steps,
        video_fps=30,
    )

    eval_envs = ManiSkillVectorEnv(
        eval_envs,  # type: ignore
        args.num_eval_envs,
        ignore_terminations=not args.eval_partial_reset,
        record_metrics=True,
    )

    print("Evaluating")

    eval_obs, _ = eval_envs.reset()
    num_episodes = 0

    observation_shape = eval_envs.single_observation_space.shape
    action_shape = eval_envs.single_action_space.shape

    agent = Agent(observation_shape, action_shape).to(device)
    agent.load_state_dict(torch.load(args.checkpoint))
    agent.eval()

    for _ in range(args.num_eval_steps):
        with torch.no_grad():
            eval_action = agent.get_action(eval_obs, deterministic=True)
            eval_obs, _, _, _, eval_infos = eval_envs.step(eval_action)

            if "final_info" in eval_infos:
                mask = eval_infos["_final_info"]
                num_episodes += mask.sum()

    total_eval_steps = args.num_eval_steps * args.num_eval_envs
    print(f"Evaluated {total_eval_steps} steps resulting in {num_episodes} episodes")

    eval_envs.close()
