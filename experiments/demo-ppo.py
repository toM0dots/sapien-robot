import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from twsim.envs import explore  # noqa: F401
from twsim.robots import transwheel  # noqa: F401


@dataclass
class Args:
    """Experiment configuration and arguments."""

    # fmt: off

    # General configuration
    exp_name: str | None = None            # Experiment name
    seed: int = 1                          # Experiment seed
    torch_deterministic: bool = True       # set `torch.backends.cudnn.deterministic=True`
    cuda: bool = True                      # Enable CUDA by default
    track: bool = False                    # Track with Wandb
    wandb_project_name: str = "ManiSkill"  # Wandb project name
    wandb_entity: str | None = None        # Wandb entity name
    capture_video: bool = True             # Save videos to ./runs/{run_name}/videos
    save_model: bool = True                # Save model ./runs/{run_name}
    evaluate: bool = False                 # Only evaluate and save trajectories
    checkpoint: str | None = None          # Path to pretrained checkpoint for initialization

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"                    # Environment ID
    total_timesteps: int = 10_000_000              # Total number of time steps for training
    learning_rate: float = 3e-4                    # Optimizer learning rate
    num_envs: int = 512                            # Number of parallel environments
    num_eval_envs: int = 8                         # Number of parallel evaluation environments
    partial_reset: bool = True                     # Let parallel environments reset upon termination instead of truncation
    eval_partial_reset: bool = False               # Let parallel evaluation environments reset upon termination instead of truncation
    num_steps: int = 50                            # Number of steps to run in each environment per policy rollout
    num_eval_steps: int = 50                       # Number of steps to run in each evaluation environment
    reconfiguration_freq: int|None = None          # How often to reconfigure the environment during training
    eval_reconfiguration_freq: int|None =  1       # Reconfigure the environment each reset to ensure objects are randomized
    control_mode: str|None = "pd_joint_delta_pos"  # Control mode
    anneal_lr: bool =  False                       # Toggle learning rate annealing for policy and value networks
    gamma: float = 0.8                             # Discount factor gamma
    gae_lambda: float = 0.9                        # Lambda for the general advantage estimation
    num_minibatches: int = 32                      # Number of mini-batches
    update_epochs: int = 4                         # K epochs to update the policy
    norm_adv: bool = True                          # Toggle advantages normalization
    clip_coef: float = 0.2                         # Surrogate clipping coefficient
    clip_vloss: bool = False                       # Toggle clipped loss for the value function
    ent_coef: float = 0.0                          # Coefficient of the entropy
    vf_coef: float = 0.5                           # Coefficient of the value function
    max_grad_norm: float = 0.5                     # Maximum norm for the gradient clipping
    target_kl: float = 0.1                         # Target KL divergence threshold
    reward_scale: float = 1.0                      # Scale the reward by this factor
    eval_freq: int = 25                            # Evaluation frequency in terms of iterations
    save_train_video_freq: int|None =  None        # Frequency to save training videos in terms of iterations )
    finite_horizon_gae: bool = False               # Horizon for generalized advantage estimation

    # Derived configurations
    batch_size: int = 0     # Batch size (derived)
    minibatch_size: int = 0 # Mini-batch size (derived)
    num_iterations: int = 0 # Number of iterations (derived)
    # fmt: on


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        observation_shape = envs.single_observation_space.shape
        action_shape = envs.single_action_space.shape

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(action_shape)), std=0.01 * np.sqrt(2)),
        )

        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(action_shape)) * -0.5)

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter | None = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            raise NotImplementedError("Wandb tracking is not implemented yet.")
            wandb.log({tag: scalar_value}, step=step)
        if self.writer:
            self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        if self.writer:
            self.writer.close()


if __name__ == "__main__":
    "Run PPO on a ManiSkill environment."

    #
    #  Parse arguments
    #

    args = tyro.cli(Args)

    # Compute derived arguments
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    #
    # Configure and create the environment
    #

    # NOTE: do not modify {
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    # NOTE: do not modify }

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")

    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    # Create the training environment
    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,  # type: ignore
    )

    # Create the evaluation environment
    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **env_kwargs,  # type: ignore
    )

    # Flatten action spaces if needed
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    #
    # Setup recording wrappers
    #

    if args.capture_video:
        if args.save_train_video_freq:

            def save_video_trigger(x):
                return (x // args.num_steps) % args.save_train_video_freq == 0

            envs = RecordEpisode(
                envs,  # type: ignore
                output_dir=f"runs/{run_name}/train_videos",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=30,
            )

        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            if not args.checkpoint:
                raise ValueError("Checkpoint must be provided for evaluation.")
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"

        print(f"Saving eval videos to {eval_output_dir}")

        eval_envs = RecordEpisode(
            eval_envs,  # type: ignore
            output_dir=eval_output_dir,
            save_trajectory=args.evaluate,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=30,
        )

    #
    # Setup vectorized environments wrappers
    #

    envs = ManiSkillVectorEnv(
        envs,  # type: ignore
        # args.num_envs,
        ignore_terminations=not args.partial_reset,
        record_metrics=True,
    )

    eval_envs = ManiSkillVectorEnv(
        eval_envs,  # type: ignore
        # args.num_eval_envs,
        ignore_terminations=not args.eval_partial_reset,
        record_metrics=True,
    )

    action_space_is_continuous = isinstance(envs.single_action_space, gym.spaces.Box)
    assert action_space_is_continuous, "Only continuous action spaces are supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None

    if not args.evaluate:
        print("Running training")

        if args.track:
            raise NotImplementedError("Wandb tracking is not implemented yet.")

            import wandb

            config = vars(args)
            config["env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_envs,
                env_id=args.env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps,
                partial_reset=args.partial_reset,
            )
            config["eval_env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_eval_envs,
                env_id=args.env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps,
                partial_reset=False,
            )
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group="PPO",
                tags=["ppo", "walltime_efficient"],
            )

        writer = SummaryWriter(f"runs/{run_name}")

        table_header = "|param|value|\n|-|-|"
        table_body = "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        writer.add_text("hyperparameters", f"{table_header}\n{table_body}")

        logger = Logger(log_wandb=args.track, tensorboard=writer)

    else:
        print("Running evaluation only")

    #
    # Algorithm storage space
    #

    experiment_shape = (args.num_steps, args.num_envs)

    obs = torch.zeros(experiment_shape + envs.single_observation_space.shape).to(device)  # type: ignore
    actions = torch.zeros(experiment_shape + envs.single_action_space.shape).to(device)  # type: ignore
    logprobs = torch.zeros(experiment_shape).to(device)
    rewards = torch.zeros(experiment_shape).to(device)
    dones = torch.zeros(experiment_shape).to(device)
    values = torch.zeros(experiment_shape).to(device)

    #
    # Setup the agent and optimizer
    #

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    #
    # Start training
    #

    # NOTE: do not modify
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    print("####")
    print(f"{args.num_iterations=} {args.num_envs=} {args.num_eval_envs=}")
    print(f"{args.minibatch_size=} {args.batch_size=} {args.update_epochs=}")
    print("####")

    action_space_low, action_space_high = (
        torch.from_numpy(envs.single_action_space.low).to(device),  # type: ignore
        torch.from_numpy(envs.single_action_space.high).to(device),  # type: ignore
    )

    def compute_clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, {global_step=}")

        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()

        if iteration % args.eval_freq == 1:
            print("Evaluating")

            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0

            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_action = agent.get_action(eval_obs, deterministic=True)
                    eval_obs, _, _, _, eval_infos = eval_envs.step(eval_action)

                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)

            total_eval_steps = args.num_eval_steps * args.num_eval_envs
            print(f"Evaluated {total_eval_steps} steps resulting in {num_episodes} episodes")

            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")

            if args.evaluate:
                break

        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lr_now = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now

        rollout_time = time.time()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs  # type: ignore
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            clip_action = compute_clip_action(action)
            next_obs, reward, terminations, truncations, infos = envs.step(clip_action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)  # type: ignore
            rewards[step] = reward.view(-1) * args.reward_scale  # type: ignore

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]

                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)  # type: ignore

                with torch.no_grad():
                    final_obs = agent.get_value(infos["final_observation"][done_mask])
                    done_mask_indices = torch.arange(args.num_envs, device=device)[done_mask]
                    final_values[step, done_mask_indices] = final_obs.view(-1)

        rollout_time = time.time() - rollout_time

        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            last_gae_lam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    next_values = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    next_values = values[t + 1]

                # t instead of t+1
                real_next_values = next_not_done * next_values + final_values[t]

                # next_not_done means next_values is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1:  # initialize
                        lam_coef_sum = 0.0
                        reward_term_sum = 0.0  # the sum of the second term
                        value_term_sum = 0.0  # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = (
                        args.gae_lambda * args.gamma * reward_term_sum
                        + lam_coef_sum * rewards[t]
                    )
                    value_term_sum = (
                        args.gae_lambda * args.gamma * value_term_sum
                        + args.gamma * real_next_values
                    )

                    advantages[t] = (
                        reward_term_sum + value_term_sum
                    ) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = last_gae_lam = (
                        delta + args.gamma * args.gae_lambda * next_not_done * last_gae_lam
                    )  # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.time()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_time

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar(
            "time/rollout_fps",
            args.num_envs * args.num_steps / rollout_time,
            global_step,
        )
    if not args.evaluate:
        if args.save_model:
            model_path = f"runs/{run_name}/final_ckpt.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        logger.close()
    envs.close()
    eval_envs.close()
