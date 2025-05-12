import sapien
from sapien import Pose
import numpy as np
import torch
from typing import Any, Dict, Union
import math

from transforms3d.euler import euler2quat

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.envs.utils import randomization, rewards
@register_env("Terrain-env", max_episode_steps=50)
class TerrainEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["tw_robot"]

    INITIAL_ROBOT_POSE_TUPLE = [0, 0, 0.1]
    last_pose = np.array(INITIAL_ROBOT_POSE_TUPLE)
    last_velocity = np.array([0.0,0.0,0.0])
    last_net_extensions = 0

    TARGET_DISTANCE = 8e-1
    TARGET_VELOCITY = 3e-1 # in/s?

    FALL_THRESHOLD = -0.1
    CHANGE_MARGIN = 1e-5
    

    previous_action = torch.zeros(8)
    max_wheel_delta = 0.5

    def __init__(self, *args, robot_uids="tw_robot", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)
    
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=self.INITIAL_ROBOT_POSE_TUPLE))

    def _load_scene(self, options: dict):
        
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose(p=[0,0,0], q=[1, 0, 0, 0])

        half_size = [10, 1.5e-1, 1e-2]
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose(p=[3e-1,0,0], q=[1, 0, 0, 0])
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, material=[.2, .2, .2])
        box = builder.build_static(name="floor")

        half_size = [3e-2, 3e-1, 2e-2]
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose(p=[1.5e-1,0,1e-2], q=[1, 0, 0, 0])
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, material=[.4, .2, .4])
        box = builder.build_static(name="wall")

        half_size = [3e-2, 3e-1, 2e-2]
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose(p=[4e-1,0,1e-2], q=[1, 0, 0, 0])
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, material=[.4, .2, .4])
        box = builder.build_static(name="wall2")


        # half_size = [1e-2, 1e-2, 1e-2]
        # builder = self.scene.create_actor_builder()
        # builder.initial_pose = sapien.Pose(p=self.TARGET_POSE, q=[1, 0, 0, 0])
        # builder.add_box_collision(half_size=half_size)
        # builder.add_box_visual(half_size=half_size, material=[.9, .1, .1])
        # box = builder.build_static(name="goal")

        return
        terrain_name = "Terrain"
        terrain_vertical_offset = 0.0
        terrain_length = 1
        terrain_material = [0.9, 0.9, 0.9]
        terrain_chunks = 1

        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose(p=[0,0,0], q=[1, 0, 0, 0])
        
        builder.add_convex_collision_from_file(
            filename=f"terrain/{terrain_name}.obj",
        )
        builder.add_visual_from_file(
            filename=f"terrain/{terrain_name}.glb",
            material=terrain_material,
        )

        for i in range(terrain_chunks):
            terrain = builder.build_static(name=f"terrain_{i}") #Kinematic entities are not affected by forces
            terrain.set_pose(
                Pose(
                    p=[i*terrain_length, 0, terrain_vertical_offset],
                    q=euler2quat(np.deg2rad(90), 0, np.deg2rad(90)) #Terrain is by default on its side, rotate to correct orientation
                )
            )


        # TEMPORARY FOR DEBUG
        half_size = [2e-2, 2e-2, 2e-2]
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose(p=self.TARGET_POSE, q=[1, 0, 0, 0])
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, material=[.2, .2, .2])
        box = builder.build_static(name="block")





    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.agent.robot.set_pose(sapien.Pose(p=self.INITIAL_ROBOT_POSE_TUPLE))


    def evaluate(self):

        fall_threshold = -0.1

        robot_pose = self.agent.robot.get_pose().raw_pose.numpy()[0][:3]

        # print(f"Pose: ({pose[0].item()},{pose[1].item()},{pose[2].item()})")

        success = robot_pose[0].item() > self.TARGET_DISTANCE 
        fail = robot_pose[2].item() < self.FALL_THRESHOLD

        return {
            "success": torch.from_numpy(np.array([success])), # If robot has moved far enough forward, success. #torch.zeros(self.num_envs, device=self.device, dtype=bool),
            "fail": torch.from_numpy(np.array([fail])) # If robot has fallen off, failed.    #torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

    #
    # Override _step_action to have all extensions on a given wheel perform the same action
    #
    def _step_action(
        self, action: Union[None, np.ndarray, torch.Tensor, Dict]
    ) -> Union[None, torch.Tensor]:
        set_action = False
        action_is_unbatched = False
        if action is None:  # simulation without action
            pass
        elif isinstance(action, np.ndarray) or isinstance(action, torch.Tensor):
            action = common.to_tensor(action, device=self.device)
            if action.shape == self._orig_single_action_space.shape:
                action_is_unbatched = True
            set_action = True
        elif isinstance(action, dict):
            if "control_mode" in action:
                if action["control_mode"] != self.agent.control_mode:
                    self.agent.set_control_mode(action["control_mode"])
                    self.agent.controller.reset()
                action = common.to_tensor(action["action"], device=self.device)
                if action.shape == self._orig_single_action_space.shape:
                    action_is_unbatched = True
            else:
                assert isinstance(
                    self.agent, MultiAgent
                ), "Received a dictionary for an action but there are not multiple robots in the environment"
                # assume this is a multi-agent action
                action = common.to_tensor(action, device=self.device)
                for k, a in action.items():
                    if a.shape == self._orig_single_action_space[k].shape:
                        action_is_unbatched = True
                        break
            set_action = True
        else:
            print(type(action))
            pass
            # raise TypeError(type(action))

        if set_action:

            # CHANGES
            # - Changed the action space to only contain 8 actions (1 per wheel and 1 per wheel's extension set)
            # - There are 16 controllers but only sampling from the 4 wheels, and 4 arbitrary extensions
            # - Duplicate for the wheel extensions so that they are in sync
            wheel_actions = action[:4]
            extension_actions = action[-4:]

            repeated_extension_actions = torch.cat([
                extension_actions[i].repeat(
                    self.agent.num_wheel_extensions) for i in range(len(extension_actions))
                ])
            new_actions = torch.cat((wheel_actions, repeated_extension_actions))
            self.previous_action = new_actions

            if self.num_envs == 1 and action_is_unbatched:
                new_actions = common.batch(new_actions)
            
            self.agent.set_action(new_actions)
            # END CHANGES

            if self._sim_device.is_cuda():
                self.scene.px.gpu_apply_articulation_target_position()
                self.scene.px.gpu_apply_articulation_target_velocity()
        self._before_control_step()
        for _ in range(self._sim_steps_per_control):
            if self.agent is not None:
                self.agent.before_simulation_step()
            self._before_simulation_step()
            self.scene.step()
            self._after_simulation_step()
        self._after_control_step()
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        return action
    
    
    #
    # Get observations
    #
    def _get_obs_state_dict(self, info: Dict):
        """Get (ground-truth) state-based observations."""
        return dict(
            agent=self._get_obs_agent(),
            extra=self._get_obs_extra(info),
        )

    def _get_obs_extra(self, info: Dict):
        pose = self.agent.robot.get_qpos().numpy()[0]
        angular_velocity = self.agent.robot.root_angular_velocity.numpy()[0]
        linear_acceleration = (
            self.last_velocity - self.agent.robot.root_linear_velocity.numpy()[0]
        ) * self._control_freq

        return dict(
            qpos=torch.tensor(pose).unsqueeze(0),             
            angular_vel=torch.tensor(angular_velocity).unsqueeze(0),  
            linear_accel=torch.tensor(linear_acceleration).unsqueeze(0), 
        )
    

    #
    # Compute rewards
    #
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        
        robot_pose = self.agent.robot.get_pose().raw_pose.numpy()[0][:3]
        linear_vel = self.agent.robot.root_linear_velocity.numpy()[0]
        angualr_vel = self.agent.robot.root_angular_velocity.numpy()[0]
        obs_arr = obs.numpy()[0]
        net_extension_angles = obs_arr[4] + obs_arr[7] + obs_arr[10] + obs_arr[13]
        net_extension_angles = (net_extension_angles + self.last_net_extensions)/2

        

        minimal_change = abs(self.last_pose[0].item()-robot_pose[0].item()) < self.CHANGE_MARGIN
        success = robot_pose[0].item() > self.TARGET_DISTANCE
        fallen = robot_pose[2].item() < self.FALL_THRESHOLD


        velocity_x_diff = linear_vel[0] - self.TARGET_VELOCITY

        # Rewards
        rew_survival = 0.01


        # scale_rew_vel = 5300 * self.TARGET_VELOCITY
        max_vel_rew = 1
        # rew_velocity =  max_vel_rew - (1/self.TARGET_VELOCITY) * abs(velocity_x_diff) 
        distribution_exp = (-1/2)*(velocity_x_diff/(self.TARGET_VELOCITY/2))**2
        rew_velocity = max_vel_rew* (math.e**distribution_exp)
        # print(f"RewVel: {rew_velocity:.3f}, VX: {linear_vel[0]:.3f}, VelDiff: {velocity_x_diff:.3f}")
        
        # Penalties
        # scale_pen_ext = 0.001
        # pen_extensions = scale_pen_ext * net_extension_angles**2
        pen_extensions = math.log10(net_extension_angles + 1)
        

        # print(f"VelRew: {rew_velocity}, ExtPen: {pen_extensions}")
        reward = rew_survival + rew_velocity - pen_extensions
        # reward = 0
        if minimal_change:
            reward = rew_survival
        elif success:
            reward = 1
        elif fallen:
            reward = -1


        self.last_pose = robot_pose
        self.last_velocity = linear_vel
        self.last_net_extensions = net_extension_angles


        # print(f"")

        print(f"Reward: {reward:.3f}, RewVel: {rew_velocity:.3f}, PenExt: {pen_extensions:.3f} VX: {linear_vel[0]:.3f}, VelDiff: {velocity_x_diff:.3f}")
        

        return torch.tensor([reward]) 

        
        
        
        
        print(f"X lin vel: {linear_vel[0]}, X diff: {velocity_x_diff}, ")

        ext_penalty_scale = 10
        extension_penalty = 0 # ext_penalty_scale * -net_extension_angles 
        
        vel_max_reward = 600
        vel_reward_ease = 200
        # target_speed_reward = min(vel_max_reward, vel_reward_ease/(max(abs(velocity_x_diff), 0.00001)))

        target_speed_reward = 50 * linear_vel[0]

        survival_reward = 5 
        reward = survival_reward + target_speed_reward + extension_penalty
        
        if minimal_change:
            reward = 0
        elif success:
            reward = 1000
        elif fallen:
            reward = -1000
            
        print(f"Pose: {robot_pose}, Lin vel: {linear_vel}, Success: {success}")
        print(f"Rew: {reward}, Vel Rew: {target_speed_reward} Ext Pen: {extension_penalty}")


        self.last_pose = robot_pose
        self.last_velocity = linear_vel
        # print("REW",reward)

        return torch.tensor([reward])
        # return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward