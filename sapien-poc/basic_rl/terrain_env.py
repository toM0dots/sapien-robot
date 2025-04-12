import sapien
from sapien import Pose
import numpy as np
import torch
from typing import Any, Dict, Union

from transforms3d.euler import euler2quat

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env

@register_env("Terrain-env", max_episode_steps=50)
class TerrainEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["tw_robot"]

    INITIAL_ROBOT_POSE_TUPLE = [0, 0, 0.1]
    last_pose = np.array(INITIAL_ROBOT_POSE_TUPLE)

    TARGET_POSE = [3e-1, 2e-1, 0]

    previous_action = torch.zeros(8)
    max_wheel_delta = 0.5

    def __init__(self, *args, robot_uids="tw_robot", **kwargs):
        # robot_uids="fetch" is possible, or even multi-robot 
        # setups via robot_uids=("fetch", "panda")
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
        print("Init episode...")
        self.agent.robot.set_pose(sapien.Pose(p=self.INITIAL_ROBOT_POSE_TUPLE))


    def evaluate(self):

        success_threshold = 0.03
        fall_threshold = -0.1

        robot_pose = self.agent.robot.get_pose().raw_pose.numpy()[0][:3]

        # print(f"Pose: ({pose[0].item()},{pose[1].item()},{pose[2].item()})")

        success = abs(np.linalg.norm(robot_pose-self.TARGET_POSE)) < success_threshold 
        fail = robot_pose[2].item() < fall_threshold

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
        # print(self._get_obs_agent())
        return dict(
            agent=self._get_obs_agent(),
            extra=self._get_obs_extra(info),
        )

    def _get_obs_extra(self, info: Dict):
        return dict()
    

    #
    # Compute rewards
    #
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        
        robot_pose = self.agent.robot.get_pose().raw_pose.numpy()[0][:3]

        success_threshold = 0.03
        fall_threshold = -0.01

        change_margin = 0.00005

        success = abs(np.linalg.norm(robot_pose-self.TARGET_POSE)) < success_threshold 
        fallen = robot_pose[2] < fall_threshold
        minimal_change = abs(np.linalg.norm(self.last_pose-robot_pose)) < change_margin
        improvement = np.linalg.norm(robot_pose-self.TARGET_POSE) < np.linalg.norm(self.last_pose-self.TARGET_POSE)
        
        reward = 10
        if minimal_change:
            pass
        elif improvement:
            reward = 300
        elif (not improvement):
            reward = -300
        elif success:
            reward = 1000
        elif fallen:
            reward = -1000
            
        self.last_pose = robot_pose
        
        return torch.tensor([reward])
        # return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1000.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward