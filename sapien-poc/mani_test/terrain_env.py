from typing import Any, Dict, Union
import sapien
from sapien import Pose
from transforms3d.euler import euler2quat
import numpy as np
import torch
from mani_skill.sensors.camera import CameraConfig

from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env

@register_env("Terrain-env", max_episode_steps=50)
class TerrainEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["tw_robot"]

    def __init__(self, *args, robot_uids="tw_robot", **kwargs):
        # robot_uids="fetch" is possible, or even multi-robot 
        # setups via robot_uids=("fetch", "panda")
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0.3]))

    def _load_scene(self, options: dict):
        
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        terrain_name = "Terrain"
        terrain_vertical_offset = 0.05
        terrain_length = 1
        terrain_material = [0.9, 0.9, 0.9]
        terrain_chunks = 1


        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.02], q=[1, 0, 0, 0])
        
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
 
        # strongly recommended to set initial poses for objects, even if you plan to modify them later
        # self.obj = builder.build(name="terrain")

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return torch.tensor([0.47, 13.1, 1.1, 0.2])
        # Not implemented yet but can't be NotImplementedError()
    
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)