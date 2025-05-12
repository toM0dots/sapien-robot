import sapien

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from gymnasium import spaces

import numpy as np
import torch

from sapien import Pose, Scene
from transforms3d.euler import euler2quat
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from mani_skill.envs.scene import ManiSkillScene

from mani_skill.agents.controllers.pd_joint_pos import (
    PDJointPosController,
    PDJointPosControllerConfig,
)
from mani_skill.agents.controllers.base_controller import (
    ControllerConfig,
)
DictControllerConfig = Dict[str, ControllerConfig]
from mani_skill.utils.structs import Link

# Robot Parameters

# control_freq = 100.0
# timestep = 1 / control_freq

static_friction = 10.0
dynamic_friction = 10.0
restitution = 0.1
joint_friction = 0.0
joint_damping = 0.0

# The coordinate frame in Sapien is: x(forward), y(left), z(upward)

chassis_length = 9e-2 #16e-2
chassis_width = 6e-2# 32e-2
chassis_thickness = 1e-2
chassis_material = [0.4, 0.4, 0.0]

wheel_radius = 1.5e-2 #5e-2
wheel_thickness = 0.2e-2 #1e-2
wheel_material = [0.0, 0.4, 0.4]

wheel_extension_radial_offset = 0.875e-2 #wheel_radius / 1.2
wheel_extension_angle_offset = np.deg2rad(20)
wheel_extension_length = 1.625e-2 #wheel_radius / 3
wheel_extension_width = 0.375e-2 #wheel_thickness
wheel_extension_thickness = 0.2e-2 #wheel_thickness
wheel_extension_material = [0.4, 0.0, 0.4]

@register_agent()
class TwRobot(BaseAgent):
    uid = "tw_robot"

    num_wheel_extensions = 3 # TODO: Currently assumes 3 wheel extensions when copied

    def create_twrobot(self, initial_pos) -> sapien.physx.PhysxArticulation:
        
        # 
        # Chassis
        #
        robot_builder = self.scene.create_articulation_builder()
        robot_builder.initial_pose = initial_pos
        
        chassis_half_size = [chassis_length / 2, chassis_width / 2, chassis_thickness / 2]
        chassis_vertical_offset = wheel_radius + 7e-2
        chassis_pose = Pose(p=[0, 0, chassis_vertical_offset])
        
        chassis = robot_builder.create_link_builder()
        chassis.set_name("chassis")
        chassis.add_box_collision(half_size=chassis_half_size)
        chassis.add_box_visual(half_size=chassis_half_size, material=chassis_material)
        
        # 
        # Wheels and revolute joints
        # 
        wheel_half_thickness = wheel_thickness / 2
        
        front_rear_placement = chassis_half_size[0]
        left_right_placement = chassis_half_size[1] + 1e-2
        ninety_deg = np.deg2rad(90)
        
        wheel_parameters = [
            ("front_left",  front_rear_placement,  left_right_placement, euler2quat(0, 0, ninety_deg)),
            ("front_right", front_rear_placement, -left_right_placement, euler2quat(0, 0, ninety_deg)),
            ("rear_left",  -front_rear_placement,  left_right_placement, euler2quat(0, 0, ninety_deg)),
            ("rear_right", -front_rear_placement, -left_right_placement, euler2quat(0, 0, ninety_deg)),
        ]
        
        wheels = {}
        
        for name, fr, lr, quat in wheel_parameters:
            
            wheel = robot_builder.create_link_builder(chassis)
            wheel.set_name(f"wheel_{name}")
            wheel.set_joint_name(f"wheel_joint_{name}")
        
            # TODO: convert to spheroid using convex mesh?
            # NOTE: by default, cylinders are oriented along the x-axis
            wheel.add_cylinder_collision(radius=wheel_radius, half_length=wheel_half_thickness)
            wheel.add_cylinder_visual(radius=wheel_radius, half_length=wheel_half_thickness, material=wheel_material)
        
            wheel.set_joint_properties(
                "revolute",
                limits=[[-np.inf, np.inf]],
                pose_in_parent=Pose(p=[fr, lr, 0], q=quat),
                pose_in_child=Pose(),
                friction=joint_friction,
                damping=joint_damping,
            )
        
            wheels[name] = wheel
        
        # 
        # Wheel extensions
        # 
        # NOTE: extensions are relative to the wheel:
        #    x -> y
        #    y -> x
        #    z -> z
        
        extension_half_size = [wheel_extension_width / 2, wheel_extension_length / 2, wheel_extension_thickness / 2]
        
        for name, fr, lr, quat in wheel_parameters:

            for i in range(self.num_wheel_extensions):
            
                extension = robot_builder.create_link_builder(wheels[name])
                extension.set_name(f"extension_{name}_{i}")
                extension.set_joint_name(f"extension_joint_{name}_{i}")
        
                # TODO: convert to capsule?
                extension.add_box_collision(half_size=extension_half_size)
                extension.add_box_visual(half_size=extension_half_size, material=wheel_extension_material)
        
                radial_angle = np.deg2rad(i/self.num_wheel_extensions*360)
                
                y = wheel_extension_radial_offset * np.cos(radial_angle)
                z = wheel_extension_radial_offset * np.sin(radial_angle)
        
                x = np.copysign(wheel_extension_width, lr)
        
                extension.set_joint_properties(
                    "revolute",
                    # TODO: Upper limit should take into account angle offset
                    limits=[[0, np.pi]],
                    pose_in_parent=Pose(p=[x, y, z]),
                    pose_in_child=Pose(p=[0, -wheel_extension_length/2, 0], q=euler2quat(np.deg2rad(90) - radial_angle + wheel_extension_angle_offset, 0, 0)),
                    friction=joint_friction,
                    damping=joint_damping,
                )

        # 
        # Finalize the articulated robot
        # 
        robot_builder.set_name("tw_robot")
        robot = robot_builder.build()
        
        robot.set_pose(chassis_pose)
        
        joints = {joint.get_name(): joint for joint in robot.get_active_joints()}
        
        joint_mode = 'force'
        
        for jname in joints:
            
            if jname.startswith("wheel_joint"):
                joints[jname].set_drive_properties(stiffness=10, damping=100, mode=joint_mode)
            
            elif jname.startswith("extension_joint"):
                joints[jname].set_drive_properties(stiffness=1000, damping=10, mode=joint_mode)
            
            else:
                print("Ignoring", jname) 

        return robot

    def _load_articulation(
        self, initial_pose: Optional[Union[sapien.Pose, Pose]] = None
    ):        
        self.robot = self.create_twrobot(initial_pose)

        # Cache robot link names
        self.robot_link_names = [link.name for link in self.robot.get_links()]

    @property
    def action_space(self) -> spaces.Space:
        # Our control mode is pd_joint_delta_pos, never None

        if self._control_mode is None:
            return spaces.Dict(
                {
                    uid: controller.action_space
                    for uid, controller in self.controllers.items()
                }
            )
        else:
            # When obtaining the action space, only keep the bottom 8 actions:
            # 4 unique wheel actions, 4 unique extensions (are copied in _step_action in env)
            original_space = self.controller.action_space
            return spaces.Box( 
                low=original_space.low[:8],
                high=original_space.high[:8],
                dtype=original_space.dtype
            )
        
    @property
    def single_action_space(self) -> spaces.Space:
        # Our control mode is pd_joint_delta_pos, never None

        if self._control_mode is None:
            return spaces.Dict(
                {
                    uid: controller.single_action_space
                    for uid, controller in self.controllers.items()
                }
            )
        else:
            # When obtaining the action space, only keep the bottom 8 actions:
            # 4 unique wheel actions, 4 unique extensions (are copied in _step_action in env)
            original_space = self.controller.single_action_space
            return spaces.Box(
                low=original_space.low[:8],
                high=original_space.high[:8],
                dtype=original_space.dtype
            )

    @property
    def _controller_configs(
        self,
    ) -> Dict[str, Union[ControllerConfig, DictControllerConfig]]:
        """Returns a dict of controller configs for this agent. By default this is a PDJointPos (delta and non delta) controller for all active joints."""

        wheel_pd_joint_delta_pos = PDJointPosControllerConfig(
            [x.name for x in self.robot.active_joints if "wheel_joint" in x.name],
            lower=-2000,
            upper=2000,
            stiffness=100,
            damping=10,
            friction=joint_friction,
            normalize_action=False,
            use_delta=True,
        )

        extension_pd_joint_pos = PDJointPosControllerConfig(
            [x.name for x in self.robot.active_joints if "extension_joint" in x.name],
            lower=0,
            upper=np.pi,
            stiffness=1000,
            damping=1000,
            friction=joint_friction,
            normalize_action=False,
            force_limit=1e1,
            # use_target=True,
            # use_delta=True,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                wheel_joint=wheel_pd_joint_delta_pos, # 4 controllers for wheels
                extension_joint=extension_pd_joint_pos, # 3 controllers per wheel
                balance_passive_force=False, # Enable gravity
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
    
    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent, default is the qpos and qvel of the robot and any controller state.
        """
        obs = dict(qpos=self.robot.get_qpos(), qvel=self.robot.get_qvel())
        # print("1:", obs)
        controller_state = self.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        # print("2:", obs)
        return obs