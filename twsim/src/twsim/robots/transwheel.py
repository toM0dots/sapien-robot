"""
NOTE: The coordinate frame in Sapien is: x(forward), y(left), z(upward)

TODO:
- Define keyframes with extensions all the way out and all the way in
- Check all physical dimensions (try different units)
"""

import numpy as np
from gymnasium import spaces
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import deepcopy_dict
from mani_skill.agents.controllers.base_controller import ControllerConfig
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosControllerConfig
from mani_skill.agents.controllers.pd_joint_vel import PDJointVelControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.utils.structs import Articulation
from sapien import Pose
from transforms3d.euler import euler2quat

#
# Robot Parameters
#

# control_freq = 100.0
# timestep = 1 / control_freq

static_friction = 10.0
dynamic_friction = 10.0
restitution = 0.1
joint_friction = 0.0
joint_damping = 0.0

chassis_length = 9e-2
chassis_width = 6e-2
chassis_thickness = 1e-2
chassis_material = (0.4, 0.4, 0.0)

wheel_radius = 1.5e-2
wheel_thickness = 0.2e-2
wheel_material = (0.0, 0.4, 0.4)

num_extensions = 3

extension_radial_offset = 0.875e-2
extension_angle_offset = np.deg2rad(20)
extension_length = 1.625e-2
extension_width = 0.375e-2
extension_thickness = 0.2e-2
extension_material = (0.4, 0.0, 0.4)


@register_agent()
class TransWheel(BaseAgent):
    uid = "transwheel"
    fix_root_link = False
    disable_self_collisions = True

    def create(self, initial_pos) -> Articulation:
        #
        # Chassis
        #

        robot_builder = self.scene.create_articulation_builder()
        robot_builder.initial_pose = initial_pos

        chassis_half_size = (
            chassis_length / 2,
            chassis_width / 2,
            chassis_thickness / 2,
        )

        # TODO: remove and use initial_pose?
        # chassis_vertical_offset = wheel_radius + 7e-2
        # chassis_pose = Pose(p=[0, 0, chassis_vertical_offset])

        chassis = robot_builder.create_link_builder()
        chassis.set_name("chassis")
        chassis.add_box_collision(half_size=chassis_half_size)
        chassis.add_box_visual(half_size=chassis_half_size, material=chassis_material)

        #
        # Wheels and revolute joints
        #

        wheel_half_thickness = wheel_thickness / 2

        front_rear_offset = chassis_half_size[0]
        left_right_offset = chassis_half_size[1] + wheel_thickness

        # NOTE: by default, cylinders are oriented along the x-axis
        cylinder_rotation = euler2quat(0, 0, np.deg2rad(90))

        wheel_parameters = [
            ("front_left", front_rear_offset, left_right_offset, cylinder_rotation),
            ("front_right", front_rear_offset, -left_right_offset, cylinder_rotation),
            ("rear_left", -front_rear_offset, left_right_offset, cylinder_rotation),
            ("rear_right", -front_rear_offset, -left_right_offset, cylinder_rotation),
        ]

        wheels = {}

        for name, fr, lr, quat in wheel_parameters:
            wheel = robot_builder.create_link_builder(chassis)
            wheel.set_name(f"wheel_{name}")
            wheel.set_joint_name(f"wheel_joint_{name}")

            # TODO: convert to spheroid using convex mesh?
            wheel.add_cylinder_collision(
                radius=wheel_radius, half_length=wheel_half_thickness
            )
            wheel.add_cylinder_visual(
                radius=wheel_radius,
                half_length=wheel_half_thickness,
                material=wheel_material,
            )

            wheel.set_joint_properties(
                "revolute",
                limits=[[-np.inf, np.inf]],
                pose_in_parent=Pose(p=[fr, lr, 0], q=quat),  # type: ignore
                pose_in_child=Pose(),
                friction=joint_friction,  # type: ignore
                damping=joint_damping,  # type: ignore
            )

            wheels[name] = wheel

        #
        # Wheel extensions
        #
        # NOTE: extensions are relative to the wheel:
        #    x -> y
        #    y -> x
        #    z -> z
        #

        extension_half_size = (
            extension_width / 2,
            extension_length / 2,
            extension_thickness / 2,
        )

        for name, fr, lr, quat in wheel_parameters:
            for i in range(num_extensions):
                extension = robot_builder.create_link_builder(wheels[name])
                extension.set_name(f"extension_{name}_{i}")
                extension.set_joint_name(f"extension_joint_{name}_{i}")

                # TODO: convert to capsule?
                extension.add_box_collision(half_size=extension_half_size)
                extension.add_box_visual(
                    half_size=extension_half_size, material=extension_material
                )

                radial_angle = np.deg2rad(i / num_extensions * 360)

                y = extension_radial_offset * np.cos(radial_angle)
                z = extension_radial_offset * np.sin(radial_angle)

                x = np.copysign(extension_width, lr)

                extension.set_joint_properties(
                    "revolute",
                    # TODO: Upper limit should take into account angle offset
                    limits=[[0, np.pi]],
                    pose_in_parent=Pose(p=[x, y, z]),
                    pose_in_child=Pose(
                        p=[0, -extension_length / 2, 0],
                        q=euler2quat(
                            np.deg2rad(90) - radial_angle + extension_angle_offset,
                            0,
                            0,
                        ),  # type: ignore
                    ),
                    friction=joint_friction,  # type: ignore
                    damping=joint_damping,  # type: ignore
                )

        #
        # Finalize the articulated robot
        #

        robot_builder.set_name("transwheel")
        robot = robot_builder.build()

        # TODO: remove?
        # robot.set_pose(chassis_pose)

        # TODO: remove? this is handled by _controller_configs
        if False:
            joints = {joint.get_name(): joint for joint in robot.get_active_joints()}

            # TODO: dig into joint mode options in more detail
            # TODO: don't hardcode stiffness and damping
            joint_mode = "force"

            for joint_name in joints:
                if joint_name.startswith("wheel_joint"):
                    joints[joint_name].set_drive_properties(
                        stiffness=10, damping=100, mode=joint_mode
                    )

                elif joint_name.startswith("extension_joint"):
                    joints[joint_name].set_drive_properties(
                        stiffness=1000, damping=10, mode=joint_mode
                    )

                else:
                    raise ValueError(
                        f"Unknown joint name: {joint_name}. "
                        "Expected to start with 'wheel_joint' or 'extension_joint'."
                    )

        return robot

    def _load_articulation(self, initial_pose: Pose | None = None):
        # TODO: remove this statement once debugged
        # print(f"{initial_pose=}", type(initial_pose))
        self.robot = self.create(initial_pose)

        # Cache robot link names
        self.robot_link_names = [link.name for link in self.robot.get_links()]

    @property
    def action_space(self) -> spaces.Space:
        "(Batched) Although we have 4 + 4*num_wheel_extensions controllers, we only need 8 actions."

        # The controller action space includes a space for each wheel and each extension
        # We only need one space for each wheel and one space for each SET of extensions
        # The controller configuration space is defined in _controller_configs, and it should
        # first define the wheel controllers and then the extension controllers.
        # We use that order here to define the action space based on the controller space so that
        # we are only setting limits in one place. Alternatively, we could define a normalized
        # action space for both the controller config and here, but it still separates the logic
        # into two places that would need to be kept in sync.
        controller_action_space = self.controller.action_space

        num_actions = 8
        return spaces.Box(
            low=controller_action_space.low[..., :num_actions],  # type: ignore
            high=controller_action_space.high[..., :num_actions],  # type: ignore
            dtype=controller_action_space.dtype,  # type: ignore
        )

    @property
    def single_action_space(self) -> spaces.Space:
        "(Not Batched) Although we have 4 + 4*num_wheel_extensions controllers, we only need 8 actions."
        # NOTE: see comments in action_space

        # NOTE: the difference here is the use of single action space on the controller
        controller_action_space = self.controller.single_action_space
        num_actions = 8
        return spaces.Box(
            low=controller_action_space.low[:num_actions],  # type: ignore
            high=controller_action_space.high[:num_actions],  # type: ignore
            dtype=controller_action_space.dtype,  # type: ignore
        )

    def set_action(self, action):
        "(Batched) Set the agent's action which is to be executed in the next environment timestep."

        # Override the default set_action method so that we can expand back to the
        # correct number of controllers.

        if not self.scene.gpu_sim_enabled:
            if np.isnan(action).any():
                raise ValueError("Action cannot be NaN. Received:", action)

        print(f"{action=}", type(action))
        print(f"{action.shape=}")

        wheel_actions = action[..., :4]
        print(f"{wheel_actions=}")
        extension_actions = action[..., 4:].repeat_interleave(num_wheel_extensions)
        print(f"{extension_actions=}")

        new_action = np.concatenate((wheel_actions, extension_actions), axis=-1)
        print(f"{new_action=}", type(new_action))
        print(f"{new_action.shape=}")

        self.controller.set_action(new_action)

    @property
    def _controller_configs(self) -> dict[str, ControllerConfig]:
        "Returns a dict of controller configs for this agent."

        joint_names = self.robot.active_joints
        wheel_joint_names = [x.name for x in joint_names if "wheel_joint" in x.name]
        ext_joint_names = [x.name for x in joint_names if "extension_joint" in x.name]

        # TODO: set reasonable values (pass in as arguments to class?)
        max_linear_velocity = 0.5  # inches / second ?
        max_angular_velocity = max_linear_velocity / wheel_radius
        velocity_damping = 100

        wheel_velocity_controllers = PDJointVelControllerConfig(
            joint_names=wheel_joint_names,
            # TODO: set to infinity?
            lower=-max_angular_velocity,
            upper=max_angular_velocity,
            damping=velocity_damping,
            # force_limit: Union[float, Sequence[float]] = 1e10
            # friction: Union[float, Sequence[float]] = 0.0
            normalize_action=False,
            # drive_mode: Union[Sequence[DriveMode], DriveMode] = "force"
        )

        # TODO: set reasonable values
        position_stiffness = 1000
        position_damping = 1000

        extension_position_controllers = PDJointPosControllerConfig(
            joint_names=ext_joint_names,
            lower=0,
            upper=np.pi,
            stiffness=position_stiffness,
            damping=position_damping,
            # force_limit: Union[float, Sequence[float]] = 1e10
            # friction: Union[float, Sequence[float]] = 0.0
            # use_delta: bool = False
            # use_target: bool = False
            # interpolate: bool = False
            # normalize_action: bool = True
            # drive_mode: Union[Sequence[DriveMode], DriveMode] = "force"
        )

        controller_configs = dict(
            wheel_vel_ext_pos=dict(
                wheel_velocity=wheel_velocity_controllers,
                extension_position=extension_position_controllers,
                balance_passive_force=False,
            ),
            # TODO: add a configuration with extension_velocity control
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def get_proprioception(self):
        "(Batched) Return the proprioceptive state of the agent. (called by default env._get_obs_agent)"

        controller_state = self.controller.get_state()
        if len(controller_state) > 0:
            raise ValueError("Controller state is not empty. This is unexpected.")

        # TODO: proprioception depends on the controller mode (extensions as position or velocity)

        # By default, the proprioceptive state is the qpos and qvel of the robot and any controller state.
        #     obs = dict(qpos=self.robot.get_qpos(), qvel=self.robot.get_qvel())
        # We only need a subset of the data
        wheel_vel = self.robot.get_qvel()[..., :4]
        extension_pos = self.robot.get_qpos()[..., 4::num_extensions]

        return dict(wheel_velocities=wheel_vel, extension_positions=extension_pos)
