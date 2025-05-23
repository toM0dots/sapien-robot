"""
NOTE: The coordinate frame in Sapien is: x(forward), y(left), z(upward)

TODO: consider GPU vs CPU in all relevant places

TODO: consider the following methods
- _default_sensor_configs
- _setup_sensors
- _load_lighting
- get_state_dict
- set_state_dict
"""

import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from twsim.robots.transwheel import TransWheel, wheel_radius


def check_collision(a, b):
    a_min_x, a_min_y, a_min_z = a[0]
    a_max_x, a_max_y, a_max_z = a[1]

    b_min_x, b_min_y, b_min_z = b[0]
    b_max_x, b_max_y, b_max_z = b[1]

    return (
        a_min_x <= b_max_x
        and a_max_x >= b_min_x
        and a_min_y <= b_max_y
        and a_max_y >= b_min_y
        and a_min_z <= b_max_z
        and a_max_z >= b_min_z
    )


# TODO: format more like the above (or vice versa)
def bbox_distance(a, b):
    """
    Compute the minimum Euclidean distance between two 3D bounding boxes (batched).
    Each bbox should be a tensor of shape (batch, 2, 3), where [:,0,:] is min_corner and [:,1,:] is max_corner.
    Returns: tensor of shape (batch,)
    """
    # a, b: (batch, 2, 3)
    a_min, a_max = a[:, 0, :], a[:, 1, :]
    b_min, b_max = b[:, 0, :], b[:, 1, :]

    # For each axis, compute the distance between the boxes (0 if overlapping)
    dx = torch.maximum(a_min[:, 0] - b_max[:, 0], b_min[:, 0] - a_max[:, 0])
    dx = torch.clamp(dx, min=0)
    dy = torch.maximum(a_min[:, 1] - b_max[:, 1], b_min[:, 1] - a_max[:, 1])
    dy = torch.clamp(dy, min=0)
    dz = torch.maximum(a_min[:, 2] - b_max[:, 2], b_min[:, 2] - a_max[:, 2])
    dz = torch.clamp(dz, min=0)

    return torch.sqrt(dx * dx + dy * dy + dz * dz)


# TODO: set a reasonable number of max episode steps
@register_env("StepVelSensor-v1", max_episode_steps=200)
class StepVelSensor(BaseEnv):
    """This environment includes a step in front of the robot on a flat plane.

    This environment does not have a success condition.
    """

    SUPPORTED_ROBOTS = ["transwheel"]
    agent: TransWheel

    def __init__(self, *args, robot_uids="transwheel", robot_init_qpos_noise=0.02, **kwargs):
        # TODO: initial position noise is not used
        self.robot_init_qpos_noise = robot_init_qpos_noise

        # TODO: set an initial position for the robot based on terrain and robot dimensions
        self.initial_pose = sapien.Pose(p=[0.0, 0.0, wheel_radius + 1e-3])

        # TODO: let user pass in target velocity
        # self.target_pose = sapien.Pose(p=[0.5, 0.5, 0.0], q=[1.0, 0.0, 0.0, 0.0])
        # self.target_radius = 0.05
        self.target_velocity = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float32)

        self.ground_threshold = -0.1

        self.chassis_lin_vel_prev = 0
        self.distance_prev = None

        # Calling super last since some of the functionality below is called and depends on the variables above
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Specify default simulation/gpu memory configurations. Note that tasks need to tune their GPU memory configurations accordingly
    # in order to save memory while also running with no errors. In general you can start with low values and increase them
    # depending on the messages that show up when you try to run more environments in parallel. Since this is a python property
    # you can also check self.num_envs to dynamically set configurations as well
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    """
    Reconfiguration Code

    below are all functions involved in reconfiguration during environment reset called in the same order. As a user
    you can change these however you want for your desired task. These functions will only ever be called once in general. In CPU simulation,
    for some tasks these may need to be called multiple times if you need to swap out object assets. In GPU simulation these will only ever be called once.
    """

    def _load_agent(self, options: dict):
        "Set the robot's initial pose."
        super()._load_agent(options, self.initial_pose)

    def _load_scene(self, options: dict):
        "Construct the scene ManiSkill will automatically create actors in every sub-scene)."

        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        self.ground = build_ground(
            self.scene,
            floor_width=2,
            floor_length=4,
            texture_square_len=1,
        )

        step_half_size = (0.03, 0.3, 0.02)
        step_material = (0.4, 0.2, 0.4)

        # TODO: why divide by 2
        # TODO: change this back to 0.15
        step_position = (9.15, 0, step_half_size[2] / 2)

        builder = self.scene.create_actor_builder()

        builder.initial_pose = sapien.Pose(p=step_position, q=(1, 0, 0, 0))  # type: ignore
        builder.add_box_collision(half_size=step_half_size)
        builder.add_box_visual(half_size=step_half_size, material=step_material)
        self.step_obj = builder.build_static(name="step")

    def _after_reconfigure(self, options):
        # self.agent_bbox = self.agent.robot.get_first_collision_mesh().bounding_box  # type: ignore
        self.agent_bbox = (
            torch.tensor(
                [[-0.06000149, -0.04700007, -0.03125], [0.06064923, 0.04700007, 0.03125]],
            )
            .unsqueeze(dim=0)
            .repeat(self.num_envs, 1, 1)
        )
        self.step_bbox = (
            torch.tensor(self.step_obj.get_first_collision_mesh().bounding_box.bounds)  # type: ignore
            .unsqueeze(dim=0)
            .repeat(self.num_envs, 1, 1)
        )
        # print(f"{self.agent_bbox.shape=}")
        # print(f"{self.step_bbox.shape=}")
        return super()._after_reconfigure(options)

    def get_bboxes(self):
        # TODO: use the robot's initial bounding box and current position
        # return self.agent.robot.get_first_collision_mesh().bounds, self.step_bbox  # type: ignore
        positions = (
            self.agent.robot.get_root_pose().get_p().unsqueeze(dim=1).repeat(1, 2, 1).cpu()
        )
        # print(f"{positions.shape=}")
        agent_bboxes = (self.agent_bbox + positions).cpu()
        # print(f"{agent_bboxes.shape=}")
        return agent_bboxes, self.step_bbox

    # @property
    # def _default_sensor_configs(self):
    #     # To customize the sensors that capture images/point clouds for the environment observations,
    #     # simply define a CameraConfig as done below for Camera sensors. You can add multiple sensors by returning a list
    #     pose = sapien_utils.look_at(
    #         eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1]
    #     )  # sapien_utils.look_at is a utility to get the pose of a camera that looks at a target

    #     # to see what all the sensors capture in the environment for observations, run env.render_sensors() which returns an rgb array you can visualize
    #     return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # this is just like _sensor_configs, but for adding cameras used for rendering when you call env.render()
        # when render_mode="rgb_array" or env.render_rgb_array()
        # Another feature here is that if there is a camera called render_camera, this is the default view shown initially when a GUI is opened
        pose = sapien_utils.look_at(eye=[0.6, 0.7, 0.6], target=[0.0, 0.0, 0.35])
        # TODO: look into other camera options (specifically the entity uid and mount)
        return [
            CameraConfig(
                "render_camera",
                pose,
                width=512,
                height=512,
                fov=1,
                mount=self.agent.robot.get_root(),
            )
        ]

    # def _setup_sensors(self, options: dict):
    #     # default code here will setup all sensors. You can add additional code to change the sensors e.g.
    #     # if you want to randomize camera positions
    #     return super()._setup_sensors()

    # def _load_lighting(self, options: dict):
    #     # default code here will setup all lighting. You can add additional code to change the lighting e.g.
    #     # if you want to randomize lighting in the scene
    #     return super()._load_lighting()

    """
    Episode Initialization Code

    below are all functions involved in episode initialization during environment reset called in the same order. As a user
    you can change these however you want for your desired task. Note that these functions are given a env_idx variable.

    `env_idx` is a torch Tensor representing the indices of the parallel environments that are being initialized/reset. This is used
    to support partial resets where some parallel envs might be reset while others are still running (useful for faster RL and evaluation).
    Generally you only need to really use it to determine batch sizes via len(env_idx). ManiSkill helps handle internally a lot of masking
    you might normally need to do when working with GPU simulation. For specific details check out the push_cube.py code
    """

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        "Set states for all non-static objects, including the robot."
        self.agent.robot.set_pose(self.initial_pose)

        # NOTE: here is where we would randomly set new poses for the environments (steps or terrain)
        # https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/intro.html#episode-initialization-randomization

    """
    Modifying observations, goal parameterization, and success conditions for your task

    the code below all impact some part of `self.step` function
    """

    def compute_velocity_error(self) -> torch.Tensor:
        "(Batched) Return the difference between the robot's velocity and the target velocity."
        robot_velocity = self.agent.robot.get_root_linear_velocity()
        velocity_diff = robot_velocity.cpu() - self.target_velocity
        velocity_error = torch.linalg.norm(velocity_diff, dim=1)
        return velocity_error

    def evaluate(self):
        "(Batched) Return success and failure conditions for the task."

        # this function is used primarily to determine success and failure of a task, both of which are optional. If a dictionary is returned
        # containing "success": bool array indicating if the env is in success state or not, that is used as the terminated variable returned by
        # self.step. Likewise if it contains "fail": bool array indicating the opposite (failure state or not) the same occurs. If both are given
        # then a logical OR is taken so terminated = success | fail. If neither are given, terminated is always all False.
        #
        # You may also include additional keys which will populate the info object returned by self.step and that will be given to
        # `_get_obs_extra` and `_compute_dense_reward`. Note that as everything is batched, you must return a batched array of
        # `self.num_envs` booleans (or 0/1 values) for success and fail as done in the example below

        robot_position = self.agent.robot.get_pose().get_p()
        vertical_position = robot_position[:, 2]
        below_ground = vertical_position < self.ground_threshold
        fail = below_ground.type(torch.bool).to(device=self.device)  # type: ignore

        # NOTE: no success condition in this environment
        return {"fail": fail}

    def _get_obs_extra(self, info: dict):
        "Additional observation data for the task."
        # should return an dict of additional observation data for your tasks
        # this will be included as part of the observation in the "extra" key when obs_mode="state_dict" or any of the visual obs_modes
        # and included as part of a flattened observation when obs_mode="state". Moreover, you have access to the info object
        # which is generated by the `evaluate` function above

        # We'll also need the robot's orientation
        # Some examples
        #   velocity = self.agent.robot.links_map["pole_1"].linear_velocity
        #   angular_velocity = self.agent.robot.links_map["pole_1"].angular_velocity
        #   root_linear_velocity = (self.agent.robot.root_linear_velocity,)
        #   root_angular_velocity = (self.agent.robot.root_angular_velocity,)
        #   link_orientations = torch.stack(
        #       [link.pose.q for link in self.active_links], dim=-1
        #   ).view(-1, len(self.active_links) * 4)

        # TODO: should we convert to euler angles? is there research on angle representation?
        chassis_orientation = self.agent.robot.get_root_pose().get_q()

        chassis_angular_velocity = self.agent.robot.get_root_angular_velocity()

        # TODO: should this be control_timestep?
        chassis_lin_vel = self.agent.robot.get_root_linear_velocity()
        chassis_linear_acceleration = (
            chassis_lin_vel - self.chassis_lin_vel_prev
        ) / self.sim_timestep
        self.chassis_lin_vel_prev = chassis_lin_vel

        robot_bbox, step_bbox = self.get_bboxes()
        collision = (
            (bbox_distance(robot_bbox, step_bbox) < 0.01)
            .type(torch.float)
            .to(device=self.device)
        )

        # print(f"{self.agent.robot.get_pose().get_p()=}")
        # print(f"{collision=}")

        # print(f"{chassis_orientation.device=}")
        # print(f"{collision.device=}")

        return dict(
            orientation=chassis_orientation,  # (N, 4)
            angular_velocity=chassis_angular_velocity,  # (N, 3)
            linear_acceleration=chassis_linear_acceleration,  # (N, 3)
            # TODO: needs to be optional
            collision=collision,  # (N, 1)
        )

    def compute_dense_reward(self, obs, action: torch.Tensor, info: dict):
        # you can optionally provide a dense reward function by returning a scalar value here. This is used when reward_mode="dense"
        # note that as everything is batched, you must return a batch of of self.num_envs rewards as done in the example below.
        # Moreover, you have access to the info object which is generated by the `evaluate` function above

        # TODO: we need a better way to maintain the max reward (it needs to match the value in compute_normalized_dense_reward)
        self.max_reward = 0
        reward = 0

        info["velocity"] = self.agent.robot.get_root_linear_velocity().cpu()
        info["extension"] = obs[..., 4:8].sum(dim=-1).cpu()

        robot_bbox, step_bbox = self.get_bboxes()
        # info["collision"] = check_collision(robot_bbox.bounds, step_bbox.bounds)  # type: ignore
        # info["collision"] = bbox_distance < 0.01  # type: ignore
        info["distance"] = bbox_distance(robot_bbox, step_bbox)  # type: ignore

        #
        # Reward for moving at the correct velocity
        #

        # NOTE: distance is always positive, so, tanh will increase to 1 as distance decreases
        velocity_error = self.compute_velocity_error()
        info["velocity_error"] = velocity_error.cpu()
        velocity_error_reward = 1 - torch.tanh(5 * velocity_error)
        info["reward_velocity"] = velocity_error_reward
        reward += velocity_error_reward
        self.max_reward += 1

        #
        # Reward part for not extending the extensions
        #

        # TODO: move to device? same for computing velocity error?

        # NOTE: distance is always positive, so, tanh will increase to 1 as distance decreases
        extension_amounts = obs[..., 4:8]
        extension_amount_reward = 1 - torch.tanh(5 * extension_amounts.sum(dim=-1).cpu())
        info["reward_extension"] = extension_amount_reward
        reward += extension_amount_reward
        self.max_reward += 1

        return reward.to(device=self.device)

    def compute_normalized_dense_reward(self, obs, action: torch.Tensor, info: dict):
        "Equal to compute_dense_reward / max possible reward."
        return self.compute_dense_reward(obs=obs, action=action, info=info) / self.max_reward

    # def get_state_dict(self):
    #     # this function is important in order to allow accurate replaying of trajectories. Make sure to specify any
    #     # non simulation state related data such as a random 3D goal position you generated
    #     # alternatively you can skip this part if the environment's rewards, observations, eval etc. are dependent on simulation data only
    #     # e.g. self.your_custom_actor.pose.p will always give you your actor's 3D position
    #     state = super().get_state_dict()
    #     # state["goal_pos"] = add_your_non_sim_state_data_here
    #     return state

    # def set_state_dict(self, state):
    #     # this function complements get_state and sets any non simulation state related data correctly so the environment behaves
    #     # the exact same in terms of output rewards, observations, success etc. should you reset state to a given state and take the same actions
    #     self.goal_pos = state["goal_pos"]
    #     super().set_state_dict(state)
