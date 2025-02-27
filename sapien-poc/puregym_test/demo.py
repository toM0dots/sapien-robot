import sapien.core as sapien
from sapien import Pose, Scene

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from transforms3d.euler import euler2quat
from gymnasium import spaces
import random
from pathlib import Path
from shutil import rmtree
from loguru import logger
from PIL import Image
import cv2
import os



class SapienEnv(gym.Env):
    """Superclass for Sapien environments."""

    def __init__(self, control_freq, timestep):
        self.control_freq = control_freq  # alias: frame_skip in mujoco_py
        self.timestep = timestep

        self._scene = sapien.Scene()
        self._scene.set_timestep(timestep)

        self._build_world()
        self.viewer = None
        self.seed()

    def _build_world(self):
        raise NotImplementedError()

    def _setup_viewer(self):
        # raise NotImplementedError()
        pass

    # ---------------------------------------------------------------------------- #
    # Override gym functions
    # ---------------------------------------------------------------------------- #
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer is not None:
            pass  # release viewer

    def render(self, mode='human'):
        self._scene.update_render()
        return
        if mode == 'human':
            if self.viewer is None:
                self._setup_viewer()
            self._scene.update_render()
            if not self.viewer.closed:
                self.viewer.render()
        else:
            raise NotImplementedError('Unsupported render mode {}.'.format(mode))

    # ---------------------------------------------------------------------------- #
    # Utilities
    # ---------------------------------------------------------------------------- #
    def get_actor(self, name):
        all_actors = self._scene.get_all_actors()
        actor = [x for x in all_actors if x.name == name]
        if len(actor) > 1:
            raise RuntimeError(f'Not a unique name for actor: {name}')
        elif len(actor) == 0:
            raise RuntimeError(f'Actor not found: {name}')
        return actor[0]

    def get_articulation(self, name):
        all_articulations = self._scene.get_all_articulations()
        articulation = [x for x in all_articulations if x.name == name]
        if len(articulation) > 1:
            raise RuntimeError(f'Not a unique name for articulation: {name}')
        elif len(articulation) == 0:
            raise RuntimeError(f'Articulation not found: {name}')
        return articulation[0]

    @property
    def dt(self):
        return self.timestep * self.control_freq



# Robot Parameters

control_freq = 107.0
timestep = 1 / control_freq

static_friction = 10.0
dynamic_friction = 10.0
restitution = 0.1
joint_friction = 0.0
joint_damping = 0.0

# The coordinate frame in Sapien is: x(forward), y(left), z(upward)

chassis_length = 16e-2
chassis_width = 32e-2
chassis_thickness = 3e-2
chassis_material = [0.4, 0.4, 0.0]

wheel_radius = 5e-2
wheel_thickness = 1e-2
wheel_material = [0.0, 0.4, 0.4]

num_wheel_extensions = 3
wheel_extension_radial_offset = wheel_radius / 1.2
wheel_extension_angle_offset = np.deg2rad(20)
wheel_extension_length = wheel_radius / 3
wheel_extension_width = wheel_thickness
wheel_extension_thickness = wheel_thickness
wheel_extension_material = [0.4, 0.0, 0.4]

joints = []

duration = 8
total_steps = int(duration / timestep)
    
capture_fps = 25
capture_interval = int((1/capture_fps) / timestep)

    
progress_interval = 200
    
wheel_speed = 3
extension_position = 0

class AntEnv(SapienEnv):
    def __init__(self):
        super().__init__(control_freq=5, timestep=0.01)

        self.actuator = self.get_articulation('ant')
        self._scene.step()  # simulate one step for steady state
        self._init_state = self._scene.physx_system.pack()

        dof = self.actuator.dof
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=[5 + dof + 6 + dof], dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=[dof], dtype=np.float32)
        # Following the original implementation, we scale the action (qf)
        self._action_scale_factor = 50.0

    # ---------------------------------------------------------------------------- #
    # Simulation world
    # ---------------------------------------------------------------------------- #
    def create_ant(self, scene):
        scene.set_timestep(timestep)
        
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
        
        scene.add_ground(altitude=0)

        # 
        # Chassis
        #

        robot_builder = scene.create_articulation_builder()
        
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
        
            # wheel_half_size = [wheel_thickness/2, wheel_radius, wheel_radius]
            # wheel.add_box_collision(half_size=wheel_half_size)
            # wheel.add_box_visual(half_size=wheel_half_size, material=wheel_material)
        
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
            for i in range(num_wheel_extensions):
            
                extension = robot_builder.create_link_builder(wheels[name])
                extension.set_name(f"extension_{name}_{i}")
                extension.set_joint_name(f"extension_joint_{name}_{i}")
        
                # TODO: convert to capsule?
                extension.add_box_collision(half_size=extension_half_size)
                extension.add_box_visual(half_size=extension_half_size, material=wheel_extension_material)
        
                radial_angle = np.deg2rad(i/num_wheel_extensions*360)
                
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
        
        robot = robot_builder.build()
        robot.set_name("ant")
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

        for jname in joints:
            if jname.startswith("wheel_joint"):
                joints[jname].set_drive_velocity_target(wheel_speed)
            
            elif jname.startswith("extension_joint"):
                joints[jname].set_drive_target(extension_position)

        # builder.initial_pose = initial_pose
        return robot


    def _build_world(self):
        physical_material = self._scene.create_physical_material(1.0, 1.0, 0.0)
        self._scene.default_physical_material = physical_material
        render_material = sapien.render.RenderMaterial()
        render_material.set_base_color([0.8, 0.9, 0.8, 1])
        self._scene.add_ground(0.0, render_material=render_material)
        ant = self.create_ant(self._scene)
        ant.set_pose(Pose([0., 0., 0.55]))

    def state_vector(self):
        return [1,2,random.uniform(0.19, 1.01),4,5]

    def step(self, action, step_num=0):
            ant = self.actuator

            x_before = ant.pose.p[0]
            # ant.set_qf(action * self._action_scale_factor)
            # for i in range(self.control_freq):
            #     self._scene.step()
            if step_num > total_steps // 2:
                extension_position = np.deg2rad(90)
        
                for jname in joints:                
                    if jname.startswith("extension_joint"):
                        ant.get.set_drive_target(extension_position)

            ant.set_qf(ant.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            self._scene.step()

            x_after = ant.pose.p[0]

            forward_reward = (x_after - x_before) / self.dt
            ctrl_cost = 0.5 * np.square(action).sum()
            survive_reward = 1.0
            # Note that we do not include contact cost as the original version
            reward = forward_reward - ctrl_cost + survive_reward

            state = self.state_vector()
            is_healthy = (np.isfinite(state).all() and 0.2 <= state[2] <= 1.0)
            done = not is_healthy

            obs = self._get_obs()

            return obs, reward, done, dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_survive=survive_reward)

    def _get_obs(self):
        return 47
    
    def reset(self):
        self._scene.physx_system.unpack(self._init_state)
        # add some random noise
        init_qpos = self.actuator.get_qpos()
        init_qvel = self.actuator.get_qvel()
        qpos = init_qpos + self.np_random.uniform(size=self.actuator.dof, low=-0.1, high=0.1)
        qvel = init_qvel + self.np_random.normal(size=self.actuator.dof) * 0.1
        self.actuator.set_qpos(qpos)
        self.actuator.set_qvel(qvel)
        obs = self._get_obs()
        return obs

def compute_camera_pose_follow(target, offset):
    "Compute the camera pose by specifying forward(x), left(y) and up(z)."
    
    position = np.array(offset)
    
    forward = np.array(target) - position
    forward = forward / np.linalg.norm(forward)

    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    
    up = np.cross(forward, left)
    
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack([forward, left, up], axis=1)
    matrix[:3, 3] = position

    return Pose(matrix)

def main():
    print("Beginning simulation")

    # 
    # Simulate
    #
    
    image_dir = Path("./image_output")
    
    if image_dir.exists():
        rmtree(image_dir)
    
    image_dir.mkdir()
    
    

    # 
    # Camera
    #

    # camera_target = [0, 0, 0]
    camera_target = [0.5, 0, 0]

    # camera_offset = [-0.2, -0.6, 1]
    # camera_offset = [-0.6, -0.6, 0.8]  # default
    camera_offset = [-1.5, -0.6, 2.4]  # fixed trailing view
    # camera_offset = [ 0.0, -0.6, 0.2]  # from right
    # camera_offset = [ 0.0,  0.6, 0.1]  # from left
    # camera_offset = [-0.1,  0.0, 1.0]  # from top
    # camera_offset = [-0.8,  0.0, 0.5]  # from back
    
    camera_pose = compute_camera_pose_follow(camera_target, camera_offset)
    
    near, far = 0.1, 100
    width, height = 640, 480
    fov = np.deg2rad(35)

    env = AntEnv()
    camera = env._scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=fov,
        near=near,
        far=far,
    )
    camera.set_entity_pose(camera_pose)
    capture_i = 0

    env.reset()
    for step in range(total_steps):
        robot = env.get_articulation("ant")

        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action, step_num=step)

        if step % 500 == 0 or step == (total_steps - 1):
            logger.debug('step:', step)
            logger.debug('Pose', robot.get_pose())
            logger.debug('Joint positions', robot.get_qpos())
            logger.debug('Joint velocities', robot.get_qvel())
    
        if step % capture_interval == 0:       
            camera.take_picture()
            rgba = camera.get_picture("Color")
            rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
            rgba_pil = Image.fromarray(rgba_img)
            rgba_pil.save(f"image_output/cam{capture_i:05}.png")
            capture_i += 1
        
        if done:
            print(f'Done at step {step}')
            obs = env.reset()
    env.close()
    print("Simulation completed")
    
    image_folder = './image_output/' # Path to the directory containing images
    video_name = 'output_video.mp4' # Name of the output video file
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), capture_fps, (width, height))
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()

    

if __name__ == "__main__":
    main()