import sapien
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
@register_agent()

class TFWheelBot(BaseAgent):
    uid = "tfwheel_bot"

    def assemble():
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
        robot.set_name("robot")
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