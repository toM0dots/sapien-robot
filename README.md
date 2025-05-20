# Transformable Wheel Robot

## Setup

Starting from the root of the repository:

```bash
cd twsim
python -m pip install --editable .
```

## Development Notes

Methods to override/implement:

- Environment creation and reset
  - set `SUPPORTED_ROBOTS = ...`
  - set `agent: ...`
  - `__init__`
- Environment reset
  - `_load_scene`
  - (optional) `_load_lighting`
  - `_initialize_episode`
- Environment step
  - (optional) `_before_control_step`
  - (optional) `_before_simulation_step`
  - (optional) `_after_simulation_step`
  - (optional) `_after_control_step`
  - (optional) `_evaluate`
  - (optional) `_get_obs_extra`
  - (optional) `_compute_dense_reward`
- Robot (which is managed by an "agent")
  - Useful `base_agent` properties and methods
    - `control_mode: str`
    - `disable_self_collisions: bool`
    - `fix_root_link: bool`
    - `get_state() -> Dict`
    - `robot: Articulation`
  - ManiSkill really wants to create one using URDF or MJCF
    - Set `urdf_path` instead of using `_load_articulation`
    - Define a `urdf_config` object
  - Set a uid and register the agent with ManiSkill
  - (optional) Define keyframes
  - `_load_articulation`
  - (optional) `action_space` (batched) (default will take from the controller configuration)
  - (optional) `single_action_space` (default will take from the controller configuration)
  - (optional) `set_action` (batched) (default will take from the controller configuration)
  - (optional) `_controller_configs` (default will create PD position controllers for active each joint)
  - (optional) `get_proprioception` (default will return joint positions and velocities)
  - (optional) `_sensor_configs` (includes only camera sensors)

Comments from the documentation:

- "Static actors must have an initial pose set before calling build_static via builder.initial_pose = ..."
- "We recommend defining normalized reward function as these tend to be easier to learn from."
- Might need to set `sim_config.spacing` so that each parallel "sub-scene" is spaced out enough to avoid collisions.
- `forces = self.scene.get_pairwise_contact_forces(actor_1, link_2)`
- `actor.get_net_contact_forces()` # shape (N, 3), N is number of environments
- `articulation.get_net_contact_forces(link_names)` # shape (N, len(link_names), 3)
- Use "scene masks" to make each sub-scene unique (e.g., different terrains or obstacles).
  - [docs](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/advanced.html#scene-masks)
- "env.get_state_dict() returns a state dictionary containing the entirety of simulation state of actors and articulations in the state dict registry"
- ManiSkill environments are batched by default
