from __future__ import annotations
import time
import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.math import quat_rotate
from isaaclab.managers import EventTermCfg
from isaaclab.envs import ManagerBasedRLEnvCfg


def step_physics_only(env, num_steps: int = 1):
    """Step the simulation without applying actions."""
    for _ in range(num_steps):
        env.scene.write_data_to_sim()  # write any pending state (e.g., after reset)
        env.sim.step()  # step physics
        env.scene.update(env.step_dt)  # update scene buffers


def reset_robot_to_home_pose(env, env_ids: torch.Tensor):
    """
    Reset the robot to a defined 'ready' pose by overriding selected joint positions,
    while respecting the initial configuration state for other joints.

    Args:
        env: The environment object containing the simulation scene and configuration.
        env_ids (torch.Tensor): Tensor of environment IDs to apply the reset on.

    Behavior:
        - Retrieves initial joint positions from configuration.
        - Applies fixed overrides for key joints to put the robot in a ready pose.
        - Applies initial joint velocities if available.
        - Writes the updated joint states to the simulation.
        - Sleeps for 2 seconds to allow the reset to take effect.
    """
    robot = env.scene["robot"]
    joint_names = robot.data.joint_names
    num_joints = len(joint_names)

    # Retrieve the robot articulation configuration and its initial state
    robot_cfg: ArticulationCfg = env.cfg.scene.robot
    init_state_cfg = robot_cfg.init_state  # type: ignore

    # Initialize joint positions tensor based on initial configuration patterns
    pos_target = torch.zeros(num_joints, device=robot.device)
    for pattern, pos_val in init_state_cfg.joint_pos.items():
        for i, name in enumerate(joint_names):
            if (
                pattern == ".*"
                or name == pattern
                or (pattern.endswith(".*") and name.startswith(pattern[:-2]))
            ):
                pos_target[i] = float(pos_val)

    # Define specific joint overrides for ready pose - mostly elbow joints
    ready_pose_overrides = {
        # "left_shoulder_pitch_joint": 1.0,
        # "left_shoulder_roll_joint": 0,
        # "left_shoulder_yaw_joint":0,
        # "left_elbow_pitch_joint": -2.2,
        # "left_elbow_yaw_joint": 1.5708,
        # "right_shoulder_pitch_joint": -0.0,
        # "right_elbow_pitch_joint": 0,
        # "right_elbow_yaw_joint": 0,
        # "neck_pitch_joint":0.5
    }

    # Apply the overrides to the target joint positions
    for joint_name, pos_val in ready_pose_overrides.items():
        if joint_name in joint_names:
            idx = joint_names.index(joint_name)
            pos_target[idx] = float(pos_val)
        else:
            # Log warning if joint is missing; robust but non-fatal
            print(
                f"[WARNING] Joint '{joint_name}' not found in robot. Cannot apply ready pose."
            )

    # Expand positions and initialize velocities for all environment instances
    N = len(env_ids)
    pos = pos_target.unsqueeze(0).repeat(N, 1)
    vel = torch.zeros_like(pos)

    # If initial joint velocities are specified, apply those patterns as well
    if init_state_cfg.joint_vel is not None:
        vel_target = torch.zeros(num_joints, device=robot.device)
        for pattern, vel_val in init_state_cfg.joint_vel.items():
            for i, name in enumerate(joint_names):
                if (
                    pattern == ".*"
                    or name == pattern
                    or (pattern.endswith(".*") and name.startswith(pattern[:-2]))
                ):
                    vel_target[i] = float(vel_val)
        vel = vel_target.unsqueeze(0).repeat(N, 1)

    # Write the computed joint states into the simulation environment
    robot.write_joint_state_to_sim(
        position=pos,
        velocity=vel,
        joint_ids=None,  # apply update to all joints
        env_ids=env_ids,
    )

    # # Pause to ensure the reset has time to propagate in the simulation


def place_sphere_at_hand(env, env_ids: torch.Tensor):
    """
    Places a sphere rigid object near the left hand of the robot after it has
    been reset to the ready pose.

    Args:
        env: The environment object containing the simulation scene and configuration.
        env_ids (torch.Tensor): Tensor of environment IDs to apply the placement on.

    Behavior:
        - Retrieves the pose of the left hand's proximal thumb base.
        - Computes an offset in the hand's frame.
        - Transforms offset into world coordinates.
        - Positions the sphere at the resulting location with the same orientation as the hand.
        - Sets the sphere's velocity to zero.
        - Writes pose and velocity to simulation.

    Raises:
        RuntimeError: If the specified hand body part is not found in the robot.
    """
    return
    robot: ArticulationCfg = env.scene["robot"]
    # sphere: RigidObjectCfg = env.scene["sphere"]

    # Name of the body link to place the sphere relative to
    body_name = "left_thumb_proximal_base"
    if body_name not in robot.data.body_names:
        raise RuntimeError(f"Body '{body_name}' not found in robot.")

    body_idx = robot.data.body_names.index(body_name)

    # Fetch current position and orientation (quaternion) of the specified body part
    hand_pos = robot.data.body_pos_w[env_ids, body_idx]  # Shape: (N, 3)
    hand_quat = robot.data.body_quat_w[
        env_ids, body_idx
    ]  # Shape: (N, 4), quaternion (w,x,y,z)

    # Define a small local offset in hand coordinates to position the sphere
    local_offset = torch.tensor([-0.03, 0.045, 0.085], device=robot.device)
    local_offset = local_offset.unsqueeze(0).repeat(len(env_ids), 1)  # (N, 3)

    # Rotate the offset from the hand frame to the world frame
    world_offset = quat_rotate(hand_quat, local_offset)  # (N, 3)

    # Calculate absolute position of the sphere in the world frame
    sphere_pos = hand_pos + world_offset

    # Use the same orientation as the hand for the sphere
    sphere_quat = hand_quat

    # Concatenate position and quaternion to form full pose (N, 7)
    new_poses = torch.cat([sphere_pos, sphere_quat], dim=1)

    # Initialize zero velocity for the sphere (6 DOF: linear + angular)
    N = len(env_ids)
    vel = torch.zeros(N, 6, device=sphere.device)

    # Write the new pose and velocity state into the simulation
    sphere.write_root_pose_to_sim(new_poses, env_ids=env_ids)
    sphere.write_root_velocity_to_sim(vel, env_ids=env_ids)



class ExampleEventsCfg():
    events = {
        "reset_robot_to_home_pose": EventTermCfg(
            func=reset_robot_to_home_pose,
            mode="reset",
            min_step_count_between_reset=0,
        ),
        "place_sphere_at_hand": EventTermCfg(
            func=place_sphere_at_hand,
            mode="reset",
            min_step_count_between_reset=0,
        ),
    }

# Example usage snippet from a configuration class:
#
# class ExampleEnvCfg(ManagerBasedRLEnvCfg):
#     events = {
#         "reset_robot_to_ready_pose": EventTermCfg(
#             func=reset_robot_to_ready_pose,
#             mode="reset",
#             min_step_count_between_reset=0,
#         ),
#         "place_sphere_at_hand": EventTermCfg(
#             func=place_sphere_at_hand,
#             mode="reset",
#             min_step_count_between_reset=0,
#         ),
#     }
