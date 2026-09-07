from __future__ import annotations
from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
)
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class ExampleObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObservationTermCfg(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.15, n_max=0.15), scale=1.0
        )
        base_ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=1.0
        )
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02), scale=1.0
        )
        legs_joint_pos = ObservationTermCfg(
            func=mdp.joint_pos,
            # noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "left_hip_roll_joint",
                        "left_hip_yaw_joint",
                        "left_knee_crank_joint",
                        "left_ankle_u_crank_joint",
                        "left_ankle_b_crank_joint",
                        "right_hip_pitch_joint",
                        "right_hip_roll_joint",
                        "right_hip_yaw_joint",
                        "right_knee_crank_joint",
                        "right_ankle_u_crank_joint",
                        "right_ankle_b_crank_joint",
                    ],
                ),
            },
        )
        torso_joint_pos = ObservationTermCfg(
            func=mdp.joint_pos,
            # noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "torso_yaw_joint",
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_pitch_joint",
                        "left_elbow_yaw_joint",
                        "left_wrist_crank_joint",
                        "left_wrist_roll_joint",
                        
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_pitch_joint",
                        "right_elbow_yaw_joint",
                        "right_wrist_crank_joint",
                        "right_wrist_roll_joint",
                        
                        "neck_yaw_joint",
                        "neck_pitch_joint",
                    ],
                ),
            },
        )
        finger_joint_pos = ObservationTermCfg(
            func=mdp.joint_pos,
            # noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_pinky_proximal_joint",
                        ".*_ring_proximal_joint",
                        ".*_middle_proximal_joint",
                        ".*_index_proximal_joint",
                        ".*_thumb_proximal_pitch_joint",
                        ".*_thumb_proximal_yaw_joint",
                    ],
                ),
            },
        )

        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            # noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=1.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "left_hip_roll_joint",
                        "left_hip_yaw_joint",
                        "left_knee_crank_joint",
                        "left_ankle_u_crank_joint",
                        "left_ankle_b_crank_joint",
                        "right_hip_pitch_joint",
                        "right_hip_roll_joint",
                        "right_hip_yaw_joint",
                        "right_knee_crank_joint",
                        "right_ankle_u_crank_joint",
                        "right_ankle_b_crank_joint",
                        "torso_yaw_joint",
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_pitch_joint",
                        "left_elbow_yaw_joint",
                        "left_wrist_crank_joint",
                        "left_wrist_roll_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_pitch_joint",
                        "right_elbow_yaw_joint",
                        "right_wrist_crank_joint",
                        "right_wrist_roll_joint",
                        "neck_yaw_joint",
                        "neck_pitch_joint",
                    ],
                ),
            },
        )
        actions = ObservationTermCfg(func=mdp.last_action)
        # clock = ObservationTermCfg(func=loco_observations.clock)
        # Commands
        # velocity_commands = ObservationTermCfg(
        #     func=mdp.generated_commands, params={"command_name": "base_velocity"}
        # )
        # gait_command = ObservationTermCfg(
        #     func=mdp.generated_commands, params={"command_name": "gait"}
        # )

        # height_command = ObservationTermCfg(
        #     func=mdp.generated_commands, params={"command_name": "height_offset"}
        # )

        def __post_init__(self):
            # self.enable_corruption = True
            self.concatenate_terms = False  # True
            self.history_length = 1

    # observation groups
    policy: PolicyCfg = PolicyCfg()
