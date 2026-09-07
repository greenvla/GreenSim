from __future__ import annotations
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class ExampleActionsPosCfg:
    legs_joint_pos = mdp.JointPositionActionCfg(    #mdp.JointPositionActionCfg(
        asset_name="robot",  # Must match your robot asset name
        joint_names=[
            # Regex to select arm joints
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
        scale=1,
        # use_default_offset=True,
        preserve_order=True,
    )

    torso_joint_pos = mdp.JointPositionActionCfg(    #mdp.JointPositionActionCfg(
        asset_name="robot",  # Must match your robot asset name
        joint_names=[
            # Regex to select arm joints
             "torso_yaw_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_elbow_yaw_joint",
            # "left_wrist_crank_joint",
            # "left_wrist_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_elbow_yaw_joint",
            # "right_wrist_crank_joint",
            # "right_wrist_roll_joint",
            "neck_yaw_joint",
            "neck_pitch_joint",
        ],
        scale=1,
        # use_default_offset=True,
        preserve_order=True,
    )

    wrist_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",  # Must match your robot asset name
        joint_names=[  # Regex to select master finger joints
            "left_wrist_crank_joint",
            "left_wrist_roll_joint",
            "right_wrist_crank_joint",
            "right_wrist_roll_joint",
        ],
        scale=1.0,
        use_default_offset=True,
        preserve_order=True,
    )

    finger_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",  # Must match your robot asset name
        joint_names=[  # Regex to select master finger joints
            "left_pinky_proximal_joint",
            "left_ring_proximal_joint",
            "left_middle_proximal_joint",
            "left_index_proximal_joint",
            "left_thumb_proximal_pitch_joint",
            "left_thumb_proximal_yaw_joint",
            "right_pinky_proximal_joint",
            "right_ring_proximal_joint",
            "right_middle_proximal_joint",
            "right_index_proximal_joint",
            "right_thumb_proximal_pitch_joint",
            "right_thumb_proximal_yaw_joint",
        ],
        scale=1.0,
        use_default_offset=True,
        preserve_order=True,
    )
