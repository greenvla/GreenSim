import re
import torch

# Effort-controlled body joints, in the exact order of the model action vector
# (legs_joint_pos[0:12] + torso_joint_pos[12:25]). Their target positions are
# converted to torques here; the remaining 16 action values (wrist + fingers)
# are position commands and are passed through unchanged.
JM_BODY_JOINT_NAMES = [
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
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "neck_yaw_joint",
    "neck_pitch_joint",
]

# Same gain scaling as heavy_green_challenge's TorqueCommandSubscriber.
JM_KP_GAIN_SCALE = 1.4
JM_KD_GAIN_SCALE = 1.4


def _build_gain_arrays_from_env(env, ordered_joint_names):
    """Build per-joint kp/kd from the robot's actuator config.

    Mirrors heavy_green_challenge's TorqueCommandSubscriber._build_gain_arrays_from_env:
    take the first actuator group and regex-match each joint name against its
    stiffness/damping tables, applying the gain scales.
    """
    robot = env.unwrapped.robot
    actuator_cfg = robot.cfg.actuators

    cfg = next(iter(actuator_cfg.values()))

    compiled_kp = [(re.compile(pat), val * JM_KP_GAIN_SCALE) for pat, val in cfg.stiffness.items()]
    compiled_kd = [(re.compile(pat), val * JM_KD_GAIN_SCALE) for pat, val in cfg.damping.items()]

    kp_list = []
    kd_list = []

    for joint_name in ordered_joint_names:
        kp_val = None
        for pattern, val in compiled_kp:
            if pattern.fullmatch(joint_name):
                kp_val = float(val)
                break
        if kp_val is None:
            raise ValueError(f"Joint '{joint_name}' not found in stiffness config.")

        kd_val = None
        for pattern, val in compiled_kd:
            if pattern.fullmatch(joint_name):
                kd_val = float(val)
                break
        if kd_val is None:
            raise ValueError(f"Joint '{joint_name}' not found in damping config.")

        kp_list.append(kp_val)
        kd_list.append(kd_val)

    return kp_list, kd_list


def _prepare(env):
    """Cache gains and joint indices on the env (computed once)."""
    if getattr(env, "_jm_ready", False):
        return

    device = env.unwrapped.device
    joint_name_to_id = {name: i for i, name in enumerate(env.unwrapped.robot.data.joint_names)}

    env._jm_body_joint_ids = [joint_name_to_id[name] for name in JM_BODY_JOINT_NAMES]

    kp_list, kd_list = _build_gain_arrays_from_env(env, JM_BODY_JOINT_NAMES)
    env._jm_kp = torch.tensor(kp_list, dtype=torch.float32, device=device)
    env._jm_kd = torch.tensor(kd_list, dtype=torch.float32, device=device)
    env._jm_num_body = len(JM_BODY_JOINT_NAMES)
    env._jm_ready = True


def jm_step(env, action_pos):
    """Convert position actions to torques for the body, the same way as heavy.

    Input ``action_pos``: tensor of shape (num_envs, 41) of target positions.
        [0:25]  -> body target positions (effort-controlled joints)
        [25:41] -> wrist + finger target positions (position-controlled joints)

    Output: tensor of shape (num_envs, 41) where the first 25 values are PD
    torques tau = kp * (q_des - q_curr) + kd * (dq_des - dq_curr) + tau_ff
    (with dq_des = 0 and tau_ff = 0, since only positions are commanded), and
    the last 16 values are the unchanged position commands.
    """
    _prepare(env)

    robot = env.unwrapped.robot
    device = env.unwrapped.device

    action_pos = torch.as_tensor(action_pos, dtype=torch.float32, device=device)
    if action_pos.dim() == 1:
        action_pos = action_pos.unsqueeze(0)

    n = env._jm_num_body
    body_ids = env._jm_body_joint_ids

    q_des = action_pos[:, :n]
    q_curr = robot.data.joint_pos[:, body_ids]
    dq_curr = robot.data.joint_vel[:, body_ids]

    # dq_des = 0 and tau_ff = 0 because only positions are commanded over the socket.
    torques = env._jm_kp * (q_des - q_curr) - env._jm_kd * dq_curr

    wrist_and_finger_positions = action_pos[:, n:]

    return torch.cat([torques, wrist_and_finger_positions], dim=1)
