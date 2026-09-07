
def positions41_to_torques_plus_positions(env, model_positions):
    """
    Input:
        model_positions: 41 target positions from model

    Output:
        41 action values:
            first 25 = torques for body
            last 16 = positions for wrists + fingers
    """
    if len(model_positions) != 41:
        raise ValueError(f"Expected 41 positions, got {len(model_positions)}")

    body_joint_names = [
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

    kp = [200, 200, 100, 350, 70, 70, 200, 200, 100, 350, 70, 70, 70, 30, 30, 20, 20, 20, 30, 30, 20, 20, 20, 20, 20]
    kd = [5, 5, 5, 5, 4, 4, 5, 5, 5, 5, 4, 4, 4, 1.5, 1.5, 1, 1, 1, 1.5, 1.5, 1, 1, 1, 1, 1]

    kp_gain_scale = 1.4
    kd_gain_scale = 1.4

    joint_name_to_id = {name: i for i, name in enumerate(env.unwrapped.robot.data.joint_names)}
    body_joint_ids = [joint_name_to_id[name] for name in body_joint_names]

    q_des = model_positions[:25]
    q_curr = env.unwrapped.robot.data.joint_pos[0, body_joint_ids].cpu().tolist()
    dq_curr = env.unwrapped.robot.data.joint_vel[0, body_joint_ids].cpu().tolist()

    torques = []
    for i in range(25):
        torque = (
            kp[i] * kp_gain_scale * (q_des[i] - q_curr[i])
            - kd[i] * kd_gain_scale * dq_curr[i]
        )
        torques.append(torque)

    wrist_and_finger_positions = model_positions[25:]

    return torques + wrist_and_finger_positions
