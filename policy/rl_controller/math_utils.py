import torch


def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def quat_unit(a):
    return normalize(a)


def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


def quat_rotate_inverse(q, v):
    q = torch.tensor(q)
    v = torch.tensor(v)
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = (
        q_vec
        * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1)
        * 2.0
    )
    return (a - b + c).numpy()


def quat_to_yaw(quat):
    """
    Args:
        quat: shape (num_envs, 4)
    """
    forward_vec = torch.tensor([1.0, 0.0, 0.0], device=quat.device).repeat(
        quat.shape[0], 1
    )
    heading_vec = quat_apply(quat, forward_vec)
    yaw = torch.atan2(heading_vec[:, 1], heading_vec[:, 0])  # (-pi; pi]
    return yaw


def align_trajectory(traj, step, pos, quat, env_ids):
    """
    Transforms trajectory in XY plane in a way that at 'step' it's located at 'pos' and oriented as 'quat'
    Args:
        step: shape (num_envs)
        pos: shape (num_envs, 3)
        quat: shape (num_envs, 4)
        env_ids: shape (num_envs)
    """
    assert traj["body_pos_world"].ndim == 3  # envs x steps x feats
    n_e, n_s = traj["body_pos_world"].shape[:2]
    assert (
        step.shape[0] == n_e
        and pos.shape[0] == n_e
        and quat.shape[0] == n_e
        and env_ids.shape[0] == n_e
    )

    # change only yaw rotation
    quat = quat.clone()
    quat[:, :2] = 0
    quat = quat_unit(quat)
    init_quat = traj["quaternion"][env_ids, step].clone()
    init_quat[:, :2] = 0
    init_quat = quat_unit(init_quat)

    # transform base pos (only xy)
    traj["body_pos_world"][env_ids, :, :2] -= traj["body_pos_world"][
        env_ids, step, :2
    ].unsqueeze(dim=1)
    quat_tr_pos = quat_mul(quat, quat_conjugate(init_quat))
    traj["body_pos_world"][env_ids] = quat_apply(
        quat_tr_pos.unsqueeze(dim=1).repeat(1, n_s, 1),
        traj["body_pos_world"][env_ids],
    )
    traj["body_pos_world"][env_ids, :, :2] += pos[:, None, :2]

    # transform base quat (only yaw)
    traj["quaternion"][env_ids] = quat_mul(
        quat_conjugate(init_quat).unsqueeze(dim=1).repeat(1, n_s, 1),
        traj["quaternion"][env_ids],
    )
    traj["quaternion"][env_ids] = quat_mul(
        quat.unsqueeze(dim=1).repeat(1, n_s, 1),
        traj["quaternion"][env_ids],
    )
