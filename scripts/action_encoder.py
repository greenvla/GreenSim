import torch


def encode_act_v2(act, device, num_envs):
    big_list = sum(act.values(), [])
    return torch.as_tensor(big_list, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)
