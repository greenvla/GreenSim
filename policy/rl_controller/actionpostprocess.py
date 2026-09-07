import torch
import numpy as np
from typing import List, Dict
from rl_controller.filters import LowPassFilter


class Action:
    def __init__(
        self,
        config: Dict,
        joint_order: List,
        default_joints_pos: torch.Tensor,
        device: torch.DeviceObjType,
    ) -> None:
        self.cfg = config
        self.joint_order = joint_order
        self.default_joints_pos = default_joints_pos
        self.device = device

        self.num_actions = len(self.joint_order)
        control_type = self.cfg["control_type"]
        if isinstance(control_type, dict):
            self.control_type = np.array(
                [control_type[joint] for joint in self.joint_order]
            )
        else:
            self.control_type = np.array([control_type] * self.num_actions)
        self.init_control_gains(self.cfg)
        self.limit_actions_by_torque = self.cfg["limit_actions_by_torque"]
        if self.limit_actions_by_torque:
            self.torque_limits = torch.tensor(
                [self.cfg["torque_limits"][joint] for joint in self.joint_order],
                dtype=torch.float32,
                device=self.device,
            )
        action_scale = self.cfg["action_scale"]
        if isinstance(action_scale, dict):
            action_scale = torch.tensor(
                [action_scale[joint] for joint in self.joint_order],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            action_scale = torch.tensor(
                [action_scale] * self.num_actions,
                dtype=torch.float32,
                device=self.device,
            )
        self.action_scale = action_scale
        self.action_clip = self.cfg["clip_actions"]
        self.use_lpf = self.cfg["use_lpf"]

        if self.use_lpf:
            raise NotImplementedError("This code requires a thorough revision")
            self.init_filter(self.cfg["dt"])

        self.reset()

    def init_filter(self, sample_time, order=1, cutoff_freq=4):
        self.filter = LowPassFilter(
            order=order,
            cutoff_freq=cutoff_freq,
            input_dim=self.num_actions,
            sampling_freq=1 / sample_time,
        )

    def init_control_gains(self, control_config):
        self.p_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.d_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )

        stiffness_conf = control_config["stiffness"]
        for k, v in stiffness_conf.items():
            indices = [i for i, name in enumerate(self.joint_order) if k in name]
            self.p_gains[indices] = v

        damping_conf = control_config["damping"]
        for k, v in damping_conf.items():
            indices = [i for i, name in enumerate(self.joint_order) if k in name]
            self.d_gains[indices] = v

    def process_action(
        self,
        action: torch.Tensor,
        joint_pos: torch.Tensor = None,
        joint_vel: torch.Tensor = None,
    ):
        """
        Process action as follows:
            1. Numerical clip
            2. Scale
            3. Add default offset
            4. [Clip by torque]
        """
        action_raw = torch.clip(action, -self.action_clip, self.action_clip)
        action = action_raw * self.action_scale

        action[self.control_type == "P"] += self.default_joints_pos[
            self.control_type == "P"
        ]

        if self.limit_actions_by_torque:  # limit actions by max torque
            clipped_pos_diff = torch.clip(
                action - joint_pos,
                (self.d_gains * joint_vel - self.torque_limits) / self.p_gains,
                (self.d_gains * joint_vel + self.torque_limits) / self.p_gains,
            )
            action[self.control_type == "P"] = (joint_pos + clipped_pos_diff)[
                self.control_type == "P"
            ]

            clipped_vel_diff = torch.clip(
                action - joint_vel,
                -self.torque_limits / self.d_gains,
                self.torque_limits / self.d_gains,
            )
            action[self.control_type == "V"] = (joint_vel + clipped_vel_diff)[
                self.control_type == "V"
            ]

            action[self.control_type == "T"] = torch.clip(
                action, -self.torque_limits, self.torque_limits
            )[self.control_type == "T"]

        return action_raw, action

    def reset(self):
        if self.use_lpf:
            self.filter.reset(self.default_joints_pos.clone())
