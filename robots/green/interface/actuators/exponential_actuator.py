from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaacsim.core.utils.types import ArticulationActions

from isaaclab.actuators.actuator_base import ActuatorBase
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .actuator_cfg import ExponentialActuatorCfg


class ExponentialActuator(ActuatorBase):
    r"""Custom actuator model. Applies exponential filtering to the target positions"""

    cfg: ExponentialActuatorCfg
    """The configuration for the actuator model."""

    """
    Operations.
    """

    def __init__(self, cfg: ExponentialActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.prev_target = torch.zeros(
            self._num_envs, self.num_joints, device=self._device
        )
        self.alpha = cfg.alpha

    def reset(self, env_ids: Sequence[int]):
        self.prev_target[env_ids] = 0

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        # apply exponential filter
        desired_position = (
            control_action.joint_positions * self.alpha
            + self.prev_target * (1 - self.alpha)
        )
        self.prev_target = desired_position

        # compute errors
        error_pos = desired_position - joint_pos
        error_vel = control_action.joint_velocities - joint_vel

        # calculate the desired joint torques
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel
        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)
        # set the computed actions back into the control action
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action
