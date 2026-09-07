from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaacsim.core.utils.types import ArticulationActions

from isaaclab.actuators import IdealPDActuator
from isaaclab.actuators.actuator_base import ActuatorBase
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .actuator_cfg import FrictionalActuatorCfg


class FrictionalActuator(IdealPDActuator):
    r"""Ideal torque-controlled actuator model with frictional effects.

    This actuator incorporates friction modeling into the torque control for the actuated joint.
    The friction is modeled using the following components:

    - **Stribeck Friction**: Exhibits a characteristic behavior at low velocities, decaying exponentially as velocity increases.
    - **Coulomb Friction**: Represents constant friction torque opposing motion at nonzero velocities.
    - **Viscous Friction**: Increases linearly with angular velocity.

    The individual friction components are calculated as follows:

    .. math::

        \tau_s = 2(\tau_b - \tau_c) \exp\left(-\left(\frac{\omega}{\omega_s}\right)^2\right) \tanh\left(\frac{\omega}{\omega_c}\right)

    .. math::

        \tau_c = \tau_c \tanh\left(\frac{\omega}{\omega_c}\right)

    .. math::

        \tau_v = b \omega

    The total torque applied is given by:

    .. math::

        \tau_{\text{total}} = \tau_s + \tau_c + \tau_v

    Where:
    - :math:`\tau_b`: Breakaway friction torque.
    - :math:`\tau_c`: Coulomb friction torque.
    - :math:`b`: Viscous friction coefficient.
    - :math:`\omega`: Relative angular velocity.
    - :math:`\omega_s`: Stribeck velocity threshold :math:`\sqrt{2} \cdot \tau_b`.
    - :math:`\omega_c`: Coulomb velocity threshold :math:`\frac{\tau_b}{10}`.

    The parameters for this model are read from the configuration instance passed to the class.

    Reference:
    `Frictional Actuator Example <https://colab.research.google.com/drive/1HpFE1tJxXutbavw92ZB8a2cLaPc10PkR#scrollTo=bxD4sG4NhFTJ>`
    """

    cfg: FrictionalActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: FrictionalActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.force_control = self._parse_joint_parameter(cfg.force_control, 0)
        self.force_control_mask = (self.force_control > 0.5).to(torch.bool)  

        # Zero stiffness & damping ONLY where force_control == 1
        if hasattr(self, "stiffness") and self.stiffness is not None:
            self.stiffness = torch.where(
                self.force_control_mask,
                torch.zeros_like(self.stiffness),
                self.stiffness
            )
        if hasattr(self, "damping") and self.damping is not None:
            self.damping = torch.where(
                self.force_control_mask,
                torch.zeros_like(self.damping),
                self.damping
            )

        # Store friction parameters from configuration
        self.breakaway_friction = self._parse_joint_parameter(cfg.breakaway_friction, 0)
        self.coulomb_friction = self._parse_joint_parameter(cfg.coulomb_friction, 0)
        self.viscous_coeff = self._parse_joint_parameter(cfg.viscous_coeff, 0)
        self.breakaway_velocity = self._parse_joint_parameter(
            cfg.breakaway_velocity, 0.1
        )

        # Calculate Stribeck and Coulomb thresholds
        self.stribeck_velocity_threshold = self.breakaway_velocity * torch.sqrt(
            torch.tensor(2.0)
        )
        self.coulomb_velocity_threshold = self.breakaway_velocity / 10

    def reset(self, env_ids: Sequence[int]):
        pass

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        """Computes the control action based on current joint positions and velocities.

        Args:
            control_action (ArticulationActions): The desired control action.
            joint_pos (torch.Tensor): The current joint positions.
            joint_vel (torch.Tensor): The current joint velocities.

        Returns:
            ArticulationActions: The updated control action with applied efforts.
        """
        # joint_index_map = dict(zip(self.joint_names, self._joint_indices))
        # for name, idx in zip(self.joint_names, self._joint_indices):
        #     print(f"{name}:{idx}")
        
        # call the base method
        control_action = super().compute(control_action, joint_pos, joint_vel)

        # Calculate desired joint torques including friction
        friction_torque = self._compute_friction_torque(joint_vel)

        # apply friction model on the torque
        self.computed_effort = control_action.joint_efforts - friction_torque
        self.applied_effort = self._clip_effort(self.computed_effort)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None

        return control_action

    # def compute(
    #     self,
    #     control_action: ArticulationActions,
    #     joint_pos: torch.Tensor,
    #     joint_vel: torch.Tensor,
    # ) -> ArticulationActions:
    #     """Computes the control action based on current joint positions and velocities.

    #     Args:
    #         control_action (ArticulationActions): The desired control action.
    #         joint_pos (torch.Tensor): The current joint positions.
    #         joint_vel (torch.Tensor): The current joint velocities.

    #     Returns:
    #         ArticulationActions: The updated control action with applied efforts.
    #     """
    #     # Call base PD logic — this sets joint_efforts = PD output for all joints
    #     control_action = super().compute(control_action, joint_pos, joint_vel)

    #     # Compute full friction torque (shape: same as joint_vel)
    #     friction_torque = self._compute_friction_torque(joint_vel)

    #     desired_effort = control_action.joint_efforts

    #     # Compute net effort: τ = τ_desired - τ_friction (to substruct physical friction to simulate it)
    #     compensated_effort = desired_effort - friction_torque

    #     # Blend using mask: force joints use compensated; PD joints keep PD output
    #     final_effort = torch.where(
    #         self.force_control_mask,
    #         compensated_effort,
    #         compensated_effort,
    #         # desired_effort  
    #     )

    #     # Clip and assign
    #     self.computed_effort = final_effort
    #     self.applied_effort = self._clip_effort(self.computed_effort)

    #     control_action.joint_efforts = self.applied_effort
    #     control_action.joint_positions = None
    #     control_action.joint_velocities = None

    #     return control_action


    def _compute_friction_torque(self, joint_vel: torch.Tensor) -> torch.Tensor:
        """Compute the friction torque based on the current joint velocity.

        Args:
            joint_vel (torch.Tensor): The current joint velocity.

        Returns:
            torch.Tensor: The computed friction torque.
        """

        coulomb_smoothed_sign = torch.tanh(joint_vel / self.coulomb_velocity_threshold)
        # Coulomb friction effect
        coulomb_friction_effect = self.coulomb_friction * coulomb_smoothed_sign
        # Stribeck friction component
        stribeck_smoothed_sign = torch.tanh(
            joint_vel / self.stribeck_velocity_threshold
        )
        stribeck_friction = (
            (self.breakaway_friction - self.coulomb_friction)
            * torch.exp(-((joint_vel / self.stribeck_velocity_threshold) ** 2))
            * stribeck_smoothed_sign
        )

        # Viscous friction
        viscous_friction = self.viscous_coeff * joint_vel

        # Total friction torque
        return stribeck_friction + coulomb_friction_effect + viscous_friction
