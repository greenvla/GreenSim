from dataclasses import MISSING

from isaaclab.actuators.actuator_cfg import ActuatorBaseCfg, IdealPDActuatorCfg
from isaaclab.utils import configclass

from .exponential_actuator import ExponentialActuator
from .frictional_actuator import FrictionalActuator


@configclass
class ExponentialActuatorCfg(ActuatorBaseCfg):
    """Configuration for an exponentially filtered actuator
    https://en.wikipedia.org/wiki/Exponential_smoothing
    """

    class_type: type = ExponentialActuator

    # Exponential filter param
    alpha: float = MISSING


@configclass
class FrictionalActuatorCfg(IdealPDActuatorCfg):
    """Configuration for a frictional actuator.

    Reference:
    `Frictional Actuator Example <https://colab.research.google.com/drive/1HpFE1tJxXutbavw92ZB8a2cLaPc10PkR#scrollTo=bxD4sG4NhFTJ>`

    Attributes:
        class_type (type): The class type of the actuator model.
        breakaway_friction (float): Torque to initiate movement (Nm).
        coulomb_friction (float): Constant opposing friction torque (Nm).
        viscous_coeff (float): Coefficient for viscous friction (Nm/(rad/s)).
        breakaway_velocity (float): Velocity threshold for Stribeck friction (rad/s).
    """

    class_type: type = FrictionalActuator
    # Friction parameters
    breakaway_friction: float = MISSING  # Torque to initiate movement (Nm)
    coulomb_friction: float = MISSING  # Constant opposing friction torque (Nm)
    viscous_coeff: float = MISSING  # Viscous friction coefficient (Nm/(rad/s))
    breakaway_velocity: float = 0.1  # Breakaway velocity threshold (rad/s)

    force_control: bool = False # force control