"""
Minimal action/observation configs used when the real robot is disabled
(see ``scenes.ENABLE_ROBOT``).

The real ``ExampleActionsCfg`` / ``ExampleObservationsCfg`` reference the real
robot's joints by name; the dummy articulation only has a single ``joint1``
DOF, so these light-weight configs keep the action/observation managers happy
without driving anything meaningful.
"""
from __future__ import annotations
from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
)
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class DummyActionsCfg:
    dummy_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint1"],
        scale=1.0,
        preserve_order=True,
    )


@configclass
class DummyObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"])},
        )
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = False
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
