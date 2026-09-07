"""
Dummy robot asset configuration.

A minimal articulation used in place of the real robot when it is disabled
(see ``scenes.ENABLE_ROBOT``). It is intentionally as light as possible: a
single fixed base box plus one prismatic DOF, authored in
``assets/dummy_robot/dummy_robot.usda``. Because it is a real articulation,
the "robot" scene entity stays valid and nothing keyed on it has to change.
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# assets/dummy_robot/dummy_robot.usda at the repo root.
_DUMMY_ROBOT_USD = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..",
        "assets", "dummy_robot", "dummy_robot.usda",
    )
)

dummy_robot_cfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_DUMMY_ROBOT_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={"joint1": 0.0},
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=1000.0,
            damping=100.0,
        ),
    },
)
