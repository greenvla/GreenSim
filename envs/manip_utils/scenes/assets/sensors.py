from typing import Dict, Any
# from isaaclab.sim.spawners.sensors.sensors_cfg import FisheyeCameraCfg
from isaaclab.sensors import CameraCfg, ImuCfg, ContactSensorCfg
import isaaclab.sim as sim_utils
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch

imu_sensor = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/root/imu", debug_vis=False)

feet_contact_sensor = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/root/.*_ankle_roll_link",   # regex for the link names of your feet
    update_period=0.0,        # 0.0 = every physics step, maximum accuracy
    history_length=3,         # for last_air_time/last_contact_time need >=1
    track_air_time=True,      # provides current_air_time / current_contact_time — handy for gait FSM
    debug_vis=False,           # visualization of contact points in the viewport
    # filter_prim_paths_expr=["/World/ground"],  # optional: count contact only with a specific body (floor)
)

# Contact Sensors for Hands

# from isaaclab.sensors import ContactSensorCfg
# import re

# FINGER_LINKS_NAMES=['left_index_intermediate', 'left_index_proximal', 'left_middle_intermediate', 'left_middle_proximal', 
#                    'left_pinky_intermediate', 'left_pinky_proximal', 'left_ring_intermediate', 'left_ring_proximal', 
#                    'left_thumb_distal', 'left_thumb_intermediate', 'left_thumb_proximal', 'right_index_intermediate', 
#                    'right_index_proximal', 'right_middle_intermediate', 'right_middle_proximal', 'right_pinky_intermediate', 
#                    'right_pinky_proximal', 'right_ring_intermediate', 'right_ring_proximal', 
#                    'right_thumb_distal', 'right_thumb_intermediate', 'right_thumb_proximal']


# def create_finger_contact_sensor_cfg(
#     link_names: list[str] = FINGER_LINKS_NAMES,
#     robot_prim_path: str = "{ENV_REGEX_NS}/Robot/root",
#     update_period: float = 0.0,
#     history_length: int = 0,
#     debug_vis: bool = False,
# ) -> ContactSensorCfg:
#     """Create ContactSensorCfg from a list of link names (e.g., ['thumb_distal', ...])."""
#     if not link_names:
#         raise ValueError("link_names must not be empty.")
#     regex = "|".join(re.escape(name) for name in link_names)
#     return ContactSensorCfg(
#         prim_path=f"{robot_prim_path}/({regex})",
#         update_period=update_period,
#         history_length=history_length,
#         debug_vis=debug_vis,
#     )
    
# contact_sensor_cfg = create_finger_contact_sensor_cfg()
