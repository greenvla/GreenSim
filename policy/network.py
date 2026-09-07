"""
Standalone wrapper around the JIT locomotion policies copied from
``sberchelx_meta/external/rl_controller``. Replaces the ROS-coupled
``controller_rl_adapter.py`` from the original repo with the minimal
glue needed to drive an IsaacLab ``ManagerBasedRLEnv``.

Design constraints:
- ``rl_controller/`` is a verbatim copy of the upstream package and uses
  absolute imports like ``from rl_controller import ...``. We make those
  resolve by prepending this folder to ``sys.path``.
- ``weights/<policy_name>/`` holds an unmodified ``config.yaml`` and
  ``body_latest.jit`` so that future updates only require copying these
  two folders.
"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

POLICY_DIR = os.path.dirname(os.path.realpath(__file__))
if POLICY_DIR not in sys.path:
    sys.path.insert(0, POLICY_DIR)

from rl_controller import RLControllerZestAP3Open  # noqa: E402

BODY_JOINT_ORDER = [
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
FINGER_JOINT_ORDER = [
    "left_pinky_proximal_joint",
    "right_pinky_proximal_joint",

    "left_ring_proximal_joint",
    "right_ring_proximal_joint",

    "left_middle_proximal_joint",
    "right_middle_proximal_joint",

    "left_index_proximal_joint",
    "right_index_proximal_joint",

    "left_thumb_proximal_pitch_joint",
    "right_thumb_proximal_pitch_joint",

    "left_thumb_proximal_yaw_joint",
    "right_thumb_proximal_yaw_joint",
]

_POLICY_NAME_BY_TYPE = {
    "zest": "no_dwell_walk_manip_data_vel_med_sigma_angmom_pen_symm",
}

_CONTROLLER_CLS_BY_TYPE = {
    "zest": RLControllerZestAP3Open,
}


def _load_policy_model(policy_dir: str, device: torch.device):
    """Load policy weights from ``policy_dir``.

    Tries ``body_latest.jit`` first, then ``body_latest.onnx``. For ONNX the
    ``onnxruntime`` import is lazy.
    """
    jit_path = os.path.join(policy_dir, "body_latest.jit")
    onnx_path = os.path.join(policy_dir, "body_latest.onnx")
    if os.path.exists(jit_path):
        return torch.jit.load(jit_path, map_location=device)
    if os.path.exists(onnx_path):
        import onnxruntime as ort  # noqa: WPS433 (lazy import is intentional)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        return ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
    raise FileNotFoundError(
        f"No policy weights found in {policy_dir} (expected body_latest.jit or body_latest.onnx)"
    )


class Network:
    """Drives an IsaacLab env with a JIT locomotion policy."""

    def __init__(
        self,
        env,
        type: str = "zest",
        device: str = "cpu",
        debug: bool = False,
        closed_kinematics: bool = True,
    ) -> None:
        if type not in _POLICY_NAME_BY_TYPE:
            raise ValueError(
                f"Invalid network type: {type!r}. "
                f"Expected one of {list(_POLICY_NAME_BY_TYPE)}."
            )

        self.type = type
        self.env = env
        self.device = torch.device(device)
        self._debug = debug
        self._closed_kinematics = closed_kinematics
        self._first_call = True
        self._dump_target_once = True
        self._call_count = 0
        self._verbose_steps = 5  # print every step's obs/target for first N calls

        policy_name = _POLICY_NAME_BY_TYPE[type]
        policy_dir = os.path.join(POLICY_DIR, "weights", policy_name)
        config_path = os.path.join(policy_dir, "config.yaml")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg

        policy_model = _load_policy_model(policy_dir, self.device)
        controller_cls = _CONTROLLER_CLS_BY_TYPE[type]
        self.controller = controller_cls(
            cfg, policy_model, self.device, closed_kinematics=closed_kinematics
        )

        self.policy_joint_order: List[str] = list(cfg["joint_order"])
        self.model_joint_order: List[str] = BODY_JOINT_ORDER

        self._validate_joint_orders()

        self.model_to_policy_idx: List[int] = [
            self.model_joint_order.index(name) for name in self.policy_joint_order
        ]
        self.policy_to_model_idx: List[int] = [
            self.policy_joint_order.index(name) for name in self.model_joint_order
        ]

        robot = env.unwrapped.scene["robot"]
        robot_joint_names: List[str] = list(robot.data.joint_names)
        self._sim_indices: List[int] = [
            robot_joint_names.index(name) for name in self.model_joint_order
        ]

        # USD/articulation default joint pos (the offset re-added by
        # JointPositionActionCfg(use_default_offset=True)).
        default_joint_pos_all = robot.data.default_joint_pos[0].detach().cpu().numpy()
        self._default_joint_pos_model_order = np.asarray(
            default_joint_pos_all[self._sim_indices], dtype=np.float32
        )

        # Policy's own expected default joint pose (in policy joint order).
        cfg_defaults = cfg["init_state"]["default_joint_angles"]
        self._policy_default_pos = np.asarray(
            [cfg_defaults[name] for name in self.policy_joint_order], dtype=np.float32
        )

        self._n_fingers = len(FINGER_JOINT_ORDER)

        # The asset's FrictionalActuator zeros stiffness/damping wherever
        # `force_control=true` in the cfg, which is the case for every
        # locomotion / arm joint. The policy was trained against PD control,
        # so we restore the gains it expects. Coulomb / viscous friction in
        # the actuator still acts on top.
        self._restore_pd_gains_from_policy_cfg(env, cfg)

        self.reset()

    def _restore_pd_gains_from_policy_cfg(self, env, cfg) -> None:
        kp_by_name = cfg["control"]["stiffness"]
        kd_by_name = cfg["control"]["damping"]

        if not self._closed_kinematics:
            # The policy config gains are MOTOR-space (closed model: two
            # ankle motors per foot acting through the crank linkage). On
            # the open model the same-named dofs are JOINT-space (u_crank =
            # pitch, b_crank = roll), so the gains must be reflected through
            # the linkage: K_joint = J2M^T*K_motor*J2M = diag(2u^2, 2b^2)*kp.
            # The open model's ankle armature values follow exactly this
            # transform (0.0795 = 2*0.957^2*0.0434), confirming the
            # convention. Without it the open ankles are ~45% too soft and
            # the robot drifts forward at zero velocity command.
            u = RLControllerZestAP3Open._ANKLE_U_RATIO
            b = RLControllerZestAP3Open._ANKLE_B_RATIO
            ankle_gain_scale = {}
            for side in ("left", "right"):
                ankle_gain_scale[f"{side}_ankle_u_crank_joint"] = 2.0 * u * u
                ankle_gain_scale[f"{side}_ankle_b_crank_joint"] = 2.0 * b * b
            kp_by_name = {
                n: v * ankle_gain_scale.get(n, 1.0) for n, v in kp_by_name.items()
            }
            kd_by_name = {
                n: v * ankle_gain_scale.get(n, 1.0) for n, v in kd_by_name.items()
            }

        robot = env.unwrapped.scene["robot"]
        overridden: List[str] = []
        for actuator in robot.actuators.values():
            for local_idx, j_name in enumerate(actuator.joint_names):
                if j_name not in kp_by_name:
                    continue
                actuator.stiffness[:, local_idx] = float(kp_by_name[j_name])
                actuator.damping[:, local_idx] = float(kd_by_name[j_name])
                overridden.append(j_name)

        missing = [n for n in self.policy_joint_order if n not in overridden]
        if missing:
            print(
                "[Network] WARNING: could not restore PD gains for joints not "
                f"found in any actuator: {missing}"
            )
        if self._debug:
            print(
                f"[Network] restored PD gains on {len(overridden)} joints "
                f"from policy config (force_control was zeroing them)."
            )

    def reset(self) -> None:
        self.controller.reset()
        self._first_call = True
        self._dump_target_once = True
        self._call_count = 0

    # Named velocity-command presets. Values are in policy units
    # (lin_vel_x / lin_vel_y in m/s, ang_vel_z in rad/s). These are
    # clipped automatically by the policy's own command ranges.
    _DIRECTION_PRESETS: Dict[str, Dict[str, float]] = {
        "stop":         {"lin_vel_x": 0.0, "lin_vel_y": 0.0, "ang_vel_z": 0.0},
        "forward":      {"lin_vel_x": 0.5, "lin_vel_y": 0.0, "ang_vel_z": 0.0},
        "backward":     {"lin_vel_x": -0.5, "lin_vel_y": 0.0, "ang_vel_z": 0.0},
        "left":         {"lin_vel_x": 0.0, "lin_vel_y": 0.5, "ang_vel_z": 0.0},
        "right":        {"lin_vel_x": 0.0, "lin_vel_y": -0.5, "ang_vel_z": 0.0},
        "rotate_left":  {"lin_vel_x": 0.0, "lin_vel_y": 0.0, "ang_vel_z": 0.8},
        "rotate_right": {"lin_vel_x": 0.0, "lin_vel_y": 0.0, "ang_vel_z": -0.8},
    }

    def set_command(self, direction: str) -> None:
        """Tell the policy which way to walk.

        Args:
            direction: one of ``stop``, ``forward``, ``backward``, ``left``,
                ``right``, ``rotate_left``, ``rotate_right``.
        """
        if direction not in self._DIRECTION_PRESETS:
            raise ValueError(
                f"Unknown direction: {direction!r}. "
                f"Expected one of {list(self._DIRECTION_PRESETS)}."
            )

        cmds = self._DIRECTION_PRESETS[direction] # {'lin_vel_x': -0.5, 'lin_vel_y': 0.0, 'ang_vel_z': 0.0}
        # Only forward commands the active policy actually advertises.
        known = set(self.controller.commander.cmd_names) # {'lin_vel_x', 'lin_vel_y', 'height_offset', 'ang_vel_z'}
        cmds = {k: v for k, v in cmds.items() if k in known} # {'lin_vel_x': -0.5, 'lin_vel_y': 0.0, 'ang_vel_z': 0.0}
        self.controller.set_cmds(cmds)
        if self._debug:
            print(f"[Network] command set to '{direction}': {cmds}")
    
    def set_cmd(self, cmds: Dict[str, List[float]]) -> None:
        self.controller.set_cmds(cmds)
        if self._debug:
            print(f"[Network] command set to direction: {cmds}")

    def set_command_detailed(self, lin_vel_x: float, lin_vel_y: float, ang_vel_z: float) -> None:
        cmds = {'lin_vel_x': lin_vel_x, 'lin_vel_y': lin_vel_y, 'ang_vel_z': ang_vel_z}
        self.controller.set_cmds(cmds)
        if self._debug:
            print(f"[Network] command set to direction: {cmds}")

    def set_velocity_command(
        self,
        lin_vel_x: float = 0.0,
        lin_vel_y: float = 0.0,
        ang_vel_z: float = 0.0,
    ) -> None:
        """Set arbitrary velocity command (m/s, m/s, rad/s)."""
        cmds = {
            "lin_vel_x": float(lin_vel_x),
            "lin_vel_y": float(lin_vel_y),
            "ang_vel_z": float(ang_vel_z),
        }
        known = set(self.controller.commander.cmd_names)
        cmds = {k: v for k, v in cmds.items() if k in known}
        self.controller.set_cmds(cmds)

    def set_joint_overrides(self, overrides: Dict[str, List[float]]) -> None:
        """Push joint-position overrides into the commander as ref targets.

        The ZEST controller reads full-body reference from the commander;
        joint names absent from ``commander.cmd_names`` are silently filtered
        out.

        Args:
            overrides: dict possibly containing ``legs_joint_pos`` (12),
                ``torso_joint_pos`` (13: 1 torso_yaw + 5+5 arms + 2 head),
                ``wrist_joint_pos`` (4, not part of policy commander),
                ``finger_joint_pos`` (12, not part of policy commander).
                Wrist/finger entries are ignored here.
        """
        known = set(self.controller.commander.cmd_names)
        cmds: Dict[str, float] = {}

        legs = overrides.get("legs_joint_pos")
        if legs is not None:
            # First 12 entries of model.joint_order are legs in the correct order.
            for name, val in zip(self.model_joint_order[:12], legs[:12]):
                if name in known:
                    cmds[name] = float(val)

        torso = overrides.get("torso_joint_pos")
        if torso is not None:
            # Next 13 entries: torso_yaw + 10 arms + 2 head.
            for name, val in zip(self.model_joint_order[12:25], torso[:13]):
                if name in known:
                    cmds[name] = float(val)
        
        velocity = overrides.get("velocity")
        if velocity is not None:
            vel_keys = ["lin_vel_x", "lin_vel_y", "lin_vel_z",
                        "ang_vel_x", "ang_vel_y", "ang_vel_z"]
            for name, val in zip(vel_keys, velocity[:6]):
                if name in known:
                    cmds[name] = float(val)

        base_cmd = overrides.get("base_command")
        if base_cmd is not None:
            for name in ("root_height", "roll", "pitch", "yaw"):
                if name in known and name in base_cmd:
                    cmds[name] = float(base_cmd[name])

        if cmds:
            self.controller.set_cmds(cmds)
            if self._debug:
                print(f"[Network] joint overrides -> commander:{cmds}")

    def obs(self) -> Dict[str, List[float]]:
        robot = self.env.unwrapped.scene["robot"]
        data = robot.data

        joint_pos_sim = (
            data.joint_pos[0, self._sim_indices].detach().cpu().numpy().astype(np.float32)
        )
        joint_vel_sim = (
            data.joint_vel[0, self._sim_indices].detach().cpu().numpy().astype(np.float32)
        )
        base_ang_vel = data.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float32)
        projected_gravity = (
            data.projected_gravity_b[0].detach().cpu().numpy().astype(np.float32)
        )

        dof_pos_policy = joint_pos_sim[self.model_to_policy_idx]
        dof_vel_policy = joint_vel_sim[self.model_to_policy_idx]

        verbose = self._debug and self._call_count < self._verbose_steps
        if self._debug and self._first_call:
            self._first_call = False
            np.set_printoptions(precision=3, suppress=True, linewidth=200)
            print("[Network] === first-call diagnostics ===")
            print(f"[Network] policy type        : {self.type}")
            print(f"[Network] USD default (model order): {self._default_joint_pos_model_order}")
            print(f"[Network] policy default (policy order): {self._policy_default_pos}")

        if verbose:
            np.set_printoptions(precision=3, suppress=True, linewidth=200)
            dof_pos_dev_policy = dof_pos_policy - self._policy_default_pos
            print(f"[Network] --- step {self._call_count} obs ---")
            print(f"[Network]   base_ang_vel       : {base_ang_vel}")
            print(f"[Network]   projected_gravity  : {projected_gravity}")
            print(f"[Network]   joint_pos - USD_default (model order):\n             {joint_pos_sim - self._default_joint_pos_model_order}")
            print(f"[Network]   dof_pos - policy_default (policy order):\n             {dof_pos_dev_policy}")
            print(f"[Network]   |dev|max={np.abs(dof_pos_dev_policy).max():.3f}")
            print(f"[Network]   joint_vel (model order):\n             {joint_vel_sim}")
            print(f"[Network]   |joint_vel|max={np.abs(joint_vel_sim).max():.3f}")

        obs_dict = {
            "base_ang_vel": torch.as_tensor(base_ang_vel),
            "projected_gravity": torch.as_tensor(projected_gravity),
            "dof_pos": torch.as_tensor(dof_pos_policy),
            "dof_vel": torch.as_tensor(dof_vel_policy),
        }
        return obs_dict   

    def __call__(self, obs_dict: Optional[Dict] = None) -> Dict[str, List[float]]:
        target_pose_policy_order = self.controller(obs_dict)
        if isinstance(target_pose_policy_order, torch.Tensor):
            target_pose_policy_order = target_pose_policy_order.detach().cpu().numpy()
        target_pose_policy_order = np.asarray(target_pose_policy_order, dtype=np.float32)

        target_pose_model_order = target_pose_policy_order[self.policy_to_model_idx]

        # JointPositionActionCfg(use_default_offset=True) re-adds default_joint_pos
        # to the action, so we subtract it here to send absolute targets.
        raw_action_model_order = target_pose_model_order - self._default_joint_pos_model_order

        self._call_count += 1

        return {
            "joint_pos": raw_action_model_order.tolist(),
            "finger_joint_pos": [0.0] * self._n_fingers,
        }

    def _validate_joint_orders(self) -> None:
        missing_in_sim = set(self.policy_joint_order) - set(self.model_joint_order)
        if missing_in_sim:
            raise RuntimeError(
                "Policy expects joints not present in sim model.joint_order: "
                f"{sorted(missing_in_sim)}"
            )
