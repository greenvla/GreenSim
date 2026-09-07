import math
import os
import torch
import numpy as np
from typing import Union, Dict, List
from rl_controller import Observation, Action, CommandGenerator
from rl_controller.math_utils import quat_to_yaw, align_trajectory


class RLController:
    def __init__(
        self,
        config: Dict,
        policy: Union[torch.ScriptModule, torch.nn.Module],
        device: torch.DeviceObjType,
        closed_kinematics: bool = True,
    ):
        """Reinforcement learning locomotion controller
        Args:
            config: controller config
            policy: loaded model instance
            device: model inference device (CPU, CUDA)
            closed_kinematics: True for the original robot model with closed
                leg kinematics (sim ankle dofs are motor/crank angles), False
                for the open-kinematics model (sim ankle dofs are joint angles)
        """
        self.cfg = config
        self.policy = policy
        self.device = device
        self.closed_kinematics = closed_kinematics

        self.joint_order = self.cfg["joint_order"]
        if "active_joint_order" in self.cfg:
            self.active_joint_idx = [
                self.joint_order.index(joint)
                for joint in self.cfg["active_joint_order"]
            ]
        else:
            self.active_joint_idx = list(range(len(self.joint_order)))
        if "obs_joint_order" in self.cfg:
            self.obs_joint_idx = [
                self.joint_order.index(joint) for joint in self.cfg["obs_joint_order"]
            ]
        else:
            self.obs_joint_idx = self.active_joint_idx.copy()
        self.num_actions = len(self.joint_order)

        # joint positions corresponding to zero action
        default_joints_pos = torch.tensor(
            [
                self.cfg["init_state"]["default_joint_angles"][name]
                for name in self.joint_order
            ],
            dtype=torch.float32,
            device=self.device,
        )

        self.commander = CommandGenerator(
            config=self.cfg["commands"],
            num_envs=1,
            device=self.device,
        )
        self.observation_packer = Observation(
            config=self.cfg["observation"],
            device=self.device,
        )
        self.action_postproc = Action(
            config=self.cfg["control"],
            joint_order=self.joint_order,
            default_joints_pos=default_joints_pos,
            device=self.device,
        )

        self.reset()  # set init state

    def __call__(self, obs_dict: Dict[str, Union[torch.Tensor, np.ndarray, List]]):
        """Get policy action
        Args:
            obs_dcit: observation dict as expected in config
        Returns:
            np.ndarray: desired joint states
        """
        joints_pos = None
        joints_vel = None
        for name, obs in obs_dict.items():
            if isinstance(obs, (np.ndarray, list)):
                obs = torch.tensor(obs)
            obs_dict[name] = obs.to(torch.float32).to(self.device)
            if name == "dof_pos":
                joints_pos = obs_dict[name].clone()
            elif name == "dof_vel":
                joints_vel = obs_dict[name].clone()

        observation = self.prepare_observation(obs_dict)

        actions = torch.zeros(self.num_actions, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions[self.active_joint_idx] = self.policy(
                observation.unsqueeze(0)
            ).squeeze()
        self.actions, target_joint_state = self.action_postproc.process_action(
            action=actions,
            joint_pos=joints_pos,
            joint_vel=joints_vel,
        )

        if target_joint_state.is_cuda:
            target_joint_state = target_joint_state.cpu()

        return target_joint_state.numpy()

    def prepare_observation(self, obs_dict: Dict[str, torch.Tensor]):
        obs_dict["actions"] = self.actions.clone()
        if "dof_pos" in obs_dict:  # observe joint pos with offset
            obs_dict["dof_pos"] = (
                obs_dict["dof_pos"] - self.action_postproc.default_joints_pos
            )
        obs_dict["commands"] = (
            self.commander.get_cmd() * self.commander.cmd_scales
        ).squeeze(dim=0)

        # select observed joints
        if "dof_pos" in obs_dict:
            obs_dict["dof_pos"] = obs_dict["dof_pos"][self.obs_joint_idx]
        if "dof_vel" in obs_dict:
            obs_dict["dof_vel"] = obs_dict["dof_vel"][self.obs_joint_idx]
        if "actions" in obs_dict:
            obs_dict["actions"] = obs_dict["actions"][self.active_joint_idx]

        observation = self.observation_packer.prepare_observations(obs_dict)
        return observation

    def set_cmds(self, cmds_dict: Dict[str, float]):
        for name, value in cmds_dict.items():
            self.commander.set_cmd(name, value)

    def reset(self):
        """Reset controller to initial state"""
        self.action_postproc.reset()
        self.observation_packer.reset()
        self.commander.reset()
        self.actions = torch.zeros(
            self.num_actions, dtype=torch.float32, device=self.device
        )


class RLControllerWTW(RLController):
    def prepare_observation(self, obs_dict):
        clock_inputs = torch.remainder(
            obs_dict["clock"] * self.commander.get_cmd("gait_frequency"), 1.0
        )
        clock_inputs = torch.tensor(
            [
                clock_inputs + self.commander.get_cmd("gait_phase"),
                clock_inputs,
            ]
        )
        obs_dict["clock"] = torch.sin(2 * torch.pi * clock_inputs)
        return super().prepare_observation(obs_dict)


class RLControllerUnitree(RLController):
    def prepare_observation(self, obs_dict):
        period = self.cfg["observation"]["phase_period"]
        phase = (obs_dict["clock"] % period) / period
        obs_dict["phase"] = torch.tensor(
            [
                torch.sin(2 * torch.pi * phase),
                torch.cos(2 * torch.pi * phase),
            ],
        )
        return super().prepare_observation(obs_dict)


class RLControllerGait(RLController):
    def gait_phase_clock(self, gait_command: torch.Tensor, episode_time: torch.Tensor):
        """
        Phase clock is a normalized time ([0,1]) of the current gait cycle
        In order to compute it we take the duration parameter(expressed in seconds)
        and take modulo of the current episode time, then normalize it using the same duration parameter.
        We then shift the normalized phase time using the offset parameter for each leg (offsets=[0,1])
        and take the modulo again to stay in the [0,1] range
        """
        per_leg_clock = episode_time.detach().unsqueeze(1).repeat(1, 2)
        duration = 2 * gait_command[:, 0:1]
        offset = gait_command[:, 3:4]
        normalized_clock = torch.remainder(per_leg_clock, duration) / duration
        normalized_clock[:, 1:2] = torch.remainder(offset + normalized_clock[:, 1:2], 1)
        return normalized_clock

    def prepare_observation(self, obs_dict):
        cmd = self.commander.get_cmd(
            ["duration", "duty_factor", "foot_height", "offset"]
        )
        clk = torch.tensor([obs_dict["time"]])
        clock = self.gait_phase_clock(cmd, clk).squeeze(0)
        obs_dict.update({"clock": clock})
        return super().prepare_observation(obs_dict)


class RLControllerFixedGait(RLController):
    def prepare_observation(self, obs_dict):
        tmp_phase = (obs_dict["time"]) / 0.6
        obs_dict.update(
            {
                "clock": torch.cat(
                    (
                        torch.sin(2 * torch.pi * torch.ones(1) * tmp_phase),
                        torch.cos(2 * torch.pi * torch.ones(1) * tmp_phase),
                    ),
                )
            }
        )
        return super().prepare_observation(obs_dict)


class RLControllerHeightScan(RLController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        H, W = self.cfg["height_scan"]["in_dims"]
        h, w = self.cfg["height_scan"]["out_dims"]
        dX, dY = self.cfg["height_scan"]["in_res"]
        dx, dy = self.cfg["height_scan"]["out_res"]

        X = (H - 1) * dX
        Y = (W - 1) * dY
        x = (h - 1) * dx
        y = (w - 1) * dy
        o_x = (X - x) / 2
        o_y = (Y - y) / 2
        o_i = round(o_x / dX)
        o_j = round(o_y / dY)

        self.hs_crop_ids = [o_i, o_j]

    def prepare_observation(self, obs_dict):
        hs = obs_dict["height_scan"].reshape(*self.cfg["height_scan"]["in_dims"]).T
        hs = hs[
            self.hs_crop_ids[0] : self.hs_crop_ids[0]
            + self.cfg["height_scan"]["out_dims"][0],
            self.hs_crop_ids[1] : self.hs_crop_ids[1]
            + self.cfg["height_scan"]["out_dims"][1],
        ]  # crop desired size at center
        hs *= self.cfg["height_scan"]["scale"]
        hs += self.cfg["height_scan"]["offset"]
        hs = hs.flip(
            dims=self.cfg["height_scan"]["flip_dims"]
        )  # reindex as expected in observation
        obs_dict["height_scan"] = hs.reshape(-1)
        return super().prepare_observation(obs_dict)


class RLControllerZestAP3Open(RLController):
    """ZEST WBC controller (open kinematics) with commander-based reference.

    Residual policy: q_target = ref + action_postproc(policy(obs)).
    Ref is built from commander on every tick (no .npz playback).

    Differences vs base RLController:
      - obs_default_joint_angles vs act_default_joint_angles are distinct in
        ZEST configs; the base class uses a single tensor, so we override
        prepare_observation to apply the *observation* default.
      - with ``closed_kinematics=True`` (original robot model) the sim ankle
        dofs are motor/crank angles: dof_pos / dof_vel and reference commands
        are mapped to joint space via M2J, and the output target is mapped
        back via J2M. With ``closed_kinematics=False`` (open-kinematics robot
        model) everything is already in joint space and no transform is applied.
      - Policy inference dispatches between TorchScript (JIT) and ONNX based
        on the loaded model type (mirrors upstream ``RLController.inference_policy``).
    """

    _ANKLE_U_RATIO = 0.957
    _ANKLE_B_RATIO = 0.826

    def __init__(self, config, policy, device, closed_kinematics=True):
        # ZEST configs ship split defaults (obs_default + act_default) and have
        # no top-level "default_joint_angles". The base RLController.__init__
        # expects exactly that key for its action-postproc offset. Synthesize
        # it from act_default (zeros for ZEST), since the ref is added inside
        # this class on top of the scaled residual.
        init_state = config["init_state"]
        if "default_joint_angles" not in init_state:
            src = init_state.get(
                "act_default_joint_angles",
                init_state.get("obs_default_joint_angles", {}),
            )
            init_state["default_joint_angles"] = dict(src)

        super().__init__(
            config=config,
            policy=policy,
            device=device,
            closed_kinematics=closed_kinematics,
        )

        # Separate observation-space default joint angles from action-space
        # ones. ZEST configs ship both: obs_default == standing pose, act_default == 0.
        obs_default_dict = self.cfg["init_state"].get(
            "obs_default_joint_angles",
            self.cfg["init_state"].get("default_joint_angles", {}),
        )
        self._obs_default_joints_pos = torch.tensor(
            [obs_default_dict.get(name, 0.0) for name in self.joint_order],
            dtype=torch.float32,
            device=self.device,
        )

        # Closed-chain ankle motor<->joint transforms (same coefficients as
        # the C++ controller in sberchelx_meta). Only needed for the original
        # robot model with closed leg kinematics.
        if self.closed_kinematics:
            laucj_id = self.joint_order.index("left_ankle_u_crank_joint")
            labcj_id = self.joint_order.index("left_ankle_b_crank_joint")
            raucj_id = self.joint_order.index("right_ankle_u_crank_joint")
            rabcj_id = self.joint_order.index("right_ankle_b_crank_joint")
            u = self._ANKLE_U_RATIO
            b = self._ANKLE_B_RATIO
            J2M = torch.eye(self.num_actions, dtype=torch.float32, device=self.device)
            J2M[laucj_id, [laucj_id, labcj_id]] = torch.tensor(
                [u, -b], dtype=torch.float32, device=self.device
            )
            J2M[labcj_id, [laucj_id, labcj_id]] = torch.tensor(
                [u, b], dtype=torch.float32, device=self.device
            )
            J2M[raucj_id, [raucj_id, rabcj_id]] = torch.tensor(
                [u, b], dtype=torch.float32, device=self.device
            )
            J2M[rabcj_id, [raucj_id, rabcj_id]] = torch.tensor(
                [u, -b], dtype=torch.float32, device=self.device
            )
            self.J2M = J2M
            self.M2J = torch.linalg.inv(J2M)
        if not self.closed_kinematics:
            # Open-kinematics robots still need M2J for the commander ref:
            # the policy was trained with M2J-transformed refs (see
            # _get_ref_joint_pos), so the same matrices are built here.
            laucj_id = self.joint_order.index("left_ankle_u_crank_joint")
            labcj_id = self.joint_order.index("left_ankle_b_crank_joint")
            raucj_id = self.joint_order.index("right_ankle_u_crank_joint")
            rabcj_id = self.joint_order.index("right_ankle_b_crank_joint")
            u = self._ANKLE_U_RATIO
            b = self._ANKLE_B_RATIO
            J2M = torch.eye(self.num_actions, dtype=torch.float32, device=self.device)
            J2M[laucj_id, [laucj_id, labcj_id]] = torch.tensor(
                [u, -b], dtype=torch.float32, device=self.device
            )
            J2M[labcj_id, [laucj_id, labcj_id]] = torch.tensor(
                [u, b], dtype=torch.float32, device=self.device
            )
            J2M[raucj_id, [raucj_id, rabcj_id]] = torch.tensor(
                [u, b], dtype=torch.float32, device=self.device
            )
            J2M[rabcj_id, [raucj_id, rabcj_id]] = torch.tensor(
                [u, -b], dtype=torch.float32, device=self.device
            )
            self.J2M = J2M
            self.M2J = torch.linalg.inv(J2M)

        # Policy format (ONNX vs JIT). ONNX policies expose .run().
        self._policy_format = "onnx" if hasattr(policy, "run") else "jit"
        if self._policy_format == "onnx":
            self._onnx_input_name = policy.get_inputs()[0].name
        else:
            self._onnx_input_name = None

    # ------------------------------------------------------------------
    # Reference building (commander-based; no .npz playback supported here)
    # ------------------------------------------------------------------

    def _get_ref_joint_pos(self) -> torch.Tensor:
        """Commander-driven 25-DoF reference, in joint space."""
        cmd = self.commander.get_cmd(self.joint_order)
        if cmd.dim() > 1:
            cmd = cmd.squeeze(0)
        cmd = cmd.to(torch.float32).to(self.device)
        if self.closed_kinematics:
            cmd = self.M2J @ cmd
        if not self.closed_kinematics:
            # Apply the same M2J as the closed model. On the closed model the
            # final J2M cancels it, so the executed ankle ref equals the
            # commander values interpreted as motor angles — the pipeline the
            # policy was trained with (it stands by leaning against the
            # resulting ankle bias). The open model must feed the policy the
            # same ref obs and execute the same physical ankle pose,
            # otherwise the learned lean makes the robot drift forward at
            # zero velocity command.
            cmd = self.M2J @ cmd
        return cmd

    # robot lean fix start
    # On by default: LEAN_FIX=0 opts out, which is what a control arm needs and nothing else.
    _LEAN_FIX = os.environ.get("LEAN_FIX", "1") not in ("0", "false", "False")
    _LEAN_FIX_LOG = os.environ.get("LEAN_FIX_LOG", "") not in ("", "0", "false", "False")
    _lean_tick = 0

    def _lean_corrected_pitch(self, pitch_cmd: float, obs_dict) -> float:
        """Pull the commanded pitch forward by however far the robot actually leans back.

        The reference gravity is built from sin(pitch_cmd) while the resulting physical
        pitch is atan2(-g_x, hypot(g_y, g_z)) -- the two have opposite sign, so commanding
        pitch alone never corrects a backward lean. Feed the measured lean back in.
        """
        if not self._LEAN_FIX:
            return pitch_cmd
        g = obs_dict.get("projected_gravity")
        if g is None:
            return pitch_cmd
        gx, gy, gz = (float(g.flatten()[0]), float(g.flatten()[1]), float(g.flatten()[2]))
        p_phys = math.atan2(-gx, math.hypot(gy, gz))
        u = pitch_cmd + 0.08 + 1.5 * max(p_phys, 0.0)
        u = min(max(u, 0.0), 0.30)
        self._lean_tick += 1
        if self._LEAN_FIX_LOG and self._lean_tick % 50 == 1:
            print("[LEAN_FIX] pitch_cmd=%+.4f p_phys=%+.4f -> %+.4f rad"
                  % (pitch_cmd, p_phys, u), flush=True)
        return u
    # robot lean fix end

    @staticmethod
    def _projected_gravity_from_rp(roll: float, pitch: float) -> torch.Tensor:
        # Mirror C++ closed form (see rl_controller_zest_ap3_open.cpp:215-222).
        # Yaw is intentionally ignored.
        return torch.tensor(
            [
                math.sin(pitch),
                -math.sin(roll) * math.cos(pitch),
                -math.cos(roll) * math.cos(pitch),
            ],
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Reset (applies commander defaults that the green_challenge commander
    # doesn't restore on its own — root_height=0.9, standing-pose joints, etc.)
    # ------------------------------------------------------------------

    def reset(self):
        super().reset()
        defaults = self.cfg.get("commands", {}).get("defaults", {})
        for name, val in defaults.items():
            if name in self.commander.cmd_names:
                self.commander.set_cmd(name, float(val))

    # ------------------------------------------------------------------
    # Inference dispatch + main __call__
    # ------------------------------------------------------------------

    def _inference_policy(self, observation: torch.Tensor) -> torch.Tensor:
        if self._policy_format == "jit":
            with torch.no_grad():
                action = self.policy(observation)
        else:
            obs_np = observation.detach().cpu().numpy()
            action = self.policy.run(None, {self._onnx_input_name: obs_np})[0]
            action = torch.from_numpy(action).to(torch.float32).to(self.device)
        return action.squeeze()

    def __call__(self, obs_dict):
        # closed kinematics: sim ankle dofs are motor angles -> map to joint
        # space (must happen before tensor coercion below). Open kinematics
        # reports joint space directly, no transform needed.
        if self.closed_kinematics:
            for name in ("dof_pos", "dof_vel"):
                if name in obs_dict:
                    v = obs_dict[name]
                    if isinstance(v, (np.ndarray, list)):
                        v = torch.tensor(v)
                    obs_dict[name] = self.M2J @ v.to(torch.float32).to(self.device)

        # cache joint-space dof_pos/vel (used by Action.process_action)
        joints_pos = None
        joints_vel = None
        for name, obs in obs_dict.items():
            if isinstance(obs, (np.ndarray, list)):
                obs = torch.tensor(obs)
            obs_dict[name] = obs.to(torch.float32).to(self.device)
            if name == "dof_pos":
                joints_pos = obs_dict[name].clone()
            elif name == "dof_vel":
                joints_vel = obs_dict[name].clone()

        # Snapshot the joint-space ref. prepare_observation will recompute it
        # for the observation_packer; we keep a copy here for the residual add.
        ref_dof_pos = self._get_ref_joint_pos()

        observation = self.prepare_observation(obs_dict)

        actions = torch.zeros(self.num_actions, dtype=torch.float32, device=self.device)
        actions[self.active_joint_idx] = self._inference_policy(
            observation.unsqueeze(0)
        )
        self.actions, target_joint_state = self.action_postproc.process_action(
            action=actions,
            joint_pos=joints_pos,
            joint_vel=joints_vel,
        )

        # residual: q_cmd = ref + scaled_residual (joint space). Closed
        # kinematics maps the target back to motor space; open kinematics
        # takes joint-space targets directly.
        target_joint_state = target_joint_state + ref_dof_pos
        if self.closed_kinematics:
            target_joint_state = self.J2M @ target_joint_state

        if target_joint_state.is_cuda:
            target_joint_state = target_joint_state.cpu()
        return target_joint_state.numpy()

    # ------------------------------------------------------------------
    # Observation packing (commander-based ref obs)
    # ------------------------------------------------------------------

    def prepare_observation(self, obs_dict: Dict[str, torch.Tensor]):
        obs_dict["actions"] = self.actions.clone()

        # observation offset uses obs_default (standing pose), NOT act_default (zeros).
        if "dof_pos" in obs_dict:
            obs_dict["dof_pos"] = obs_dict["dof_pos"] - self._obs_default_joints_pos

        # Reference observations from commander.
        obs_dict["ref_dof_pos"] = self._get_ref_joint_pos()
        obs_dict["ref_base_position_z"] = (
            self.commander.get_cmd(["root_height"]).squeeze(0)
            .to(torch.float32).to(self.device)
        )
        obs_dict["ref_base_lin_vel"] = (
            self.commander.get_cmd(["lin_vel_x", "lin_vel_y", "lin_vel_z"]).squeeze(0)
            .to(torch.float32).to(self.device)
        )
        obs_dict["ref_base_ang_vel"] = (
            self.commander.get_cmd(["ang_vel_x", "ang_vel_y", "ang_vel_z"]).squeeze(0)
            .to(torch.float32).to(self.device)
        )
        roll = float(self.commander.get_cmd("roll").flatten()[0])
        pitch = float(self.commander.get_cmd("pitch").flatten()[0])
        obs_dict["ref_projected_gravity"] = (
            self._projected_gravity_from_rp(roll, pitch).to(self.device)
        )

        # Commands vector (kept for parity with base; harmless if unused).
        obs_dict["commands"] = (
            self.commander.get_cmd() * self.commander.cmd_scales
        ).squeeze(dim=0)

        # select observed joints
        if "dof_pos" in obs_dict:
            obs_dict["dof_pos"] = obs_dict["dof_pos"][self.obs_joint_idx]
        if "dof_vel" in obs_dict:
            obs_dict["dof_vel"] = obs_dict["dof_vel"][self.obs_joint_idx]
        if "actions" in obs_dict:
            obs_dict["actions"] = obs_dict["actions"][self.active_joint_idx]

        return self.observation_packer.prepare_observations(obs_dict)
