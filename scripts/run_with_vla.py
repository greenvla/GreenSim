"""VLA + ZEST pipeline: VLA produces full-body reference targets,
ZEST balance-policy adds residual corrections."""

import argparse
import json
import logging
import numpy as np
import torch
from isaaclab.app import AppLauncher
import os
import math
import sys
import time

logger = logging.getLogger("run")
logger.addHandler(logging.StreamHandler())
logger.propagate = False
logger.setLevel(logging.INFO)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.dirname(REPO_ROOT))

# atomatic parser start
from green_challenge.scripts.args_json_parser import (
    build_parser_from_json,
    apply_scenario_args,
)

parser = build_parser_from_json("configs/cli_args.json")
# VLA-specific arguments (not in cli_args.json)
parser.add_argument("--policy_host", type=str, default="127.0.0.1")
parser.add_argument(
    "--policy_port", type=int, default=int(os.environ.get("POLICY_PORT", "8999"))
)
parser.add_argument("--use_policy", action="store_true", default=False)
parser.add_argument(
    "--record_video",
    action="store_true",
    default=False,
    help="Record viewport video (lighter than camera sensors)",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default="/tmp/sim_videos",
    help="Output directory for recorded videos",
)
parser.add_argument(
    "--video_fps", type=int, default=30, help="Target frame rate for recorded video"
)
parser.add_argument(
    "--max_time",
    type=float,
    default=0.0,
    help="Max simulation time in seconds (0 = unlimited)",
)
parser.add_argument(
    "--control_mode",
    type=str,
    default="none",
    choices=["keyboard", "vla", "none"],
    help="Joint control source: keyboard (manual), vla (policy server), none (ZEST only)",
)
parser.add_argument(
    "--action_horizon",
    type=int,
    default=int(os.environ.get("ACTION_HORIZON", "5")),
    help="Number of VLA action steps to execute before next inference (1, 2, 5, 10, 50)",
)
parser.add_argument(
    "--policy_wait_timeout",
    type=float,
    default=300.0,
    help="Seconds to wait for the policy server before exiting (vla mode)",
)
parser.add_argument(
    "--benchmark_output_dir",
    type=str,
    default="./benchmark_logs",
    help="Directory for benchmark episode reports (one subdir per scenario)",
)
parser.add_argument(
    "--benchmark_total_episodes",
    type=int,
    default=0,
    help="Total episodes across the whole benchmark (for progress logging)",
)
parser.set_defaults(enable_cameras=True)
parser.set_defaults(headless=True)
parser.set_defaults(rendering_mode="balanced")
args = parser.parse_args()
apply_scenario_args(args, "assets/scenarios.json")
print("Resolved args:", vars(args))
args.kit_args = "--/physics/asyncSimulation=false"
if args.deactivate_robot:
    args.deactivate_cameras = True
    args.benchmark_task = None
os.environ["scene"] = os.path.join(
    REPO_ROOT, "assets", "scenes", args.scene + ".json"
)
os.environ["deactivate_cameras"] = str(int(args.deactivate_cameras))
os.environ["deactivate_room"] = str(int(args.deactivate_room))
os.environ["frame_mode"] = args.frame_mode
os.environ["deactivate_gui"] = str(int(args.deactivate_gui))
os.environ["deactivate_robot"] = str(int(args.deactivate_robot))
os.environ["deactivate_torque"] = str(int(args.deactivate_torque))
os.environ["robot"] = args.robot
os.environ["env_simdt_denominator"] = str(int(args.env_simdt_denominator))
os.environ["env_decimation"] = str(int(args.env_decimation))
os.environ["env_render_interval"] = str(int(args.env_render_interval))
os.environ["enable_third_person_camera"] = str(int(args.record_video))

# Synchronize render rate with video recording target fps.
if args.record_video:
    _render_int = max(1, int(args.env_simdt_denominator / args.video_fps))
    print("render = ", _render_int)
    os.environ["env_render_interval"] = str(_render_int)

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import isaaclab.utils.assets as assets

assets.is_usd_path_available = lambda usd_path, timeout=1: True

from green_challenge.envs.env_registration import register_envs, make_env
from green_challenge.policy.network import Network
from green_challenge.policy.pi_policy import PiPolicy
from green_challenge.scripts.keyboard_control import KeyboardControl, _KEY_MAP, _STEP

# Activate sber control panel
import omni

EXTS_DIR = os.path.join(REPO_ROOT, "exts")
CONTROL_PANEL_EXT_DIR = os.path.join(EXTS_DIR, "control_panel")
ext_mgr = omni.kit.app.get_app().get_extension_manager()
ext_mgr.add_path(EXTS_DIR)
sys.path.insert(0, CONTROL_PANEL_EXT_DIR)
ext_mgr.set_extension_enabled_immediate("control_panel", True)
from control_panel.extension import SberRoboticsControlExtension

DEACTIVATE_GUI = bool(int(os.environ.get("deactivate_gui", "0")))

# Start poses, held for the first ticks of an episode in VLA mode (the `holding` branch
# in main()).
#
#   zest_manipulate — the home pose policy v1 was trained from: elbows bent ~1.8 rad,
#       hands over the table (the policy never sees the arms hanging down).
#   canonical        — the robot's canonical joint pose from its asset config
#       (robots/<robot>/dependencies/cfgs/*.interface.json -> init_state.joint_pos):
#       arms hang straight down, fingers open with the thumb yaw at its -0.1 rest angle.
#
# The 13 torso numbers are in Network.model_joint_order[12:25] order:
#   torso_yaw, L shoulder pitch/roll/yaw, L elbow pitch/yaw,
#   R shoulder pitch/roll/yaw, R elbow pitch/yaw, neck_yaw, neck_pitch
# fingers are the 12 simulator finger dofs (6 left, 6 right: pinky, ring, middle, index,
# thumb_pitch, thumb_yaw); wrists are the 4 dofs of the wrist action group, in its
# order: [L crank, L roll, R crank, R roll].
_START_POSES = {
    "zest_manipulate": {
        "torso": [
            +0.03795719,
            -0.07438755,
            +0.22735882,
            -0.42763424,
            -1.85206413,
            +0.22278214,
            -0.03547764,
            -0.41084862,
            +0.36392784,
            -1.81277180,
            -0.16975689,
            -0.10528755,
            +0.30518007,
        ],
        "fingers": [
            0.17627118,
            0.34539050,
            0.24500000,
            0.34866628,
            0.00375171,
            0.83728814,
            0.55963355,
            0.41705608,
            0.37423888,
            0.36302084,
            0.01610692,
            0.58220339,
        ],
        "wrists": [0.18912393, 0.00753975, 0.35876989, 0.02576113],
    },
    "canonical": {
        "torso": [0.0] * 13,
        "fingers": [0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1],
        "wrists": [0.0] * 4,
    },
}
START_POSE_NAME = os.environ.get("START_POSE", "").strip()
if START_POSE_NAME and START_POSE_NAME not in _START_POSES:
    raise ValueError(
        "Unknown START_POSE %r. Available: %s"
        % (START_POSE_NAME, ", ".join(sorted(_START_POSES)))
    )
START_POSE = _START_POSES.get(START_POSE_NAME)
# How long the start pose is held, in SECONDS of model time. It used to be a count of
# control ticks, which quietly made it five times shorter when the control rate went from
# 50 Hz to 250 Hz: 0.2 s instead of 1.0 s, not enough for the arms to travel from hanging
# down to the trained home pose. What matters is how long the robot has to get there, and
# that is a duration, so it is written as one. Ticks are derived once dt is known.
START_POSE_SEC = float(os.environ.get("START_POSE_SEC", "1.0"))

# PROPRIO_CMD=all: send the policy its own last command back as proprioception instead
# of the measurement. That is what the checkpoint was trained on — with a positional PD
# the arm always lags its setpoint, and the policy has never seen that lag.
PROPRIO_CMD = os.environ.get("PROPRIO_CMD", "")
# Idle leg pose from the reference client (HMND_S0S1_IDLE_LEG_JOINTS_ZEST): while
# manipulating, the reference feeds the policy exactly this, not the balancing angles.
_IDLE_LEGS_ZEST = [
    +0.01945496,
    +0.01602268,
    -0.01678562,
    +0.17776775,
    -0.12741280,
    -0.15030193,
    +0.08735752,
    -0.01411533,
    -0.09308052,
    +0.15144539,
    -0.12970161,
    -0.18272686,
]

# ARM_FF: static feed-forward against the right-arm sag. A positional PD holds the arm
# with a constant error from its own weight, so the elbow ends up ~0.12 rad less bent
# than commanded and the hand hangs below where the policy aimed. Measured on this
# stand as (setpoint - fact) in rad; adding it back to the command cancels the droop.
# Indices are into torso_joint_pos: 6..10 are the right shoulder pitch/roll/yaw and
# elbow pitch/yaw, everything else is left alone.
ARM_FF = float(os.environ.get("ARM_FF", "0") or 0)
_ARM_FF_OFFSETS = [0.0] * 6 + [-0.107, -0.070, -0.106, -0.120, +0.008] + [0.0, 0.0]


def _print_key_reference():
    """Print keyboard control reference."""
    print("=" * 60)
    print("KEYBOARD JOINT CONTROL  (step=%.3f rad, ESC=quit)" % _STEP)
    print("-" * 60)
    groups: dict = {}
    for key, (group, idx, desc) in sorted(_KEY_MAP.items()):
        groups.setdefault(group, []).append(f"  {key:4s} → {desc}")
    for group in ("torso", "finger", "wrist"):
        if group in groups:
            print(f"[{group}]")
            for line in groups[group]:
                print(line)
    print("=" * 60)




def _build_obs_for_policy(
    robot, leg_idx, torso_idx, finger_idx, reset_flag=False, last_cmd=None
) -> dict:
    """Build the observation dict sent to the VLA policy server."""
    all_pos = robot.data.joint_pos[0].cpu().numpy()
    legs = all_pos[leg_idx]
    torso = all_pos[torso_idx]
    fingers = all_pos[finger_idx]
    root_height = robot.data.root_pos_w[0, 2].cpu().item()

    # PROPRIO_CMD=all — echo the last command back instead of the measurement, which is
    # the proprioception the checkpoint was trained on. Anything else keeps the measured
    # state, i.e. exactly the behaviour this file had before.
    if PROPRIO_CMD == "all" and last_cmd is not None:
        if last_cmd.get("torso") is not None and len(last_cmd["torso"]) == 13:
            torso = np.asarray(last_cmd["torso"], dtype=np.float64)
        cmd_f, cmd_w = last_cmd.get("fingers"), last_cmd.get("wrists")
        if (
            cmd_f is not None
            and cmd_w is not None
            and len(cmd_f) == 12
            and len(cmd_w) == 4
        ):
            # finger_idx interleaves the wrists: 6 left fingers, 2 left wrist dofs,
            # 6 right fingers, 2 right wrist dofs.
            fingers = np.array(
                list(cmd_f[0:6])
                + [cmd_w[0], cmd_w[1]]
                + list(cmd_f[6:12])
                + [cmd_w[2], cmd_w[3]],
                dtype=np.float64,
            )
        legs = np.asarray(_IDLE_LEGS_ZEST, dtype=np.float64)
        # projected_gravity deliberately stays measured: the server reads the attitude
        # from root_quat and only falls back to projected_gravity when root_quat is
        # absent, which it never is here (policy_adapter._roll_pitch).
        base_cmd = last_cmd.get("base_command")
        if isinstance(base_cmd, dict) and base_cmd.get("root_height") is not None:
            root_height = float(base_cmd["root_height"])

    observation = {
        "legs_joint_pos": legs,
        "torso_joint_pos": torso,
        "finger_joint_pos": fingers,
        "base_lin_vel": robot.data.root_lin_vel_b[0].cpu().numpy(),
        "base_ang_vel": robot.data.root_ang_vel_b[0].cpu().numpy(),
        "projected_gravity": robot.data.projected_gravity_b[0].cpu().numpy(),
        "root_height": root_height,
        "root_quat": robot.data.root_quat_w[0].cpu().numpy(),
        "reset": reset_flag,
    }
    if observation.get("reset"):
        observation["policy_seed"] = int(os.environ.get("POLICY_SEED", "42"))
    return observation


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    register_envs()
    # The benchmark subprocess is launched with --benchmark_task but no --scenario,
    # so derive the scenario key (structure.json uses the task name sans "task.").
    scenario_name = args.scenario
    if scenario_name is None and (args.benchmark_task or "").startswith("task."):
        scenario_name = args.benchmark_task[len("task.") :]
    env = make_env(
        args.task,
        num_envs=args.num_envs,
        benchmark_task=args.benchmark_task,
        benchmark_output_dir=args.benchmark_output_dir,
        num_episodes=args.num_episodes,
        scenario_name=scenario_name,
    )

    model = Network(
        env,
        type=args.controller,
        device="cpu",
        debug=False,
        closed_kinematics=(args.robot == "green"),
    )


    # Task shown to the policy as "prompt". Defaults to the scenario's top-level task
    # (SubtaskRuntime.get_task); the web panel's /instruction endpoint overrides it.
    _task_override = None

    def _current_task():
        if _task_override is not None:
            return _task_override
        sr = env.unwrapped.subtask_runtime
        return sr.get_task(env_id=0) if sr is not None else None

    def _current_subtask():
        sr = env.unwrapped.subtask_runtime
        return sr.get_subtask(env_id=0) if sr is not None else None

    pi_policy = None
    if args.control_mode == "vla":
        pi_policy = PiPolicy(
            host=args.policy_host,
            port=args.policy_port,
            task_provider=_current_task,
            action_horizon=args.action_horizon,
            latest_frame_handler=env.unwrapped.latest_frame_handler,
            subtask_provider=_current_subtask,
        )

    if not DEACTIVATE_GUI:
        if SberRoboticsControlExtension._instance:
            print("[ControlPanel] instance found, setting env")
            SberRoboticsControlExtension._instance.set_env(env.unwrapped)
        else:
            print("[ControlPanel] ERROR: instance is None!")

    obs, info = env.reset()
    model.reset()

    # Build index maps to read actual joint positions from the USD stage.
    robot = env.unwrapped.scene["robot"]
    robot_joint_names = list(robot.data.joint_names)

    # Build obs_for_policy directly from USD joint order (bypassing
    # env observation which has incorrect interleaved leg order).
    _LEG_NAMES = model.model_joint_order[:12]
    _TORSO_NAMES = model.model_joint_order[12:25]
    _FINGER_NAMES = [
        # left hand (8)
        "left_pinky_proximal_joint",
        "left_ring_proximal_joint",
        "left_middle_proximal_joint",
        "left_index_proximal_joint",
        "left_thumb_proximal_pitch_joint",
        "left_thumb_proximal_yaw_joint",
        # crank, not pitch: on green `*_wrist_pitch_joint` is the PASSIVE output of the
        # linkage, while `*_wrist_crank_joint` is the dof the wrist action group drives
        # (envs/manip_utils/mdp/actions_pos.py). Reading pitch here reported a joint the
        # policy never commands, and in the wrong slot order — the server expects
        # (crank, roll).
        "left_wrist_crank_joint",
        "left_wrist_roll_joint",
        # right hand (8)
        "right_pinky_proximal_joint",
        "right_ring_proximal_joint",
        "right_middle_proximal_joint",
        "right_index_proximal_joint",
        "right_thumb_proximal_pitch_joint",
        "right_thumb_proximal_yaw_joint",
        "right_wrist_crank_joint",
        "right_wrist_roll_joint",
    ]

    _leg_indices = [robot_joint_names.index(n) for n in _LEG_NAMES]
    _torso_indices = [robot_joint_names.index(n) for n in _TORSO_NAMES]
    _finger_indices = [robot_joint_names.index(n) for n in _FINGER_NAMES]

    # Keyboard joint controller (keyboard control mode only).
    kbd = KeyboardControl() if args.control_mode == "keyboard" else None
    if kbd is not None:
        kbd.start()
        _print_key_reference()

    # Persistent keyboard joint state (accumulated each frame).
    _kb_torso = [0.0] * 13
    _kb_fingers = [0.0] * 12
    _kb_wrists = [0.0] * 4
    _kb_velocity = [0.0] * 6  # lin_vel xyz, ang_vel xyz

    # Velocity command: gentle forward walk.
    i = 0  # control tick within the current episode
    start_time = time.time()
    # A fresh process is a fresh episode, and the server has to be told: it may still be
    # holding the plans of whatever ran before, whose clock is a hundred seconds ahead of
    # the one starting now. Nothing announces the boundary except this flag.
    _episode_ended = True
    movie_path = ""

    if args.control_mode == "vla":
        # One audit line: which of the switches above actually reached this run.
        logger.info(
            "VLA setup: start_pose=%s/%.2fs proprio=%s arm_ff=%.2f settle_steps=%s "
            "horizon=%d %s",
            START_POSE_NAME or "-",
            START_POSE_SEC,
            PROPRIO_CMD or "-",
            ARM_FF,
            os.environ.get("SETTLE_STEPS", "0"),
            args.action_horizon,
            "abs_hands=%s" % os.environ.get("ABS_HAND_TARGETS", "0"),
        )
    _policy_started = False  # has the policy driven at all in this episode?
    _policy_wait_started = (
        None  # wall-clock time when we first tried to reach the server
    )
    _episode = 0
    _dt_ctrl = float(args.env_decimation) / float(args.env_simdt_denominator)
    _last_task = None
    _last_subtask = None
    _log_every = max(
        1, int(1.0 / _dt_ctrl)
    )  # periodic progress log every ~1s of model time

    # pick fix start
    START_POSE_TICKS = max(1, int(round(START_POSE_SEC / _dt_ctrl)))
    # Which side turns action chunks into commands. The server declares it once, in the
    # handshake, and we ask before the first tick: learning it afterwards would be too
    # late, the first chunk would already have been played by the wrong rules.
    FEED_BY_SERVER = False
    _server_meta = {}
    if pi_policy is not None:
        _server_meta = pi_policy.connect()
        FEED_BY_SERVER = bool(_server_meta.get("feed_by_server", False))
    if FEED_BY_SERVER:
        # The server schedules its commands for one particular control grid and says
        # which in the handshake. A mismatch is not a setting to reconcile at runtime --
        # it is a schedule that is wrong by a factor, and wrong invisibly: every command
        # still looks perfectly plausible. So compare and refuse.
        _server_dt = float(_server_meta.get("control_dt", 0.02))
        if abs(_dt_ctrl - _server_dt) > 1e-9:
            raise RuntimeError(
                "The policy server schedules commands for a %.4f s control tick, but "
                "this run ticks at %.4f s (env_decimation=%s, denominator=%s). Every "
                "command would land at the wrong moment. Refusing to run — start the "
                "server with --control-dt %.4f, or run at its tick."
                % (
                    _server_dt,
                    _dt_ctrl,
                    args.env_decimation,
                    args.env_simdt_denominator,
                    _dt_ctrl,
                )
            )
    else:
        print(
            "FEED: the policy server returns raw action chunks, so this run plays one "
            "chunk row per control tick. That is a much weaker feed than scheduling "
            "the rows by time and blending overlapping plans -- start the server with "
            "--feed-by-server to have it do that.",
            flush=True,
        )
    # pick fix end

    _policy_ms = None  # last real round trip to the policy server
    # Last command sent to the robot, remembered for PROPRIO_CMD and for holding the
    # hands still while the policy is silent.
    _last_cmd = {"torso": None, "fingers": None, "wrists": None, "base_command": None}

    # Start viewport video recording (lighter than sensor cameras).
    if args.record_video:
        os.makedirs(args.video_dir, exist_ok=True)

        # Save cadence: match renders to target fps.
        _ctrl_steps_per_sec = args.env_simdt_denominator / args.env_decimation
        _record_interval = max(1, round(_ctrl_steps_per_sec / args.video_fps))

        # Try viewport capture first (fastest), fall back to head camera.
        _record_mode = "head_camera"
        try:
            import omni.syntheticdata as _sd
            from omni.kit.viewport_legacy import get_viewport_interface

            _vp = get_viewport_interface().get_viewport_window().get_active_viewport()
            _rp = _vp.get_render_product()
            _record_mode = "viewport"
        except Exception:
            pass

        movie_path = os.path.join(
            args.video_dir, f"{scenario_name}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        _video_writer = None
        try:
            import imageio

            _video_writer = imageio.get_writer(
                movie_path, fps=args.video_fps, codec="libx264", format="FFMPEG"
            )
        except Exception:
            logger.warning("imageio unavailable — no video will be written")
        logger.info("mode=%s target=%dfps -> %s", _record_mode, args.video_fps, movie_path)
    else:
        _record_mode = ""
        _record_interval = 1
        movie_path = ""
        _video_writer = None

    while sim_app.is_running() and not env.unwrapped.episodes.reached_total(
        args.num_episodes
    ):

        # ── Time limit ──
        if args.max_time > 0 and (time.time() - start_time) >= args.max_time:
            logger.info("max_time=%.0fs reached, stopping", args.max_time)
            break

        # ── 1. Read keyboard (keyboard control mode only) ──
        if args.control_mode == "keyboard":
            kb = kbd.get_deltas()
            if kbd.quit_requested:
                break
            for idx, delta in kb["torso"].items():
                _kb_torso[idx] += delta
            for idx, delta in kb["finger"].items():
                _kb_fingers[idx] += delta
            for idx, delta in kb["wrist"].items():
                _kb_wrists[idx] += delta
            for idx, delta in kb["velocity"].items():
                _kb_velocity[idx] += delta
            # Use keyboard state as ZEST reference.
            model.set_joint_overrides(
                {
                    "legs_joint_pos": np.zeros(12, dtype=np.float64),
                    "torso_joint_pos": np.array(_kb_torso, dtype=np.float64),
                    "velocity": np.array(_kb_velocity, dtype=np.float64),
                    "base_command": {
                        "root_height": 0.9,
                        "roll": 0.0,
                        "pitch": 0.0,
                        "yaw": 0.0,
                    },
                }
            )

        # ── 1b. Hold the start pose the policy was trained from ──
        # Held for the first START_POSE_TICKS ticks of an episode so the arms travel from
        # hanging down to the trained home pose before the policy drives.
        holding = (
            START_POSE is not None
            and args.control_mode == "vla"
            and i < START_POSE_TICKS
        )
        if holding:
            model.set_joint_overrides(
                {
                    "torso_joint_pos": np.array(START_POSE["torso"], dtype=np.float64),
                }
            )
            # The hands bypass ZEST, so they are held through the command below.
            _last_cmd["fingers"] = list(START_POSE["fingers"])
            _last_cmd["wrists"] = list(START_POSE["wrists"])

        # ── 2. VLA inference (if control_mode == "vla") ──
        vla_fingers = None
        vla_wrist = None
        if args.control_mode == "vla" and pi_policy is not None and not holding:
            # Give up if the policy server never answers: start the clock on the first
            # inference attempt and exit once --policy_wait_timeout has elapsed.
            if not _policy_started:
                if _policy_wait_started is None:
                    _policy_wait_started = time.time()
                elif time.time() - _policy_wait_started > args.policy_wait_timeout:
                    logger.error(
                        "policy server %s:%d not reachable within %.0fs, exiting",
                        args.policy_host,
                        args.policy_port,
                        args.policy_wait_timeout,
                    )
                    break
            obs_for_policy = _build_obs_for_policy(
                robot,
                _leg_indices,
                _torso_indices,
                _finger_indices,
                reset_flag=_episode_ended,
                last_cmd=_last_cmd,
            )
            _infer_started = time.time()
            policy_actions = pi_policy.act(obs_for_policy, i * _dt_ctrl)
            _elapsed_ms = (time.time() - _infer_started) * 1000.0
            if _elapsed_ms >= 1.0:
                # act() only talks to the server when the buffered chunk has run dry;
                # every other tick just pops a row and would report ~0 ms.
                _policy_ms = round(_elapsed_ms, 1)
            if policy_actions is not None:
                # Only now is the payload known to have reached the server, so only now
                # may the "new episode" flag be dropped: act() also returns None while
                # it waits for camera frames or sits out the settle pause.
                _episode_ended = False
                model.set_joint_overrides(
                    {
                        "legs_joint_pos": policy_actions.get("legs_joint_pos"),
                        "torso_joint_pos": policy_actions.get("torso_joint_pos"),
                        "velocity": policy_actions.get("velocity"),
                        "base_command": policy_actions.get("base_command"),
                    }
                )
                vla_fingers = policy_actions.get("finger_joint_pos")
                vla_wrist = policy_actions.get("wrist_joint_pos")
                if vla_fingers is None or vla_wrist is None:
                    # the two go into env.step() as one flat vector, so a half answer is
                    # no answer: hold the previous hand pose instead
                    vla_fingers = vla_wrist = None
                _policy_wait_started = None
                _policy_started = True
                _last_cmd["torso"] = policy_actions.get("torso_joint_pos")
                _last_cmd["base_command"] = policy_actions.get("base_command")
                if vla_fingers is not None:
                    _last_cmd["fingers"] = vla_fingers
                    _last_cmd["wrists"] = vla_wrist

        # ── 3. ZEST balances (always runs) ──
        act = {
            "legs_joint_pos": [0.0] * 12,
            "torso_joint_pos": [0.0] * 13,
            "wrist_joint_pos": [0.0] * 4,
            "finger_joint_pos": [0.0] * 12,
        }
        model_act = model(model.obs())

        i += 1

        act["legs_joint_pos"] = model_act["joint_pos"][:12]
        act["torso_joint_pos"] = model_act["joint_pos"][12:25]

        if args.control_mode == "keyboard":
            act["finger_joint_pos"] = _kb_fingers
            act["wrist_joint_pos"] = _kb_wrists
        elif args.control_mode == "vla":
            if vla_fingers is not None:
                act["finger_joint_pos"] = vla_fingers
                act["wrist_joint_pos"] = vla_wrist
            elif _last_cmd["fingers"] is not None and _last_cmd["wrists"] is not None:
                # start-pose hold, paused, or no answer this tick: keep the last hand
                # pose instead of springing the fingers open to zero
                act["finger_joint_pos"] = _last_cmd["fingers"]
                act["wrist_joint_pos"] = _last_cmd["wrists"]

        # Right-arm gravity compensation, see ARM_FF above.
        if ARM_FF:
            act["torso_joint_pos"] = [
                v + ARM_FF * d for v, d in zip(act["torso_joint_pos"], _ARM_FF_OFFSETS)
            ]

        # ── 4. Step the simulator ──
        flat = (
            act["legs_joint_pos"]
            + act["torso_joint_pos"]
            + act["wrist_joint_pos"]
            + act["finger_joint_pos"]
        )
        act_tensor = (
            torch.as_tensor(
                flat,
                dtype=torch.float32,
                device=env.unwrapped.device,
            )
            .unsqueeze(0)
            .repeat(args.num_envs, 1)
        )

        obs, rew, done, trunc, info = env.step(act_tensor)

        # Progress log: on every task/subtask change and periodically (every
        # _log_every control ticks, ~1s of model time), so the run isn't silent.
        _sr = env.unwrapped.subtask_runtime
        _cur_task = _sr.get_task(env_id=0) if _sr is not None else None
        _cur_subtask = _sr.get_subtask(env_id=0) if _sr is not None else None
        if (_cur_task, _cur_subtask) != (
            _last_task,
            _last_subtask,
        ) or i % _log_every == 0:
            _last_task, _last_subtask = _cur_task, _cur_subtask
            logger.info(
                "scenario=%s episode=%d/%d remaining=%d benchmark_total=%d "
                "step=%d t=%.1fs task=%r subtask=%r",
                scenario_name,
                _episode + 1,
                args.num_episodes,
                max(0, args.num_episodes - (_episode + 1)),
                args.benchmark_total_episodes,
                i,
                i * _dt_ctrl,
                _cur_task,
                _cur_subtask,
            )

        # Track episode end for VLA reset signal.
        if done or trunc:
            _episode_ended = True
            # Put the run back into exactly its just-launched state, so episode N
            # behaves like episode 1. Anything remembered here describes a pose the
            # robot is no longer in, and with PROPRIO_CMD it would be sent to the
            # policy as if it still were.
            _episode += 1
            i = 0
            _policy_started = False
            _last_cmd = {
                "torso": None,
                "fingers": None,
                "wrists": None,
                "base_command": None,
            }
            if pi_policy is not None:
                pi_policy.reset()
            model.reset()

        if _video_writer is not None and i % _record_interval == 0:
            frame = None  # viewport capture below may produce nothing at all
            if _record_mode == "viewport":
                rgb = _sd.get_syntheticdata("rgb", render_product_path=_rp.path)
                if rgb is not None and rgb.size > 0:
                    frame = rgb
            else:
                _tp = env.unwrapped.scene.sensors.get("third_person_camera")
                if _tp is not None and "rgb" in _tp.data.output:
                    _rgb = _tp.data.output["rgb"]
                    frame = _rgb[0].cpu().numpy() if _rgb.ndim == 4 else _rgb.cpu().numpy()
            if frame is not None and frame.size > 0:
                _video_writer.append_data(frame)

    # ── Finalize video ──
    if _video_writer is not None:
        _video_writer.close()
        logger.info("record saved: %s", movie_path)

    # ── Clean shutdown ──
    if args.control_mode == "vla" and pi_policy is not None:
        # Only signal reset if the policy actually drove; otherwise infer() would block
        # another minute trying to reach a server that was never there.
        if _policy_started:
            obs_for_policy = _build_obs_for_policy(
                robot,
                _leg_indices,
                _torso_indices,
                _finger_indices,
                reset_flag=True,
                last_cmd=_last_cmd,
            )
            # Force-send via infer() — act() would return buffered actions instead.
            pi_policy.reset()
            payload = pi_policy.get_payload(obs_for_policy)
            pi_policy.infer(payload)
        pi_policy.close()
    if kbd is not None:
        kbd.stop()
    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
