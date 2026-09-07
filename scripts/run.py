# 1. Standard library imports
import os
import torch
from isaaclab.app import AppLauncher
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.dirname(REPO_ROOT))

# 2. Related third party imports
from green_challenge.scripts.args_json_parser import (
    build_parser_from_json,
    apply_scenario_args,
)

parser = build_parser_from_json("configs/cli_args.json")
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
app_launcher = AppLauncher(args)
sim_app = app_launcher.app

# speed up sim loading
import isaaclab.utils.assets as assets
assets.is_usd_path_available = lambda usd_path, timeout=1: True
print(
    f"Headless mode: {app_launcher._headless}, Enable cameras: {app_launcher._enable_cameras}"
)
print(f"Experience loaded: {app_launcher._sim_experience_file}")

# 3. Local application/library specific imports
import carb
settings = carb.settings.get_settings()
from green_challenge.envs.env_registration import register_envs, make_env
from green_challenge.envs.manip_utils.scenes.scenes import ENABLE_ROBOT
from green_challenge.policy.network import Network
from green_challenge.scripts.pos2torque import positions41_to_torques_plus_positions
from green_challenge.scripts.action_encoder import encode_act_v2

# Activate sber control panel
import omni

EXTS_DIR = os.path.join(REPO_ROOT, "exts")
CONTROL_PANEL_EXT_DIR = os.path.join(EXTS_DIR, "control_panel")
ext_mgr = omni.kit.app.get_app().get_extension_manager()
ext_mgr.add_path(EXTS_DIR)
sys.path.insert(0, CONTROL_PANEL_EXT_DIR)
ext_mgr.set_extension_enabled_immediate("control_panel", True)
from control_panel.extension import SberRoboticsControlExtension

# 4. Constants
DEACTIVATE_GUI = bool(int(os.environ.get("deactivate_gui", "0")))

def predict_locomotion(net, obs, conditioning):
    net.set_command_detailed(
        conditioning["lin_vel_x"], conditioning["lin_vel_y"], conditioning["ang_vel_z"]
    )
    net.set_joint_overrides(
        {
            "torso_joint_pos": conditioning["torso_joint_pos"],
            "wrist_joint_pos": conditioning["wrist_joint_pos"],
            "finger_joint_pos": conditioning["finger_joint_pos"],
        }
    )
    obs_zest = net.obs()
    loco_action = net(obs_zest)
    return loco_action


def main():
    env = make_env(
        args.task,
        num_envs=args.num_envs,
        benchmark_task=args.benchmark_task,
        num_episodes=args.num_episodes,
        scenario_name=args.scenario,
    )
    locomotion_net = (
        Network(
            env,
            type=args.controller,
            device="cpu",
            closed_kinematics=(args.robot == "green"),
        )
        if ENABLE_ROBOT
        else None
    )

    if not DEACTIVATE_GUI:
        if SberRoboticsControlExtension._instance:
            print("[ControlPanel] instance found, setting env")
            SberRoboticsControlExtension._instance.set_env(env.unwrapped)
        else:
            print("[ControlPanel] ERROR: instance is None!")

    obs, info = env.reset()
    while not env.unwrapped.episodes.reached_total(args.num_episodes):
        action = {
            "legs_joint_pos": [0.0] * 12,
            "torso_joint_pos": [0.0] * 13,
            "wrist_joint_pos": [0.0] * 4,
            "finger_joint_pos": [0.0] * 12,
            "lin_vel_x": 0.5,  # go fwd
            "lin_vel_y": 0.0,
            "ang_vel_z": 0.0,
        }
        if ENABLE_ROBOT:
            locomotion_prediction = predict_locomotion(
                locomotion_net, obs, conditioning=action
            )
            action["legs_joint_pos"] = locomotion_prediction["joint_pos"][:12]
            act_tensor = (
                torch.as_tensor(
                    action["legs_joint_pos"]
                    + action["torso_joint_pos"]
                    + action["wrist_joint_pos"]
                    + action["finger_joint_pos"],
                    dtype=torch.float32,
                    device=env.unwrapped.device,
                )
                .unsqueeze(0)
                .repeat(args.num_envs, 1)
            )
        else:
            act_tensor = torch.zeros(
                (args.num_envs, env.unwrapped.action_manager.total_action_dim),
                dtype=torch.float32,
                device=env.unwrapped.device,
            )
        obs, rew, done, trunc, info = env.step(act_tensor)

    env.close()
    sim_app.close()


if __name__ == "__main__":
    register_envs()
    main()
