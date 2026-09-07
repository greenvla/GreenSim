# 1. Standard library imports
from __future__ import annotations
from typing import Optional
import os

# 2. Related third party imports
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg
import torch

# 3. Local application/library specific imports
from .manip_utils.scenes.scenes import DefaultSceneCfg, ENABLE_ROBOT
from green_challenge.envs.manip_utils.scenes.assets.rigid_objects import SceneSpawner
from green_challenge.robots.robot_selector import ConfigManager, ROBOT_FOLDER
from green_challenge.scripts.joint_manager_step import jm_step
from .manip_utils.mdp.events_cfg import ExampleEventsCfg
from .manip_utils.mdp.actions import ExampleActionsCfg
from .manip_utils.mdp.actions_pos import ExampleActionsPosCfg
from .manip_utils.mdp.observations import ExampleObservationsCfg
from .manip_utils.mdp.dummy import DummyActionsCfg, DummyObservationsCfg
from .manip_utils.mdp.terminations import ExampleTerminationsCfg
from .manip_utils.mdp.rewards import NoRewardsCfg
from .episode_service import EpisodeService
from .manip_utils.managers.common.shm_image_handler import ShmRosImagePublisherHandler
from .manip_utils.managers.common.sensor_manager import SensorManager, CompositeHandler
from .manip_utils.managers.common.joint_manager import JointManager
from .manip_utils.managers.benchmark.bench_utils import _create_relation_monitor, handle_termination_event, _create_subtask_runtime
from .manip_utils.managers.common.latest_frame_handler import LatestFrameHandler

# 4. Constants
ENABLE_SPAWNER = not bool(int(os.environ.get("deactivate_room", "0")))
FRAME_MODE = os.environ.get("frame_mode", "shm")  # "shm" | "latest" | "both" | "none"
TRACK_EPISODES = True # ExampleTerminationsCfg depends on it
SHOULD_LOG_FREQ = False
ENABLE_JOINT_MANAGER_TORQUE_CONTROL = not bool(int(os.environ.get("deactivate_torque", "0")))
ENABLE_CONFIG_MANAGER = True # joint manager depends on it
CAMERA_TARGET_WIDTH = int(os.environ.get("camera_target_width", "448"))
CAMERA_TARGET_HEIGHT = int(os.environ.get("camera_target_height", "448"))
_ENABLE_THIRD_PERSON = bool(int(os.environ.get("enable_third_person_camera", "0")))
_CAMS = ['left_head_camera', 'left_wrist_camera', 'right_wrist_camera']
if _ENABLE_THIRD_PERSON:
    _CAMS.append('third_person_camera')
CAMERA_NAMES = _CAMS
DISABLE_ROBOT_SPAWN_RANDOMIZATION = False

# Camera masks (native 960×600) applied before resize in LatestFrameHandler.
import numpy as np

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_CFG_DIR = os.path.join(_PROJECT_DIR, "robots", ROBOT_FOLDER, "dependencies", "settings", "camera_mask")
_head_mask = np.load(os.path.join(_CFG_DIR, "mask_head.npy"))
_arm_l_mask = np.load(os.path.join(_CFG_DIR, "mask_left_arm.npy"))
_arm_r_mask = np.load(os.path.join(_CFG_DIR, "mask_right_arm.npy"))

CAMERA_MASKS = {
    "left_head_camera": _head_mask,
    "right_head_camera": _head_mask,
    "left_wrist_camera": _arm_l_mask,
    "right_wrist_camera": _arm_r_mask,
}


class ManipEnv(ManagerBasedRLEnv):
    cfg: ManipEnvCfg
    def __init__(self, cfg: ManipEnvCfg, scenario_name: str | None, **kwargs):
        self.scenario_name = scenario_name
        self.subtask_runtime = None
        self.success_grace_seconds: float = 1.0
        if ENABLE_SPAWNER:
            json_objects_list_path = os.environ.get("scene")
            cfg.scene_spawner = SceneSpawner(json_objects_list_path)
            cfg.collision_applier = cfg.scene.spawn_assets(cfg.scene_spawner)
        if not ENABLE_ROBOT:
            cfg.actions = DummyActionsCfg()
            cfg.observations = DummyObservationsCfg()
        elif not ENABLE_JOINT_MANAGER_TORQUE_CONTROL:
            cfg.actions = ExampleActionsPosCfg()
        super().__init__(cfg, **kwargs)
        if TRACK_EPISODES:
            self.episodes = EpisodeService(self.num_envs, self.device)
        self.robot: Articulation = self.scene["robot"]
        if ENABLE_CONFIG_MANAGER:
            self.config_manager = ConfigManager.instance()
            cm = self.config_manager
            self.joint_groups = cm.load_settings("joint_groups")
        if FRAME_MODE != "none":
            handlers = []
            if FRAME_MODE in ("shm", "both"):
                handlers.append(ShmRosImagePublisherHandler(
                    camera_names=CAMERA_NAMES, node_name="isaac_cameras",
                    max_fps=60, should_log=SHOULD_LOG_FREQ,
                ))
            if FRAME_MODE in ("latest", "both"):
                self.latest_frame_handler = LatestFrameHandler(
                    target_resolution=(CAMERA_TARGET_WIDTH, CAMERA_TARGET_HEIGHT),
                    exclude_cameras={"third_person_camera"},
                    masks=CAMERA_MASKS,
                )
                handlers.append(self.latest_frame_handler)
            self.frame_handler = handlers[0] if len(handlers) == 1 else CompositeHandler(handlers)
            self.sensor_manager = SensorManager(
                env=self, capture_freq_hz=60, env_index=0,
                camera_names=CAMERA_NAMES, handler=self.frame_handler,
                should_log=SHOULD_LOG_FREQ,
            )
            if FRAME_MODE not in ("shm", "both"):
                # latest or none — no shared memory, so no sensor_manager.start_episode
                pass
        if ENABLE_JOINT_MANAGER_TORQUE_CONTROL and ENABLE_ROBOT:
            self.joint_manager = JointManager(self, self.joint_groups)
        if ENABLE_SPAWNER:
            self.scene_spawner = self.cfg.scene_spawner
            self.scene_spawner.update_group_visuals(self.scene)
            self.scene_spawner.capture_initial_poses(self.scene)
            self.scene_spawner.apply_viewer(self.viewport_camera_controller)
        self.cfg.collision_applier.apply_fingertip_friction(env_regex_ns=self.scene.env_regex_ns, robot_name="Robot")

        self.benchmark_enabled = bool(getattr(cfg, "benchmark_enabled", False))
        if self.benchmark_enabled:
            _create_relation_monitor(self, cfg)
            _create_subtask_runtime(self)

        self._reset_requested = False
        self._episode_count = 0

    def handle_termination_event(self, relation_events: list[dict], event_type: str):
        return handle_termination_event(self, relation_events, event_type)

    def request_manual_termination(self, env_ids=None):
        self._reset_requested = True # flag a reset that is serviced in step().

    def _on_episode_start(self, env_ids: torch.Tensor):
        if ENABLE_SPAWNER:
            _seed = os.environ.get("SIM_SEED")
            if _seed:
                seed = int(_seed) + self._episode_count
            else:
                seed = None
            self.scene_spawner.reset_objects(self.scene, env_ids, seed=seed)
            self._episode_count += 1
            # self.scene_spawner.apply_viewer(self.viewport_camera_controller)
        if hasattr(self, "robot"):
            root_state = self.robot.data.default_root_state.clone()
            root_state[env_ids, 7:] = 0.0  # zero linear and angular velocities
            if DISABLE_ROBOT_SPAWN_RANDOMIZATION:
                self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
            joint_pos = self.robot.data.default_joint_pos.clone()
            joint_vel = torch.zeros_like(self.robot.data.default_joint_vel)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            self.robot.set_joint_position_target(joint_pos[env_ids], env_ids=env_ids)
            self.robot.write_data_to_sim()
        # Aim the third-person camera at the scene's viewer pose (chosen per scene in
        # the scene JSON, so it is not occluded by furniture).
        if "third_person_camera" in self.scene.sensors:
            _cam = self.scene["third_person_camera"]
            _spawner = getattr(self, "scene_spawner", None)
            _vm = getattr(_spawner, "viewer_metadata", None) if _spawner is not None else None
            _eye = _vm.get("eye") if _vm else None
            _lookat = _vm.get("lookat") if _vm else None
            if _eye is not None and _lookat is not None:
                _eye_t = torch.tensor(_eye, dtype=torch.float32, device=self.device)
                _lookat_t = torch.tensor(_lookat, dtype=torch.float32, device=self.device)
                _cam.set_world_poses_from_view(_eye_t.unsqueeze(0), _lookat_t.unsqueeze(0))
        if self.benchmark_enabled:
            self.relation_monitor.reset(env_ids, sim_time=float(self.sim.current_time))
            self.relation_monitor.capture_upright_references(env_ids)
            self.benchmark_runtime.reset()
        if TRACK_EPISODES:
            episode_id = self.episodes.get_current_episode_id(env_id=0)
        if self.benchmark_enabled:
            num_episodes = getattr(self.cfg, "num_episodes", None)
            within_target = num_episodes is None or episode_id < num_episodes # prevents saving redundant episode.json in benchmark_log after final episode.
            if (env_ids == 0).any() and within_target:
                self.relation_recorder.on_reset_start_new_episode(episode_id=f"episode_{int(episode_id):03d}", sim_time=float(self.sim.current_time))
                if self.subtask_runtime is not None:
                    subtask_events = self.subtask_runtime.begin_episode(env_ids=[0], sim_time=float(self.sim.current_time),)
                    self.relation_recorder.record_subtask_events(subtask_events)
        if ENABLE_JOINT_MANAGER_TORQUE_CONTROL and ENABLE_ROBOT:
            self.joint_manager.reset_buffer()
        if FRAME_MODE != "none":
            self.sensor_manager.start_episode(episode_id)

    def step(self, action):
        if ENABLE_JOINT_MANAGER_TORQUE_CONTROL and ENABLE_ROBOT and not getattr(self, "_external_torque_from_ros2", False):
            action = jm_step(self, action) # convert positions->torques
        obs, reward, terminated, truncated, info = super().step(action)
        if self._check_robot_fallen():
            self._reset_requested = True

        # self.camera_spawner.apply_masks()
        # scene_object_manipulator = self.scene_spawner.build_object_manipulator(self.scene,)

        if FRAME_MODE != "none":
            self.sensor_manager.update()
        sim_time = float(self.sim.current_time)
        if self.benchmark_enabled:
            sim_step = int(getattr(self, "common_step_counter", 0))
            # sim_time = float(getattr(self.sim, "current_time", sim_step))
            self.benchmark_runtime.process(
                sim_step=sim_step,
                sim_time=sim_time,
                manual_reset=self._reset_requested,
                success_episode=False,
            )
        if self._reset_requested:
            if not self.benchmark_enabled:
                self.episodes.request_event("aborted")
            self._reset_requested = False
        return obs, reward, terminated, truncated, info
    
    def close(self):
        super().close()
        if FRAME_MODE in ("shm", "both"):
            self.sensor_manager.close()
        if self.benchmark_enabled:
            self.relation_recorder.close()

    def _reset_idx(self, env_ids: torch.Tensor):
        if TRACK_EPISODES:
            self.episodes.on_reset_idx(env_ids, self.episode_length_buf)
        super()._reset_idx(env_ids)
        if TRACK_EPISODES:
            self._on_episode_start(env_ids)

    def _check_robot_fallen(self) -> bool:
        root_z = self.robot.data.root_pos_w[0, 2]
        up_proj = -self.robot.data.projected_gravity_b[0, 2]
        return bool((root_z < 0.35) or (up_proj < 0.5))


@configclass
class ManipEnvCfg(ManagerBasedRLEnvCfg):
    scene: DefaultSceneCfg = DefaultSceneCfg(num_envs=1, env_spacing=2.0)
    observations: ExampleObservationsCfg = ExampleObservationsCfg()
    actions: ExampleActionsCfg = ExampleActionsCfg()
    rewards: NoRewardsCfg = NoRewardsCfg()
    terminations: ExampleTerminationsCfg = ExampleTerminationsCfg() # NoTerminationsCfg
    events: ExampleEventsCfg = ExampleEventsCfg()
    num_actions: Optional[int] = None
    num_observations: Optional[int] = None
    num_states: Optional[int] = None
    max_episode_steps: int = 50000000
    seed = 42
    benchmark_enabled: bool = False
    benchmark_task: Optional[str] = None
    benchmark_output_dir: str = "./benchmark_logs"
    benchmark_every_n_steps: int = 1
    num_episodes: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        # simdt=150 -> The robot completes 12 steps in 500 env.step() calls. This takes 13 real seconds.
        # simdt=550 -> The robot completes 4 steps in 500 env.step() calls. This takes 11 real seconds.
        self.sim.dt = 1.0/int(os.environ.get("env_simdt_denominator", "150")) #1.0 /150.0 #500=sameFpsButSlowMove, 200.0=walksInRoom  # 50=tooBigStepsAndFall, 250.0=default  # stable was 500
        self.decimation = int(os.environ.get("env_decimation", "1")) #1 # 5==do 5 phys steps per 1 simple step. stable was 4  in qwentest was 1, 1=robotFalls, 5=robotWalks
        self.sim.render_interval = int(os.environ.get("env_render_interval", "1")) #1 #2 #5 # 1=15fps and 13 step/sec #5=8fps and 21 step()/sec 
        print("sim.dt =", self.sim.dt, "decimation =", self.decimation, "render_interval =", self.sim.render_interval)
        control_dt = self.decimation * self.sim.dt  # seconds per action step
        self.episode_length_s = self.max_episode_steps * control_dt # 60  # Episode duration in seconds
        self.viewer.lookat = [0.0, 0.0, 1.2]
        self.viewer.eye = (3.1, 1.3, 1.8)
        
        # Enlarge PhysX GPU buffers. A randomized reset can momentarily cluster/overlap
        self.sim.physx.gpu_found_lost_pairs_capacity = 2 ** 23
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2 ** 25
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2 ** 23
        self.sim.physx.gpu_max_rigid_contact_count = 2 ** 24
        self.sim.physx.gpu_max_rigid_patch_count = 2 ** 21
        self.sim.physx.gpu_collision_stack_size = 2 ** 26
