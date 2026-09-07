import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
import os

from green_challenge.envs.manip_utils.scenes.assets.rigid_objects import SceneSpawner


def register_envs():
    gym.envs.registration.register(
        id="ManipEnv-v0",
        entry_point="green_challenge.envs.manip_env:ManipEnv",
        kwargs={"env_cfg_entry_point": "green_challenge.envs.manip_env:ManipEnvCfg"},
    )

def make_env(
    env_id: str,
    num_envs: int = 1,
    benchmark_task: str | None = None,
    benchmark_output_dir: str = "./benchmark_logs",
    benchmark_every_n_steps: int = 1,
    num_episodes: int | None = None,
    scenario_name: str | None = None,
) -> ManagerBasedRLEnv:
    env_cfg = parse_env_cfg(
        env_id,  # type: ignore
        num_envs=num_envs,
    )

    env_cfg.sim.device = "cpu"
    env_cfg.sim.use_gpu_pipeline = False
    env_cfg.sim.physx.use_gpu = False
    
    env_cfg.sim.enable_scene_query_support=False
    env_cfg.sim.physx.num_threads=30
    
    env_cfg.sim.render.enable_translucency = True
    # IMPORTANT: In RTX Real-time mode, Fractional Cutout must be enabled,
    # otherwise the transparency of the results will be "binary" (0 or 1).
    # In Isaac Lab 2025, this is done via direct parameter expansion:
    if env_cfg.sim.render.antialiasing_mode is not None:
        import carb
        settings = carb.settings.get_settings()
        settings.set("/rtx/translucency/enabled", True)
        settings.set("/rtx/translucency/fractionalCutout/enabled", True)
        settings.set("/rtx/translucency/cutoff", 0.0)
        settings.set("/rtx/translucency/fractionalCutout/threshold", 0.0)

    # experimental settings
    env_cfg.sim.enable_scene_query_support = True   # cameras, distance calculation to objects
    env_cfg.sim.physx.solver_type = 0               # PGS (0) faster than TGS (1), but less stable
    env_cfg.sim.physx.num_threads = os.cpu_count()  # Often 4-8 cores faster than 30 due to cache
    env_cfg.sim.physx.num_position_iterations = 4   # default is often 16
    env_cfg.sim.physx.num_velocity_iterations = 0   # Minimum for stability
    env_cfg.sim.physx.contact_collection_mode = 0   # 0: none, 1: all, 2: report
    env_cfg.sim.physx.enable_stabilization = True   # Disable collision visualization and other things
    env_cfg.sim.use_fabric = True # Keep True, don't duplicate data via USD API (get_prim_attribute), since USD API on CPU is "death" for FPS.

    env_cfg.sim.physx.bounce_threshold_velocity = 0.5   # Don't count micro-bounces
    env_cfg.sim.physx.sleep_threshold = 0.05            # Put objects to sleep earlier
    env_cfg.sim.physx.stabilization_threshold = 0.01

    env_cfg.sim.use_flatcache = False   # structure for fast data transfer to GPU

    print("[INFO] Using CPU dynamics.")
    env_cfg.benchmark_task = benchmark_task
    env_cfg.benchmark_enabled = benchmark_task is not None
    env_cfg.benchmark_output_dir = benchmark_output_dir
    env_cfg.benchmark_every_n_steps = benchmark_every_n_steps
    env_cfg.num_episodes = num_episodes

    env: ManagerBasedRLEnv = gym.make(env_id, cfg=env_cfg, scenario_name=scenario_name)  # type: ignore
    return env
