from __future__ import annotations
from dataclasses import MISSING
import os
import random

from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.assets.articulation import Articulation
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ImuCfg
import isaaclab.sim as sim_utils

from .assets.robots import robot_cfg
from .assets.dummy_robot import dummy_robot_cfg
from .assets.odometry import imu_sensor
from .assets.collisions import CollisionMaterialApplier
from .assets.camera_spawner import CameraSpawner
from .assets.sensors import feet_contact_sensor
from .assets.terrains import ground_cfg
# from .assets.sensors import create_camera_cfg, CAMERA_PARAMETERS
# from .assets.sensors import create_camera_cfgs, cameras_config
from .assets import lights

ENABLE_ROBOT = not bool(int(os.environ.get("deactivate_robot", "0")))


def _make_third_person_camera() -> CameraCfg:
    """Fixed world-space pinhole camera looking at the robot root (~0, 0, 0.68)."""
    import isaaclab.sim as sim_utils
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/third_person_camera",
        update_period=0.067,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            clipping_range=(0.1, 1e5),
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            vertical_aperture=11.787,
        ),
        offset=CameraCfg.OffsetCfg(
            # Match the GUI startup viewport (manip_env.py viewer):
            # eye (3.1, 1.3, 1.8) looking at (0.0, 0.0, 1.2) — robot face + table.
            pos=(3.1, 1.3, 1.8),
            rot=(0.1965, 0.0865, 0.0174, -0.9765),  # yaw -157.2°, pitch 10.1° down
            convention="world",
        ),
    )


@configclass
class DefaultSceneCfg(InteractiveSceneCfg):
    def __post_init__(self) -> None:
        self.robot: Articulation = robot_cfg if ENABLE_ROBOT else dummy_robot_cfg
        if ENABLE_ROBOT:
            self.imu_sensor: ImuCfg = imu_sensor

        # camera
        # if not bool(int(os.environ.get("deactivate_cameras", "0"))):
        #     self.left_head_camera: CameraCfg = create_camera_cfgs(camera_params=cameras_config, add_depth=False)['left_head_camera']
        #     # self.right_head_camera: CameraCfg = create_camera_cfg(camera_params=cameras_config, add_depth=False)['right_head_camera']
        #     self.left_wrist_camera: CameraCfg = create_camera_cfgs(camera_params=cameras_config, add_depth=False)['left_wrist_camera']
        #     self.right_wrist_camera: CameraCfg = create_camera_cfgs(camera_params=cameras_config, add_depth=False)['right_wrist_camera']

        if bool(int(os.environ.get("deactivate_room", "0"))):
            self.ground: TerrainImporterCfg = ground_cfg

        # Third-person camera: fixed world position, simple pinhole.
        # Only created when video recording is enabled (env var set by run_with_vla.py).
        if bool(int(os.environ.get("enable_third_person_camera", "0"))):
            self.third_person_camera: CameraCfg = _make_third_person_camera()

        # light
        self.disk_light: AssetBaseCfg = lights.disk_light
        self.disk_light_0: AssetBaseCfg = lights.disk_light_0
        self.disk_light_1: AssetBaseCfg = lights.disk_light_1

        self.feet_contact_sensor = feet_contact_sensor

    def spawn_assets(self, scene_spawner):
        if not bool(int(os.environ.get("deactivate_cameras", "0"))):
            camera_spawner = CameraSpawner.from_config_manager(
                settings_name="camera_parameters",
                add_depth=False,
            )
            for cam_name, cam_cfg in camera_spawner.camera_cfgs.items():
                setattr(self, cam_name, cam_cfg)
        else: camera_spawner = None

        scene_spawner.spawn_scene_static_objects(num_envs=self.num_envs)
        scene_spawner.spawn_lighting()

        print("\nSpawner objects:", sorted(scene_spawner.object_metadata.keys()),"\n")
        assets_cfgs, articulation_cfgs, friction_specs = scene_spawner.get_asset_cfgs()
        # append rigid objects
        for asset_name, asset_cfg in assets_cfgs.items():
            setattr(self, asset_name, asset_cfg)
        # append articulated objects
        for asset_name, articulation_cfg in articulation_cfgs.items():
            setattr(self, asset_name, articulation_cfg)
        collision_applier = CollisionMaterialApplier.from_config_manager(settings_name="fingertip_collision",) # Check physMat on this: /World/envs/env_0/Robot/root/left_thumb_distal/collisions/link14_l/link14_l
        return collision_applier
    

    # def spawn_assets(self, json_path: str):
    #     camera_spawner = CameraSpawner.from_config_manager(
    #         settings_name="camera_parameters",
    #         add_depth=False,
    #     )
    #     for cam_name, cam_cfg in camera_spawner.camera_cfgs.items():
    #         setattr(self, cam_name, cam_cfg)
    #     scene_spawner = SceneSpawner(json_path)
    #     scene_spawner.spawn_scene_static_objects(num_envs=self.num_envs)
    #     scene_spawner.spawn_lighting()
    #     print(
    #         "\n[SceneSpawner] Object List:",
    #         sorted(scene_spawner.object_metadata.keys()),
    #         "\n",
    #     )
    #     assets_cfgs, articulation_cfgs, friction_specs = scene_spawner.get_asset_cfgs()
    #     # append rigid objects
    #     for asset_name, asset_cfg in assets_cfgs.items():
    #         setattr(self, asset_name, asset_cfg)
    #     # append articulated objects
    #     for asset_name, articulation_cfg in articulation_cfgs.items():
    #         setattr(self, asset_name, articulation_cfg)
    #     collision_applier = CollisionMaterialApplier.from_config_manager(
    #         settings_name="fingertip_collision",
    #     )
    #     return scene_spawner, camera_spawner, collision_applier

# random 10cm cube resting on the desk center
# DESK_CENTER = (0.6, 0.2, 0.8)
# CUBE_SIZE = 0.1  # 10 cm
# self.random_cube: RigidObjectCfg = RigidObjectCfg(
#     prim_path="{ENV_REGEX_NS}/random_cube",
#     spawn=sim_utils.CuboidCfg(
#         size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
#         visual_material=sim_utils.PreviewSurfaceCfg(
#             diffuse_color=(random.random(), random.random(), random.random()),
#         ),
#         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             rigid_body_enabled=True,
#             kinematic_enabled=True,
#             disable_gravity=True,
#         ),
#         mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
#         physics_material=sim_utils.RigidBodyMaterialCfg(
#             static_friction=0.5,
#             dynamic_friction=0.4,
#             restitution=0.2,
#             friction_combine_mode="average",
#             restitution_combine_mode="average",
#         ),
#     ),
#     init_state=RigidObjectCfg.InitialStateCfg(
#         # rest the cube on the tabletop: surface z + half the cube height
#         pos=(DESK_CENTER[0], DESK_CENTER[1], DESK_CENTER[2] + CUBE_SIZE / 2.0),
#         rot=(1.0, 0.0, 0.0, 0.0),
#     ),
# )


# # camera
# if not bool(int(os.environ.get("deactivate_cameras", "0"))):
#     self.left_head_camera: CameraCfg = create_camera_cfg(cam_name='left_head_camera', cfg=CAMERA_PARAMETERS["left_head_camera"], add_depth=True)
#     # self.right_head_camera: CameraCfg = create_camera_cfg(cam_name='right_head_camera', cfg=CAMERA_PARAMETERS["right_head_camera"], add_depth=True)
#     self.left_wrist_camera: CameraCfg = create_camera_cfg(cam_name='left_wrist_camera', cfg=CAMERA_PARAMETERS["left_wrist_camera"], add_depth=True)
#     self.right_wrist_camera: CameraCfg = create_camera_cfg(cam_name='right_wrist_camera', cfg=CAMERA_PARAMETERS["right_wrist_camera"], add_depth=True)
