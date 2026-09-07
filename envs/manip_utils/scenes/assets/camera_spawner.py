from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

from green_challenge.robots.robot_selector import ConfigManager

# Aperture ≈ sensor width in mm computed for ZED X One GS from:
# pixel pitch × pixel count ~ 5.784 mm  (1/2.6" sensor, 3µm pixels, 1928 px width).
# FoV=2arctan(sensor size/ 2xfocal_length)
# intrinsic_matrix = [fx, 0,  cx], 
#                    [ 0, fy, cy], 
#                    [ 0, 0,  1]

class CameraSpawner:
    def __init__(
        self,
        camera_parameters,
        prim_path_template="{ENV_REGEX_NS}/Robot/root/{camera_name}/{camera_name}_sensor",
        add_depth=True,
        mask_dir=None,
    ):
        self._raw_cfg = deepcopy(camera_parameters)
        self._prim_path_template = prim_path_template
        self._add_depth = add_depth
        self._mask_dir = Path(mask_dir) if mask_dir is not None else None

        self.camera_params = self._build_camera_parameters(self._raw_cfg)
        self.camera_cfgs = self._create_camera_cfgs(self.camera_params)

        self._scene = None
        self._mask_runtime = {}

    @classmethod
    def from_config_manager(
        cls,
        settings_name="camera_parameters",
        prim_path_template="{ENV_REGEX_NS}/Robot/root/{camera_name}/{camera_name}_sensor",
        add_depth=True,
        mask_dir=None,
    ):
        cm = ConfigManager.instance()
        camera_parameters = cm.load_settings(settings_name)

        if mask_dir is None:
            mask_dir = cm.settings_dir 

        return cls(
            camera_parameters=camera_parameters,
            prim_path_template=prim_path_template,
            add_depth=add_depth,
            mask_dir=mask_dir,
        )

    def _resolve_mask_path(self, mask_file: str) -> Path:
        path = Path(mask_file)
        if path.is_absolute():
            return path
        if self._mask_dir is not None:
            return self._mask_dir / path
        return path

    @staticmethod
    def _build_camera_parameters(cfg: dict) -> dict:
        presets = cfg["presets"]
        mapping = cfg["camera_mapping"]

        return {
            camera_key: presets[preset_name]
            for camera_key, preset_name in mapping.items()
            if preset_name is not None
        }

    @staticmethod
    def _build_pinhole_cfg(cfg: dict):
        intrinsic = cfg["intrinsic"]

        focal_length = (
            intrinsic["fx"] * intrinsic["horizontal_aperture"]
        ) / intrinsic["resolution"][0]

        intrinsic_matrix = [
            intrinsic["fx"], 0, intrinsic["cx"],
            0, intrinsic["fy"], intrinsic["cy"],
            0, 0, 1,
        ]

        return sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsic_matrix,
            width=intrinsic["resolution"][0],
            height=intrinsic["resolution"][1],
            focal_length=focal_length,
            projection_type="pinhole",
        )

    @staticmethod
    def _build_fisheye_cfg(cfg: dict, pinhole_cfg):
        intrinsic = cfg["intrinsic"]

        dist = list(intrinsic.get("distortion", [0.0, 0.0, 0.0, 0.0]))
        dist = (dist + [0.0, 0.0, 0.0, 0.0])[:4]
        k1, k2, k3, k4 = dist

        return sim_utils.FisheyeCameraCfg(
            projection_type=cfg.get("projection_type", "fisheyeKannalaBrandtK3"),
            fisheye_nominal_width=float(intrinsic.get("ftheta_width", intrinsic["resolution"][0])),
            fisheye_nominal_height=float(intrinsic.get("ftheta_height", intrinsic["resolution"][1])),
            fisheye_optical_centre_x=intrinsic["cx"],
            fisheye_optical_centre_y=intrinsic["cy"],
            fisheye_polynomial_a=k1,
            fisheye_polynomial_b=k2,
            fisheye_polynomial_c=k3,
            fisheye_polynomial_d=k4,
            fisheye_polynomial_e=0.0,
            fisheye_polynomial_f=0.0,
            fisheye_max_fov=intrinsic["fov"],
            clipping_range=tuple(cfg["clipping_range"]),
            focal_length=pinhole_cfg.focal_length,
            focus_distance=pinhole_cfg.focus_distance,
            f_stop=pinhole_cfg.f_stop,
            horizontal_aperture=pinhole_cfg.horizontal_aperture,
            vertical_aperture=pinhole_cfg.vertical_aperture,
            horizontal_aperture_offset=pinhole_cfg.horizontal_aperture_offset,
            vertical_aperture_offset=pinhole_cfg.vertical_aperture_offset,
            lock_camera=pinhole_cfg.lock_camera,
        )

    def _create_camera_cfgs(
        self,
        camera_params: Dict[str, Any],
    ) -> Dict[str, CameraCfg]:
        camera_cfgs = {}

        for cam_name, cfg in camera_params.items():
            if cfg is None:
                continue

            intrinsic = cfg["intrinsic"]
            extrinsic = cfg["extrinsic"]

            pinhole_cfg = self._build_pinhole_cfg(cfg)
            fisheye_cfg = self._build_fisheye_cfg(cfg, pinhole_cfg)

            projection_type = cfg.get("projection_type", "fisheyeKannalaBrandtK3")
            spawn_cfg = pinhole_cfg if projection_type == "pinhole" else fisheye_cfg

            data_types = list(cfg["data_types"])
            if self._add_depth and "depth" not in data_types:
                data_types.append("depth")

            prim_path = cfg.get(
                "prim_path",
                self._prim_path_template.format(
                    ENV_REGEX_NS="{ENV_REGEX_NS}",
                    camera_name=cam_name,
                ),
            )

            pos = tuple(float(x) for x in extrinsic["position"])
            rot = tuple(float(x) for x in extrinsic["rotation"])

            cam_cfg = CameraCfg(
                prim_path=prim_path,
                update_period=cfg["update_period"],
                height=cfg["height"],
                width=cfg["width"],
                data_types=data_types,
                spawn=spawn_cfg,
                offset=CameraCfg.OffsetCfg(
                    pos=pos,
                    rot=rot,
                    convention=extrinsic.get("convention", "opengl"),
                ),
                debug_vis=cfg.get("debug_vis", False),
            )

            camera_cfgs[cam_name] = cam_cfg

        return camera_cfgs

    def bind_scene(self, scene, device: Optional[str] = None) -> None:
        self._scene = scene
        self._init_masks(device=device)

    def _init_masks(self, device=None) -> None:
        runtime = {}

        for cam_name, cfg in self.camera_params.items():
            mask_file = cfg.get("mask_file")
            if not mask_file:
                continue

            mask_path = self._resolve_mask_path(mask_file)
            mask_np = np.load(mask_path)

            if mask_np.dtype != np.bool_:
                mask_np = mask_np > 0

            mask = torch.from_numpy(mask_np).bool()
            if device is not None:
                mask = mask.to(device=device)

            runtime[cam_name] = {"mask": mask}

        self._mask_runtime = runtime

    def apply_masks(self) -> None:
        if self._scene is None or not self._mask_runtime:
            return

        for cam_name, item in self._mask_runtime.items():
            rgb = self._scene[cam_name].data.output.get("rgb")
            if rgb is None:
                continue

            mask = item["mask"]
            if mask.device != rgb.device:
                mask = mask.to(rgb.device)
                item["mask"] = mask

            if rgb.ndim != 4:
                continue

            _, h, w, c = rgb.shape
            if tuple(mask.shape) != (h, w) or c < 3:
                continue

            inv = (~mask).unsqueeze(0).unsqueeze(-1)   # (1, H, W, 1)
            rgb.masked_fill_((~mask).unsqueeze(0).unsqueeze(-1), 0)

    @staticmethod
    def rescale_camera_preset(
        preset: dict,
        new_width: int,
        new_height: int,
        mode: str = "resize",
    ) -> dict:
        out = deepcopy(preset)

        intrinsic = out["intrinsic"]
        old_width, old_height = intrinsic["resolution"]
        fx, fy = intrinsic["fx"], intrinsic["fy"]
        cx, cy = intrinsic["cx"], intrinsic["cy"]

        if mode == "resize":
            sx = new_width / old_width
            sy = new_height / old_height

            intrinsic["fx"] = fx * sx
            intrinsic["fy"] = fy * sy
            intrinsic["cx"] = cx * sx
            intrinsic["cy"] = cy * sy

        elif mode == "center_crop_resize":
            crop_size = min(old_width, old_height)
            crop_x = (old_width - crop_size) / 2.0
            crop_y = (old_height - crop_size) / 2.0

            cx_crop = cx - crop_x
            cy_crop = cy - crop_y

            sx = new_width / crop_size
            sy = new_height / crop_size

            intrinsic["fx"] = fx * sx
            intrinsic["fy"] = fy * sy
            intrinsic["cx"] = cx_crop * sx
            intrinsic["cy"] = cy_crop * sy

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        intrinsic["resolution"] = [new_width, new_height]
        out["width"] = new_width
        out["height"] = new_height

        return out

    @classmethod
    def rescale_camera_config(
        cls,
        cfg: dict,
        new_width: int,
        new_height: int,
        mode: str = "resize",
    ) -> dict:
        out = deepcopy(cfg)
        out["presets"] = {
            preset_name: cls.rescale_camera_preset(preset, new_width, new_height, mode=mode)
            for preset_name, preset in cfg["presets"].items()
        }
        return out
