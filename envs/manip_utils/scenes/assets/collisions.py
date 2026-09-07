from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import isaaclab.sim as sim_utils

from green_challenge.robots.robot_selector import ConfigManager #from simple_example.utils.config_manager import ConfigManager

@dataclass(slots=True)
class CollisionMaterialApplier:
    fingertip_collision_suffixes: list[str]
    static_friction: float = 5.0
    dynamic_friction: float = 5.0
    restitution: float = 0.0
    friction_combine_mode: str = "max"
    restitution_combine_mode: str = "min"
    # improve_patch_friction: bool = True
    material_path: str = "/World/PhysicsMaterials/fingertip_grip"

    @classmethod
    def from_config_manager(
        cls,
        settings_name: str = "fingertip_collision",
    ) -> "CollisionMaterialApplier":
        cm = ConfigManager.instance()
        data = cm.load_settings(settings_name)

        if isinstance(data, dict):
            suffixes = data.get("collision_links", [])
            material = data.get("physics_material", {})
            material_path = data.get(
                "material_path",
                "/World/PhysicsMaterials/fingertip_grip",
            )

            return cls(
                fingertip_collision_suffixes=list(suffixes),
                static_friction=float(material.get("static_friction", 5.0)),
                dynamic_friction=float(material.get("dynamic_friction", 5.0)),
                restitution=float(material.get("restitution", 0.0)),
                friction_combine_mode=str(
                    material.get("friction_combine_mode", "max")
                ),
                restitution_combine_mode=str(
                    material.get("restitution_combine_mode", "min")
                ),
                # improve_patch_friction=bool(
                #     material.get("improve_patch_friction", True)
                # ),
                material_path=material_path,
            )

        if isinstance(data, list):
            return cls(fingertip_collision_suffixes=list(data))

        raise ValueError(
            f"Settings '{settings_name}' must be list[str] or dict."
        )

    def _ensure_material(self) -> str:
        stage = sim_utils.get_current_stage()
        prim = stage.GetPrimAtPath(self.material_path)

        if not prim.IsValid():
            cfg = sim_utils.RigidBodyMaterialCfg(
                static_friction=self.static_friction,
                dynamic_friction=self.dynamic_friction,
                restitution=self.restitution,
                friction_combine_mode=self.friction_combine_mode,
                restitution_combine_mode=self.restitution_combine_mode,
                # improve_patch_friction=self.improve_patch_friction,
            )
            cfg.func(self.material_path, cfg)

        return self.material_path

    def apply_fingertip_friction(
        self,
        env_regex_ns: str,
        robot_name: str = "Robot",
    ) -> None:
        stage = sim_utils.get_current_stage()
        material_path = self._ensure_material()

        robot_roots = sim_utils.find_matching_prim_paths(
            f"{env_regex_ns}/{robot_name}",
            stage=stage,
        )
        if not robot_roots:
            raise RuntimeError(
                f"No robot roots found for pattern: {env_regex_ns}/{robot_name}"
            )

        for robot_root in robot_roots:
            sim_utils.make_uninstanceable(robot_root, stage=stage)

            for suffix in self.fingertip_collision_suffixes:
                prim_path = f"{robot_root}/{suffix}"
                prim = stage.GetPrimAtPath(prim_path)

                if not prim.IsValid():
                    raise RuntimeError(f"Collision prim not found: {prim_path}")

                sim_utils.bind_physics_material(
                    prim_path=prim_path,
                    material_path=material_path,
                    stage=stage,
                    stronger_than_descendants=True,
                )