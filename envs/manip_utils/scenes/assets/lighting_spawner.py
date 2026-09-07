import re
from typing import Any, Dict, Optional

import omni.usd
import isaaclab.sim as sim_utils


class LightingSpawner:
    """Parses compact lighting JSON and spawns real lights + visible light geometry."""

    LIGHT_REGISTRY = {
        "dome": {
            "cls": sim_utils.DomeLightCfg,
            "tuple": ("color",),
            "float": ("intensity", "exposure", "color_temperature"),
            "plain": (
                "enable_color_temperature",
                "normalize",
                "texture_file",
                "texture_format",
                "visible_in_primary_ray",
            ),
        },
        "disk": {
            "cls": sim_utils.DiskLightCfg,
            "tuple": ("color",),
            "float": ("intensity", "exposure", "color_temperature", "radius"),
            "plain": ("enable_color_temperature", "normalize"),
        },
        "sphere": {
            "cls": sim_utils.SphereLightCfg,
            "tuple": ("color",),
            "float": ("intensity", "exposure", "color_temperature", "radius"),
            "plain": ("enable_color_temperature", "normalize"),
        },
        "distant": {
            "cls": sim_utils.DistantLightCfg,
            "tuple": ("color",),
            "float": ("intensity", "exposure", "color_temperature", "angle"),
            "plain": ("enable_color_temperature", "normalize"),
        },
        "cylinder": {
            "cls": sim_utils.CylinderLightCfg,
            "tuple": ("color",),
            "float": ("intensity", "exposure", "color_temperature", "radius", "length"),
            "plain": ("enable_color_temperature", "normalize"),
        },
    }

    VISUAL_REGISTRY = {
        "cylinder": {
            "cls": sim_utils.CylinderCfg,
            "float": ("radius", "height"),
            "plain": ("axis",),
        },
    }

    PREVIEW_SURFACE_SPEC = {
        "tuple": ("diffuse_color", "emissive_color"),
        "float": ("roughness", "metallic", "opacity"),
    }

    def __init__(self, config: Dict[str, Any], prim_root: str = "/World/Light"):
        self.config = config
        self.prim_root = prim_root.rstrip("/")
        self.stage = omni.usd.get_context().get_stage()

        self.light_definitions = config.get("light_definitions", {})
        self.visual_definitions = config.get("light_visual_definitions", {})
        self.lighting = config.get("lighting", [])

        if not isinstance(self.light_definitions, dict):
            raise ValueError("'light_definitions' must be a dict")
        if not isinstance(self.visual_definitions, dict):
            raise ValueError("'light_visual_definitions' must be a dict")
        if not isinstance(self.lighting, list):
            raise ValueError("'lighting' must be a list")

        self.light_metadata: Dict[str, Dict[str, Any]] = {}
        self.light_visual_metadata: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_") or "unnamed"

    @staticmethod
    def _vec3(value: Any, field: str, default=None) -> tuple[float, float, float]:
        if value is None:
            if default is not None:
                return tuple(float(x) for x in default)
            raise ValueError(f"Field '{field}' is required")
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"Field '{field}' must be length-3, got: {value}")
        return tuple(float(x) for x in value)

    @staticmethod
    def _quat(value: Any, field: str, default=(1.0, 0.0, 0.0, 0.0)) -> tuple[float, float, float, float]:
        if value is None:
            return tuple(float(x) for x in default)
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError(f"Field '{field}' must be length-4 quaternion, got: {value}")
        return tuple(float(x) for x in value)

    @staticmethod
    def _vadd(
        a: tuple[float, float, float],
        b: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    @staticmethod
    def _collect_kwargs(meta: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = {}

        for key in spec.get("tuple", ()):
            value = meta.get(key)
            if value is not None:
                kwargs[key] = tuple(float(x) for x in value)

        for key in spec.get("float", ()):
            value = meta.get(key)
            if value is not None:
                kwargs[key] = float(value)

        for key in spec.get("plain", ()):
            value = meta.get(key)
            if value is not None:
                kwargs[key] = value

        return kwargs

    def _build_light_cfg(self, meta: Dict[str, Any]):
        light_type = meta["type"]
        if light_type not in self.LIGHT_REGISTRY:
            raise ValueError(f"Unknown light type: {light_type}")
        spec = self.LIGHT_REGISTRY[light_type]
        return spec["cls"](**self._collect_kwargs(meta, spec))

    def _build_preview_surface_cfg(self, meta: Dict[str, Any]):
        if meta.get("type", "preview_surface") != "preview_surface":
            raise ValueError(f"Unsupported visual material type: {meta.get('type')}")
        return sim_utils.PreviewSurfaceCfg(
            **self._collect_kwargs(meta, self.PREVIEW_SURFACE_SPEC)
        )

    def _build_visual_cfg(self, meta: Dict[str, Any]):
        visual_type = meta["type"]
        if visual_type not in self.VISUAL_REGISTRY:
            raise ValueError(f"Unsupported light visual type: {visual_type}")

        spec = self.VISUAL_REGISTRY[visual_type]
        kwargs = self._collect_kwargs(meta, spec)
        kwargs["visual_material"] = self._build_preview_surface_cfg(meta["visual_material"])
        kwargs["collision_props"] = sim_utils.CollisionPropertiesCfg() if meta.get("collision", False) else None
        kwargs["rigid_props"] = sim_utils.RigidBodyPropertiesCfg() if meta.get("rigid", False) else None
        return spec["cls"](**kwargs)

    def _prim_path(self, name: str) -> str:
        return f"{self.prim_root}/{name}"

    def spawn(self) -> None:
        for item in self.lighting:
            if not isinstance(item, dict):
                raise ValueError("Each item in 'lighting' must be a dict")

            name = self._sanitize_name(item["name"])
            pos = self._vec3(item.get("position"), f"lighting[{name}].position")
            rot = self._quat(item.get("rotation"), f"lighting[{name}].rotation")

            light_ref = item.get("light")
            visual = item.get("visual")

            if light_ref is None and visual is None:
                raise ValueError(f"lighting[{name}] must define at least 'light' or 'visual'")

            if light_ref is not None:
                if not isinstance(light_ref, str):
                    raise ValueError(f"'light' in lighting[{name}] must be a string")
                if light_ref not in self.light_definitions:
                    raise KeyError(f"Unknown light ref '{light_ref}' in lighting[{name}]")

                light_name = f"{name}_light"
                light_prim_path = self._prim_path(light_name)
                light_meta = dict(self.light_definitions[light_ref])

                self.light_metadata[light_name] = {
                    "prim_path": light_prim_path,
                    "position": pos,
                    "rotation": rot,
                    "ref": light_ref,
                    **light_meta,
                }

                if not self.stage.GetPrimAtPath(light_prim_path).IsValid():
                    cfg = self._build_light_cfg(light_meta)
                    cfg.func(light_prim_path, cfg, translation=pos, orientation=rot)

            if visual is not None:
                if isinstance(visual, str):
                    visual_ref = visual
                    visual_offset = self._vec3(
                        item.get("visual_offset"),
                        f"lighting[{name}].visual_offset",
                        default=(0.0, 0.0, 0.0),
                    )
                elif isinstance(visual, dict):
                    if "ref" not in visual:
                        raise ValueError(f"'visual' dict in lighting[{name}] must contain 'ref'")
                    visual_ref = visual["ref"]
                    visual_offset = self._vec3(
                        visual.get("offset", item.get("visual_offset")),
                        f"lighting[{name}].visual.offset",
                        default=(0.0, 0.0, 0.0),
                    )
                else:
                    raise ValueError(f"'visual' in lighting[{name}] must be string or dict")

                if visual_ref not in self.visual_definitions:
                    raise KeyError(f"Unknown visual ref '{visual_ref}' in lighting[{name}]")

                visual_name = f"{name}_visual"
                visual_prim_path = self._prim_path(visual_name)
                visual_pos = self._vadd(pos, visual_offset)
                visual_meta = dict(self.visual_definitions[visual_ref])

                self.light_visual_metadata[visual_name] = {
                    "prim_path": visual_prim_path,
                    "position": visual_pos,
                    "rotation": rot,
                    "ref": visual_ref,
                    **visual_meta,
                }

                if not self.stage.GetPrimAtPath(visual_prim_path).IsValid():
                    cfg = self._build_visual_cfg(visual_meta)
                    cfg.func(visual_prim_path, cfg, translation=visual_pos, orientation=rot)