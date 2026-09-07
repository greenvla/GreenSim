"""
Scene Spawner for IsaacLab with Consistent Initialization and Reset.

This module enables deterministic scene initialization and fast reset of object poses
while respecting the original orientation embedded in USD files.

Key features:
- Reads local transform (position + rotation) from USD files once at startup.
- Combines user-specified rotation with USD's local rotation for consistent behavior.
- Supports stack/grid layouts, multi-instance spawning, and collision-aware placement.
- Caches USD-derived data to avoid redundant I/O during resets.
- Fully compatible with IsaacLab's RigidObject API.

Usage:
    spawner = SceneSpawner("scene.json")
    asset_cfgs, friction_specs = spawner.get_asset_cfgs()  # for SceneCfg
    # ...
    spawner.reset_objects(scene, env_ids)  # in _reset_idx
"""

from __future__ import annotations
import json
from typing import Dict, Tuple, List, Any, Optional
from pathlib import Path
import numpy as np
import torch
import isaaclab.sim as sim_utils

import warp as wp
from isaaclab.sim import SimulationContext

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, PhysxSchema
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg, ArticulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
# from .scene_object_manipulator import SceneObjectManipulator

from .mdl_register import _rewrite_usd_mdl_paths_to_package_paths
from .lighting_spawner import LightingSpawner

sphere_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/sphere",
    spawn=sim_utils.SphereCfg(
        radius=0.015,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.4,
            restitution=0.2,
            friction_combine_mode="average",
            restitution_combine_mode="average",
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0, 0, 0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

box_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/box",
    spawn=sim_utils.CuboidCfg(
        size=(0.70, 1.40, 0.025),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.4,
            restitution=0.2,
            friction_combine_mode="average",
            restitution_combine_mode="average",
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(1.6, 0.0, 0.7),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

PARKING_POSITION = [10.0, 10.0, 0.5]
# Cache for computed oriented AABB sizes: (usd_path, rotation, scale) -> [x, y, z]
_USD_SIZE_CACHE: dict[tuple[str, tuple[float, ...], tuple[float, ...]], np.ndarray] = {}
# Cache for local transforms: usd_path -> (local_pos, local_rot)
_USD_LOCAL_TRANSFORM_CACHE: dict[str, Tuple[np.ndarray, np.ndarray]] = {}


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3D vector v by quaternion q=(w, x, y, z)."""
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)

    q_w = q[0]
    q_xyz = q[1:4]

    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def _quat_mul_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication for one [w, x, y, z] torch quaternion."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _quat_rotate_torch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate one 3-vector v by one quaternion q=[w, x, y, z]."""
    q = q / torch.clamp(
        torch.linalg.vector_norm(q),
        min=1e-12,
    )

    q_w = q[0]
    q_xyz = q[1:4]

    t = 2.0 * torch.cross(q_xyz, v, dim=0)
    return v + q_w * t + torch.cross(q_xyz, t, dim=0)


def _compute_usd_local_transform(usd_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts local position and rotation from a USD file with caching.

    Opens a USD stage and retrieves the local transformation (translation and
    rotation) from the default or first available Xformable prim. Results are
    cached to avoid redundant file I/O operations.

    Args:
        usd_path (str): Absolute or relative path to the USD file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - position (np.ndarray): Translation vector of shape (3,) [x, y, z].
            - rotation (np.ndarray): Quaternion of shape (4,) [w, x, y, z].
            Returns identity transform (zeros position, [1,0,0,0] rotation) if
            extraction fails.

    Note:
        Results are cached in _USD_LOCAL_TRANSFORM_CACHE to improve performance
        when the same USD file is queried multiple times.
        Compatible with IsaacLab 2.3.2+ (uses GetLocalTransformation instead of
        ComputeLocalTransformation for newer USD bindings).
        Debug output is printed to stdout for transformation values.
    """
    if usd_path in _USD_LOCAL_TRANSFORM_CACHE:
        return _USD_LOCAL_TRANSFORM_CACHE[usd_path]

    try:
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            raise ValueError(f"Failed to open USD stage: {usd_path}")

        root_prim = stage.GetDefaultPrim()
        if not root_prim:
            # Fallback: find first Xformable prim in the stage
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Xformable):
                    root_prim = prim
                    break
        if not root_prim:
            raise ValueError(f"No Xformable prim found in {usd_path}")

        xform = UsdGeom.Xformable(root_prim)
        time = Usd.TimeCode.Default()
        # IsaacLab 2.3.2 / newer USD: GetLocalTransformation instead of ComputeLocalTransformation
        tf = xform.GetLocalTransformation(time)
        transform = tf[0] if isinstance(tf, tuple) else tf
        # Extract translation
        pos = np.array(transform.ExtractTranslation())

        # Extract rotation as quaternion (w, x, y, z)
        rotation = transform.ExtractRotation()
        quat = rotation.GetQuat()
        rot = np.array(
            [
                quat.GetReal(),
                quat.GetImaginary()[0],
                quat.GetImaginary()[1],
                quat.GetImaginary()[2],
            ]
        )

        result = (pos, rot)
        _USD_LOCAL_TRANSFORM_CACHE[usd_path] = result
        return result

    except Exception as e:
        print(f"Warning: Failed to read local transform from {usd_path}: {e}")
        fallback = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        _USD_LOCAL_TRANSFORM_CACHE[usd_path] = fallback
        return fallback


def _compute_oriented_aabb_size(
    usd_path: str,
    obj_rotation: list[float],
    scale: list[float] | None = None,
) -> np.ndarray:
    """Computes world-aligned AABB size of a USD asset with caching.

    Calculates the effective bounding box size of a USD asset in world coordinates,
    accounting for the object's local scale and rotation. This is useful for
    collision-aware placement and spatial queries.

    The function computes the oriented bounding box by transforming all 8 corners
    of the local AABB through the rotation matrix, then finding the new min/max
    bounds in world space.

    Args:
        usd_path (str): Absolute or relative path to the USD file.
        obj_rotation (list[float]): Initial orientation as a quaternion [w, x, y, z].
        scale (list[float] | None): Scaling factors applied to the asset [sx, sy, sz].
            Defaults to [1.0, 1.0, 1.0] if not provided.

    Returns:
        np.ndarray: Array of shape (3,) containing the effective size [size_x, size_y, size_z]
            in world coordinates after applying scale and rotation.

    Note:
        Results are cached in _USD_SIZE_CACHE using a composite key of
        (usd_path, rotation, scale) to avoid redundant computations.
        Uses UsdGeom.BBoxCache for stable bounding box computation across USD versions.
        Returns fallback value [0.1, 0.1, 0.1] if computation fails.
        Rotation is skipped if quaternion equals identity [1.0, 0.0, 0.0, 0.0].
    """
    if scale is None:
        scale = [1.0, 1.0, 1.0]

    key = (
        usd_path,
        tuple(round(x, 6) for x in obj_rotation),
        tuple(round(x, 6) for x in scale),
    )
    if key in _USD_SIZE_CACHE:
        return _USD_SIZE_CACHE[key]

    try:
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            raise ValueError(f"Failed to open USD stage: {usd_path}")

        root_prim = stage.GetDefaultPrim()
        if not root_prim:
            # Fallback: find first Mesh prim in the stage
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Mesh):
                    root_prim = prim
                    break
        if not root_prim:
            raise ValueError(f"No valid geometry found in {usd_path}")

        time = Usd.TimeCode.Default()
        # More reliable via BBoxCache (works more stably between USD versions)
        bbox_cache = UsdGeom.BBoxCache(
            time,
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )

        local_bounds = bbox_cache.ComputeLocalBound(root_prim)
        rng = local_bounds.GetRange()

        # In newer bindings, check emptiness on range, not on bbox
        if rng.IsEmpty():
            world_bounds = bbox_cache.ComputeWorldBound(root_prim)
            rng = world_bounds.GetRange()

        if rng.IsEmpty():
            raise ValueError(f"Empty bbox range for {usd_path}")

        local_size = np.array(rng.GetSize(), dtype=float)
        scaled_size = np.abs(local_size * np.array(scale, dtype=float))

        if obj_rotation == [1.0, 0.0, 0.0, 0.0]:
            result = scaled_size
        else:
            # Compute oriented bounding box by rotating all 8 corners
            cx, cy, cz = scaled_size / 2.0
            corners_local = np.array(
                [
                    [-cx, -cy, -cz],
                    [-cx, -cy, cz],
                    [-cx, cy, -cz],
                    [-cx, cy, cz],
                    [cx, -cy, -cz],
                    [cx, -cy, cz],
                    [cx, cy, -cz],
                    [cx, cy, cz],
                ]
            )

            w, x, y, z = obj_rotation
            R = np.array(
                [
                    [
                        1 - 2 * y * y - 2 * z * z,
                        2 * x * y - 2 * z * w,
                        2 * x * z + 2 * y * w,
                    ],
                    [
                        2 * x * y + 2 * z * w,
                        1 - 2 * x * x - 2 * z * z,
                        2 * y * z - 2 * x * w,
                    ],
                    [
                        2 * x * z - 2 * y * w,
                        2 * y * z + 2 * x * w,
                        1 - 2 * x * x - 2 * y * y,
                    ],
                ]
            )

            corners_world = (R @ corners_local.T).T
            min_w = corners_world.min(axis=0)
            max_w = corners_world.max(axis=0)
            oriented_size = max_w - min_w
            result = np.abs(oriented_size)

        _USD_SIZE_CACHE[key] = result
        return result

    except Exception as e:
        print(
            f"Warning: Failed to compute size for {usd_path}, using fallback [0.1, 0.1, 0.1]. Error: {e}"
        )
        fallback = np.array([0.1, 0.1, 0.1])
        _USD_SIZE_CACHE[key] = fallback
        return fallback


def _determine_spawn_count(
    spawn_count_config: int | List[int] | None,
    total_objects: int,
    rng: np.random.Generator,
) -> int:
    """
    Determines the actual number of objects to spawn based on configuration.

    Args:
        spawn_count_config: None (spawn all objects), an integer (fixed count),
                            or [min, max] (random count within range).
        total_objects: Total number of objects in the group.
        rng: Random number generator for reproducibility.

    Returns:
        Actual number of objects to spawn (0 <= count <= total_objects).
    """
    if spawn_count_config is None:
        return total_objects

    if isinstance(spawn_count_config, (int, float)):
        count = int(spawn_count_config)
        return max(0, min(count, total_objects))

    if isinstance(spawn_count_config, (list, tuple)) and len(spawn_count_config) == 2:
        min_count, max_count = int(spawn_count_config[0]), int(spawn_count_config[1])
        # Clamp range to actual number of objects
        min_count = max(0, min(min_count, total_objects))
        max_count = max(0, min(max_count, total_objects))

        if min_count > max_count:
            min_count, max_count = max_count, min_count  # Ensure correct order

        # Randomly select count from the range
        return int(rng.integers(min_count, max_count + 1))

    # Invalid configuration — spawn all objects
    print(
        f"Warning: Invalid spawn_count format {spawn_count_config}, spawning all {total_objects} objects"
    )
    return total_objects


def _apply_random_rotation(
    base_rotation: list[float], spec: Dict[str, List[float]], rng: np.random.Generator
) -> list[float]:
    """
    Applies random rotation around specified axes (in degrees) to a base quaternion.


    Args:
        base_rotation: Base orientation as [w, x, y, z]
        spec: Dict like {"z": [0, 360], "x": [-5, 5]}
        rng: Random generator

    Returns:
        New quaternion with applied random rotation
    """
    base_quat = np.array(base_rotation, dtype=np.float64)
    q_total = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # identity

    # Apply rotations in X→Y→Z order (world space)
    for axis in ["x", "y", "z"]:
        if axis in spec:
            min_deg, max_deg = spec[axis]
            angle_rad = np.deg2rad(rng.uniform(min_deg, max_deg))
            half = angle_rad / 2.0

            if axis == "x":
                q_axis = np.array([np.cos(half), np.sin(half), 0.0, 0.0])
            elif axis == "y":
                q_axis = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
            else:  # 'z'
                q_axis = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])

            q_total = _quat_mul(q_axis, q_total)

    # Combine: random perturbation FIRST, then base rotation (world space order)
    final = _quat_mul(q_total, base_quat)
    return (final / np.linalg.norm(final)).tolist()


def _get_robot_spawn_areas(robot_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Returns robot.spawn_areas"""
    areas = robot_cfg.get("spawn_areas")

    # Backward compatibility:
    # robot: {center, size_xy, z_rotation_deg}
    if areas is None:
        return [
            {
                "name": "legacy",
                "center": robot_cfg.get("center", [0.0, 0.0, 0.0]),
                "size_xy": robot_cfg.get("size_xy", [0.0, 0.0]),
                "z_rotation_deg": robot_cfg.get("z_rotation_deg", [0.0, 0.0]),
                "weight": 1.0,
                "visible": robot_cfg.get("visible", True),
            }
        ]

    if not isinstance(areas, list) or not areas:
        raise ValueError("robot.spawn_areas must be a non-empty list")

    normalized = []
    for index, area in enumerate(areas):
        if not isinstance(area, dict):
            raise ValueError(f"robot.spawn_areas[{index}] must be an object")

        center = area.get("center")
        size_xy = area.get("size_xy")

        if not _is_position3(center):
            raise ValueError(f"robot.spawn_areas[{index}].center must be [x, y, z]")

        if (
            not isinstance(size_xy, (list, tuple))
            or len(size_xy) != 2
            or not all(isinstance(v, (int, float)) for v in size_xy)
            or size_xy[0] < 0.0
            or size_xy[1] < 0.0
        ):
            raise ValueError(
                f"robot.spawn_areas[{index}].size_xy must be [size_x, size_y]"
            )

        yaw_range = area.get("z_rotation_deg", [0.0, 0.0])
        if (
            not isinstance(yaw_range, (list, tuple))
            or len(yaw_range) != 2
            or not all(isinstance(v, (int, float)) for v in yaw_range)
        ):
            raise ValueError(
                f"robot.spawn_areas[{index}].z_rotation_deg must be [min_deg, max_deg]"
            )

        weight = float(area.get("weight", 1.0))
        if weight < 0.0:
            raise ValueError(f"robot.spawn_areas[{index}].weight must be >= 0")

        normalized.append(
            {
                "name": str(area.get("name", f"area_{index}")),
                "center": [float(v) for v in center],
                "size_xy": [float(v) for v in size_xy],
                "z_rotation_deg": [float(v) for v in yaw_range],
                "weight": weight,
                "visible": bool(area.get("visible", True)),
            }
        )

    if not any(area["weight"] > 0.0 for area in normalized):
        raise ValueError("robot.spawn_areas: at least one weight must be > 0")

    return normalized


def _generate_robot_spawn_pose(
    cfg: dict,
    rng: np.random.Generator,
) -> tuple[list[float], list[float]]:
    """Selects a spawn area by weight and a uniform pose inside it."""

    areas = _get_robot_spawn_areas(cfg)

    # 1. Choose an area according to its weight.
    weights = np.asarray(
        [float(area["weight"]) for area in areas],
        dtype=np.float64,
    )

    if np.any(weights < 0.0):
        raise ValueError("robot.spawn_areas[].weight must be >= 0")

    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        raise ValueError("robot.spawn_areas: at least one area must have weight > 0")

    probabilities = weights / total_weight
    area = areas[int(rng.choice(len(areas), p=probabilities))]

    # 2. Uniform XY selection inside the chosen area.
    cx, cy, cz = map(float, area["center"])
    sx, sy = map(float, area["size_xy"])

    if sx < 0.0 or sy < 0.0:
        raise ValueError("robot.spawn_areas[].size_xy values must be >= 0")

    pos = [
        cx + rng.uniform(-0.5 * sx, 0.5 * sx),
        cy + rng.uniform(-0.5 * sy, 0.5 * sy),
        cz,
    ]

    # 3. Uniform yaw in the chosen area's z_rotation_deg.
    # Supports both equivalent orderings:
    # [90, 135]  -> U(90, 135)
    # [0, -45]   -> U(-45, 0)
    z_rotation_deg = area["z_rotation_deg"]

    if not isinstance(z_rotation_deg, (list, tuple)) or len(z_rotation_deg) != 2:
        raise ValueError(
            "robot.spawn_areas[].z_rotation_deg must be [start_deg, end_deg]"
        )

    yaw_start_deg = float(z_rotation_deg[0])
    yaw_end_deg = float(z_rotation_deg[1])

    yaw_deg = rng.uniform(
        min(yaw_start_deg, yaw_end_deg),
        max(yaw_start_deg, yaw_end_deg),
    )

    yaw_rad = np.deg2rad(yaw_deg)
    half_yaw = 0.5 * yaw_rad

    rot = [
        float(np.cos(half_yaw)),
        0.0,
        0.0,
        float(np.sin(half_yaw)),
    ]

    return pos, rot


def _is_position3(value: Any) -> bool:
    """True only for one numeric [x, y, z] position."""
    return (
        isinstance(value, (list, tuple, np.ndarray))
        and len(value) == 3
        and all(isinstance(x, (int, float, np.integer, np.floating)) for x in value)
    )


def _resolve_group_base_position(
    group_name: str,
    group_templates: Dict[str, Dict[str, Any]],
    object_metadata: Dict[str, Dict[str, Any]],
    rng: np.random.Generator,
    resolved_positions: Dict[str, np.ndarray],
    resolving_groups: set[str],
) -> np.ndarray:
    """
    Resolves spawn-group base_position to one [x, y, z].

    Supported JSON formats:
      [x, y, z]
      [[x1, y1, z1], [x2, y2, z2], ...]
      "another_spawn_group"
      ["spawn_group_a", "spawn_group_b", ...]
    """
    if group_name in resolved_positions:
        return resolved_positions[group_name].copy()

    if group_name not in group_templates:
        print(
            f"Warning: spawn group '{group_name}' not found, " "using PARKING_POSITION"
        )
        return np.array(PARKING_POSITION, dtype=float)

    if group_name in resolving_groups:
        cycle = " -> ".join([*sorted(resolving_groups), group_name])
        raise ValueError(f"Cyclic base_position group reference: {cycle}")

    resolving_groups.add(group_name)

    tmpl = group_templates[group_name]
    raw_base_position = tmpl["base_position"]

    # Case 1:
    # "base_position": [0.55, -0.6, 0.96]
    if _is_position3(raw_base_position):
        base = np.array(raw_base_position, dtype=float)

    # Case 2:
    # "base_position": "plate_on_table"
    elif isinstance(raw_base_position, str):
        target = raw_base_position

        if target in group_templates:
            base = _resolve_group_base_position(
                group_name=target,
                group_templates=group_templates,
                object_metadata=object_metadata,
                rng=rng,
                resolved_positions=resolved_positions,
                resolving_groups=resolving_groups,
            )

            # Preserve your old behavior:
            # a reference to a grid group is placed on its surface.
            if group_templates[target].get("type") == "grid":
                base = base.copy()
                base[2] += group_templates[target].get("z_offset", 0.0)

        # Preserve old support for object references.
        elif target in object_metadata:
            obj_pos = object_metadata[target].get("position")

            if _is_position3(obj_pos):
                base = np.array(obj_pos, dtype=float)
            else:
                print(
                    f"Warning: object '{target}' has no valid absolute position, "
                    "using PARKING_POSITION"
                )
                base = np.array(PARKING_POSITION, dtype=float)

        else:
            print(
                f"Warning: base_position target '{target}' not found, "
                "using PARKING_POSITION"
            )
            base = np.array(PARKING_POSITION, dtype=float)

    # Case 3 and 4:
    # [[x1, y1, z1], [x2, y2, z2], ...]
    # OR
    # ["plate_left", "plate_center", "plate_right"]
    elif isinstance(raw_base_position, (list, tuple)) and len(raw_base_position) > 0:
        selected = raw_base_position[int(rng.integers(0, len(raw_base_position)))]

        # Selected item is one coordinate.
        if _is_position3(selected):
            base = np.array(selected, dtype=float)

        # Selected item is a group name.
        elif isinstance(selected, str):
            if selected not in group_templates:
                print(
                    f"Warning: base_position target group '{selected}' not found, "
                    "using PARKING_POSITION"
                )
                base = np.array(PARKING_POSITION, dtype=float)
            else:
                base = _resolve_group_base_position(
                    group_name=selected,
                    group_templates=group_templates,
                    object_metadata=object_metadata,
                    rng=rng,
                    resolved_positions=resolved_positions,
                    resolving_groups=resolving_groups,
                )

                # Preserve your old string-reference semantics.
                if group_templates[selected].get("type") == "grid":
                    base = base.copy()
                    base[2] += group_templates[selected].get("z_offset", 0.0)

        else:
            print(
                f"Warning: invalid base_position candidate {selected!r} "
                f"for group '{group_name}', using PARKING_POSITION"
            )
            base = np.array(PARKING_POSITION, dtype=float)

    else:
        print(
            f"Warning: invalid base_position for '{group_name}', "
            "using PARKING_POSITION"
        )
        base = np.array(PARKING_POSITION, dtype=float)

    resolving_groups.remove(group_name)
    resolved_positions[group_name] = base.copy()
    return base.copy()


def _resolve_parent_object_poses(
    object_metadata: Dict[str, Dict[str, Any]],
    poses: Dict[str, Tuple[list[float], list[float]]],
    active_objects: Dict[str, bool],
) -> None:
    """Resolve parent_object links for already generated object poses.

    For an object with parent_object, its JSON `position` is interpreted as
    a local translation in the physical root frame of its parent.

    The function mutates:
    - poses: replaces child world positions;
    - active_objects: disables children whose parent is unavailable.

    The child's `rotation` retains its existing semantics. Only `position`
    becomes parent-local.
    """

    resolving: set[str] = set()
    resolved: set[str] = set()

    def resolve(name: str) -> None:
        if name in resolved:
            return

        meta = object_metadata[name]
        parent_name = meta.get("parent_object")

        if parent_name is None:
            resolved.add(name)
            return

        if parent_name not in object_metadata:
            raise ValueError(
                f"Object '{name}' references unknown parent_object " f"'{parent_name}'"
            )

        if name in resolving:
            cycle = " -> ".join([*sorted(resolving), name])
            raise ValueError(f"Cyclic parent_object reference: {cycle}")

        resolving.add(name)
        resolve(parent_name)
        resolving.remove(name)

        # If parent was excluded by spawn_count or its group could not find
        # a collision-free pose, then its child must be inactive too.
        if not active_objects.get(parent_name, True) or parent_name not in poses:
            active_objects[name] = False
            poses.pop(name, None)
            resolved.add(name)
            return

        # Do not reactivate a child disabled by its own spawn_count.
        if not active_objects.get(name, True) or name not in poses:
            resolved.add(name)
            return

        local_position = meta.get("position")
        if not _is_position3(local_position):
            raise ValueError(
                f"Object '{name}' with parent_object='{parent_name}' "
                "must define position as [x, y, z]"
            )

        parent_position, parent_user_rotation = poses[parent_name]

        # The pose generator stores user rotation; get_asset_cfgs() and
        # reset_objects() later apply USD local rotation to form the actual
        # physical root quaternion. Use the same frame for the local offset.
        parent_root_rotation = _quat_mul(
            np.asarray(parent_user_rotation, dtype=np.float64),
            np.asarray(
                object_metadata[parent_name]["usd_local_rotation"],
                dtype=np.float64,
            ),
        )
        parent_root_rotation /= np.linalg.norm(parent_root_rotation)

        world_position = np.asarray(parent_position, dtype=np.float64) + _quat_rotate(
            parent_root_rotation,
            np.asarray(local_position, dtype=np.float64),
        )

        # parent_object changes position semantics only.
        _, child_rotation = poses[name]
        poses[name] = (world_position.tolist(), child_rotation)

        resolved.add(name)

    for name in sorted(object_metadata.keys()):
        resolve(name)


class SlotsLayout:
    """Places objects into pre-defined absolute XYZ slots."""

    def __init__(
        self,
        positions: List[List[float]],
        object_names: List[str],
        object_metadata: Dict[str, Dict[str, Any]],
        rng: np.random.Generator,
        *,
        min_slot_gap: int = 1,
        pick_slots: Optional[List[int]] = None,
        pick_clearance: int = 0,
        pick_origin_xy: Optional[List[float]] = None,
    ):
        if not positions:
            raise ValueError("slots layout requires non-empty positions")

        self.positions = [
            [float(pos[0]), float(pos[1]), float(pos[2])] for pos in positions
        ]
        self.object_metadata = object_metadata
        self.assignment: Dict[str, int] = {}

        slot_count = len(self.positions)
        all_slots = list(range(slot_count))
        pick_slots = pick_slots or all_slots
        pick_slots = [
            int(idx) for idx in pick_slots if 0 <= int(idx) < slot_count
        ] or all_slots

        pick_names = [
            name
            for name in object_names
            if object_metadata[name].get("pick_target", False)
        ]
        filler_names = [
            name
            for name in object_names
            if not object_metadata[name].get("pick_target", False)
        ]

        occupied: set[int] = set()
        reserved: set[int] = set()

        def choose(pool: List[int]) -> Optional[int]:
            if not pool:
                return None

            slot_idx = int(pool[int(rng.integers(0, len(pool)))])
            occupied.add(slot_idx)
            return slot_idx

        # 1. First place the pick-target.
        for name in pick_names:
            pool = [idx for idx in pick_slots if idx not in occupied]
            if not pool:
                pool = [idx for idx in all_slots if idx not in occupied]

            slot_idx = choose(pool)
            if slot_idx is None:
                break

            self.assignment[name] = slot_idx

            for neighbor in range(
                slot_idx - max(0, pick_clearance),
                slot_idx + max(0, pick_clearance) + 1,
            ):
                if neighbor != slot_idx and 0 <= neighbor < slot_count:
                    reserved.add(neighbor)

        # 2. Filler objects take the remaining valid slots.
        for name in filler_names:
            pool = [
                idx
                for idx in all_slots
                if idx not in occupied
                and idx not in reserved
                and all(
                    abs(idx - occupied_slot) >= max(1, min_slot_gap)
                    for occupied_slot in occupied
                )
            ]

            slot_idx = choose(pool)
            if slot_idx is not None:
                self.assignment[name] = slot_idx

        # 3. Make the pick-target the closest to the robot among occupied positions.
        if pick_names and self.assignment:
            pick_name = pick_names[0]

            def dist2(slot_idx: int) -> float:
                if pick_origin_xy is None:
                    return float(slot_idx)

                pos = self.positions[slot_idx]
                dx = pos[0] - float(pick_origin_xy[0])
                dy = pos[1] - float(pick_origin_xy[1])
                return dx * dx + dy * dy

            nearest_slot = min(self.assignment.values(), key=dist2)
            nearest_name = next(
                name
                for name, slot_idx in self.assignment.items()
                if slot_idx == nearest_slot
            )

            if nearest_name != pick_name:
                self.assignment[pick_name], self.assignment[nearest_name] = (
                    self.assignment[nearest_name],
                    self.assignment[pick_name],
                )

    def has_object(self, object_name: str) -> bool:
        return object_name in self.assignment

    def get_position(
        self,
        object_name: str,
        offset: Optional[List[float]] = None,
    ) -> List[float]:
        slot_idx = self.assignment[object_name]
        pos = np.array(self.positions[slot_idx], dtype=float)

        if offset is not None:
            pos += np.asarray(offset, dtype=float)

        return pos.tolist()


def _generate_object_poses(
    object_metadata: Dict[str, Dict[str, Any]],
    group_templates: Dict[str, Dict[str, Any]],
    group_objects: Dict[str, List[str]],
    seed: Optional[int] = None,
) -> Tuple[Dict[str, Tuple[list[float], list[float]]], Dict[str, bool]]:
    """Unified pose generation logic — used both at startup and on reset.

    This function generates world-space poses that respect:
    - User-specified rotations
    - USD file's local orientation (handled externally via metadata)
    - Group-based layout constraints
    - Optional spawn_count limitation per group (random subset selection)

    Args:
        object_metadata: Precomputed per-object data (including sizes).
        group_templates: Group definitions from JSON.
        seed: Random seed for reproducibility (None = random).

    Returns:
        Tuple of (poses_dict, active_objects_dict)
        active_objects_dict: {object_name: should_spawn} for post-reset visibility/collision control
    """
    rng = np.random.default_rng(seed)

    # Allow relative references in base_position (groups OR objects)
    # resolved_positions = {}

    # # Pass 1: Absolute Group Positions
    # for name, tmpl in group_templates.items():
    #     bp = tmpl["base_position"]
    #     if isinstance(bp, (list, tuple)) and len(bp) == 3:
    #         resolved_positions[name] = np.array(bp, dtype=float)
    #     elif isinstance(bp, np.ndarray) and bp.shape == (3,):
    #         resolved_positions[name] = bp.copy()

    # # Pass 2: Relative Links
    # for name, tmpl in group_templates.items():
    #     bp = tmpl["base_position"]
    #     if isinstance(bp, str):
    #         target = bp

    #         # Case 1: Link to another group
    #         if target in group_templates and target in resolved_positions:
    #             base = resolved_positions[target].copy()
    #             # For "grid" type groups, add their z_offset to the base position
    #             if group_templates[target].get("type") == "grid":
    #                 base[2] += group_templates[target].get("z_offset", 0.0)
    #             resolved_positions[name] = base
    #             tmpl["base_position"] = base

    #         # Case 2: Reference to an object with an absolute position
    #         elif target in object_metadata:
    #             obj_pos = object_metadata[target]["position"]
    #             if obj_pos is not None and len(obj_pos) == 3:
    #                 base = np.array(obj_pos, dtype=float).copy()
    #                 resolved_positions[name] = base
    #                 tmpl["base_position"] = base
    #             else:
    #                 print(
    #                     f"Warning: object '{target}' has no valid absolute position (likely in a spawn_group), using PARKING_POSITION"
    #                 )
    #                 base = np.array(PARKING_POSITION, dtype=float)
    #                 resolved_positions[name] = base
    #                 tmpl["base_position"] = base

    #         # Case 3: Not found
    #         else:
    #             print(
    #                 f"Warning: base_position target '{target}' not found (not a group or object with absolute position), using PARKING_POSITION"
    #             )
    #             base = np.array(PARKING_POSITION, dtype=float)
    #             resolved_positions[name] = base
    #             tmpl["base_position"] = base

    # # Format unification
    # for name, tmpl in group_templates.items():
    #     bp = tmpl["base_position"]
    #     if not (isinstance(bp, np.ndarray) and bp.shape == (3,)):
    #         if isinstance(bp, (list, tuple)) and len(bp) == 3:
    #             tmpl["base_position"] = np.array(bp, dtype=float)
    #         else:
    #             print(
    #                 f"Warning: invalid base_position for '{name}', using PARKING_POSITION"
    #             )
    #             tmpl["base_position"] = np.array(PARKING_POSITION, dtype=float)

    # For each group, one base_position is chosen per current spawn/reset.
    # Key: group name, value: already resolved [x, y, z].
    resolved_positions: Dict[str, np.ndarray] = {}

    for group_name, group_cfg in group_templates.items():
        resolved_base_position = _resolve_group_base_position(
            group_name=group_name,
            group_templates=group_templates,
            object_metadata=object_metadata,
            rng=rng,
            resolved_positions=resolved_positions,
            resolving_groups=set(),
        )

        # Do not mutate group_cfg["base_position"]:
        # it must keep the original JSON: list of positions / list of names.
        #
        # This field is only needed as the runtime result of the current spawn/reset,
        # in particular for update_group_visuals().
        group_cfg["_resolved_base_position"] = resolved_base_position

    active_objects: Dict[str, bool] = {name: True for name in object_metadata.keys()}

    for grp_name, grp in group_templates.items():
        if grp_name not in group_objects:
            continue

        objects_in_group = group_objects[grp_name]
        total_count = len(objects_in_group)
        spawn_count_cfg = grp.get("spawn_count")
        actual_count = _determine_spawn_count(spawn_count_cfg, total_count, rng)

        if actual_count >= total_count:
            selected = set(objects_in_group)
        elif actual_count <= 0:
            selected = set()
        else:
            selected = set(
                rng.choice(objects_in_group, size=actual_count, replace=False).tolist()
            )

        for obj_name in objects_in_group:
            active_objects[obj_name] = obj_name in selected

    # Reconstruct fresh group states
    group_states = {}
    for name, tmpl in group_templates.items():
        if tmpl["type"] == "stack":
            axis_index = {"x": 0, "y": 1, "z": 2}.get(tmpl["stack_axis"].lower(), 2)
            group_states[name] = {
                "type": "stack",
                # "base_position": tmpl["base_position"].copy(),
                "base_position": tmpl["_resolved_base_position"].copy(),
                "z_offset": tmpl["z_offset"],
                "axis_index": axis_index,
                "spacing": tmpl["spacing"],
                "next_offset": 0.0,
                "rotation": tmpl["rotation"],
            }
        elif tmpl["type"] == "grid":
            group_states[name] = {
                "type": "grid",
                # "base_position": tmpl["base_position"].copy(),
                "base_position": tmpl["_resolved_base_position"].copy(),
                "half_extents": tmpl["size"] / 2.0,
                "z_offset": tmpl["z_offset"],
                "min_spacing": tmpl["min_spacing"],
                "packing_mode": tmpl["packing_mode"],
                "rotation": tmpl["rotation"],
                "placed_positions": [],
                "current_row_y": None,
                "current_x": None,
                "max_obj_height_in_row": 0.0,
            }
        elif tmpl["type"] == "slots":
            active_names = [
                obj_name
                for obj_name in group_objects.get(name, [])
                if active_objects.get(obj_name, True)
            ]

            layout = SlotsLayout(
                positions=tmpl["positions"],
                object_names=active_names,
                object_metadata=object_metadata,
                rng=rng,
                min_slot_gap=tmpl.get("min_slot_gap", 1),
                pick_slots=tmpl.get("pick_slots"),
                pick_clearance=tmpl.get("pick_clearance", 0),
                pick_origin_xy=tmpl.get("pick_origin_xy"),
            )

            for obj_name in group_objects.get(name, []):
                if not layout.has_object(obj_name):
                    active_objects[obj_name] = False

            group_states[name] = {
                "type": "slots",
                "layout": layout,
            }

    poses: Dict[str, Tuple[list[float], list[float]]] = {}

    # Process in sorted order for determinism
    for name in sorted(object_metadata.keys()):
        meta = object_metadata[name]
        size = meta["size"]
        radius_x, radius_y = size[0] / 2.0, size[1] / 2.0

        # Skip position generation for inactive group objects
        if meta["spawn_group"] is not None and not active_objects.get(name, True):
            continue

        obj_rot = meta["rotation"]

        if meta["spawn_group"] is not None:
            grp = group_states[meta["spawn_group"]]

            if grp["type"] == "stack":
                pos = grp["base_position"].copy()
                pos[2] += grp.get("z_offset", 0.0)
                pos[grp["axis_index"]] += grp["next_offset"]
                if meta["offset_in_group"] is not None:
                    pos += np.array(meta["offset_in_group"])
                spacing = (
                    meta["spacing_override"]
                    if meta["spacing_override"] is not None
                    else grp["spacing"]
                )
                grp["next_offset"] += spacing
                poses[name] = (pos.tolist(), obj_rot)

            elif grp["type"] == "grid":
                area_min_x = grp["base_position"][0] - grp["half_extents"][0]
                area_max_x = grp["base_position"][0] + grp["half_extents"][0]
                area_min_y = grp["base_position"][1] - grp["half_extents"][1]
                area_max_y = grp["base_position"][1] + grp["half_extents"][1]

                if grp["packing_mode"] == "random":
                    if meta.get("random_rotation") is not None:
                        obj_rot = _apply_random_rotation(
                            obj_rot,
                            meta["random_rotation"],
                            rng,
                        )

                        if meta.get("object_kind") == "guide":
                            # Guide has no USD asset. Its size is already its
                            # local placement bbox.
                            size = np.asarray(meta["size"], dtype=float)
                        else:
                            final_rot_for_aabb = _quat_mul(
                                np.asarray(obj_rot, dtype=float),
                                meta["usd_local_rotation"],
                            )
                            final_rot_for_aabb /= np.linalg.norm(final_rot_for_aabb)

                            size = _compute_oriented_aabb_size(
                                meta["usd_path"],
                                final_rot_for_aabb.tolist(),
                                meta["scale"],
                            )

                        radius_x, radius_y = size[0] / 2.0, size[1] / 2.0

                    placed = False
                    for _ in range(100):
                        ox = rng.uniform(
                            -grp["half_extents"][0], grp["half_extents"][0]
                        )
                        oy = rng.uniform(
                            -grp["half_extents"][1], grp["half_extents"][1]
                        )
                        cx = grp["base_position"][0] + ox
                        cy = grp["base_position"][1] + oy
                        cz = grp["base_position"][2] + grp["z_offset"]

                        overlap = False
                        for px, py, r_x, r_y in grp["placed_positions"]:
                            dx = abs(cx - px)
                            dy = abs(cy - py)
                            if dx < (radius_x + r_x + grp["min_spacing"]) and dy < (
                                radius_y + r_y + grp["min_spacing"]
                            ):
                                overlap = True
                                break

                        if not overlap:
                            poses[name] = ([cx, cy, cz], obj_rot)
                            grp["placed_positions"].append((cx, cy, radius_x, radius_y))
                            placed = True
                            break

                    if not placed:
                        # poses[name] = (
                        #     [
                        #         grp["base_position"][0],
                        #         grp["base_position"][1],
                        #         grp["base_position"][2] + grp["z_offset"],
                        #     ],
                        #     obj_rot,
                        # )
                        active_objects[name] = False

                elif grp["packing_mode"] == "grid":
                    if grp["current_row_y"] is None:
                        grp["current_row_y"] = area_max_y - radius_y
                        grp["current_x"] = area_min_x + radius_x
                        grp["max_obj_height_in_row"] = size[1]

                    right_edge = grp["current_x"] + radius_x
                    if right_edge > area_max_x + 1e-6:
                        # row_bottom = (
                        #     grp["current_row_y"] - grp["max_obj_height_in_row"] / 2.0
                        # )
                        # new_row_top = row_bottom - radius_y
                        row_bottom = (
                            grp["current_row_y"] - grp["max_obj_height_in_row"] / 2.0
                        )
                        new_row_top = row_bottom - grp["min_spacing"] - radius_y
                        if new_row_top - radius_y < area_min_y:
                            poses[name] = (
                                [
                                    grp["base_position"][0],
                                    grp["base_position"][1],
                                    grp["base_position"][2] + grp["z_offset"],
                                ],
                                obj_rot,
                            )
                        else:
                            grp["current_row_y"] = new_row_top
                            grp["current_x"] = area_min_x + radius_x
                            grp["max_obj_height_in_row"] = size[1]
                            poses[name] = (
                                [
                                    grp["current_x"],
                                    grp["current_row_y"],
                                    grp["base_position"][2] + grp["z_offset"],
                                ],
                                obj_rot,
                            )
                    else:
                        if size[1] > grp["max_obj_height_in_row"]:
                            grp["max_obj_height_in_row"] = size[1]
                        poses[name] = (
                            [
                                grp["current_x"],
                                grp["current_row_y"],
                                grp["base_position"][2] + grp["z_offset"],
                            ],
                            obj_rot,
                        )

                    grp["current_x"] += size[0] + grp["min_spacing"]

            elif grp["type"] == "slots":
                layout: SlotsLayout = grp["layout"]

                if not layout.has_object(name):
                    active_objects[name] = False
                    continue

                poses[name] = (
                    layout.get_position(
                        name,
                        offset=meta.get("offset_in_group"),
                    ),
                    obj_rot,
                )

        else:
            # Explicit placement
            poses[name] = (meta["position"], obj_rot)

    _resolve_parent_object_poses(
        object_metadata=object_metadata,
        poses=poses,
        active_objects=active_objects,
    )

    return poses, active_objects


def _json_tuple(d: dict, key: str):
    value = d.get(key)
    return None if value is None else tuple(value)


class SceneSpawner:
    """Manages scene configuration with consistent initialization and reset."""

    def __init__(self, json_path: str, env_regex_ns: str = "{ENV_REGEX_NS}"):
        """
        Create SceneSpawner and read the initial scene description from JSON.

        The constructor initializes persistent runtime caches once and then parses
        the JSON scene description into metadata dictionaries used later by spawn
        and reset methods.

        Args:
            json_path: Path to the JSON configuration file.
            env_regex_ns: Namespace template for prim paths.
        """

        self.json_path = json_path
        self.env_regex_ns = env_regex_ns

        self._initial_poses: Optional[Dict[str, Tuple[list[float], list[float]]]] = None
        self._initial_joint_pos: Dict[str, Dict[str, float]] = {}
        self.last_active_objects: Dict[str, bool] = {}
        self._group_objects: Dict[str, List[str]] = {}
        self._prim_paths_cache: Dict[str, List[str]] = {}
        self._group_visual_prim_paths: Dict[str, List[str]] = {}
        self._anchor_physx_views: Dict[str, Any] = {}
        self._reload_pose_overrides: set[str] = set()
        self._robot_spawn_visual_prim_paths: Dict[str, List[str]] = {}

        (
            self.config,
            self.robot_cfg,
            self.object_metadata,
            self.group_templates,
            self.light_metadata,
            self.light_visual_metadata,
            parsed_group_objects,
            self.viewer_metadata,
        ) = self._read_scene_description()

        self._group_objects.update(parsed_group_objects)

    def _read_scene_description(self):
        """
        Read self.json_path and build fresh scene description dictionaries.

        This method only parses JSON and prepares metadata.
        It does not touch runtime state such as:
        - _initial_poses
        - _initial_joint_pos
        - _prim_paths_cache
        - last_active_objects

        Returns:
            tuple:
                (
                    config,
                    robot_cfg,
                    object_metadata,
                    group_templates,
                    light_metadata,
                    light_visual_metadata,
                    group_objects,
                )
        """
        with open(self.json_path, "r") as f:
            config = json.load(f)

        # Foolproof
        identity_quat = [1.0, 0.0, 0.0, 0.0]

        for obj in config.get("objects", []):
            if obj.get("rotation") == [0, 0, 0, 0]:
                obj["rotation"] = identity_quat.copy()

        for group_cfg in config.get("spawn_groups", {}).values():
            if group_cfg.get("rotation") == [0, 0, 0, 0]:
                group_cfg["rotation"] = identity_quat.copy()

        ###

        robot_cfg = config.get("robot")
        object_metadata: Dict[str, Dict[str, Any]] = {}
        group_templates: Dict[str, Dict[str, Any]] = {}
        light_metadata: Dict[str, Dict[str, Any]] = {}
        light_visual_metadata: Dict[str, Dict[str, Any]] = {}
        group_objects: Dict[str, List[str]] = {}

        if "spawn_groups" in config:
            for name, cfg in config["spawn_groups"].items():
                visible = cfg["visible"] if "visible" in cfg else None
                layout = cfg.get("layout", "stack")
                base_pos_raw = cfg.get("base_position", [0.0, 0.0, 0.0])

                if layout == "stack":
                    group_templates[name] = {
                        "type": "stack",
                        "base_position": base_pos_raw,
                        "anchor_prim": cfg.get("anchor_prim"),
                        "stack_axis": cfg.get("stack_axis", "z"),
                        "spacing": float(cfg.get("spacing", 0.1)),
                        "rotation": cfg.get("rotation", [1.0, 0.0, 0.0, 0.0]),
                        "spawn_count": cfg.get("spawn_count", None),
                        "z_offset": float(cfg.get("z_offset", 0.0)),
                        "visible": visible,
                    }
                elif layout == "grid":
                    group_templates[name] = {
                        "type": "grid",
                        "base_position": base_pos_raw,
                        "anchor_prim": cfg.get("anchor_prim"),
                        "size": np.array(cfg["size"], dtype=float),
                        "z_offset": float(cfg.get("z_offset", 0.01)),
                        "min_spacing": float(cfg.get("min_spacing", 0.005)),
                        "packing_mode": cfg.get("packing_mode", "random"),
                        "rotation": cfg.get("rotation", [1.0, 0.0, 0.0, 0.0]),
                        "spawn_count": cfg.get("spawn_count", None),
                        "visible": visible,
                    }
                elif layout == "slots":
                    positions = cfg.get("positions", [])

                    if not isinstance(positions, list) or not positions:
                        raise ValueError(
                            f"slots group '{name}' requires non-empty 'positions'"
                        )

                    group_templates[name] = {
                        "type": "slots",
                        "base_position": positions[0],
                        "xy_jitter": float(cfg.get("xy_jitter", 0.0) or 0.0),
                        "positions": positions,
                        "rotation": cfg.get("rotation", [1.0, 0.0, 0.0, 0.0]),
                        "spawn_count": cfg.get("spawn_count"),
                        "min_slot_gap": int(cfg.get("min_slot_gap", 1)),
                        "pick_clearance": int(cfg.get("pick_clearance", 0)),
                        "pick_slots": cfg.get("pick_slots"),
                        "pick_origin_xy": cfg.get("pick_origin_xy"),
                        "visible": visible,
                    }
                else:
                    raise ValueError(f"Unknown layout: {layout}")

        if "lights" in config:
            lights_cfg = config["lights"]

            if isinstance(lights_cfg, dict):
                lights_iter = lights_cfg.items()
            elif isinstance(lights_cfg, list):
                lights_iter = []
                for light in lights_cfg:
                    if "name" not in light:
                        raise ValueError(
                            "Each light in 'lights' list must have a 'name'"
                        )
                    lights_iter.append((light["name"], light))
            else:
                raise ValueError("'lights' must be either a dict or a list")

            for light_name, light in lights_iter:
                prim_path = f"/World/{light_name}"
                light_metadata[light_name] = {
                    "prim_path": prim_path,
                    "type": light["type"],
                    "position": _json_tuple(light, "position"),
                    "rotation": _json_tuple(light, "rotation"),
                    "intensity": light.get("intensity"),
                    "color": _json_tuple(light, "color"),
                    "exposure": light.get("exposure"),
                    "radius": light.get("radius"),
                    "length": light.get("length"),
                    "angle": light.get("angle"),
                    "enable_color_temperature": light.get("enable_color_temperature"),
                    "color_temperature": light.get("color_temperature"),
                    "normalize": light.get("normalize"),
                    "texture_file": light.get("texture_file"),
                    "texture_format": light.get("texture_format"),
                    "visible_in_primary_ray": light.get("visible_in_primary_ray"),
                }

        for obj in config["objects"]:
            count = obj.get("instance_count", 1)
            base_name = obj["name"]

            if "usd_path" not in obj:
                for i in range(count):
                    name = f"{base_name}_{i}" if count > 1 else base_name

                    if "spawn_group" in obj:
                        group_rot = group_templates[obj["spawn_group"]]["rotation"]
                        obj_rot = obj.get("rotation", group_rot)
                    else:
                        obj_rot = obj.get("rotation", [1.0, 0.0, 0.0, 0.0])

                    guide_size = obj.get("size", 0.1)
                    if isinstance(guide_size, (int, float)):
                        guide_radius = float(guide_size)
                        guide_bbox_size = [
                            guide_radius * 2.0,
                            guide_radius * 2.0,
                            guide_radius * 2.0,
                        ]
                    elif _is_position3(guide_size):
                        guide_radius = None
                        guide_bbox_size = [
                            float(guide_size[0]),
                            float(guide_size[1]),
                            float(guide_size[2]),
                        ]
                    else:
                        raise ValueError(
                            f"Guide '{name}': size must be a radius or [size_x, size_y, size_z]"
                        )

                    object_metadata[name] = {
                        "object_kind": "guide",
                        "parent_object": obj.get("parent_object"),
                        "spawn_group": obj.get("spawn_group"),
                        "position": obj.get("position"),
                        "rotation": obj_rot,
                        "random_rotation": obj.get("random_rotation"),
                        "scale": [1.0, 1.0, 1.0],
                        "fixed": obj.get("fixed", True),
                        "mass": 0.0,
                        "physics_material": None,
                        "offset_in_group": obj.get("offset_in_group"),
                        "spacing_override": obj.get("spacing_override"),
                        "size": guide_bbox_size,
                        "guide_radius": guide_radius,
                        "color": obj.get("color", "red"),
                        "visible": bool(obj.get("visible", True)),
                        "usd_local_rotation": np.array([1.0, 0.0, 0.0, 0.0]),
                        "articulated": False,
                        "collision_groups": [],
                        "scene_static": False,
                        "pick_target": bool(obj.get("pick_target", False)),
                    }

                    grp_name = obj.get("spawn_group")
                    if grp_name is not None:
                        group_objects.setdefault(grp_name, []).append(name)

                continue

            usd_path = str(Path(obj["usd_path"]).resolve())
            scale = obj.get("scale", [1.0, 1.0, 1.0])

            is_articulated = obj.get("articulated", False)
            collision_groups = obj.get("collision_groups", [])

            _, local_rot = _compute_usd_local_transform(usd_path)

            for i in range(count):
                name = f"{base_name}_{i}" if count > 1 else base_name

                if "spawn_group" in obj:
                    group_rot = group_templates[obj["spawn_group"]]["rotation"]
                    obj_rot = obj.get("rotation", group_rot)
                else:
                    obj_rot = obj["rotation"]

                # size = _compute_oriented_aabb_size(usd_path, obj_rot, scale)
                final_rot_for_aabb = _quat_mul(
                    np.asarray(obj_rot, dtype=float),
                    local_rot,
                )
                final_rot_for_aabb /= np.linalg.norm(final_rot_for_aabb)

                size = _compute_oriented_aabb_size(
                    usd_path,
                    final_rot_for_aabb.tolist(),
                    scale,
                )

                joint_pos = obj.get("joint_pos", {})

                if not isinstance(joint_pos, dict):
                    raise ValueError(
                        f"Object '{name}': joint_pos must be a dictionary "
                        f"of {{joint_name: position}}"
                    )

                joint_pos = {
                    str(joint_name): float(joint_value)
                    for joint_name, joint_value in joint_pos.items()
                }

                object_metadata[name] = {
                    "object_kind": "asset",
                    "usd_path": usd_path,
                    "parent_object": obj.get("parent_object"),
                    "spawn_group": obj.get("spawn_group"),
                    "position": obj.get("position"),
                    "rotation": obj_rot,
                    "random_rotation": obj.get("random_rotation"),
                    "scale": scale,
                    "fixed": obj["fixed"],
                    "mass": obj.get("mass", 0.1),
                    "physics_material": obj.get("physics_material"),
                    "offset_in_group": obj.get("offset_in_group"),
                    "spacing_override": obj.get("spacing_override"),
                    "size": size,
                    "usd_local_rotation": local_rot,
                    "articulated": is_articulated,
                    "joint_pos": joint_pos,
                    "collision_groups": collision_groups,
                    "scene_static": bool(obj.get("scene_static", False)),
                    "pick_target": bool(obj.get("pick_target", False)),
                }

                grp_name = obj.get("spawn_group")
                if grp_name is not None:
                    group_objects.setdefault(grp_name, []).append(name)

        viewer_metadata = self._get_viewer_metadata(config)
        return (
            config,
            robot_cfg,
            object_metadata,
            group_templates,
            light_metadata,
            light_visual_metadata,
            group_objects,
            viewer_metadata,
        )

    def reload_scene(self):
        """
        Re-read self.json_path and update only runtime-safe scene parameters.

        What this method does:
        - reloads JSON
        - updates robot_cfg
        - updates group_templates
        - updates placement-related fields of already existing objects

        What this method does not do:
        - does not create new objects
        - does not remove existing objects
        - does not replace USD assets
        - does not reset runtime caches
        - does not apply changes to the simulator by itself

        Notes:
            - New object names from JSON are ignored.
            - Missing old object names are kept.
            - Changes for fixed or scene_static objects are ignored, because the current
            reset pipeline does not reposition them.
            - Lighting changes are ignored in the current runtime workflow.
            - After calling reload_scene(), call the existing reset path to apply the
            new layout to the simulation.
        """
        (
            new_config,
            new_robot_cfg,
            new_object_metadata,
            new_group_templates,
            new_light_metadata,
            new_light_visual_metadata,
            _new_group_objects,
            self.viewer_metadata,
        ) = self._read_scene_description()

        allowed_object_keys = (
            "position",
            "rotation",
            "random_rotation",
            "parent_object",
            "spawn_group",
            "offset_in_group",
            "spacing_override",
            "size",
            "guide_radius",
            "color",
            "visible",
        )

        old_names = set(self.object_metadata.keys())
        new_names = set(new_object_metadata.keys())

        for name in sorted(new_names - old_names):
            print(f"[reload_scene] WARNING: new object '{name}' ignored")

        for name in sorted(old_names - new_names):
            print(
                f"[reload_scene] WARNING: object '{name}' is missing in JSON, keeping old one"
            )

        self.config = new_config
        self.robot_cfg = new_robot_cfg

        for name in sorted(old_names & new_names):
            old_meta = self.object_metadata[name]
            new_meta = new_object_metadata[name]

            position_or_rotation_changed = old_meta.get("position") != new_meta.get(
                "position"
            ) or old_meta.get("rotation") != new_meta.get("rotation")

            if old_meta.get("object_kind") != new_meta.get("object_kind"):
                print(
                    f"[reload_scene] WARNING: object '{name}' changed kind, keeping old one"
                )
                continue

            if old_meta.get("scene_static", False):
                continue

            if old_meta.get("object_kind") != "guide" and old_meta.get("fixed", False):
                continue

            old_group = old_meta.get("spawn_group")
            new_group = new_meta.get("spawn_group")
            if new_group is not None and new_group not in new_group_templates:
                print(
                    f"[reload_scene] WARNING: object '{name}' refers to missing group "
                    f"'{new_group}', keeping old group '{old_group}'"
                )
                continue

            for key in allowed_object_keys:
                old_meta[key] = new_meta.get(key)

            if position_or_rotation_changed:
                self._reload_pose_overrides.add(name)

        self.group_templates.clear()
        self.group_templates.update(new_group_templates)
        self._anchor_physx_views.clear()

        rebuilt_group_objects: Dict[str, List[str]] = {}
        for obj_name, meta in self.object_metadata.items():
            grp_name = meta.get("spawn_group")
            if grp_name is not None and grp_name in self.group_templates:
                rebuilt_group_objects.setdefault(grp_name, []).append(obj_name)

        self._group_objects.clear()
        self._group_objects.update(rebuilt_group_objects)

        if new_light_metadata != self.light_metadata:
            print("[reload_scene] WARNING: lighting changes ignored")
        if new_light_visual_metadata != self.light_visual_metadata:
            print("[reload_scene] WARNING: light visual changes ignored")

        self._rt_stage = None
        # self._get_viewer_metadata(self.config)

    def _get_group_anchor_prim(
        self,
        meta: Dict[str, Any],
    ) -> Optional[str]:
        group_name = meta.get("spawn_group")
        if group_name is None:
            return None

        group_cfg = self.group_templates.get(group_name)
        if group_cfg is None:
            return None

        return group_cfg.get("anchor_prim")

    def _get_env_prim_path(
        self,
        prim_ref: str,
        env_id: int,
    ) -> str:
        """Convert an env-relative prim reference to an absolute stage path."""
        if prim_ref.startswith("/"):
            raise ValueError(
                "anchor_prim must be env-relative, without "
                "'/World/envs/env_N/'. "
                f"Got: {prim_ref}"
            )

        return f"/World/envs/env_{env_id}/{prim_ref.strip('/')}"

    def _get_rendered_prim_world_pose(self, prim_path: str, device: torch.device):
        import usdrt

        if getattr(self, "_fabric_hierarchy", None) is None:
            stage_id = omni.usd.get_context().get_stage_id()
            rt_stage = usdrt.Usd.Stage.Attach(stage_id)
            self._fabric_hierarchy = (
                usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
                    rt_stage.GetFabricId(), stage_id
                )
            )

        # WITHOUT this call, fabric world transforms do not exist at all
        self._fabric_hierarchy.update_world_xforms()

        m = self._fabric_hierarchy.get_world_xform(usdrt.Sdf.Path(prim_path))

        m_np = np.array(
            [[m[r][c] for c in range(4)] for r in range(4)], dtype=np.float64
        )
        pos_np = m_np[3, :3].copy()

        # upper 3x3: rows = basis vectors (row-vector convention USD)
        basis = m_np[:3, :3]
        basis /= np.linalg.norm(basis, axis=1, keepdims=True)  # remove scale
        R = basis.T

        # matrix -> quaternion [w, x, y, z]
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            q = [
                0.25 * s,
                (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
            ]
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q = [
                (R[2, 1] - R[1, 2]) / s,
                0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s,
            ]
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q = [
                (R[0, 2] - R[2, 0]) / s,
                (R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s,
            ]
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q = [
                (R[1, 0] - R[0, 1]) / s,
                (R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s,
                0.25 * s,
            ]

        pos = torch.tensor(pos_np, dtype=torch.float, device=device)
        quat = torch.tensor(q, dtype=torch.float, device=device)
        quat = quat / torch.clamp(torch.linalg.vector_norm(quat), min=1e-12)

        # print("[FABRIC]", prim_path, pos_np.tolist())

        return pos, quat

    def _get_live_anchor_body_pose(
        self,
        scene,
        anchor_prim: str,
        env_id: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read live PhysX pose of any env-relative rigid-body anchor prim.

        The anchor does not have to be an InteractiveScene RigidObject or
        Articulation. It only must have UsdPhysics.RigidBodyAPI.
        """
        if not isinstance(anchor_prim, str) or not anchor_prim.strip():
            raise RuntimeError(
                f"anchor_prim must be a non-empty env-relative path, "
                f"got {anchor_prim!r}"
            )

        anchor_prim = anchor_prim.strip("/")
        anchor_path = self._get_env_prim_path(anchor_prim, env_id)

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(anchor_path)

        if not prim.IsValid():
            raise RuntimeError(
                f"anchor_prim does not exist for env={env_id}: " f"'{anchor_path}'"
            )

        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                "anchor_prim must point directly to a PhysX rigid body.\n"
                f"anchor_prim: '{anchor_prim}'\n"
                f"resolved path: '{anchor_path}'"
            )

        physx_view = self._anchor_physx_views.get(anchor_prim)

        if physx_view is None:
            sim_context = SimulationContext.instance()

            if sim_context is None:
                raise RuntimeError(
                    "SimulationContext is not initialized; "
                    "cannot create anchor RigidBodyView."
                )

            physx_pattern = f"/World/envs/env_*/{anchor_prim}"
            physx_view = sim_context.physics_sim_view.create_rigid_body_view(
                physx_pattern
            )

            if physx_view.count == 0:
                raise RuntimeError(
                    "PhysX did not find any rigid bodies for anchor_prim.\n"
                    f"anchor_prim: '{anchor_prim}'\n"
                    f"pattern: '{physx_pattern}'"
                )

            self._anchor_physx_views[anchor_prim] = physx_view

        transforms = physx_view.get_transforms()

        if not isinstance(transforms, torch.Tensor):
            transforms = wp.to_torch(transforms)

        if env_id < 0 or env_id >= transforms.shape[0]:
            raise RuntimeError(
                f"anchor_prim '{anchor_prim}' has only "
                f"{transforms.shape[0]} PhysX instances, "
                f"but env_id={env_id} was requested."
            )

        transform = transforms[env_id].to(device=device, dtype=torch.float)

        # PhysX RigidBodyView transform format:
        # [position_x, position_y, position_z, quat_x, quat_y, quat_z, quat_w]
        pos_w = transform[0:3]
        quat_w = transform[[6, 3, 4, 5]]  # convert PhysX xyzw -> Isaac Lab wxyz

        quat_w = quat_w / torch.clamp(
            torch.linalg.vector_norm(quat_w),
            min=1e-12,
        )

        return pos_w, quat_w

    def _get_live_prim_world_pose(
        self,
        prim_ref: str,
        env_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read live world pose of an env-relative USD prim."""
        stage = omni.usd.get_context().get_stage()
        prim_path = self._get_env_prim_path(prim_ref, env_id)

        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(
                f"anchor_prim does not exist for env={env_id}: {prim_path}"
            )

        xform = UsdGeom.Xformable(prim)
        world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        translation = world_transform.ExtractTranslation()
        rotation = world_transform.ExtractRotation()
        quat = rotation.GetQuat()

        pos = np.array(
            [translation[0], translation[1], translation[2]],
            dtype=np.float64,
        )
        rot = np.array(
            [
                quat.GetReal(),
                quat.GetImaginary()[0],
                quat.GetImaginary()[1],
                quat.GetImaginary()[2],
            ],
            dtype=np.float64,
        )
        rot /= np.linalg.norm(rot)

        return pos, rot

    def _make_anchored_root_pose_tensors(
        self,
        scene,
        meta: Dict[str, Any],
        local_pos: list[float],
        local_rot: list[float],
        env_ids: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert anchor-local object pose to live world root poses."""
        anchor_prim = self._get_group_anchor_prim(meta)

        if anchor_prim is None:
            raise RuntimeError(
                "Attempted anchored placement for object without anchor_prim"
            )

        local_pos_tensor = torch.tensor(
            local_pos,
            dtype=torch.float,
            device=device,
        )
        local_rot_tensor = torch.tensor(
            local_rot,
            dtype=torch.float,
            device=device,
        )
        usd_local_rot_tensor = torch.tensor(
            meta["usd_local_rotation"],
            dtype=torch.float,
            device=device,
        )

        pos_list = []
        rot_list = []

        for env_id in env_ids.tolist():
            anchor_pos_w, anchor_quat_w = self._get_live_anchor_body_pose(
                scene=scene,
                anchor_prim=anchor_prim,
                env_id=int(env_id),
                device=device,
            )

            world_user_pos = anchor_pos_w + _quat_rotate_torch(
                anchor_quat_w,
                local_pos_tensor,
            )

            world_user_rot = _quat_mul_torch(
                anchor_quat_w,
                local_rot_tensor,
            )
            world_user_rot = world_user_rot / torch.clamp(
                torch.linalg.vector_norm(world_user_rot),
                min=1e-12,
            )

            final_rot = _quat_mul_torch(
                world_user_rot,
                usd_local_rot_tensor,
            )
            final_rot = final_rot / torch.clamp(
                torch.linalg.vector_norm(final_rot),
                min=1e-12,
            )

            pos_list.append(world_user_pos)
            rot_list.append(final_rot)

        return (
            torch.stack(pos_list, dim=0),
            torch.stack(rot_list, dim=0),
        )

    def _get_guide_prim_path(
        self,
        name: str,
        meta: Dict[str, Any],
        env_id: int,
    ) -> str:
        """Return the root path or anchor-child path for a guide."""
        anchor_prim = self._get_group_anchor_prim(meta)

        if anchor_prim is None:
            return f"/World/envs/env_{env_id}/{name}"

        return f"{self._get_env_prim_path(anchor_prim, env_id)}/{name}"

    def capture_initial_poses(self, scene):
        """Capture actual initial poses from the simulation after spawn.

        Call this AFTER the scene has been created and initialized.
        """
        self._initial_poses = {}
        self._initial_joint_pos = {}

        for name, meta in self.object_metadata.items():
            if meta.get("object_kind") == "guide":
                continue
            if meta.get("scene_static", False):
                continue

            if meta["articulated"]:
                if name in scene.articulations:
                    art = scene.articulations[name]
                    pos = art.data.root_pos_w[0].cpu().numpy().tolist()
                    quat = art.data.root_quat_w[0].cpu().numpy().tolist()
                    self._initial_poses[name] = (pos, quat)

                    joint_pos = art.data.joint_pos[0].cpu().numpy()
                    joint_names = art.joint_names
                    self._initial_joint_pos[name] = {
                        joint_names[i]: float(joint_pos[i])
                        for i in range(len(joint_names))
                    }
            else:
                if name in scene.rigid_objects:
                    obj = scene.rigid_objects[name]
                    pos = obj.data.root_pos_w[0].cpu().numpy().tolist()
                    quat = obj.data.root_quat_w[0].cpu().numpy().tolist()
                    self._initial_poses[name] = (pos, quat)

        # for name in self.object_metadata.keys():
        #     prim_paths = []
        #     for env_id in range(num_envs):
        #         # Form a full path taking into account the environment namespace
        #         obj = scene[name]
        #         # obj is your RigidObject
        #         prim_path = obj.root_physx_view.prim_paths[env_id]
        #         prim_paths.append(prim_path)
        #     self._prim_paths_cache[name] = prim_paths

        num_envs = scene.num_envs
        for name, meta in self.object_metadata.items():

            # if meta.get("object_kind") == "guide":
            #     self._prim_paths_cache[name] = [
            #         f"/World/envs/env_{env_id}/{name}" for env_id in range(num_envs)
            #     ]
            #     continue
            if meta.get("object_kind") == "guide":
                self._prim_paths_cache[name] = [
                    self._get_guide_prim_path(name, meta, env_id)
                    for env_id in range(num_envs)
                ]
                continue

            # scene_static objects were spawned directly via spawn_from_usd()
            # and are not registered in InteractiveScene.
            if meta.get("scene_static", False):
                self._prim_paths_cache[name] = [
                    f"/World/envs/env_{env_id}/{name}" for env_id in range(num_envs)
                ]
                continue

            # InteractiveScene-managed articulation
            if meta["articulated"] and name in scene.articulations:
                art = scene.articulations[name]
                self._prim_paths_cache[name] = [
                    art.root_physx_view.prim_paths[env_id] for env_id in range(num_envs)
                ]
                continue

            # InteractiveScene-managed rigid object
            if name in scene.rigid_objects:
                obj = scene.rigid_objects[name]
                self._prim_paths_cache[name] = [
                    obj.root_physx_view.prim_paths[env_id] for env_id in range(num_envs)
                ]
                continue

            # Fallback: entity not found in scene registry
            print(
                f"Warning: '{name}' not found in scene.articulations or scene.rigid_objects; "
                f"skipping prim path cache population for this object."
            )

        self.update_guide_objects(scene)
        self.update_robot_spawn_visuals(scene)
        self.update_group_visuals(scene)

    def get_asset_cfgs(
        self,
    ) -> Tuple[
        Dict[str, RigidObjectCfg],
        Dict[str, ArticulationCfg],
        Dict[str, Dict[str, float]],
    ]:
        """Generate AssetBaseCfg using deterministic pose generation (seed=0).

        The initial state respects both user-specified rotation and USD's local orientation.
        """

        # pose_object_metadata = {
        #     name: meta
        #     for name, meta in self.object_metadata.items()
        #     if meta.get("object_kind") != "guide"
        # }
        pose_object_metadata = {
            name: meta
            for name, meta in self.object_metadata.items()
            if not meta.get("scene_static", False)
        }

        poses, _ = _generate_object_poses(
            pose_object_metadata,
            self.group_templates,
            self._group_objects,
            seed=0,
        )

        # Persist initial local poses for guides.  They are not Isaac Lab
        # assets, so update_guide_objects() reads their pose from metadata.
        for name, meta in self.object_metadata.items():
            if meta.get("object_kind") != "guide":
                continue

            if name not in poses:
                continue

            local_pos, local_rot = poses[name]
            meta["position"] = list(local_pos)
            meta["rotation"] = list(local_rot)

        asset_cfgs: Dict[str, RigidObjectCfg] = {}
        articulation_cfgs: Dict[str, ArticulationCfg] = {}
        friction_specs: Dict[str, Dict[str, float]] = {}

        for name, meta in self.object_metadata.items():
            if meta.get("object_kind") == "guide":
                continue
            if meta.get("scene_static", False):
                continue
            prim_path = f"{self.env_regex_ns}/{name}"
            # if name not in poses:
            #     # For inactive objects, use the "safe" position ON THE FLOOR (not in the sump!)
            #     user_pos = PARKING_POSITION
            #     user_rot = [1.0, 0.0, 0.0, 0.0]
            # else:
            #     user_pos, user_rot = poses[name]
            anchor_prim = self._get_group_anchor_prim(meta)

            if name not in poses or anchor_prim is not None:
                user_pos, user_rot = self._parking_pose_for(name, env_id=0)
            else:
                user_pos, user_rot = poses[name]

            # Combine user rotation with USD local rotation
            final_rot = _quat_mul(user_rot, meta["usd_local_rotation"])
            final_rot /= np.linalg.norm(final_rot)

            if meta.get("articulated", False):
                articulation_cfgs[name] = ArticulationCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=meta["usd_path"],
                        scale=meta["scale"],
                        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                            fix_root_link=meta["fixed"],
                        ),
                    ),
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=user_pos,
                        rot=final_rot.tolist(),
                        # joint_pos={".*": 0.0},
                    ),
                    actuators={},
                )
            else:

                rigid_props = sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=meta["fixed"],
                    disable_gravity=meta["fixed"],
                    max_linear_velocity=0.0 if meta["fixed"] else 1000.0,
                    max_angular_velocity=0.0 if meta["fixed"] else 1000.0,
                )
                collision_props = (
                    sim_utils.CollisionPropertiesCfg()
                )  # collision_enabled=True)

                asset_cfgs[name] = RigidObjectCfg(
                    prim_path=prim_path,
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=meta["usd_path"],
                        scale=meta["scale"],
                        mass_props=sim_utils.MassPropertiesCfg(mass=meta["mass"]),
                        rigid_props=rigid_props,
                        collision_props=collision_props,
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=user_pos, rot=final_rot.tolist()
                    ),
                )

                if meta["physics_material"]:
                    friction_specs[prim_path] = {
                        "static_friction": float(
                            meta["physics_material"].get("static_friction", 0.5)
                        ),
                        "dynamic_friction": float(
                            meta["physics_material"].get("dynamic_friction", 0.5)
                        ),
                        "restitution": float(
                            meta["physics_material"].get("restitution", 0.0)
                        ),
                    }

        return asset_cfgs, articulation_cfgs, friction_specs

    def generate_reset_poses(
        self,
        seed: Optional[int] = None,
    ) -> Tuple[
        Dict[str, Tuple[list[float], list[float]]],
        Dict[str, bool],
    ]:
        pose_object_metadata = {
            name: meta
            for name, meta in self.object_metadata.items()
            if not meta.get("scene_static", False)
        }

        return _generate_object_poses(
            pose_object_metadata,
            self.group_templates,
            self._group_objects,
            seed=seed,
        )

    def _apply_collision_state(
        self,
        active_objects: Dict[str, bool],
        env_ids: torch.Tensor,
        enable: bool = True,
    ):
        """
        Efficiently manages collisions via USD API.
        Call BEFORE moving objects (to avoid collisions during teleportation).

        Args:
            active_objects: {object_name: should_be_active}
            env_ids: environment indices to update
            enable: True = enable collisions for active objects, False = disable collisions for inactive objects
        """

        stage = omni.usd.get_context().get_stage()

        def _set_collision_enabled(prim, enabled: bool):
            # Only actual collider prims have CollisionAPI
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                return False
            col_api = UsdPhysics.CollisionAPI(prim)
            attr = col_api.GetCollisionEnabledAttr()
            if not attr:
                attr = col_api.CreateCollisionEnabledAttr()
            attr.Set(enabled)
            return True

        for env_id in env_ids.tolist():
            env_id = int(env_id)
            for obj_name, is_active in active_objects.items():
                if self.object_metadata[obj_name]["fixed"]:
                    continue

                should_enable = is_active if enable else not is_active
                full_path = self._prim_paths_cache[obj_name][env_id]

                root_prim = stage.GetPrimAtPath(full_path)
                if not root_prim.IsValid():
                    print(f"Warning: invalid prim path: {full_path}")
                    continue

                try:
                    # Traverse the ENTIRE prim hierarchy of the object
                    for prim in Usd.PrimRange(root_prim):
                        _set_collision_enabled(prim, should_enable)

                except Exception as e:
                    print(
                        f"Warning: Failed to set collision state for {full_path}: {e}"
                    )

    def get_physx_prim_path(self, object_name: str, env_id: int = 0) -> str:
        if not self._prim_paths_cache:
            raise RuntimeError(
                "Prim path cache is empty. Call capture_initial_poses(scene) first."
            )
        if object_name not in self._prim_paths_cache:
            raise KeyError(f"Unknown object name: {object_name}")

        prim_paths = self._prim_paths_cache[object_name]
        if env_id < 0 or env_id >= len(prim_paths):
            raise IndexError(
                f"env_id={env_id} out of range for '{object_name}', num_envs={len(prim_paths)}"
            )
        return prim_paths[env_id]

    def resolve_policy_ref(self, ref: str, env_id: int = 0) -> str:
        if not isinstance(ref, str) or not ref:
            raise ValueError("Policy ref must be a non-empty string")

        # 1) already absolute path
        if ref.startswith("/"):
            return ref

        # 2) scene object name
        if ref in self._prim_paths_cache:
            return self.get_physx_prim_path(ref, env_id=env_id)

        # 3) env-relative prim, e.g. "Robot/root"
        return f"/World/envs/env_{env_id}/{ref.lstrip('/')}"

    def build_object_pose_resolver(self, scene):
        def _resolve(target_ref: str, env_index: int, device: torch.device):
            env_index = int(env_index)
            env_prefix = f"/World/envs/env_{env_index}/"

            # if target_ref.startswith(env_prefix):
            #     absolute_target_ref = target_ref
            #     target_ref = target_ref[len(env_prefix) :]

            #     for guide_name, guide_meta in self.object_metadata.items():
            #         if guide_meta.get("object_kind") != "guide":
            #             continue

            #         guide_path = self._get_guide_prim_path(
            #             guide_name,
            #             guide_meta,
            #             env_index,
            #         )

            #         if (
            #             absolute_target_ref == guide_path
            #             or target_ref == guide_name
            #             or target_ref.endswith(f"/{guide_name}")
            #         ):
            #             target_ref = guide_name
            #             break

            absolute_target_ref = None

            if target_ref.startswith(env_prefix):
                absolute_target_ref = target_ref
                target_ref = target_ref[len(env_prefix) :]

            for guide_name, guide_meta in self.object_metadata.items():
                if guide_meta.get("object_kind") != "guide":
                    continue

                guide_path = self._get_guide_prim_path(
                    guide_name,
                    guide_meta,
                    env_index,
                )

                if (
                    absolute_target_ref == guide_path
                    or target_ref == guide_name
                    or target_ref.endswith(f"/{guide_name}")
                ):
                    target_ref = guide_name
                    break

            if target_ref in scene.rigid_objects:
                obj = scene.rigid_objects[target_ref]
                return obj.data.root_pos_w[env_index].to(device), obj.data.root_quat_w[
                    env_index
                ].to(device)

            if target_ref in scene.articulations:
                art = scene.articulations[target_ref]
                return art.data.root_pos_w[env_index].to(device), art.data.root_quat_w[
                    env_index
                ].to(device)

            if (
                target_ref in self.object_metadata
                and self.object_metadata[target_ref].get("object_kind") == "guide"
            ):
                meta = self.object_metadata[target_ref]
                anchor_prim = self._get_group_anchor_prim(meta)

                # Guide is attached to an anchor: read the pose of the guide prim itself
                # as the renderer collects it (Fabric-aware).
                if anchor_prim is not None:
                    guide_path = self._get_guide_prim_path(target_ref, meta, env_index)
                    return self._get_rendered_prim_world_pose(guide_path, device)

                # Guide WITHOUT an anchor (static, like drawer_is_opened_guide):
                # the pose lives in metadata — this branch must remain!
                pos = torch.tensor(
                    meta["position"] or [0.0, 0.0, 0.0],
                    dtype=torch.float,
                    device=device,
                )
                quat = torch.tensor(
                    meta["rotation"] or [1.0, 0.0, 0.0, 0.0],
                    dtype=torch.float,
                    device=device,
                )

                if hasattr(scene, "env_origins"):
                    pos = pos + scene.env_origins[env_index].to(device)

                return pos, quat
            # if (
            #     target_ref in self.object_metadata
            #     and self.object_metadata[target_ref].get("object_kind") == "guide"
            # ):
            #     meta = self.object_metadata[target_ref]
            #     anchor_prim = self._get_group_anchor_prim(meta)

            # if anchor_prim is not None:
            #     anchor_pos_w, anchor_quat_w = self._get_live_anchor_body_pose(
            #         scene=scene,
            #         anchor_prim=anchor_prim,
            #         env_id=env_index,
            #         device=device,
            #     )

            #     local_pos = torch.tensor(
            #         meta["position"] or [0.0, 0.0, 0.0],
            #         dtype=torch.float,
            #         device=device,
            #     )
            #     local_quat = torch.tensor(
            #         meta["rotation"] or [1.0, 0.0, 0.0, 0.0],
            #         dtype=torch.float,
            #         device=device,
            #     )

            #     guide_pos_w = anchor_pos_w + _quat_rotate_torch(
            #         anchor_quat_w,
            #         local_pos,
            #     )

            #     guide_quat_w = _quat_mul_torch(
            #         anchor_quat_w,
            #         local_quat,
            #     )
            #     guide_quat_w = guide_quat_w / torch.clamp(
            #         torch.linalg.vector_norm(guide_quat_w),
            #         min=1e-12,
            #     )

            #     return guide_pos_w, guide_quat_w

            # pos = torch.tensor(
            #     meta["position"] or [0.0, 0.0, 0.0],
            #     dtype=torch.float,
            #     device=device,
            # )
            # quat = torch.tensor(
            #     meta["rotation"] or [1.0, 0.0, 0.0, 0.0],
            #     dtype=torch.float,
            #     device=device,
            # )

            # if hasattr(scene, "env_origins"):
            #     pos = pos + scene.env_origins[env_index].to(device)

            # return pos, quat

            resolved_ref = self.resolve_policy_ref(target_ref, env_id=env_index)

            for name, prim_paths in self._prim_paths_cache.items():
                if (
                    env_index < len(prim_paths)
                    and prim_paths[env_index] == resolved_ref
                ):
                    if name in scene.rigid_objects:
                        obj = scene.rigid_objects[name]
                        return obj.data.root_pos_w[env_index].to(
                            device
                        ), obj.data.root_quat_w[env_index].to(device)
                    if name in scene.articulations:
                        art = scene.articulations[name]
                        return art.data.root_pos_w[env_index].to(
                            device
                        ), art.data.root_quat_w[env_index].to(device)

            raise RuntimeError(
                f"Unable to resolve live pose for target_ref={target_ref}, env_index={env_index}"
            )

        return _resolve

    def _get_live_guide_world_pose(
        self,
        name: str,
        meta: Dict[str, Any],
        env_id: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        guide_path = self._get_guide_prim_path(name, meta, env_id)

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(guide_path)

        if not prim.IsValid():
            raise RuntimeError(f"Guide prim does not exist: '{guide_path}'")

        xform = UsdGeom.Xformable(prim)
        world_tf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        translation = world_tf.ExtractTranslation()
        quat = world_tf.ExtractRotation().GetQuat()

        pos_w = torch.tensor(
            [translation[0], translation[1], translation[2]],
            dtype=torch.float,
            device=device,
        )

        quat_w = torch.tensor(
            [
                quat.GetReal(),
                quat.GetImaginary()[0],
                quat.GetImaginary()[1],
                quat.GetImaginary()[2],
            ],
            dtype=torch.float,
            device=device,
        )

        quat_w = quat_w / torch.clamp(
            torch.linalg.vector_norm(quat_w),
            min=1e-12,
        )

        return pos_w, quat_w

    def is_object_center_inside(
        self,
        scene,
        object_name: str,
        container_name: str,
    ) -> torch.Tensor:
        """
        Checks whether the object's root-frame center is inside the container
        for each environment.

        The container is determined from object_metadata:
        - guide with guide_radius != None: sphere;
        - guide with guide_radius == None: oriented box with dimensions size;
        - regular asset: oriented box with dimensions metadata["size"].

        Args:
            scene: InteractiveScene.
            object_name: Name of the rigid object or articulation whose center
                is being checked.
            container_name: Name of the container object from the scene JSON.
                The caller does not need to know whether it is a guide.

        Returns:
            torch.BoolTensor shape [num_envs].
            True means the center of object_name lies inside container_name
            in the corresponding environment.

        Raises:
            KeyError: If object_name or container_name is unknown.
            ValueError: If container_name does not have a valid size.
            RuntimeError: If the pose for the guide has not been determined yet.
        """
        if object_name in scene.rigid_objects:
            object_position_w = scene.rigid_objects[object_name].data.root_pos_w

        elif object_name in scene.articulations:
            object_position_w = scene.articulations[object_name].data.root_pos_w

        elif (
            object_name in self.object_metadata
            and self.object_metadata[object_name].get("object_kind") == "guide"
        ):
            device = scene.env_origins.device
            pose_resolver = self.build_object_pose_resolver(scene)

            object_position_w = torch.stack(
                [
                    pose_resolver(object_name, env_id, device)[0]
                    for env_id in range(scene.num_envs)
                ],
                dim=0,
            )
        else:
            robot_key = (
                self.robot_cfg.get("name", "robot")
                if self.robot_cfg is not None
                else "robot"
            )
            robot_art = scene.articulations.get(robot_key)

            if robot_art is not None and object_name in robot_art.body_names:
                body_idx = robot_art.body_names.index(object_name)
                object_position_w = robot_art.data.body_pos_w[:, body_idx, :]
            else:
                available_bodies = (
                    [] if robot_art is None else list(robot_art.body_names)
                )
                raise KeyError(
                    f"Object '{object_name}' not found in scene or robot bodies. "
                    f"Robot='{robot_key}', available body names: {available_bodies}"
                )

        if container_name not in self.object_metadata:
            raise KeyError(f"Unknown container: '{container_name}'")

        container_meta = self.object_metadata[container_name]
        device = object_position_w.device
        dtype = object_position_w.dtype
        num_envs = object_position_w.shape[0]

        if container_name in scene.rigid_objects:
            container_position_w = scene.rigid_objects[container_name].data.root_pos_w
            container_rotation_w = scene.rigid_objects[container_name].data.root_quat_w

        elif container_name in scene.articulations:
            container_position_w = scene.articulations[container_name].data.root_pos_w
            container_rotation_w = scene.articulations[container_name].data.root_quat_w

        # elif container_meta.get("object_kind") == "guide":
        #     if container_meta["position"] is None:
        #         raise RuntimeError(
        #             f"Guide '{container_name}' has no resolved position. "
        #             "Call reset_objects() before checking containment."
        #         )

        #     container_position_w = (
        #         torch.tensor(
        #             container_meta["position"],
        #             dtype=dtype,
        #             device=device,
        #         )
        #         .unsqueeze(0)
        #         .repeat(num_envs, 1)
        #     )

        #     container_rotation_w = (
        #         torch.tensor(
        #             container_meta["rotation"] or [1.0, 0.0, 0.0, 0.0],
        #             dtype=dtype,
        #             device=device,
        #         )
        #         .unsqueeze(0)
        #         .repeat(num_envs, 1)
        #     )

        #     if hasattr(scene, "env_origins"):
        #         container_position_w += scene.env_origins.to(
        #             device=device,
        #             dtype=dtype,
        #         )
        elif container_meta.get("object_kind") == "guide":
            pose_resolver = self.build_object_pose_resolver(scene)

            container_poses = [
                pose_resolver(container_name, env_id, device)
                for env_id in range(num_envs)
            ]

            container_position_w = torch.stack(
                [pose[0] for pose in container_poses],
                dim=0,
            )

            container_rotation_w = torch.stack(
                [pose[1] for pose in container_poses],
                dim=0,
            )

        else:
            raise KeyError(
                f"Container '{container_name}' is not available in scene "
                "and is not a guide"
            )

        guide_radius = container_meta.get("guide_radius")

        if guide_radius is not None:
            delta_w = object_position_w - container_position_w
            inside = torch.sum(delta_w * delta_w, dim=-1) <= float(guide_radius) ** 2

            # if (
            #     object_name == "drawer_handle_guide"
            #     and container_name == "drawer_is_opened_guide"
            # ):
            #     print(
            #         "[INSIDE]",
            #         object_position_w[0].detach().cpu().tolist(),
            #         container_position_w[0].detach().cpu().tolist(),
            #         inside[0].item(),
            #     )

            return inside

        container_size = container_meta.get("size")
        if not _is_position3(container_size):
            raise ValueError(f"Container '{container_name}' must have size [x, y, z]")

        container_size = torch.tensor(
            container_size,
            dtype=dtype,
            device=device,
        ).unsqueeze(0)

        container_rotation_w = container_rotation_w / torch.linalg.vector_norm(
            container_rotation_w,
            dim=-1,
            keepdim=True,
        )

        delta_w = object_position_w - container_position_w

        q_w = container_rotation_w[:, 0:1]
        q_xyz = -container_rotation_w[:, 1:4]

        t = 2.0 * torch.cross(q_xyz, delta_w, dim=-1)

        object_position_local = delta_w + q_w * t + torch.cross(q_xyz, t, dim=-1)

        inside = torch.all(
            torch.abs(object_position_local) <= 0.5 * container_size,
            dim=-1,
        )

        # if (
        #     object_name == "drawer_handle_guide"
        #     and container_name == "drawer_is_opened_guide"
        # ):
        #     print(
        #         "[INSIDE]",
        #         object_position_w[0].detach().cpu().tolist(),
        #         container_position_w[0].detach().cpu().tolist(),
        #         object_position_local[0].detach().cpu().tolist(),
        #         container_size[0].detach().cpu().tolist(),
        #         inside[0].item(),
        #     )

        return inside

    # def build_object_manipulator(
    #     self,
    #     scene,
    # ) -> SceneObjectManipulator:
    #     return SceneObjectManipulator(
    #         scene=scene,
    #         scene_spawner=self,
    #     )

    def _parking_pose_for(
        self,
        obj_name: str,
        env_id: int,
    ) -> tuple[list[float], list[float]]:
        names = sorted(self.object_metadata.keys())
        obj_index = names.index(obj_name) if obj_name in names else 0

        pos = [
            100.0 + 0.5 * obj_index,
            100.0 + 0.5 * int(env_id),
            2.0,
        ]
        rot = [1.0, 0.0, 0.0, 0.0]
        return pos, rot

    def _reset_configured_articulation_joints(
        self,
        name: str,
        meta: Dict[str, Any],
        art,
        env_ids: torch.Tensor,
    ) -> None:
        """Set explicitly configured articulation joint positions on reset."""

        requested_joint_pos = meta.get("joint_pos", {})
        if not requested_joint_pos:
            return

        available_joint_names = list(art.joint_names)
        unknown_joint_names = sorted(
            set(requested_joint_pos.keys()) - set(available_joint_names)
        )

        if unknown_joint_names:
            raise ValueError(
                f"Object '{name}': unknown joint name(s) in JSON joint_pos: "
                f"{unknown_joint_names}. "
                f"Available articulation joints: {available_joint_names}"
            )

        # Use the authored/default state as a base so as not to break the other joints.
        joint_pos = art.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)

        for joint_name, joint_value in requested_joint_pos.items():
            joint_index = available_joint_names.index(joint_name)
            joint_pos[:, joint_index] = float(joint_value)

        art.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            env_ids=env_ids,
        )

    def _make_parking_tensors(
        self,
        obj_name: str,
        env_ids: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_list = []
        rot_list = []

        for eid in env_ids.tolist():
            pos, rot = self._parking_pose_for(obj_name, int(eid))
            pos_list.append(pos)
            rot_list.append(rot)

        pos_tensor = torch.tensor(pos_list, dtype=torch.float, device=device)
        rot_tensor = torch.tensor(rot_list, dtype=torch.float, device=device)

        return pos_tensor, rot_tensor

    def reset_objects(self, scene, env_ids: torch.Tensor, seed: Optional[int] = None):
        """Reset rigid objects and articulations to initial or newly generated poses.

        Restores the root state (position, orientation, and zero linear/angular velocity)
        for all non-fixed objects in the specified environments. Articulated objects
        additionally have their joint positions reset to captured initial values when
        available; otherwise joints are set to zero. Objects without a spawn group revert
        to poses captured via `capture_initial_poses()`, while objects belonging to a
        spawn group receive newly generated poses from `generate_reset_poses()`.

        Objects marked as fixed in metadata are skipped. Objects not present in the
        scene's `articulations` or `rigid_objects` dictionaries are silently ignored.
        Velocities are always zeroed during reset regardless of object type.

        Args:
            scene: Simulation scene object containing `articulations` and `rigid_objects`
                dictionaries keyed by object name.
            env_ids (torch.Tensor): 1-D tensor of environment indices to reset. Must be
                on the same device as simulation tensors.
            seed (int, optional): Random seed for pose generation when resetting grouped
                objects. If None, uses the generator's internal state.

        Raises:
            RuntimeError: If `capture_initial_poses()` was not called prior to this method
                (i.e., `_initial_poses` is None).

        Note:
            - For articulated objects without a spawn group, joint positions revert to
            values stored in `_initial_joint_pos` (if available).
            - For grouped articulated objects, joint positions default to zero.
            - USD-local rotations (`usd_local_rotation` in metadata) are combined with
            generated/user rotations using quaternion multiplication and normalized.
            - This method modifies simulation state directly via `write_root_state_to_sim`
            and `write_joint_state_to_sim`.
        """
        if self._initial_poses is None:
            raise RuntimeError(
                "capture_initial_poses() must be called before reset_objects()"
            )
        if not self._prim_paths_cache:
            raise RuntimeError(
                "capture_initial_poses() must populate _prim_paths_cache before reset"
            )

        new_poses, active_objects = self.generate_reset_poses(seed=seed)
        self.last_active_objects = active_objects

        parent_names = {
            meta["parent_object"]
            for meta in self.object_metadata.values()
            if meta.get("parent_object") is not None
        }
        # self._apply_collision_state(active_objects, env_ids, enable=False)

        rng = np.random.default_rng(seed)
        device = env_ids.device

        # Apply reset poses to guides.
        # They have no PhysX body: the pose is stored in metadata,
        # and is applied to the USD sphere in update_guide_objects() at the end of reset.
        for name, meta in self.object_metadata.items():
            if meta.get("object_kind") != "guide":
                continue

            # A guide without a group keeps its JSON-specified pose.
            if meta.get("spawn_group") is None:
                continue

            # The guide was not selected according to spawn_count:
            # we do not change its pose; update_guide_objects() will hide it later.
            if not active_objects.get(name, True):
                continue

            position, rotation = new_poses[name]

            if meta.get("random_rotation") is not None:
                rotation = _apply_random_rotation(
                    rotation,
                    meta["random_rotation"],
                    rng,
                )

            meta["position"] = list(position)
            meta["rotation"] = list(rotation)

        for name, meta in self.object_metadata.items():
            if meta.get("object_kind") == "guide":
                continue
            if meta.get("scene_static", False):
                continue

            if meta.get("articulated", False):
                if name not in scene.articulations:
                    continue

                art = scene.articulations[name]

                self._reset_configured_articulation_joints(
                    name=name,
                    meta=meta,
                    art=art,
                    env_ids=env_ids,
                )

                if meta["fixed"]:
                    continue

                # For non-group articulations, use captured states
                # if meta["spawn_group"] is None and name in self._initial_poses:
                use_initial_pose = (
                    meta["spawn_group"] is None
                    and meta.get("parent_object") is None
                    and name not in parent_names
                    and name not in self._reload_pose_overrides
                    and name in self._initial_poses
                )

                if use_initial_pose:
                    pos, rot = self._initial_poses[name]
                    # Reset joints to captured values (if available)
                    joint_targets = self._initial_joint_pos.get(name, {})
                else:
                    # For grouped objects — new poses + joints to zero
                    if not active_objects.get(name, True):
                        # For the inactive - a safe position ON THE FLOOR
                        pos_tensor, rot_tensor = self._make_parking_tensors(
                            name, env_ids, device
                        )
                        joint_targets = {}
                    else:
                        # user_pos, user_rot = new_poses[name]
                        # # if meta.get("random_rotation"):
                        # #     user_rot = _apply_random_rotation(
                        # #         user_rot, meta["random_rotation"], rng
                        # #     )
                        # final_rot = _quat_mul(user_rot, meta["usd_local_rotation"])
                        # final_rot /= np.linalg.norm(final_rot)
                        # pos_tensor = torch.tensor(
                        #     user_pos, dtype=torch.float, device=device
                        # ).repeat(len(env_ids), 1)
                        # rot_tensor = torch.tensor(
                        #     final_rot.tolist(), dtype=torch.float, device=device
                        # ).repeat(len(env_ids), 1)
                        user_pos, user_rot = new_poses[name]
                        anchor_prim = self._get_group_anchor_prim(meta)

                        if anchor_prim is not None:
                            pos_tensor, rot_tensor = (
                                self._make_anchored_root_pose_tensors(
                                    scene=scene,
                                    meta=meta,
                                    local_pos=user_pos,
                                    local_rot=user_rot,
                                    env_ids=env_ids,
                                    device=device,
                                )
                            )
                        else:
                            final_rot = _quat_mul(user_rot, meta["usd_local_rotation"])
                            final_rot /= np.linalg.norm(final_rot)

                            pos_tensor = torch.tensor(
                                user_pos,
                                dtype=torch.float,
                                device=device,
                            ).repeat(len(env_ids), 1)

                            rot_tensor = torch.tensor(
                                final_rot.tolist(),
                                dtype=torch.float,
                                device=device,
                            ).repeat(len(env_ids), 1)

                        joint_targets = {}

                # Reset root state
                # Reset root state
                root_state = art.data.default_root_state.clone().to(device)
                root_state[env_ids, :3] = pos_tensor  # <-- now always pos_tensor
                root_state[env_ids, 3:7] = rot_tensor  # <-- now always rot_tensor
                root_state[env_ids, 7:] = 0.0
                art.write_root_state_to_sim(root_state, env_ids=env_ids)

                # Reset joints (if captured values exist)
                if joint_targets:
                    joint_pos = torch.zeros(
                        (len(env_ids), art.num_joints), device=device
                    )
                    for j_idx, j_name in enumerate(art.joint_names):
                        if j_name in joint_targets:
                            joint_pos[:, j_idx] = joint_targets[j_name]
                    art.write_joint_state_to_sim(
                        joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids
                    )

            else:  # simple rigid objects
                if name not in scene.rigid_objects:
                    continue

                obj = scene.rigid_objects[name]

                # we do not move a fixed rigid object at all.
                if meta.get("fixed", False):
                    continue

                # if meta["spawn_group"] is None and name in self._initial_poses:
                use_initial_pose = (
                    meta["spawn_group"] is None
                    and meta.get("parent_object") is None
                    and name not in parent_names
                    and name not in self._reload_pose_overrides
                    and name in self._initial_poses
                )

                if use_initial_pose:
                    pos, rot = self._initial_poses[name]
                    pos_tensor = torch.tensor(
                        pos, dtype=torch.float, device=device
                    ).repeat(len(env_ids), 1)
                    rot_tensor = torch.tensor(
                        rot, dtype=torch.float, device=device
                    ).repeat(len(env_ids), 1)
                else:
                    if not active_objects.get(name, True):
                        pos_tensor, rot_tensor = self._make_parking_tensors(
                            name, env_ids, device
                        )
                    else:
                        # user_pos, user_rot = new_poses[name]
                        # # if meta.get("random_rotation"):
                        # #     user_rot = _apply_random_rotation(
                        # #         user_rot, meta["random_rotation"], rng
                        # #     )
                        # final_rot = _quat_mul(user_rot, meta["usd_local_rotation"])
                        # final_rot /= np.linalg.norm(final_rot)
                        # pos_tensor = torch.tensor(
                        #     user_pos, dtype=torch.float, device=device
                        # ).repeat(len(env_ids), 1)
                        # rot_tensor = torch.tensor(
                        #     final_rot.tolist(), dtype=torch.float, device=device
                        # ).repeat(len(env_ids), 1)
                        user_pos, user_rot = new_poses[name]
                        anchor_prim = self._get_group_anchor_prim(meta)

                        if anchor_prim is not None:
                            pos_tensor, rot_tensor = (
                                self._make_anchored_root_pose_tensors(
                                    scene=scene,
                                    meta=meta,
                                    local_pos=user_pos,
                                    local_rot=user_rot,
                                    env_ids=env_ids,
                                    device=device,
                                )
                            )
                        else:
                            final_rot = _quat_mul(user_rot, meta["usd_local_rotation"])
                            final_rot /= np.linalg.norm(final_rot)

                            pos_tensor = torch.tensor(
                                user_pos,
                                dtype=torch.float,
                                device=device,
                            ).repeat(len(env_ids), 1)

                            rot_tensor = torch.tensor(
                                final_rot.tolist(),
                                dtype=torch.float,
                                device=device,
                            ).repeat(len(env_ids), 1)

                root_state = obj.data.default_root_state.clone().to(device)
                root_state[env_ids, :3] = pos_tensor
                root_state[env_ids, 3:7] = rot_tensor
                root_state[env_ids, 7:] = 0.0
                obj.write_root_state_to_sim(root_state, env_ids=env_ids)

        if self.robot_cfg is not None:
            robot_key = self.robot_cfg.get("name", "robot")
            robot_art = scene.articulations.get(robot_key)

            if robot_art is not None:
                n_envs = len(env_ids)

                root_state = (
                    robot_art.data.default_root_state[env_ids].clone().to(device)
                )

                for i, env_id in enumerate(env_ids):
                    pos, rot = _generate_robot_spawn_pose(self.robot_cfg, rng)

                    pos_t = torch.tensor(pos, dtype=torch.float, device=device)
                    rot_t = torch.tensor(rot, dtype=torch.float, device=device)

                    # If the JSON specifies a position locally relative to the env
                    if hasattr(scene, "env_origins"):
                        pos_t = pos_t + scene.env_origins[env_id].to(device)

                    root_state[i, 0:3] = pos_t
                    root_state[i, 3:7] = rot_t
                    root_state[i, 7:13] = 0.0

                root_state = root_state.contiguous()

                robot_art.write_root_state_to_sim(root_state, env_ids=env_ids)

                print("[SPAWNER] robot randomized root pose written")
                print("env_ids:", env_ids)
                print("first root pos:", root_state[0, 0:3].detach().cpu().tolist())
                print("first root rot:", root_state[0, 3:7].detach().cpu().tolist())
            else:
                print(f"[SPAWNER] Robot '{robot_key}' not found in scene.articulations")
                print("Available articulations:", list(scene.articulations.keys()))

        # Apply visibility/collisions AFTER moving objects
        # self._apply_collision_state(active_objects, env_ids, enable=True)
        self._apply_visibility_state(active_objects, env_ids)

        self.update_robot_spawn_visuals(scene, env_ids)
        self.update_group_visuals(scene, env_ids)
        self.update_guide_objects(scene, env_ids)

    def is_object_active(self, object_name: str) -> bool:
        """Checks whether the object was active at the last reset."""
        return self.last_active_objects.get(object_name, True)

    def _apply_visibility_state(
        self, active_objects: Dict[str, bool], env_ids: torch.Tensor
    ):
        stage = omni.usd.get_context().get_stage()

        for env_id in env_ids.tolist():
            env_id = int(env_id)
            for obj_name, is_active in active_objects.items():
                if self.object_metadata[obj_name]["fixed"]:
                    continue

                full_path = self._prim_paths_cache[obj_name][env_id]
                root_prim = stage.GetPrimAtPath(full_path)
                if not root_prim.IsValid():
                    continue

                try:
                    imageable = UsdGeom.Imageable(root_prim)
                    if imageable:
                        if is_active:
                            imageable.MakeVisible()
                        else:
                            imageable.MakeInvisible()
                except Exception as e:
                    print(f"Warning: Failed to set visibility for {full_path}: {e}")

    def _resolve_guide_color(self, color_name: str) -> tuple[float, float, float]:
        named_colors = {
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
            "white": (1.0, 1.0, 1.0),
            "black": (0.0, 0.0, 0.0),
            "orange": (1.0, 0.5, 0.0),
            "purple": (0.5, 0.0, 1.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
        }
        return named_colors.get(str(color_name).lower(), named_colors["red"])

    def spawn_scene_static_objects(self, num_envs: int):
        """
        Spawns ready-made composite USD objects directly into the stage,
        without creating RigidObjectCfg/ArticulationCfg for them.

        Use BEFORE creating the InteractiveScene.
        """
        for env_id in range(num_envs):
            for name, meta in self.object_metadata.items():
                if meta.get("object_kind") == "guide":
                    continue
                if not meta.get("scene_static", False):
                    continue

                prim_path = f"/World/envs/env_{env_id}/{name}"

                user_pos = (
                    meta["position"]
                    if meta["position"] is not None
                    else [0.0, 0.0, 0.0]
                )
                user_rot = (
                    meta["rotation"]
                    if meta["rotation"] is not None
                    else [1.0, 0.0, 0.0, 0.0]
                )

                final_rot = _quat_mul(user_rot, meta["usd_local_rotation"])
                final_rot /= np.linalg.norm(final_rot)

                fixed_usd_path = _rewrite_usd_mdl_paths_to_package_paths(
                    meta["usd_path"]
                )

                cfg = sim_utils.UsdFileCfg(
                    usd_path=fixed_usd_path,
                    scale=meta["scale"],
                )

                spawn_from_usd(
                    prim_path=prim_path,
                    cfg=cfg,
                    translation=user_pos,
                    orientation=final_rot.tolist(),
                )

                print(
                    f"[SCENE_STATIC] spawned: env={env_id} name={name} -> {prim_path}"
                )

    def spawn_lighting(self) -> None:
        lighting_spawner = LightingSpawner(self.config, prim_root="/World/Light")
        lighting_spawner.spawn()

        self.light_metadata = lighting_spawner.light_metadata
        self.light_visual_metadata = lighting_spawner.light_visual_metadata

    def _get_viewer_metadata(self, config: dict) -> dict:
        viewer = config.get("viewer") or {}
        if not isinstance(viewer, dict):
            raise ValueError("viewer must be an object")

        origin_type = viewer.get("origin_type", "world")
        if origin_type not in {"world", "env", "asset_root", "asset_body"}:
            raise ValueError(f"invalid viewer.origin_type: {origin_type}")

        return {
            "origin_type": origin_type,
            "eye": tuple(viewer["eye"]) if "eye" in viewer else None,
            "lookat": tuple(viewer["lookat"]) if "lookat" in viewer else None,
            "env_index": viewer.get("env_index"),
            "asset_name": viewer.get("asset_name"),
            "body_name": viewer.get("body_name"),
        }

    def apply_viewer(self, viewport_camera_controller) -> None:
        viewer = self.viewer_metadata
        if not viewer:
            return

        c = viewport_camera_controller

        if viewer["env_index"] is not None:
            c.set_view_env_index(viewer["env_index"])

        origin_type = viewer["origin_type"]

        if origin_type == "world":
            c.update_view_to_world()
        elif origin_type == "env":
            c.update_view_to_env()
        elif origin_type == "asset_root":
            c.update_view_to_asset_root(viewer["asset_name"])
        elif origin_type == "asset_body":
            c.update_view_to_asset_body(
                viewer["asset_name"],
                viewer["body_name"],
            )
        else:
            raise ValueError(f"Unsupported viewer origin_type: {origin_type}")

        eye = viewer.get("eye")
        lookat = viewer.get("lookat")

        if eye is not None and lookat is not None:
            c.update_view_location(eye, lookat)

    def update_group_visuals(
        self,
        scene,
        env_ids: Optional[torch.Tensor] = None,
    ):
        stage = omni.usd.get_context().get_stage()

        if env_ids is None:
            env_ids_list = list(range(scene.num_envs))
        else:
            env_ids_list = [int(env_id) for env_id in env_ids.tolist()]

        for group_name, group_cfg in self.group_templates.items():
            group_visible = group_cfg.get("visible", None)
            base_position = group_cfg.get("_resolved_base_position")

            if base_position is None:
                preview_rng = np.random.default_rng(0)
                base_position = _resolve_group_base_position(
                    group_name=group_name,
                    group_templates=self.group_templates,
                    object_metadata=self.object_metadata,
                    rng=preview_rng,
                    resolved_positions={},
                    resolving_groups=set(),
                )
                group_cfg["_resolved_base_position"] = base_position

            size = group_cfg.get("size")
            local_rotation = group_cfg.get(
                "rotation",
                [1.0, 0.0, 0.0, 0.0],
            )
            anchor_prim = group_cfg.get("anchor_prim")

            paths = self._group_visual_prim_paths.setdefault(group_name, [])

            for env_id in env_ids_list:
                anchor_prim = group_cfg.get("anchor_prim")

                if anchor_prim is None:
                    prim_path = f"/World/envs/env_{env_id}/group_visuals/{group_name}"
                else:
                    prim_path = (
                        f"{self._get_env_prim_path(anchor_prim, env_id)}"
                        f"/group_visuals/{group_name}"
                    )

                if env_id >= len(paths):
                    paths.extend([""] * (env_id + 1 - len(paths)))
                paths[env_id] = prim_path

                visual_position = list(base_position)
                visual_rotation = list(local_rotation)

                new_size = (
                    [float(size[0]), float(size[1])] if size is not None else None
                )

                # Cache per environment: a moving anchor has a different
                # world transform for every env and every reset/step.
                visual_state = group_cfg.setdefault(
                    "_visual_state_by_env",
                    {},
                )
                old_state = visual_state.get(env_id)

                current_state = {
                    "size": new_size,
                    "position": visual_position,
                    "rotation": visual_rotation,
                }

                need_respawn = old_state != current_state

                prim = stage.GetPrimAtPath(prim_path)

                if group_visible is None:
                    if prim.IsValid():
                        imageable = UsdGeom.Imageable(prim)
                        if imageable:
                            imageable.MakeInvisible()
                    visual_state[env_id] = current_state
                    continue

                if need_respawn and prim.IsValid():
                    stage.RemovePrim(prim_path)
                    prim = stage.GetPrimAtPath(prim_path)

                if not prim.IsValid():
                    if size is None:
                        continue

                    rect_cfg = sim_utils.CuboidCfg(
                        size=(float(size[0]), float(size[1]), 0.003),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 0.6, 1.0),
                            opacity=0.35,
                        ),
                    )

                    rect_cfg.func(
                        prim_path,
                        rect_cfg,
                        translation=tuple(visual_position),
                        orientation=tuple(visual_rotation),
                    )

                    prim = stage.GetPrimAtPath(prim_path)

                if prim.IsValid():
                    imageable = UsdGeom.Imageable(prim)
                    if imageable:
                        if group_visible:
                            imageable.MakeVisible()
                        else:
                            imageable.MakeInvisible()

                visual_state[env_id] = current_state

    def update_guide_objects(self, scene, env_ids: Optional[torch.Tensor] = None):
        stage = omni.usd.get_context().get_stage()

        if env_ids is None:
            env_ids_list = list(range(scene.num_envs))
        else:
            env_ids_list = [int(env_id) for env_id in env_ids.tolist()]

        for name, meta in self.object_metadata.items():
            if meta.get("object_kind") != "guide":
                continue

            guide_position = (
                meta["position"] if meta["position"] is not None else [0.0, 0.0, 0.0]
            )
            guide_rotation = (
                meta["rotation"]
                if meta["rotation"] is not None
                else [1.0, 0.0, 0.0, 0.0]
            )
            guide_size_value = meta.get("size", [0.2, 0.2, 0.2])
            guide_radius = meta.get("guide_radius")
            guide_color = self._resolve_guide_color(meta.get("color", "red"))
            guide_visible = bool(meta.get("visible", True))
            # guide_visible = (
            #     bool(meta.get("visible", True))
            #     and self.last_active_objects.get(name, True)
            # )

            new_size = [
                float(guide_size_value[0]),
                float(guide_size_value[1]),
                float(guide_size_value[2]),
            ]
            new_position = list(guide_position)
            new_rotation = list(guide_rotation)
            new_color = list(guide_color)

            old_size = meta.get("_last_visual_size")
            old_position = meta.get("_last_visual_position")
            old_rotation = meta.get("_last_visual_rotation")
            old_color = meta.get("_last_visual_color")

            need_respawn = (
                old_size != new_size
                or old_position != new_position
                or old_rotation != new_rotation
                or old_color != new_color
            )

            for env_id in env_ids_list:
                # prim_path = f"/World/envs/env_{env_id}/{name}"
                prim_path = self._get_guide_prim_path(name, meta, env_id)
                prim = stage.GetPrimAtPath(prim_path)

                if need_respawn and prim.IsValid():
                    stage.RemovePrim(prim_path)
                    prim = stage.GetPrimAtPath(prim_path)

                if not prim.IsValid():
                    if guide_radius is not None:
                        guide_cfg = sim_utils.SphereCfg(
                            radius=float(guide_radius),
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=guide_color,
                                opacity=1.0,
                            ),
                        )
                    else:
                        guide_cfg = sim_utils.CuboidCfg(
                            size=tuple(new_size),
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=guide_color,
                                opacity=1.0,
                            ),
                        )

                    guide_cfg.func(
                        prim_path,
                        guide_cfg,
                        translation=tuple(guide_position),
                        orientation=tuple(guide_rotation),
                    )
                    prim = stage.GetPrimAtPath(prim_path)

                if prim.IsValid():
                    imageable = UsdGeom.Imageable(prim)
                    if imageable:
                        if guide_visible:
                            imageable.MakeVisible()
                        else:
                            imageable.MakeInvisible()

            meta["_last_visual_size"] = new_size
            meta["_last_visual_position"] = new_position
            meta["_last_visual_rotation"] = new_rotation
            meta["_last_visual_color"] = new_color

    def update_robot_spawn_visuals(
        self,
        scene,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Visualizes robot.spawn_areas in each environment."""

        if self.robot_cfg is None:
            return

        stage = omni.usd.get_context().get_stage()
        areas = _get_robot_spawn_areas(self.robot_cfg)

        if env_ids is None:
            env_ids_list = list(range(scene.num_envs))
        else:
            env_ids_list = [int(env_id) for env_id in env_ids.tolist()]

        visual_state = self.robot_cfg.setdefault(
            "_robot_spawn_visual_state_by_env",
            {},
        )

        for area_index, area in enumerate(areas):
            center = area["center"]
            size_xy = area["size_xy"]
            visible = bool(area.get("visible", True))

            sx = float(size_xy[0])
            sy = float(size_xy[1])

            # CuboidCfg does not allow zero dimensions.
            # size_xy=[0, 0] is interpreted as a point spawn.
            visual_kind = "point" if sx <= 1e-8 or sy <= 1e-8 else "rectangle"

            # If the area needs to be drawn at a height other than the robot root,
            # add visual_z_offset to the JSON.
            visual_z_offset = float(area.get("visual_z_offset", 0.0))

            visual_position = [
                float(center[0]),
                float(center[1]),
                float(center[2]) + visual_z_offset,
            ]

            current_state = {
                "kind": visual_kind,
                "position": visual_position,
                "size_xy": [sx, sy],
                "visible": visible,
            }

            for env_id in env_ids_list:
                prim_path = (
                    f"/World/envs/env_{env_id}/robot_spawn_visuals/"
                    f"area_{area_index}"
                )

                state_key = f"area_{area_index}:env_{env_id}"
                old_state = visual_state.get(state_key)

                prim = stage.GetPrimAtPath(prim_path)

                # When the JSON changes, remove the old geometry:
                # this is important when transitioning point <-> rectangle.
                if old_state != current_state and prim.IsValid():
                    stage.RemovePrim(prim_path)
                    prim = stage.GetPrimAtPath(prim_path)

                if not prim.IsValid():
                    material = sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.45, 0.0),
                        opacity=0.45,
                    )

                    if visual_kind == "point":
                        marker_cfg = sim_utils.SphereCfg(
                            radius=0.025,
                            visual_material=material,
                        )

                        marker_cfg.func(
                            prim_path,
                            marker_cfg,
                            translation=tuple(visual_position),
                            orientation=(1.0, 0.0, 0.0, 0.0),
                        )

                    else:
                        rect_cfg = sim_utils.CuboidCfg(
                            size=(sx, sy, 0.002),
                            visual_material=material,
                        )

                        rect_cfg.func(
                            prim_path,
                            rect_cfg,
                            translation=tuple(visual_position),
                            orientation=(1.0, 0.0, 0.0, 0.0),
                        )

                    prim = stage.GetPrimAtPath(prim_path)

                if prim.IsValid():
                    imageable = UsdGeom.Imageable(prim)

                    if imageable:
                        if visible:
                            imageable.MakeVisible()
                        else:
                            imageable.MakeInvisible()

                visual_state[state_key] = current_state


print("Loading asssets..")
