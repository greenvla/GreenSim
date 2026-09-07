# Copyright (c) 2026 Sber Robotics Center. All rights reserved.
# All rights reserved.
from __future__ import annotations
from typing import TYPE_CHECKING, Type
import os
import shutil
import math

import re
import defusedxml.ElementTree as ET
from collections import deque

from isaaclab.sim.converters import MjcfConverter
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import (
    clone,
    make_uninstanceable,
    bind_visual_material,
    get_all_matching_child_prims,
)

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, PhysxSchema
except ImportError:
    raise ImportError(
        "usd-core failed to import, install with pip install usd-core"
        + " NOTE: Do not install this if using with ISAAC SIM."
    )

if TYPE_CHECKING:
    from .mjcf_cfg import MJCFFileCfg


def _indent(elem, level=0):
    """Pretty-print an ElementTree element in place (drop-in for ET.indent)."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def preprocess_robot_menagerie_mjcf(mjcf_path: str) -> str:

    def sanitize(s: str) -> str:
        s = re.sub(r"[^\w\-]", "_", s)
        return re.sub(r"_+", "_", s).strip("_") or "unnamed"

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    base_dir = os.path.dirname(os.path.abspath(mjcf_path))

    # Remove meshdir and get its value
    meshdir = ""
    for compiler in root.findall(".//compiler"):
        if "meshdir" in compiler.attrib:
            meshdir = compiler.attrib["meshdir"].rstrip("/\\")
            del compiler.attrib["meshdir"]

    mesh_abs_dir = os.path.join(base_dir, meshdir) if meshdir else base_dir
    os.makedirs(mesh_abs_dir, exist_ok=True)

    # Process meshes
    asset = root.find("asset")
    if asset is not None:
        for mesh in asset.findall("mesh"):
            name = mesh.get("name")
            src_file = mesh.get("file")
            if not name or not src_file:
                continue

            src_path = (
                os.path.join(base_dir, meshdir, src_file)
                if meshdir
                else os.path.join(base_dir, src_file)
            )
            if not os.path.exists(src_path):
                continue

            dst_file = sanitize(name) + ".stl"
            dst_path = os.path.join(mesh_abs_dir, dst_file)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

            mesh.set(
                "file",
                (
                    os.path.join(meshdir, dst_file).replace("\\", "/")
                    if meshdir
                    else dst_file
                ),
            )

    # Remove all connect tags
    for elem in root.iter():
        connects_to_remove = [child for child in elem if child.tag == "connect"]
        for conn in connects_to_remove:
            elem.remove(conn)
    # Save to cache
    cache_dir = os.path.join(
        os.path.expanduser("~/.cache/dirol"), os.path.basename(base_dir)
    )
    os.makedirs(cache_dir, exist_ok=True)
    shutil.copytree(base_dir, cache_dir, dirs_exist_ok=True)

    out_path = os.path.join(cache_dir, "tmp_" + os.path.basename(mjcf_path))
    _indent(root)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def get_mjcf_equality_constraints(mjcf_path: str) -> dict:
    """
    Removes floating base joint and related links
    Returns preprocessed temporary file path
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    constraints = {}
    for child in root.findall(".//equality/connect"):
        constraint_name = child.attrib["name"]
        constraints[constraint_name] = {}
        constraints[constraint_name]["body1"] = child.attrib["body1"]
        constraints[constraint_name]["body2"] = child.attrib["body2"]
        constraints[constraint_name]["anchor"] = list(
            map(float, child.attrib["anchor"].split(" "))
        )
    return constraints


def get_mjcf_articulated_joints(mjcf_path: str) -> list:
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    joints = []
    for child in root.findall(".//motor"):
        joints.append(child.attrib["joint"])
    for child in root.findall(".//general"):
        if child.attrib["joint"] not in joints:
            joints.append(child.attrib["joint"])
    return joints


def find_prim(stage: Usd.Stage, name: str) -> Usd.Prim | None:
    """
    Search recursively for a prim in the USD stage by its name.

    Args:
        stage: The USD stage to search within.
        name: The target prim name.

    Returns:
        The first prim with the specified name or None if not found.
    """
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


def get_mjcf_joint_param_dicts(mjcf_path: str) -> dict:
    articulated_joints = get_mjcf_articulated_joints(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    params = {"damping": {}, "frictionloss": {}}

    for child in root.findall(".//joint"):
        joint_name = child.attrib["name"]
        if joint_name in articulated_joints:
            params["damping"][joint_name] = float(child.attrib.get("damping", 0.0))
            params["frictionloss"][joint_name] = float(
                child.attrib.get("frictionloss", 0.0)
            )
    return params

def get_mjcf_mimic_info(mjcf_path: str) -> dict[str, dict[str, object]]:
    """
    Extracts mimic joint information from an MJCF file.

    This function parses the MJCF XML file, collects the axes of all declared
    joints, and then inspects the `<equality>` section for joint constraints
    whose names end with `_mimic`. For each such constraint, it builds a
    dictionary describing the slave-master mimic relationship, including the
    multiplier, offset, and both joint axes.

    Args:
        mjcf_path (str): Path to the MJCF XML file.

    Returns:
        dict[str, dict[str, object]]: A mapping from slave joint name to mimic
        metadata. Each entry contains:
            - "mimic_master_joint": Name of the master joint.
            - "mimic_multiplier": Linear multiplier applied to the master joint.
            - "mimic_offset": Constant offset applied to the mimic relation.
            - "slave_axis": Axis of the slave joint.
            - "master_axis": Axis of the master joint.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    mimic_info = {}

    # Collect axes for all joints.
    joint_axes = {}
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        axis_str = joint.get("axis", "0 0 1")  # MJCF default axis is Z
        joint_axes[name] = [float(x) for x in axis_str.strip().split()]

    equality_elem = root.find("equality")
    if equality_elem is not None:
        for joint in equality_elem.findall("joint"):
            joint_name = joint.attrib.get("name", "")
            if joint_name.endswith("_mimic"):
                master_joint = joint.attrib.get("joint1", "")
                slave_joint = joint.attrib.get("joint2", "")
                polycoef_str = joint.attrib.get("polycoef", "")

                offset, multiplier = 0.0, 1.0
                if polycoef_str:
                    try:
                        polycoef = [float(x) for x in polycoef_str.strip().split()]
                        if len(polycoef) >= 2:
                            offset, multiplier = polycoef[0], polycoef[1]
                        elif len(polycoef) == 1:
                            offset, multiplier = 0.0, polycoef[0]
                    except ValueError:
                        print(
                            f"[get_mjcf_mimic_info] Invalid polycoef '{polycoef_str}'"
                        )

                mimic_info[slave_joint] = {
                    "mimic_master_joint": master_joint,
                    "mimic_multiplier": multiplier,
                    "mimic_offset": offset,
                    "slave_axis": joint_axes.get(slave_joint, [0, 0, 1]),
                    "master_axis": joint_axes.get(master_joint, [0, 0, 1]),
                }
    else:
        print("[get_mjcf_mimic_info] No <equality> tag found.")

    return mimic_info

def insert_fixed_joints(mjcf_path: str) -> str:
    """
    Insert fixed joints into MJCF bodies lacking joints but having geometry,
    and save the updated MJCF XML to a new file.

    Args:
        mjcf_path (str): Path to the original MJCF XML file.

    Returns:
        str: Path to the new MJCF file with fixed joints inserted.
    """
    print(f"[FixedJointMjcfConverter] Processing MJCF for fixed joints: {mjcf_path}")
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        print(
            "[FixedJointMjcfConverter] No <worldbody> element; aborting fixed joint insertion."
        )
        return mjcf_path  # Return original path if no modification

    def has_joint(body: ET.Element) -> bool:
        return any(child.tag == "joint" for child in body)

    def has_geom(body: ET.Element) -> bool:
        return any(child.tag == "geom" for child in body)

    def recursive_process(body: ET.Element) -> None:
        if has_geom(body) and not has_joint(body):
            body_name = body.attrib.get("name")
            if body_name:
                joint_name = f"{body_name}_fixed_joint"
                print(
                    f"[FixedJointMjcfConverter] Adding fixed joint '{joint_name}' to body '{body_name}'"
                )
                ET.SubElement(
                    body,
                    "joint",
                    {
                        "name": joint_name,
                        "type": "hinge",
                        "pos": "0 0 0",
                        "axis": "1 0 0",
                        "limited": "true",
                        "range": "0 0.0000000000000001",
                        "damping": "100",
                    },
                )
        for child in body:
            if child.tag == "body":
                recursive_process(child)

    recursive_process(worldbody)

    # def remove_mimic_joints_from_equality(element: ET.Element):
    #     to_remove = []
    #     for child in list(element):
    #         # Check if this is a <joint> tag with the 'name' attribute ending with '_mimic'
    #         if child.tag == "joint" and child.attrib.get("name", "").endswith("_mimic"):
    #             print(f"Removing <joint> with mimic suffix: {child.attrib.get('name')}")
    #             to_remove.append(child)
    #         else:
    #             # Recurse into child elements to find nested joints
    #             remove_mimic_joints_from_equality(child)
    #     for child in to_remove:
    #         element.remove(child)

    # # Use this function on every <equality> element in root
    # for equality in root.findall("equality"):
    #     remove_mimic_joints_from_equality(equality)

    base, ext = os.path.splitext(mjcf_path)
    fixed_file_path = base + "_fixed.xml"
    tree.write(fixed_file_path, encoding="utf-8", xml_declaration=True)
    print(f"[FixedJointMjcfConverter] Fixed MJCF saved to {fixed_file_path}")

    return fixed_file_path


def _explode_instancing(robot_prim: Usd.Prim) -> None:
    """Break every instance below robot_prim so meshes are real prims."""
    stage = robot_prim.GetStage()
    queue = deque([robot_prim])
    while queue:
        prim = queue.popleft()
        if not prim or not prim.IsValid():
            continue
        if prim.IsInstance():
            path = prim.GetPath().pathString
            make_uninstanceable(path, stage=stage)
            prim = stage.GetPrimAtPath(path)  # refresh handle
        queue.extend(prim.GetChildren())


def _repair_instanced_material_bindings(robot_prim: Usd.Prim) -> None:
    """
    Breaks USD instancing and rebinds the proper MJCF visual materials after spawning.

    When spawning MJCF converted assets, USD instancing can sometimes cause
    material bindings to be lost or incorrectly evaluated on instanceable prims.
    This function iterates through all meshes within the robot's hierarchy,
    disables instancing on their closest instanceable ancestor, and explicitly
    rebinds their previously resolved visual materials.

    Args:
        robot_prim (Usd.Prim): The root USD Prim of the spawned robot.

    Returns:
        None
    """
    stage = robot_prim.GetStage()

    def _is_mesh(prim: Usd.Prim) -> bool:
        """Predicate helper to verify if a given prim is a USD Mesh."""
        return UsdGeom.Mesh(prim) is not None

    # 1. Retrieve all mesh prims under the robot hierarchy
    mesh_prims = get_all_matching_child_prims(
        robot_prim.GetPath().pathString,
        predicate=_is_mesh,
        stage=stage,
        # traverse_instance_prims=True,
    )

    if not mesh_prims:
        print("[spawn_from_mjcf] No mesh prims found; nothing to fix")
        return

    uninstanced = set()

    # 2. Iterate through all discovered meshes to repair their bindings
    for mesh_prim in mesh_prims:
        if not mesh_prim or not mesh_prim.IsValid():
            continue

        mesh_path = mesh_prim.GetPath().pathString

        # 3. Resolve whichever material the MJCF converter had previously assigned
        binding_api = UsdShade.MaterialBindingAPI(mesh_prim)
        resolved_material, _ = binding_api.ComputeBoundMaterial()

        if not resolved_material:
            continue  # The mesh never had a material assigned in the MJCF

        material_path = resolved_material.GetPath().pathString

        # 4. Find the closest instanceable ancestor (which could be the mesh itself)
        instance_root = mesh_prim
        while instance_root and not instance_root.IsInstanceable():
            instance_root = instance_root.GetParent()

        # 5. Break instancing on the ancestor if it hasn't been done yet
        if instance_root:
            instance_path = instance_root.GetPath().pathString
            if instance_path not in uninstanced:
                make_uninstanceable(instance_path, stage=stage)
                uninstanced.add(instance_path)

        # 6. Rebind the resolved material directly back onto this specific mesh
        bind_visual_material(mesh_path, material_path, stage=stage)
        # print(f"[spawn_from_mjcf] Bound {material_path} to {mesh_path}")


def set_convex_decomposition_for_all_collision_meshes(
    stage: Usd.Stage,
    root_prim: Usd.Prim | None = None,
    max_hulls: int = 16,
    hull_vertex_limit: int = 64,
) -> int:
    """Sets convex decomposition approximation for all collision meshes in a USD stage.

    Iterates through all Prims of type Mesh that have CollisionAPI enabled and sets
    physics:approximation to "convexDecomposition". Additionally, attempts to apply
    PhysX-specific decomposition parameters (max hulls and vertex limit) if the
    PhysxSchema is available in the current environment.

    Args:
        stage (Usd.Stage): The USD stage to traverse.
        root_prim (Usd.Prim | None): Optional root Prim to start traversal from.
            If provided and valid, traversal begins from this Prim. Otherwise,
            the entire stage is traversed. Defaults to None.
        max_hulls (int): Maximum number of convex hulls to generate during
            decomposition. Used for PhysX schema configuration. Defaults to 16.
        hull_vertex_limit (int): Maximum number of vertices allowed per convex
            hull. Used for PhysX schema configuration. Defaults to 64.

    Returns:
        int: The number of collision meshes successfully patched.

    Note:
        This function modifies the USD stage in-place.
        PhysX schema application is wrapped in a try-except block to ensure
        compatibility across different USD/PhysX versions where specific
        attributes might not exist.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

    patched = 0
    it = (
        Usd.PrimRange(root_prim)
        if root_prim and root_prim.IsValid()
        else stage.Traverse()
    )

    for prim in it:
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not UsdPhysics.CollisionAPI(prim):
            continue

        # USD Physics
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_api.CreateApproximationAttr().Set("convexDecomposition")

        # Optionally apply PhysX parameters (if available in this version)
        try:
            decomp = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            if hasattr(decomp, "CreateMaxConvexHullsAttr"):
                decomp.CreateMaxConvexHullsAttr().Set(max_hulls)
            if hasattr(decomp, "CreateHullVertexLimitAttr"):
                decomp.CreateHullVertexLimitAttr().Set(hull_vertex_limit)
        except Exception:
            pass

        patched += 1

    print(
        f"[spawn_from_mjcf] convexDecomposition applied to {patched} collision meshes"
    )
    return patched


def _fix_duplicate_articulation_roots(robot_prim: Usd.Prim) -> None:
    """
    Resolves duplicate articulation roots within a robot's USD prim hierarchy.

    This function traverses the prim hierarchy starting from `robot_prim` to find
    all prims with the UsdPhysics.ArticulationRootAPI applied. If multiple roots
    are found, it prioritizes keeping the one whose path ends with "/root/root".
    It then strips the articulation and PhysX APIs from all other duplicates,
    re-applies them to the chosen primary root, enables mimic joints, and saves
    the USD stage.

    Args:
        robot_prim (Usd.Prim): The root USD Prim of the robot to process.

    Returns:
        None
    """
    articulation_roots = []

    # 1. Identify all prims containing an ArticulationRootAPI
    for p in Usd.PrimRange(robot_prim):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(p)

    print("[articulation fix] found roots:")
    for p in articulation_roots:
        print(f"   {p.GetPath()}")

    if not articulation_roots:
        print("[articulation fix] ERROR: no articulation roots found")
        return

    # 2. Determine which prim to keep as the sole articulation root
    # Prioritize keeping the prim with the "/root/root" suffix
    keep_prim = None
    for p in articulation_roots:
        if p.GetPath().pathString.endswith("/root/root"):
            keep_prim = p
            break

    # Fallback to the first discovered root if "/root/root" is not present
    if keep_prim is None:
        keep_prim = articulation_roots[0]

    print(f"[articulation fix] keep: {keep_prim.GetPath()}")

    # 3. Remove articulation APIs from all other duplicate roots
    for p in articulation_roots:
        if p.GetPath() == keep_prim.GetPath():
            continue

        print(f"[articulation fix] removing ArticulationRootAPI from: {p.GetPath()}")

        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            p.RemoveAPI(UsdPhysics.ArticulationRootAPI)

        if p.HasAPI(PhysxSchema.PhysxArticulationAPI):
            p.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

        # Clear the mimic joints attribute on duplicates if it exists
        attr = p.GetAttribute("physxArticulation:enabledMimicJoints")
        if attr:
            attr.Clear()

    # 4. Enforce required APIs on the selected primary root
    UsdPhysics.ArticulationRootAPI.Apply(keep_prim)
    PhysxSchema.PhysxArticulationAPI.Apply(keep_prim)

    # 5. Enable mimic joints for the primary articulation root
    attr = keep_prim.CreateAttribute(
        "physxArticulation:enabledMimicJoints",
        Sdf.ValueTypeNames.Bool,
        custom=False,
    )
    attr.Set(True)

    print(f"[articulation fix] enabled mimic on: {keep_prim.GetPath()} {attr.Get()}")

    # 6. Save the stage to persist API removals and attribute changes
    robot_prim.GetStage().Save()


def _apply_physx_mimic_joints(
    robot_prim: Usd.Prim,
    mimic_info: dict[str, dict[str, object]],
) -> None:
    """
    Configures PhysX mimic joints for a robot based on provided mimic information.

    This function traverses the USD hierarchy to enable mimic joints on the
    articulation root, maps all revolute joints, and applies the PhysxMimicJointAPI
    to slave joints. It handles Isaac Sim specific axis conversions (inversions)
    from MJCF and ensures that competing Drive APIs are removed from slave joints
    so the mimic constraints function correctly.

    Args:
        robot_prim (Usd.Prim): The root USD Prim of the robot.
        mimic_info (dict): A dictionary mapping slave joint names to their metadata.
            Expected metadata keys include 'mimic_master_joint', 'mimic_multiplier',
            and 'mimic_offset'.

    Returns:
        None
    """
    stage = robot_prim.GetStage()

    def _usd_axis_to_mimic_token(joint_prim: Usd.Prim) -> str:
        """Helper to convert a joint's local axis to a UsdPhysics token."""
        joint = UsdPhysics.RevoluteJoint(joint_prim)
        axis = joint.GetAxisAttr().Get()
        axis_str = str(axis).lower()

        if axis_str == "x":
            return UsdPhysics.Tokens.rotX
        if axis_str == "y":
            return UsdPhysics.Tokens.rotY
        if axis_str == "z":
            return UsdPhysics.Tokens.rotZ

        raise RuntimeError(f"[mimic] bad axis for {joint_prim.GetPath()}: {axis}")

    # 1. Enable mimic joints on the articulation root BEFORE creating PhysX articulation
    found_articulation = False
    for p in Usd.PrimRange(robot_prim):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            if not p.HasAPI(PhysxSchema.PhysxArticulationAPI):
                PhysxSchema.PhysxArticulationAPI.Apply(p)

            attr = p.GetAttribute("physxArticulation:enabledMimicJoints")
            if not attr:
                attr = p.CreateAttribute(
                    "physxArticulation:enabledMimicJoints",
                    Sdf.ValueTypeNames.Bool,
                    custom=False,
                )
            attr.Set(True)

            print(f"[mimic] enabledMimicJoints on {p.GetPath()} = {attr.Get()}")
            found_articulation = True

    if not found_articulation:
        print(f"[mimic] WARNING: no ArticulationRootAPI under {robot_prim.GetPath()}")

    # 2. Build a map of all Revolute Joints by their names
    joint_map: dict[str, Usd.Prim] = {}
    for prim in Usd.PrimRange(robot_prim):
        if prim.GetTypeName() == "PhysicsRevoluteJoint":
            joint_map[prim.GetName()] = prim

    # 3. Process each slave joint defined in the mimic_info dictionary
    for slave_name, meta in mimic_info.items():
        master_name = str(meta["mimic_master_joint"])

        slave_prim = joint_map.get(slave_name)
        master_prim = joint_map.get(master_name)

        if not slave_prim or not slave_prim.IsValid():
            print(f"[mimic] slave not found: {slave_name}")
            continue

        if not master_prim or not master_prim.IsValid():
            print(f"[mimic] master not found: {master_name}")
            continue

        slave_axis_token = _usd_axis_to_mimic_token(slave_prim)
        master_axis_token = _usd_axis_to_mimic_token(master_prim)

        # 4. Calculate gearing (multiplier) and offset
        # The Isaac Sim converter redefines axes to local X by rotating phalanx frames.
        # Because of this, the local rotation direction is inverted relative to MJCF.
        multiplier = -1.0 * float(meta.get("mimic_multiplier", 1.0))

        # Convert the offset from radians (MJCF) to degrees (USD/PhysX)
        offset = -1.0 * float(meta.get("mimic_offset", 0.0)) * (180.0 / math.pi)

        # 5. IMPORTANT: Completely remove all drives from the slave joint.
        # If DriveAPI:angular remains, PhysX will disable the mimic constraint,
        # and the fingers will hang limply.
        drive_instances = ["angular", "linear", "rotX", "rotY", "rotZ", "X", "Y", "Z"]
        for inst in drive_instances:
            if slave_prim.HasAPI(UsdPhysics.DriveAPI, inst):
                slave_prim.RemoveAPI(UsdPhysics.DriveAPI, inst)
                print(
                    f"[mimic] REMOVED DriveAPI:{inst} from slave {slave_prim.GetPath()}"
                )

        # 6. Widen the limits on the slave joint to prevent restriction of the mimic motion
        slave_joint = UsdPhysics.RevoluteJoint(slave_prim)
        if slave_joint.GetLowerLimitAttr():
            slave_joint.GetLowerLimitAttr().Set(-100.0)
        if slave_joint.GetUpperLimitAttr():
            slave_joint.GetUpperLimitAttr().Set(100.0)

        # 7. Apply the Mimic API strictly to the physical axis (rotX, rotY, rotZ)
        mimic_api = PhysxSchema.PhysxMimicJointAPI.Get(slave_prim, slave_axis_token)
        if not mimic_api:
            mimic_api = PhysxSchema.PhysxMimicJointAPI.Apply(
                slave_prim, slave_axis_token
            )

        # Link slave to master and set mimic properties
        mimic_api.GetReferenceJointRel().ClearTargets(True)
        mimic_api.GetReferenceJointRel().AddTarget(master_prim.GetPath())
        mimic_api.GetReferenceJointAxisAttr().Set(master_axis_token)
        mimic_api.GetGearingAttr().Set(multiplier)
        mimic_api.GetOffsetAttr().Set(offset)

        print(
            f"[mimic] slave='{slave_name}'({slave_axis_token}) -> "
            f"master='{master_name}'({master_axis_token}) "
            f"gearing={multiplier:.4f} offset={offset:.4f}"
        )

    # 8. Save the stage to persist API removals and mimic additions
    stage.Save()


@clone
def spawn_from_mjcf(
    prim_path: str,
    cfg: MJCFFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    converter_cls: type[MjcfConverter] = MjcfConverter,
) -> Usd.Prim:
    """
    Converts an MJCF file to USD, applies necessary patches, and spawns it into the active stage.

    This function orchestrates the conversion of an MJCF robot model into a USD template.
    It applies several essential fixes to the generated USD: resolves duplicate articulation
    roots, enforces equality constraints via spherical joints, strips drive APIs from passive
    joints, configures PhysX mimic joints, and repairs structural bugs related to the
    'worldBody' prim and root joint offsets. Finally, it instances the corrected USD into
    the active simulation stage and applies convex decomposition for collision meshes.

    Args:
        prim_path (str): The target path in the current stage to spawn the robot.
        cfg (MJCFFileCfg): The configuration object containing the MJCF asset path and settings.
        translation (tuple[float, float, float] | None): Optional (x, y, z) global translation.
        orientation (tuple[float, float, float, float] | None): Optional (w, x, y, z) global quaternion.
        converter_cls (type[MjcfConverter]): The converter class used for MJCF-to-USD translation.

    Returns:
        Usd.Prim: The root prim of the newly spawned robot in the active stage.

    Raises:
        ValueError: If the converted USD stage contains an invalid root prim.
    """
    # 1. Extract metadata and preprocess the MJCF file
    mimic_info = get_mjcf_mimic_info(cfg.asset_path)
    constraints = get_mjcf_equality_constraints(cfg.asset_path)

    cfg.asset_path = preprocess_robot_menagerie_mjcf(cfg.asset_path)
    mjcf_loader = converter_cls(cfg)

    # 2. Open the converted USD template stage and validate the root
    stage = Usd.Stage.Open(mjcf_loader.usd_path)
    default_prim = stage.GetPrimAtPath("/").GetChildren()[0]
    if not default_prim or not default_prim.IsValid():
        raise ValueError(f"[spawn_from_mjcf] Invalid root prim at {prim_path}")

    stage.SetDefaultPrim(default_prim)

    # Clean up duplicate articulation roots generated by the converter
    _fix_duplicate_articulation_roots(default_prim)

    # 3. Apply MJCF equality constraints as USD Spherical Joints
    xf_cache = UsdGeom.XformCache()
    for constraint_name, constraint_data in constraints.items():
        body1_name = constraint_data["body1"]
        body2_name = constraint_data["body2"]

        print(
            f"[spawn_from_mjcf] Creating joint constraint at: {body1_name} -> {body2_name}"
        )
        from_prim = find_prim(stage, body1_name)
        to_prim = find_prim(stage, body2_name)

        joint_path = f"{default_prim.GetPath().pathString}/joints/{constraint_name}"
        joint = UsdPhysics.SphericalJoint.Define(stage, joint_path)

        joint.CreateBody0Rel().SetTargets([Sdf.Path(from_prim.GetPath().pathString)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(to_prim.GetPath().pathString)])

        # Calculate local anchors based on global poses
        to_pose = xf_cache.GetLocalToWorldTransform(to_prim)
        from_pose = xf_cache.GetLocalToWorldTransform(from_prim)

        anchor_pos_world = Gf.Quatd(from_pose.ExtractRotationQuat()).Transform(
            Gf.Vec3d(*constraint_data["anchor"])
        ) + Gf.Vec3d(from_pose.ExtractTranslation())

        anchor_pose_rel0 = (
            Gf.Transform(anchor_pos_world).GetMatrix() * from_pose.GetInverse()
        )
        anchor_pose_rel0 = anchor_pose_rel0.RemoveScaleShear()

        anchor_pose_rel1 = (
            Gf.Transform(anchor_pos_world).GetMatrix() * to_pose.GetInverse()
        )
        anchor_pose_rel1 = anchor_pose_rel1.RemoveScaleShear()

        # Extract positions and rotations for the joint local attributes
        pos0 = Gf.Vec3f(anchor_pose_rel0.ExtractTranslation())
        rot0 = Gf.Quatf(anchor_pose_rel0.ExtractRotationQuat())
        pos1 = Gf.Vec3f(anchor_pose_rel1.ExtractTranslation())
        rot1 = Gf.Quatf(anchor_pose_rel1.ExtractRotationQuat())

        joint.CreateLocalPos0Attr().Set(pos0)
        joint.CreateLocalRot0Attr().Set(rot0)
        joint.CreateLocalPos1Attr().Set(pos1)
        joint.CreateLocalRot1Attr().Set(rot1)
        joint.GetExcludeFromArticulationAttr().Set(True)

    # 4. Remove Drive APIs from passive/mimic joints
    articulated_joints = get_mjcf_articulated_joints(cfg.asset_path)

    for p in stage.Traverse():
        if "RevoluteJoint" in p.GetTypeName():
            if p.GetName() not in articulated_joints:
                p.RemoveAPI(UsdPhysics.DriveAPI, "X")
                p.RemoveAPI(UsdPhysics.DriveAPI, "Y")
                p.RemoveAPI(UsdPhysics.DriveAPI, "Z")

    # 5. Apply PhysX mimic joints setup
    _apply_physx_mimic_joints(default_prim, mimic_info)

    # 6. Patch for Isaac Lab bug involving the anomalous 'worldBody' prim
    world_body_prim = default_prim.GetChild("worldBody")
    if world_body_prim and world_body_prim.IsValid():
        print(
            f"[spawn_from_mjcf] Removing 'worldBody' prim at {world_body_prim.GetPath()}"
        )
        world_body_prim.SetActive(False)
        world_body_prim.SetHidden(True)

    # 7. FIX: Repair the buggy 'rootJoint_*' joint offset mappings
    root_joint_prim = None
    joints_scope = stage.GetPrimAtPath(default_prim.GetPath().AppendChild("joints"))

    if joints_scope and joints_scope.IsValid():
        for child in joints_scope.GetChildren():
            if UsdPhysics.Joint(child) and child.GetName().startswith("rootJoint_"):
                root_joint_prim = child
                break

    if root_joint_prim and root_joint_prim.IsValid():
        defected_joint_name = root_joint_prim.GetName()
        joint = UsdPhysics.Joint(root_joint_prim)

        if joint:
            xfc = UsdGeom.XformCache()
            b0_targets = joint.GetBody0Rel().GetTargets()
            b1_targets = joint.GetBody1Rel().GetTargets()

            if b1_targets:
                b1_prim = stage.GetPrimAtPath(b1_targets[0])
                if b1_prim and b1_prim.IsValid():
                    # Calculate absolute transformations to align local joint frames
                    t1_matrix = Gf.Matrix4d(xfc.GetLocalToWorldTransform(b1_prim))

                    pos1 = joint.GetLocalPos1Attr().Get() or Gf.Vec3f(0.0, 0.0, 0.0)
                    rot1 = joint.GetLocalRot1Attr().Get() or Gf.Quatf(
                        1.0, 0.0, 0.0, 0.0
                    )

                    tf1 = Gf.Transform()
                    q1i = rot1.GetImaginary()
                    q1d = Gf.Quatd(
                        float(rot1.GetReal()),
                        float(q1i[0]),
                        float(q1i[1]),
                        float(q1i[2]),
                    )

                    tf1.SetRotation(Gf.Rotation(q1d))
                    tf1.SetTranslation(Gf.Vec3d(pos1[0], pos1[1], pos1[2]))
                    x1_matrix = tf1.GetMatrix().RemoveScaleShear()

                    a_matrix = (t1_matrix * x1_matrix).RemoveScaleShear()

                    if b0_targets:
                        b0_prim = stage.GetPrimAtPath(b0_targets[0])
                        if b0_prim and b0_prim.IsValid():
                            t0_matrix = Gf.Matrix4d(
                                xfc.GetLocalToWorldTransform(b0_prim)
                            )
                        else:
                            t0_matrix = Gf.Matrix4d(1.0)

                        x0_new = t0_matrix.GetInverse() * a_matrix
                        p0 = Gf.Vec3f(x0_new.ExtractTranslation())
                        q0d = x0_new.ExtractRotation().GetQuat()
                    else:
                        p0 = Gf.Vec3f(a_matrix.ExtractTranslation())
                        q0d = a_matrix.ExtractRotation().GetQuat()

                    qi = q0d.GetImaginary()
                    q0 = Gf.Quatf(
                        float(q0d.GetReal()), float(qi[0]), float(qi[1]), float(qi[2])
                    )

                    joint.CreateLocalPos0Attr().Set(p0)
                    joint.CreateLocalRot0Attr().Set(q0)
                    print(
                        f"[spawn_from_mjcf] Repaired joint locals at {defected_joint_name}"
                    )
                else:
                    print(
                        f"[spawn_from_mjcf] body1 target invalid for {defected_joint_name}"
                    )
            else:
                print(
                    f"[spawn_from_mjcf] No body1 on {defected_joint_name}; cannot repair"
                )
        else:
            print(
                f"[spawn_from_mjcf] Prim at {defected_joint_name} is not a UsdPhysics.Joint"
            )

    # 8. Save the template USD stage before spawning
    stage.Save()

    # 9. Spawn the live clone in Isaac Lab using the saved USD template
    prim = _spawn_from_usd_file(
        prim_path, mjcf_loader.usd_path, cfg, translation, orientation
    )

    # 10. Post-spawn cleanup and physics material assignments
    _explode_instancing(prim)
    _repair_instanced_material_bindings(prim)

    set_convex_decomposition_for_all_collision_meshes(
        stage=prim.GetStage(),
        root_prim=prim,
        max_hulls=16,
        hull_vertex_limit=64,
    )

    return prim
