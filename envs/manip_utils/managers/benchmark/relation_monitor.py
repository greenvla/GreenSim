# relation_monitor.py
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional, Set

from .benchmark_analyzer import BenchmarkAnalyzer
from .annotation_manager import AnnotationManager

import torch
import math


@dataclass
class RelationInstanceMeta:
    relation_key: str  # e.g. "in_hand"
    relation_type: str  # distance/spawn/...
    pattern_filled: str  # "pick green_apple_0 by left_thumb_proximal_base"


@dataclass
class CompiledRelationSpec:
    relation_key: str
    relation_type: str
    pattern: str
    token_a: str | None
    token_b: str | None
    names_a: List[str]
    names_b: List[str]
    threshold: float | None
    rid_start: int
    rid_end: int
    comparison: str = "less"
    duration_sec: float | None = None
    latch: bool = False
    source_relation: str | None = None
    source_rid_start: int | None = None
    source_rid_end: int | None = None
    ang_threshold: float | None = None

    local_up_axis: Tuple[float, float, float] | None = None
    world_up_axis: Tuple[float, float, float] | None = None
    upright_cos_threshold: float | None = None
    reference_mode: str | None = None


class RelationMonitor:
    """
    Relation monitor:
      - reads task.json,
      - on update() computes the current bool values of all relation instances,
      - returns only the events that changed.
    """

    TOKEN_RE = re.compile(r"{([^}]+)}")
    INDEXED_NAME_RE = re.compile(r"^(.*)_(\d+)$")

    DEFAULT_STATIC_LIN_VEL_EPS: float = 0.02  # m/s
    DEFAULT_STATIC_ANG_VEL_EPS: float = 0.05  # rad/s (~2.9°/s)
    ROBOT_ROOT_NAME = "robot_root"

    DEFAULT_UPRIGHT_LOCAL_UP_AXIS: Tuple[float, float, float] = (
        0.0,
        0.0,
        1.0,
    )
    DEFAULT_UPRIGHT_WORLD_UP_AXIS: Tuple[float, float, float] = (
        0.0,
        0.0,
        1.0,
    )

    def __init__(
        self,
        env,
        task_json: Union[str, Path, Dict[str, Any]],
        every_n_steps: int = 1,
        device: str | torch.device | None = None,
    ):
        self.env = env
        self.scene = env.scene
        self.device = torch.device(device if device is not None else env.device)
        self.num_envs = int(env.num_envs)
        self.every_n_steps = max(1, int(every_n_steps))

        self.annotation_managers: list[AnnotationManager] = [
            AnnotationManager(
                use_sim_time=bool(getattr(self.env, "enable_sim_time", False)),
            )
            for _ in range(self.num_envs)
        ]

        self.task = self._load_task(task_json)
        self.object_groups = self._compile_object_groups(self.task["objects"])

        self.compiled_specs: List[CompiledRelationSpec] = []
        self.relation_meta: List[RelationInstanceMeta] = []
        self._event_queue: List[Dict[str, Any]] = []

        self._compile_relations(self.task["relations"])

        # ---- debug: print the names of the objects being monitored ----
        print("\n[RelationMonitor] Monitored object names:")
        printed = set()
        for spec in self.compiled_specs:
            for n in spec.names_a + spec.names_b:
                if n not in printed:
                    printed.add(n)
                    print(" ", n)
        print("[RelationMonitor] total:", len(printed), "\n")

        self.num_relation_instances = len(self.relation_meta)

        self._duration_started_at = torch.full(
            (self.num_envs, self.num_relation_instances),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )

        self._duration_satisfied = torch.zeros(
            (self.num_envs, self.num_relation_instances),
            dtype=torch.bool,
            device=self.device,
        )
        self.prev_values = torch.zeros(
            (self.num_envs, self.num_relation_instances),
            dtype=torch.bool,
            device=self.device,
        )
        self._distance_values = torch.full(
            (self.num_envs, self.num_relation_instances),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )

        self._upright_reference_local_up_axis = torch.zeros(
            (
                self.num_envs,
                self.num_relation_instances,
                3,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        self._upright_reference_valid = torch.zeros(
            (
                self.num_envs,
                self.num_relation_instances,
            ),
            dtype=torch.bool,
            device=self.device,
        )

        self._last_sim_time = 0.0
        self._episode_start_time = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._robot_body_names = set(self.scene["robot"].body_names)
        self.analyzers = [
            BenchmarkAnalyzer(
                self.task,
                robot_body_names=self._robot_body_names,
            )
            for _ in range(self.num_envs)
        ]
        self._success_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._manual_success_latched = False

        self._relation_to_specs: Dict[str, CompiledRelationSpec] = {}
        self._relation_to_rids: Dict[str, List[int]] = {}
        self._build_relation_indexes()

    # -------------------- public API --------------------

    def success_mask(self) -> torch.Tensor:
        return self._success_mask.clone()

    @torch.no_grad()
    def update(
        self,
        sim_step: int,
        sim_time: float,
        manual_reset: bool = False,
        success_episode: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Returns the list of events only for relation instances that changed.
        """
        if (sim_step % self.every_n_steps) != 0:
            return []

        self._last_sim_time = float(sim_time)
        self._last_sim_step = int(sim_step)
        curr = self._compute_all_relations()  # [E, R], bool

        # print("\n[DEBUG] spawn relations at init:")
        # dummy_active_cache = {}
        # for spec in self.compiled_specs:
        #     if spec.relation_type == "spawn":
        #         m = self._get_active_mask(spec.names_a, dummy_active_cache)
        #         print(f"  rel_key={spec.relation_key}, names={spec.names_a}")
        #         print("   active mask:", m)  # [E, Na]

        changed = curr ^ self.prev_values
        changed_idx = torch.nonzero(changed, as_tuple=False)  # [K,2]

        self.prev_values[changed] = curr[changed]

        events: List[Dict[str, Any]] = []

        for event in self._event_queue:
            event["sim_step"] = int(sim_step)
            event["sim_time"] = float(sim_time)
            events.append(event)

        self._event_queue.clear()

        if changed_idx.numel() > 0:
            changed_idx_cpu = changed_idx.cpu()
            curr_cpu = curr.cpu()

            for k in range(changed_idx_cpu.shape[0]):
                env_id = int(changed_idx_cpu[k, 0].item())
                rid = int(changed_idx_cpu[k, 1].item())
                value = int(curr_cpu[env_id, rid].item())
                meta = self.relation_meta[rid]

                events.append(
                    {
                        "sim_step": int(sim_step),
                        "sim_time": float(sim_time),
                        "env_id": env_id,
                        "rid": rid,
                        "relation": meta.relation_key,
                        "type": meta.relation_type,
                        "pattern": meta.pattern_filled,
                        "value": value,
                    }
                )

        # Sending events to BenchmarkAnalyzer (batched per env) ----
        batched_by_env: Dict[int, Dict[str, int]] = {}
        for e in events:
            env_id = int(e["env_id"])
            # Skip already-finalized episodes
            if getattr(self.analyzers[env_id], "_finalized", False):
                continue
            batched_by_env.setdefault(env_id, {})
            batched_by_env[env_id][e["pattern"]] = int(e["value"])

        # Update analyzers and check success ----
        for env_id, payload_dict in batched_by_env.items():
            # Send the batch of events for this step
            self.analyzers[env_id].update(json.dumps(payload_dict), float(sim_time))
            # Check success
            if self.analyzers[env_id].is_success_now():
                events.append(
                    {
                        "sim_step": int(sim_step),
                        "sim_time": float(sim_time),
                        "env_id": env_id,
                        "relation": "episode_success",
                        "type": "success",
                        "pattern": "episode_success",
                        "value": 1,
                    }
                )
                # Mark as finalized so we don't send further events
                # self.analyzers[env_id]._finalized = True
        if success_episode:
            self._manual_success_latched = True

        if manual_reset:
            for env_id in range(self.num_envs):
                if self._manual_success_latched or success_episode:
                    events.append(
                        {
                            "sim_step": int(sim_step),
                            "sim_time": float(sim_time),
                            "env_id": env_id,
                            "relation": "episode_success",
                            "type": "success",
                            "pattern": "episode_success",
                            "value": 1,
                        }
                    )
                else:
                    events.append(
                        {
                            "sim_step": int(sim_step),
                            "sim_time": float(sim_time),
                            "env_id": env_id,
                            "relation": "episode_aborted",
                            "type": "aborted",
                            "pattern": "episode_aborted",
                            "value": 1,
                        }
                    )

        return events

    def reset(self, env_ids=None, sim_time: float | None = None):
        if env_ids is None:
            self.prev_values.zero_()
            self._distance_values.fill_(float("nan"))
            self._duration_started_at.fill_(float("nan"))
            self._duration_satisfied.zero_()
            self._upright_reference_valid.zero_()
            self._success_mask[:] = False
            self._manual_success_latched = False
            for i in range(self.num_envs):
                self.analyzers[i] = BenchmarkAnalyzer(self.task, self._robot_body_names)
            if sim_time is not None:
                self._episode_start_time[:] = float(sim_time)
            for annotation_manager in self.annotation_managers:
                annotation_manager.reset()
        else:
            self.prev_values[env_ids] = False
            self._distance_values[env_ids] = float("nan")
            self._duration_started_at[env_ids] = float("nan")
            self._duration_satisfied[env_ids] = False
            self._upright_reference_valid[env_ids] = False
            self._success_mask[env_ids] = False
            self._manual_success_latched = False
            for i in env_ids.tolist():
                self.analyzers[i] = BenchmarkAnalyzer(self.task, self._robot_body_names)
            if sim_time is not None:
                self._episode_start_time[env_ids] = float(sim_time)
            for env_id in env_ids.tolist():
                self.annotation_managers[int(env_id)].reset()

    @torch.no_grad()
    def capture_upright_references(self, env_ids=None) -> None:
        """
        Remembers the semantic local up-axis upright relation from the current pose.

        Call after SceneSpawner.reset_objects(), when the scene data already
        contains the new root_quat_w. The current pose must be the reference one:
        the object visually stands correctly.
        """
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs,
                dtype=torch.long,
                device=self.device,
            )
        else:
            env_ids = env_ids.to(
                device=self.device,
                dtype=torch.long,
            )

        quat_cache: Dict[Tuple[str, ...], torch.Tensor] = {}

        for spec in self.compiled_specs:
            if spec.relation_type != "upright":
                continue

            if spec.reference_mode != "spawn":
                continue

            quat_a = self._get_orientations(
                spec.names_a,
                quat_cache,
            )  # [E, Na, 4]

            world_up_axis = (
                torch.tensor(
                    spec.world_up_axis,
                    dtype=quat_a.dtype,
                    device=quat_a.device,
                )
                .view(1, 1, 3)
                .expand(
                    self.num_envs,
                    len(spec.names_a),
                    3,
                )
            )

            local_up_axis = self._quat_apply_inverse_wxyz(
                quat_a,
                world_up_axis,
            )  # [E, Na, 3]

            self._upright_reference_local_up_axis[
                env_ids,
                spec.rid_start : spec.rid_end,
                :,
            ] = local_up_axis[env_ids]

            self._upright_reference_valid[
                env_ids,
                spec.rid_start : spec.rid_end,
            ] = True

    def resolve_relation_rids(
        self,
        relation_key: str,
        object_a: str | None = None,
        object_b: str | None = None,
    ) -> List[int]:
        """Return the rid of a specific relation instance without computing it."""
        if relation_key not in self._relation_to_specs:
            raise KeyError(f"Unknown relation_key: {relation_key}")

        spec = self._relation_to_specs[relation_key]

        if object_a is None and object_b is None:
            return list(range(spec.rid_start, spec.rid_end))

        if spec.relation_type == "spawn":
            if object_a is None or object_b is not None:
                raise ValueError(f"Relation {relation_key!r} requires object_a only")
            return [spec.rid_start + spec.names_a.index(object_a)]

        if spec.relation_type in {"static", "upright"}:
            if object_a is None or object_b is not None:
                raise ValueError(f"Relation {relation_key!r} requires object_a only")
            return [spec.rid_start + spec.names_a.index(object_a)]

        if spec.relation_type in {"distance", "inside", "not_inside", "duration"}:
            if object_a is None or object_b is None:
                raise ValueError(
                    f"Relation {relation_key!r} requires object_a and object_b"
                )
            idx_a = spec.names_a.index(object_a)
            idx_b = spec.names_b.index(object_b)
            return [spec.rid_start + idx_a * len(spec.names_b) + idx_b]

        if spec.relation_type == "timeout":
            if object_a is not None or object_b is not None:
                raise ValueError(f"Relation {relation_key!r} does not accept objects")
            return [spec.rid_start]

        raise NotImplementedError(
            f"Relation type {spec.relation_type!r} is unsupported"
        )

    def get_relation_spec(self, relation_key: str) -> CompiledRelationSpec:
        """Return the compiled spec for a benchmark relation."""
        try:
            return self._relation_to_specs[relation_key]
        except KeyError as exc:
            known = ", ".join(sorted(self._relation_to_specs))
            raise KeyError(
                f"Unknown benchmark relation {relation_key!r}. "
                f"Known relations: [{known}]"
            ) from exc

    def _get_guide_pose_resolver(self):
        resolver = getattr(self, "_guide_pose_resolver", None)
        if resolver is None:
            resolver = self.env.scene_spawner.build_object_pose_resolver(self.scene)
            self._guide_pose_resolver = resolver
        return resolver

    def _resolve_guide_world_pose(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        World pose of a guide: pos [E, 3], quat [E, 4] (w, x, y, z).

        - guide bound to a spawn group with anchor_prim: pose is read live from
          the stage via the spawner's pose resolver (the guide moves together
          with its anchor);
        - static guide: metadata position/rotation + env_origins
        """
        spawner = self.env.scene_spawner
        meta = spawner.object_metadata[name]

        anchor_prim = None
        if hasattr(spawner, "_get_group_anchor_prim"):
            anchor_prim = spawner._get_group_anchor_prim(meta)

        if anchor_prim is not None:
            try:
                pose_resolver = self._get_guide_pose_resolver()
                poses = [
                    pose_resolver(name, env_id, self.device)
                    for env_id in range(self.num_envs)
                ]
                pos = torch.stack([p[0] for p in poses], dim=0).to(torch.float32)
                quat = torch.stack([p[1] for p in poses], dim=0).to(torch.float32)
                return pos, quat
            except Exception as e:
                warnings.warn(
                    f"Failed to resolve live pose for anchored guide "
                    f"'{name}': {e}. Falling back to metadata pose."
                )

        guide_position = (
            meta["position"] if meta["position"] is not None else [0.0, 0.0, 0.0]
        )
        guide_rotation = (
            meta["rotation"] if meta["rotation"] is not None else [1.0, 0.0, 0.0, 0.0]
        )

        pos = (
            torch.tensor(guide_position, dtype=torch.float32, device=self.device)
            .view(1, 3)
            .repeat(self.num_envs, 1)
            + self.scene.env_origins
        )
        quat = (
            torch.tensor(guide_rotation, dtype=torch.float32, device=self.device)
            .view(1, 4)
            .repeat(self.num_envs, 1)
        )
        return pos, quat

    def get_relation(
        self,
        relation_key: str,
        object_a: str | None = None,
        object_b: str | None = None,
        env_id: int = 0,
        report_query: bool = True,
    ) -> bool:
        """
        Universal relation query.

        Behavior:
        - if object_a/object_b are not given -> any across all relation instances
        - if they are given -> exact lookup of a specific relation instance

        Examples:
            get_relation("in_hand")
            get_relation("in_hand", object_a="hshoulders", object_b="left_thumb_proximal_base")
            get_relation("exists_fruits", object_a="hshoulders")
            get_relation("episode_timeout")
        """
        if relation_key not in self._relation_to_specs:
            raise KeyError(f"Unknown relation_key: {relation_key}")

        spec = self._relation_to_specs[relation_key]

        if not (0 <= int(env_id) < self.num_envs):
            raise IndexError(f"env_id out of range: {env_id}")

        # 1) aggregate any
        if object_a is None and object_b is None:
            rids = self._relation_to_rids.get(relation_key, [])
            if not rids:
                return False
            result = bool(self.prev_values[env_id, rids].any().item())
            if report_query:
                self._report_relation_request(
                    relation_key=relation_key,
                    object_a=object_a,
                    object_b=object_b,
                    env_id=env_id,
                    result=result,
                )
            return result

        # 2) typed exact lookup
        if spec.relation_type == "timeout":
            if object_a is not None or object_b is not None:
                raise ValueError(
                    f"Relation '{relation_key}' of type 'timeout' does not accept object arguments"
                )
            rid = spec.rid_start
            result = bool(self.prev_values[env_id, rid].item())
            if report_query:
                self._report_relation_request(
                    relation_key=relation_key,
                    object_a=object_a,
                    object_b=object_b,
                    env_id=env_id,
                    result=result,
                )
            return result

        if spec.relation_type == "spawn":
            if object_a is None:
                raise ValueError(
                    f"Relation '{relation_key}' of type 'spawn' requires object_a"
                )
            if object_b is not None:
                raise ValueError(
                    f"Relation '{relation_key}' of type 'spawn' does not accept object_b"
                )

            try:
                idx_a = spec.names_a.index(object_a)
            except ValueError as e:
                raise KeyError(
                    f"Object '{object_a}' is not part of relation '{relation_key}'"
                ) from e

            rid = spec.rid_start + idx_a
            result = bool(self.prev_values[env_id, rid].item())
            if report_query:
                self._report_relation_request(
                    relation_key=relation_key,
                    object_a=object_a,
                    object_b=object_b,
                    env_id=env_id,
                    result=result,
                )
            return result

        if spec.relation_type in {
            "distance",
            "inside",
            "not_inside",
            "duration",
        }:
            if object_a is None or object_b is None:
                raise ValueError(
                    f"Relation '{relation_key}' of type '{spec.relation_type}' "
                    "requires object_a and object_b"
                )

            try:
                idx_a = spec.names_a.index(object_a)
            except ValueError as e:
                raise KeyError(
                    f"Object '{object_a}' is not part of relation '{relation_key}'"
                ) from e

            try:
                idx_b = spec.names_b.index(object_b)
            except ValueError as e:
                raise KeyError(
                    f"Object '{object_b}' is not part of relation '{relation_key}'"
                ) from e

            rid = spec.rid_start + idx_a * len(spec.names_b) + idx_b
            result = bool(self.prev_values[env_id, rid].item())

            distance = None
            threshold = None
            details = None

            if spec.relation_type == "distance":
                cached_distance = float(self._distance_values[env_id, rid].item())
                if math.isfinite(cached_distance):
                    distance = cached_distance
                threshold = float(spec.threshold)
            elif spec.relation_type == "duration":
                details = self._describe_duration_state(
                    spec=spec,
                    rid=rid,
                    idx_a=idx_a,
                    idx_b=idx_b,
                    env_id=env_id,
                )
            elif spec.relation_type in {"inside", "not_inside"}:
                details = self._describe_inside_state(object_a, object_b, env_id)
            if report_query:
                self._report_relation_request(
                    relation_key=relation_key,
                    object_a=object_a,
                    object_b=object_b,
                    env_id=env_id,
                    result=result,
                    distance=distance,
                    threshold=threshold,
                    details=details,
                )

            return result

        if spec.relation_type in {"static", "upright"}:
            if object_a is None:
                raise ValueError(
                    f"Relation '{relation_key}' of type "
                    f"'{spec.relation_type}' requires object_a"
                )

            if object_b is not None:
                raise ValueError(
                    f"Relation '{relation_key}' of type "
                    f"'{spec.relation_type}' does not accept object_b"
                )

            try:
                idx_a = spec.names_a.index(object_a)
            except ValueError as e:
                raise KeyError(
                    f"Object '{object_a}' is not part of " f"relation '{relation_key}'"
                ) from e

            rid = spec.rid_start + idx_a

            result = bool(self.prev_values[env_id, rid].item())
            if report_query:
                self._report_relation_request(
                    relation_key=relation_key,
                    object_a=object_a,
                    object_b=object_b,
                    env_id=env_id,
                    result=result,
                )
            return result

        raise NotImplementedError(
            f"Relation type '{spec.relation_type}' is not supported in get_relation()"
        )

    def _report_relation_request(
        self,
        relation_key: str,
        object_a: str | None,
        object_b: str | None,
        env_id: int,
        result: bool,
        distance: float | None = None,
        threshold: float | None = None,
        details: str | None = None,
    ) -> None:
        pattern = f"query relation {relation_key}"

        if object_a is not None:
            pattern += f" {object_a}"

        if object_b is not None:
            pattern += f" {object_b}"

        if distance is not None:
            pattern += f" distance={distance:.6f}m"

        if threshold is not None:
            pattern += f" threshold={threshold:.6f}m"

        if details is not None:
            pattern += f" [{details}]"

        self._event_queue.append(
            {
                "env_id": int(env_id),
                "relation": relation_key,
                "type": "query",
                "pattern": pattern,
                "value": int(result),
            }
        )

    def drain_event_queue(
        self, sim_step: int | None = None, sim_time: float | None = None
    ) -> List[Dict[str, Any]]:
        step = (
            int(sim_step)
            if sim_step is not None
            else int(getattr(self, "_last_sim_step", 0))
        )
        t = float(sim_time) if sim_time is not None else float(self._last_sim_time)
        events = []
        for event in self._event_queue:
            event["sim_step"] = step
            event["sim_time"] = t
            events.append(event)
        self._event_queue.clear()
        return events

    # -------------------- compile --------------------

    def _load_task(self, task_json: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(task_json, dict):
            return task_json
        p = Path(task_json)
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _build_relation_indexes(self) -> None:
        self._relation_to_specs = {}
        self._relation_to_rids = {}

        for spec in self.compiled_specs:
            self._relation_to_specs[spec.relation_key] = spec

        for rid, meta in enumerate(self.relation_meta):
            self._relation_to_rids.setdefault(meta.relation_key, []).append(rid)

    def _compile_object_groups(
        self, objects_cfg: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        From:
          "fruits": {"green_apple":[0,1], "orange":[0], "potato":[]}
        to:
          "fruits": ["green_apple_0","green_apple_1","orange_0", "potato"]
        """
        groups: Dict[str, List[str]] = {}
        for group_name, group_desc in objects_cfg.items():
            names: List[str] = []
            if not isinstance(group_desc, dict):
                raise ValueError(f"objects.{group_name} must be dict")
            for base_name, ids in group_desc.items():
                if ids is None or len(ids) == 0:
                    names.append(base_name)
                else:
                    for i in ids:
                        names.append(f"{base_name}_{i}")
            groups[group_name] = names
        return groups

    def _compile_relations(self, relations_cfg: Dict[str, Any]):
        rid = 0

        for rel_key, rel_cfg in relations_cfg.items():
            rel_type = rel_cfg["type"]
            pattern = rel_cfg["pattern"]
            tokens = self.TOKEN_RE.findall(pattern)

            if rel_type == "distance":
                comparison = str(rel_cfg.get("comparison", "less")).lower()
                if comparison not in {"less", "greater"}:
                    raise ValueError(
                        f"{rel_key}: distance comparison must be 'less' or 'greater', "
                        f"got {comparison!r}"
                    )
                if len(tokens) != 2:
                    raise ValueError(
                        f"{rel_key}: distance pattern must contain exactly 2 tokens"
                    )

                token_a, token_b = tokens[0], tokens[1]
                names_a = self.object_groups[token_a]
                names_b = self.object_groups[token_b]
                threshold = float(rel_cfg["distance"])

                rid_start = rid
                for a in names_a:
                    for b in names_b:
                        pattern_filled = pattern.replace(f"{{{token_a}}}", a).replace(
                            f"{{{token_b}}}", b
                        )
                        self.relation_meta.append(
                            RelationInstanceMeta(
                                relation_key=rel_key,
                                relation_type=rel_type,
                                pattern_filled=pattern_filled,
                            )
                        )
                        rid += 1
                rid_end = rid

                self.compiled_specs.append(
                    CompiledRelationSpec(
                        relation_key=rel_key,
                        relation_type=rel_type,
                        pattern=pattern,
                        token_a=token_a,
                        token_b=token_b,
                        names_a=names_a,
                        names_b=names_b,
                        threshold=threshold,
                        rid_start=rid_start,
                        rid_end=rid_end,
                        comparison=comparison,
                    )
                )

            elif rel_type == "spawn":
                if len(tokens) != 1:
                    raise ValueError(
                        f"{rel_key}: spawn pattern must contain exactly 1 token"
                    )

                token_a = tokens[0]
                names_a = self.object_groups[token_a]

                rid_start = rid
                for a in names_a:
                    pattern_filled = pattern.replace(f"{{{token_a}}}", a)
                    self.relation_meta.append(
                        RelationInstanceMeta(
                            relation_key=rel_key,
                            relation_type=rel_type,
                            pattern_filled=pattern_filled,
                        )
                    )
                    rid += 1
                rid_end = rid

                self.compiled_specs.append(
                    CompiledRelationSpec(
                        relation_key=rel_key,
                        relation_type=rel_type,
                        pattern=pattern,
                        token_a=token_a,
                        token_b=None,
                        names_a=names_a,
                        names_b=[],
                        threshold=None,
                        rid_start=rid_start,
                        rid_end=rid_end,
                    )
                )
            elif rel_type == "duration":
                source_relation = str(rel_cfg.get("relation", ""))
                if not source_relation:
                    raise ValueError(
                        f"{rel_key}: duration relation requires field 'relation'"
                    )
                duration_sec = float(rel_cfg.get("duration_sec", 0.0))
                if duration_sec <= 0.0:
                    raise ValueError(
                        f"{rel_key}: duration_sec must be > 0, got {duration_sec}"
                    )
                latch = bool(rel_cfg.get("latch", False))
                source_spec = next(
                    (
                        spec
                        for spec in self.compiled_specs
                        if spec.relation_key == source_relation
                    ),
                    None,
                )

                if source_spec is None:
                    raise ValueError(
                        f"{rel_key}: source relation {source_relation!r} was not found. "
                        "Declare it before this duration relation."
                    )
                if source_spec.token_a is None or source_spec.token_b is None:
                    raise ValueError(
                        f"{rel_key}: source relation {source_relation!r} must have "
                        "exactly two object placeholders"
                    )
                duration_tokens = self.TOKEN_RE.findall(pattern)
                expected_tokens = [source_spec.token_a, source_spec.token_b]

                if duration_tokens != expected_tokens:
                    raise ValueError(
                        f"{rel_key}: duration pattern placeholders must match source "
                        f"relation {source_relation!r} and preserve their order: "
                        f"{expected_tokens!r}; got {duration_tokens!r}"
                    )

                rid_start = rid
                for a in source_spec.names_a:
                    for b in source_spec.names_b:
                        pattern_filled = pattern.replace(
                            f"{{{source_spec.token_a}}}", a
                        ).replace(f"{{{source_spec.token_b}}}", b)

                        self.relation_meta.append(
                            RelationInstanceMeta(
                                relation_key=rel_key,
                                relation_type=rel_type,
                                pattern_filled=pattern_filled,
                            )
                        )
                        rid += 1
                rid_end = rid

                self.compiled_specs.append(
                    CompiledRelationSpec(
                        relation_key=rel_key,
                        relation_type=rel_type,
                        pattern=pattern,
                        token_a=source_spec.token_a,
                        token_b=source_spec.token_b,
                        names_a=source_spec.names_a,
                        names_b=source_spec.names_b,
                        threshold=None,
                        rid_start=rid_start,
                        rid_end=rid_end,
                        duration_sec=duration_sec,
                        latch=latch,
                        source_relation=source_relation,
                        source_rid_start=source_spec.rid_start,
                        source_rid_end=source_spec.rid_end,
                    )
                )
            elif rel_type == "timeout":
                duration = float(rel_cfg["duration_sec"])
                rid_start = rid
                self.relation_meta.append(
                    RelationInstanceMeta(
                        relation_key=rel_key,
                        relation_type=rel_type,
                        pattern_filled=rel_cfg.get("pattern", f"{rel_key} timeout"),
                    )
                )
                rid += 1
                rid_end = rid
                self.compiled_specs.append(
                    CompiledRelationSpec(
                        relation_key=rel_key,
                        relation_type=rel_type,
                        pattern=rel_cfg.get("pattern", f"{rel_key} timeout"),
                        token_a=None,
                        token_b=None,
                        names_a=[],
                        names_b=[],
                        threshold=None,
                        rid_start=rid_start,
                        rid_end=rid_end,
                        duration_sec=duration,
                    )
                )
            elif rel_type == "static":
                if len(tokens) != 1:
                    raise ValueError(
                        f"{rel_key}: static pattern must contain exactly 1 token"
                    )

                token_a = tokens[0]
                names_a = self.object_groups[token_a]

                # both epsilons are optional — if not given, we use physically
                # motivated defaults consistent with the PhysX sleep threshold
                lin_eps = float(
                    rel_cfg.get("lin_vel_eps", self.DEFAULT_STATIC_LIN_VEL_EPS)
                )

                # ang_vel_eps can be explicitly set to null in JSON -> disables
                # the angular velocity check; if the key is absent -> default
                if "ang_vel_eps" in rel_cfg:
                    ang_eps_raw = rel_cfg["ang_vel_eps"]
                    ang_eps = float(ang_eps_raw) if ang_eps_raw is not None else None
                else:
                    ang_eps = self.DEFAULT_STATIC_ANG_VEL_EPS

                rid_start = rid
                for a in names_a:
                    pattern_filled = pattern.replace(f"{{{token_a}}}", a)
                    self.relation_meta.append(
                        RelationInstanceMeta(
                            relation_key=rel_key,
                            relation_type=rel_type,
                            pattern_filled=pattern_filled,
                        )
                    )
                    rid += 1
                rid_end = rid

                self.compiled_specs.append(
                    CompiledRelationSpec(
                        relation_key=rel_key,
                        relation_type=rel_type,
                        pattern=pattern,
                        token_a=token_a,
                        token_b=None,
                        names_a=names_a,
                        names_b=[],
                        threshold=lin_eps,
                        rid_start=rid_start,
                        rid_end=rid_end,
                        ang_threshold=ang_eps,
                    )
                )
            elif rel_type == "upright":
                if len(tokens) != 1:
                    raise ValueError(
                        f"{rel_key}: upright pattern must contain exactly 1 token"
                    )

                token_a = tokens[0]
                names_a = self.object_groups[token_a]
                reference_mode = rel_cfg.get("reference_mode", None)

                if reference_mode not in {None, "spawn"}:
                    raise ValueError(
                        f"{rel_key}: reference_mode must be null or 'spawn', "
                        f"got {reference_mode!r}"
                    )

                if reference_mode == "spawn" and "local_up_axis" in rel_cfg:
                    raise ValueError(
                        f"{rel_key}: local_up_axis cannot be used together "
                        "with reference_mode='spawn'"
                    )

                if "max_tilt_deg" not in rel_cfg:
                    raise ValueError(
                        f"{rel_key}: upright relation requires 'max_tilt_deg'"
                    )

                max_tilt_deg = float(rel_cfg["max_tilt_deg"])

                if not (0.0 <= max_tilt_deg < 90.0):
                    raise ValueError(
                        f"{rel_key}: max_tilt_deg must be in [0, 90), "
                        f"got {max_tilt_deg}"
                    )

                local_up_axis_raw = rel_cfg.get(
                    "local_up_axis",
                    self.DEFAULT_UPRIGHT_LOCAL_UP_AXIS,
                )
                world_up_axis_raw = rel_cfg.get(
                    "world_up_axis",
                    self.DEFAULT_UPRIGHT_WORLD_UP_AXIS,
                )

                local_up_axis = torch.tensor(
                    local_up_axis_raw,
                    dtype=torch.float32,
                )
                world_up_axis = torch.tensor(
                    world_up_axis_raw,
                    dtype=torch.float32,
                )

                if tuple(local_up_axis.shape) != (3,):
                    raise ValueError(
                        f"{rel_key}: local_up_axis must be a 3-vector, "
                        f"got {local_up_axis_raw!r}"
                    )

                if tuple(world_up_axis.shape) != (3,):
                    raise ValueError(
                        f"{rel_key}: world_up_axis must be a 3-vector, "
                        f"got {world_up_axis_raw!r}"
                    )

                local_up_axis_norm = torch.linalg.norm(local_up_axis)
                world_up_axis_norm = torch.linalg.norm(world_up_axis)

                if float(local_up_axis_norm.item()) < 1e-8:
                    raise ValueError(f"{rel_key}: local_up_axis must be non-zero")

                if float(world_up_axis_norm.item()) < 1e-8:
                    raise ValueError(f"{rel_key}: world_up_axis must be non-zero")

                local_up_axis = local_up_axis / local_up_axis_norm
                world_up_axis = world_up_axis / world_up_axis_norm

                rid_start = rid

                for a in names_a:
                    pattern_filled = pattern.replace(f"{{{token_a}}}", a)

                    self.relation_meta.append(
                        RelationInstanceMeta(
                            relation_key=rel_key,
                            relation_type=rel_type,
                            pattern_filled=pattern_filled,
                        )
                    )
                    rid += 1

                rid_end = rid

                self.compiled_specs.append(
                    CompiledRelationSpec(
                        relation_key=rel_key,
                        relation_type=rel_type,
                        pattern=pattern,
                        token_a=token_a,
                        token_b=None,
                        names_a=names_a,
                        names_b=[],
                        threshold=None,
                        rid_start=rid_start,
                        rid_end=rid_end,
                        local_up_axis=tuple(local_up_axis.tolist()),
                        world_up_axis=tuple(world_up_axis.tolist()),
                        upright_cos_threshold=math.cos(math.radians(max_tilt_deg)),
                        reference_mode=reference_mode,
                    )
                )
            elif rel_type in {"inside", "not_inside"}:
                if len(tokens) != 2:
                    raise ValueError(
                        f"{rel_key}: {rel_type} pattern must contain exactly 2 tokens"
                    )

                token_a, token_b = tokens[0], tokens[1]
                names_a = self.object_groups[token_a]
                names_b = self.object_groups[token_b]

                rid_start = rid
                for a in names_a:
                    for b in names_b:
                        pattern_filled = pattern.replace(f"{{{token_a}}}", a).replace(
                            f"{{{token_b}}}", b
                        )

                        self.relation_meta.append(
                            RelationInstanceMeta(
                                relation_key=rel_key,
                                relation_type=rel_type,
                                pattern_filled=pattern_filled,
                            )
                        )
                        rid += 1

                rid_end = rid

                self.compiled_specs.append(
                    CompiledRelationSpec(
                        relation_key=rel_key,
                        relation_type=rel_type,
                        pattern=pattern,
                        token_a=token_a,
                        token_b=token_b,
                        names_a=names_a,
                        names_b=names_b,
                        threshold=None,
                        rid_start=rid_start,
                        rid_end=rid_end,
                    )
                )
            else:
                # Contact can be added later (the format is already accounted for)
                warnings.warn(
                    f"Relation '{rel_key}' type '{rel_type}' is not implemented yet. Skipped."
                )

    def build_report(self, env_id: int, episode_id: str) -> Dict[str, Any]:
        analyzer = self.analyzers[env_id]

        # if already finalized — return the last report, don't crash
        if getattr(analyzer, "_finalized", False):
            last = getattr(analyzer, "_final_report", None)
            if last is not None:
                return dict(last)
            # fallback: safely recreate an empty one
            return {
                "episode_id": episode_id,
                "success": False,
                "error": "analyzer already finalized and no cached report",
            }

        duration_seconds = max(
            0.0, float(self._last_sim_time - self._episode_start_time[env_id].item())
        )
        report = analyzer.finalize(
            episode_id=episode_id, duration_seconds=duration_seconds
        )

        # cache for repeated calls
        analyzer._final_report = dict(report)
        return report

    def finalize_reports_for_events(self, episode_events):
        reports = []
        for ev in episode_events:
            env_id = int(ev["env_id"])
            finished_ep = int(ev["episode_id"])
            ep_id = f"episode_{finished_ep:03d}"

            report = self.build_report(env_id=env_id, episode_id=ep_id)
            reports.append((env_id, report))
        return reports

    # -------------------- runtime compute --------------------
    @torch.no_grad()
    def _compute_all_relations(self) -> torch.Tensor:
        out = torch.zeros(
            (self.num_envs, self.num_relation_instances),
            dtype=torch.bool,
            device=self.device,
        )

        # cache valid for one update()
        pos_cache: Dict[Tuple[str, ...], torch.Tensor] = {}
        active_cache: Dict[Tuple[str, ...], torch.Tensor] = {}
        lin_vel_cache: Dict[Tuple[str, ...], torch.Tensor] = {}
        ang_vel_cache: Dict[Tuple[str, ...], torch.Tensor] = {}
        quat_cache: Dict[Tuple[str, ...], torch.Tensor] = {}

        for spec in self.compiled_specs:
            if spec.relation_type == "spawn":
                active_a = self._get_active_mask(spec.names_a, active_cache)
                out[:, spec.rid_start : spec.rid_end] = active_a

            elif spec.relation_type == "distance":
                pos_a = self._get_positions(spec.names_a, pos_cache)
                pos_b = self._get_positions(spec.names_b, pos_cache)
                active_a = self._get_active_mask(spec.names_a, active_cache)
                active_b = self._get_active_mask(spec.names_b, active_cache)

                d = torch.cdist(pos_a, pos_b)
                if spec.comparison == "less":
                    rel = d < float(spec.threshold)
                elif spec.comparison == "greater":
                    rel = d > float(spec.threshold)
                else:
                    raise RuntimeError(
                        f"Unsupported distance comparison {spec.comparison!r} "
                        f"for relation {spec.relation_key!r}"
                    )

                valid = active_a.unsqueeze(-1) & active_b.unsqueeze(1)
                rel = rel & valid

                flat = rel.reshape(self.num_envs, -1)
                out[:, spec.rid_start : spec.rid_end] = flat

                self._distance_values[:, spec.rid_start : spec.rid_end] = d.reshape(
                    self.num_envs, -1
                )

            elif spec.relation_type == "timeout":
                elapsed = self._last_sim_time - self._episode_start_time
                alive = ~self._success_mask
                rel = ((elapsed >= float(spec.duration_sec)) & alive).unsqueeze(1)
                out[:, spec.rid_start : spec.rid_end] = rel

            elif spec.relation_type in {"static"}:
                speed_a = self._get_linear_speed(spec.names_a, lin_vel_cache)  # [E, Na]
                active_a = self._get_active_mask(spec.names_a, active_cache)  # [E, Na]

                rel = speed_a < float(spec.threshold)

                if spec.ang_threshold is not None:
                    ang_speed_a = self._get_angular_speed(
                        spec.names_a, ang_vel_cache
                    )  # [E, Na]
                    rel = rel & (ang_speed_a < float(spec.ang_threshold))

                rel = rel & active_a
                out[:, spec.rid_start : spec.rid_end] = rel
            elif spec.relation_type == "upright":
                # [E, Na, 4], wxyz
                quat_a = self._get_orientations(
                    spec.names_a,
                    quat_cache,
                )
                # [E, Na]
                active_a = self._get_active_mask(
                    spec.names_a,
                    active_cache,
                )

                world_up_axis = torch.tensor(
                    spec.world_up_axis,
                    dtype=quat_a.dtype,
                    device=quat_a.device,
                ).view(1, 1, 3)

                if spec.reference_mode == "spawn":
                    local_up_axis = self._upright_reference_local_up_axis[
                        :,
                        spec.rid_start : spec.rid_end,
                        :,
                    ].to(
                        dtype=quat_a.dtype,
                        device=quat_a.device,
                    )

                    reference_valid = self._upright_reference_valid[
                        :,
                        spec.rid_start : spec.rid_end,
                    ]
                else:
                    local_up_axis = (
                        torch.tensor(
                            spec.local_up_axis,
                            dtype=quat_a.dtype,
                            device=quat_a.device,
                        )
                        .view(1, 1, 3)
                        .expand(
                            self.num_envs,
                            len(spec.names_a),
                            3,
                        )
                    )

                    reference_valid = torch.ones(
                        (
                            self.num_envs,
                            len(spec.names_a),
                        ),
                        dtype=torch.bool,
                        device=self.device,
                    )

                object_up_axis_w = self._quat_apply_wxyz(
                    quat_a,
                    local_up_axis,
                )

                alignment = torch.sum(
                    object_up_axis_w * world_up_axis,
                    dim=-1,
                )

                rel = (
                    (alignment >= float(spec.upright_cos_threshold))
                    & active_a
                    & reference_valid
                )

                out[:, spec.rid_start : spec.rid_end] = rel

            elif spec.relation_type in {"inside", "not_inside"}:
                spawner = getattr(self.env, "scene_spawner", None)
                if spawner is None:
                    raise RuntimeError("Relation 'inside' requires env.scene_spawner")

                active_a = self._get_active_mask(
                    spec.names_a,
                    active_cache,
                )  # [E, Na]

                relation_columns = []

                for object_name in spec.names_a:
                    for container_name in spec.names_b:
                        inside = spawner.is_object_center_inside(
                            scene=self.scene,
                            object_name=object_name,
                            container_name=container_name,
                        )  # [E], torch.BoolTensor

                        relation_columns.append(
                            inside.to(
                                device=self.device,
                                dtype=torch.bool,
                            )
                        )

                rel = torch.stack(
                    relation_columns,
                    dim=1,
                )  # [E, Na * Nb]

                # Order matches the compilation loop:
                # a0-b0, a0-b1, ..., a1-b0, a1-b1, ...
                valid = (
                    active_a.unsqueeze(-1)
                    .expand(
                        self.num_envs,
                        len(spec.names_a),
                        len(spec.names_b),
                    )
                    .reshape(self.num_envs, -1)
                )

                if spec.relation_type == "not_inside":
                    rel = ~rel
                out[:, spec.rid_start : spec.rid_end] = rel & valid

            elif spec.relation_type == "duration":
                source_start = int(spec.source_rid_start)
                source_end = int(spec.source_rid_end)

                source_value = out[:, source_start:source_end]

                duration_start = self._duration_started_at[
                    :, spec.rid_start : spec.rid_end
                ]

                started_now = source_value & torch.isnan(duration_start)

                duration_start = torch.where(
                    started_now,
                    torch.full_like(duration_start, self._last_sim_time),
                    duration_start,
                )

                elapsed = self._last_sim_time - duration_start

                completed_now = source_value & (elapsed >= float(spec.duration_sec))

                if spec.latch:
                    satisfied = self._duration_satisfied[
                        :, spec.rid_start : spec.rid_end
                    ]

                    satisfied = satisfied | completed_now

                    self._duration_satisfied[:, spec.rid_start : spec.rid_end] = (
                        satisfied
                    )

                    rel = satisfied
                else:
                    rel = completed_now

                duration_start = torch.where(
                    source_value,
                    duration_start,
                    torch.full_like(duration_start, float("nan")),
                )

                self._duration_started_at[:, spec.rid_start : spec.rid_end] = (
                    duration_start
                )

                out[:, spec.rid_start : spec.rid_end] = rel
        return out

    @staticmethod
    def _quat_apply_wxyz(
        quat: torch.Tensor,
        vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rotates vec by quaternion quat.

        Args:
            quat: [..., 4], quaternion in (w, x, y, z) format
            vec:  [..., 3]

        Returns:
            torch.Tensor: [..., 3]
        """
        quat = torch.nn.functional.normalize(quat, dim=-1)

        quat_w = quat[..., :1]
        quat_xyz = quat[..., 1:]

        t = 2.0 * torch.cross(quat_xyz, vec, dim=-1)

        return vec + quat_w * t + torch.cross(quat_xyz, t, dim=-1)

    @staticmethod
    def _quat_apply_inverse_wxyz(
        quat: torch.Tensor,
        vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rotates vec by the inverse of quaternion quat.

        Args:
            quat: [..., 4], quaternion in (w, x, y, z) format
            vec:  [..., 3]

        Returns:
            torch.Tensor: [..., 3]
        """
        quat = torch.nn.functional.normalize(quat, dim=-1)

        quat_inverse = quat.clone()
        quat_inverse[..., 1:] = -quat_inverse[..., 1:]

        return RelationMonitor._quat_apply_wxyz(
            quat_inverse,
            vec,
        )

    def _get_orientations(
        self,
        names: List[str],
        quat_cache: Dict[Tuple[str, ...], torch.Tensor],
    ) -> torch.Tensor:
        key = tuple(names)

        if key in quat_cache:
            return quat_cache[key]

        quat_list = [
            self._resolve_name_orientation(name) for name in names
        ]  # each [E, 4]

        quat = torch.stack(quat_list, dim=1)  # [E, N, 4]

        quat_cache[key] = quat
        return quat

    def _get_positions(
        self,
        names: List[str],
        pos_cache: Dict[Tuple[str, ...], torch.Tensor],
    ) -> torch.Tensor:
        key = tuple(names)
        if key in pos_cache:
            return pos_cache[key]

        pos_list = [self._resolve_name_position(n) for n in names]  # each [E,3]
        pos = torch.stack(pos_list, dim=1)  # [E,N,3]
        pos_cache[key] = pos
        return pos

    def _get_linear_speed(
        self,
        names: List[str],
        cache: Dict[Tuple[str, ...], torch.Tensor],
    ) -> torch.Tensor:
        key = tuple(names)
        if key in cache:
            return cache[key]

        vel_list = [
            self._resolve_name_velocity(n, angular=False) for n in names
        ]  # each [E,3]
        vel = torch.stack(vel_list, dim=1)  # [E,N,3]
        speed = torch.linalg.norm(vel, dim=-1)  # [E,N]
        cache[key] = speed
        return speed

    def _get_angular_speed(
        self,
        names: List[str],
        cache: Dict[Tuple[str, ...], torch.Tensor],
    ) -> torch.Tensor:
        key = tuple(names)
        if key in cache:
            return cache[key]

        vel_list = [self._resolve_name_velocity(n, angular=True) for n in names]
        vel = torch.stack(vel_list, dim=1)
        speed = torch.linalg.norm(vel, dim=-1)
        cache[key] = speed
        return speed

    def _get_active_mask(
        self,
        names: List[str],
        active_cache: Dict[Tuple[str, ...], torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns [E, N] bool.
        For robot_parts it is always True.
        For objects: taken from scene_spawner, if available.
        """
        key = tuple(names)
        if key in active_cache:
            return active_cache[key]

        # robot parts always active
        if all(self._is_robot_part_name(n) for n in names):
            m = torch.ones(
                (self.num_envs, len(names)), dtype=torch.bool, device=self.device
            )
            active_cache[key] = m
            return m

        m = torch.zeros(
            (self.num_envs, len(names)), dtype=torch.bool, device=self.device
        )

        spawner = getattr(self.env, "scene_spawner", None)
        if spawner is not None:
            # 1) preferred path: explicit API
            # if hasattr(spawner, "is_object_active"):
            #     for j, n in enumerate(names):
            #         # try per-env signature first
            #         col = []
            #         for env_id in range(self.num_envs):
            #             try:
            #                 v = bool(spawner.is_object_active(n, env_id=env_id))
            #             except TypeError:
            #                 # fallback: global signature
            #                 v = bool(spawner.is_object_active(n))
            #             col.append(v)
            #         m[:, j] = torch.tensor(col, dtype=torch.bool, device=self.device)

            #     active_cache[key] = m
            #     return m

            if hasattr(spawner, "is_object_active"):
                for j, n in enumerate(names):
                    # Normalize aliases of the form base_idx -> base
                    query = n

                    if not self._object_in_scene(n):
                        mm = self.INDEXED_NAME_RE.match(n)
                        if mm and self._object_in_scene(mm.group(1)):
                            query = mm.group(1)
                        elif (
                            n in getattr(spawner, "object_metadata", {})
                            and spawner.object_metadata[n].get("object_kind") == "guide"
                        ):
                            query = n
                        else:
                            continue

                    col = []
                    for env_id in range(self.num_envs):
                        try:
                            v = bool(spawner.is_object_active(query, env_id=env_id))
                        except TypeError:
                            v = bool(spawner.is_object_active(query))
                        col.append(v)
                    m[:, j] = torch.tensor(col, dtype=torch.bool, device=self.device)

                active_cache[key] = m
                return m

            # 2) fallback: last_active_objects (global)
            active_set = set(getattr(spawner, "last_active_objects", []))
            if len(active_set) > 0:
                per_name = [(n in active_set) for n in names]
                row = torch.tensor(
                    per_name, dtype=torch.bool, device=self.device
                ).unsqueeze(0)
                m = row.repeat(self.num_envs, 1)
                active_cache[key] = m
                return m

        # 3) safe fallback: unknown -> inactive (to avoid false positives)
        # print("\n[DEBUG _get_active_mask]")
        # print("names:", [repr(n) for n in names])
        # print("spawner:", type(spawner).__name__ if spawner is not None else None)
        # print("has is_object_active:", hasattr(spawner, "is_object_active") if spawner is not None else False)
        # print("last_active_objects:", [repr(x) for x in getattr(spawner, "last_active_objects", [])])

        # active_set = set(getattr(spawner, "last_active_objects", []))
        # for n in names:
        #     print(f"name={n!r}, in_last_active={n in active_set}")

        active_cache[key] = m
        return m

    # -------------------- scene resolve helpers --------------------

    def _is_robot_part_name(self, name: str) -> bool:
        return name == self.ROBOT_ROOT_NAME or name in self.object_groups.get(
            "robot_parts", []
        )

    def _object_in_scene(self, name: str) -> bool:
        try:
            _ = self.scene[name]
            return True
        except Exception:
            return False

    def _resolve_name_position(self, name: str) -> torch.Tensor:
        """
        Returns the object position [E, 3] by name.

        Lookup order:
        1. robot_parts -> body_pos_w
        2. scene[name] -> root_pos_w (exact match)
        3. scene[name_0] -> root_pos_w (fallback for unindexed names)
        4. scene[base][idx] -> root_pos_w[:, idx] (for names like base_1)
        5. Not found -> zeros

        Args:
            name: Object name (e.g. "green_apple", "basket_1", "left_thumb_proximal_base")

        Returns:
            torch.Tensor: Positions [E, 3]. Zeros if the object is not found.
        """
        if self._is_robot_part_name(name):
            return self._get_robot_body_pos(name)

        spawner = getattr(self.env, "scene_spawner", None)
        if (
            spawner is not None
            and name in getattr(spawner, "object_metadata", {})
            and spawner.object_metadata[name].get("object_kind") == "guide"
        ):
            return self._resolve_guide_world_pose(name)[0]

        # 1) Direct name in scene
        if self._object_in_scene(name):
            return self._get_asset_root_pos(self.scene[name])

        # 2) Fallback: unindexed name -> try to find it with _0
        m = self.INDEXED_NAME_RE.match(name)
        if not m:
            name_with_idx = f"{name}_0"
            if self._object_in_scene(name_with_idx):
                return self._get_asset_root_pos(self.scene[name_with_idx])

        # 3) Indexed name: base_1 -> scene[base][:, 1]
        if m:
            base, idx_s = m.group(1), m.group(2)
            idx = int(idx_s)
            if self._object_in_scene(base):
                return self._get_asset_root_pos(self.scene[base], index=idx)

        # 4) Not found
        return torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    def _resolve_name_orientation(self, name: str) -> torch.Tensor:
        """
        Returns the quaternion [E, 4] in (w, x, y, z) format.

        Lookup order repeats _resolve_name_position():
        1. robot_parts -> body_quat_w
        2. scene[name] -> root_quat_w
        3. scene[name_0] -> root_quat_w
        4. scene[base][idx] -> root_quat_w[:, idx]
        5. Not found -> identity quaternion
        """
        if self._is_robot_part_name(name):
            return self._get_robot_body_quat(name)

        spawner = getattr(self.env, "scene_spawner", None)
        object_metadata = getattr(spawner, "object_metadata", {})

        spawner = getattr(self.env, "scene_spawner", None)
        object_metadata = getattr(spawner, "object_metadata", {})

        # Guide: the pose is computed by a single method (taking anchor_prim into account)
        if (
            object_metadata
            and object_metadata.get(name, {}).get("object_kind") == "guide"
        ):
            return self._resolve_guide_world_pose(name)[1]

        # If the spawner knows the set of spawnable objects and the name is not in it,
        # don't mistakenly treat an arbitrary scene asset as a task object.
        if object_metadata and name not in object_metadata:
            m = self.INDEXED_NAME_RE.match(name)

            if not (m is not None and m.group(1) in object_metadata):
                quat = torch.zeros(
                    (self.num_envs, 4),
                    dtype=torch.float32,
                    device=self.device,
                )
                quat[:, 0] = 1.0
                return quat

        if self._object_in_scene(name):
            return self._get_asset_root_quat(self.scene[name])

        m = self.INDEXED_NAME_RE.match(name)

        if not m:
            name_with_idx = f"{name}_0"

            if self._object_in_scene(name_with_idx):
                return self._get_asset_root_quat(self.scene[name_with_idx])

        if m:
            base, idx_s = m.group(1), m.group(2)
            idx = int(idx_s)

            if self._object_in_scene(base):
                return self._get_asset_root_quat(
                    self.scene[base],
                    index=idx,
                )

        quat = torch.zeros(
            (self.num_envs, 4),
            dtype=torch.float32,
            device=self.device,
        )
        quat[:, 0] = 1.0
        return quat

    def _get_robot_body_pos(self, body_name: str) -> torch.Tensor:
        robot = self.scene["robot"]

        # Alias: the robot root articulation position in the world frame.
        if body_name == self.ROBOT_ROOT_NAME:
            root_pos = robot.data.root_pos_w
            if root_pos.ndim != 2 or root_pos.shape[-1] != 3:
                raise RuntimeError(
                    f"Unexpected robot.data.root_pos_w shape: {tuple(root_pos.shape)}"
                )
            return root_pos

        body_id = None
        if hasattr(robot, "body_names"):
            if body_name in robot.body_names:
                body_id = robot.body_names.index(body_name)
        elif hasattr(robot, "data") and hasattr(robot.data, "body_names"):
            if body_name in robot.data.body_names:
                body_id = robot.data.body_names.index(body_name)

        if body_id is None and hasattr(robot, "find_bodies"):
            ids, _ = robot.find_bodies(body_name)
            if len(ids) > 0:
                body_id = int(ids[0])

        if body_id is None:
            raise KeyError(
                f"Robot body '{body_name}' not found. "
                "Check objects.robot_parts in task.json."
            )

        return robot.data.body_pos_w[:, body_id, :]

    def _get_robot_body_quat(
        self,
        body_name: str,
    ) -> torch.Tensor:
        robot = self.scene["robot"]

        if body_name == self.ROBOT_ROOT_NAME:
            return robot.data.root_quat_w

        body_id = None

        if hasattr(robot, "body_names"):
            if body_name in robot.body_names:
                body_id = robot.body_names.index(body_name)

        elif hasattr(robot, "data") and hasattr(robot.data, "body_names"):
            if body_name in robot.data.body_names:
                body_id = robot.data.body_names.index(body_name)

        if body_id is None and hasattr(robot, "find_bodies"):
            ids, _ = robot.find_bodies(body_name)

            if len(ids) > 0:
                body_id = int(ids[0])

        if body_id is None:
            raise KeyError(
                f"Robot body '{body_name}' not found. "
                "Check objects.robot_parts in task.json"
            )

        return robot.data.body_quat_w[:, body_id, :]

    def _get_asset_root_pos(self, asset, index: int | None = None) -> torch.Tensor:
        """
        Unified handling for the variants:
          - root_pos_w: [E,3]
          - root_pos_w: [E,N,3]
        """
        root_pos = asset.data.root_pos_w
        if root_pos.ndim == 2:
            # [E,3] — singleton
            if index is None or index == 0:
                return root_pos
            return torch.zeros(
                (self.num_envs, 3), dtype=root_pos.dtype, device=root_pos.device
            )
        elif root_pos.ndim == 3:
            # [E,N,3]
            if index is None:
                # if no index is given, use 0
                return root_pos[:, 0, :]
            if index < root_pos.shape[1]:
                return root_pos[:, index, :]
            # out-of-range -> zeros
            return torch.zeros(
                (self.num_envs, 3), dtype=torch.float32, device=self.device
            )
        else:
            raise RuntimeError(f"Unsupported root_pos_w shape: {tuple(root_pos.shape)}")

    def _get_asset_root_quat(
        self,
        asset,
        index: int | None = None,
    ) -> torch.Tensor:
        """
        Unified handling for the variants:
          - root_quat_w: [E, 4]
          - root_quat_w: [E, N, 4]

        Quaternion format: (w, x, y, z).
        """
        root_quat = asset.data.root_quat_w

        if root_quat.ndim == 2:
            if index is None or index == 0:
                return root_quat

            quat = torch.zeros(
                (self.num_envs, 4),
                dtype=root_quat.dtype,
                device=root_quat.device,
            )
            quat[:, 0] = 1.0
            return quat

        if root_quat.ndim == 3:
            if index is None:
                return root_quat[:, 0, :]

            if index < root_quat.shape[1]:
                return root_quat[:, index, :]

            quat = torch.zeros(
                (self.num_envs, 4),
                dtype=root_quat.dtype,
                device=root_quat.device,
            )
            quat[:, 0] = 1.0
            return quat

        raise RuntimeError(f"Unsupported root_quat_w shape: {tuple(root_quat.shape)}")

    def _resolve_name_velocity(self, name: str, angular: bool = False) -> torch.Tensor:
        """
        Analogous to _resolve_name_position, but for velocities.
        Returns [E, 3]. If the object/body is not found — zeros (treated as stationary).
        """
        if self._is_robot_part_name(name):
            return self._get_robot_body_vel(name, angular=angular)

        if self._object_in_scene(name):
            return self._get_asset_root_vel(self.scene[name], angular=angular)

        m = self.INDEXED_NAME_RE.match(name)
        if not m:
            name_with_idx = f"{name}_0"
            if self._object_in_scene(name_with_idx):
                return self._get_asset_root_vel(
                    self.scene[name_with_idx], angular=angular
                )

        if m:
            base, idx_s = m.group(1), m.group(2)
            idx = int(idx_s)
            if self._object_in_scene(base):
                return self._get_asset_root_vel(
                    self.scene[base], index=idx, angular=angular
                )

        return torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    def _get_robot_body_vel(
        self, body_name: str, angular: bool = False
    ) -> torch.Tensor:
        robot = self.scene["robot"]

        if body_name == self.ROBOT_ROOT_NAME:
            return robot.data.root_ang_vel_w if angular else robot.data.root_lin_vel_w

        body_id = None
        if hasattr(robot, "body_names"):
            if body_name in robot.body_names:
                body_id = robot.body_names.index(body_name)
        elif hasattr(robot, "data") and hasattr(robot.data, "body_names"):
            if body_name in robot.data.body_names:
                body_id = robot.data.body_names.index(body_name)

        if body_id is None and hasattr(robot, "find_bodies"):
            ids, _ = robot.find_bodies(body_name)
            if len(ids) > 0:
                body_id = int(ids[0])

        if body_id is None:
            raise KeyError(
                f"Robot body '{body_name}' not found. Check objects.robot_parts in task.json"
            )

        data = robot.data
        if angular:
            if hasattr(data, "body_ang_vel_w"):
                return data.body_ang_vel_w[:, body_id, :]
            return data.body_vel_w[:, body_id, 3:6]
        else:
            if hasattr(data, "body_lin_vel_w"):
                return data.body_lin_vel_w[:, body_id, :]
            return data.body_vel_w[:, body_id, 0:3]

    def _get_asset_root_vel(
        self, asset, index: int | None = None, angular: bool = False
    ) -> torch.Tensor:
        data = asset.data
        if angular:
            vel = getattr(data, "root_ang_vel_w", None)
        else:
            vel = getattr(data, "root_lin_vel_w", None)

        if vel is None:
            return torch.zeros(
                (self.num_envs, 3), dtype=torch.float32, device=self.device
            )

        if vel.ndim == 2:
            if index is None or index == 0:
                return vel
            return torch.zeros((self.num_envs, 3), dtype=vel.dtype, device=vel.device)
        elif vel.ndim == 3:
            if index is None:
                return vel[:, 0, :]
            if index < vel.shape[1]:
                return vel[:, index, :]
            return torch.zeros(
                (self.num_envs, 3), dtype=torch.float32, device=self.device
            )
        else:
            raise RuntimeError(f"Unsupported velocity shape: {tuple(vel.shape)}")

    def emit_episode_aborted(
        self, env_ids, sim_step: int, sim_time: float
    ) -> List[Dict[str, Any]]:
        events = []
        for env_id in env_ids.tolist():
            events.append(
                {
                    "sim_step": int(sim_step),
                    "sim_time": float(sim_time),
                    "env_id": int(env_id),
                    "relation": "episode_aborted",
                    "type": "aborted",
                    "pattern": "episode_aborted",
                    "value": 1,
                }
            )
        return events

    def _describe_duration_state(
        self,
        spec: CompiledRelationSpec,
        rid: int,
        idx_a: int,
        idx_b: int,
        env_id: int,
    ) -> str:
        """Collects diagnostics: why the duration relation is in its current state."""
        parts: List[str] = []

        # 1) the value of the source relation right now
        source_rid = int(spec.source_rid_start) + idx_a * len(spec.names_b) + idx_b
        source_value = bool(self.prev_values[env_id, source_rid].item())
        parts.append(f"source[{spec.source_relation}]={int(source_value)}")

        # 2) how long source has been continuously satisfied
        started_at = float(self._duration_started_at[env_id, rid].item())
        if math.isnan(started_at):
            parts.append("elapsed=none(source_not_active)")
        else:
            elapsed = max(0.0, float(self._last_sim_time) - started_at)
            parts.append(f"elapsed={elapsed:.3f}s")

        # 3) required duration
        parts.append(f"required={float(spec.duration_sec):.3f}s")

        # 4) latch state
        if spec.latch:
            latched = bool(self._duration_satisfied[env_id, rid].item())
            parts.append(f"latched={int(latched)}")

        # 5) source relation details: why it is in its current state
        source_spec = self._relation_to_specs.get(spec.source_relation)
        if source_spec is not None:
            object_a = spec.names_a[idx_a]
            object_b = spec.names_b[idx_b]

            if source_spec.relation_type in {"inside", "not_inside"}:
                parts.append(self._describe_inside_state(object_a, object_b, env_id))
            elif source_spec.relation_type == "distance":
                d = float(self._distance_values[env_id, source_rid].item())
                if math.isfinite(d):
                    parts.append(
                        f"distance={d:.3f}m {source_spec.comparison} "
                        f"threshold={float(source_spec.threshold):.3f}m"
                    )

            # inside/distance are gated by object activity
            spawner = getattr(self.env, "scene_spawner", None)
            if spawner is not None and hasattr(spawner, "is_object_active"):
                try:
                    parts.append(
                        f"active[{object_a}]="
                        f"{int(bool(spawner.is_object_active(object_a)))}"
                    )
                except Exception:
                    pass

        return " ".join(parts)

    def _describe_inside_state(
        self,
        object_a: str,
        object_b: str,
        env_id: int,
    ) -> str:
        """Why object_a is (not) inside container object_b: positions and axes."""
        spawner = getattr(self.env, "scene_spawner", None)
        if spawner is None:
            return "inside: no scene_spawner"

        try:
            obj_pos = self._resolve_name_position(object_a)[env_id]

            meta = spawner.object_metadata.get(object_b)
            if meta is None:
                return f"inside: unknown container '{object_b}'"

            if meta.get("object_kind") == "guide":
                cont_pos_all, cont_quat_all = self._resolve_guide_world_pose(object_b)
                cont_pos = cont_pos_all[env_id]
                cont_quat = cont_quat_all[env_id]
            else:
                cont_pos = self._resolve_name_position(object_b)[env_id]
                cont_quat = self._resolve_name_orientation(object_b)[env_id]

            def fmt(v) -> str:
                return "[" + ",".join(f"{float(x):.3f}" for x in v) + "]"

            delta_w = obj_pos - cont_pos

            guide_radius = meta.get("guide_radius")
            if guide_radius is not None:
                dist = float(torch.linalg.norm(delta_w).item())
                return (
                    f"obj_pos={fmt(obj_pos)} zone_center={fmt(cont_pos)} "
                    f"dist={dist:.3f}m radius={float(guide_radius):.3f}m"
                )

            local = self._quat_apply_inverse_wxyz(
                cont_quat.view(1, 4),
                delta_w.view(1, 3),
            ).view(3)

            half = [0.5 * float(s) for s in meta["size"]]

            axis_parts = []
            for axis_name, off, h in zip("xyz", local.tolist(), half):
                mark = "ok" if abs(off) <= h else "OUT"
                axis_parts.append(f"{axis_name}:{off:+.3f}/{h:.3f}={mark}")

            return (
                f"obj_pos={fmt(obj_pos)} zone_center={fmt(cont_pos)} "
                f"axes({' '.join(axis_parts)})"
            )
        except Exception as e:
            return f"inside_diag_error={e!r}"

    # ------- annotations -----------------------------------
    def add_annotation(
        self,
        *,
        command: Dict[str, Any],
        sim_time: float,
        env_id: int = 0,
    ) -> None:
        """Register one executed scripted-policy annotation command."""
        env_id = int(env_id)
        self._validate_annotation_env_id(env_id)

        self.annotation_managers[env_id].add_annotation(
            command=command,
            sim_time=float(sim_time),
        )

    def get_episode_annotations(
        self,
        *,
        env_id: int = 0,
        sim_time: float,
    ) -> Dict[str, list[Dict[str, Any]]]:
        """
        Finalize the annotations of the episode being completed and return the
        payload in the DatasetLoggingManager format.

        The method closes the last open action with end_timestamp=sim_time.
        Calling it again is safe: after the first call there is no open action
        left, and the payload stays the same.
        """
        env_id = int(env_id)
        self._validate_annotation_env_id(env_id)

        return self.annotation_managers[env_id].finalize(
            sim_time=float(sim_time),
        )

    def get_annotations(
        self,
        *,
        env_id: int = 0,
    ) -> Dict[str, list[Dict[str, Any]]]:
        """Return a defensive copy of annotations accumulated so far."""
        env_id = int(env_id)
        self._validate_annotation_env_id(env_id)

        return self.annotation_managers[env_id].result()

    def _validate_annotation_env_id(self, env_id: int) -> None:
        if not 0 <= env_id < self.num_envs:
            raise IndexError(
                f"Annotation env_id out of range: {env_id}; "
                f"expected [0, {self.num_envs})"
            )
