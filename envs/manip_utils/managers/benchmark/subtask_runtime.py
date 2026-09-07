from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


@dataclass(frozen=True)
class SubtaskTransition:
    name: str
    object_a: Optional[str] = None
    object_b: Optional[str] = None
    rids: tuple[int, ...] = ()


@dataclass(frozen=True)
class SubtaskSpec:
    task: str
    subtask: str
    relations: List[SubtaskTransition]


class SubtaskRuntime:
    """
    Sequential runtime for subtasks.

    RelationMonitor remains the sole owner of relation computation.
    This class only receives the transition events from monitor.update() and
    maintains its own current-state cache for the relation instances needed by
    annotations.
    """

    def __init__(
        self,
        relation_monitor,
        annotations: Dict[str, Dict[str, Any]],
        num_envs: int,
    ):
        self._relation_monitor = relation_monitor
        self._num_envs = int(num_envs)
        self._subtasks = self._parse_annotations(annotations)

        self._current_indices: List[Optional[int]] = [
            None for _ in range(self._num_envs)
        ]
        self._started: List[bool] = [False for _ in range(self._num_envs)]

        self._tracked_rids: Set[int] = {
            rid
            for subtask in self._subtasks
            for relation in subtask.relations
            for rid in relation.rids
        }

        self._relation_states: List[Dict[int, bool]] = [
            {rid: False for rid in self._tracked_rids} for _ in range(self._num_envs)
        ]

    def reset(
        self,
        env_ids=None,
        sim_time: float = 0.0,
    ) -> List[Dict[str, Any]]:
        if env_ids is None:
            env_ids = range(self._num_envs)

        events: List[Dict[str, Any]] = []

        for env_id in env_ids:
            env_id = int(env_id)
            self._validate_env_id(env_id)

            self._relation_states[env_id] = {rid: False for rid in self._tracked_rids}

            if self._started[env_id]:
                continue

            if not self._subtasks:
                self._current_indices[env_id] = None
                self._started[env_id] = True
                continue

            self._current_indices[env_id] = 0
            self._started[env_id] = True

            print(
                "\033[1;93m"
                f"[SubtaskRuntime]\033[1;36m Started subtask {0}: {self._subtasks[0].subtask}"
                "\033[0m"
            )

            events.append(
                self._make_started_event(
                    env_id=env_id,
                    sim_time=sim_time,
                    subtask_index=0,
                    subtask=self._subtasks[0],
                )
            )

        return events

    def begin_episode(
        self,
        env_ids=None,
        sim_time: float = 0.0,
    ) -> List[Dict[str, Any]]:
        if env_ids is None:
            env_ids = range(self._num_envs)

        for env_id in env_ids:
            env_id = int(env_id)
            self._validate_env_id(env_id)

            self._current_indices[env_id] = None
            self._started[env_id] = False
            self._relation_states[env_id] = {rid: False for rid in self._tracked_rids}

        return self.reset(
            env_ids=env_ids,
            sim_time=sim_time,
        )

    def process(
        self,
        relation_events: List[Dict[str, Any]],
        sim_step: int,
        sim_time: float,
        env_ids=None,
    ) -> List[Dict[str, Any]]:
        """
        Apply the RelationMonitor transition events and advance each env by at
        most one subtask.

        The relation instance state is updated exclusively from event["rid"]
        and event["value"]. There is no re-querying of RelationMonitor.
        """
        if env_ids is None:
            env_ids = range(self._num_envs)

        env_ids_set = {int(env_id) for env_id in env_ids}
        for env_id in env_ids_set:
            self._validate_env_id(env_id)

        for event in relation_events:
            env_id = int(event.get("env_id", -1))
            rid = event.get("rid")

            if env_id not in env_ids_set or rid is None:
                continue

            rid = int(rid)
            if rid not in self._tracked_rids:
                continue

            self._relation_states[env_id][rid] = bool(event.get("value", 0))

        events: List[Dict[str, Any]] = []

        for env_id in env_ids_set:
            current_subtask = self._get_current_subtask(env_id)
            if current_subtask is None:
                continue

            if not self._are_relations_satisfied(
                subtask=current_subtask,
                env_id=env_id,
            ):
                continue

            completed_index = self._current_indices[env_id]
            if completed_index is None:
                continue

            events.append(
                self._make_completed_event(
                    env_id=env_id,
                    sim_step=sim_step,
                    sim_time=sim_time,
                    subtask_index=completed_index,
                    subtask=current_subtask,
                )
            )

            next_index = completed_index + 1

            if next_index >= len(self._subtasks):
                self._current_indices[env_id] = None
                continue

            self._current_indices[env_id] = next_index

            print(
                "\033[1;93m"
                f"[SubtaskRuntime]\033[1;36m Started subtask {next_index}: "
                f"{self._subtasks[next_index].subtask}"
                "\033[0m"
            )

            events.append(
                self._make_started_event(
                    env_id=env_id,
                    sim_time=sim_time,
                    subtask_index=next_index,
                    subtask=self._subtasks[next_index],
                )
            )

        return events

    def get_task(self, env_id: int = 0) -> Optional[str]:
        subtask = self._get_current_subtask(env_id)
        return None if subtask is None else subtask.task

    def get_subtask(self, env_id: int = 0) -> Optional[str]:
        subtask = self._get_current_subtask(env_id)
        return None if subtask is None else subtask.subtask

    def is_finished(self, env_id: int = 0) -> bool:
        env_id = int(env_id)
        self._validate_env_id(env_id)
        return self._current_indices[env_id] is None

    def _get_current_subtask(self, env_id: int) -> Optional[SubtaskSpec]:
        env_id = int(env_id)
        self._validate_env_id(env_id)

        current_index = self._current_indices[env_id]
        if current_index is None:
            return None

        return self._subtasks[current_index]

    def _are_relations_satisfied(
        self,
        subtask: SubtaskSpec,
        env_id: int,
    ) -> bool:
        if not subtask.relations:
            return False

        states = self._relation_states[env_id]

        return all(
            all(states.get(rid, False) for rid in relation.rids)
            for relation in subtask.relations
        )

    def _parse_annotations(
        self,
        annotations: Dict[str, Dict[str, Any]],
    ) -> List[SubtaskSpec]:
        subtasks: List[SubtaskSpec] = []

        for annotation_index in sorted(annotations, key=int):
            annotation = annotations[annotation_index]

            task = annotation.get("task")
            subtask = annotation.get("subtask")
            relations_cfg = annotation.get("relations", [])

            if not isinstance(task, str) or not task:
                raise ValueError(
                    f"annotations[{annotation_index!r}].task "
                    "must be a non-empty string"
                )

            if not isinstance(subtask, str) or not subtask:
                raise ValueError(
                    f"annotations[{annotation_index!r}].subtask "
                    "must be a non-empty string"
                )

            if not isinstance(relations_cfg, list):
                raise ValueError(
                    f"annotations[{annotation_index!r}].relations must be a list"
                )

            if not relations_cfg:
                print(
                    "\033[1;93m"
                    "[SubtaskRuntime] \033[1;31mEmpty relation set: "
                    f"annotation={annotation_index!r}, "
                    "\033[0m"
                    f"task={task!r}, "
                    f"subtask={subtask!r}. "
                    "This subtask will remain active until episode termination."
                )

            relations: List[SubtaskTransition] = []

            for relation_index, relation_cfg in enumerate(relations_cfg):
                if not isinstance(relation_cfg, dict):
                    raise ValueError(
                        f"annotations[{annotation_index!r}].relations"
                        f"[{relation_index}] must be an object"
                    )

                name = relation_cfg.get("name")
                object_a = relation_cfg.get("object_a")
                object_b = relation_cfg.get("object_b")

                if not isinstance(name, str) or not name:
                    raise ValueError(
                        f"annotations[{annotation_index!r}].relations"
                        f"[{relation_index}].name must be a non-empty string"
                    )

                if object_a is not None and not isinstance(object_a, str):
                    raise ValueError(
                        f"annotations[{annotation_index!r}].relations"
                        f"[{relation_index}].object_a must be a string or null"
                    )

                if object_b is not None and not isinstance(object_b, str):
                    raise ValueError(
                        f"annotations[{annotation_index!r}].relations"
                        f"[{relation_index}].object_b must be a string or null"
                    )

                try:
                    rids = tuple(
                        self._relation_monitor.resolve_relation_rids(
                            relation_key=name,
                            object_a=object_a,
                            object_b=object_b,
                        )
                    )
                except (KeyError, ValueError) as exc:
                    print(
                        "\033[1;31m"
                        "[SubtaskRuntime] structure/benchmark mismatch:"
                        "\033[0m\n"
                        f"  annotation={annotation_index!r}, "
                        f"relation_index={relation_index}\n"
                        f"  subtask={subtask!r}\n"
                        f"  relation={name!r}, "
                        f"object_a={object_a!r}, object_b={object_b!r}\n"
                        f"  error={exc}\n"
                        "  Check that structure.json was updated after changing the "
                        "benchmark relation or its object placeholders."
                    )
                    raise

                if not rids:
                    raise ValueError(
                        f"annotations[{annotation_index!r}].relations"
                        f"[{relation_index}] resolved to no relation instances"
                    )

                relations.append(
                    SubtaskTransition(
                        name=name,
                        object_a=object_a,
                        object_b=object_b,
                        rids=rids,
                    )
                )

            subtasks.append(
                SubtaskSpec(
                    task=task,
                    subtask=subtask,
                    relations=relations,
                )
            )

        return subtasks

    @staticmethod
    def _make_started_event(
        env_id: int,
        sim_time: float,
        subtask_index: int,
        subtask: SubtaskSpec,
    ) -> Dict[str, Any]:
        event = {
            "env_id": int(env_id),
            "sim_time": float(sim_time),
            "event": "started",
            "subtask_index": int(subtask_index),
            "task": subtask.task,
            "subtask": subtask.subtask,
        }

        if not subtask.relations:
            event["warning"] = "Relation list is empty"

        return event

    @staticmethod
    def _make_completed_event(
        env_id: int,
        sim_step: int,
        sim_time: float,
        subtask_index: int,
        subtask: SubtaskSpec,
    ) -> Dict[str, Any]:
        return {
            "env_id": int(env_id),
            "sim_step": int(sim_step),
            "sim_time": float(sim_time),
            "event": "completed",
            "subtask_index": int(subtask_index),
            "task": subtask.task,
            "subtask": subtask.subtask,
        }

    def _validate_env_id(self, env_id: int) -> None:
        if env_id < 0 or env_id >= self._num_envs:
            raise IndexError(f"env_id={env_id} is outside [0, {self._num_envs})")
