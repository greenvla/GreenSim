#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Union, Any
from collections import defaultdict

# ============================================================
# TERMINAL COLORS
# ============================================================


class TermColor:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


# ============================================================
# DATA MODELS
# ============================================================


@dataclass(frozen=True)
class TaskInfo:
    name: str = "unknown"
    description: str = ""


@dataclass(frozen=True)
class RelationSpec:
    name: str
    pattern: str
    type: str
    distance: Optional[float] = None
    duration_sec: Optional[float] = None
    source_relation: Optional[str] = None
    lin_vel_eps: Optional[float] = None
    ang_vel_eps: Optional[float] = None
    placeholder_groups: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GoalSpec:
    name: str
    description: str
    for_each: str
    must_have: Tuple[str, ...]
    required: bool = False
    match_mode: str = "all"
    min_satisfied: int = 1


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    description: str = ""
    forbid_transition: Optional[Tuple[str, Tuple[int, int]]] = None
    limit_transitions: Optional[Tuple[str, Tuple[int, int]]] = None
    forbid_value: Optional[Tuple[str, int]] = None
    before: Optional[Tuple[str, int]] = None
    if_object_type: Optional[str] = None
    per_object: bool = False
    max_count: Optional[int] = None


@dataclass(frozen=True)
class SuccessSpec:
    required_goals: Tuple[str, ...] = field(default_factory=tuple)
    max_violations: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedRelationKey:
    message_key: str
    relation_name: str
    relation_type: str
    object_1: Optional[str]
    object_2: Optional[str]
    matched_groups: Dict[str, str]


@dataclass(frozen=True)
class TransitionEvent:
    message_key: str
    relation_name: str
    object_1: Optional[str]
    object_2: Optional[str]
    from_value: int
    to_value: int
    timestamp: float


@dataclass
class GoalEvaluation:
    status: str
    required: bool
    description: str
    details: Dict[str, Any]


@dataclass
class ConstraintEvaluation:
    violations: int
    limit: int
    passed: bool
    description: str
    details: Any
    violation_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AnalyzerState:
    spawned_objects: Dict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    current_snapshot: Dict[str, int] = field(default_factory=dict)
    previous_snapshot: Dict[str, int] = field(default_factory=dict)
    transitions_by_key: Dict[str, List[TransitionEvent]] = field(
        default_factory=lambda: defaultdict(list)
    )
    recognized_relations: Dict[str, ParsedRelationKey] = field(default_factory=dict)
    processed_messages: int = 0


# ============================================================
# TASK MODEL
# ============================================================


class TaskModel:
    def __init__(self, raw_config: Dict[str, Any]):
        self.raw = raw_config

        self.task = self._parse_task_info(raw_config.get("task", {}))
        self.objects_cfg: Dict[str, Dict[str, List[int]]] = raw_config.get(
            "objects", {}
        )
        self.relations: Dict[str, RelationSpec] = self._parse_relations(
            raw_config.get("relations", {})
        )
        self.goals: Dict[str, GoalSpec] = self._parse_goals(raw_config.get("goals", {}))
        self.constraints: Dict[str, ConstraintSpec] = self._parse_constraints(
            raw_config.get("constraints", {})
        )
        self.success = self._parse_success(raw_config.get("success", {}))

        self.group_object_lookup: Dict[str, Set[str]] = defaultdict(set)
        self.object_to_groups: Dict[str, Set[str]] = defaultdict(set)
        self.static_objects_by_group: Dict[str, Set[str]] = defaultdict(set)

        self._build_object_indexes()
        self._validate()

    @staticmethod
    def _parse_task_info(task_cfg: Dict[str, Any]) -> TaskInfo:
        return TaskInfo(
            name=task_cfg.get("name", "unknown"),
            description=task_cfg.get("description", ""),
        )

    @staticmethod
    def _parse_relations(relations_cfg: Dict[str, Any]) -> Dict[str, RelationSpec]:
        result = {}
        for rel_name, rel_cfg in relations_cfg.items():
            pattern = rel_cfg.get("pattern", "")
            placeholders = tuple(re.findall(r"\{(\w+)\}", pattern))
            result[rel_name] = RelationSpec(
                name=rel_name,
                pattern=pattern,
                type=rel_cfg.get("type"),
                distance=rel_cfg.get("distance"),
                duration_sec=rel_cfg.get("duration_sec"),
                source_relation=rel_cfg.get("relation"),
                lin_vel_eps=rel_cfg.get("lin_vel_eps"),
                ang_vel_eps=rel_cfg.get("ang_vel_eps"),
                placeholder_groups=placeholders,
            )
        return result

    @staticmethod
    def _parse_goals(goals_cfg: Dict[str, Any]) -> Dict[str, GoalSpec]:
        result = {}
        for goal_name, goal_cfg in goals_cfg.items():
            raw_must_have = goal_cfg.get("must_have")
            if isinstance(raw_must_have, str):
                must_have = (raw_must_have,)
            elif isinstance(raw_must_have, (list, tuple)):
                must_have = tuple(raw_must_have)
            else:
                raise ValueError(
                    f"goals.{goal_name}.must_have must be a string or a list of strings"
                )
            match_mode = goal_cfg.get("match_mode", "all")
            if match_mode not in {"all", "any"}:
                raise ValueError(
                    f"goals.{goal_name}.match_mode must be 'all' or 'any', "
                    f"got '{match_mode}'"
                )

            min_satisfied = int(goal_cfg.get("min_satisfied", 1))
            if min_satisfied < 1:
                raise ValueError(f"goals.{goal_name}.min_satisfied must be >= 1")

            result[goal_name] = GoalSpec(
                name=goal_name,
                description=goal_cfg.get("description", ""),
                for_each=goal_cfg.get("for_each"),
                must_have=must_have,
                required=bool(goal_cfg.get("required", False)),
                match_mode=match_mode,
                min_satisfied=min_satisfied,
            )
        return result

    @staticmethod
    def _parse_constraints(
        constraints_cfg: Dict[str, Any],
    ) -> Dict[str, ConstraintSpec]:
        result = {}
        for const_name, const_cfg in constraints_cfg.items():
            forbid_transition = None
            if "forbid_transition" in const_cfg:
                rel_name, values = const_cfg["forbid_transition"]
                forbid_transition = (rel_name, (int(values[0]), int(values[1])))

            limit_transitions = None
            if "limit_transitions" in const_cfg:
                rel_name, values = const_cfg["limit_transitions"]
                limit_transitions = (rel_name, (int(values[0]), int(values[1])))

            forbid_value = None
            if "forbid_value" in const_cfg:
                rel_name, value = const_cfg["forbid_value"]
                forbid_value = (rel_name, int(value))

            before = None
            if "before" in const_cfg:
                rel_name, value = const_cfg["before"]
                before = (rel_name, int(value))

            result[const_name] = ConstraintSpec(
                name=const_name,
                description=const_cfg.get("description", ""),
                forbid_transition=forbid_transition,
                limit_transitions=limit_transitions,
                forbid_value=forbid_value,
                before=before,
                if_object_type=const_cfg.get("if_object_type"),
                per_object=bool(const_cfg.get("per_object", False)),
                max_count=const_cfg.get("max_count"),
            )
        return result

    @staticmethod
    def _parse_success(success_cfg: Dict[str, Any]) -> SuccessSpec:
        return SuccessSpec(
            required_goals=tuple(success_cfg.get("required_goals", [])),
            max_violations=dict(success_cfg.get("max_violations", {})),
        )

    def _build_object_indexes(self):
        for group_name, group_cfg in self.objects_cfg.items():
            for prefix, indices in group_cfg.items():
                if indices == []:
                    obj_name = prefix
                    self.group_object_lookup[group_name].add(obj_name)
                    self.object_to_groups[obj_name].add(group_name)
                    self.static_objects_by_group[group_name].add(obj_name)
                else:
                    for idx in indices:
                        obj_name = f"{prefix}_{idx}"
                        self.group_object_lookup[group_name].add(obj_name)
                        self.object_to_groups[obj_name].add(group_name)

    def _validate(self):
        for rel_name, rel in self.relations.items():
            if rel.type not in {
                "distance",
                "spawn",
                "collision",
                "timeout",
                "static",
                "upright",
                "inside",
                "not_inside",
                "duration",
            }:
                raise ValueError(
                    f"relations.{rel_name}.type must be one of: distance, spawn, collision, static, upright, inside, not_inside"
                )
            if rel.type == "distance" and rel.distance is None:
                raise ValueError(
                    f"relations.{rel_name}.distance is required for type='distance'"
                )
            if rel.type == "timeout" and rel.duration_sec is None:
                raise ValueError(
                    f"relations.{rel_name}.duration_sec is required for type='timeout'"
                )
            if rel.type == "duration":
                if rel.duration_sec is None:
                    raise ValueError(
                        f"relations.{rel_name}.duration_sec is required "
                        "for type='duration'"
                    )

                if float(rel.duration_sec) <= 0.0:
                    raise ValueError(f"relations.{rel_name}.duration_sec must be > 0")

                if not rel.source_relation:
                    raise ValueError(
                        f"relations.{rel_name}.relation is required "
                        "for type='duration'"
                    )

                if rel.source_relation not in self.relations:
                    raise ValueError(
                        f"relations.{rel_name}.relation references unknown relation "
                        f"'{rel.source_relation}'"
                    )

                if not rel.pattern:
                    raise ValueError(
                        f"relations.{rel_name}.pattern is required "
                        "for type='duration'"
                    )
            for group_name in rel.placeholder_groups:
                if group_name not in self.objects_cfg:
                    raise ValueError(
                        f"relations.{rel_name}.pattern references unknown group '{group_name}'"
                    )

        for goal_name, goal in self.goals.items():
            if goal.for_each not in self.objects_cfg:
                raise ValueError(
                    f"goals.{goal_name}.for_each references unknown group '{goal.for_each}'"
                )
            for rel_name in goal.must_have:
                if rel_name not in self.relations:
                    raise ValueError(
                        f"goals.{goal_name}.must_have references unknown relation '{rel_name}'"
                    )

        for const_name, const in self.constraints.items():
            if const.forbid_transition:
                if const.forbid_transition[0] not in self.relations:
                    raise ValueError(
                        f"constraints.{const_name}.forbid_transition references unknown relation '{const.forbid_transition[0]}'"
                    )
            if const.limit_transitions:
                if const.limit_transitions[0] not in self.relations:
                    raise ValueError(
                        f"constraints.{const_name}.limit_transitions references unknown relation '{const.limit_transitions[0]}'"
                    )
            if const.forbid_value:
                if const.forbid_value[0] not in self.relations:
                    raise ValueError(
                        f"constraints.{const_name}.forbid_value references unknown relation '{const.forbid_value[0]}'"
                    )
            if const.before:
                if const.before[0] not in self.relations:
                    raise ValueError(
                        f"constraints.{const_name}.before references unknown relation '{const.before[0]}'"
                    )
            if const.if_object_type and const.if_object_type not in self.objects_cfg:
                raise ValueError(
                    f"constraints.{const_name}.if_object_type references unknown group '{const.if_object_type}'"
                )

        for goal_name in self.success.required_goals:
            if goal_name not in self.goals:
                raise ValueError(
                    f"success.required_goals references unknown goal '{goal_name}'"
                )

        for const_name in self.success.max_violations:
            if const_name not in self.constraints:
                raise ValueError(
                    f"success.max_violations references unknown constraint '{const_name}'"
                )


# ============================================================
# RELATION MATCHER
# ============================================================


class RelationMatcher:
    def __init__(self, task_model: TaskModel):
        self.task = task_model
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        self._compile()

    def _compile(self):
        for rel_name, rel in self.task.relations.items():
            regex = rel.pattern
            for group_name in rel.placeholder_groups:
                regex = regex.replace(f"{{{group_name}}}", f"(?P<{group_name}>[\\w_]+)")
            regex = f"^{regex}$"
            self._compiled_patterns[rel_name] = re.compile(regex)

    def normalize_object_name(self, obj_name: str, group_name: str) -> str:
        if obj_name.endswith("_spawn"):
            candidate = obj_name[:-6]
            if candidate in self.task.group_object_lookup.get(group_name, set()):
                return candidate
        return obj_name

    def parse(self, message_key: str) -> Optional[ParsedRelationKey]:
        for rel_name, rel in self.task.relations.items():
            match = self._compiled_patterns[rel_name].match(message_key)
            if not match:
                continue

            raw_groups = match.groupdict()
            normalized_groups: Dict[str, str] = {}

            valid = True
            for group_name, obj_name in raw_groups.items():
                obj_name = self.normalize_object_name(obj_name, group_name)
                if obj_name not in self.task.group_object_lookup.get(group_name, set()):
                    valid = False
                    break
                normalized_groups[group_name] = obj_name

            if not valid:
                continue

            ordered = [normalized_groups[g] for g in rel.placeholder_groups]

            return ParsedRelationKey(
                message_key=message_key,
                relation_name=rel_name,
                relation_type=rel.type,
                object_1=ordered[0] if len(ordered) >= 1 else None,
                object_2=ordered[1] if len(ordered) >= 2 else None,
                matched_groups=normalized_groups,
            )
        return None


# ============================================================
# STATE TRACKER
# ============================================================


class StateTracker:
    def __init__(
        self,
        task_model: TaskModel,
        matcher: RelationMatcher,
        robot_body_names: Optional[Set[str]] = None,
    ):
        self.task = task_model
        self.matcher = matcher
        self.state = AnalyzerState()

        if robot_body_names is None:
            self.state.spawned_objects["robot_parts"].update(
                self.task.group_object_lookup.get("robot_parts", set())
            )
        else:
            robot_body_names = set(robot_body_names)

            for group_name, group_objects in self.task.group_object_lookup.items():
                if group_objects and group_objects.issubset(robot_body_names):
                    self.state.spawned_objects[group_name].update(group_objects)

    def update(self, snapshot: Dict[str, int], timestamp: float):
        self.state.processed_messages += 1

        for key in snapshot.keys():
            if key not in self.state.recognized_relations:
                parsed = self.matcher.parse(key)
                if parsed:
                    self.state.recognized_relations[key] = parsed

        for key, value in snapshot.items():
            if int(value) != 1:
                continue

            parsed = self.state.recognized_relations.get(key)
            if not parsed or parsed.relation_type != "spawn" or parsed.object_1 is None:
                continue

            for group_name in self.task.object_to_groups.get(parsed.object_1, set()):
                self.state.spawned_objects[group_name].add(parsed.object_1)

        for key, curr_value in snapshot.items():
            curr_value = int(curr_value)
            prev_value = self.state.current_snapshot.get(key, 0)

            if prev_value != curr_value:
                parsed = self.state.recognized_relations.get(key)
                if parsed:
                    event = TransitionEvent(
                        message_key=key,
                        relation_name=parsed.relation_name,
                        object_1=parsed.object_1,
                        object_2=parsed.object_2,
                        from_value=prev_value,
                        to_value=curr_value,
                        timestamp=timestamp,
                    )
                    self.state.transitions_by_key[key].append(event)

        self.state.previous_snapshot = self.state.current_snapshot.copy()
        # self.state.current_snapshot = {k: int(v) for k, v in snapshot.items()}
        for k, v in snapshot.items():
            self.state.current_snapshot[k] = int(v)

    def get_active_objects(self, group_name: str) -> List[str]:
        return sorted(self.state.spawned_objects.get(group_name, set()))

    def get_all_active_objects(self) -> List[str]:
        result = set()
        for objs in self.state.spawned_objects.values():
            result.update(objs)
        return sorted(result)

    def _requires_active_second_object(
        self,
        parsed: ParsedRelationKey,
    ) -> bool:
        return (
            parsed.relation_type
            not in {
                "inside",
                "not_inside",
                "duration",
            }
            and parsed.object_2 is not None
            and parsed.object_2 in self.task.object_to_groups
        )

    def relation_keys(
        self,
        relation_name: str,
        object_1: Optional[str] = None,
        object_2: Optional[str] = None,
        active_only: bool = True,
    ) -> List[str]:
        result = []
        active_objects = set(self.get_all_active_objects())

        for key, parsed in self.state.recognized_relations.items():
            if parsed.relation_name != relation_name:
                continue
            if object_1 is not None and parsed.object_1 != object_1:
                continue
            if object_2 is not None and parsed.object_2 != object_2:
                continue

            if active_only:
                if parsed.object_1 and parsed.object_1 not in active_objects:
                    continue
                if self._requires_active_second_object(parsed):
                    if parsed.object_2 not in active_objects:
                        continue

            result.append(key)

        return result

    def aggregated_relation_value(
        self,
        relation_name: str,
        object_1: Optional[str] = None,
        object_2: Optional[str] = None,
        active_only: bool = True,
    ) -> int:
        keys = self.relation_keys(
            relation_name=relation_name,
            object_1=object_1,
            object_2=object_2,
            active_only=active_only,
        )
        if not keys:
            return 0
        return (
            1
            if any(self.state.current_snapshot.get(key, 0) == 1 for key in keys)
            else 0
        )

    def positive_relation_matches(
        self, relation_name: str, object_1: Optional[str]
    ) -> List[ParsedRelationKey]:
        result = []
        for key in self.relation_keys(
            relation_name=relation_name, object_1=object_1, active_only=True
        ):
            if self.state.current_snapshot.get(key, 0) == 1:
                result.append(self.state.recognized_relations[key])
        return result

    def transition_events(
        self,
        relation_name: str,
        from_value: Optional[int] = None,
        to_value: Optional[int] = None,
        object_filter_group: Optional[str] = None,
        active_only: bool = True,
    ) -> List[TransitionEvent]:
        result = []
        active_objects = set(self.get_all_active_objects())

        for key, events in self.state.transitions_by_key.items():
            parsed = self.state.recognized_relations.get(key)
            if not parsed or parsed.relation_name != relation_name:
                continue

            if active_only:
                if parsed.object_1 and parsed.object_1 not in active_objects:
                    continue
                if self._requires_active_second_object(parsed):
                    if parsed.object_2 not in active_objects:
                        continue

            if object_filter_group:
                if parsed.object_1 not in self.state.spawned_objects.get(
                    object_filter_group, set()
                ):
                    continue

            for event in events:
                if from_value is not None and event.from_value != from_value:
                    continue
                if to_value is not None and event.to_value != to_value:
                    continue
                result.append(event)

        return result


# ============================================================
# GOAL EVALUATOR
# ============================================================


class GoalEvaluator:
    def __init__(self, task_model: TaskModel, tracker: StateTracker):
        self.task = task_model
        self.tracker = tracker

    def evaluate_one(self, goal: GoalSpec) -> GoalEvaluation:
        objects_to_check = self.tracker.get_active_objects(goal.for_each)

        satisfied_objects = []
        unsatisfied_objects = []

        for obj_name in objects_to_check:
            values = [
                self.tracker.aggregated_relation_value(
                    relation_name=rel_name,
                    object_1=obj_name,
                    active_only=True,
                )
                for rel_name in goal.must_have
            ]
            if all(v == 1 for v in values):
                satisfied_objects.append(obj_name)
            else:
                unsatisfied_objects.append(obj_name)

        relation_targets = {}
        for obj_name in satisfied_objects:
            targets_all = []
            for rel_name in goal.must_have:
                positives = self.tracker.positive_relation_matches(rel_name, obj_name)
                targets_all.extend(
                    parsed.object_2
                    for parsed in positives
                    if parsed.object_2 is not None
                )
            relation_targets[obj_name] = sorted(set(targets_all))

        details = {
            "group": goal.for_each,
            "relation": list(goal.must_have),  # now a list, not a string
            "total_objects": len(objects_to_check),
            "satisfied_objects_count": len(satisfied_objects),
            "unsatisfied_objects_count": len(unsatisfied_objects),
            "satisfied_objects": satisfied_objects,
            "unsatisfied_objects": unsatisfied_objects,
            "relation_targets": relation_targets,
            "match_mode": goal.match_mode,
            "min_satisfied": goal.min_satisfied,
        }

        if goal.match_mode == "any":
            status = (
                "satisfied"
                if len(satisfied_objects) >= goal.min_satisfied
                else "failed"
            )
        else:
            status = (
                "satisfied"
                if len(objects_to_check) > 0 and len(unsatisfied_objects) == 0
                else "failed"
            )

        return GoalEvaluation(
            status=status,
            required=goal.required,
            description=goal.description,
            details=details,
        )

    def evaluate_all(self) -> Dict[str, GoalEvaluation]:
        return {
            goal_name: self.evaluate_one(goal_spec)
            for goal_name, goal_spec in self.task.goals.items()
        }


# ============================================================
# CONSTRAINT EVALUATOR
# ============================================================


class ConstraintEvaluator:
    def __init__(self, task_model: TaskModel, tracker: StateTracker):
        self.task = task_model
        self.tracker = tracker

    def _before_condition_matches(
        self, object_name: Optional[str], before: Optional[Tuple[str, int]]
    ) -> bool:
        if before is None:
            return True
        if object_name is None:
            return False

        relation_name, required_value = before
        current_value = self.tracker.aggregated_relation_value(
            relation_name=relation_name,
            object_1=object_name,
            active_only=True,
        )
        return current_value == required_value

    def evaluate_one(
        self, constraint: ConstraintSpec, success_limit: int
    ) -> ConstraintEvaluation:
        if constraint.limit_transitions is not None:
            relation_name, (from_value, to_value) = constraint.limit_transitions
            max_count = int(
                constraint.max_count if constraint.max_count is not None else 0
            )

            events = self.tracker.transition_events(
                relation_name=relation_name,
                from_value=from_value,
                to_value=to_value,
                object_filter_group=constraint.if_object_type,
                active_only=True,
            )

            counts_by_object = defaultdict(int)
            for event in events:
                if event.object_1 is not None:
                    counts_by_object[event.object_1] += 1

            violation_details = []

            if constraint.per_object:
                relevant_objects = (
                    self.tracker.get_active_objects(constraint.if_object_type)
                    if constraint.if_object_type
                    else sorted(set(counts_by_object.keys()))
                )

                violations = 0
                for obj_name in relevant_objects:
                    cnt = counts_by_object.get(obj_name, 0)
                    if cnt > max_count:
                        violations += 1
                        violation_details.append(
                            {
                                "object": obj_name,
                                "count": cnt,
                                "limit": max_count,
                                "relation": relation_name,
                                "transition": [from_value, to_value],
                            }
                        )

                details = dict(sorted(counts_by_object.items()))
            else:
                total_count = len(events)
                violations = 1 if total_count > max_count else 0
                if violations:
                    violation_details.append(
                        {
                            "count": total_count,
                            "limit": max_count,
                            "relation": relation_name,
                            "transition": [from_value, to_value],
                        }
                    )
                details = {
                    "count": total_count,
                    "relation": relation_name,
                    "transition": [from_value, to_value],
                }

            return ConstraintEvaluation(
                violations=violations,
                limit=success_limit,
                passed=violations <= success_limit,
                description=constraint.description,
                details=details,
                violation_details=violation_details,
            )

        if constraint.forbid_transition is not None:
            relation_name, (from_value, to_value) = constraint.forbid_transition

            events = self.tracker.transition_events(
                relation_name=relation_name,
                from_value=from_value,
                to_value=to_value,
                object_filter_group=constraint.if_object_type,
                active_only=True,
            )

            filtered = [
                event
                for event in events
                if self._before_condition_matches(event.object_1, constraint.before)
            ]

            if constraint.per_object:
                by_object = defaultdict(int)
                for event in filtered:
                    if event.object_1 is not None:
                        by_object[event.object_1] += 1
                violations = len([obj for obj, cnt in by_object.items() if cnt > 0])
            else:
                violations = len(filtered)

            details = []
            for event in filtered:
                item = {
                    "time": event.timestamp,
                    "relation": relation_name,
                    "transition": [from_value, to_value],
                }
                if event.object_1 is not None:
                    item["object"] = event.object_1
                if event.object_2 is not None:
                    item["object_2"] = event.object_2
                details.append(item)

            return ConstraintEvaluation(
                violations=violations,
                limit=success_limit,
                passed=violations <= success_limit,
                description=constraint.description,
                details=details,
                violation_details=list(details),
            )

        if constraint.forbid_value is not None:
            relation_name, forbidden_value = constraint.forbid_value
            active_objects = set(self.tracker.get_all_active_objects())

            matched_entries = []

            for key, value in self.tracker.state.current_snapshot.items():
                parsed = self.tracker.state.recognized_relations.get(key)
                if not parsed or parsed.relation_name != relation_name:
                    continue

                if parsed.object_1 and parsed.object_1 not in active_objects:
                    continue
                if self.tracker._requires_active_second_object(parsed):
                    if parsed.object_2 not in active_objects:
                        continue

                if constraint.if_object_type:
                    if parsed.object_1 not in self.tracker.state.spawned_objects.get(
                        constraint.if_object_type, set()
                    ):
                        continue

                if int(value) == forbidden_value:
                    entry = {
                        "relation": relation_name,
                        "value": forbidden_value,
                    }
                    if parsed.object_1 is not None:
                        entry["object"] = parsed.object_1
                    if parsed.object_2 is not None:
                        entry["object_2"] = parsed.object_2
                    matched_entries.append(entry)

            if constraint.per_object:
                violations = len(
                    set(item["object"] for item in matched_entries if "object" in item)
                )
            else:
                violations = len(matched_entries)

            return ConstraintEvaluation(
                violations=violations,
                limit=success_limit,
                passed=violations <= success_limit,
                description=constraint.description,
                details=matched_entries,
                violation_details=list(matched_entries),
            )

        return ConstraintEvaluation(
            violations=0,
            limit=success_limit,
            passed=True,
            description=constraint.description,
            details=[],
            violation_details=[],
        )

    def evaluate_all(self) -> Dict[str, ConstraintEvaluation]:
        result = {}
        for name, spec in self.task.constraints.items():
            success_limit = int(self.task.success.max_violations.get(name, 0))
            result[name] = self.evaluate_one(spec, success_limit)
        return result


# ============================================================
# HISTORY BUILDER
# ============================================================


class HistoryBuilder:
    def __init__(
        self,
        task_model: TaskModel,
        tracker: StateTracker,
        goal_results: Dict[str, GoalEvaluation],
        constraint_results: Dict[str, ConstraintEvaluation],
    ):
        self.task = task_model
        self.tracker = tracker
        self.goal_results = goal_results
        self.constraint_results = constraint_results

    def build_timeline(self) -> List[Dict[str, Any]]:
        events = []
        for key, key_events in self.tracker.state.transitions_by_key.items():
            for ev in key_events:
                events.append(
                    {
                        "time": round(ev.timestamp, 6),
                        "message_key": ev.message_key,
                        "relation": ev.relation_name,
                        "object_1": ev.object_1,
                        "object_2": ev.object_2,
                        "transition": [ev.from_value, ev.to_value],
                    }
                )

        events.sort(key=lambda x: (x["time"], x["message_key"]))
        return events

    def build_final_relation_states(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        active_objects = set(self.tracker.get_all_active_objects())

        for key, parsed in sorted(self.tracker.state.recognized_relations.items()):
            if parsed.object_1 and parsed.object_1 not in active_objects:
                continue
            if self.tracker._requires_active_second_object(parsed):
                if parsed.object_2 not in active_objects:
                    continue

            result[key] = {
                "relation": parsed.relation_name,
                "object_1": parsed.object_1,
                "object_2": parsed.object_2,
                "value": self.tracker.state.current_snapshot.get(key, 0),
            }

        return result

    def build_goal_diagnostics(self) -> Dict[str, Any]:
        diagnostics = {}

        for goal_name, result in self.goal_results.items():
            details = result.details
            unsatisfied_objects = details.get("unsatisfied_objects", [])

            if result.status == "satisfied":
                reason = "goal conditions satisfied"
            elif details.get("match_mode") == "any":
                reason = "fewer objects satisfy required relations than min_satisfied"
            else:
                reason = "some active objects do not satisfy required relations"

            diagnostics[goal_name] = {
                "status": result.status,
                "description": result.description,
                "group": details.get("group"),
                "reason": reason,
                "relation": details.get("relation"),
                "match_mode": details.get("match_mode"),
                "min_satisfied": details.get("min_satisfied"),
                "satisfied_objects_count": details.get("satisfied_objects_count"),
                "unsatisfied_objects": unsatisfied_objects,
                "unsatisfied_explanations": [],
            }

            relation_name = details.get("relation")
            for obj_name in unsatisfied_objects:
                diagnostics[goal_name]["unsatisfied_explanations"].append(
                    {
                        "object": obj_name,
                        "final_relation_values": {
                            rel_name: self.tracker.aggregated_relation_value(
                                relation_name=rel_name,
                                object_1=obj_name,
                                active_only=True,
                            )
                            for rel_name in relation_name
                        },
                        "matching_relation_keys": {
                            rel_name: self.tracker.relation_keys(
                                relation_name=rel_name,
                                object_1=obj_name,
                                active_only=True,
                            )
                            for rel_name in relation_name
                        },
                    }
                )

        return diagnostics

    def build_constraint_diagnostics(self) -> Dict[str, Any]:
        diagnostics = {}

        for const_name, result in self.constraint_results.items():
            diagnostics[const_name] = {
                "passed": result.passed,
                "violations": result.violations,
                "limit": result.limit,
                "description": result.description,
                "reason": (
                    "violations within allowed limit"
                    if result.passed
                    else "violations exceed allowed limit"
                ),
                "violation_details": result.violation_details,
            }

        return diagnostics

    def build(self) -> Dict[str, Any]:
        return {
            "timeline": self.build_timeline(),
            "goal_diagnostics": self.build_goal_diagnostics(),
            "constraint_diagnostics": self.build_constraint_diagnostics(),
            "final_relation_states": self.build_final_relation_states(),
        }


# ============================================================
# METRICS BUILDER
# ============================================================


class SemanticMetricsBuilder:
    def __init__(
        self,
        task_model: TaskModel,
        goal_results: Dict[str, GoalEvaluation],
        constraint_results: Dict[str, ConstraintEvaluation],
        tracker: StateTracker,
    ):
        self.task = task_model
        self.goal_results = goal_results
        self.constraint_results = constraint_results
        self.tracker = tracker

    @staticmethod
    def _metric_key(name: str) -> str:
        name = name.strip().lower()
        name = re.sub(r"[^a-zA-Z0-9а-яА-Я_]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name

    def build(self) -> Dict[str, Any]:
        metrics = {}

        total_goals = len(self.task.goals)
        satisfied_goals = sum(
            1 for g in self.goal_results.values() if g.status == "satisfied"
        )
        metrics["goal_completion_rate"] = (
            round(satisfied_goals / total_goals, 4) if total_goals else 1.0
        )

        required_goals = list(self.task.success.required_goals)
        required_satisfied = sum(
            1
            for g in required_goals
            if self.goal_results.get(g) and self.goal_results[g].status == "satisfied"
        )
        metrics["required_goal_completion_rate"] = (
            round(required_satisfied / len(required_goals), 4)
            if required_goals
            else 1.0
        )

        total_constraints = len(self.task.constraints)
        passed_constraints = sum(
            1 for c in self.constraint_results.values() if c.passed
        )
        metrics["constraint_pass_rate"] = (
            round(passed_constraints / total_constraints, 4)
            if total_constraints
            else 1.0
        )

        required_constraints = list(self.task.success.max_violations.keys())
        required_constraints_passed = sum(
            1
            for c in required_constraints
            if self.constraint_results.get(c) and self.constraint_results[c].passed
        )
        metrics["required_constraint_pass_rate"] = (
            round(required_constraints_passed / len(required_constraints), 4)
            if required_constraints
            else 1.0
        )

        for goal_name, goal_res in self.goal_results.items():
            k = self._metric_key(goal_name)
            total = goal_res.details.get("total_objects", 0)
            sat = goal_res.details.get("satisfied_objects_count", 0)
            metrics[f"{k}_completion_rate"] = round(sat / total, 4) if total else 1.0
            metrics[f"{k}_satisfied_objects"] = sat
            metrics[f"{k}_total_objects"] = total

        for const_name, const_res in self.constraint_results.items():
            k = self._metric_key(const_name)
            metrics[f"{k}_violations"] = const_res.violations
            metrics[f"{k}_limit"] = const_res.limit
            metrics[f"{k}_passed"] = const_res.passed

        metrics["active_objects_count"] = len(self.tracker.get_all_active_objects())
        metrics["recognized_relation_keys"] = len(
            self.tracker.state.recognized_relations
        )
        metrics["observed_transitions"] = sum(
            len(v) for v in self.tracker.state.transitions_by_key.values()
        )
        metrics["processed_messages"] = self.tracker.state.processed_messages

        return metrics


# ============================================================
# REPORT BUILDER
# ============================================================


class ReportBuilder:
    def __init__(
        self,
        task_model: TaskModel,
        tracker: StateTracker,
        goal_results: Dict[str, GoalEvaluation],
        constraint_results: Dict[str, ConstraintEvaluation],
    ):
        self.task = task_model
        self.tracker = tracker
        self.goal_results = goal_results
        self.constraint_results = constraint_results

    def build(
        self, episode_id: str, duration_seconds: float, rosbag_name: str = ""
    ) -> Dict[str, Any]:
        active_objects = self.tracker.get_all_active_objects()

        required_goal_names = list(self.task.success.required_goals)
        required_constraint_names = list(self.task.success.max_violations.keys())

        goals_ok = (
            all(self.goal_results[g].status == "satisfied" for g in required_goal_names)
            if required_goal_names
            else True
        )

        constraints_ok = (
            all(self.constraint_results[c].passed for c in required_constraint_names)
            if required_constraint_names
            else True
        )

        success = goals_ok and constraints_ok

        metrics = SemanticMetricsBuilder(
            task_model=self.task,
            goal_results=self.goal_results,
            constraint_results=self.constraint_results,
            tracker=self.tracker,
        ).build()

        history = HistoryBuilder(
            task_model=self.task,
            tracker=self.tracker,
            goal_results=self.goal_results,
            constraint_results=self.constraint_results,
        ).build()

        timeout_rel_names = [
            n for n, r in self.task.relations.items() if r.type == "timeout"
        ]
        timeout_hit = any(
            len(
                self.tracker.transition_events(
                    relation_name=n, to_value=1, active_only=False
                )
            )
            > 0
            for n in timeout_rel_names
        )

        scoring = {
            goal_name: int(goal_result.status == "satisfied")
            for goal_name, goal_result in self.goal_results.items()
            if self.task.raw.get("goals", {})
            .get(goal_name, {})
            .get("include_in_scoring", True)
        }

        report = {
            "task_name": self.task.task.name,
            "task_description": self.task.task.description,
            "episode_id": episode_id,
            "rosbag_name": rosbag_name,
            "duration_seconds": duration_seconds,
            "success": success,
            "scoring": scoring,
            "episode_timeout": {
                "hit": timeout_hit,
                "relations": timeout_rel_names,
            },
            "objects": {
                "total_spawned": len(active_objects),
                "by_group": {
                    group_name: len(sorted(objs))
                    for group_name, objs in sorted(
                        self.tracker.state.spawned_objects.items()
                    )
                },
                "active_objects": active_objects,
            },
            "goals": {
                name: {
                    "status": res.status,
                    "required": res.required,
                    "description": res.description,
                    "details": res.details,
                }
                for name, res in self.goal_results.items()
            },
            "constraints": {
                name: {
                    "violations": res.violations,
                    "limit": res.limit,
                    "passed": res.passed,
                    "description": res.description,
                    "details": res.details,
                }
                for name, res in self.constraint_results.items()
            },
            "metrics": metrics,
            "history": history,
        }

        violations_summary = []
        for const_name, res in self.constraint_results.items():
            for detail in res.violation_details:
                item = {
                    "constraint": const_name,
                    "description": res.description,
                }
                item.update(detail)
                violations_summary.append(item)

        if violations_summary:
            report["violations_summary"] = violations_summary

        return report


# ============================================================
# REPORT PRINTER
# ============================================================


class ConsoleReportPrinter:
    def print(
        self, report: Dict[str, Any], show_history: bool = True, history_limit: int = 40
    ):
        task = report["task_name"]
        episode_id = report["episode_id"]
        duration = report["duration_seconds"]
        success = report["success"]

        print(f"\n{TermColor.BOLD}{'=' * 80}{TermColor.RESET}")
        print(
            f"{TermColor.BOLD}BENCHMARK: {task} | EPISODE: {episode_id}{TermColor.RESET}"
        )
        print(f"{TermColor.CYAN}TIME: {duration:.1f} sec{TermColor.RESET}")
        print(f"{TermColor.BOLD}{'=' * 80}{TermColor.RESET}\n")

        objects = report["objects"]
        by_group = ", ".join(f"{k}: {v}" for k, v in objects["by_group"].items())
        active = (
            ", ".join(objects["active_objects"]) if objects["active_objects"] else "—"
        )
        print(
            f"{TermColor.BOLD}📦 OBJECTS:{TermColor.RESET} {objects['total_spawned']} ({by_group})"
        )
        print(f"   Active: {active}\n")

        print(f"{TermColor.BOLD}🎯 GOALS:{TermColor.RESET}")
        for goal_name, goal in report["goals"].items():
            icon = (
                f"{TermColor.GREEN}✓{TermColor.RESET}"
                if goal["status"] == "satisfied"
                else f"{TermColor.RED}❌{TermColor.RESET}"
            )
            d = goal["details"]
            print(
                f"   {icon} {goal_name}: {d.get('satisfied_objects_count', 0)}/{d.get('total_objects', 0)}"
            )
            if goal.get("description"):
                print(f"      {goal['description']}")
            if d.get("unsatisfied_objects"):
                print(f"      Unsatisfied: {', '.join(d['unsatisfied_objects'])}")
        print()

        print(f"{TermColor.BOLD}⚠️  CONSTRAINTS:{TermColor.RESET}")
        for const_name, const in report["constraints"].items():
            icon = (
                f"{TermColor.GREEN}✓{TermColor.RESET}"
                if const["passed"]
                else f"{TermColor.RED}❌{TermColor.RESET}"
            )
            print(
                f"   {icon} {const_name}: {const['violations']} (limit: {const['limit']})"
            )
            if const.get("description"):
                print(f"      {const['description']}")
        print()

        print(f"{TermColor.BOLD}📈 METRICS:{TermColor.RESET}")
        for key, value in sorted(report["metrics"].items()):
            print(f"   {key}: {value}")
        print()

        if not success:
            print(f"{TermColor.BOLD}🧭 WHY FAILED:{TermColor.RESET}")
            for goal_name, gd in report["history"]["goal_diagnostics"].items():
                if gd["status"] != "satisfied":
                    print(f"   • Goal {goal_name}: {gd['reason']}")
                    if gd["unsatisfied_objects"]:
                        print(f"     objects: {', '.join(gd['unsatisfied_objects'])}")

            for const_name, cd in report["history"]["constraint_diagnostics"].items():
                if not cd["passed"]:
                    print(f"   • Constraint {const_name}: {cd['reason']}")
            print()

        if show_history:
            timeline = report.get("history", {}).get("timeline", [])
            print(f"{TermColor.BOLD}🕘 HISTORY (state changes only):{TermColor.RESET}")
            if not timeline:
                print("   —")
            else:
                items = timeline[:history_limit]
                for item in items:
                    obj2 = (
                        f", {item['object_2']}"
                        if item.get("object_2") is not None
                        else ""
                    )
                    print(
                        f"   [{item['time']:.3f}] {item['relation']} "
                        f"({item.get('object_1')}{obj2}) "
                        f"{item['transition'][0]}→{item['transition'][1]}"
                    )
                if len(timeline) > history_limit:
                    print(f"   ... {len(timeline) - history_limit} more events")
            print()

        if success:
            print(f"{TermColor.GREEN}✅ STATUS: TASK COMPLETED{TermColor.RESET}")
        else:
            print(f"{TermColor.RED}❌ STATUS: TASK NOT COMPLETED{TermColor.RESET}")

        print(f"{TermColor.BOLD}{'=' * 80}{TermColor.RESET}")


# ============================================================
# PUBLIC ANALYZER
# ============================================================


class BenchmarkAnalyzer:
    def __init__(
        self,
        task_source: Union[str, Dict[str, Any]],
        robot_body_names: Optional[Set[str]] = None,
    ):

        if isinstance(task_source, str):
            with open(task_source, "r", encoding="utf-8") as f:
                raw_config = json.load(f)
        else:
            raw_config = task_source

        self.task_model = TaskModel(raw_config)
        self.matcher = RelationMatcher(self.task_model)
        self.tracker = StateTracker(
            self.task_model,
            self.matcher,
            robot_body_names=robot_body_names,
        )
        self.printer = ConsoleReportPrinter()

        self._debug_mode = False
        self._finalized = False
        self._last_report: Optional[Dict[str, Any]] = None

    def set_debug_mode(self, enabled: bool = True):
        self._debug_mode = enabled
        print(
            f"{TermColor.CYAN}🔧 Debug mode: {'ON' if enabled else 'OFF'}{TermColor.RESET}"
        )

    def update(self, json_message: str, timestamp: float):
        if self._finalized:
            raise RuntimeError("Cannot update finalized analyzer")

        try:
            raw = json.loads(json_message)
        except json.JSONDecodeError as e:
            print(f"{TermColor.RED}[Parse Error] {e}{TermColor.RESET}")
            return

        if not isinstance(raw, dict):
            return

        snapshot = {}
        for key, value in raw.items():
            if isinstance(value, (int, float)):
                snapshot[key] = int(value)

        self.tracker.update(snapshot, timestamp)

        if self._debug_mode:
            self.debug_state(show_snapshot=True, show_transitions=False)

    def is_success_now(self) -> bool:
        goal_results = GoalEvaluator(self.task_model, self.tracker).evaluate_all()
        constraint_results = ConstraintEvaluator(
            self.task_model, self.tracker
        ).evaluate_all()

        required_goal_names = list(self.task_model.success.required_goals)
        required_constraint_names = list(self.task_model.success.max_violations.keys())

        goals_ok = (
            all(goal_results[g].status == "satisfied" for g in required_goal_names)
            if required_goal_names
            else True
        )
        constraints_ok = (
            all(constraint_results[c].passed for c in required_constraint_names)
            if required_constraint_names
            else True
        )
        return goals_ok and constraints_ok

    def finalize(
        self, episode_id: str, duration_seconds: float, rosbag_name: str = ""
    ) -> Dict[str, Any]:
        if self._finalized:
            raise RuntimeError("Analyzer already finalized")

        self._finalized = True

        goal_results = GoalEvaluator(self.task_model, self.tracker).evaluate_all()
        constraint_results = ConstraintEvaluator(
            self.task_model, self.tracker
        ).evaluate_all()

        report = ReportBuilder(
            task_model=self.task_model,
            tracker=self.tracker,
            goal_results=goal_results,
            constraint_results=constraint_results,
        ).build(
            episode_id=episode_id,
            duration_seconds=duration_seconds,
            rosbag_name=rosbag_name,
        )

        self._last_report = report
        return report

    def print_report(
        self,
        report: Optional[Dict[str, Any]] = None,
        show_history: bool = True,
        history_limit: int = 40,
    ):
        report = report or self._last_report
        if not report:
            print(f"{TermColor.RED}No report to print{TermColor.RESET}")
            return
        self.printer.print(
            report, show_history=show_history, history_limit=history_limit
        )

    def save_report(self, report: Dict[str, Any], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def debug_state(self, show_transitions: bool = True, show_snapshot: bool = False):
        state = self.tracker.state

        print(f"\n{TermColor.MAGENTA}{TermColor.BOLD}{'=' * 84}{TermColor.RESET}")
        print(f"{TermColor.MAGENTA}{TermColor.BOLD}DEBUG STATE{TermColor.RESET}")
        print(f"{TermColor.MAGENTA}{TermColor.BOLD}{'=' * 84}{TermColor.RESET}\n")

        print(
            f"{TermColor.BOLD}Processed messages:{TermColor.RESET} {state.processed_messages}"
        )
        print(
            f"{TermColor.BOLD}Recognized relation keys:{TermColor.RESET} {len(state.recognized_relations)}"
        )
        print(
            f"{TermColor.BOLD}Observed transitions:{TermColor.RESET} {sum(len(v) for v in state.transitions_by_key.values())}"
        )
        print()

        print(f"{TermColor.BOLD}Active objects by group:{TermColor.RESET}")
        for group_name, objs in sorted(state.spawned_objects.items()):
            print(f"  - {group_name}: {sorted(objs)}")
        print()

        print(f"{TermColor.BOLD}Recognized relations:{TermColor.RESET}")
        for key, parsed in sorted(state.recognized_relations.items()):
            print(
                f"  - {key} -> relation={parsed.relation_name}, "
                f"object_1={parsed.object_1}, object_2={parsed.object_2}"
            )
        print()

        if show_snapshot:
            print(f"{TermColor.BOLD}Current snapshot:{TermColor.RESET}")
            for key, value in sorted(state.current_snapshot.items()):
                print(f"  - {key}: {value}")
            print()

        if show_transitions:
            print(f"{TermColor.BOLD}Transitions by key:{TermColor.RESET}")
            for key, events in sorted(state.transitions_by_key.items()):
                print(f"  - {key}")
                for event in events:
                    print(
                        f"      {event.from_value}->{event.to_value} @ {event.timestamp:.3f}s "
                        f"(relation={event.relation_name}, object_1={event.object_1}, object_2={event.object_2})"
                    )
            print()

        print(f"{TermColor.MAGENTA}{TermColor.BOLD}{'=' * 84}{TermColor.RESET}\n")


if __name__ == "__main__":
    pass
