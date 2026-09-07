#!/usr/bin/env python3
"""Build assets/structure.json from assets/scenarios.json.

Expected layout:
assets/
├── scenarios.json
├── policies/
│   └── policy.<...>.json
└── tasks/
    └── task.<...>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSON value must be an object: {path}")
    return data


def referenced_name(value: Any, field: str, scenario_name: str) -> str:
    """Extract the filename stem from [name, enabled] or directly from a string."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and isinstance(value[0], str):
        return value[0]
    raise ValueError(
        f"Scenario {scenario_name!r}: {field!r} must be a string or a non-empty "
        f"list whose first item is a string; got {value!r}"
    )


def json_path(directory: Path, name: str) -> Path:
    return directory / (name if name.endswith(".json") else f"{name}.json")


def resolve_policy_object(
    objects: dict[str, Any],
    object_ref: Any,
    policy_path: Path,
    command_id: str,
    field_name: str,
) -> str:
    if not isinstance(object_ref, str) or not object_ref:
        raise ValueError(
            f"{policy_path}, command {command_id!r}: relation field "
            f"{field_name!r} must be a non-empty string"
        )

    resolved_object = objects.get(object_ref, object_ref)

    if not isinstance(resolved_object, str) or not resolved_object:
        raise ValueError(
            f"{policy_path}, command {command_id!r}: object alias "
            f"{object_ref!r} from field {field_name!r} must resolve "
            "to a non-empty string"
        )

    return resolved_object


def extract_annotations(
    policy: dict[str, Any],
    policy_path: Path,
) -> dict[str, dict[str, Any]]:
    commands = policy.get("commands")
    if not isinstance(commands, dict):
        raise ValueError(f"{policy_path}: field 'commands' must be an object")

    objects = policy.get("objects", {})
    if not isinstance(objects, dict):
        raise ValueError(f"{policy_path}: field 'objects' must be an object")

    annotations: dict[str, dict[str, Any]] = {}
    current_annotation_id: str | None = None

    for command_id, command in commands.items():
        if not isinstance(command, dict):
            raise ValueError(
                f"{policy_path}, command {command_id!r}: command must be an object"
            )

        command_type = command.get("type")

        if command_type == "annotation":
            task = command.get("task")
            subtask = command.get("subtask")

            if not isinstance(task, str) or not isinstance(subtask, str):
                raise ValueError(
                    f"{policy_path}, command {command_id!r}: annotation requires "
                    "string fields 'task' and 'subtask'"
                )

            current_annotation_id = str(len(annotations))
            annotations[current_annotation_id] = {
                "task": task,
                "subtask": subtask,
                "relations": [],
            }
            continue

        if command_type != "relation" or current_annotation_id is None:
            continue

        if command.get("required_for_transition", True) is False:
            continue

        relation_name = command.get("name")
        if not isinstance(relation_name, str) or not relation_name:
            raise ValueError(
                f"{policy_path}, command {command_id!r}: relation requires "
                "a non-empty string field 'name'"
            )

        relation: dict[str, str] = {
            "name": relation_name,
        }

        for field_name in ("object_a", "object_b"):
            if field_name not in command:
                continue

            relation[field_name] = resolve_policy_object(
                objects=objects,
                object_ref=command[field_name],
                policy_path=policy_path,
                command_id=command_id,
                field_name=field_name,
            )

        relation_key = (
            relation["name"],
            relation.get("object_a"),
            relation.get("object_b"),
        )

        current_relations = annotations[current_annotation_id]["relations"]

        if not any(
            (
                existing_relation["name"],
                existing_relation.get("object_a"),
                existing_relation.get("object_b"),
            ) == relation_key
            for existing_relation in current_relations
        ):
            current_relations.append(relation)

    return annotations


def build_structure(assets_dir: Path, output_path: Path) -> dict[str, Any]:
    scenarios_path = assets_dir / "scenarios.json"
    scenarios = load_json(scenarios_path)
    policies_dir = assets_dir / "policies"
    tasks_dir = assets_dir / "tasks"

    result: dict[str, Any] = {}
    errors: list[str] = []

    for scenario_name, scenario in scenarios.items():
        try:
            if not isinstance(scenario, dict):
                raise ValueError(f"Scenario {scenario_name!r} must be an object")

            scene = scenario.get("scene")
            if not isinstance(scene, str):
                raise ValueError(f"Scenario {scenario_name!r}: 'scene' must be a string")

            benchmark_name = referenced_name(
                scenario.get("benchmark"), "benchmark", scenario_name
            )
            policy_name = referenced_name(scenario.get("policy"), "policy", scenario_name)

            task_path = json_path(tasks_dir, benchmark_name)
            policy_path = json_path(policies_dir, policy_name)
            task_json = load_json(task_path)
            policy_json = load_json(policy_path)

            task = task_json.get("task")
            if not isinstance(task, dict):
                raise ValueError(f"{task_path}: field 'task' must be an object")
            if not isinstance(task.get("name"), str) or not isinstance(task.get("description"), str):
                raise ValueError(
                    f"{task_path}: 'task' must contain string fields 'name' and 'description'"
                )

            result[scenario_name] = {
                "scene": scene,
                "benchmark": benchmark_name,
                "policy": policy_name,
                "task": {
                    "name": task["name"],
                    "description": task["description"],
                },
                "annotations": extract_annotations(policy_json, policy_path),
            }
        except (ValueError, FileNotFoundError) as exc:
            errors.append(str(exc))

    if errors:
        raise RuntimeError("structure.json was not created:\n- " + "\n- ".join(errors))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets-dir", type=Path, default=Path("assets"),
        help="Path to the assets directory (default: assets)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path (default: <assets-dir>/structure.json)",
    )
    args = parser.parse_args()

    output_path = args.output or args.assets_dir / "structure.json"
    try:
        result = build_structure(args.assets_dir, output_path)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Created {output_path} for {len(result)} scenarios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
