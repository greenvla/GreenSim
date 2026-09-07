from __future__ import annotations
import time
from copy import deepcopy
from typing import Any, Optional
def get_ros_timestamp(sim_time: float, use_sim_time: bool): return time.time() # from simple_example.envs.managers.ros.isaac_clock_publisher import get_ros_timestamp


class AnnotationManager:
    """Accumulates scripted-policy action annotations for one episode."""

    def __init__(
        self,
        *,
        annotator_username: str = "scripted_policy",
        use_sim_time: bool = False,
    ) -> None:
        self.annotator_username = annotator_username
        self.use_sim_time = bool(use_sim_time)
        self.reset()

    def reset(self) -> None:
        self._current_task: Optional[str] = None
        self._sequence_of_actions: list[dict[str, Any]] = []
        self._open_action: Optional[dict[str, Any]] = None
        self._next_action_id = 0

    def _to_ros_timestamp(self, raw_sim_time: Any) -> float:
        return get_ros_timestamp(
            sim_time=self._finite_sim_time(raw_sim_time),
            use_sim_time=self.use_sim_time,
        )

    def add_annotation(
        self,
        *,
        command: dict[str, Any],
        sim_time: float,
    ) -> None:
        if command.get("type") != "annotation":
            raise ValueError(
                "AnnotationManager.add_annotation() requires an annotation command"
            )

        timestamp = self._to_ros_timestamp(sim_time)
        subtask = self._required_nonempty_string(command, "subtask")

        task = command.get("task")
        if task is not None:
            if not isinstance(task, str) or not task.strip():
                raise ValueError("Annotation command field 'task' must be a non-empty string")
            self._current_task = task.strip()

        if self._current_task is None:
            raise ValueError(
                "Annotation command has no task and no current_task exists for this episode"
            )

        self._close_open_action(timestamp)
        self._open_action = {
            "action_id": self._next_action_id,
            "action_task": self._current_task,
            "action_name": subtask,
            "start_timestamp": timestamp,
            "end_timestamp": None,
            "error_codes": self._normalize_error_codes(command.get("error_codes", [])),
            "annotator_username": self.annotator_username,
            "annotated_frames": [],
        }
        self._next_action_id += 1

    def finalize(self, *, sim_time: float) -> dict[str, list[dict[str, Any]]]:
        self._close_open_action(self._to_ros_timestamp(sim_time))
        return self.result()

    def result(self) -> dict[str, list[dict[str, Any]]]:
        actions = list(self._sequence_of_actions)
        if self._open_action is not None:
            actions.append(self._open_action)
        return {"sequence_of_actions": deepcopy(actions)}

    def _close_open_action(self, end_timestamp: float) -> None:
        if self._open_action is None:
            return
        self._open_action["end_timestamp"] = end_timestamp
        self._sequence_of_actions.append(self._open_action)
        self._open_action = None

    @staticmethod
    def _required_nonempty_string(command: dict[str, Any], key: str) -> str:
        value = command.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Annotation command field '{key}' must be a non-empty string"
            )
        return value.strip()

    @staticmethod
    def _finite_sim_time(value: Any) -> float:
        try:
            timestamp = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"sim_time must be numeric, got {value!r}") from exc
        if timestamp != timestamp or timestamp in (float("inf"), float("-inf")):
            raise ValueError(f"sim_time must be finite, got {value!r}")
        return timestamp

    def _normalize_error_codes(self, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("Annotation command field 'error_codes' must be a list")

        normalized: list[dict[str, Any]] = []
        for index, error_code in enumerate(value):
            if not isinstance(error_code, dict):
                raise ValueError(f"error_codes[{index}] must be a dict")

            code = error_code.get("code")
            if not isinstance(code, str):
                raise ValueError(f"error_codes[{index}].code must be a string")

            timestamps = error_code.get("timestamp", [])
            if not isinstance(timestamps, list):
                raise ValueError(f"error_codes[{index}].timestamp must be a list")

            normalized.append(
                {
                    "code": code,
                    "timestamp": [self._to_ros_timestamp(t) for t in timestamps],
                }
            )
        return normalized
