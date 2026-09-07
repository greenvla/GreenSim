# episode_relations_recorder.py
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import json, time, os
from typing import Any, Dict, List, Optional
def get_ros_timestamp(sim_time: float, use_sim_time: bool): return time.time() # from simple_example.envs.managers.ros.isaac_clock_publisher import get_ros_timestamp


class EpisodeRecorder:
    def __init__(
        self,
        output_dir,
        topic="/benchmark/rewards",
        task_name=None,
        target_env_id=0,
        use_sim_time: bool = False,
    ):
        self.output_dir = self.make_run_dir(
            base=output_dir, task_name=task_name or "no_task"
        )
        self.episodes_dir = self.output_dir / "episodes"
        self.reports_dir = self.output_dir / "reports"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.topic = topic
        self.task_name = task_name
        self.target_env_id = int(target_env_id)
        self.use_sim_time = bool(use_sim_time)

        self._active = False
        self._episode_id = None
        self._rosbag_path = ""
        self._start_ts = None
        self._start_sim_ts = None
        self._last_sim_ts = None
        self._messages: List[Dict[str, Any]] = []
        self._subtask_history: List[Dict[str, Any]] = []

    def make_run_dir(self, base="benchmark_log", task_name="unknown"):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(base) / f"{task_name}_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _to_ros_timestamp(self, sim_time: float) -> float:
        """Raw Isaac sim.current_time -> timestamp on the ROS /clock scale."""
        return get_ros_timestamp(
            sim_time=float(sim_time),
            use_sim_time=self.use_sim_time,
        )

    def get_output_dir(self):
        return self.output_dir

    def on_reset_start_new_episode(
        self,
        episode_id: str,
        rosbag_path: str = "",
        sim_time: float | None = None,
    ) -> None:
        """Open an episode.

        sim_time — raw self.sim.current_time.
        start_timestamp is recorded in the ROS /clock epoch time.
        """
        if sim_time is None:
            raise ValueError("sim_time is required to start an EpisodeRecorder episode")

        raw_sim_time = float(sim_time)

        self._active = True
        self._episode_id = str(episode_id)
        self._rosbag_path = rosbag_path

        self._start_sim_ts = raw_sim_time
        self._last_sim_ts = raw_sim_time
        self._start_ts = self._to_ros_timestamp(raw_sim_time)

        self._messages = []
        self._subtask_history: List[Dict[str, Any]] = []

    def record(self, events: List[Dict[str, Any]]):
        if not self._active or not events:
            return (None, None, None)

        payload = {}
        sim_time_for_message = None
        for e in events:
            if int(e.get("env_id", -1)) != self.target_env_id:
                continue
            payload[e["pattern"]] = int(e["value"])
            sim_time_for_message = e.get("sim_time", sim_time_for_message)

        if not payload:
            return (None, None, None)

        if sim_time_for_message is None:
            raise RuntimeError(
                "Benchmark event has no sim_time; cannot create ROS-aligned timestamp"
            )

        raw_sim_time = float(sim_time_for_message)
        self._last_sim_ts = raw_sim_time

        message = {
            "timestamp": self._to_ros_timestamp(raw_sim_time),
            "sim_timestamp": raw_sim_time,
            "payload": json.dumps(payload, ensure_ascii=False),
        }

        self._messages.append(message)
        return (self.task_name, self._episode_id, message)

    def close(self):
        """Optionally call at env shutdown."""
        if self._active:
            self._finalize_current()

    def _finalize_current(
        self,
        env_id: int = 0,
        sim_time: float | None = None,
    ) -> None:
        if self._episode_id is None:
            return

        end_sim_ts = float(sim_time) if sim_time is not None else self._last_sim_ts

        if end_sim_ts is None:
            raise RuntimeError(
                "sim_time is required to finalize an EpisodeRecorder episode"
            )

        start_sim_ts = float(
            self._start_sim_ts if self._start_sim_ts is not None else end_sim_ts
        )

        end_ts = self._to_ros_timestamp(end_sim_ts)
        start_ts = float(
            self._start_ts
            if self._start_ts is not None
            else self._to_ros_timestamp(start_sim_ts)
        )

        duration_sim_seconds = (
            round(end_sim_ts - start_sim_ts, 3)
            if (end_sim_ts is not None and start_sim_ts is not None)
            else None
        )

        result = {
            "task_name": self.task_name,
            "episode_id": self._episode_id,
            "rosbag_path": self._rosbag_path,
            "topic": self.topic,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "duration_seconds": round(end_ts - start_ts, 3),
            "start_sim_timestamp": start_sim_ts,
            "end_sim_timestamp": end_sim_ts,
            "duration_sim_seconds": duration_sim_seconds,
            "message_count": len(self._messages),
            "subtask_history": self._subtask_history,
            "messages": self._messages,
        }

        out = self.episodes_dir / f"{self._episode_id}.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())  # for durability on disk

        self._active = False
        self._episode_id = None
        self._rosbag_path = ""
        self._start_ts = None
        self._start_sim_ts = None
        self._last_sim_ts = None
        self._messages = []
        self._subtask_history = []

    def record_subtask_events(self, events: List[Dict[str, Any]]) -> None:
        if not self._active or not events:
            return

        for event in events:
            if int(event.get("env_id", -1)) != self.target_env_id:
                continue

            sim_time = event.get("sim_time")
            if sim_time is None:
                raise RuntimeError(
                    "Subtask event has no sim_time; cannot create ROS-aligned timestamp"
                )

            raw_sim_time = float(sim_time)
            ros_timestamp = self._to_ros_timestamp(raw_sim_time)
            event_name = event.get("event")
            subtask_index = int(event["subtask_index"])

            if event_name == "started":
                history_item = {
                    "subtask_index": subtask_index,
                    "task": str(event["task"]),
                    "subtask": str(event["subtask"]),
                    "start_timestamp": ros_timestamp,
                    "start_sim_timestamp": raw_sim_time,
                    "end_timestamp": None,
                    "end_sim_timestamp": None,
                    "completed": False,
                }

                if "warning" in event:
                    history_item["warning"] = str(event["warning"])

                self._subtask_history.append(history_item)
                continue

            if event_name == "completed":
                if (
                    not self._subtask_history
                    or self._subtask_history[-1]["subtask_index"] != subtask_index
                ):
                    raise RuntimeError(
                        "Received subtask completion without matching active subtask: "
                        f"{subtask_index}"
                    )

                active_subtask = self._subtask_history[-1]

                if active_subtask["completed"]:
                    raise RuntimeError(f"Subtask {subtask_index} was already completed")

                active_subtask["end_timestamp"] = ros_timestamp
                active_subtask["end_sim_timestamp"] = raw_sim_time
                active_subtask["completed"] = True
                self._last_sim_ts = raw_sim_time
                continue

            raise ValueError(f"Unknown subtask event: {event_name!r}")

    def save_report(self, report: Dict[str, Any]):
        episode_id = report["episode_id"]
        ep_path = self.reports_dir / f"{episode_id}_report.json"
        with ep_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
