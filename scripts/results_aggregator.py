"""Aggregate per-episode benchmark reports into a run-level ``results.json``.

The final ``score`` is the subtask-level integral metric (variant 4): every
subtask is a scoring unit. A completed subtask contributes its speed, an
incomplete subtask contributes zero. The run ``score`` is the raw sum of the
speeds of all completed subtasks (no averaging).

- ``Speed_{i,g} = clamp(1 - t_{i,g} / T_max, 0, 1)``
- ``t_{i,g}`` — sim-time duration of subtask ``g`` in episode ``i``, taken as
  ``end_sim_timestamp - start_sim_timestamp`` from the episode's
  ``subtask_history``.
- ``T_max`` — timeout limit from the task's ``timeout`` relation (``duration_sec``).
- ``score = Σ(completed subtask speeds)``, rounded to 8 decimals.

This module is stdlib-only so it can run outside Isaac Sim.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_optional_feature(cfg: Any) -> Optional[str]:
    """Resolve the ``[name, enabled]`` / ``null`` convention used in scenarios.json."""
    if cfg is None:
        return None
    if not isinstance(cfg, list) or len(cfg) != 2:
        raise ValueError(f"Expected [name, enabled] or null, got: {cfg!r}")
    name, enabled = cfg
    if not enabled:
        return None
    return name


def get_timeout_seconds(task_json_path: str | Path) -> Optional[float]:
    """Return ``duration_sec`` of the first ``timeout`` relation, or None."""
    task = load_json(task_json_path)
    for rel_cfg in task.get("relations", {}).values():
        if rel_cfg.get("type") == "timeout" and "duration_sec" in rel_cfg:
            return float(rel_cfg["duration_sec"])
    return None


def _round(value: float, ndigits: int) -> float:
    return round(float(value), ndigits)


def _load_subtask_counts(structure_path: Optional[str | Path]) -> Dict[str, int]:
    """Map scenario name -> number of subtasks (len of ``annotations``).

    Returns an empty dict if the structure file is missing/unreadable.
    """
    if structure_path is None:
        return {}
    path = Path(structure_path)
    if not path.is_file():
        return {}
    structure = load_json(path)
    result: Dict[str, int] = {}
    for name, cfg in structure.items():
        if isinstance(cfg, dict) and isinstance(cfg.get("annotations"), dict):
            result[name] = len(cfg["annotations"])
    return result


def _episode_metrics(
    reports: List[Dict[str, Any]],
    timeout_sec: Optional[float],
) -> Dict[str, Any]:
    """Compute episode-level metrics (SR, speed, goal scoring) for one scenario."""
    n = len(reports)
    successes = 0
    speed_sum = 0.0

    scoring_achieved: Dict[str, int] = defaultdict(int)
    scoring_runs: Dict[str, int] = defaultdict(int)

    for report in reports:
        success = 1 if report.get("success") else 0
        duration = float(report.get("duration_seconds", 0.0))

        if timeout_sec is not None and timeout_sec > 0:
            speed = 1.0 - duration / timeout_sec
        else:
            speed = 1.0 if success else 0.0
        speed = max(0.0, min(1.0, speed))

        successes += success
        speed_sum += speed

        scoring = report.get("scoring", {})
        if not isinstance(scoring, dict):
            continue

        for target_name, target_status in scoring.items():
            scoring_runs[target_name] += 1
            scoring_achieved[target_name] += int(bool(target_status))

    sr = successes / n if n else 0.0

    scoring_stats = {
        target_name: {
            "achieved": scoring_achieved[target_name],
            "runs": runs,
            "success_rate": (
                f"{scoring_achieved[target_name] / runs * 100:.2f}% "
                f"({scoring_achieved[target_name]}/{runs})"
            ),
        }
        for target_name, runs in sorted(scoring_runs.items())
    }

    goals_score = sum(
        target_stats["achieved"] for target_stats in scoring_stats.values()
    )

    return {
        "episodes": n,
        "success_rate": f"{sr * 100:.2f}% ({successes}/{n})",
        "speed": _round(speed_sum / n, 6) if n else 0.0,
        "scoring": scoring_stats,
        "goals_score": goals_score,
        "_successes": successes,
        "_speed_sum": speed_sum,
        "_scoring_achieved": dict(scoring_achieved),
        "_scoring_runs": dict(scoring_runs),
    }


def _subtask_metrics(
    episodes: List[Dict[str, Any]],
    timeout_sec: Optional[float],
    total_subtasks: Optional[int],
) -> Dict[str, Any]:
    """Compute the subtask-level score for one scenario.

    ``episodes`` are the raw ``episodes/episode_*.json`` files, each carrying a
    ``subtask_history`` list. ``total_subtasks`` is the number of subtasks
    declared in ``assets/structure.json`` (or None when unknown).
    """
    completed = 0
    started = 0
    completed_speed_sum = 0.0

    for ep in episodes:
        history = ep.get("subtask_history", [])
        if not isinstance(history, list):
            continue
        for st in history:
            if not isinstance(st, dict):
                continue
            started += 1
            if not st.get("completed"):
                continue
            start_sim = st.get("start_sim_timestamp")
            end_sim = st.get("end_sim_timestamp")
            if start_sim is None or end_sim is None:
                continue

            t = float(end_sim) - float(start_sim)
            if timeout_sec is not None and timeout_sec > 0:
                speed = 1.0 - t / timeout_sec
            else:
                speed = 1.0
            speed = max(0.0, min(1.0, speed))

            completed += 1
            completed_speed_sum += speed

    if total_subtasks is not None and total_subtasks > 0:
        subtask_total = len(episodes) * total_subtasks
    else:
        subtask_total = started

    return {
        "subtask_completed": completed,
        "subtask_total": subtask_total,
        "score": _round(completed_speed_sum, 8),
        "_completed_speed_sum": completed_speed_sum,
    }


def compute_run_results(
    run_dir: str | Path,
    scenarios: List[Dict[str, str]],
    tasks_dir: str | Path = "assets/tasks",
    structure_path: str | Path = "assets/structure.json",
    benchmark_name: str = "green_challenge",
    run_name: str = "",
) -> Dict[str, Any]:
    """Aggregate all scenarios of a run into a results dict.

    ``scenarios`` is a list of resolved entries::

        {"scenario": "sort_fruits_1_test", "benchmark_task": "sort_fruits_random"}

    Reports are expected under ``<run_dir>/<scenario>/.../reports/`` and raw
    episode files under ``<run_dir>/<scenario>/.../episodes/``.
    """
    run_dir = Path(run_dir)
    tasks_dir = Path(tasks_dir)
    subtask_counts = _load_subtask_counts(structure_path)

    subtasks: Dict[str, Dict[str, Any]] = {}
    total_episodes = 0
    total_successes = 0
    speed_sum = 0.0
    scoring_achieved: Dict[str, int] = defaultdict(int)
    scoring_runs: Dict[str, int] = defaultdict(int)

    subtask_speed_sum = 0.0
    subtask_completed_total = 0
    subtask_total_total = 0

    for entry in scenarios:
        scenario_name = entry["scenario"]
        benchmark_task = entry.get("benchmark_task")

        scenario_dir = run_dir / scenario_name
        reports: List[Dict[str, Any]] = []
        episodes: List[Dict[str, Any]] = []

        if scenario_dir.is_dir():
            report_paths = sorted(scenario_dir.rglob("reports/episode_*_report.json"))
            reports = [load_json(path) for path in report_paths]
            episode_paths = sorted(scenario_dir.rglob("episodes/episode_*.json"))
            episodes = [load_json(path) for path in episode_paths]

        if not reports and not episodes:
            print(f"[aggregator] WARNING: no reports/episodes for scenario '{scenario_name}'")
            continue

        timeout_sec = None
        if benchmark_task:
            task_path = tasks_dir / f"{benchmark_task}.json"
            if task_path.is_file():
                timeout_sec = get_timeout_seconds(task_path)
            else:
                print(f"[aggregator] WARNING: task file not found: {task_path}")

        episode_metrics = _episode_metrics(reports, timeout_sec)
        subtask_metrics = _subtask_metrics(
            episodes, timeout_sec, subtask_counts.get(scenario_name)
        )

        subtasks[scenario_name] = {
            k: v for k, v in episode_metrics.items() if not k.startswith("_")
        }
        subtasks[scenario_name]["score"] = subtask_metrics["score"]
        subtasks[scenario_name]["subtask_completed"] = subtask_metrics["subtask_completed"]
        subtasks[scenario_name]["subtask_total"] = subtask_metrics["subtask_total"]

        total_episodes += episode_metrics["episodes"]
        total_successes += episode_metrics["_successes"]
        speed_sum += episode_metrics["_speed_sum"]
        for target_name, achieved in episode_metrics["_scoring_achieved"].items():
            scoring_achieved[target_name] += achieved
        for target_name, runs in episode_metrics["_scoring_runs"].items():
            scoring_runs[target_name] += runs

        subtask_speed_sum += subtask_metrics["_completed_speed_sum"]
        subtask_completed_total += subtask_metrics["subtask_completed"]
        subtask_total_total += subtask_metrics["subtask_total"]

    if total_episodes == 0:
        success_rate_str = "0.00%"
        overall_speed = 0.0
    else:
        success_rate_str = f"{total_successes / total_episodes * 100:.2f}%"
        overall_speed = _round(speed_sum / total_episodes, 6)

    overall_score = _round(subtask_speed_sum, 8)

    overall_scoring = {
        target_name: {
            "achieved": scoring_achieved[target_name],
            "runs": runs,
            "success_rate": (
                f"{scoring_achieved[target_name] / runs * 100:.2f}% "
                f"({scoring_achieved[target_name]}/{runs})"
            ),
        }
        for target_name, runs in sorted(scoring_runs.items())
    }

    goals_score = sum(
        target_stats["achieved"] for target_stats in overall_scoring.values()
    )

    return {
        "benchmark": benchmark_name,
        "run_name": run_name,
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_episodes": total_episodes,
        "success_rate": success_rate_str,
        "speed": overall_speed,
        "score": overall_score,
        "subtask_completed": subtask_completed_total,
        "subtask_total": subtask_total_total,
        "scoring": overall_scoring,
        "goals_score": goals_score,
        "subtasks": subtasks,
    }


def write_results(results: Dict[str, Any], run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    out_path = run_dir / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        f.write("\n")
    return out_path


def _resolve_scenarios_from_configs(
    runs_config: Dict[str, Any], scenarios_path: str | Path
) -> List[Dict[str, str]]:
    scenarios_cfg = load_json(scenarios_path)
    resolved = []
    for entry in runs_config["scenarios"]:
        scenario_name = entry["scenario"]
        if scenario_name not in scenarios_cfg:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. Available: {sorted(scenarios_cfg)}"
            )
        resolved.append(
            {
                "scenario": scenario_name,
                "benchmark_task": resolve_optional_feature(
                    scenarios_cfg[scenario_name].get("benchmark")
                ),
            }
        )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate a benchmark run into results.json"
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to the run folder (benchmark_logs/{run_name}_{time})",
    )
    parser.add_argument("--runs", required=True, help="Path to the runs JSON file")
    parser.add_argument(
        "--scenarios", default="assets/scenarios.json", help="Path to scenarios.json"
    )
    parser.add_argument("--benchmark", default="green_challenge", help="Benchmark name")
    args = parser.parse_args()

    runs_config = load_json(args.runs)
    run_name = runs_config.get("run_name") or Path(args.runs).stem
    resolved = _resolve_scenarios_from_configs(runs_config, args.scenarios)

    # Resolve asset paths relative to this file so they work from any cwd.
    repo_root = Path(__file__).resolve().parent.parent
    tasks_dir = repo_root / "assets" / "tasks"
    structure_path = repo_root / "assets" / "structure.json"

    results = compute_run_results(
        args.run_dir,
        resolved,
        tasks_dir=str(tasks_dir),
        structure_path=str(structure_path),
        benchmark_name=args.benchmark,
        run_name=run_name,
    )
    out_path = write_results(results, args.run_dir)
    print(f"[aggregator] wrote {out_path}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
