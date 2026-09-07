#!/usr/bin/env python3
"""Sequential multi-scenario benchmark runner.

Reads a run config from ``runs/*.json``, launches ``run_with_vla.py`` once per
scenario as a subprocess (each gets a fresh Isaac Sim app), then aggregates the
per-episode reports into a run-level ``results.json``.

Usage:
    python scripts/run_benchmark.py [assets/runs/green_challenge.json]

The run config may also be selected with the RUN_PATH environment variable,
which is how ``make up`` passes it into the container.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "scripts"))

import results_aggregator as agg  # noqa: E402


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else SCRIPT_DIR / path


def _args_to_flags(args: dict) -> list:
    """Convert ``{"flag": value}`` into CLI tokens.

    ``True`` -> ``--flag``, ``False``/``None`` -> omitted, other -> ``--flag value``.
    """
    flags = []
    for name, value in args.items():
        flag = f"--{name}"
        if value is True:
            flags.append(flag)
        elif value is False or value is None:
            continue
        else:
            flags.extend([flag, str(value)])
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-scenario benchmark")
    parser.add_argument(
        "runs",
        nargs="?",
        # ``or`` so RUN_PATH exported empty (Makefile always passes -e) falls through.
        default=os.environ.get("RUN_PATH") or "assets/runs/green_challenge.json",
        help="Path to the runs JSON file (default: $RUN_PATH, else the green_challenge run)",
    )
    parser.add_argument(
        "--scenarios", default="assets/scenarios.json", help="Path to scenarios.json"
    )
    parser.add_argument(
        "--benchmark_logs_dir",
        default="benchmark_logs",
        help="Base directory for run folders",
    )
    parser.add_argument(
        "--benchmark",
        default="green_challenge",
        help="Benchmark name recorded in results.json",
    )

    parser.add_argument(
        "--ros-launcher",
        nargs="?",
        const="run_in_conda.sh",
        metavar="PATH",
        help=(
            "Path to the ROS launcher script. Activating this switches "
            "Isaac Lab to heavy mode. If omitted, runs in lightweight mode."
        ),
    )

    args = parser.parse_args()

    runs_path = _resolve_path(args.runs)
    if not runs_path.is_file():
        raise FileNotFoundError(f"Run config not found: {runs_path}")
    print(f"[runner] run config: {runs_path}")
    runs_config = agg.load_json(runs_path)
    scenarios_cfg = agg.load_json(_resolve_path(args.scenarios))

    run_name = runs_config.get("run_name") or Path(args.runs).stem
    global_args = runs_config.get("args", {})
    scenarios = runs_config["scenarios"]

    # Validate scenario names up front so a typo fails before any run starts.
    for entry in scenarios:
        if entry["scenario"] not in scenarios_cfg:
            available = ", ".join(sorted(scenarios_cfg))
            raise ValueError(
                f"Unknown scenario '{entry['scenario']}'. Available: {available}"
            )

    run_dir = (
        _resolve_path(args.benchmark_logs_dir)
        / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"[runner] run dir: {run_dir}")

    # Total episodes across all scenarios, passed to each subprocess for its progress log.
    total_episodes = 0
    for entry in scenarios:
        _cfg = scenarios_cfg[entry["scenario"]]
        _n = entry.get("num_episodes")
        if _n is None:
            _n = _cfg.get("num_episodes", 10)
        total_episodes += int(_n)

    resolved = []
    for entry in scenarios:
        scenario_name = entry["scenario"]
        scenario_cfg = scenarios_cfg[scenario_name]
        scene = scenario_cfg["scene"]
        benchmark_task = agg.resolve_optional_feature(scenario_cfg.get("benchmark"))
        policy = agg.resolve_optional_feature(scenario_cfg.get("policy"))
        num_episodes = entry.get("num_episodes")
        if num_episodes is None:
            num_episodes = scenario_cfg.get("num_episodes", 10)

        resolved.append({"scenario": scenario_name, "benchmark_task": benchmark_task})

        cmd_args = [
            "--scene",
            scene,
            "--benchmark_output_dir",
            str(run_dir / scenario_name),
        ]
        if benchmark_task:
            cmd_args += ["--benchmark_task", benchmark_task]

        cmd_args += ["--num_episodes", str(num_episodes)]
        cmd_args += ["--benchmark_total_episodes", str(total_episodes)]

        if args.ros_launcher is None:
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "scripts" / "run_with_vla.py"),
                "--task",
                "ManipEnv-v0",
                *cmd_args,
                *_args_to_flags(global_args),
            ]
        else:
            ros_launcher = _resolve_path(args.ros_launcher)

            if not ros_launcher.is_file():
                raise FileNotFoundError(f"ROS launcher not found: {ros_launcher}")
            if not os.access(ros_launcher, os.X_OK):
                raise PermissionError(f"ROS launcher is not executable: {ros_launcher}")

            cmd = [
                str(ros_launcher),
                *cmd_args,
            ]
            if policy:
                cmd += ["--policy", policy]

        print(
            f"\n[runner] scenario '{scenario_name}' -> scene '{scene}', "
            f"benchmark_task '{benchmark_task}', {num_episodes} episodes"
        )
        print("[runner] command: " + " ".join(cmd))
        # argv-array invocation without a shell: elements are passed to exec
        # directly, so no command injection is possible (taint rule false
        # positive — shlex.quote would corrupt argv paths, not sanitize them).
        proc = subprocess.run(  # nosemgrep
            cmd, cwd=str(SCRIPT_DIR), shell=False
        )
        if proc.returncode != 0:
            print(
                f"[runner] WARNING: scenario '{scenario_name}' exited with code {proc.returncode}"
            )

    print("\n[runner] aggregating results...")
    results = agg.compute_run_results(
        run_dir,
        resolved,
        tasks_dir=str(_resolve_path("assets/tasks")),
        structure_path=str(_resolve_path("assets/structure.json")),
        benchmark_name=args.benchmark,
        run_name=run_name,
    )
    out_path = agg.write_results(results, run_dir)
    print(f"[runner] results written to {out_path}")

    # Mirror the final results.json to the location in RESULTS_PATH (if set).
    results_path = os.environ.get("RESULTS_PATH")
    if results_path:
        os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
        shutil.copyfile(out_path, results_path)
        print(f"[runner] results copied to {results_path}")
    else:
        print("[runner] RESULTS_PATH not set — skipping the mirror copy")

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
