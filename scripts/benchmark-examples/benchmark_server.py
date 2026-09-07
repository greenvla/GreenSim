import argparse
import glob
import json
import os
from typing import Any

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = "benchmark_logs"

FIXED_FILE_PATH: str | None = None
SELECTED_RUN_DIR: str | None = None
ALL_RUNS_MODE = False


@app.route("/")
def index():
    return send_from_directory(SCRIPT_DIR, "index.html")


def normalize_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def normalize_run_dir(path: str) -> str:
    """Accept a run directory or its reports/ directory and return run dir."""
    path = normalize_path(path)
    if os.path.basename(os.path.normpath(path)) == "reports":
        return os.path.dirname(path)
    return path


def is_valid_run_dir(run_dir: str) -> bool:
    return os.path.isdir(run_dir) and os.path.isdir(
        os.path.join(run_dir, "reports")
    )


def get_all_run_folders() -> list[str]:
    """Find run directories directly inside BASE_DIR, ordered by mtime."""
    base_dir = normalize_path(BASE_DIR)
    if not os.path.isdir(base_dir):
        return []

    folders = [
        path
        for path in glob.glob(os.path.join(base_dir, "*"))
        if is_valid_run_dir(path)
    ]
    return sorted(folders, key=os.path.getmtime)


def get_latest_run_folder() -> str | None:
    folders = get_all_run_folders()
    return folders[-1] if folders else None


def get_current_run_folder() -> str | None:
    if SELECTED_RUN_DIR:
        return SELECTED_RUN_DIR
    return get_latest_run_folder()


def get_reports_dir() -> str | None:
    if FIXED_FILE_PATH:
        return os.path.dirname(FIXED_FILE_PATH)

    if ALL_RUNS_MODE:
        return None

    run_dir = get_current_run_folder()
    if not run_dir:
        return None

    reports_dir = os.path.join(run_dir, "reports")
    return reports_dir if os.path.isdir(reports_dir) else None


def get_episodes_dir() -> str | None:
    if ALL_RUNS_MODE:
        return None

    run_dir = get_current_run_folder()
    if not run_dir:
        return None

    episodes_dir = os.path.join(run_dir, "episodes")
    return episodes_dir if os.path.isdir(episodes_dir) else None


def get_mode() -> str:
    if FIXED_FILE_PATH:
        return "file"
    if ALL_RUNS_MODE:
        return "all"
    if SELECTED_RUN_DIR:
        return "path"
    return "latest"


def duration_stats(durations: list[float]) -> dict[str, Any]:
    if not durations:
        return {
            "count": 0,
            "min_seconds": None,
            "max_seconds": None,
            "avg_seconds": None,
            "sum_seconds": 0.0,
        }

    return {
        "count": len(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "avg_seconds": sum(durations) / len(durations),
        "sum_seconds": sum(durations),
    }


def find_report_files(reports_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(reports_dir, "episode_*_report.json")))


def find_episode_files(episodes_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(episodes_dir, "episode_*.json")))


def collect_report_statistics(reports_dir: str) -> dict[str, Any]:
    files = find_report_files(reports_dir)

    durations: list[float] = []
    successful_durations: list[float] = []
    failed_durations: list[float] = []
    invalid_reports: list[dict[str, Any]] = []
    skipped_reports: list[dict[str, Any]] = []

    for report_path in files:
        filename = os.path.basename(report_path)

        try:
            with open(report_path, "r", encoding="utf-8") as file:
                report = json.load(file)

            duration = report.get("duration_seconds")
            if not isinstance(duration, (int, float)):
                invalid_reports.append({
                    "filename": filename,
                    "error": "duration_seconds is missing or invalid",
                })
                continue

            duration = float(duration)
            if duration <= 0.1:
                skipped_reports.append({
                    "filename": filename,
                    "reason": "duration_seconds <= 0.1",
                })
                continue

            durations.append(duration)
            if report.get("success") is True:
                successful_durations.append(duration)
            else:
                failed_durations.append(duration)

        except Exception as exc:
            invalid_reports.append({
                "filename": filename,
                "error": str(exc),
            })

    total_episodes = len(durations)

    return {
        "reports_dir": reports_dir,
        "total_files": len(files),
        "total_episodes": total_episodes,
        "successful_episodes": len(successful_durations),
        "failed_episodes": len(failed_durations),
        "success_rate": (
            len(successful_durations) / total_episodes
            if total_episodes > 0
            else 0.0
        ),
        "total_duration_seconds": duration_stats(durations),
        "successful_duration_seconds": duration_stats(successful_durations),
        "failed_duration_seconds": duration_stats(failed_durations),
        "skipped_reports": skipped_reports,
        "invalid_reports": invalid_reports,
        "_durations": durations,
        "_successful_durations": successful_durations,
        "_failed_durations": failed_durations,
    }


def public_report_statistics(stats: dict[str, Any]) -> dict[str, Any]:
    result = dict(stats)
    result.pop("_durations", None)
    result.pop("_successful_durations", None)
    result.pop("_failed_durations", None)
    return result


def collect_episode_statistics(run_dir: str) -> dict[str, Any]:
    reports_dir = os.path.join(run_dir, "reports")
    episodes_dir = os.path.join(run_dir, "episodes")

    if not os.path.isdir(episodes_dir):
        return {
            "run": os.path.basename(run_dir),
            "episodes_dir": episodes_dir,
            "error": "Episodes folder not found",
            "total_files": 0,
            "total_episodes": 0,
            "classified_episodes": 0,
            "successful_episodes": 0,
            "failed_episodes": 0,
            "success_rate": 0.0,
            "total_duration_seconds": duration_stats([]),
            "successful_duration_seconds": duration_stats([]),
            "failed_duration_seconds": duration_stats([]),
            "skipped_episodes": [],
            "invalid_episodes": [],
            "episodes_without_report": [],
            "invalid_reports": [],
            "_durations": [],
            "_successful_durations": [],
            "_failed_durations": [],
        }

    report_success_by_episode: dict[str, bool] = {}
    invalid_reports: list[dict[str, Any]] = []

    for report_path in find_report_files(reports_dir):
        try:
            with open(report_path, "r", encoding="utf-8") as file:
                report = json.load(file)

            episode_id = report.get("episode_id")
            if not episode_id:
                invalid_reports.append({
                    "filename": os.path.basename(report_path),
                    "error": "episode_id is missing",
                })
                continue

            report_success_by_episode[str(episode_id)] = report.get("success") is True

        except Exception as exc:
            invalid_reports.append({
                "filename": os.path.basename(report_path),
                "error": str(exc),
            })

    durations: list[float] = []
    successful_durations: list[float] = []
    failed_durations: list[float] = []
    invalid_episodes: list[dict[str, Any]] = []
    skipped_episodes: list[dict[str, Any]] = []
    episodes_without_report: list[dict[str, Any]] = []

    episode_files = find_episode_files(episodes_dir)
    for episode_path in episode_files:
        filename = os.path.basename(episode_path)

        try:
            with open(episode_path, "r", encoding="utf-8") as file:
                episode = json.load(file)

            episode_id = episode.get("episode_id")
            duration = episode.get("duration_seconds")

            if not episode_id:
                invalid_episodes.append({
                    "filename": filename,
                    "error": "episode_id is missing",
                })
                continue

            if not isinstance(duration, (int, float)):
                invalid_episodes.append({
                    "filename": filename,
                    "error": "duration_seconds is missing or invalid",
                })
                continue

            duration = float(duration)
            if duration <= 0.1:
                skipped_episodes.append({
                    "filename": filename,
                    "episode_id": episode_id,
                    "reason": "duration_seconds <= 0.1",
                })
                continue

            durations.append(duration)
            episode_id = str(episode_id)

            if episode_id not in report_success_by_episode:
                episodes_without_report.append({
                    "filename": filename,
                    "episode_id": episode_id,
                })
                continue

            if report_success_by_episode[episode_id]:
                successful_durations.append(duration)
            else:
                failed_durations.append(duration)

        except Exception as exc:
            invalid_episodes.append({
                "filename": filename,
                "error": str(exc),
            })

    classified_episodes = len(successful_durations) + len(failed_durations)

    return {
        "run": os.path.basename(run_dir),
        "episodes_dir": episodes_dir,
        "total_files": len(episode_files),
        "total_episodes": len(durations),
        "classified_episodes": classified_episodes,
        "successful_episodes": len(successful_durations),
        "failed_episodes": len(failed_durations),
        "success_rate": (
            len(successful_durations) / classified_episodes
            if classified_episodes > 0
            else 0.0
        ),
        "total_duration_seconds": duration_stats(durations),
        "successful_duration_seconds": duration_stats(successful_durations),
        "failed_duration_seconds": duration_stats(failed_durations),
        "skipped_episodes": skipped_episodes,
        "invalid_episodes": invalid_episodes,
        "episodes_without_report": episodes_without_report,
        "invalid_reports": invalid_reports,
        "_durations": durations,
        "_successful_durations": successful_durations,
        "_failed_durations": failed_durations,
    }


def public_episode_statistics(stats: dict[str, Any]) -> dict[str, Any]:
    result = dict(stats)
    result.pop("_durations", None)
    result.pop("_successful_durations", None)
    result.pop("_failed_durations", None)
    return result


def get_latest_report_file() -> str | None:
    if FIXED_FILE_PATH:
        return FIXED_FILE_PATH if os.path.isfile(FIXED_FILE_PATH) else None

    if ALL_RUNS_MODE:
        report_files: list[str] = []
        for run_dir in get_all_run_folders():
            report_files.extend(
                find_report_files(os.path.join(run_dir, "reports"))
            )
        return max(report_files, key=os.path.getmtime) if report_files else None

    reports_dir = get_reports_dir()
    if not reports_dir:
        return None

    report_files = find_report_files(reports_dir)
    return max(report_files, key=os.path.getmtime) if report_files else None


@app.route("/api/latest_episode")
def get_latest_episode():
    latest_file = get_latest_report_file()

    if not latest_file:
        return jsonify({
            "message": "No reports yet",
            "mode": get_mode(),
        }), 200

    try:
        with open(latest_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        data["_filename"] = os.path.basename(latest_file)
        data["_folder"] = os.path.basename(
            os.path.dirname(os.path.dirname(latest_file))
        )
        data["_fixed"] = FIXED_FILE_PATH is not None
        data["_mode"] = get_mode()
        return jsonify(data)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/statistics")
def get_statistics():
    if ALL_RUNS_MODE:
        run_dirs = get_all_run_folders()
        if not run_dirs:
            return jsonify({
                "error": f"No benchmark run directories found in: {normalize_path(BASE_DIR)}"
            }), 404

        runs: list[dict[str, Any]] = []
        all_durations: list[float] = []
        all_successful_durations: list[float] = []
        all_failed_durations: list[float] = []
        invalid_reports: list[dict[str, Any]] = []
        skipped_reports: list[dict[str, Any]] = []
        total_files = 0
        total_episodes = 0
        successful_episodes = 0
        failed_episodes = 0

        for run_dir in run_dirs:
            run_name = os.path.basename(run_dir)
            stats = collect_report_statistics(os.path.join(run_dir, "reports"))

            all_durations.extend(stats["_durations"])
            all_successful_durations.extend(stats["_successful_durations"])
            all_failed_durations.extend(stats["_failed_durations"])

            total_files += stats["total_files"]
            total_episodes += stats["total_episodes"]
            successful_episodes += stats["successful_episodes"]
            failed_episodes += stats["failed_episodes"]

            invalid_reports.extend([
                {"run": run_name, **item}
                for item in stats["invalid_reports"]
            ])
            skipped_reports.extend([
                {"run": run_name, **item}
                for item in stats["skipped_reports"]
            ])

            run_stats = public_report_statistics(stats)
            run_stats["run"] = run_name
            runs.append(run_stats)

        return jsonify({
            "mode": "all",
            "base_dir": normalize_path(BASE_DIR),
            "total_runs": len(run_dirs),
            "total_files": total_files,
            "total_episodes": total_episodes,
            "successful_episodes": successful_episodes,
            "failed_episodes": failed_episodes,
            "success_rate": (
                successful_episodes / total_episodes
                if total_episodes > 0
                else 0.0
            ),
            "total_duration_seconds": duration_stats(all_durations),
            "successful_duration_seconds": duration_stats(all_successful_durations),
            "failed_duration_seconds": duration_stats(all_failed_durations),
            "invalid_reports": invalid_reports,
            "skipped_reports": skipped_reports,
            "runs": runs,
        })

    reports_dir = get_reports_dir()
    if not reports_dir:
        return jsonify({
            "error": "Reports folder not found",
            "mode": get_mode(),
        }), 404

    stats = public_report_statistics(collect_report_statistics(reports_dir))
    return jsonify({
        "mode": get_mode(),
        "run": os.path.basename(os.path.dirname(reports_dir)),
        **stats,
    })


@app.route("/api/episode_statistics")
def get_episode_statistics():
    if ALL_RUNS_MODE:
        run_dirs = get_all_run_folders()
        if not run_dirs:
            return jsonify({
                "error": f"No benchmark run directories found in: {normalize_path(BASE_DIR)}"
            }), 404

        runs: list[dict[str, Any]] = []
        all_durations: list[float] = []
        all_successful_durations: list[float] = []
        all_failed_durations: list[float] = []
        skipped_episodes: list[dict[str, Any]] = []
        invalid_episodes: list[dict[str, Any]] = []
        episodes_without_report: list[dict[str, Any]] = []
        invalid_reports: list[dict[str, Any]] = []
        total_files = 0
        total_episodes = 0
        classified_episodes = 0
        successful_episodes = 0
        failed_episodes = 0

        for run_dir in run_dirs:
            run_name = os.path.basename(run_dir)
            stats = collect_episode_statistics(run_dir)

            all_durations.extend(stats["_durations"])
            all_successful_durations.extend(stats["_successful_durations"])
            all_failed_durations.extend(stats["_failed_durations"])

            total_files += stats["total_files"]
            total_episodes += stats["total_episodes"]
            classified_episodes += stats["classified_episodes"]
            successful_episodes += stats["successful_episodes"]
            failed_episodes += stats["failed_episodes"]

            skipped_episodes.extend([
                {"run": run_name, **item}
                for item in stats["skipped_episodes"]
            ])
            invalid_episodes.extend([
                {"run": run_name, **item}
                for item in stats["invalid_episodes"]
            ])
            episodes_without_report.extend([
                {"run": run_name, **item}
                for item in stats["episodes_without_report"]
            ])
            invalid_reports.extend([
                {"run": run_name, **item}
                for item in stats["invalid_reports"]
            ])

            runs.append(public_episode_statistics(stats))

        return jsonify({
            "mode": "all",
            "base_dir": normalize_path(BASE_DIR),
            "total_runs": len(run_dirs),
            "total_files": total_files,
            "total_episodes": total_episodes,
            "classified_episodes": classified_episodes,
            "successful_episodes": successful_episodes,
            "failed_episodes": failed_episodes,
            "success_rate": (
                successful_episodes / classified_episodes
                if classified_episodes > 0
                else 0.0
            ),
            "total_duration_seconds": duration_stats(all_durations),
            "successful_duration_seconds": duration_stats(all_successful_durations),
            "failed_duration_seconds": duration_stats(all_failed_durations),
            "skipped_episodes": skipped_episodes,
            "invalid_episodes": invalid_episodes,
            "episodes_without_report": episodes_without_report,
            "invalid_reports": invalid_reports,
            "runs": runs,
        })

    run_dir = get_current_run_folder()
    if not run_dir:
        return jsonify({
            "error": "Benchmark run directory not found",
            "mode": get_mode(),
        }), 404

    return jsonify({
        "mode": get_mode(),
        **public_episode_statistics(collect_episode_statistics(run_dir)),
    })


def configure_from_args() -> argparse.Namespace:
    global ALL_RUNS_MODE
    global FIXED_FILE_PATH
    global SELECTED_RUN_DIR

    parser = argparse.ArgumentParser(description="Benchmark Monitor Server")
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default=None,
        help="Path to a benchmark run directory or directly to its reports/",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Aggregate statistics across all run directories in benchmark_logs/",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Pin to a specific episode_*_report.json",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Server port, default 5000",
    )

    args = parser.parse_args()

    selected_count = sum(bool(value) for value in (args.path, args.all, args.file))
    if selected_count > 1:
        parser.error("Use only one of the keys: --path, --all or --file")

    ALL_RUNS_MODE = args.all

    if args.file:
        FIXED_FILE_PATH = normalize_path(args.file)
        if not os.path.isfile(FIXED_FILE_PATH):
            parser.error(f"Report file not found: {FIXED_FILE_PATH}")

    if args.path:
        SELECTED_RUN_DIR = normalize_run_dir(args.path)
        if not is_valid_run_dir(SELECTED_RUN_DIR):
            parser.error(
                "Invalid benchmark run directory. "
                f"Expected a folder with reports/: {SELECTED_RUN_DIR}"
            )

    return args


def print_startup_info(port: int) -> None:
    print("🚀 Starting monitoring server...")

    if FIXED_FILE_PATH:
        print("Mode: specific report file")
        print(f"File: {FIXED_FILE_PATH}")
    elif ALL_RUNS_MODE:
        print("Mode: all benchmark run directories")
        print(f"Base folder: {normalize_path(BASE_DIR)}")
        print(f"Run directories found: {len(get_all_run_folders())}")
    elif SELECTED_RUN_DIR:
        print("Mode: selected benchmark run")
        print(f"Run: {SELECTED_RUN_DIR}")
    else:
        latest_run = get_latest_run_folder()
        print("Mode: latest benchmark run")
        print(f"Run: {latest_run or 'not found'}")

    print(f"Web interface: http://127.0.0.1:{port}")
    print(f"API: http://127.0.0.1:{port}/api/latest_episode")
    print("⏹ Press Ctrl+C to stop")
    print("-" * 50)


if __name__ == "__main__":
    args = configure_from_args()
    print_startup_info(args.port)
    app.run(debug=True, port=args.port, use_reloader=False)
