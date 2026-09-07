# Green Challenge Sim — Isaac Lab / Isaac Sim Manipulation Simulator

- [Prerequisites](#prerequisites)
- [1. Clone](#1-clone)
- [2. Start & enter the container](#2-start--enter-the-container)
- [3. Multi-scenario benchmark](#3-multi-scenario-benchmark)
- [4. Open-source Green robot models](#4-open-source-green-robot-models)
- [5. Watching the sim (WebRTC client)](#5-watching-the-sim-webrtc-client)

---

## Prerequisites

- [CUDA-capable GPU](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- Docker

---

## 1. Clone

Clone the project. The large files (USD models, assets) are not in the repo — download them
separately in [step 2](#download-assets-one-time).

```bash
git clone https://github.com/greenvla/GreenSim green_challenge
cd green_challenge
```

---

## 2. Start & enter the container

```bash
# Build the Docker image
make build
```

### Download assets (one-time)

The large USD models and the benchmark dataset are not in the repo — download them once from
HuggingFace into the repo root:

```bash
# Assets (USD models: baskets, tables, fruit, …)
huggingface-cli download SberRoboticsCenter/GreenChallengeAssets --repo-type dataset --local-dir assets/usd_models

# Benchmark dataset
huggingface-cli download SberRoboticsCenter/GreenChallengeData --repo-type dataset --local-dir data
```

`docker/docker-compose.yaml` bind-mounts the whole repo to `/workspace/green_challenge`, so
anything placed next to this file is visible inside the container without rebuilding the image.

### Start the container and launch the benchmark

`make up` starts the container and immediately runs the multi-scenario benchmark
([section 3](#3-multi-scenario-benchmark)). It needs the assets downloaded and the VLA policy
server already running.

```bash
make up                    # container + benchmark
make up RUN_BENCHMARK=0    # container only, no benchmark
```

Arguments example (container + benchmark):

```bash
LIVESTREAM=2 PUBLIC_IP=127.0.0.1 RUN_PATH=assets/runs/green_challenge.json RESULTS_PATH=/workspace/green_challenge/results.json make up
```

Arguments:

`LIVESTREAM` / `PUBLIC_IP` (optional) enable WebRTC livestreaming of the benchmark. They are
forwarded to every scenario's Isaac Sim app; connect the WebRTC client to `PUBLIC_IP:49100`
(`LIVESTREAM=1` = public network, `LIVESTREAM=2` = private network). The container uses
`network_mode: host`, so no port needs publishing.

`RUN_PATH` (optional) selects the run config, relative to the repo root or absolute. If unset,
`assets/runs/green_challenge.json` is used. `make up` invokes `run_benchmark.sh` without
arguments, so inside `make up` this variable is the only way to choose a config.

`RESULTS_PATH` (optional) is the file where the aggregated `results.json` is copied. If unset,
the mirror copy is skipped (the run-level `results.json` stays in the run directory).

`POLICY_PORT` (optional) is the port of the VLA policy server the benchmark connects to
(default `8999`).

### Make targets

| Target | Effect |
| --- | --- |
| `make build` | Build the `green-challenge:latest` image. |
| `make up` | Start the container, then run the benchmark unless `RUN_BENCHMARK=0`. |
| `make enter` | Open a shell in the running container. |
| `make stop` | Stop the container, keeping it around. |
| `make down` | Stop and remove the container. |
| `make logs` | Follow the container logs. |
| `make clean` | `down -v` and remove the image. |

To enter the running container in a separate terminal:

```bash
make enter
# Optional: run manually inside the container
cd /workspace/green_challenge
./run_benchmark.sh assets/runs/green_challenge.json
```

The green_challenge package is automatically added to the `PYTHONPATH` on container start. No
manual `pip install` is needed.

---

## 3. Multi-scenario benchmark

`scripts/run_benchmark.py` (wrapped by `run_benchmark.sh`) runs a list of scenarios sequentially and
aggregates the per-episode reports into a run-level score. Each scenario launches a fresh
Isaac Sim app through `scripts/run_with_vla.py` (VLA policy + ZEST balance), so the VLA policy server
must already be running.

### Prerequisites

- The VLA policy server listening on `127.0.0.1:8999` over a plaintext WebSocket.
- Assets downloaded ([step 2](#download-assets-one-time)).

### Run configuration

A run is described by a JSON file in `assets/runs/`, e.g. `assets/runs/test.json`:

```json
{
  "run_name" : "green_challenge",
  "args"     : {
    "controller"           : "zest",
    "robot"                : "green",
    "control_mode"         : "vla",
    "frame_mode"           : "latest",
    "env_simdt_denominator": 250,
    "env_decimation"       : 1,
    "deactivate_torque"    : true
  },
  "scenarios": [
    {"scenario": "test_pick.e1.a1", "num_episodes": 2},
    {"scenario": "test_pick.e1.a2", "num_episodes": 2}
  ]
}
```

- `run_name` — prefix of the run directory under `benchmark_logs/`.
- `args` — CLI flags forwarded to every `scripts/run_with_vla.py` invocation. Names come from
  `configs/cli_args.json` (plus the VLA-specific flags declared in `scripts/run_with_vla.py`).
- `scenarios` — scenario names from `assets/scenarios.json`, each with the number of episodes.
  `num_episodes` may be omitted, in which case the scenario's own value is used, defaulting
  to 10. An unknown scenario name aborts the run before any Isaac Sim app starts.

### Tasks

The competition tasks of the [Green Challenge 2026](https://dsworks.ru/en/champ/aij26-green), defined
in `assets/runs/green_challenge.json`. The time shown is the episode timeout.

- **`dry_kitchen.e3.a2`** *(52 s)* — Raise your right hand, pick up the sugar bowl with your right hand, place the sugar bowl onto the tray with your right hand, move your right hand away from the sugar bowl.
  - Raise your right hand.
  - Pick the sugar bowl from the shelf with your right hand.
  - Place the sugar bowl onto the tray with your right hand.
  - Move your right hand away from the sugar bowl.
- **`sber_shop.e1.a2`** *(70 s)* — Lift the lid of the Sber Ring box with your right hand, place the lid aside with your right hand, push the Sber Ring box towards the visitor with your right hand, move the Sber Ring box back with your right hand.
  - Raise your right hand above the table.
  - Lift the lid of the Sber Ring box with your right hand.
  - Place the lid aside with your right hand.
  - Push the Sber Ring box towards the visitor with your right hand.
  - Move the Sber Ring box back to yourself with your right hand.
- **`darkstore.e1.a2`** *(35 s)* — Pick the red soda can up with your right hand, place the red soda can in the empty space next to the matching red soda cans.
  - Pick the red soda can up with your right hand.
  - Place the red soda can in the empty space next to the matching red soda cans.

### Run

Start the VLA policy server, then
[launch the benchmark](#start-the-container-and-launch-the-benchmark). Scenarios start
automatically.

### Output

```text
benchmark_logs/<run_name>_<timestamp>/
├── <scenario>/
│   └── <task_name>_<timestamp>/                # one per scripts/run_with_vla.py launch
│       ├── episodes/episode_<NNN>.json         # raw episode messages + subtask_history
│       └── reports/episode_<NNN>_report.json   # per-episode report
└── results.json                                # aggregated run score (also copied to RESULTS_PATH, if set)
```

The `<task_name>_<timestamp>` level is created by `EpisodeRecorder`; the aggregator finds the
`episodes/` and `reports/` folders with `rglob`, so the extra nesting is transparent to it.

### Scoring

Aggregation is done by `scripts/results_aggregator.py`. `T_max` is the `duration_sec` of the
task's `timeout` relation, read from `assets/tasks/<benchmark_task>.json`.

Per-episode metrics, from `reports/episode_<NNN>_report.json`:

- `s_i` — binary success (0/1); `success_rate = mean(s_i)`.
- `t_i` — episode duration in simulation seconds (the report's `duration_seconds`).
- `Speed_i = clamp(1 - t_i / T_max, 0, 1)`; `speed = mean(Speed_i)`. Note `Speed_i` is *not*
  gated on success. If the task declares no `timeout` relation, `Speed_i` falls back to `s_i`.

The run `score` is subtask-level, from `subtask_history` in `episodes/episode_<NNN>.json`:

- `t_{i,g} = end_sim_timestamp - start_sim_timestamp` of subtask `g` in episode `i`.
- `Speed_{i,g} = clamp(1 - t_{i,g} / T_max, 0, 1)` for **completed** subtasks; an incomplete
  subtask contributes 0.
- `score = Σ Speed_{i,g}` — a raw sum over completed subtasks, **not** an average, so it grows
  with the number of episodes and subtasks.
- `subtask_completed` / `subtask_total`, where `subtask_total = episodes × len(annotations)`
  from `assets/structure.json` (falling back to the number of started subtasks if the scenario
  is missing there).

Also reported: `scoring` — per-goal `achieved` / `runs` / `success_rate` — and `goals_score`,
the total number of achieved goals. Every metric appears per scenario under the `subtasks` key
and at the run level in `results.json`. `score` is rounded to 8 decimals and `speed` to 6;
`success_rate` and the per-goal rates are percentages rounded to 2 decimals.

---

## 4. Open-source Green robot models

`assets/green_open_source/` contains the open-source Green robot models, reusable outside this
repository:

- `green.xml` — the Green robot as a MuJoCo model (closed kinematics, crank-linkage wrists/ankles).
- `green_open.xml` — the open-kinematics variant (directly actuated wrists/ankles).
- `green_open.urdf` — the URDF export of the open variant.
- `meshes/` — the `.stl` visual/collision meshes for every link.

These are the same models the simulator loads (selected via the `--robot` flag / `robot`
environment variable); the XML/URDF files can be reused directly in other MuJoCo/URDF-based
tools.

---

## 5. Watching the sim (WebRTC client)

Requires the container to have been started with `LIVESTREAM=1` or `LIVESTREAM=2`
([section 2](#start-the-container-and-launch-the-benchmark)); a default `make up` renders
headless and streams nothing.

Download [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)<br>
Read [docs](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/manual_livestream_clients.html)

Run:

```bash
./isaacsim-webrtc-streaming-client-<version>-linux-x64.AppImage --no-sandbox
# Enter the host (machine with Isaac Sim) IP in the client — the same address as PUBLIC_IP
```

## License and third-party assets

The project source code is distributed under the MIT License; see `LICENSE`.

The MIT License does not relicense third-party 3D assets. Available source,
attribution, license, and modification information for assets distributed from
`assets/usd_models/` is maintained in:

[`assets/usd_models/THIRD_PARTY_ASSET_NOTICES.md`](assets/usd_models/THIRD_PARTY_ASSET_NOTICES.md)

This repository does not distribute NVIDIA Isaac Sim, Omniverse Kit, NVIDIA
container images, or NVIDIA asset packs as part of the project.