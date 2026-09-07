#!/bin/bash

# Sequential multi-scenario benchmark runner (see scripts/run_benchmark.py).
# Run with the Isaac Sim Python so spawned scripts/run_with_vla.py subprocesses
# inherit the correct interpreter.
#
# Example:
#   ./run_benchmark.sh assets/runs/green_challenge.json
#   RUN_PATH=assets/runs/test.json ./run_benchmark.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

unset DISPLAY

export MDL_SYSTEM_PATH="${SCRIPT_DIR}/assets/shaders:${MDL_SYSTEM_PATH:-}"

# Same VLA calibration as run_with_vla.sh — see vla_env.sh.
source "${SCRIPT_DIR}/vla_env.sh"

# Passed straight through: with no args scripts/run_benchmark.py falls back to
# $RUN_PATH, then to the default run config.
/isaac-sim/python.sh scripts/run_benchmark.py "$@"
