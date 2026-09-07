#!/usr/bin/env bash
# run_with_vla.sh — apple-pick run driven by the VLA policy server.
#
# Run INSIDE the green-challenge container (it needs a TTY — the run reads the keyboard):
#
#     make up && make enter                     # on the host
#     cd /workspace/green_challenge && ./run_with_vla.sh
#
# Expects on the host:
#   1) the policy server:  S0S1_SWAP_WRISTS=1 S0S1_JPEG_Q=50 S0S1_THUMB_HW_MAP=1 ./gv.sh up v1
#   2) the container:      make up
#
# Every variable below can be overridden from the environment, e.g.
#   LIVESTREAM=0   ./run_with_vla.sh     headless, no WebRTC
#   MAX_TIME=600   ./run_with_vla.sh     stop after 10 minutes of wall clock
#   ARM_FF=0       ./run_with_vla.sh     without right-arm gravity compensation
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export MDL_SYSTEM_PATH="${SCRIPT_DIR}/assets/shaders:${MDL_SYSTEM_PATH:-}"

# ── live viewing: WebRTC stream on PUBLIC_IP:49100 ───────────────────────────
# LIVESTREAM>=1 also selects isaaclab.python.rendering.kit, the only experience with a
# viewport. With LIVESTREAM=0 a WebRTC client would connect to a black screen.
export LIVESTREAM="${LIVESTREAM:-1}"
export PUBLIC_IP="${PUBLIC_IP:-127.0.0.1}"

# ── VLA policy calibration ───────────────────────────────────────────────────
# Single source of truth shared with the benchmark runner (run_benchmark.sh), so the
# policy is always driven in the distribution it was trained on. See vla_env.sh for
# the values and why each one is set.
source "${SCRIPT_DIR}/vla_env.sh"

# ── benchmark scoring ────────────────────────────────────────────────────────
# OFF by default, because it does not just score the episode — it ENDS it.
# assets/tasks/rl_apple_pick.json counts "right thumb base within 0.15 m of the apple" as
# the success condition, and a success terminates the episode (relation_monitor ->
# handle_termination_event -> ExampleTerminationsCfg.relation_success), so the run was
# being cut off a few seconds in, while the hand was merely NEAR the fruit and long
# before it had closed on it.
# The apple is spawned by the scene, not by the task, so the pick works either way.
# Turn this on only when you want the reports in benchmark_logs/, and expect the episode
# to end at first contact again:
#     BENCHMARK_TASK=rl_apple_pick ./run_with_vla.sh
BENCHMARK_TASK="${BENCHMARK_TASK:-}"

LIVESTREAM=1 /isaac-sim/python.sh scripts/run_with_vla.py \
    --task ManipEnv-v0 --scene scene.test_pick.e1.a1 \
    ${BENCHMARK_TASK:+--benchmark_task "$BENCHMARK_TASK"} \
    --robot green --controller zest --deactivate_torque \
    --control_mode vla --policy_host 127.0.0.1 --policy_port 8999 \
    --task_instruction "${TASK_INSTRUCTION:-Pick big green apple from table with your right hand}" \
    --frame_mode latest \
    --env_simdt_denominator 250 --env_decimation 5 \
    --max_time "${MAX_TIME:-7200}" \
    "$@"
