#!/usr/bin/env bash
# Shared VLA policy calibration — single source of truth for the observation/action
# settings the VLA checkpoint was trained on. Sourced by both run_with_vla.sh
# (interactive apple-pick) and run_benchmark.sh (multi-scenario benchmark) so the
# policy is always driven in-distribution. Every value stays overridable via env.

# PROPRIO_CMD=all: send the policy its own last command as proprioception instead of the
# measurement — a positional PD always lags its setpoint, and the policy never saw that lag.
export PROPRIO_CMD="${PROPRIO_CMD:-all}"

# Hold the start pose for the first ticks of an episode. Default `canonical` keeps the
# robot's canonical joint pose (arms hanging down); `zest_manipulate` holds the trained
# home pose (elbows bent, hands over the table).
export START_POSE="${START_POSE:-canonical}"
export START_POSE_STEPS="${START_POSE_STEPS:-50}"
# Seconds, not ticks: the arms need about a second to travel there, whatever the
# control rate. As a tick count this silently became 0.2 s at a 0.004 s tick.
export START_POSE_SEC="${START_POSE_SEC:-1.0}"

# Let the arm reach the last target of a chunk before asking for the next one, so the
# next chunk is not planned from a pose the robot never reached.
export SETTLE_STEPS="${SETTLE_STEPS:-20}"   # only without the server feed

# Static feed-forward against the right-arm sag: a positional PD holds the elbow ~0.12 rad
# less bent than commanded, so the hand hangs below where the policy aimed.
export ARM_FF="${ARM_FF:-0}"

# Hand commands are absolute joint targets, not offsets from the default pose (the right
# thumb yaw defaults to its lower limit and would bias every grasp open).
export ABS_HAND_TARGETS="${ABS_HAND_TARGETS:-1}"

# Honour the ftheta_* calibration of the fisheye presets, so the head and wrist cameras
# have the optical axis the checkpoint was trained with instead of a shifted one.
export CAMERA_FTHETA="${CAMERA_FTHETA:-1}"

# Ambient fill on top of the three disk lights, the way the policy's own runs were lit.
export DOME_LIGHT="${DOME_LIGHT:-1}"

# SIM_SEED=random draws a fresh seed per process, so a series covers the spawn box
# instead of one point in it. Left at the code default here: the contest run should
# be reproducible, a measurement should not be.
export SIM_SEED="${SIM_SEED:-42}"

# Number of VLA action steps executed before the next inference.
export ACTION_HORIZON="${ACTION_HORIZON:-50}"    # only without the server feed
