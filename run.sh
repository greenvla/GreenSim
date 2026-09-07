#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MDL_SYSTEM_PATH="${SCRIPT_DIR}/assets/shaders:${MDL_SYSTEM_PATH:-}"
/isaac-sim/python.sh -m pip install onnxruntime
/isaac-sim/python.sh -m pip install defusedxml

PUBLIC_IP=127.0.0.1 LIVESTREAM=2 /isaac-sim/python.sh scripts/run.py \
    --scenario dry_kitchen.e2.a2 \
    --controller zest \
    --deactivate_torque \
    --robot green \
    --env_simdt_denominator 250
