#!/usr/bin/env bash
# Zero-shot cube: two floating Wuji hands, non-RL scripted controller, scramble U2 R' F, 51 s (two of three moves).
#   PY=... cube/run_zero_shot.sh OUT
set -euo pipefail
out=$(realpath -m "$1")
cd "$(dirname "$0")"
MUJOCO_GL=egl "$PY" sim/run.py --controller zero_shot/controller.py --embodiment floating \
  --scramble "U2 R' F" --seconds 51 --out "$out" --video
