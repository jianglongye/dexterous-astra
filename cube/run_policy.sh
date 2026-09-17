#!/usr/bin/env bash
# Policy cube: scripted pickup, then five learned one-hand maneuvers (turn L', roll, turn F', roll, turn U').
# Seed 5005, mix U F L. Renders the full solve.
#   PY=... cube/run_policy.sh OUT
set -euo pipefail
out=$(realpath -m "$1")
cd "$(dirname "$0")"
export MUJOCO_GL=egl OMP_NUM_THREADS=1   # single-threaded torch: other thread counts change the rollout
"$PY" policy/solve.py --out "$out" --seed 5005 --scramble "U F L"
OMP_NUM_THREADS=2 PYTHONPATH=sim "$PY" sim/render.py --model "$out/scene.mjb" --trajectory "$out/trajectory.npz" --out "$out/video.mp4"
