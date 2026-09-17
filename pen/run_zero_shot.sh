#!/usr/bin/env bash
# Zero-shot pen spin: non-RL sampled finger-keyframe planner, seed 3, then an MP4 of the recorded run.
#   ISAAC_PY=... PY=... pen/run_zero_shot.sh OUT [GPU]
# ISAAC_PY: Isaac Lab python (GPU). PY: CPU python with pen/requirements.txt (rendering).
set -euo pipefail
out=$(realpath -m "$1") gpu=${2:-0}
cd "$(dirname "$0")"
mkdir -p "$out"
export CUDA_VISIBLE_DEVICES=$gpu OMNI_KIT_ACCEPT_EULA=YES OMP_NUM_THREADS=3
# The actor simulates the recorded run; the planner evaluates candidate finger targets in separate copies.
"$ISAAC_PY" zero_shot/mpc_pen.py --role actor --out "$out" --seed 3 > "$out/actor.log" 2>&1 &
actor=$!
"$ISAAC_PY" zero_shot/mpc_pen.py --role planner --out "$out" --seed 3 > "$out/planner.log" 2>&1 &
planner=$!
trap 'kill $actor $planner 2>/dev/null || true' EXIT   # Isaac can hang on shutdown after saving
while [[ ! -s "$out/finished.json" ]]; do kill -0 $actor; sleep 2; done
cat "$out/report.json"; echo

"$PY" zero_shot/export_for_render.py "$out"
"$PY" sim/render.py "$out" --title "zero-shot planner, no learning"
