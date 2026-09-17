#!/usr/bin/env bash
# Pen spin with the trained policy: evaluate checkpoints/best_policy.pt on 32 seeded trials (seeds 41000000-41000031),
# score them, and render trial 4 (seed 41000004), the approved video.
#   ISAAC_PY=... PY=... pen/run_policy.sh OUT [GPU]
set -euo pipefail
out=$(realpath -m "$1") gpu=${2:-0}
cd "$(dirname "$0")"
mkdir -p "$out"
export CUDA_VISIBLE_DEVICES=$gpu OMNI_KIT_ACCEPT_EULA=YES OMP_NUM_THREADS=4
"$ISAAC_PY" policy/evaluate.py --ckpt checkpoints/best_policy.pt --run_cfg checkpoints/env_cfg.json \
  --out "$out" --seed0 41000000 --trials 32 --seconds 12 > "$out/eval.log" 2>&1 &
eval_pid=$!
trap 'kill $eval_pid 2>/dev/null || true' EXIT   # Isaac can hang on shutdown after saving
while [[ ! -s "$out/meta.json" ]]; do kill -0 $eval_pid; sleep 2; done
sleep 5

"$PY" policy/score.py "$out" > "$out/score.log"
"$PY" sim/render.py "$out" --trial 4 --views close top
