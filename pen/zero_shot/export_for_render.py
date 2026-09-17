"""Convert a zero-shot run (trajectory.npz from mpc_pen.py) to the evaluate.py layout so sim/render.py can draw it.

python zero_shot/export_for_render.py RUN      # writes RUN/trajectories.npz and RUN/meta.json
"""

import json
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
z = np.load(run / "trajectory.npz")
axis = z["pen_axis"]
arrays = {k: z[k][None, 1:] for k in ("q", "pen_pos", "pen_quat", "pen_axis", "finger_force", "physx_penetration")}
arrays.update(
    q0=z["q"][None, 0],
    pen_pos0=z["pen_pos"][None, 0],
    pen_quat0=z["pen_quat"][None, 0],
    heading0=np.arctan2(axis[None, 0, 1], axis[None, 0, 0]),
)
np.savez_compressed(run / "trajectories.npz", **arrays)
report = json.loads((run / "report.json").read_text())
meta = dict(control_dt=1 / 60, seed0=report["seed"], trials=1)
(run / "meta.json").write_text(json.dumps(meta, indent=1))
