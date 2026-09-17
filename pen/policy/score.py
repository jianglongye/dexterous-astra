"""Score recorded Isaac Lab trials: MuJoCo convex-hull penetration replay plus the frozen gate."""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from gate import A, check_trial  # noqa: E402
from mj_scene import Index, build, pen_penetration  # noqa: E402
from spec import FINGERS, JOINTS  # noqa: E402

SLICES = {f: [i for i, j in enumerate(JOINTS) if f"_{f}_" in j] for f in FINGERS}


def penetration_series(job):
    """Penetration (mm) at the settled start and after every control step of one trial."""
    q0, p0, r0, q, pos, quat = job
    m = build(cameras=False)
    d = mujoco.MjData(m)
    ix = Index(m)
    frames = [(q0, p0, r0)] + list(zip(q, pos, quat))
    out = np.zeros(len(frames))
    for i, (qi, pi, ri) in enumerate(frames):
        ix.set_state(d, qi, pi, ri)
        mujoco.mj_kinematics(m, d)
        out[i] = 1000 * pen_penetration(m, d, ix)
    return out


def posture(z):
    """Diagnostics only (not acceptance): joint-range usage after the first second."""
    u = (z["q"][:, 60:] - z["q_lo"]) / (z["q_hi"] - z["q_lo"])
    f = z["finger_force"][:, 60:]
    return {
        "near_limit_frac": float((np.abs(u - 0.5) > 0.45).mean()),
        "mean_abs_from_mid": float(np.abs(u - 0.5).mean()),
        "thumb_contact_frac": float((f[..., 0] > A["contact_n"]).mean()),
        "finger_force_median_n": float(np.median(f.sum(-1))),
        "finger_force_p99_n": float(np.quantile(f.sum(-1), 0.99)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    run = Path(a.run)
    z = np.load(run / "trajectories.npz")
    meta = json.loads((run / "meta.json").read_text())
    n = meta["trials"]
    jobs = [
        (z["q0"][i], z["pen_pos0"][i], z["pen_quat0"][i], z["q"][i], z["pen_pos"][i], z["pen_quat"][i])
        for i in range(n)
    ]
    with Pool(a.workers) as pool:
        pens = pool.map(penetration_series, jobs)
    results, passed = [], 0
    for i in range(n):
        rec = {k: z[k][i] for k in ("pen_pos", "pen_axis", "pen_angvel", "q", "qd", "finger_force")}
        rec["ref_pos"], rec["heading0"] = z["ref_pos"][i], z["heading0"][i]
        ok, det = check_trial(rec, SLICES, z["q_lo"], z["q_hi"], penetration_mm=pens[i])
        det["initial_penetration_mm"] = float(pens[i][0])
        det["physx_max_penetration_mm"] = float(1000 * z["physx_penetration"][i].max())
        det["unsettled"] = bool(z["unsettled"][i])
        ok = ok and not det["unsettled"]
        det.update(trial=i, seed=meta["seed0"] + i, passed=ok)
        results.append(det)
        passed += ok
    keys = [
        "turns_ok",
        "timing_ok",
        "hold_ok",
        "drop",
        "limits_ok",
        "penetration_ok",
        "participation_ok",
        "all_fingers_ok",
        "thumb_release_ok",
    ]
    summary = {
        "run": str(run),
        "meta": meta,
        "passed": int(passed),
        "trials": n,
        "check_rates": {k: float(np.mean([bool(r[k]) for r in results])) for k in keys},
        "net_turns_mean": float(np.mean([r["net_turns"] for r in results])),
        "net_turns_quantiles": np.quantile([r["net_turns"] for r in results], [0.1, 0.5, 0.9]).tolist(),
        "max_penetration_mm_p95": float(np.quantile([r["max_penetration_mm"] for r in results], 0.95)),
        "posture": posture(z),
    }
    (run / "score.json").write_text(json.dumps({"summary": summary, "trials": results}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
