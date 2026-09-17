"""Frozen numerical acceptance gate for recorded trials (numpy only).

Thresholds live in acceptance.json and were fixed before any policy was trained or evaluated.
Penetration comes from an independent MuJoCo convex-hull replay of the recorded states (score.py).
"""

import json
from pathlib import Path

import numpy as np

A = json.loads(Path(__file__).with_name("acceptance.json").read_text())
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _runs(mask):
    """(start, end_exclusive) index pairs of True runs."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def check_trial(rec, finger_joint_slices, q_lo, q_hi, penetration_mm):
    """rec: dict of per-step arrays for one trial (T = steps). Returns (passed, details)."""
    a = A
    dt = 1.0 / a["control_hz"]
    T = len(rec["pen_pos"])
    t = np.arange(1, T + 1) * dt
    axis = rec["pen_axis"]
    heading = np.unwrap(np.arctan2(axis[:, 1], axis[:, 0]))
    progress = a["spin_sign"] * (heading - rec["heading0"])
    progress = np.concatenate([[0.0], progress])  # sample 0 = settled state before the first action
    t = np.concatenate([[0.0], t])
    det = {}
    turn_t = []
    for k in range(1, a["turns"] + 1):
        hit = np.flatnonzero(progress >= 2 * np.pi * k)
        turn_t.append(float(t[hit[0]]) if len(hit) else None)
    det["turn_times"] = turn_t
    ok_turns = all(x is not None for x in turn_t)
    ok_timing = ok_turns and all(
        (turn_t[i] - (turn_t[i - 1] if i else 0.0)) <= a["max_turn_s"] + 1e-9 for i in range(a["turns"])
    )
    det["turns_ok"], det["timing_ok"] = ok_turns, ok_timing
    det["net_turns"] = float(progress.max() / (2 * np.pi))

    force = np.concatenate([np.zeros((1, 5)), rec["finger_force"]])
    touch = force > a["contact_n"]
    omega = np.abs(np.concatenate([[0.0], rec["pen_angvel"][:, 2]]))
    rel = np.concatenate([[np.zeros(3)], rec["pen_pos"] - rec["ref_pos"]])
    tilt = np.degrees(np.arcsin(np.clip(np.abs(np.concatenate([[0.0], axis[:, 2]])), 0, 1)))
    dropped = (
        (rel[:, 2] < -a["drop_z"]) | (np.linalg.norm(rel[:, :2], axis=1) > a["drop_xy"]) | (tilt > a["max_tilt_deg"])
    )

    # hold: a window of hold_s that starts within max_stop_s after the last turn
    hold_ok, t_end = False, T
    if ok_turns:
        i3 = int(np.flatnonzero(t >= turn_t[-1])[0])
        n_hold = int(round(a["hold_s"] / dt))
        for s in range(i3, min(i3 + int(round(a["max_stop_s"] / dt)) + 1, len(t) - n_hold + 1)):
            w = slice(s, s + n_hold)
            if (
                (omega[w] < np.radians(a["hold_max_omega_deg_s"])).all()
                and not dropped[w].any()
                and touch[w].any(axis=1).mean() >= a["hold_min_support_frac"]
                and np.ptp(progress[w]) < np.radians(a["hold_max_heading_change_deg"])
            ):
                hold_ok, t_end = True, s + n_hold
                break
    det["hold_ok"] = hold_ok
    span = slice(0, t_end)
    det["drop"] = bool(dropped[span].any())

    q = rec["q"][: max(t_end - 1, 1)]
    viol = np.maximum(q_lo - q, q - q_hi).max() if len(q) else 0.0
    det["max_limit_violation_rad"] = float(viol)
    det["limits_ok"] = bool(viol <= a["joint_limit_tol_rad"])

    det["max_penetration_mm"] = float(np.max(penetration_mm[:t_end]))
    det["penetration_ok"] = det["max_penetration_mm"] <= a["max_penetration_mm"]

    # participation per revolution
    qd = np.concatenate([np.zeros((1, rec["qd"].shape[1])), rec["qd"]])
    fwd = np.concatenate([[False], np.diff(progress) > 0])
    part_ok = ok_turns
    det["participation"] = []
    if ok_turns:
        bounds = [0.0] + turn_t
        for k in range(a["turns"]):
            w = (t > bounds[k]) & (t <= bounds[k + 1])
            active_fingers = []
            for fi, f in enumerate(FINGERS):
                if f not in a["participation_fingers"]:
                    continue
                speed = np.sqrt((qd[:, finger_joint_slices[f]] ** 2).mean(axis=1))
                active = touch[:, fi] & (speed > a["active_joint_speed_rad_s"]) & fwd & w
                if active.sum() * dt >= a["participation_min_s"]:
                    active_fingers.append(f)
            det["participation"].append(active_fingers)
            part_ok = part_ok and len(active_fingers) >= a["participation_min_fingers"]
    det["participation_ok"] = bool(part_ok)

    # added check (stricter): every non-thumb finger takes part in most revolutions
    extra = a.get("added_checks", {}).get("all_fingers")
    all_ok = True
    if extra is not None:
        all_ok = ok_turns
        det["all_fingers_revolutions"] = {}
        if ok_turns:
            bounds = [0.0] + turn_t
            for fi, f in enumerate(FINGERS):
                if f not in extra["fingers"]:
                    continue
                speed = np.sqrt((qd[:, finger_joint_slices[f]] ** 2).mean(axis=1))
                active = touch[:, fi] & (speed > a["active_joint_speed_rad_s"]) & fwd
                n = sum(
                    int(active[(t > bounds[k]) & (t <= bounds[k + 1])].sum() * dt >= extra["min_active_s"])
                    for k in range(a["turns"])
                )
                det["all_fingers_revolutions"][f] = n
                all_ok = all_ok and n >= extra["min_revolutions"]
    det["all_fingers_ok"] = bool(all_ok)

    # thumb release interval
    thumb_ok = False
    best = 0.0
    if ok_turns:
        end = int(np.flatnonzero(t >= turn_t[-1])[0]) + 1
        free = ~touch[:end, 0]
        for s, e in _runs(free):
            if (e - s) * dt < a["thumb_free_min_s"]:
                continue
            others = touch[s:e, 1:].sum(axis=1) >= a["thumb_free_min_other_fingers"]
            gain = progress[e - 1] - progress[s]
            best = max(best, float(np.degrees(gain)))
            if (
                others.mean() >= a["thumb_free_support_frac"]
                and np.degrees(gain) >= a["thumb_free_min_rotation_deg"]
                and not dropped[s:e].any()
            ):
                thumb_ok = True
    det["thumb_release_ok"] = thumb_ok
    det["best_thumb_free_rotation_deg"] = best

    passed = (
        ok_turns
        and ok_timing
        and hold_ok
        and not det["drop"]
        and det["limits_ok"]
        and part_ok
        and all_ok
        and thumb_ok
        and det["penetration_ok"]
    )
    return bool(passed), det
