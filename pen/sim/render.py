"""Render a recorded Isaac Lab trial at normal speed (60 fps, one frame per control step), uncut.

Hand joint angles and the pen pose come from the Isaac Lab / PhysX recording; MuJoCo only draws them
with the official Sharpa meshes. The overlay shows time, net turns, and per-finger pen contact.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from mj_scene import Index, build  # noqa: E402

SIZE = 720
VIEWS = {
    "top": dict(lookat=(0.14, 0.0, 0.56), distance=0.30, azimuth=180, elevation=-89),
    "close": dict(lookat=(0.14, 0.0, 0.575), distance=0.22, azimuth=215, elevation=-28),
}
FONT = None
for f in (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
):
    if Path(f).exists():
        FONT = f
GATE_CHECKS = (
    "turns_ok",
    "timing_ok",
    "hold_ok",
    "limits_ok",
    "penetration_ok",
    "participation_ok",
    "all_fingers_ok",
    "thumb_release_ok",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--views", nargs="*", default=["close"], choices=list(VIEWS))
    ap.add_argument("--title", default="PPO policy")
    a = ap.parse_args()
    run = Path(a.run)
    z = np.load(run / "trajectories.npz")
    meta = json.loads((run / "meta.json").read_text())
    score = json.loads((run / "score.json").read_text()) if (run / "score.json").exists() else None
    i = a.trial
    m = build()
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, SIZE)
    m.vis.global_.offheight = max(m.vis.global_.offheight, SIZE)
    d = mujoco.MjData(m)
    ix = Index(m)
    r = mujoco.Renderer(m, SIZE, SIZE)
    T = z["q"].shape[1]
    axis = z["pen_axis"][i]
    heading = np.unwrap(np.arctan2(axis[:, 1], axis[:, 0]))
    turns = (heading - z["heading0"][i]) / (2 * np.pi)
    force = z["finger_force"][i]
    font = ImageFont.truetype(FONT, SIZE // 34) if FONT else ImageFont.load_default()
    small = ImageFont.truetype(FONT, SIZE // 44) if FONT else ImageFont.load_default()
    verdict = ""
    if score:
        t = score["trials"][i]
        verdict = (
            "numerical gate: PASS"
            if t["passed"]
            else "numerical gate: FAIL ("
            + ", ".join(k for k in GATE_CHECKS if not t.get(k))
            + ("drop" if t.get("drop") else "")
            + ")"
        )
    out_dir = run / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    for view in a.views:
        path = out_dir / f"trial{i:03d}_seed{meta['seed0'] + i}_{view}.mp4"
        writer = imageio.get_writer(
            path, fps=round(1 / meta["control_dt"]), codec="libx264", quality=8, macro_block_size=1
        )
        cam = mujoco.MjvCamera()
        for k, v in VIEWS[view].items():
            setattr(cam, k, v) if k != "lookat" else cam.lookat.__setitem__(slice(None), v)
        for s in range(-1, T):
            if s < 0:
                q, p, qt, fr, tr = z["q0"][i], z["pen_pos0"][i], z["pen_quat0"][i], np.zeros(5), 0.0
            else:
                q, p, qt, fr, tr = z["q"][i, s], z["pen_pos"][i, s], z["pen_quat"][i, s], force[s], turns[s]
            ix.set_state(d, q, p, qt)
            mujoco.mj_kinematics(m, d)
            r.update_scene(d, cam)
            img = Image.fromarray(r.render())
            dr = ImageDraw.Draw(img)
            t = (s + 1) * meta["control_dt"]
            dr.text(
                (12, 8), f"Sharpa Wave | {a.title} | t={t:5.2f}s | net turns={tr:+.2f}", font=font, fill=(255, 255, 255)
            )
            dr.text(
                (12, 8 + SIZE // 28),
                f"Isaac Lab PhysX states drawn in MuJoCo | nominal physics (g=9.81) | seed {meta['seed0'] + i}",
                font=small,
                fill=(200, 200, 200),
            )
            x = 12
            for name, f in zip("TIMRP", fr):
                on = f > 0.01
                dr.rectangle([x, SIZE - 40, x + 30, SIZE - 12], fill=(40, 200, 90) if on else (70, 70, 70))
                dr.text((x + 9, SIZE - 38), name, font=small, fill=(0, 0, 0) if on else (220, 220, 220))
                x += 36
            dr.text((x + 8, SIZE - 38), "pen contact per finger", font=small, fill=(200, 200, 200))
            if verdict:
                dr.text((12, SIZE - 40 - SIZE // 24), verdict, font=small, fill=(255, 220, 120))
            writer.append_data(np.asarray(img))
        writer.close()
        print(path)


if __name__ == "__main__":
    main()
