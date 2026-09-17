"""Render a recorded policy solve: overview plus cube close-up, labelled by which controller acts.

python sim/render.py --model OUT/scene.mjb --trajectory OUT/trajectory.npz --out OUT/video.mp4
"""

import argparse
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import scene


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def controller_label(phase):
    phase = str(phase)
    if phase == "rl_one_hand_face_turn":
        return "RL: one-hand face turn (left fingers only)", (0, 110, 40)
    if phase == "rl_single_hand_tumble":
        return "RL: one-hand roll to the next face (left fingers only)", (0, 110, 40)
    if phase.startswith("one_hand"):
        return "scripted: hold current finger posture", (90, 90, 90)
    return f"scripted pickup (left hand): {phase}", (90, 90, 90)


def render(model_path, trajectory_path, out, fps=50, width=640, height=480):
    m = mujoco.MjModel.from_binary_path(str(model_path))
    d = mujoco.MjData(m)
    trace = np.load(trajectory_path)
    qpos, times, phases = trace["qpos"], trace["time"], trace["phase"]
    core = m.body("cube/core").id
    big, small = font(18), font(14)
    writer = imageio.get_writer(
        out, fps=fps, codec="libx264", quality=8, macro_block_size=1, ffmpeg_params=["-threads", "2"]
    )
    with mujoco.Renderer(m, height, width) as r:
        for i in range(len(qpos)):
            d.qpos[:] = qpos[i]
            mujoco.mj_forward(m, d)
            r.update_scene(d, "overview", scene.VISUAL)
            panels = [r.render()]
            cam = mujoco.MjvCamera()
            cam.distance, cam.azimuth, cam.elevation = 0.26, 240, -30
            cam.lookat[:] = d.xpos[core]
            r.update_scene(d, cam, scene.VISUAL)
            panels.append(r.render())
            image = Image.fromarray(np.concatenate(panels, 1))
            draw = ImageDraw.Draw(image)
            text, color = controller_label(phases[i])
            draw.text((12, 10), text, fill=color, font=big)
            draw.text((12, 34), f"t = {times[i]:6.2f} s (simulation time)", fill=(40, 40, 40), font=small)
            writer.append_data(np.asarray(image))
    writer.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fps", type=int, default=50)
    a = p.parse_args()
    render(a.model, a.trajectory, a.out, a.fps)


if __name__ == "__main__":
    main()
