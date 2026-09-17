# /// script
# requires-python = ">=3.10"
# dependencies = ["mujoco==3.13.0", "numpy", "imageio[ffmpeg]"]
# ///
"""Run a controller on the scrambled cube and check the result.

    uv run run.py --controller controller.py --scramble "U'" --out runs/u1 [--seconds 30] [--video]

Writes <out>/report.json, traj.npz, frame_*.png, and video.mp4 (--video).
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import time
from pathlib import Path

import scene  # isort: skip (sets MUJOCO_GL before mujoco is imported)

import mujoco
import numpy as np
from arms import HAND_ACCEL, HAND_SPEED, ArmDrive, JointFilter, descendants

CONTROL_DT = 0.01  # controller runs at 100 Hz
RECORD_FPS = 50
DETENT = 0.02  # N*m/rad passive spring pulling each face centre to the nearest quarter turn
MAX_WRIST_SPEED = (0.5, 3.0)  # m/s, rad/s: wrist targets are rate-limited like an arm would be
LIMITS = {"penetration_m": 1e-3, "joint_violation_rad": 0.05, "misalign_deg": 10.0}
HOLD_S = 1.0  # the solved, aligned state must hold this long


def load_controller(path, info):
    spec = importlib.util.spec_from_file_location("controller", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Controller(info)


class Sim:
    def __init__(self, scramble, out, embodiment="arms"):
        self.embodiment = embodiment
        spec = scene.build_spec(embodiment=embodiment)
        self.m = m = spec.compile()
        self.d = mujoco.MjData(m)
        mujoco.mj_saveModel(m, str(out / "scene.mjb"), None)  # exact copy for the controller's own FK/IK
        scene.apply_scramble(m, self.d, scramble)
        self.core = m.body("cube/core").id
        self.cube_bodies = descendants(m, "cube/core")
        self.hand_bodies = descendants(m, "l_wrist") | descendants(m, "r_wrist")
        self.robot_bodies = descendants(m, "robot_base") if embodiment == "arms" else self.hand_bodies
        self.robot_geom = np.isin(m.geom_bodyid, list(self.robot_bodies))
        self.hand_act = np.array([a for a in range(m.nu) if m.jnt_bodyid[m.actuator_trnid[a, 0]] in self.hand_bodies])
        self.pedestal = m.geom("pedestal").id
        self.act_qadr = m.jnt_qposadr[m.actuator_trnid[self.hand_act, 0]]
        self.hand_dofs = m.jnt_dofadr[m.actuator_trnid[self.hand_act, 0]]
        self.hand_joints = [j for j in range(m.njnt) if m.jnt_bodyid[j] in self.robot_bodies and m.jnt_limited[j]]
        self.centres = [m.joint(f"cube/{n}") for n in scene.CENTRES]
        self.mocap = (
            {s: m.body_mocapid[m.body(f"{s}_wrist_target").id] for s in "lr"} if embodiment == "floating" else {}
        )
        self.arms = ArmDrive(m, self.d) if embodiment == "arms" else None
        self.fingers = JointFilter(self.d.qpos[self.act_qadr], HAND_SPEED, HAND_ACCEL, frequency=40.0)
        self.hand_target = self.d.qpos[self.act_qadr].copy()
        self.phase = "initial"
        self.substep_worst = {"penetration_m": 0.0, "joint_violation_rad": 0.0}
        if self.arms:
            # Legacy geometric finger planners use independent hand poses in a
            # private model. Only the actuated arm scene is physically executed.
            planning = scene.build_spec(embodiment="floating").compile()
            mujoco.mj_saveModel(planning, str(out / "planning.mjb"), None)
        self.tips = [m.site(i).name for i in range(m.nsite) if m.site(i).name.endswith("_tip")]
        self.cube_actuators = int(sum(m.jnt_bodyid[m.actuator_trnid[a, 0]] in self.cube_bodies for a in range(m.nu)))

    def observe(self):
        m, d = self.m, self.d
        if self.arms:
            # mj_step integrates qpos after computing its Cartesian fields.
            # Controllers need joints, wrists, cubies, and contacts from the
            # same instant when reconstructing fingertip force Jacobians.
            mujoco.mj_forward(m, d)
        rots, _ = scene.cubie_rotations(m, d)
        facelets, misalign = scene.facelets(m, d)
        contacts = []
        for (g1, g2), dist in zip(d.contact.geom, d.contact.dist):
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            if b2 in self.hand_bodies:
                b1, b2 = b2, b1
            if b1 in self.hand_bodies and b2 not in self.hand_bodies:
                contacts.append((m.body(b1).name, m.body(b2).name, float(dist)))
        observation = {
            "time": float(d.time),
            "hand_qpos": d.qpos[self.act_qadr].copy(),  # actuator order
            "wrist": {
                s: (d.xpos[m.body(f"{s}_wrist").id].copy(), d.xquat[m.body(f"{s}_wrist").id].copy()) for s in "lr"
            },
            "tips": {n: d.site(n).xpos.copy() for n in self.tips},
            "cube_pos": d.xpos[self.core].copy(),
            "cube_quat": d.xquat[self.core].copy(),
            "cube_vel": d.qvel[m.jnt_dofadr[m.body_jntadr[self.core]] :][:6].copy(),
            "on_pedestal": self.on_pedestal(),
            "face_angles": {n.name[-2:]: float(d.qpos[n.qposadr[0]]) for n in self.centres},
            "cubie_rot": rots,  # 26 x 3 x 3, relative to the cube core
            "facelets": facelets,
            "misalign_deg": misalign,
            "contacts": contacts,  # (hand body, other body, signed distance)
            "arm_qpos": {s: d.qpos[a].copy() for s, a in self.arms.qadr.items()} if self.arms else {},
            "arm_qvel": {s: d.qvel[a].copy() for s, a in self.arms.dofs.items()} if self.arms else {},
            "wrist_error": {s: e.copy() for s, e in self.arms.error.items()} if self.arms else {},
            "arm_collision_blocked": self.arms.blocked if self.arms else False,
            "arm_collision_pairs": self.arms.collision_pairs if self.arms else [],
            "arm_path_pending": {s: bool(p) for s, p in self.arms.paths.items()} if self.arms else {},
        }
        if os.environ.get("RUBIKS_CONTACT_WRENCH", "0") == "1":
            observation["cube_contact_wrenches"] = self.cube_contact_wrenches()
        return observation

    def cube_contact_wrenches(self):
        """External contact wrenches on the cube, in world axes about its core."""
        m, d = self.m, self.d
        result = {key: np.zeros(6) for key in ["l", "r", "environment"]}
        for index, contact in enumerate(d.contact):
            bodies = m.geom_bodyid[contact.geom]
            inside = [body in self.cube_bodies for body in bodies]
            if sum(inside) != 1:
                continue
            cube_side = inside.index(True)
            name = m.body(bodies[1 - cube_side]).name
            key = name[0] if name.startswith(("l_", "r_")) else "environment"
            local = np.zeros(6)
            mujoco.mj_contactForce(m, d, index, local)
            # Contact-frame force acts on geom 2; its axes are matrix rows.
            # https://mujoco.readthedocs.io/en/stable/computation/index.html#contact
            rotation = contact.frame.reshape(3, 3).T
            force = rotation @ local[:3] * (1 if cube_side == 1 else -1)
            torque = rotation @ local[3:] * (1 if cube_side == 1 else -1)
            torque += np.cross(contact.pos - d.xpos[self.core], force)
            result[key] += np.r_[force, torque]
        return result

    def apply(self, cmd, dt):
        m, d = self.m, self.d
        lo, hi = m.actuator_ctrlrange[self.hand_act].T
        hand = np.asarray(cmd["hand"], float).reshape(len(self.hand_act))
        goals = {}
        for side, (pos, quat) in cmd.get("wrist", {}).items():
            pos, quat = np.asarray(pos, float).reshape(3), np.asarray(quat, float).reshape(4)
            goals[side] = (pos, quat / np.linalg.norm(quat))
        self.hand_target = np.clip(hand, lo, hi)
        self.phase = str(cmd.get("phase", ""))
        if self.arms:
            self.arms.allow_carry_replan = bool(cmd.get("carry_replan", False))
            self.arms.plan_targets(d, goals, dt)
            return
        d.ctrl[self.hand_act] = self.hand_target
        for s, (pos, quat) in goals.items():
            i = self.mocap[s]
            step = np.subtract(pos, d.mocap_pos[i])
            d.mocap_pos[i] += step * min(1.0, MAX_WRIST_SPEED[0] * dt / max(np.linalg.norm(step), 1e-12))
            q = np.asarray(quat, float) / np.linalg.norm(quat)
            dq = np.zeros(3)
            mujoco.mju_subQuat(dq, q, d.mocap_quat[i])
            ang = np.linalg.norm(dq)
            mujoco.mju_quatIntegrate(d.mocap_quat[i], dq, min(1.0, MAX_WRIST_SPEED[1] * dt / max(ang, 1e-12)))
        scene.apply_detents(m, d, DETENT)

    def step(self, nsub):
        m, d = self.m, self.d
        if not self.arms:
            # Preserve the integration schedule of historical floating runs.
            mujoco.mj_step(m, d, nstep=nsub)
            self.substep_worst = {"penetration_m": self.penetration(), "joint_violation_rad": self.joint_violation()}
            return
        self.substep_worst = {"penetration_m": 0.0, "joint_violation_rad": 0.0}
        for _ in range(nsub):
            if self.arms:
                lo, hi = m.actuator_ctrlrange[self.hand_act].T
                target = self.fingers.step(self.hand_target, m.opt.timestep)
                target += d.qfrc_bias[self.hand_dofs] / m.actuator_gainprm[self.hand_act, 0]
                d.ctrl[self.hand_act] = np.clip(target, lo, hi)
                self.arms.actuate(d)
            scene.apply_detents(m, d, DETENT)
            mujoco.mj_step(m, d)
            self.substep_worst["penetration_m"] = max(self.substep_worst["penetration_m"], self.penetration())
            self.substep_worst["joint_violation_rad"] = max(
                self.substep_worst["joint_violation_rad"], self.joint_violation()
            )

    def controller_info(self, scramble, out, seconds):
        m = self.m
        out = out.resolve()
        return {
            "model_path": str(out / "scene.mjb"),
            "planning_model_path": str(out / ("planning.mjb" if self.arms else "scene.mjb")),
            "embodiment": self.embodiment,
            "scramble": scramble,
            "solution": scene.invert(scramble),
            "control_dt": CONTROL_DT,
            "seconds": seconds,
            "actuators": [m.actuator(i).name for i in self.hand_act],
            "ctrlrange": m.actuator_ctrlrange[self.hand_act].copy(),
            "wrist_init": {s: tuple(map(np.array, scene.WRISTS[s])) for s in "lr"},
            "max_wrist_speed": MAX_WRIST_SPEED,
            "pitch": scene.PITCH,
        }

    def penetration(self):
        geom, dist = self.d.contact.geom, self.d.contact.dist
        robot = self.robot_geom[geom[:, 0]] | self.robot_geom[geom[:, 1]]
        return float(max(0.0, -dist[robot].min())) if robot.any() else 0.0

    def on_pedestal(self):
        geom = self.d.contact.geom
        other = np.where(geom[:, 0] == self.pedestal, geom[:, 1], np.where(geom[:, 1] == self.pedestal, geom[:, 0], -1))
        return bool(np.isin(self.m.geom_bodyid[other[other >= 0]], list(self.cube_bodies)).any())

    def held_in_hand(self):
        for contact in self.d.contact:
            b1, b2 = self.m.geom_bodyid[contact.geom]
            if contact.dist <= 0 and (
                (b1 in self.hand_bodies and b2 in self.cube_bodies)
                or (b2 in self.hand_bodies and b1 in self.cube_bodies)
            ):
                return True

    def joint_violation(self):
        m, d = self.m, self.d
        q = d.qpos[m.jnt_qposadr[self.hand_joints]]
        lo, hi = m.jnt_range[self.hand_joints].T
        return float(max(0.0, (lo - q).max(), (q - hi).max()))


def render(sim, traj, out, video, n_frames=6, camera="overview", fps=RECORD_FPS):
    import imageio

    m, d = sim.m, mujoco.MjData(sim.m)
    picks = set(np.linspace(0, len(traj) - 1, n_frames).astype(int))
    writer = imageio.get_writer(out / "video.mp4", fps=fps, ffmpeg_params=["-threads", "1"]) if video else None
    detail_camera = mujoco.MjvCamera()
    detail_camera.distance, detail_camera.azimuth, detail_camera.elevation = 0.40, 225, -25
    with mujoco.Renderer(m, 480, 640) as r:
        for k, q in enumerate(traj):
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            view = camera
            r.update_scene(d, view, scene.VISUAL)
            img = r.render()
            if k in picks:
                imageio.imwrite(out / f"frame_{k / fps:06.2f}s.png", img)
            if writer is not None:
                writer.append_data(img)
    if writer is not None:
        writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", required=True)
    ap.add_argument("--scramble", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--embodiment", choices=("arms", "floating"), default="arms")
    ap.add_argument("--camera", choices=("overview", "closeup"), default="overview")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    sim = Sim(a.scramble, out, a.embodiment)
    m, d = sim.m, sim.d
    info = sim.controller_info(a.scramble, out, a.seconds)
    ctrl = load_controller(a.controller, info)
    sources = list(Path(__file__).parent.glob("*.py")) + [Path(a.controller)]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}

    nsub = round(CONTROL_DT / m.opt.timestep)
    rec_every = round(1 / (RECORD_FPS * CONTROL_DT))
    traj, timeline = [d.qpos.copy()], []
    phases = [sim.phase]
    worst = {"penetration_m": 0.0, "joint_violation_rad": 0.0}
    prev_rots, continuous, intact, solved_since, reason, error = None, True, True, None, "timeout", None
    _, home = scene.cubies(m)
    t0 = time.time()
    for k in range(round(a.seconds / CONTROL_DT)):
        obs = sim.observe()
        rots = obs["cubie_rot"]
        if prev_rots is not None:  # nothing may teleport the cube between control steps
            jump = np.arccos(np.clip((np.einsum("nij,nij->n", prev_rots, rots) - 1) / 2, -1, 1)).max()
            continuous &= bool(jump < 0.5)
        prev_rots = rots
        if obs["misalign_deg"] < 20:  # when aligned: every cubie in its own slot, and no twisted/flipped cubie
            slots = np.einsum("nij,nj->ni", scene.snap(rots)[0], home)
            intact &= len(np.unique(slots, axis=0)) == len(home) and scene.legal(obs["facelets"])
        if k % 50 == 0:
            timeline.append(
                {
                    "t": round(obs["time"], 2),
                    "facelets": obs["facelets"],
                    "misalign_deg": round(obs["misalign_deg"], 2),
                    "cube_pos": obs["cube_pos"].round(4).tolist(),
                    "contacts": sorted({c[0] for c in obs["contacts"]}),
                    "wrist_error": {s: e.tolist() for s, e in obs["wrist_error"].items()},
                    "arm_collision_pairs": obs["arm_collision_pairs"],
                }
            )
            timeline[-1]["phase"] = sim.phase
        solved = obs["facelets"] == scene.SOLVED and obs["misalign_deg"] < LIMITS["misalign_deg"]
        solved_since = (solved_since if solved_since is not None else obs["time"]) if solved else None
        try:
            sim.apply(ctrl.act(obs["time"], obs), CONTROL_DT)
        except Exception as e:  # noqa: BLE001
            reason, error = "controller_error", f"{type(e).__name__}: {e}"
            break
        sim.step(nsub)
        for metric, value in sim.substep_worst.items():
            worst[metric] = max(worst[metric], value)
        if (k + 1) % rec_every == 0:
            traj.append(d.qpos.copy())
            phases.append(sim.phase)
    wall = time.time() - t0

    final = sim.observe()
    checks = {
        "solved": reason == "solved",
        "continuous": continuous,
        "cube_intact": bool(intact),
        "cube_actuators_zero": sim.cube_actuators == 0,
        "penetration_ok": worst["penetration_m"] <= LIMITS["penetration_m"],
        "joint_limits_ok": worst["joint_violation_rad"] <= LIMITS["joint_violation_rad"],
    }
    report = {
        "passed": all(checks.values()),
        "reason": reason,
        "error": error,
        "checks": checks,
        "scramble": a.scramble,
        "solution": info["solution"],
        "sim_time": round(float(d.time), 3),
        "wall_time": round(wall, 1),
        "physics_steps": round(d.time / m.opt.timestep),
        "state_resets": 0,
        "cube_actuators": sim.cube_actuators,
        "embodiment": a.embodiment,
        "arm_collision_blocked": sim.arms.blocked if sim.arms else False,
        "arm_collision_pairs": sim.arms.collision_pairs if sim.arms else [],
        "wrist_error": {s: e.tolist() for s, e in sim.arms.error.items()} if sim.arms else {},
        "wrist_goals": {s: [p.tolist(), q.tolist()] for s, (p, q) in sim.arms.goals.items()} if sim.arms else {},
        **worst,
        "limits": LIMITS,
        "final": {
            "facelets": final["facelets"],
            "misalign_deg": round(final["misalign_deg"], 2),
            "cube_pos": final["cube_pos"].round(4).tolist(),
            "on_pedestal": sim.on_pedestal(),
            "hand_contacts": sorted({c[0] for c in final["contacts"]}),
        },
        "timeline": timeline,
        "environment": {"host": platform.node(), "mujoco": mujoco.__version__, "python": platform.python_version()},
        "source_sha256": source_hashes,
        "controller_settings": {k: v for k, v in os.environ.items() if k.startswith("RUBIKS_")},
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    np.savez_compressed(out / "traj.npz", qpos=np.array(traj), fps=RECORD_FPS, phase=np.array(phases))
    if not a.no_render:
        render(sim, traj, out, a.video, camera=a.camera)
    print(json.dumps({k: report[k] for k in ("passed", "reason", "error", "checks", "sim_time", "wall_time")}))


if __name__ == "__main__":
    main()
