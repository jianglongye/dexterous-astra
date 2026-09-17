"""Learned one-hand roll: the left fingers roll the whole cube 90 degrees so the next face is up.

All 26 cubies are welded to the core during the roll (physics.set_roll_welds). A roll learned for one
(top, target) face pair serves every pair related by a cube symmetry G: the policy sees the cube
orientation relabelled by G. Quantities are expressed in the live wrist frame.
"""

import math
from pathlib import Path

import mujoco
import numpy as np
import torch

import physics
import scene

FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
SITES = [f"l_{n}_tip" for n in FINGERS]
FACE_NORMAL = {
    "U": (0.0, 0.0, 1.0),
    "R": (1.0, 0.0, 0.0),
    "F": (0.0, -1.0, 0.0),
    "D": (0.0, 0.0, -1.0),
    "L": (-1.0, 0.0, 0.0),
    "B": (0.0, 1.0, 0.0),
}  # cube core frame
CONTROL_DT = 0.02
RELEASE_STEPS = 3  # 60 ms without contact
REGRASP_DISTANCE = 0.01  # new contact at least 1 cm from the release point (cube frame)
SUCCESS_DEGREES = 10.0  # remaining angle between the target face and the entry "up" direction
HOLD_SECONDS = 1.0
MIN_REGRASPS = 2


def label_symmetry(train_top, train_target, top, target):
    """Rotation G (cube frame) with G n(train_top) = n(top) and G n(train_target) = n(target)."""
    for g in scene.GROUP:
        g = g.astype(float)
        if np.allclose(g @ np.array(scene.NORMAL[train_top], float), scene.NORMAL[top]) and np.allclose(
            g @ np.array(scene.NORMAL[train_target], float), scene.NORMAL[target]
        ):
            return g
    raise ValueError(f"No cube symmetry maps {train_top}->{top} and {train_target}->{target}")


def _axis_angle(axis, angle):
    k = torch.stack(
        (
            torch.zeros_like(axis[0]),
            -axis[2],
            axis[1],
            axis[2],
            torch.zeros_like(axis[0]),
            -axis[0],
            -axis[1],
            axis[0],
            torch.zeros_like(axis[0]),
        )
    ).reshape(3, 3)
    return torch.eye(3, device=axis.device) + torch.sin(angle) * k + (1 - torch.cos(angle)) * (k @ k)


def world_grip_orientations(face):
    def rot(axis, angle):
        c, s = math.cos(angle), math.sin(angle)
        x, y, z = axis
        return np.array(
            [
                [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
            ]
        )

    faceq = {
        "U": np.eye(3),
        "D": rot((1, 0, 0), math.pi),
        "R": rot((0, 1, 0), math.pi / 2),
        "L": rot((0, 1, 0), -math.pi / 2),
        "F": rot((1, 0, 0), math.pi / 2),
        "B": rot((1, 0, 0), -math.pi / 2),
    }
    return np.stack([(faceq[face] @ rot((0, 0, 1), k * math.pi / 2)).T for k in range(4)])


class Reference:
    """Entry-derived constants in the wrist frame."""

    def __init__(self, wrist_rot, wrist_pos, cube_rot, cube_pos, low, high, top, target, align):
        t = lambda v: torch.as_tensor(v, dtype=torch.float32)
        wrist_rot, cube_rot = t(wrist_rot), t(cube_rot)
        self.up = wrist_rot.T @ cube_rot @ t(FACE_NORMAL[top])  # direction to reach, wrist frame
        self.face = t(FACE_NORMAL[target])  # cube-frame normal to bring there
        self.home = wrist_rot.T @ (t(cube_pos) - t(wrist_pos))  # cube centre, wrist frame
        # Full-orientation goals: the minimal roll bringing the target face up, then
        # the four yaw symmetries about "up" (all give the same grip up to relabeling).
        entry = wrist_rot.T @ cube_rot
        start = entry @ self.face
        axis = torch.linalg.cross(start, self.up)
        roll = _axis_angle(axis / axis.norm(), torch.acos((start * self.up).sum().clamp(-1, 1)))
        self.goals = torch.stack([_axis_angle(self.up, torch.tensor(k * math.pi / 2)) @ roll @ entry for k in range(4)])
        if align == "world":
            # The scripted pickup's support grip is world-aligned.
            self.goals = wrist_rot.T @ t(world_grip_orientations(target))
        self.low, self.high = t(low), t(high)
        self.mid, self.half = (self.low + self.high) / 2, (self.high - self.low).clamp_min(0.01) / 2


def measure(ref, wrist_rot, wrist_pos, cube_rot, cube_pos, cube_lin, cube_ang_local, tips):
    """Batch geometry. Rotations (B,3,3), positions (B,3), tips (B,5,3) in world."""
    wt = wrist_rot.transpose(-1, -2)
    rel_rot = wt @ cube_rot
    rel_pos = (wt @ (cube_pos - wrist_pos).unsqueeze(-1)).squeeze(-1)
    normal = rel_rot @ ref.face
    angle = torch.acos((normal * ref.up).sum(-1).clamp(-1, 1))
    traces = torch.einsum("kij,bij->bk", ref.goals, rel_rot)
    aligned = torch.acos(((traces.max(-1).values - 1) / 2).clamp(-1, 1))
    lin = (wt @ cube_lin.unsqueeze(-1)).squeeze(-1)
    ang = (wt @ (cube_rot @ cube_ang_local.unsqueeze(-1))).squeeze(-1)
    tips_cube = (cube_rot.transpose(-1, -2).unsqueeze(1) @ (tips - cube_pos.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
    tips_wrist = (wt.unsqueeze(1) @ (tips - cube_pos.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
    return dict(
        rel_rot=rel_rot,
        offset=rel_pos - ref.home,
        normal=normal,
        angle=angle,
        aligned=aligned,
        lin=lin,
        ang=ang,
        tips_cube=tips_cube,
        tips_wrist=tips_wrist,
    )


def observe(ref, q, qd, targets, action, geo, flags, regrasps, elapsed, horizon):
    return torch.cat(
        (
            (q - ref.mid) / ref.half,
            (qd / 4).clamp(-5, 5),
            (targets - ref.mid) / ref.half,
            action,
            geo["offset"] / 0.03,
            geo["rel_rot"].flatten(1),
            geo["normal"],
            (geo["lin"] / 0.2).clamp(-5, 5),
            (geo["ang"] / 3).clamp(-5, 5),
            geo["tips_wrist"].flatten(1) / 0.06,
            flags,
            geo["angle"][:, None] / (math.pi / 2),
            (regrasps[:, None] / MIN_REGRASPS).clamp(max=2),
            elapsed[:, None] / horizon,
        ),
        -1,
    )


class Gaiting:
    """Counts release/reposition/recontact cycles per finger from contact flags and tip positions."""

    def __init__(self):
        self.last = torch.zeros((1, 5, 3))
        self.seen = torch.zeros((1, 5), dtype=torch.bool)
        self.off = torch.zeros((1, 5))
        self.count = torch.zeros(1)

    def update(self, contact, tips_cube):
        contact = contact > 0.5
        moved = (tips_cube - self.last).norm(dim=-1) >= REGRASP_DISTANCE
        event = contact & self.seen & (self.off >= RELEASE_STEPS) & moved
        self.last = torch.where(contact.unsqueeze(-1), tips_cube, self.last)
        self.seen |= contact
        self.off = torch.where(contact, torch.zeros_like(self.off), self.off + 1)
        self.count += event.sum(-1).float()


def execute(sim, cp, policy, top, target, relabel, after_step, seconds=6.0, unlock_hold=0.5):
    m, d = sim.m, sim.d
    args = cp["arguments"]
    scale = float(args["action_scale"])
    names = [m.actuator(int(a)).name for a in sim.hand_act]
    ids = np.array([i for i, n in enumerate(names) if n.startswith("l_")])
    joints = m.actuator_trnid[sim.hand_act[ids], 0]
    qa, va = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
    low, high = m.jnt_range[joints].T
    wrist, core = m.body("l_wrist").id, sim.core
    core_v = int(m.joint("cube/core").dofadr[0])
    sites = [m.site(n).id for n in SITES]
    finger_geom = np.array([physics.finger_channel(m.body(int(b)).name) for b in m.geom_bodyid])
    tip_geom = np.array([m.body(int(b)).name.endswith("tip_sensor_frame") for b in m.geom_bodyid])
    tips_only = bool(args.get("tips_only", False))
    free_layer = bool(args.get("free_layer", False))
    target_normal = np.array(FACE_NORMAL[target])
    palm_geom = m.geom_bodyid == wrist
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)

    start = float(d.time)
    start_facelets = sim.observe()["facelets"]
    locks = physics.set_roll_welds(sim, True)
    mujoco.mj_forward(m, d)
    wrist_rot0 = d.xmat[wrist].reshape(3, 3).copy()
    align = args.get("align", "none")
    ref = Reference(
        wrist_rot0,
        d.xpos[wrist],
        (d.xmat[core].reshape(3, 3) @ relabel),
        d.xpos[core],
        low,
        high,
        top,
        target,
        "world" if align == "world" else "entry",
    )
    targets = t(sim.hand_target[ids])[None].clone()
    entry_joints = d.qpos[qa].copy()
    if args.get("goal_grasp"):
        # Checkpoints store the training machine's path; the goal grasp ships beside them.
        goal = np.load(Path(__file__).resolve().parents[1] / "checkpoints" / Path(args["goal_grasp"]).name)
        entry_joints = np.asarray(goal["hand_qpos"], float)
        ref.home = t(np.asarray(goal["cube_pos_wrist"], float))
    return_grasp = bool(args.get("return_grasp", False))
    horizon = float(args.get("horizon", 6.0))
    seconds = max(seconds, horizon)
    action = torch.zeros((1, 20))
    gait = Gaiting()
    elapsed = torch.zeros(1)
    worst = dict(penetration_m=0.0, joint_violation_rad=0.0)
    rows = []
    hold, success, failure, airborne = 0.0, False, None, 0.0
    max_wrist, other_contacts = 0.0, set()

    def contact_flags():
        f = torch.zeros((1, 8))
        core_rot = d.xmat[core].reshape(3, 3) @ relabel
        for c in d.contact[: d.ncon]:
            if c.dist > 0:
                continue
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            for ga, bb in ((g1, b2), (g2, b1)):
                if bb in sim.cube_bodies:
                    if 0 <= finger_geom[ga] < 5:
                        f[0, finger_geom[ga]] = 1
                        if not tip_geom[ga]:
                            f[0, 6] = 1
                        gb = g2 if ga == g1 else g1
                        if (core_rot.T @ (d.geom_xpos[gb] - d.xpos[core])) @ target_normal > 0.5 * 0.019:
                            f[0, 7] = 1
                    if palm_geom[ga]:
                        f[0, 5] = 1
        return f

    for step in range(round(seconds / CONTROL_DT) + 1):
        mujoco.mj_forward(m, d)
        wrist_rot = d.xmat[wrist].reshape(3, 3)
        geo = measure(
            ref,
            t(wrist_rot)[None],
            t(d.xpos[wrist])[None],
            t((d.xmat[core].reshape(3, 3) @ relabel))[None],
            t(d.xpos[core])[None],
            t(d.qvel[core_v : core_v + 3])[None],
            t(d.qvel[core_v + 3 : core_v + 6])[None],
            t(d.site_xpos[sites])[None],
        )
        flags = contact_flags()
        if step:
            gait.update(flags[:, :5], geo["tips_cube"])
        others = {n for n in physics.cube_contacts(sim) if not n.startswith("l_")}
        other_contacts |= others
        delta = wrist_rot @ wrist_rot0.T
        max_wrist = max(max_wrist, math.degrees(math.acos(np.clip((np.trace(delta) - 1) / 2, -1, 1))))
        angle = float(geo["aligned"][0] if align != "none" else geo["angle"][0])
        offset = float(geo["offset"][0].norm())
        spin, speed = float(geo["ang"][0].norm()), float(geo["lin"][0].norm())
        airborne = 0.0 if flags[0, :6].sum() > 0 else airborne + CONTROL_DT
        joint_error = np.abs(d.qpos[qa] - entry_joints)[:16]
        deviation = float(joint_error.max() if args.get("max_deviation") else joint_error.mean())
        rows.append(
            dict(
                time=float(d.time) - start,
                angle_deg=math.degrees(angle),
                grasp_deviation_rad=deviation,
                offset_m=offset,
                regrasps=int(gait.count[0]),
            )
        )
        checks = dict(
            penetration=worst["penetration_m"] <= 0.001,
            joint_limits=worst["joint_violation_rad"] <= 0.05,
            wrist_under_10deg=max_wrist < 10.0,
            other_hand_and_pedestal_clear=not others,
            not_dropped=offset < 0.06 and airborne <= 0.2,
        )
        if not all(checks.values()):
            failure = [k for k, v in checks.items() if not v]
            break
        settled = (
            angle < math.radians(SUCCESS_DEGREES)
            and offset < 0.025
            and flags[0, 0] > 0.5
            and int(flags[0, :5].sum()) >= 3
            and spin < 0.5
            and speed < 0.05
            and gait.count[0] >= MIN_REGRASPS
        )
        if tips_only:
            settled = settled and flags[0, 6] < 0.5
        if free_layer:
            settled = settled and flags[0, 7] < 0.5
        if return_grasp:
            settled = (
                settled
                and deviation < float(args.get("grasp_tolerance", 0.12))
                and offset < float(args.get("position_tolerance", 0.008))
            )
        hold = hold + CONTROL_DT if settled and step else 0.0
        if hold >= HOLD_SECONDS - 1e-6:
            success = True
            break
        if step == round(seconds / CONTROL_DT):
            failure = ["timeout"]
            break
        obs = observe(
            ref,
            t(d.qpos[qa])[None],
            t(d.qvel[va])[None],
            targets,
            action,
            geo,
            flags[:, :6],
            gait.count,
            elapsed,
            horizon,
        )
        with torch.no_grad():
            action = policy.mean_action(obs).clamp(-1, 1)
        targets = (targets + scale * action).clamp(t(low), t(high))
        sim.hand_target[ids] = targets[0].numpy()
        sim.phase = "rl_single_hand_tumble"
        sim.step(20)
        elapsed += CONTROL_DT
        for k in worst:
            worst[k] = max(worst[k], sim.substep_worst[k])
        after_step()
    physics.set_roll_welds(sim, False)
    held = None
    if success and unlock_hold > 0:
        # Welds off; the cube must stay held and aligned.
        for _ in range(round(unlock_hold / CONTROL_DT)):
            sim.step(20)
            for k in worst:
                worst[k] = max(worst[k], sim.substep_worst[k])
            after_step()
        touching = physics.cube_contacts(sim)
        held = sim.held_in_hand()
        other_contacts |= {n for n in touching if not n.startswith("l_")}
    end_obs = sim.observe()
    physical = worst["penetration_m"] <= 0.001 and worst["joint_violation_rad"] <= 0.05
    return dict(
        success=bool(
            success
            and physical
            and not other_contacts
            and max_wrist < 10
            and held
            and end_obs["misalign_deg"] < 10
            and scene.legal(end_obs["facelets"])
        ),
        failure=failure,
        duration_s=float(d.time) - start,
        regrasps=int(gait.count[0]),
        max_wrist_rotation_deg=max_wrist,
        other_body_contacts=sorted(other_contacts),
        welds=len(locks),
        held_after_unweld=held,
        start_facelets=start_facelets,
        end_facelets=end_obs["facelets"],
        end_misalign_deg=end_obs["misalign_deg"],
        **worst,
    ), rows
