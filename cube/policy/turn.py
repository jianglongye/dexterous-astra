"""Learned one-hand face turn: the left fingers turn the top layer 90 degrees while holding the rest.

Only the 20 left-hand finger targets come from the policy (50 Hz); arms and the right hand keep their
targets. During the turn the other layers are welded to the core and the turning layer to its centre
hinge (physics.set_turn_welds). Success needs >= 2 pushes: a push is a finger contact episode during
which the layer advances >= 8 degrees; an episode ends after 60 ms without contact.
"""

import math

import mujoco
import numpy as np
import torch

import physics
import scene

CONTROL_DT = 0.02
RELEASE_STEPS = 3
SUCCESS_DEGREES = 6.0
HOLD_SECONDS = 1.0
MIN_REGRASPS = 2
MAX_LOWER_ROTATION_DEG = 12.0
MAX_LOWER_OFFSET_M = 0.012
PUSH_DEGREES = 8.0
TOP_NORMAL = np.array([-1.0, 0.0, 0.0])  # the layer the policy was trained on, in cube coordinates
TIPS = [f"l_{n}_tip" for n in ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")]


def up_face(sim):
    core = sim.d.xmat[sim.core].reshape(3, 3)
    up = sim.d.xmat[sim.m.body("l_wrist").id].reshape(3, 3)[:, 0]
    return max(scene.FACES, key=lambda f: float(up @ (core @ np.array(scene.NORMAL[f], float))))


def up_symmetry(face):
    """Rotation G (cube frame) with G @ TOP_NORMAL = NORMAL[face]: relabel `face` as the trained top layer."""
    target = np.array(scene.NORMAL[face], float)
    return min(scene.GROUP, key=lambda g: np.linalg.norm(g @ TOP_NORMAL - target)).astype(float)


def rotation_angle(r):
    return torch.acos(((r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))


def measure(
    wrist_rot,
    wrist_pos,
    lower_rot,
    lower_pos,
    home_rot,
    home_pos,
    hinge,
    hinge_vel,
    target,
    lower_lin,
    lower_ang_local,
    tips,
):
    wt = wrist_rot.transpose(-1, -2)
    rel_rot = wt @ lower_rot
    rel_pos = (wt @ (lower_pos - wrist_pos).unsqueeze(-1)).squeeze(-1)
    lin = (wt @ lower_lin.unsqueeze(-1)).squeeze(-1)
    ang = (wt @ (lower_rot @ lower_ang_local.unsqueeze(-1))).squeeze(-1)
    tips_wrist = (wt.unsqueeze(1) @ (tips - lower_pos.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
    return dict(
        rel_rot=rel_rot,
        offset=rel_pos - home_pos,
        lower_rotation=rotation_angle(home_rot.transpose(-1, -2) @ rel_rot),
        error=hinge - target,
        hinge=hinge,
        hinge_vel=hinge_vel,
        lin=lin,
        ang=ang,
        tips_wrist=tips_wrist,
    )


def observe(mid, half, q, qd, targets, action, geo, command, flags, pushes, elapsed, horizon):
    return torch.cat(
        (
            (q - mid) / half,
            (qd / 4).clamp(-5, 5),
            (targets - mid) / half,
            action,
            geo["offset"] / 0.03,
            geo["rel_rot"].flatten(1),
            command[:, None],
            torch.sin(geo["hinge"])[:, None],
            torch.cos(geo["hinge"])[:, None],
            (geo["error"] / (math.pi / 2))[:, None].clamp(-2, 2),
            (geo["hinge_vel"] / 3)[:, None].clamp(-5, 5),
            (geo["lin"] / 0.2).clamp(-5, 5),
            (geo["ang"] / 3).clamp(-5, 5),
            geo["tips_wrist"].flatten(1) / 0.06,
            flags,
            (pushes[:, None] / MIN_REGRASPS).clamp(max=3),
            elapsed[:, None] / horizon,
        ),
        -1,
    )


class Pushes:
    """Counts pushes per finger (see module docstring)."""

    def __init__(self):
        self.off = torch.full((1, 5), float(RELEASE_STEPS))
        self.start = torch.zeros((1, 5))
        self.counted = torch.zeros((1, 5), dtype=torch.bool)
        self.count = torch.zeros(1)

    def update(self, contact, progress):
        """contact (1,5) layer contact flags; progress (1,) signed layer angle toward the target (rad)."""
        contact = contact > 0.5
        began = contact & (self.off >= RELEASE_STEPS)
        self.start = torch.where(began, progress[:, None].expand_as(self.start), self.start)
        self.counted &= ~began
        advanced = (progress[:, None] - self.start) >= math.radians(PUSH_DEGREES)
        event = contact & advanced & ~self.counted
        self.counted |= event
        self.off = torch.where(contact, torch.zeros_like(self.off), self.off + 1)
        self.count += event.sum(-1).float()
        return event


def execute(sim, cp, policy, command, after_step, internal_collisions=True, unweld_hold=0.5):
    m, d = sim.m, sim.d
    args = cp["arguments"]
    scale = float(args["action_scale"])
    horizon = float(args.get("horizon", 8.0))
    min_pushes = int(args.get("min_regrasps", MIN_REGRASPS))
    ids = np.arange(20)
    joints = m.actuator_trnid[sim.hand_act[ids], 0]
    qa, va = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
    low, high = m.jnt_range[joints].T
    wrist, core = m.body("l_wrist").id, sim.core
    core_v = int(m.joint("cube/core").dofadr[0])
    sites = [m.site(n).id for n in TIPS]
    finger_geom = np.array([physics.finger_channel(m.body(int(b)).name) for b in m.geom_bodyid])
    palm_geom = m.geom_bodyid == wrist
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)

    start = float(d.time)
    face = up_face(sim)
    start_facelets = sim.observe()["facelets"]
    names, centre = physics.layer_cubies(sim, face)
    layer_bodies = {m.body("cube/" + n).id for n in names}
    hinge_q = int(m.joint("cube/" + centre).qposadr[0])
    hinge_v = int(m.joint("cube/" + centre).dofadr[0])
    hinge0 = round(float(d.qpos[hinge_q]) / (math.pi / 2)) * (math.pi / 2)
    relabel = t(up_symmetry(face))
    welds, _ = physics.set_turn_welds(sim, face)
    cube_geoms = np.isin(m.geom_bodyid, list(sim.cube_bodies))
    saved_contype = m.geom_contype[cube_geoms].copy()
    if not internal_collisions:
        # While welded, cubie-cubie contacts are redundant with the hinge; hand and stand contacts stay.
        m.geom_contype[cube_geoms] = 0
    # The turning layer's welded ball joints carry no relative motion, so their friction, damping and armature
    # are removed for the turn; the centre hinge keeps its own.
    layer_dofs = np.array(
        [int(m.joint("cube/" + n).dofadr[0]) + k for n in names if n not in scene.CENTRES for k in range(3)]
    )
    saved_dof = [
        m.dof_frictionloss[layer_dofs].copy(),
        m.dof_damping[layer_dofs].copy(),
        m.dof_armature[layer_dofs].copy(),
    ]
    m.dof_frictionloss[layer_dofs] = 0.0
    m.dof_damping[layer_dofs] = 0.0
    m.dof_armature[layer_dofs] = 0.0
    mujoco.mj_forward(m, d)
    wrist_rot0 = d.xmat[wrist].reshape(3, 3).copy()
    home_rot = t(wrist_rot0.T @ d.xmat[core].reshape(3, 3))[None] @ relabel
    home_pos = t(wrist_rot0.T @ (d.xpos[core] - d.xpos[wrist]))[None]
    low_t, high_t = t(low), t(high)
    mid, half = (low_t + high_t) / 2, (high_t - low_t).clamp_min(0.01) / 2
    cmd = torch.tensor([float(command)])
    targets = t(sim.hand_target[ids])[None].clone()
    action = torch.zeros((1, 20))
    pushes = Pushes()
    elapsed = torch.zeros(1)
    worst = dict(penetration_m=0.0, joint_violation_rad=0.0)
    rows = []
    hold, success, failure, airborne, max_wrist, others = 0.0, False, None, 0.0, 0.0, set()

    def contact_flags():
        f = torch.zeros((1, 11))
        for c in d.contact[: d.ncon]:
            if c.dist > 0:
                continue
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            for ga, gb in ((g1, g2), (g2, g1)):
                bb = m.geom_bodyid[gb]
                if bb in sim.cube_bodies:
                    if 0 <= finger_geom[ga] < 5:
                        f[0, (5 if bb in layer_bodies else 0) + finger_geom[ga]] = 1
                    if palm_geom[ga]:
                        f[0, 10] = 1
        return f

    for step in range(round(horizon / CONTROL_DT) + 1):
        mujoco.mj_forward(m, d)
        wrist_rot = d.xmat[wrist].reshape(3, 3)
        lower_rot = t(d.xmat[core].reshape(3, 3))[None] @ relabel
        hinge = torch.tensor([float(d.qpos[hinge_q]) - hinge0])
        geo = measure(
            t(wrist_rot)[None],
            t(d.xpos[wrist])[None],
            lower_rot,
            t(d.xpos[core])[None],
            home_rot,
            home_pos,
            hinge,
            torch.tensor([float(d.qvel[hinge_v])]),
            cmd * math.pi / 2,
            t(d.qvel[core_v : core_v + 3])[None],
            t(d.qvel[core_v + 3 : core_v + 6])[None],
            t(d.site_xpos[sites])[None],
        )
        flags = contact_flags()
        if step:
            pushes.update(flags[:, 5:10], cmd * hinge)
        new_others = {n for n in physics.cube_contacts(sim) if not n.startswith("l_")}
        others |= new_others
        delta = wrist_rot @ wrist_rot0.T
        max_wrist = max(max_wrist, math.degrees(math.acos(np.clip((np.trace(delta) - 1) / 2, -1, 1))))
        error = abs(float(geo["error"][0]))
        offset = float(geo["offset"][0].norm())
        rot = float(geo["lower_rotation"][0])
        airborne = 0.0 if flags.sum() > 0 else airborne + CONTROL_DT
        rows.append(
            dict(
                time=float(d.time) - start,
                turn_deg=math.degrees(float(hinge[0])),
                error_deg=math.degrees(error),
                lower_rotation_deg=math.degrees(rot),
                offset_m=offset,
                pushes=int(pushes.count[0]),
            )
        )
        checks = dict(
            penetration=worst["penetration_m"] <= 0.001,
            joint_limits=worst["joint_violation_rad"] <= 0.05,
            wrist_under_10deg=max_wrist < 10.0,
            other_bodies_clear=not new_others,
            not_dropped=offset < 0.04 and airborne <= 0.2 and rot < math.radians(45),
        )
        if not all(checks.values()):
            failure = [k for k, v in checks.items() if not v]
            break
        settled = (
            error < math.radians(SUCCESS_DEGREES)
            and rot < math.radians(MAX_LOWER_ROTATION_DEG)
            and offset < MAX_LOWER_OFFSET_M
            and int(flags[0, :5].sum()) >= 3
            and abs(float(d.qvel[hinge_v])) < 0.3
            and float(geo["ang"][0].norm()) < 0.5
            and float(geo["lin"][0].norm()) < 0.05
            and pushes.count[0] >= min_pushes
        )
        hold = hold + CONTROL_DT if settled and step else 0.0
        if hold >= HOLD_SECONDS - 1e-6:
            success = True
            break
        if step == round(horizon / CONTROL_DT):
            failure = ["timeout"]
            break
        obs = observe(
            mid,
            half,
            t(d.qpos[qa])[None],
            t(d.qvel[va])[None],
            targets,
            action,
            geo,
            cmd,
            flags,
            pushes.count,
            elapsed,
            horizon,
        )
        with torch.no_grad():
            action = policy.mean_action(obs).clamp(-1, 1)
        targets = (targets + scale * action).clamp(low_t, high_t)
        sim.hand_target[ids] = targets[0].numpy()
        sim.phase = "rl_one_hand_face_turn"
        sim.step(20)
        elapsed += CONTROL_DT
        for k in worst:
            worst[k] = max(worst[k], sim.substep_worst[k])
        after_step()
    m.geom_contype[cube_geoms] = saved_contype
    m.dof_frictionloss[layer_dofs], m.dof_damping[layer_dofs], m.dof_armature[layer_dofs] = saved_dof
    physics.release_welds(sim)
    held = None
    if success and unweld_hold > 0:
        # Welds off; the cube must stay held and aligned.
        for _ in range(round(unweld_hold / CONTROL_DT)):
            sim.step(20)
            for k in worst:
                worst[k] = max(worst[k], sim.substep_worst[k])
            after_step()
        held = sim.held_in_hand()
        others |= {n for n in physics.cube_contacts(sim) if not n.startswith("l_")}
    end_obs = sim.observe()
    physical = worst["penetration_m"] <= 0.001 and worst["joint_violation_rad"] <= 0.05
    return dict(
        success=bool(
            success
            and physical
            and not others
            and max_wrist < 10
            and held
            and end_obs["misalign_deg"] < 10
            and scene.legal(end_obs["facelets"])
        ),
        failure=failure,
        face=face,
        command=int(command),
        duration_s=float(d.time) - start,
        pushes=int(pushes.count[0]),
        max_wrist_rotation_deg=max_wrist,
        other_body_contacts=sorted(others),
        held_after_unweld=held,
        welds=len(welds),
        start_facelets=start_facelets,
        end_facelets=end_obs["facelets"],
        end_misalign_deg=end_obs["misalign_deg"],
        **worst,
    ), rows
