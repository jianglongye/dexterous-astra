"""Physics for the one-hand policy solve, on top of the original two-hand scene (scene.py, run.py).

- Simplified profile (physics.json): wider stand, fingertip friction x1.5, half centre-hinge friction,
  weaker detents, and passive alignment wells near the legal cubie orientations.
- Stiffer hand contacts and per-seed friction/actuator-strength jitter.
- Inactive weld constraints between cube parts, switched on only during learned maneuvers:
  `lock_<cubie>` (cubie -> core) and `layer_<face>_<cubie>` (edge/corner -> that face's centre).
  Activation copies the current relative pose, so it never moves anything.
No hidden supports, state edits or cube actuators.
"""

import json
from pathlib import Path

import mujoco
import numpy as np

import run
import scene

PROFILE = json.loads((Path(__file__).with_name("physics.json")).read_text())
LOCK_PREFIX = "lock_"
LAYER_PREFIX = "layer_"
FINGERS = tuple(s + "_" + f for s in "lr" for f in ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky"))


def finger_channel(body_name):
    return next((i for i, prefix in enumerate(FINGERS) if body_name.startswith(prefix + "_")), -1)


def _centre_normal(name):
    sign = 1.0 if name[0] == "p" else -1.0
    return tuple(sign * (np.arange(3) == "XYZ".index(name[1])))


def centre_face(centre):
    normal = _centre_normal(centre)
    return next(f for f in scene.FACES if np.allclose(scene.NORMAL[f], normal))


def _build_with_welds(original):
    def build_spec(free_cube=True, embodiment="arms"):
        spec = original(free_cube=free_cube, embodiment=embodiment)
        spec.compiler.usethread = False
        if embodiment != "arms":
            return spec
        core = spec.body("cube/core")
        for body in core.find_all(mujoco.mjtObj.mjOBJ_BODY):
            eq = spec.add_equality(
                type=mujoco.mjtEq.mjEQ_WELD,
                name1=body.name,
                name2="cube/core",
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                active=False,
                solref=(0.005, 1),
            )
            eq.name = LOCK_PREFIX + body.name.split("/")[-1]
            eq.data = [0] * 10 + [1]
        for body in core.find_all(mujoco.mjtObj.mjOBJ_BODY):
            name = body.name.split("/")[-1]
            if name in scene.CENTRES:
                continue
            # Any edge/corner can occupy any layer after turns, so every centre gets a weld from every cubie.
            for face in scene.FACES:
                centre = next(c for c in scene.CENTRES if np.allclose(_centre_normal(c), scene.NORMAL[face]))
                eq = spec.add_equality(
                    type=mujoco.mjtEq.mjEQ_WELD,
                    name1=body.name,
                    name2="cube/" + centre,
                    objtype=mujoco.mjtObj.mjOBJ_BODY,
                    active=False,
                    solref=(0.005, 1),
                )
                eq.name = f"{LAYER_PREFIX}{face}_{name}"
                eq.data = [0] * 10 + [1]
        return spec

    return build_spec


def wells(quaternions, velocities, group, stiffness, width, damping):
    """Torque toward the nearest legal cubie orientation, active only near it."""
    q = np.asarray(quaternions)
    goal = group[np.abs(q @ group.T).argmax(axis=-1)].copy()
    goal *= np.where((q * goal).sum(-1) < 0, -1.0, 1.0)[:, None]
    # conj(q) * goal: tangent error expressed in the current joint frame.
    scalar = (q * goal).sum(-1)
    vector = q[:, :1] * goal[:, 1:] - goal[:, :1] * q[:, 1:] - np.cross(q[:, 1:], goal[:, 1:])
    norm = np.linalg.norm(vector, axis=-1)
    angle = 2 * np.arctan2(norm, np.clip(scalar, 0, 1))
    error = vector * (angle / np.maximum(norm, 1e-12))[:, None]
    near = np.exp(-0.5 * (angle / width) ** 2)
    return near[:, None] * (stiffness * error - damping * velocities)


def _install_wells(sim):
    model = sim.m
    base = scene.apply_detents
    ids = [
        j
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL and model.body(model.jnt_bodyid[j]).name.startswith("cube/")
    ]
    qa = np.asarray(model.jnt_qposadr[ids])[:, None] + np.arange(4)
    va = np.asarray(model.jnt_dofadr[ids])[:, None] + np.arange(3)
    group = np.empty((len(scene.GROUP), 4))
    for i, rotation in enumerate(scene.GROUP):
        mujoco.mju_mat2Quat(group[i], rotation.astype(float).ravel())
    p = PROFILE["passive_cubie_alignment"]

    def apply(m, d, k=0.02):
        base(m, d, k)
        d.qfrc_applied[va] = wells(
            d.qpos[qa],
            d.qvel[va],
            group,
            p["stiffness_Nm_per_rad"],
            np.deg2rad(p["width_degrees"]),
            p["damping_Nm_s_per_rad"],
        )

    scene.apply_detents = apply


def make_sim(scramble, out, seed=0, friction_jitter=0.02, strength_jitter=0.01, hand_contact_time_constant=0.018):
    scene.build_spec = _build_with_welds(scene.build_spec)
    scene.PEDESTAL = PROFILE["pedestal_half_width_m"]
    sim = run.Sim(scramble, out)
    fingers = np.array([finger_channel(sim.m.body(int(b)).name) >= 0 for b in sim.m.geom_bodyid])
    sim.m.geom_friction[fingers, 0] *= PROFILE["finger_friction_multiplier"]
    dofs = np.array([int(j.dofadr[0]) for j in sim.centres])
    sim.m.dof_frictionloss[dofs] *= PROFILE["centre_friction_multiplier"]
    run.DETENT = PROFILE["detent_stiffness_Nm_per_rad"]
    _install_wells(sim)
    mujoco.mj_forward(sim.m, sim.d)
    rng = np.random.default_rng(seed)
    friction = rng.uniform(1 - friction_jitter, 1 + friction_jitter)
    strength = rng.uniform(1 - strength_jitter, 1 + strength_jitter)
    sim.m.geom_friction[:] *= friction
    sim.m.actuator_gainprm[:] *= strength
    sim.m.actuator_biasprm[:] *= strength
    hand = np.isin(sim.m.geom_bodyid, list(sim.hand_bodies))
    sim.m.geom_solref[hand] = [hand_contact_time_constant, 1.0]
    sim.m.geom_priority[hand] = 1
    mujoco.mj_forward(sim.m, sim.d)
    mujoco.mj_saveModel(sim.m, str(out / "scene.mjb"), None)
    return sim, dict(
        profile=PROFILE,
        hand_contact_time_constant_s=hand_contact_time_constant,
        randomization=dict(seed=seed, friction_scale=float(friction), actuator_scale=float(strength)),
    )


def layer_cubies(sim, face):
    """Names of cubies currently in the layer of `face` (by snapped slot), and its centre name."""
    rots, home = scene.cubie_rotations(sim.m, sim.d)
    slots = np.einsum("nij,nj->ni", scene.snap(rots)[0], home)
    ids, _ = scene.cubies(sim.m)
    normal = np.array(scene.NORMAL[face], float)
    names = [sim.m.body(int(b)).name.split("/")[-1] for b, s in zip(ids, slots) if s @ normal > 0.5]
    centre = next(n for n in names if n in scene.CENTRES)
    return names, centre


def _activate(sim, ids, active):
    m, d = sim.m, sim.d
    if active:
        mujoco.mj_kinematics(m, d)
        for i in ids:
            # relpose = pose of body2 expressed in the body1 frame.
            b1, b2 = m.eq_obj1id[i], m.eq_obj2id[i]
            inv = np.zeros(4)
            mujoco.mju_negQuat(inv, d.xquat[b1])
            pos = np.zeros(3)
            mujoco.mju_rotVecQuat(pos, d.xpos[b2] - d.xpos[b1], inv)
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, inv, d.xquat[b2])
            m.eq_data[i, :3] = 0
            m.eq_data[i, 3:6] = pos
            m.eq_data[i, 6:10] = quat
            m.eq_data[i, 10] = 1
    d.eq_active[ids] = bool(active)
    mujoco.mj_forward(m, d)


def set_roll_welds(sim, active):
    """All 26 cubies rigid to the core (whole-cube roll)."""
    ids = [i for i in range(sim.m.neq) if sim.m.eq(i).name.startswith(LOCK_PREFIX)]
    _activate(sim, ids, active)
    return ids


def set_turn_welds(sim, face):
    """The `face` layer rigid on its centre hinge, all other cubies rigid to the core."""
    m = sim.m
    names, centre = layer_cubies(sim, face)
    home_face = centre_face(centre)
    ids = []
    for i in range(m.neq):
        n = m.eq(i).name
        if n.startswith(LOCK_PREFIX) and n[len(LOCK_PREFIX) :] not in names:
            ids.append(i)
        elif n.startswith(LAYER_PREFIX):
            f, cubie = n[len(LAYER_PREFIX) :].split("_", 1)
            if f == home_face and cubie in names:
                ids.append(i)
    _activate(sim, ids, True)
    return ids, centre


def release_welds(sim):
    sim.d.eq_active[:] = False
    mujoco.mj_forward(sim.m, sim.d)


def cube_contacts(sim):
    """Names of non-cube bodies touching the cube (distance <= 0)."""
    m, d = sim.m, sim.d
    names = set()
    for c in d.contact[: d.ncon]:
        if c.dist > 0:
            continue
        b1, b2 = m.geom_bodyid[c.geom[0]], m.geom_bodyid[c.geom[1]]
        in1, in2 = b1 in sim.cube_bodies, b2 in sim.cube_bodies
        if in1 != in2:
            other = b2 if in1 else b1
            names.add(
                m.body(int(other)).name
                if other
                else "world:" + (m.geom(int(c.geom[1] if in1 else c.geom[0])).name or "geom")
            )
    return names
