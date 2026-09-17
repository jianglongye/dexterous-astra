"""MuJoCo copy of the scene, built from the official Sharpa MJCF.

Used for replaying recorded Isaac Lab states (rendering) and for an independent penetration check
on those states. Isaac Lab / PhysX remains the simulator.
"""

import numpy as np
import mujoco

from spec import FRICTION, HAND_MJCF, HAND_POS, JOINTS, PALM_UP_QUAT, PEN


def build(cameras=True):
    spec = mujoco.MjSpec.from_file(str(HAND_MJCF))
    spec.option.timestep = 0.002
    root = spec.body("right_hand_C_MC")
    root.pos = HAND_POS
    root.quat = PALM_UP_QUAT
    for g in spec.geoms:
        if g.contype or g.conaffinity:
            g.friction = [FRICTION, 0.005, 0.0001]
    body = spec.worldbody.add_body(name="pen", pos=[0.14, 0, HAND_POS[2] + 0.12])
    body.add_freejoint(name="pen_free")
    half = PEN["length"] / 2 - PEN["radius"]
    # capsule along local x
    body.add_geom(
        name="pen",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[PEN["radius"], half, 0],
        quat=[np.sqrt(0.5), 0, np.sqrt(0.5), 0],
        mass=PEN["mass"],
        friction=[FRICTION, 0.005, 0.0001],
        condim=4,
        rgba=[0.15, 0.45, 0.9, 1],
    )
    body.add_geom(
        name="pen_cap",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[PEN["radius"] * 1.02, 0, 0],
        pos=[PEN["length"] / 2 - PEN["radius"], 0, 0],
        contype=0,
        conaffinity=0,
        mass=0,
        rgba=[0.95, 0.45, 0.1, 1],
    )
    if cameras:
        # visual-only backdrop 20 cm below the palm (no floor exists in the Isaac Lab scene)
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0.6, 0.6, 0.01],
            pos=[0.1, 0, HAND_POS[2] - 0.2],
            rgba=[0.32, 0.34, 0.38, 1],
            contype=0,
            conaffinity=0,
        )
        spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
        spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
        spec.worldbody.add_light(pos=[0.1, -0.3, HAND_POS[2] + 0.6], dir=[0, 0.4, -1], diffuse=[0.8, 0.8, 0.8])
        spec.worldbody.add_light(pos=[0.3, 0.3, HAND_POS[2] + 0.5], dir=[-0.3, -0.3, -1], diffuse=[0.4, 0.4, 0.4])
    return spec.compile()


class Index:
    def __init__(self, m):
        self.qadr = np.array([m.joint(j).qposadr[0] for j in JOINTS])
        self.pen_qadr = m.joint("pen_free").qposadr[0]
        self.pen_geom = m.geom("pen").id
        pen_body = m.body("pen").id
        self.hand_geoms = np.array(
            [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_bodyid[g] != pen_body]
        )

    def set_state(self, d, q, pen_pos, pen_quat):
        d.qpos[self.qadr] = q
        d.qpos[self.pen_qadr : self.pen_qadr + 3] = pen_pos
        d.qpos[self.pen_qadr + 3 : self.pen_qadr + 7] = pen_quat


def pen_penetration(m, d, ix, distmax=0.01):
    """Deepest penetration between the pen capsule and any hand collision geom (convex hulls)."""
    worst = 0.0
    fromto = np.zeros(6)
    for g in ix.hand_geoms:
        dist = mujoco.mj_geomDistance(m, d, int(g), ix.pen_geom, distmax, fromto)
        worst = max(worst, -dist)
    return worst
