"""Collision checks on the controller's private compiled scene."""

import numpy as np, mujoco


def sync(ctrl, obs):
    m, d = ctrl.m, ctrl.d
    d.qpos[ctrl.aa] = obs["hand_qpos"]
    for side in "lr":
        a = m.joint(side + "_wrist").qposadr[0]
        d.qpos[a : a + 3] = obs["wrist"][side][0]
        d.qpos[a + 3 : a + 7] = obs["wrist"][side][1]
    a = m.joint("cube/core").qposadr[0]
    d.qpos[a : a + 3] = obs["cube_pos"]
    d.qpos[a + 3 : a + 7] = obs["cube_quat"]


def set_hand(ctrl, side, pose, joints):
    m, d = ctrl.m, ctrl.d
    a = m.joint(side + "_wrist").qposadr[0]
    p, q = pose
    d.qpos[a : a + 3] = p
    d.qpos[a + 3 : a + 7] = q
    off = 0 if side == "l" else 20
    d.qpos[ctrl.aa[off : off + 20]] = joints


def collision(ctrl):
    m, d = ctrl.m, ctrl.d
    mujoco.mj_kinematics(m, d)
    mujoco.mj_collision(m, d)
    worst = 0.0
    for co in d.contact:
        bs = [m.body(m.geom_bodyid[x]).name for x in co.geom]
        if (
            set(b[:2] for b in bs) == {"l_", "r_"}
            or not ctrl.bracing
            and any(b.startswith(("r_wrist", "r_middle", "r_ring", "r_pinky")) for b in bs)
            and any(b.startswith("cube/") for b in bs)
        ):
            worst = max(worst, -co.dist)
    return worst


def support_candidates(ctrl, obs, face):
    from controller import quat, mul, mat

    sync(ctrl, obs)
    c = obs["cube_pos"]
    best = []
    for yaw in [np.pi, 0, np.pi / 2, -np.pi / 2]:
        local = mul(ctrl.faceq[face], mul(quat([1, 0, 0], np.pi), quat([0, 0, 1], yaw)))
        frame = mul(obs["cube_quat"], local)
        p, q = ctrl.pose("l", c, frame)
        worst = 0.0
        for gap, j in [(x, ctrl.open["l"]) for x in np.linspace(0.12, 0, 8)] + [(0, ctrl.closed["l"])]:
            set_hand(ctrl, "l", (p + mat(frame)[:, 2] * gap, q), j)
            worst = max(worst, collision(ctrl))
        best.append((worst, yaw, frame))
    best.sort(key=lambda x: x[0])
    print("SUPPORT_PLAN", [(round(x[0], 5), round(x[1], 3)) for x in best], flush=True)
    return best[0][2]


def turn_candidates(ctrl, obs, move):
    from controller import quat, mul, mat

    sync(ctrl, obs)
    c = obs["cube_pos"]
    face = move[0]
    delta = {"": -np.pi / 2, "'": np.pi / 2, "2": np.pi}[move[1:]]
    best = []
    for yaw in np.arange(0, 2 * np.pi, np.pi / 4):
        worst = 0.0
        for angle, gap, j in [(0, x, ctrl.open["r"]) for x in [0.08, 0.04, 0]] + [
            (x, 0, ctrl.closed["r"]) for x in np.linspace(0, delta, 13)
        ]:
            frame = mul(mul(obs["cube_quat"], ctrl.faceq[face]), quat([0, 0, 1], yaw + angle))
            set_hand(ctrl, "r", ctrl.pose("r", c, frame, dz=gap), j)
            worst = max(worst, collision(ctrl))
        end = mul(mul(obs["cube_quat"], ctrl.faceq[face]), quat([0, 0, 1], yaw + delta))
        preference = 0.0001 * (1 - abs(mat(end)[2, 0]))
        best.append((worst + preference, yaw))
    best.sort()
    print("TURN_PLAN", [(round(x, 5), round(y, 3)) for x, y in best], flush=True)
    ctrl.next_yaw_cost = float(best[0][0])
    return best[0][1]
