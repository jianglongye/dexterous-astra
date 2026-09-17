"""Centre-cubie support pinch and fingertip layer turns."""

from pathlib import Path
import json
import numpy as np, mujoco


def quat(axis, angle):
    return np.r_[np.cos(angle / 2), np.asarray(axis) * np.sin(angle / 2)]


def mul(a, b):
    q = np.empty(4)
    mujoco.mju_mulQuat(q, a, b)
    return q


def mat(q):
    r = np.empty(9)
    mujoco.mju_quat2Mat(r, q)
    return r.reshape(3, 3)


def blend(a, b, u):
    u = np.clip(u, 0, 1)
    u = u * u * (3 - 2 * u)
    return np.asarray(a) * (1 - u) + np.asarray(b) * u


I = np.array([1.0, 0, 0, 0])


class Controller:
    def __init__(self, info):
        self.info = info
        self.m = mujoco.MjModel.from_binary_path(info["model_path"])
        self.d = mujoco.MjData(self.m)
        self.aa = self.m.jnt_qposadr[self.m.actuator_trnid[:, 0]]
        self.dofs = self.m.jnt_dofadr[self.m.actuator_trnid[:, 0]]
        self.grasps = json.loads((Path(__file__).parent / "data" / "baseline_grasps.json").read_text())
        self.grasps["l"]["pos"][2] -= 0.021
        self.closed = {s: self.grip_joints(s, -0.008) for s in "lr"}
        self.open = {s: self.grip_joints(s, 0.014) for s in "lr"}
        for s in "lr":
            for h in [self.closed[s], self.open[s]]:
                for i in [8, 12, 16]:
                    h[i : i + 4] = [1.4, 0.3 if i == 16 else 0, 0, 0]
        self.faceq = {
            "U": I,
            "D": quat([1, 0, 0], np.pi),
            "R": quat([0, 1, 0], np.pi / 2),
            "L": quat([0, 1, 0], -np.pi / 2),
            "F": quat([1, 0, 0], np.pi / 2),
            "B": quat([1, 0, 0], -np.pi / 2),
        }
        self.keys = dict(zip("URFDLB", ["pZ", "pX", "nY", "nZ", "nX", "pY"]))
        self.moves = info["solution"].split()
        self.mi = 0
        self.phase = "init"
        self.pt = 0
        self.lastlog = -1
        self.center = np.array([0.0, 0.12, 0.33])
        self.lframe = quat([0, 0, 1], np.pi)
        self.baseq = quat([0, 0, 1], -np.pi / 2)
        self.bracing = False
        self.brace_j = np.array([1.03045, -0.10373, 0.29002, 0.50819])
        self.cube_target = I.copy()
        self.support_kind = "initial"
        self.current_yaw = np.pi if self.moves[0][0] == "F" else -np.pi / 2

    def phase_to(self, name, t):
        print("PHASE", round(t, 2), name, flush=True)
        self.phase = name
        self.pt = t

    def pose(self, s, c, q=I, dz=0):
        g = self.grasps[s]
        return np.asarray(c) + mat(q) @ (np.array(g["pos"]) + [0, 0, dz]), mul(q, g["quat"])

    def grip_joints(self, s, gap):
        m, d = self.m, self.d
        g = self.grasps[s]
        off = 0 if s == "l" else 20
        a = m.joint(s + "_wrist").qposadr[0]
        d.qpos[a : a + 3] = g["pos"]
        d.qpos[a + 3 : a + 7] = g["quat"]
        j = np.array(g["joints"])
        d.qpos[self.aa[off : off + 20]] = j
        mujoco.mj_kinematics(m, d)
        bs = [m.body(s + "_" + f + "_distal").id for f in ["thumb", "index_finger"]]
        ps = [
            np.array([0, (-1 if s == "l" else 1) * (0.009 if i == 0 else 0.0078), -0.024 if i == 0 else -0.02])
            for i in range(2)
        ]
        targets = [
            d.xpos[b] + d.xmat[b].reshape(3, 3) @ p + np.array([1 if i == 0 else -1, 0, 0]) * gap
            for i, (b, p) in enumerate(zip(bs, ps))
        ]
        for _ in range(40):
            d.qpos[self.aa[off : off + 20]] = j
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            for i, (b, p, tgt) in enumerate(zip(bs, ps, targets)):
                point = d.xpos[b] + d.xmat[b].reshape(3, 3) @ p
                jp = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jp, None, point, b)
                ids = slice(off + i * 4, off + i * 4 + 4)
                J = jp[:, self.dofs[ids]]
                step = J.T @ np.linalg.solve(J @ J.T + np.eye(3) * 1e-6, tgt - point)
                j[i * 4 : i * 4 + 4] = np.clip(
                    j[i * 4 : i * 4 + 4] + np.clip(step, -0.06, 0.06),
                    m.actuator_ctrlrange[ids, 0] + 0.015,
                    m.actuator_ctrlrange[ids, 1] - 0.015,
                )
        return j
