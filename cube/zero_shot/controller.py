"""Centre-cubie support pinch and fingertip layer turns."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
import grasp_planner
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
        self.grasps = json.loads(Path(__file__).with_name("grasps.json").read_text())
        self.grasps["l"]["pos"][2] -= 0.021
        self.closed = {s: self.grip_joints(s, -0.008) for s in "lr"}
        self.open = {s: self.grip_joints(s, 0.014) for s in "lr"}
        for s in "lr":
            for h in [self.closed[s], self.open[s]]:
                for i in [8, 12, 16]:
                    h[i : i + 4] = [1.4, 0.3 if i == 16 else 0, 0.25, 0.15]
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

    def force_grip(self, obs, s, frame, target, desired, weight=1.0, turn_torque=None):
        m, d = self.m, self.d
        off = 0 if s == "l" else 20
        d.qpos[self.aa] = obs["hand_qpos"]
        for ss in "lr":
            a = m.joint(ss + "_wrist").qposadr[0]
            d.qpos[a : a + 3] = obs["wrist"][ss][0]
            d.qpos[a + 3 : a + 7] = obs["wrist"][ss][1]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        bs = [m.body(s + "_" + f + "_distal").id for f in ["thumb", "index_finger"]]
        ps = [
            d.xpos[b]
            + d.xmat[b].reshape(3, 3)
            @ np.array([0, (-1 if s == "l" else 1) * (0.009 if i == 0 else 0.0078), -0.024 if i == 0 else -0.02])
            for i, b in enumerate(bs)
        ]
        if s == "r" and self.bracing:
            bs.append(m.body("r_middle_finger_middle").id)
            ps.append(d.xpos[bs[-1]] + d.xmat[bs[-1]].reshape(3, 3) @ np.array([0.0075, -0.005, 0.0025]))
        A = np.zeros((6, 3 * len(bs)))
        for i, p in enumerate(ps):
            x, y, z = p - obs["cube_pos"]
            A[:3, i * 3 : i * 3 + 3] = np.eye(3)
            A[3:, i * 3 : i * 3 + 3] = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
        er = np.zeros(3)
        mujoco.mju_subQuat(er, desired, obs["cube_quat"])
        er = mat(obs["cube_quat"]) @ er
        wrench = np.r_[
            np.array([0, 0, 0.672 * weight]) + 20 * (target - obs["cube_pos"]) - 0.3 * obs["cube_vel"][:3],
            0.08 * er - 0.004 * obs["cube_vel"][3:],
        ]
        if turn_torque is not None:
            wrench = np.r_[np.zeros(3), turn_torque]
        preload = [[-1.5, 0, 0], [1.5, 0, 0]] + ([[0, -1.5, -0.4]] if s == "r" and self.bracing else [])
        fs = (mat(frame) @ np.array(preload).T).T.ravel()
        fs += A.T @ np.linalg.solve(A @ A.T + np.diag([1e-5] * 3 + [2e-6] * 3), wrench - A @ fs)
        out = obs["hand_qpos"][off : off + 4 * len(bs)].copy()
        for i, (b, p, f) in enumerate(zip(bs, ps, fs.reshape(-1, 3))):
            jp = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, None, p, b)
            ids = slice(off + i * 4, off + i * 4 + 4)
            out[i * 4 : i * 4 + 4] += jp[:, self.dofs[ids]].T @ f / m.actuator_gainprm[ids, 0]
        return out + 0.03 * (self.closed[s][: len(out)] - obs["hand_qpos"][off : off + len(out)])

    def start_handoff(self, t, obs, frame):
        self.hc = obs["cube_pos"].copy()
        self.hq = obs["cube_quat"].copy()
        self.hrframe = frame.copy()
        self.hright = tuple(x.copy() for x in obs["wrist"]["r"])
        self.hleft = tuple(x.copy() for x in obs["wrist"]["l"])
        self.hold_lframe = self.lframe.copy()
        self.hleftj = obs["hand_qpos"][:20].copy()
        self.hrj = obs["hand_qpos"][20:28].copy()
        self.new_lframe = grasp_planner.support_candidates(self, obs, self.moves[self.mi + 1][0])
        self.new_lpose = self.pose("l", self.hc, self.new_lframe)
        self.phase_to("handoff_brace" if self.moves[self.mi][0] in "RL" else "handoff_open", t)

    def handoff(self, t, obs, h, w):
        ph = self.phase
        dt = t - self.pt
        h[20:] = self.closed["r"]
        share = float(blend(0, 1, (dt - 1) / 0.8)) if ph == "handoff_close" else 0.0
        f = self.force_grip(obs, "r", self.hrframe, self.hc, self.hq, weight=1 - share)
        h[20 : 20 + len(f)] = f + (0 if self.bracing else 0.6) * (self.hrj - obs["hand_qpos"][20 : 20 + len(f)])
        w["r"] = self.hright
        h[:20] = self.open["l"]
        if ph == "handoff_brace":
            w["l"] = self.pose("l", self.center, self.lframe)
            h[:20] = self.closed["l"]
            h[:8] = self.force_grip(obs, "l", self.lframe, self.center, self.cube_target)
            h[28:32] = blend(self.open["r"][8:12], self.brace_j, dt / 1.2)
            if dt >= 1.5:
                self.bracing = True
                self.closed["r"][8:12] = self.brace_j
                self.hrj = obs["hand_qpos"][20:32].copy()
                self.hc = obs["cube_pos"].copy()
                self.hq = obs["cube_quat"].copy()
                self.new_lframe = grasp_planner.support_candidates(self, obs, self.moves[self.mi + 1][0])
                self.new_lpose = self.pose("l", self.hc, self.new_lframe)
                self.phase_to("handoff_open", t)
        elif ph == "handoff_open":
            w["l"] = self.hleft
            h[:20] = blend(self.hleftj, self.open["l"], dt)
            if dt >= 1:
                self.phase_to("handoff_out", t)
        elif ph == "handoff_out":
            p, q = self.hleft
            w["l"] = (p + mat(self.hold_lframe)[:, 2] * float(blend(0, 0.15, dt / 1.2)), q)
            if dt >= 1.2:
                self.phase_to("handoff_route", t)
        elif ph == "handoff_route":
            p, q = self.new_lpose
            w["l"] = (p + mat(self.new_lframe)[:, 2] * 0.15, q)
            if dt >= 1.6:
                self.phase_to("handoff_in", t)
        elif ph == "handoff_in":
            p, q = self.new_lpose
            w["l"] = (p + mat(self.new_lframe)[:, 2] * float(blend(0.15, 0, dt / 1.8)), q)
            if dt >= 1.8:
                self.phase_to("handoff_close", t)
        elif ph == "handoff_close":
            w["l"] = self.new_lpose
            h[:20] = blend(self.open["l"], self.closed["l"], dt / 1.5)
            if share > 0:
                h[:8] = blend(h[:8], self.force_grip(obs, "l", self.new_lframe, self.hc, self.hq, weight=share), share)
            if dt >= 1.8:
                self.lframe = self.new_lframe
                self.center = self.hc
                self.cube_target = self.hq
                self.support_kind = "generic"
                self.next_yaw = grasp_planner.turn_candidates(self, obs, self.moves[self.mi + 1])
                self.bracing = False
                self.phase_to("release", t)

    def next_action(self, t, obs, frame):
        if self.mi + 1 == len(self.moves):
            self.phase_to("release", t)
            return
        nf = self.moves[self.mi + 1][0]
        if self.support_kind == "initial" and nf in "DF":
            self.phase_to("release", t)
            return
        axis = mat(self.lframe)[:, 0]
        normal = mat(mul(obs["cube_quat"], self.faceq[nf]))[:, 2]
        if abs(axis @ normal) < 0.7:
            self.next_yaw = grasp_planner.turn_candidates(self, obs, self.moves[self.mi + 1])
            if self.next_yaw_cost < 0.001:
                self.phase_to("release", t)
                return
        if self.moves[self.mi][0] in "FB":
            self.start_handoff(t, obs, frame)
        else:
            self.park_pending = True
            self.phase_to("release", t)

    def start_park(self, t, obs):
        self.pc = obs["cube_pos"].copy()
        self.pq = obs["cube_quat"].copy()
        self.pl = tuple(x.copy() for x in obs["wrist"]["l"])
        self.pframe = self.lframe.copy()
        n = mat(self.faceq[self.moves[self.mi][0]])[:, 2]
        rel = mat(self.cube_target).T @ mat(self.lframe)
        axis = rel[:, 0]
        palm = rel[:, 2]
        axis = np.eye(3)[:, np.argmax(abs(axis))] * np.sign(axis[np.argmax(abs(axis))])
        goals = []
        for b in [x * sgn for x in np.eye(3) for sgn in [-1, 1]]:
            if abs(n @ b) > 0.5:
                continue
            R = np.array([n, b, np.cross(n, b)])
            if abs((R @ axis)[2]) > 0.5 or (R @ palm)[2] < -0.3:
                continue
            q = np.empty(4)
            mujoco.mju_mat2Quat(q, R.ravel())
            angle = 2 * np.arccos(np.clip(abs(q @ self.pq), 0, 1))
            goals.append((angle - 0.4 * (R @ palm)[2], q))
        self.pgoal = min(goals, key=lambda x: x[0])[1]
        dq = mul(self.pgoal, self.pq * np.array([1, -1, -1, -1]))
        dq = dq if dq[0] >= 0 else -dq
        self.pa = 2 * np.arccos(np.clip(dq[0], -1, 1))
        self.paxis = dq[1:] / max(np.linalg.norm(dq[1:]), 1e-9)
        self.pduration = max(1.0, self.pa / 1.0)
        self.phase_to("park_rotate", t)

    def parking(self, t, obs, h, w):
        ph = self.phase
        dt = t - self.pt
        w["r"] = self.right_away
        h[:20] = self.closed["l"]
        if ph == "park_rotate":
            rot = quat(self.paxis, self.pa * float(blend(0, 1, dt / self.pduration)))
            frame = mul(rot, self.pframe)
            cq = mul(rot, self.pq)
            p, q = self.pl
            w["l"] = (self.pc + mat(rot) @ (p - self.pc), mul(rot, q))
            h[:8] = self.force_grip(obs, "l", frame, self.pc, cq)
            if dt >= self.pduration + 0.2:
                self.park_offset = w["l"][0] - self.pc
                self.park_wq = w["l"][1]
                self.lframe = frame
                self.cube_target = self.pgoal
                self.phase_to("park_lower", t)
        elif ph in ["park_lower", "park_open", "park_out"]:
            center = blend(self.pc, [0, 0, 0.2015], dt / 2.5) if ph == "park_lower" else np.array([0.0, 0.0, 0.2015])
            w["l"] = (center + self.park_offset, self.park_wq)
            if ph == "park_lower":
                h[:8] = self.force_grip(obs, "l", self.lframe, center, self.pgoal)
                if dt >= 2.7:
                    self.park_j = obs["hand_qpos"][:20].copy()
                    self.phase_to("park_open", t)
            elif ph == "park_open":
                h[:20] = blend(self.park_j, self.open["l"], dt / 0.8)
                if dt >= 1:
                    self.phase_to("park_out", t)
            else:
                h[:20] = self.open["l"]
                w["l"] = (w["l"][0] + mat(self.lframe)[:, 2] * float(blend(0, 0.10, dt / 0.7)), w["l"][1])
                if dt >= 0.8:
                    self.lframe = quat([0, 0, 1], -np.pi / 2)
                    self.phase_to("park_approach", t)
        elif ph in ["park_approach", "park_grasp_lower", "park_close"]:
            h[:20] = self.open["l"]
            gap = (
                0.08 if ph == "park_approach" else float(blend(0.08, 0, dt / 1.5)) if ph == "park_grasp_lower" else 0.0
            )
            w["l"] = self.pose("l", [0, 0, 0.2], self.lframe, dz=gap)
            if ph == "park_approach" and dt >= 1.5:
                self.phase_to("park_grasp_lower", t)
            if ph == "park_grasp_lower" and dt >= 1.5:
                self.phase_to("park_close", t)
            if ph == "park_close":
                h[:20] = blend(self.open["l"], self.closed["l"], dt / 1.2)
                if dt >= 1.6:
                    self.center = np.array([0.0, 0.12, 0.33])
                    self.support_kind = "generic"
                    self.current_yaw = -np.pi / 2
                    self.phase_to("lift", t)

    def act(self, t, obs):
        ph = self.phase
        dt = t - self.pt
        c = obs["cube_pos"].copy()
        h = np.r_[self.open["l"], self.open["r"]]
        w = {s: (np.array([-0.3, 0.35 if s == "l" else -0.3, 0.5]), self.info["wrist_init"][s][1]) for s in "lr"}
        if ph.startswith("park_"):
            self.parking(t, obs, h, w)
        elif ph.startswith("handoff_"):
            self.handoff(t, obs, h, w)
        elif ph == "init":
            h = blend(np.zeros(40), h, t / 2)
            for s, (p, q) in self.info["wrist_init"].items():
                w[s] = (p + blend([0, 0, 0], [-0.15, 0.18 if s == "l" else -0.18, 0.23], t / 2), q)
            if dt >= 2:
                self.phase_to("pickup_approach", t)
        elif ph == "pickup_approach":
            w["l"] = self.pose("l", [0, 0, 0.2], self.lframe, dz=0.08)
            if dt >= 1.8:
                self.phase_to("pickup_lower", t)
        elif ph == "pickup_lower":
            w["l"] = self.pose("l", [0, 0, 0.2], self.lframe, dz=float(blend(0.08, 0, dt / 2)))
            if dt >= 2:
                self.phase_to("pickup_close", t)
        elif ph == "pickup_close":
            w["l"] = self.pose("l", [0, 0, 0.2], self.lframe)
            h[:20] = blend(self.open["l"], self.closed["l"], dt / 1.5)
            if dt >= 2:
                self.phase_to("lift", t)
        else:
            target = blend([0, 0, 0.2], self.center, dt / 3) if ph == "lift" else self.center
            w["l"] = self.pose("l", target, self.lframe)
            h[:20] = self.closed["l"]
            h[:8] = self.force_grip(obs, "l", self.lframe, target, self.cube_target)
            if ph == "lift":
                if dt >= 3.5:
                    self.phase_to("route", t)
            elif ph != "done":
                face = self.moves[self.mi][0]
                fq = mul(obs["cube_quat"], self.faceq[face])
                rq = mul(fq, quat([0, 0, 1], self.current_yaw))
                if ph == "route":
                    p, q = self.pose("r", c, rq, dz=0.07)
                    tangent = np.cross([0, 0, 1], mat(fq)[:, 2])
                    if np.linalg.norm(tangent) < 0.2 or self.support_kind == "initial":
                        tangent = np.array([1.0, 0, 0])
                    tangent /= np.linalg.norm(tangent)
                    w["r"] = (p - 0.3 * tangent, q)
                    if dt >= (1.2 if self.support_kind == "generic" else 2):
                        self.phase_to("approach_outer", t)
                elif ph == "approach_outer":
                    p, q = self.pose("r", c, rq, dz=0.07)
                    w["r"] = (p, q)
                    if dt >= (1.2 if self.support_kind == "generic" else 2):
                        self.phase_to("approach", t)
                elif ph == "approach":
                    p, q = self.pose("r", c, rq, dz=0.07)
                    w["r"] = (p, q)
                    if dt >= (0.2 if self.support_kind == "generic" else 2):
                        self.phase_to("lower", t)
                elif ph == "lower":
                    w["r"] = self.pose("r", c, rq, dz=float(blend(0.07, 0, dt / 2)))
                    if dt >= 2:
                        self.phase_to("close", t)
                elif ph == "close":
                    w["r"] = self.pose("r", c, rq)
                    h[20:] = blend(self.open["r"], self.closed["r"], dt / 1.5)
                    if dt >= 2:
                        self.start = obs["face_angles"][self.keys[face]]
                        self.delta = {"": -np.pi / 2, "'": np.pi / 2, "2": np.pi}[self.moves[self.mi][1:]]
                        self.goal = round(self.start / (np.pi / 2)) * (np.pi / 2) + self.delta
                        self.progress = 0
                        self.phase_to("turn", t)
                elif ph == "turn":
                    h[20:] = self.closed["r"]
                    actual = obs["face_angles"][self.keys[face]]
                    error = self.goal - actual
                    self.progress += float(np.clip(error, -0.5, 0.5)) * 0.01
                    self.progress = float(
                        np.clip(self.progress, actual - self.start - 0.35, actual - self.start + 0.35)
                    )
                    frame = mul(rq, quat([0, 0, 1], self.progress))
                    w["r"] = self.pose("r", c, frame)
                    horizontal = abs(mat(mul(self.cube_target, self.faceq[face]))[2, 2]) < 0.7
                    limit = 0.025 if horizontal else 0.035
                    axis = mat(fq)[:, 2]
                    torque = axis * float(np.clip(0.08 * error, -limit, limit))
                    if horizontal:
                        torque *= float(blend(0, 1, dt / 0.8))
                    if horizontal and self.support_kind == "generic":
                        er = np.zeros(3)
                        mujoco.mju_subQuat(er, self.cube_target, obs["cube_quat"])
                        restore = 0.025 * (mat(obs["cube_quat"]) @ er) - 0.002 * (
                            mat(obs["cube_quat"]) @ obs["cube_vel"][3:]
                        )
                        restore -= axis * (axis @ restore)
                        restore *= min(1, 0.012 / max(np.linalg.norm(restore), 1e-9))
                        torque += restore
                    h[20:28] = self.force_grip(obs, "r", frame, c, obs["cube_quat"], weight=0, turn_torque=torque)
                    if abs(error) < 0.06 and dt > 1:
                        self.next_action(t, obs, frame)
                elif ph == "release":
                    h[20:] = blend(self.closed["r"], self.open["r"], dt)
                    w["r"] = self.pose(
                        "r",
                        c,
                        mul(rq, quat([0, 0, 1], self.progress)),
                        dz=float(blend(0, 0.16 if self.support_kind == "generic" else 0.10, (dt - 1) / 1.5)),
                    )
                    if dt >= 2.5:
                        self.closed["r"][8:12] = [1.4, 0, 0, 0]
                        self.mi += 1
                        if self.mi < len(self.moves):
                            self.current_yaw = getattr(
                                self, "next_yaw", np.pi if self.moves[self.mi][0] == "F" else -np.pi / 2
                            )
                        if getattr(self, "park_pending", False):
                            self.park_pending = False
                            self.right_away = tuple(x.copy() for x in w["r"])
                            self.start_park(t, obs)
                        else:
                            self.phase_to("route" if self.mi < len(self.moves) else "done", t)
        if int(t) != self.lastlog:
            self.lastlog = int(t)
            print(
                "STATE",
                round(t, 2),
                ph,
                np.round(c, 4),
                np.round(obs["cube_quat"], 3),
                round(obs["misalign_deg"], 2),
                {k: round(v, 3) for k, v in obs["face_angles"].items()},
                sorted(set(x[0] for x in obs["contacts"])),
                flush=True,
            )
        return {"hand": h, "wrist": w}
