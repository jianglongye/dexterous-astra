"""Three-contact support with bounded, friction-constrained force estimates."""

import json
import os
from pathlib import Path
import mujoco
import numpy as np
from baseline import mat, mul


class Tripod:
    def __init__(self, controller, side="l", grasp_path=None):
        self.c = controller
        self.side = side
        self.off = 0 if side == "l" else 20
        self.grip_in_cube = np.array([1.0, 0, 0, 0])
        right_path = os.environ.get("RUBIKS_RIGHT_SUPPORT_GRASP") if side == "r" else None
        self.g = json.loads(
            (
                Path(__file__).parent
                / "data"
                / (grasp_path or right_path or os.environ.get("RUBIKS_SUPPORT_GRASP", "support_grasp.json"))
            ).read_text()
        )
        shift = float(os.environ.get("RUBIKS_TRIPOD_Z", "0"))
        self.g["pos"][2] += shift
        for target in self.g["targets"]:
            target[2] += shift
        self.fingers = self.g.get("fingers", ["thumb", "index_finger", "middle_finger"])
        self.n = len(self.fingers)
        self.closed = np.r_[self.g["joints"], self.g.get("rest_joints", [1.4, 0, 0, 0] * (5 - self.n))]
        self.pickup_ring = np.asarray(self.g.get("pickup_ring", [1.4, 0, 0, 0]), float)
        self.bodies = [controller.m.body(side + "_" + f + "_distal").id for f in self.fingers]
        self.offsets = np.array(self.g["offsets"])
        self.normals = np.array(self.g["normals"])
        self.open = self.open_joints()

    def pose(self, center, orientation):
        frame = mul(orientation, self.grip_in_cube)
        return np.asarray(center) + mat(frame) @ self.g["pos"], mul(frame, self.g["quat"])

    @property
    def pickup_open(self):
        q = self.open.copy()
        if self.n > 3:
            q[12:16] = getattr(self, "pickup_ring", np.array([1.4, 0, 0, 0]))
        return q

    def open_joints(self):
        c = self.c
        m, d = c.m, c.d
        a = m.joint(self.side + "_wrist").qposadr[0]
        d.qpos[a : a + 3], d.qpos[a + 3 : a + 7] = self.g["pos"], self.g["quat"]
        q = self.closed.copy()
        targets = np.array(self.g["targets"]) + 0.015 * self.normals
        for _ in range(60):
            d.qpos[c.aa[self.off : self.off + 20]] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            for i, body in enumerate(self.bodies):
                p = d.xpos[body] + d.xmat[body].reshape(3, 3) @ self.offsets[i]
                jac = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jac, None, p, body)
                local = slice(4 * i, 4 * i + 4)
                ids = slice(self.off + 4 * i, self.off + 4 * i + 4)
                J = jac[:, c.dofs[ids]]
                dq = J.T @ np.linalg.solve(J @ J.T + np.eye(3) * 1e-6, targets[i] - p)
                q[local] = np.clip(
                    q[local] + np.clip(dq, -0.05, 0.05),
                    m.actuator_ctrlrange[ids, 0] + 0.015,
                    m.actuator_ctrlrange[ids, 1] - 0.015,
                )
        return q

    def control(self, obs, center, orientation, weight=1):
        c = self.c
        m, d = c.m, c.d
        c.sync(obs)
        pickup = not getattr(c, "pickup_done", False) and c.phase not in ["tripod_hold", "tripod_rotate"]
        n = min(self.n, 3) if pickup and (c.phase != "tripod_lift" or obs["time"] - c.pt < 4) else self.n
        bodies = self.bodies[:n]
        points = [d.xpos[b] + d.xmat[b].reshape(3, 3) @ off for b, off in zip(bodies, self.offsets[:n])]
        normals = (mat(mul(obs["cube_quat"], self.grip_in_cube)) @ self.normals[:n].T).T
        active = np.ones(n)
        points_only = (
            self.side == "l" and c.phase == "arc_align" and os.environ.get("RUBIKS_FINAL_CONTACT_POINTS", "0") == "1"
        )
        points_only |= os.environ.get("RUBIKS_SUPPORT_CONTACT_POINTS", "0") == "1" and not pickup
        use_contacts = (os.environ.get("RUBIKS_SUPPORT_CONTACTS", "0") == "1" and not pickup) or points_only
        if use_contacts:
            c.contact_points(obs)
            groups = [[] for _ in range(n)]
            for co in d.contact:
                if co.dist > 0:
                    continue
                bs = [m.geom_bodyid[g] for g in co.geom]
                names = [m.body(b).name for b in bs]
                cube_side = next((j for j, name in enumerate(names) if name.startswith("cube/")), None)
                if cube_side is None:
                    continue
                hand_side = 1 - cube_side
                for i, finger in enumerate(self.fingers[:n]):
                    if names[hand_side].startswith(self.side + "_" + finger):
                        outward = co.frame[:3].copy() * (1 if cube_side == 0 else -1)
                        groups[i].append((max(-float(co.dist), 1e-5), co.pos.copy(), outward, bs[hand_side]))
            if not hasattr(self, "contact_start"):
                self.contact_start = obs["time"]
            mix = min(1, (obs["time"] - self.contact_start) / 1.5)
            for i, contacts in enumerate(groups):
                if not contacts:
                    active[i] = 1 - mix
                    continue
                weights = np.array([x[0] for x in contacts])
                weights /= weights.sum()
                point = sum(w * x[1] for w, x in zip(weights, contacts))
                normal = sum(w * x[2] for w, x in zip(weights, contacts))
                points[i] = (1 - mix) * points[i] + mix * point
                if not points_only:
                    normals[i] = (1 - mix) * normals[i] + mix * normal
                    normals[i] /= max(np.linalg.norm(normals[i]), 1e-8)
                bodies[i] = max(contacts, key=lambda x: x[0])[3]
        A = np.zeros((6, 3 * n))
        for i, p in enumerate(points):
            x, y, z = p - obs["cube_pos"]
            A[:3, 3 * i : 3 * i + 3] = np.eye(3)
            A[3:, 3 * i : 3 * i + 3] = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]]) / 0.03
            A[:, 3 * i : 3 * i + 3] *= active[i]
        er = np.zeros(3)
        mujoco.mju_subQuat(er, orientation, obs["cube_quat"])
        er = mat(obs["cube_quat"]) @ er
        stiffness = float(os.environ.get("RUBIKS_TRIPOD_KP", "70"))
        damping = 0.8
        if c.phase.startswith("arc_"):
            stiffness = float(os.environ.get("RUBIKS_ARC_SUPPORT_KP", str(stiffness)))
            damping = float(os.environ.get("RUBIKS_ARC_SUPPORT_KD", str(damping)))
        rotation_stiffness = (
            float(os.environ.get("RUBIKS_PARK_ROT_KP", ".07")) if c.phase.startswith("pedestal_") else 0.07
        )
        wrench = np.r_[
            np.array([0, 0, 0.672 * weight])
            + stiffness * (np.asarray(center) - obs["cube_pos"])
            - damping * obs["cube_vel"][:3],
            (rotation_stiffness * er - 0.004 * obs["cube_vel"][3:]) / 0.03,
        ]
        unloading = c.phase == "pedestal_unload" and os.environ.get("RUBIKS_PARK_UNLOAD", "0") == "1"
        continuous_bottom = os.environ.get("RUBIKS_BOTTOM_CONTINUOUS", "0") == "1" and c.phase.startswith("arc_")
        continuous_bottom |= os.environ.get("RUBIKS_BOTTOM_TRANSPORT", "0") == "1" and c.phase in [
            "pedestal_rotate",
            "pedestal_lower",
            "pedestal_unload",
        ]
        normal_limits = np.array([1.8, 1.2, 1.2, 2.5][:n])
        if (
            self.side == "l"
            and os.environ.get("RUBIKS_THUMB_PRESSURE_GUARD", "0") == "1"
            and c.phase.startswith("arc_")
        ):
            depth = max(
                [0.0]
                + [
                    -float(distance)
                    for body, other, distance in obs["contacts"]
                    if body.startswith("l_thumb") and other.startswith("cube/")
                ]
            )
            target = max(1.5, 1.8 - 1200 * max(0.0, depth - 0.00065))
            # Reduce squeeze promptly, then restore it gradually after the contact relaxes.
            self.thumb_pressure_limit = min(getattr(self, "thumb_pressure_limit", 1.8) + 0.01, target)
            normal_limits[0] = self.thumb_pressure_limit
        else:
            self.thumb_pressure_limit = 1.8
        scale = float(getattr(self, "preload_scale", 1.0))
        if not 0.25 <= scale <= 1.0:
            raise ValueError("Support preload scale must be within [0.25, 1]")
        normal_limits *= scale
        squeeze = np.array([1.6, 0.8, 0.8, 0.7][:n]) * scale
        preload = -normals * squeeze[:, None]
        if os.environ.get("RUBIKS_SUPPORT_SOLVER", "legacy") == "hierarchical" and not pickup:
            from grasp_force import allocate

            mode = os.environ.get("RUBIKS_SUPPORT_PRIORITY", "wrench")
            if c.phase.startswith("pedestal_"):
                mode = os.environ.get("RUBIKS_PARK_PRIORITY", mode)
            priority = 3 if mode == "force" else 6
            motor_constraints = None
            if os.environ.get("RUBIKS_SUPPORT_MOTOR_LIMITS", "0") == "1" and c.phase.startswith("arc_"):
                rows = []
                bounds = []
                for i, (body, p) in enumerate(zip(bodies, points)):
                    jac = np.zeros((3, m.nv))
                    mujoco.mj_jac(m, d, jac, None, p, body)
                    ids = slice(self.off + 4 * i, self.off + 4 * i + 4)
                    J = jac[:, c.dofs[ids]]
                    value = obs["hand_qpos"][ids]
                    lo, hi = m.actuator_ctrlrange[ids].T
                    posture = float(os.environ.get("RUBIKS_SUPPORT_POSTURE", ".006")) * (
                        self.closed[4 * i : 4 * i + 4] - value
                    )
                    kp = m.actuator_gainprm[ids, 0]
                    for joint in range(4):
                        row = np.zeros(3 * n)
                        row[3 * i : 3 * i + 3] = J[:, joint]
                        if value[joint] < lo[joint] + 0.06:
                            rows.append(-row)
                            bounds.append(posture[joint] - kp[joint] * (lo[joint] + 0.005 - value[joint]))
                        if value[joint] > hi[joint] - 0.06:
                            rows.append(row)
                            bounds.append(kp[joint] * (hi[joint] - 0.005 - value[joint]) - posture[joint])
                if rows:
                    motor_constraints = (np.asarray(rows), np.asarray(bounds))
            f = allocate(
                A,
                wrench,
                normals,
                normal_limits,
                preload.ravel(),
                primary_rows=priority,
                motor_constraints=motor_constraints,
            )
        else:
            f = preload.ravel() + A.T @ np.linalg.solve(
                A @ A.T + np.eye(len(wrench)) * 1e-4, wrench - A @ preload.ravel()
            )
            step = 0.8 / (np.linalg.norm(A, 2) ** 2 + 0.05)
            for _ in range(12):
                f += step * (A.T @ (wrench - A @ f) + 0.05 * (preload.ravel() - f))
                for i, normal_axis in enumerate(normals):
                    force = f[3 * i : 3 * i + 3]
                    normal = np.clip(-force @ normal_axis, 0.15, normal_limits[i])
                    tangent = force - (force @ normal_axis) * normal_axis
                    tangent *= min(1, 0.8 * normal / max(np.linalg.norm(tangent), 1e-9))
                    f[3 * i : 3 * i + 3] = -normal * normal_axis + tangent
        self.requested_wrench = wrench[:6] * np.array([1, 1, 1, 0.03, 0.03, 0.03])
        self.allocated_wrench = (A[:6] @ f) * np.array([1, 1, 1, 0.03, 0.03, 0.03])
        q = obs["hand_qpos"][self.off : self.off + 20].copy()
        q[4 * self.n :] = self.closed[4 * self.n :]
        for i, (body, p) in enumerate(zip(bodies, points)):
            jac = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jac, None, p, body)
            local = slice(4 * i, 4 * i + 4)
            ids = slice(self.off + 4 * i, self.off + 4 * i + 4)
            J = jac[:, c.dofs[ids]]
            posture = float(os.environ.get("RUBIKS_SUPPORT_POSTURE", ".006"))
            if unloading:
                posture *= weight
            torque = J.T @ f[3 * i : 3 * i + 3] + posture * (self.closed[local] - q[local])
            q[local] += torque / m.actuator_gainprm[ids, 0]
        return np.clip(
            q,
            m.actuator_ctrlrange[self.off : self.off + 20, 0] + 0.005,
            m.actuator_ctrlrange[self.off : self.off + 20, 1] - 0.005,
        )
