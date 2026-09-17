"""Fingertip arc controller. Wrist poses remain fixed throughout contact strokes."""

import json
import os
from pathlib import Path

import mujoco
import numpy as np

from finger_probe import Controller as Pickup
from baseline import quat, mul, mat, blend


class Controller(Pickup):
    def __init__(self, info):
        super().__init__(info)
        path = Path(__file__).parent / "data" / os.environ.get("RUBIKS_ARC_GRASP", "turn_grasp_45.json")
        self.arc = json.loads(path.read_text())
        self.contact_mode = self.arc.get("mode", "side")
        lift = float(os.environ.get("RUBIKS_CONTACT_LIFT", "0"))
        self.arc["pos"][2] += lift
        self.contact_height = float(self.arc.get("height", 0.019)) + lift
        self.angles = np.array(self.arc["angles"])
        self.joints = np.array(self.arc["joints"])
        self.turn_fingers = self.arc.get("fingers", ["thumb", "index_finger"])
        order = ["thumb", "index_finger", "middle_finger", "ring_finger", "pinky_finger"]
        self.turn_ids = np.array([20 + 4 * order.index(f) + j for f in self.turn_fingers for j in range(4)])
        self.pad_bodies = [self.m.body("r_" + f + "_distal").id for f in self.turn_fingers]
        self.pad_offsets = [
            np.array([0, 0.009, -0.024]) if f == "thumb" else np.array([0, 0.0078, -0.020]) for f in self.turn_fingers
        ]
        for f in order:
            if f not in self.turn_fingers:
                j = 4 * order.index(f)
                self.open["r"][j : j + 4] = [1.4, 0.3 if f == "thumb" else 0, 0, 0]
        self.arc_ready = False
        self.arc_log_t = -1
        self.stroke_count = 0
        self.last_points = None
        self.preload = float(os.environ.get("RUBIKS_PRELOAD", ".9"))
        self.tangent_gain = float(os.environ.get("RUBIKS_TASK_KP", "180"))

    def sync(self, obs):
        self.d.qpos[self.aa] = obs["hand_qpos"]
        for side in "lr":
            a = self.m.joint(side + "_wrist").qposadr[0]
            p, q = obs["wrist"][side]
            self.d.qpos[a : a + 3], self.d.qpos[a + 3 : a + 7] = p, q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def contact_points(self, obs):
        """Reconstruct observed cube geometry in the private model for contact Jacobians."""
        core = self.m.body("cube/core").id
        a = self.m.jnt_qposadr[self.m.body_jntadr[core]]
        self.d.qpos[a : a + 3] = obs["cube_pos"]
        self.d.qpos[a + 3 : a + 7] = obs["cube_quat"]
        cubies = [b for b in range(self.m.nbody) if self.m.body_parentid[b] == core]
        for b, R in zip(cubies, obs["cubie_rot"]):
            j = self.m.body_jntadr[b]
            a = self.m.jnt_qposadr[j]
            if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL:
                q = np.empty(4)
                mujoco.mju_mat2Quat(q, R.ravel())
                self.d.qpos[a : a + 4] = q
            else:
                self.d.qpos[a] = obs["face_angles"][self.m.body(b).name[-2:]]
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        points = [[] for _ in self.turn_fingers]
        for co in self.d.contact:
            if co.dist > 0:
                continue
            names = [self.m.body(self.m.geom_bodyid[g]).name for g in co.geom]
            if not any(n.startswith("cube/") for n in names):
                continue
            for i, f in enumerate(self.turn_fingers):
                if any(n.startswith("r_" + f) for n in names):
                    points[i].append(co.pos.copy())
        return [np.mean(p, axis=0) if p else None for p in points]

    def arc_joints(self, angle):
        return np.array([np.interp(angle, self.angles, self.joints[:, i]) for i in range(8)])

    def guard(self, hand, obs):
        if os.environ.get("RUBIKS_RING_GUARD", "0") == "1":
            self.sync(obs)
            self.contact_points(obs)
            actual = obs["hand_qpos"][12:16]
            delta = hand[12:16] - actual
            constraints = []
            for co in self.d.contact:
                bodies = [self.m.geom_bodyid[g] for g in co.geom]
                names = [self.m.body(b).name for b in bodies]
                if not any(n.startswith("l_ring") for n in names):
                    continue
                jac = []
                for body in bodies:
                    J = np.zeros((3, self.m.nv))
                    mujoco.mj_jac(self.m, self.d, J, None, co.pos, body)
                    jac.append(J[:, self.dofs[12:16]])
                gradient = co.frame[:3] @ (jac[1] - jac[0])
                constraints.append((gradient, -0.0006 - float(co.dist)))
            for _ in range(3):
                for gradient, minimum in constraints:
                    norm = gradient @ gradient
                    if norm > 1e-8 and gradient @ delta < minimum:
                        delta += (minimum - gradient @ delta) * gradient / norm
            hand[12:16] = np.clip(
                actual + delta, self.m.actuator_ctrlrange[12:16, 0] + 0.015, self.m.actuator_ctrlrange[12:16, 1] - 0.015
            )
        if os.environ.get("RUBIKS_CONTACT_GUARD", "0") != "1":
            return hand
        depths = np.zeros(10)
        for body, other, distance in obs["contacts"]:
            side = 0 if body.startswith("l_") else 5
            for i, name in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
                if body.startswith(("l_" + name, "r_" + name)):
                    depths[side + i] = max(depths[side + i], -distance)
        return hand

    def point_ik(self, angle, radius, height=None):
        """Private-model IK for an open posture at a given arc endpoint."""
        a = self.m.joint("r_wrist").qposadr[0]
        self.d.qpos[a : a + 3] = self.arc["pos"]
        self.d.qpos[a + 3 : a + 7] = self.arc["quat"]
        q = self.arc_joints(angle)
        height = self.contact_height if height is None else height
        R = mat(quat([0, 0, 1], angle))
        for _ in range(70):
            self.d.qpos[self.aa[self.turn_ids]] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            for i, body in enumerate(self.pad_bodies):
                target = radius * (R @ np.array([1 if i == 0 else -1, 0, 0])) + [0, 0, height]
                p = self.d.xpos[body] + self.d.xmat[body].reshape(3, 3) @ self.pad_offsets[i]
                jac = np.zeros((3, self.m.nv))
                mujoco.mj_jac(self.m, self.d, jac, None, p, body)
                ids = self.turn_ids[4 * i : 4 * i + 4]
                J = jac[:, self.dofs[ids]]
                dq = J.T @ np.linalg.solve(J @ J.T + np.eye(3) * 1e-6, target - p)
                q[i * 4 : i * 4 + 4] = np.clip(
                    q[i * 4 : i * 4 + 4] + np.clip(dq, -0.05, 0.05),
                    self.m.actuator_ctrlrange[ids, 0] + 0.015,
                    self.m.actuator_ctrlrange[ids, 1] - 0.015,
                )
        return q

    def prepare_stroke(self, t, obs, new_move=False):
        self.face = self.moves[self.mi][0]
        self.key = self.keys[self.face]
        actual = obs["face_angles"][self.key]
        if new_move:
            self.stroke_count = 0
            delta = {"": -np.pi / 2, "'": np.pi / 2, "2": np.pi}[self.moves[self.mi][1:]]
            self.move_start = round(actual / (np.pi / 2)) * (np.pi / 2)
            self.move_goal = self.move_start + delta
        self.sign = np.sign(self.move_goal - actual)
        self.local_start = self.angles[0] if self.sign > 0 else self.angles[-1]
        self.stroke_start = actual
        span = np.radians(float(os.environ.get("RUBIKS_STROKE_DEGREES", self.arc.get("degrees", 45))))
        self.stroke_goal = actual + self.sign * min(abs(self.move_goal - actual), span)
        self.fq = mul(obs["cube_quat"], self.faceq[self.face])
        yaw = self.current_yaw + actual - self.move_start - self.local_start
        self.arc_yaw = yaw
        self.arc_frame = mul(self.fq, quat([0, 0, 1], yaw))
        self.arc_rotation = mat(self.arc_frame)
        self.stroke_wrist = (
            obs["cube_pos"] + self.arc_rotation @ self.arc["pos"],
            mul(self.arc_frame, self.arc["quat"]),
        )
        self.stroke_open = (
            self.open_ik(self.local_start) if self.contact_mode == "top" else self.point_ik(self.local_start, 0.038)
        )
        self.route_start_joints = obs["hand_qpos"][self.turn_ids].copy()
        self.stroke_closed = self.arc_joints(self.local_start)
        self.last_points = None
        self.stroke_count += 1
        self.phase_to("arc_route", t)

    def act(self, t, obs):
        if not self.arc_ready:
            self.arc_ready = True
            self.prepare_stroke(t, obs, new_move=True)
        h = np.r_[self.closed["l"], self.open["r"]]
        w = {"l": self.pose("l", self.center, self.lframe), "r": self.stroke_wrist}
        if hasattr(self, "support_action"):
            self.support_action(obs, h, w)
        dt = t - self.pt
        phase = self.phase
        p, q = self.stroke_wrist
        outward = self.arc_rotation[:, 2]
        h[self.turn_ids] = self.stroke_open
        if phase == "arc_route":
            h[self.turn_ids] = blend(self.route_start_joints, self.stroke_open, dt / 1.2)
            tangent = np.cross([0, 0, 1], outward)
            if np.linalg.norm(tangent) < 0.2:
                tangent = np.array([1.0, 0, 0])
            tangent /= np.linalg.norm(tangent)
            w["r"] = (p + 0.085 * outward - 0.24 * tangent, q)
            if dt >= 1.8:
                self.phase_to("arc_outer", t)
        if int(t) != self.arc_log_t:
            self.arc_log_t = int(t)
            print(
                "ARC",
                round(t, 2),
                phase,
                "face",
                self.face,
                "angle",
                round(obs["face_angles"][self.key], 4),
                "cube",
                np.round(obs["cube_pos"], 4),
                "misalign",
                round(obs["misalign_deg"], 2),
                "contacts",
                sorted(set(x[0] for x in obs["contacts"])),
                flush=True,
            )
        return {"hand": self.guard(h, obs), "wrist": w}
