"""Generic bimanual mechanism and bounded joint-space control (SI units).

These are simulation arms, not a calibrated commercial robot. Each has a
three-axis shoulder, an elbow, and a three-axis wrist on a fixed shared base.
"""

import mujoco
import numpy as np
from arm_ik import step as ik_step

JOINTS = ("shoulder_yaw", "shoulder_pitch", "shoulder_roll", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
RANGES = np.array([[-2.7, 2.7], [-2.4, 2.4], [-2.7, 2.7], [-1.6, 1.9], [-2.9, 2.9], [-2.5, 2.5], [-2.9, 2.9]])
TORQUE = np.array([70.0, 70.0, 45.0, 50.0, 15.0, 15.0, 15.0])
KP = np.array([900.0, 900.0, 600.0, 700.0, 160.0, 160.0, 160.0])
KD = np.array([55.0, 55.0, 35.0, 40.0, 8.0, 8.0, 8.0])
SPEED = 1.5  # rad/s, joint target
ACCEL = 5.0  # rad/s^2, joint target
HAND_SPEED = 4.0
HAND_ACCEL = 60.0


def add_base(spec):
    base = spec.worldbody.add_body(name="robot_base", pos=(-0.62, 0, 0))
    base.add_geom(
        name="base_foot",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=(0, 0, 0.025),
        size=(0.23, 0.025, 0),
        rgba=(0.18, 0.22, 0.26, 1),
    )
    base.add_geom(
        name="base_column",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=(0, 0, 0.25),
        size=(0.065, 0.20, 0),
        rgba=(0.35, 0.40, 0.44, 1),
    )
    base.add_geom(
        name="torso",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0, 0, 0.48),
        size=(0.065, 0.22, 0.075),
        rgba=(0.75, 0.79, 0.81, 1),
    )
    return base


def add_arm(spec, base, side, wrist_pose):
    sign = 1 if side == "l" else -1
    shoulder = np.array([-0.62, sign * 0.28, 0.48])
    elbow = np.array([-0.48, sign * 0.48, 0.20])
    wrist, quat = map(np.asarray, wrist_pose)
    upper = elbow - shoulder
    adapter = np.array([0.07, 0, 0])
    lower = wrist - adapter - elbow
    elbow_axis = np.cross(upper, lower)
    elbow_axis /= np.linalg.norm(elbow_axis)
    shoulder_body = base.add_body(name=f"{side}_arm_upper", pos=shoulder - np.asarray(base.pos))
    elbow_body = shoulder_body.add_body(name=f"{side}_arm_forearm", pos=upper)
    mount = elbow_body.add_body(name=f"{side}_arm_mount", pos=lower)
    for body, endpoint, radius, mass in ((shoulder_body, upper, 0.036, 2.2), (elbow_body, lower, 0.029, 1.4)):
        body.add_geom(
            name=body.name + "_shell",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, *endpoint],
            size=(radius, 0, 0),
            mass=mass,
            margin=0.0,
            gap=0.01,
            rgba=(0.78, 0.82, 0.85, 1),
        )
    mount.add_geom(
        name=f"{side}_arm_adapter",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, 0, 0, *adapter],
        size=(0.014, 0, 0),
        mass=0.18,
        rgba=(0.25, 0.30, 0.34, 1),
    )
    bodies = [shoulder_body] * 3 + [elbow_body] + [mount] * 3
    axes = [(0, 0, 1), (0, 1, 0), (1, 0, 0), elbow_axis, (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    for i, (name, body, axis) in enumerate(zip(JOINTS, bodies, axes)):
        name = f"{side}_arm_{name}"
        body.add_joint(
            name=name,
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=axis,
            limited=True,
            range=RANGES[i],
            damping=0.3,
            armature=0.03,
        )
        spec.add_actuator(
            name=name,
            target=name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrllimited=True,
            ctrlrange=(-TORQUE[i], TORQUE[i]),
            forcelimited=True,
            forcerange=(-TORQUE[i], TORQUE[i]),
        )
    return mount.add_frame(pos=adapter, quat=quat)


def descendants(model, root):
    ids = {model.body(root).id}
    for i in range(model.nbody):
        if model.body_parentid[i] in ids:
            ids.add(i)
    return ids


class JointFilter:
    """Continuous targets with bounded velocity and acceleration, including reversals."""

    def __init__(self, q, speed, acceleration, synchronized=False, frequency=10.0):
        self.q = np.array(q, dtype=float, copy=True)
        self.v = np.zeros_like(self.q)
        self.speed, self.acceleration = speed, acceleration
        self.synchronized = synchronized
        self.frequency = frequency

    def step(self, target, dt):
        error = np.asarray(target) - self.q
        # A critically damped attractor avoids the chatter of bang-bang braking.
        acceleration = self.frequency**2 * error - 2 * self.frequency * self.v
        if self.synchronized:
            acceleration *= min(1.0, self.acceleration / max(np.max(np.abs(acceleration)), 1e-12))
            velocity = self.v + acceleration * dt
            velocity *= min(1.0, self.speed / max(np.max(np.abs(velocity)), 1e-12))
            change = velocity - self.v
            change *= min(1.0, self.acceleration * dt / max(np.max(np.abs(change)), 1e-12))
            self.v += change
        else:
            acceleration = np.clip(acceleration, -self.acceleration, self.acceleration)
            self.v = np.clip(self.v + acceleration * dt, -self.speed, self.speed)
        self.q += self.v * dt
        return self.q.copy()


def replan_order(errors, collision_pairs=()):
    """Resolve a collision's participating arm before unrelated tracking error."""
    blocked = {side for pair in collision_pairs for name in pair for side in "lr" if name.startswith(side + "_")}
    return sorted("lr", key=lambda side: (side not in blocked, -errors[side][0]))


class ArmDrive:
    """Damped IK on private data, then finite motor torque on the physical arms."""

    def __init__(self, model, data):
        self.m = model
        self.plan = mujoco.MjData(model)
        self.joints = {s: np.array([model.joint(f"{s}_arm_{j}").id for j in JOINTS]) for s in "lr"}
        self.qadr = {s: model.jnt_qposadr[j] for s, j in self.joints.items()}
        self.dofs = {s: model.jnt_dofadr[j] for s, j in self.joints.items()}
        self.act = {s: np.array([model.actuator(f"{s}_arm_{j}").id for j in JOINTS]) for s in "lr"}
        self.filters = {s: JointFilter(data.qpos[self.qadr[s]], SPEED, ACCEL, synchronized=True) for s in "lr"}
        self.goals = {s: (data.body(f"{s}_wrist").xpos.copy(), data.body(f"{s}_wrist").xquat.copy()) for s in "lr"}
        self.target = {s: data.qpos[self.qadr[s]].copy() for s in "lr"}
        self.error = {s: np.zeros(2) for s in "lr"}
        self.robot = descendants(model, "robot_base")
        self.hand = descendants(model, "l_wrist") | descendants(model, "r_wrist")
        self.arm = self.robot - self.hand
        self.blocked = False
        self.collision_pairs = []
        self.paths = {s: [] for s in "lr"}
        self.path_goals = {}
        self.stuck = {s: 0.0 for s in "lr"}
        self.last_replan = -100.0
        self.allow_carry_replan = False

    def plan_targets(self, data, goals, dt):
        m, p = self.m, self.plan
        large_moves = [
            s
            for s, (pos, quat) in goals.items()
            if np.linalg.norm(pos - self.goals[s][0]) > 0.06 or abs(quat @ self.goals[s][1]) < 0.98
        ]
        self.goals.update(goals)
        for s, path in self.paths.items():
            if not path:
                continue
            pos, quat = self.goals[s]
            previous_pos, previous_quat = self.path_goals[s]
            if np.linalg.norm(pos - previous_pos) > 0.025 or abs(quat @ previous_quat) < 0.99:
                path.clear()
            elif np.linalg.norm(self.filters[s].q - path[0]) < 0.005 and np.max(np.abs(self.filters[s].v)) < 0.02:
                path.pop(0)
                self.filters[s].v[:] = 0
        for s in large_moves:
            if not self.paths[s] and self.can_replan(data, s):
                self.replan(data, s)
        p.qpos[:] = data.qpos
        # Start at the last command, so IK doesn't chase servo deflection.
        for s in "lr":
            p.qpos[self.qadr[s]] = self.filters[s].q
        for _ in range(8):
            mujoco.mj_kinematics(m, p)
            mujoco.mj_comPos(m, p)
            for s, (pos, quat) in self.goals.items():
                if self.paths[s]:
                    p.qpos[self.qadr[s]] = self.paths[s][0]
                    continue
                body = m.body(f"{s}_wrist").id
                dq = np.zeros(3)
                mujoco.mju_subQuat(dq, quat, p.xquat[body])
                rotation_error = p.xmat[body].reshape(3, 3) @ dq
                err = np.r_[np.asarray(pos) - p.xpos[body], 0.15 * rotation_error]
                jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                mujoco.mj_jacBody(m, p, jp, jr, body)
                J = np.vstack((jp[:, self.dofs[s]], 0.15 * jr[:, self.dofs[s]]))
                q = p.qpos[self.qadr[s]]
                p.qpos[self.qadr[s]] = q + ik_step(J, err, q, RANGES)
            # Repel approaching arm collisions while they still have clearance.
            mujoco.mj_kinematics(m, p)
            mujoco.mj_comPos(m, p)
            mujoco.mj_collision(m, p)
            for contact in p.contact:
                b1, b2 = m.geom_bodyid[contact.geom]
                if not (b1 in self.arm or b2 in self.arm) or contact.dist >= 0.005:
                    continue
        old = {s: (f.q.copy(), f.v.copy()) for s, f in self.filters.items()}
        candidates = {s: p.qpos[self.qadr[s]].copy() for s in "lr"}
        for s in "lr":
            p.qpos[self.qadr[s]] = old[s][0]
        mujoco.mj_kinematics(m, p)
        mujoco.mj_collision(m, p)
        previous_clearance = {}
        for contact in p.contact:
            pair = tuple(sorted(contact.geom))
            previous_clearance[pair] = min(previous_clearance.get(pair, np.inf), contact.dist)
        for s in "lr":
            carrying_path = bool(self.paths[s]) and self.allow_carry_replan and self.touches_cube(data, s)
            self.filters[s].speed = 0.45 if carrying_path else SPEED
            self.filters[s].acceleration = 1.0 if carrying_path else ACCEL
            self.target[s] = self.filters[s].step(candidates[s], dt)
            p.qpos[self.qadr[s]] = self.target[s]
        mujoco.mj_kinematics(m, p)
        mujoco.mj_collision(m, p)
        self.blocked = False
        self.collision_pairs = []
        # Finger/cube contact is intentional. Arm contact with anything except
        # the stationary floor/base mounting is rejected before commanding it.
        for contact in p.contact:
            b1, b2 = m.geom_bodyid[contact.geom]
        for s, (pos, quat) in self.goals.items():
            body = data.body(f"{s}_wrist")
            dq = np.zeros(3)
            mujoco.mju_subQuat(dq, quat, body.xquat)
            self.error[s] = np.array([np.linalg.norm(np.asarray(pos) - body.xpos), np.linalg.norm(dq)])
            stalled = self.blocked or (
                np.linalg.norm(self.filters[s].v) < 0.08 and (self.error[s][0] > 0.012 or self.error[s][1] > 0.08)
            )
            self.stuck[s] = self.stuck[s] + dt if stalled else 0.0
        if data.time - self.last_replan > 3:
            pairs = self.collision_pairs if getattr(self, "prioritize_blocked_arm", False) else ()
            for s in replan_order(self.error, pairs):
                if self.stuck[s] < 0.5 or self.paths[s]:
                    continue

    def touches_cube(self, data, side):
        hand = descendants(self.m, f"{side}_wrist")
        return any(
            c.dist <= 0
            and any(self.m.geom_bodyid[g] in hand for g in c.geom)
            and any(self.m.body(self.m.geom_bodyid[g]).name.startswith("cube/") for g in c.geom)
            for c in data.contact
        )

    def can_replan(self, data, side):
        if not self.touches_cube(data, side):
            return True

    def replan(self, data, side):
        from arm_planner import RegraspPlanner

        m = self.m
        planner = RegraspPlanner(
            m, self.joints[side], descendants(m, f"{side}_arm_upper"), RANGES, carry=self.touches_cube(data, side)
        )
        self.last_replan = data.time
        start = mujoco.MjData(m)
        start.qpos[:] = data.qpos
        for s in "lr":
            start.qpos[self.qadr[s]] = self.filters[s].q
        mujoco.mj_forward(m, start)
        path = planner.plan(start, m.body(f"{side}_wrist").id, *self.goals[side])
        print("ARM_REPLAN", round(data.time, 2), side, "waypoints", len(path) if path else 0, flush=True)
        if path:
            self.paths[side] = path
            self.path_goals[side] = tuple(x.copy() for x in self.goals[side])
            self.stuck[side] = 0.0

    def actuate(self, data):
        for s in "lr":
            torque = KP * (self.target[s] - data.qpos[self.qadr[s]])
            torque += KD * (self.filters[s].v - data.qvel[self.dofs[s]])
            # Gravity/Coriolis feedforward still goes through bounded motors.
            torque += data.qfrc_bias[self.dofs[s]]
            data.ctrl[self.act[s]] = np.clip(torque, -TORQUE, TORQUE)
