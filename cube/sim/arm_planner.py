"""Bounded, deterministic IK restarts and joint-space RRT for one arm.

All queries use private MuJoCo data. Planning freezes the other arm and either
freezes the cube or carries it with the wrist; execution rechecks collisions.
"""

import mujoco
import numpy as np


class RegraspPlanner:
    def __init__(self, model, joints, moving_bodies, limits, carry=False):
        self.m = model
        self.d = mujoco.MjData(model)
        self.qadr = model.jnt_qposadr[joints]
        self.dofs = model.jnt_dofadr[joints]
        self.moving = set(moving_bodies)
        self.hand = {b for b in self.moving if model.body(b).name.endswith("_wrist")}
        for b in range(model.nbody):
            if model.body_parentid[b] in self.hand:
                self.hand.add(b)
        self.limits = limits
        self.rng = np.random.default_rng(7)
        self.carry = carry
        self.cube = {model.body("cube/core").id}
        for b in range(model.nbody):
            if model.body_parentid[b] in self.cube:
                self.cube.add(b)

    def free(self, q, clearance=0.0005):
        m, d = self.m, self.d
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        for c in d.contact:
            bodies = m.geom_bodyid[c.geom]
            # With finger joints fixed, same-hand contacts are invariant under
            # arm motion. They remain enabled in the physical simulation.
            if all(b in self.cube for b in bodies):
                continue
            moving = self.moving | self.cube if self.carry else self.moving
            if c.dist < clearance and any(b in moving for b in bodies):
                return False
        return True

    def edge(self, a, b):
        count = max(1, int(np.ceil(np.max(np.abs(b - a)) / 0.06)))
        return all(self.free(a + (b - a) * u) for u in np.linspace(0, 1, count + 1)[1:])

    def goals(self, body, pos, quat, start):
        m, d = self.m, self.d
        candidates = []
        lo, hi = self.limits[:, 0] + 0.02, self.limits[:, 1] - 0.02
        for seed in range(64):
            d.qpos[self.qadr] = start if seed == 0 else self.rng.uniform(lo, hi)
            for _ in range(100):
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                dq = np.zeros(3)
                mujoco.mju_subQuat(dq, quat, d.xquat[body])
                err = np.r_[pos - d.xpos[body], 0.15 * d.xmat[body].reshape(3, 3) @ dq]
                if np.linalg.norm(err) < 0.0003:
                    break
                jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                mujoco.mj_jacBody(m, d, jp, jr, body)
                J = np.vstack((jp[:, self.dofs], 0.15 * jr[:, self.dofs]))
                step = J.T @ np.linalg.solve(J @ J.T + 0.0001 * np.eye(6), err)
                d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + np.clip(step, -0.15, 0.15), lo, hi)
            q = d.qpos[self.qadr].copy()
            if np.linalg.norm(err) < 0.001 and self.free(q, clearance=0.001):
                candidates.append(q)
        return sorted(candidates, key=lambda q: np.linalg.norm(q - start))[:4]

    @staticmethod
    def branch(nodes, parents, index):
        path = []
        while index >= 0:
            path.append(nodes[index])
            index = parents[index]
        return path[::-1]

    def connect(self, start, goal):
        if self.edge(start, goal):
            return [start, goal]
        trees = [([start], [-1]), ([goal], [-1])]

        def extend(tree, target):
            nodes, parents = tree
            near = int(np.argmin(np.linalg.norm(np.asarray(nodes) - target, axis=1)))
            delta = target - nodes[near]
            distance = np.linalg.norm(delta)
            q = nodes[near] + delta * min(1.0, 0.3 / max(distance, 1e-12))
            if not self.edge(nodes[near], q):
                return None, False
            nodes.append(q)
            parents.append(near)
            return len(nodes) - 1, distance <= 0.3

        for iteration in range(800):
            side = iteration % 2
            tree, other = trees[side], trees[1 - side]
            sample = self.rng.uniform(self.limits[:, 0] + 0.02, self.limits[:, 1] - 0.02)
            if iteration % 5 == 0:
                sample = other[0][0]
            index, _ = extend(tree, sample)
            if index is None:
                continue
            for _ in range(60):
                opposite, reached = extend(other, tree[0][index])
                if opposite is None:
                    break
                if reached:
                    a = self.branch(*tree, index)
                    b = self.branch(*other, opposite)
                    path = a + b[-2::-1] if side == 0 else b + a[-2::-1]
                    # Deterministic shortcutting removes unnecessary bends.
                    smooth = [path[0]]
                    i = 0
                    while i < len(path) - 1:
                        j = len(path) - 1
                        while j > i + 1 and not self.edge(path[i], path[j]):
                            j -= 1
                        smooth.append(path[j])
                        i = j
                    return smooth

    def plan(self, data, body, pos, quat):
        self.d.qpos[:] = data.qpos
        self.wrist = body
        start = data.qpos[self.qadr].copy()
        # Real compliant contacts can have a few micrometres of penetration.
        # Permit starting there; every subsequent edge sample must clear the
        # ordinary positive margin, and physical acceptance stays unchanged.
        for goal in self.goals(body, np.asarray(pos), np.asarray(quat), start):
            path = self.connect(start, goal)
            if path:
                return path[1:]
        return None
