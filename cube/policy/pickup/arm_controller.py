"""Finger-turn solver adapted to finite-speed arms.

Private floating-hand geometry is used for finger planning only. Physical
execution uses the arm model and motor limits supplied by the runner.
"""

import os

import numpy as np

from baseline import mat, mul
from deterministic_controller import Controller as FingerController
from regrasp_controller import conjugate

GRASP_DEFAULTS = {
    "RUBIKS_ARM_BENT_SUPPORT": "1",
    "RUBIKS_ARM_BENT_MOVES": "L",
    "RUBIKS_ARM_BENT_GRASP": "support_grasp4_pickup.json",
}

SEQUENCE_DEFAULTS = {
    "RUBIKS_PICKUP_OBSERVED": "1",
    "RUBIKS_SUPPORT_SOLVER": "hierarchical",
    "RUBIKS_SUPPORT_PRIORITY": "force",
    "RUBIKS_BOTTOM_CONTINUOUS": "1",
    "RUBIKS_BOTTOM_TRANSPORT": "1",
    "RUBIKS_BOTTOM_ALIGNMENT": ".08",
    "RUBIKS_BOTTOM_DAMPING": ".003",
    "RUBIKS_INTERMEDIATE_ALIGN": "1",
    "RUBIKS_ARM_LEVEL_PLACE": "1",
    "RUBIKS_ARM_LEVEL_CHOICE": "1",
    "RUBIKS_PARK_SETTLE": "1",
    "RUBIKS_PARK_UNLOAD": "1",
    "RUBIKS_PARK_FEEDBACK": "1",
    "RUBIKS_PARK_CAPTURE": "0",
    "RUBIKS_PARK_SERVO": "1",
    "RUBIKS_ARM_PICKUP_PLAN": "1",
    "RUBIKS_ARM_ADAPTIVE_SUPPORT": "1",
    "RUBIKS_ALIGN_ON_PEDESTAL": "1",
    "RUBIKS_PARK_ACTIVE_ALIGNMENT": "1",
    "RUBIKS_PARK_ALIGNMENT": ".16",
    "RUBIKS_PARK_UNLOAD_TIMEOUT": "20",
    "RUBIKS_PARK_CONTACT_RELEASE": "1",
    "RUBIKS_ARM_BENT_RETAIN": "1",
    "RUBIKS_ARM_PREPARE_PICKUP": "1",
    "RUBIKS_ARM_BENT_PICKUP_GUARD": "1",
    "RUBIKS_ARM_BENT_CONTINUE": "1",
    "RUBIKS_ARM_BENT_PARK": "single",
    "RUBIKS_ARM_BENT_PLACE_PLAN": "1",
    "RUBIKS_ARM_BENT_CLOCKWISE_TORQUE": ".045",
    "RUBIKS_ARM_BENT_ALIGNMENT_FACES": "U",
    "RUBIKS_ARC_RELEASE_RECOVERY": "1",
}


class ArmFingerController(FingerController):
    def continues_grasp(self):
        return (
            os.environ.get("RUBIKS_ARM_BENT_CONTINUE", "0") == "1"
            and getattr(self, "bent_support", False)
            and not self.parking
            and self.mi != self.last_started_move
            and self.mi < len(self.moves)
            and self.moves[self.mi][0] == self.grip_face
        )

    def act(self, t, obs):
        return super().act(t, obs)

    def select_support(self):
        """Change grasp geometry only before a pickup, with the cube supported."""
        from tripod import Tripod

        move = self.moves[self.mi]
        choices = os.environ.get("RUBIKS_ARM_BENT_MOVES", "D' L").split()
        retained = getattr(self, "bent_support", False) and os.environ.get("RUBIKS_ARM_BENT_RETAIN", "0") == "1"
        bent = (move in choices or move[0] in choices or retained) and self.mi >= int(
            os.environ.get("RUBIKS_ARM_BENT_FROM", "0")
        )
        path = (
            os.environ.get("RUBIKS_ARM_BENT_GRASP", "support_grasp_bent.json") if bent else "support_grasp_force.json"
        )
        if getattr(self, "support_path", "support_grasp_force.json") != path:
            frame = self.tripod.grip_in_cube.copy()
            self.tripod = Tripod(self, grasp_path=path)
            self.tripod.grip_in_cube = frame
            self.nominal_grasp = {key: np.array(self.tripod.g[key]).copy() for key in ["pos", "quat"]}
        self.support_path, self.bent_support = path, bent
        print("ARM_SUPPORT_SELECTED", self.mi, move, path, flush=True)


class Controller:
    def __init__(self, info):
        for key, value in GRASP_DEFAULTS.items():
            os.environ.setdefault(key, value)
        quarters = sum(2 if move.endswith("2") else 1 for move in info["solution"].split())
        if quarters > 1:
            for key, value in SEQUENCE_DEFAULTS.items():
                os.environ.setdefault(key, value)
        planning_info = dict(info, model_path=info["planning_model_path"], arm_model_path=info["model_path"])
        self.fingers = ArmFingerController(planning_info)
        self.fingers.select_support()
        if os.environ.get("RUBIKS_INTERMEDIATE_ALIGN", "0") == "1":
            import mujoco
            from scene import apply_scramble, facelets

            reference = mujoco.MjData(self.fingers.m)
            self.fingers.expected_facelets = []
            for index in range(len(self.fingers.moves)):
                sequence = info["scramble"] + " " + " ".join(self.fingers.moves[: index + 1])
                apply_scramble(self.fingers.m, reference, sequence)
                self.fingers.expected_facelets.append(facelets(self.fingers.m, reference)[0])
        self.nominal_grasp = {key: np.array(self.fingers.tripod.g[key]).copy() for key in ["pos", "quat"]}
        self.grasp_calibrated = False
        self.fingers.center = np.array([-0.18, 0.0, 0.35])
        offset = float(os.environ.setdefault("RUBIKS_ARM_CONTACT_OFFSET", "-0.007"))
        self.fingers.arc["pos"][2] += offset
        for finger in range(5):
            if 20 + 4 * finger not in self.fingers.turn_ids:
                # Curl the distal joints instead of folding the proximal link
                # back against the forearm, as the floating-hand solver did.
                self.fingers.open["r"][4 * finger : 4 * finger + 4] = [0.9, 0.0, 1.4, 0.7]
                if finger == 2:
                    self.fingers.open["r"][4 * finger] = 1.0
        self.clock = 0.0
        self.previous_time = 0.0
        self.stalled_since = None
        self.blocked_since = None
        self.transport_phase = None
        self.transport_joints = None
        self.transport_mode = os.environ.setdefault("RUBIKS_ARM_TRANSPORT_MODE", "force")
        self.rest_wrist = (np.array([-0.30, -0.32, 0.48]), np.array(info["wrist_init"]["r"][1]))

    def act(self, t, obs):
        # A calibrated grasp uses its own support policy. Keep the original
        # sequence policy for other grasps and, by default, for placement.
        f = self.fingers
        use_profile = getattr(f, "bent_support", False)
        placing = f.parking or (f.mi != f.last_started_move and not f.continues_grasp())
        use_profile &= not placing or os.environ.get("RUBIKS_ARM_BENT_PARK", "sequence") == "single"
        use_profile &= os.environ.get("RUBIKS_ARM_BENT_POLICY", "single") == "single"
        settings = {}
        if use_profile:
            face = f.grip_face
            settings = {
                "RUBIKS_ARM_ADAPTIVE_SUPPORT": "0",
                "RUBIKS_SUPPORT_SOLVER": "legacy",
                "RUBIKS_BOTTOM_CONTINUOUS": "0",
                "RUBIKS_BOTTOM_TRANSPORT": "0",
                "RUBIKS_BOTTOM_ALIGNMENT": ".035",
                "RUBIKS_BOTTOM_DAMPING": "0",
                "RUBIKS_BOTTOM_LEGACY_ORDER": "0",
                "RUBIKS_ALIGN_LOAD": ".7" if face == "L" else ".3",
                "RUBIKS_ALIGN_FEEDBACK": "0",
                "RUBIKS_FINAL_RELIEF_SCALE": "0" if face == "L" else "1",
                "RUBIKS_FINAL_CONTACT_POINTS": "1" if face == "L" else "0",
                "RUBIKS_FINAL_INDEX_RELIEF": "4000" if face == "L" else "0",
            }
            if not f.pickup_done and os.environ.get("RUBIKS_ARM_BENT_PICKUP_GUARD", "0") == "1":
                settings.update(RUBIKS_RING_GUARD="1", RUBIKS_CONTACT_GUARD="1")
        previous = {key: os.environ.get(key) for key in settings}
        os.environ.update(settings)
        try:
            return self._act(t, obs)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _act(self, t, obs):
        # Pedestal regrasp in the legacy planner restores its floating-hand
        # workspace. Keep every subsequent pickup within the arms' workspace.
        self.fingers.center[:] = [-0.18, 0.0, 0.35]
        dt = max(0.0, t - self.previous_time)
        self.previous_time = t
        self.prepare_relaxed_route(t, obs)
        errors = list(obs["wrist_error"].values())
        position_error = max((e[0] for e in errors), default=0.0)
        rotation_error = max((e[1] for e in errors), default=0.0)
        lagging = position_error > 0.012 or rotation_error > 0.12 or obs["arm_collision_blocked"]
        if obs["arm_collision_blocked"]:
            since = getattr(self, "blocked_since", None)
            self.blocked_since = t if since is None else since
            if t - self.blocked_since > 15:
                raise RuntimeError("Arm motion remained collision-blocked despite replanning")
        else:
            self.blocked_since = None
        if lagging and not any(obs.get("arm_path_pending", {}).values()):
            self.stalled_since = t if self.stalled_since is None else self.stalled_since
        else:
            self.stalled_since = None
        # Give abrupt approach/regrasp targets time to settle before closing.
        self.clock += dt * (0.0 if lagging else 0.75)
        previous_phase = self.fingers.phase
        was_inverting = self.fingers.inverting
        finger_obs = dict(obs, time=self.clock, physics_time=t)
        self.fingers.nominal_grasp = self.nominal_grasp
        command = self.fingers.act(self.clock, finger_obs)
        if was_inverting and not self.fingers.inverting:
            # The achieved grasp differs slightly from its geometric seed.
            # Preserve it when switching from whole-cube transport to strokes.
            p, q = obs["wrist"]["l"]
            frame = mul(obs["cube_quat"], self.fingers.tripod.grip_in_cube)
            self.fingers.tripod.g["pos"] = mat(frame).T @ (p - obs["cube_pos"])
            self.fingers.tripod.g["quat"] = mul(conjugate(frame), q)
            self.grasp_calibrated = True
            self.fingers.support_strained = bool(obs["hand_qpos"][4] < self.fingers.tripod.closed[4] - 0.002)
        command["phase"] = previous_phase if previous_phase.startswith("arc_") else self.fingers.phase
        command["hand"] = np.asarray(command["hand"], float)
        if previous_phase in {"tripod_lift", "invert_cube", "pedestal_rotate", "pedestal_lower"}:
            command["carry_replan"] = True
            # Carry the grasp with the achieved wrist. Otherwise finger forces
            # try to move the cube ahead of a finite-speed arm and lose contact.
            f = self.fingers
            if previous_phase == "tripod_lift":
                relative_pos = mat(f.tripod.grip_in_cube) @ f.tripod.g["pos"]
                relative_quat = mul(f.tripod.grip_in_cube, f.tripod.g["quat"])
            elif previous_phase == "invert_cube":
                relative_pos, relative_quat = f.invert_relative_pos, f.invert_relative_quat
            p, q = obs["wrist"]["l"]
            orientation = mul(q, conjugate(relative_quat))
            center = p - mat(orientation) @ relative_pos
            support = f.tripod.control(finger_obs, center, orientation)
            if previous_phase == "tripod_lift" and self.clock - f.pt < 4:
                support[12:16] = command["hand"][12:16]
            command["hand"][:20] = support
            if os.environ.get("RUBIKS_RING_GUARD", "0") == "1" or os.environ.get("RUBIKS_CONTACT_GUARD", "0") == "1":
                command["hand"] = f.guard(command["hand"], finger_obs)
        self.transport_phase = previous_phase
        if self.fingers.phase == "arc_route":
            outward = self.fingers.arc_rotation[:, 2]
            tangent = np.cross([0, 0, 1], outward)
            if np.linalg.norm(tangent) < 0.2:
                tangent = np.array([1.0, 0, 0])
            tangent /= np.linalg.norm(tangent)
            p, q = self.fingers.stroke_wrist
            command["wrist"]["r"] = (p + 0.085 * outward - 0.10 * tangent, q)
            if self.relaxed_route_enabled():
                command["hand"][20:] = np.tile([0.15, 0.0, 0.3, 0.2], 5)
                if not self.relaxed_route_ready():
                    command["wrist"]["r"] = tuple(x.copy() for x in obs["wrist"]["r"])
        if previous_phase == "arc_route" and self.relaxed_route_enabled():
            command["hand"][20:] = np.tile([0.15, 0.0, 0.3, 0.2], 5)
        right_idle = not self.fingers.pickup_done or self.fingers.inverting or self.fingers.parking
        right_touching = any(body.startswith("r_") and other.startswith("cube/") for body, other, _ in obs["contacts"])
        if right_idle and not right_touching:
            # The original free wrist parks beyond arm reach during inversion.
            command["wrist"]["r"] = self.rest_wrist
            command["hand"][20:] = np.tile([0.15, 0.0, 0.3, 0.2], 5)
        return command

    def relaxed_route_ready(self):
        f = self.fingers
        return getattr(self, "relaxed_route_key", None) == (f.mi, f.stroke_count, f.pt)

    def prepare_relaxed_route(self, t, obs):
        f = self.fingers
        if (
            not self.relaxed_route_enabled()
            or f.phase != "arc_route"
            or f.parking
            or (f.mi != f.last_started_move and not f.continues_grasp())
            or self.relaxed_route_ready()
        ):
            return None
