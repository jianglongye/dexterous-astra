"""Pick a face-specific support grasp, then perform finger turns with that face pointing down."""

import os
import numpy as np
from tripod_controller import Controller as TripodController
from regrasp_controller import interpolate, conjugate
from baseline import quat, mul, mat


class Controller(TripodController):
    def __init__(self, info):
        defaults = {
            "RUBIKS_SUPPORT_GRASP": "support_grasp_force.json",
            "RUBIKS_ARC_GRASP": "turn_grasp_force.json",
            "RUBIKS_ARC_CONTROL": "wrench",
            "RUBIKS_PRELOAD": "1.1",
            "RUBIKS_STROKE_DEGREES": "35",
            "RUBIKS_ARC_LEAD": ".4",
            "RUBIKS_TRIPOD_KP": "120",
            "RUBIKS_BOTTOM_ALIGNMENT": ".035",
        }
        for k, v in defaults.items():
            os.environ.setdefault(k, v)
        super().__init__(info)
        self.grip_face = self.moves[0][0]
        yaw = {"U": 0, "D": 0, "R": np.pi / 2, "L": -np.pi / 2, "F": 0, "B": np.pi}[self.grip_face]
        self.current_yaw = yaw
        self.tripod.grip_in_cube = mul(self.faceq[self.grip_face], quat([0, 0, 1], yaw))
        self.face_down = os.environ.get("RUBIKS_FACE_DOWN", "1") == "1"
        self.center[2] = 0.45 if self.face_down else 0.33
        self.inverted = False
        self.inverting = False
        self.invert_log = -1

    def act(self, t, obs):
        if self.pickup_done and not self.inverted and not self.inverting:
            self.inverting = True
            self.invert_start = obs["cube_quat"].copy()
            self.invert_goal = mul(
                quat([1, 0, 0], np.pi) if self.face_down else np.array([1.0, 0, 0, 0]),
                conjugate(self.tripod.grip_in_cube),
            )
            self.invert_duration = max(
                2.0, 2 * np.arccos(np.clip(abs(self.invert_start @ self.invert_goal), 0, 1)) / 0.6
            )
            p, q = obs["wrist"]["l"]
            self.invert_relative_pos = mat(self.invert_start).T @ (p - obs["cube_pos"])
            self.invert_relative_quat = mul(conjugate(self.invert_start), q)
            self.phase_to("invert_cube", t)
        if self.inverting:
            dt = t - self.pt
            self.cube_target = interpolate(self.invert_start, self.invert_goal, dt / self.invert_duration)
            h = np.r_[self.tripod.control(obs, self.center, self.cube_target), self.open["r"]]
            w = {
                "l": (
                    self.center + mat(self.cube_target) @ self.invert_relative_pos,
                    mul(self.cube_target, self.invert_relative_quat),
                ),
                "r": (np.array([0.3, -0.3, 0.65]), self.info["wrist_init"]["r"][1]),
            }
            if int(t) != self.invert_log:
                self.invert_log = int(t)
                print("INVERT", round(t, 2), np.round(obs["cube_pos"], 4), np.round(obs["cube_quat"], 3), flush=True)
            if dt >= self.invert_duration + 1:
                self.inverted = True
                self.inverting = False
                self.arc_ready = False
                self.phase_to("route", t)
            return {"hand": self.guard(h, obs), "wrist": w}
        return super().act(t, obs)
