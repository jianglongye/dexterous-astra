"""Side pickup and three-finger support, followed by fixed-wrist layer strokes."""

import os
import numpy as np
from arc_controller import Controller as Arc
from baseline import blend, mul, mat
from tripod import Tripod


class Controller(Arc):
    def __init__(self, info):
        super().__init__(info)
        self.tripod = Tripod(self)
        self.pickup_done = False
        self.current_yaw = float(os.environ.get("RUBIKS_TURN_YAW", "0"))
        self.last_pickup_log = -1

    def support_action(self, obs, h, w):
        h[:20] = self.tripod.control(obs, self.center, self.cube_target)
        follow = float(os.environ.get("RUBIKS_SUPPORT_FOLLOW", "0"))
        pose_center = self.center + follow * (obs["cube_pos"] - self.center)
        w["l"] = self.tripod.pose(pose_center, self.cube_target)

    def act(self, t, obs):
        if self.pickup_done:
            return super().act(t, obs)
        dt = t - self.pt
        phase = self.phase
        center = np.array(getattr(self, "pickup_center", [0.0, 0.0, 0.2]))
        h = np.r_[self.tripod.pickup_open, self.open["r"]]
        w = {s: (np.array([-0.3, 0.35 if s == "l" else -0.3, 0.5]), self.info["wrist_init"][s][1]) for s in "lr"}
        p, q = self.tripod.pose(center, self.cube_target)
        approach = mat(mul(self.cube_target, self.tripod.grip_in_cube))[:, 1]
        if phase == "init":
            h = blend(np.zeros(40), h, t / 2)
            for s, (ip, iq) in self.info["wrist_init"].items():
                w[s] = (ip + blend([0, 0, 0], [-0.15, 0.18 if s == "l" else -0.18, 0.25], t / 2), iq)
            if dt >= 2:
                self.phase_to("tripod_high", t)
        elif phase == "tripod_high":
            w["l"] = (p + 0.10 * approach + [0, 0, 0.20], q)
            if dt >= 2:
                self.phase_to("tripod_outer", t)
        elif phase == "tripod_outer":
            w["l"] = (p + 0.10 * approach, q)
            if dt >= 2:
                self.phase_to("tripod_approach", t)
        elif phase == "tripod_approach":
            w["l"] = (p + float(blend(0.10, 0, dt / 2)) * approach, q)
            if dt >= 2.2:
                self.phase_to("tripod_close", t)
        elif phase == "tripod_close":
            w["l"] = (p, q)
            h[:20] = blend(self.tripod.open, self.tripod.closed, dt / 2)
            if dt >= 2.2:
                self.phase_to("tripod_settle", t)
        elif phase == "tripod_settle":
            w["l"] = (p, q)
            h[:20] = blend(self.tripod.closed, self.tripod.control(obs, center, self.cube_target), dt / 1.2)
            if dt >= 2:
                self.phase_to("tripod_lift", t)
        elif phase == "tripod_lift":
            target = blend(center, self.center, dt / 4)
            w["l"] = self.tripod.pose(target, self.cube_target)
            h[:20] = self.tripod.control(obs, target, self.cube_target)
            if dt >= 4.5:
                if os.environ.get("RUBIKS_MODE") == "hold":
                    self.phase_to("tripod_hold", t)
                elif os.environ.get("RUBIKS_MODE") == "rotate":
                    self.rotation_start = self.cube_target.copy()
                    self.phase_to("tripod_rotate", t)
                else:
                    self.pickup_done = True
                    self.phase_to("route", t)
        if int(t) != self.last_pickup_log:
            self.last_pickup_log = int(t)
            print(
                "TRIPOD",
                round(t, 2),
                phase,
                np.round(obs["cube_pos"], 4),
                np.round(obs["cube_quat"], 3),
                sorted(set(x[0] for x in obs["contacts"])),
                flush=True,
            )
        if self.tripod.n > 3 and not self.pickup_done:
            parked = getattr(self.tripod, "pickup_ring", np.array([1.4, 0, 0, 0]))
            if phase == "init":
                h[12:16] = blend(np.zeros(4), parked, t / 2)
            elif phase == "tripod_lift" and dt < 4:
                h[12:16] = blend(parked, self.tripod.closed[12:16], (dt - 2) / 1.5)
            elif phase != "tripod_lift" and not phase.startswith("tripod_hold"):
                h[12:16] = parked
        return {"hand": self.guard(h, obs), "wrist": w}
