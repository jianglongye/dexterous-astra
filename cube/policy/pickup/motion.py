"""V2 motion preferences layered over the unchanged v1 contact controller."""

import json
import os
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
from arm_controller import Controller as ArmController


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u**3 * (10.0 + u * (-15.0 + 6.0 * u))


def load_config():
    profiles = json.loads((DATA / "profiles.json").read_text())
    name = os.environ.get("RUBIKS_V2_PROFILE", "coordinated")
    config = dict(profiles[name])
    return name, config


class Controller(ArmController):
    FREE_PHASES = {"arc_route", "arc_outer", "arc_retreat"}

    def __init__(self, info):
        self.profile, self.config = load_config()
        super().__init__(info)
        self.rest_wrist = (self.rest_wrist[0] + np.asarray(self.config["rest_offset_m"]), self.rest_wrist[1])
        self.unused = np.array([j for j in range(20, 40) if j // 4 not in set(self.fingers.turn_ids // 4)])
        self.relaxed = np.tile([0.55, 0.0, 0.85, 0.5], 5)
        self.relax_started = None
        print("V2_CONFIG", self.profile, json.dumps(self.config, sort_keys=True), flush=True)

    def _act(self, t, obs):
        f = self.fingers
        touching = any(
            body.startswith("r_") and other.startswith("cube/") and distance <= 0
            for body, other, distance in obs["contacts"]
        )
        # Adjust only the controller's free-motion clock. Physical time, contact
        # dwell, tracking gates, motor bounds, and checker timing remain intact.
        command = super()._act(t, obs)
        phase = command.get("phase", f.phase)
        if f.phase == "arc_route" and not touching:
            outward = f.arc_rotation[:, 2]
            tangent = np.cross([0.0, 0.0, 1.0], outward)
            if np.linalg.norm(tangent) < 0.2:
                tangent = np.array([1.0, 0.0, 0.0])
            tangent /= np.linalg.norm(tangent)
            p, q = f.stroke_wrist
            command["wrist"]["r"] = (p + 0.085 * outward - self.config["route_lateral_m"] * tangent, q)

        # Relax only unused fingers during released transit. Return to the
        # validated clearance posture before the approach/contact phases.
        if phase == "arc_route" and not touching:
            if self.relax_started is None:
                self.relax_started = t
            weight = smoothstep((t - self.relax_started) / 0.6)
        elif phase == "arc_outer" and not touching:
            weight = 1.0 - smoothstep((self.clock - f.pt) / 1.0)
        else:
            self.relax_started = None
            weight = 0.0
        weight *= self.config["unused_relaxation"]
        hand = command["hand"]
        hand[self.unused] = (1.0 - weight) * hand[self.unused] + weight * self.relaxed[self.unused - 20]

        # Prepare the free hand above the next turning face once the supporting
        # arm has finished most of its cube reorientation. The arm planner still
        # checks reachability and collisions before motor commands execute.
        return command
