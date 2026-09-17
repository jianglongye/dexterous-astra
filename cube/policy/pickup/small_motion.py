"""Keep released-hand commands local and consistent across phase boundaries."""

import json
import os
from pathlib import Path

import numpy as np

from motion import Controller as MotionController

DATA = Path(__file__).resolve().parent / "data"


class Controller(MotionController):
    def __init__(self, info):
        self.small_profile = os.environ.get("RUBIKS_V2_SMALL_PROFILE", "phase")
        self.small = json.loads((DATA / "small_profiles.json").read_text())[self.small_profile]
        super().__init__(info)
        print("SMALL_CONFIG", self.small_profile, json.dumps(self.small, sort_keys=True), flush=True)

    def _act(self, t, obs):
        f = self.fingers
        command = super()._act(t, obs)
        touching = any(
            body.startswith("r_") and other.startswith("cube/") and distance <= 0
            for body, other, distance in obs["contacts"]
        )
        # The parent reports the previous arc phase even when a new pedestal
        # placement has begun. Preserve its idle-hand command in that case.
        if touching or not f.pickup_done or f.inverting or f.parking:
            return command
        phase = command.get("phase", f.phase)
        if phase not in {"arc_route", "arc_outer", "arc_approach", "arc_retreat"}:
            return command

        p, q = f.stroke_wrist
        outward = f.arc_rotation[:, 2]
        clearance = self.small["clearance_m"]
        if phase == "arc_route":
            # act() may already have advanced f.phase to arc_outer, while its
            # returned command still belongs to arc_route. Keying on f.phase
            # lets the old 24 cm floating-hand waypoint escape for one tick.
            tangent = np.cross([0.0, 0.0, 1.0], outward)
            if np.linalg.norm(tangent) < 0.2:
                tangent = np.array([1.0, 0.0, 0.0])
            tangent /= np.linalg.norm(tangent)
            command["wrist"]["r"] = (p + clearance * outward - self.small["lateral_m"] * tangent, q)
        return command
