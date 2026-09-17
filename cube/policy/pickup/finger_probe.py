"""Fixed-wrist diagnostic, retaining the verified pickup/support controller."""

import os
import numpy as np

from baseline import Controller as Baseline
from baseline import quat, mul


class Controller(Baseline):
    def __init__(self, info):
        super().__init__(info)
        roll = float(os.environ.get("RUBIKS_SUPPORT_ROLL", "0"))
        self.lframe = mul(quat([1, 0, 0], roll), self.lframe)
        self.current_yaw = float(os.environ.get("RUBIKS_TURN_YAW", str(self.current_yaw)))
        self.side_pickup = abs(roll) > 0.1
        self.support_z = float(os.environ.get("RUBIKS_SUPPORT_Z", "0"))

    def pose(self, s, c, q=np.array([1.0, 0, 0, 0]), dz=0):
        c = np.asarray(c) + (np.array([0, 0, getattr(self, "support_z", 0)]) if s == "l" else 0)
        return super().pose(s, c, q, dz)
