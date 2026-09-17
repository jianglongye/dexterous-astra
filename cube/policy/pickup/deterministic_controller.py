"""Finger-driven solver with fixed, face-specific final support calibration.

This configures the shared geometric planner and feedback controller. It does
not learn a policy, alter cube state, or modify simulation parameters.
"""

import os
from cube_controller import Controller as CubeController


class Controller(CubeController):
    def __init__(self, info):
        final_face = info["solution"].split()[-1][0]
        # U, D, and L use the load-sharing settings validated for their final grasp.
        # The other final grasps add vertical support only when the cube sinks.
        defaults = {
            "RUBIKS_FINAL_BILATERAL": "1",
            "RUBIKS_THUMB_PRESSURE_GUARD": "1",
            "RUBIKS_THUMB_RELIEF": "4000",
            "RUBIKS_BOTTOM_DAMPING": "0",
            "RUBIKS_FINAL_RELIEF_SCALE": "1",
            "RUBIKS_FINAL_CONTACT_POINTS": "0",
            "RUBIKS_FINAL_INDEX_RELIEF": "0",
            "RUBIKS_BOTTOM_LEGACY_ORDER": "0",
        }
        if final_face in {"U", "D", "L"}:
            defaults.update(RUBIKS_ALIGN_LOAD={"U": "0", "D": ".3", "L": ".7"}[final_face], RUBIKS_ALIGN_FEEDBACK="0")
            if final_face == "U":
                defaults["RUBIKS_BOTTOM_DAMPING"] = ".003"
        for key, value in defaults.items():
            os.environ.setdefault(key, value)
        print("FINAL_SUPPORT", final_face, {key: os.environ[key] for key in defaults}, flush=True)
        super().__init__(info)
