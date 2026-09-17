"""Shared constants for the Sharpa pen-spinning scene (no simulator imports)."""

import math
import os
from pathlib import Path

# Official Sharpa Wave assets, commit 0d447b6889e6d993758169dfc0aa75ee9f6ad8d7
SHARPA_ROOT = Path(os.environ.get("SHARPA_ROOT", Path.home() / "code/library/sharpa-urdf-usd-xml"))
HAND_DIR = SHARPA_ROOT / "wave_01/right_sharpa_wave"
HAND_USD = HAND_DIR / "right_sharpa_wave.usda"
HAND_MJCF = HAND_DIR / "right_sharpa_wave.xml"

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# Canonical joint order: MJCF order, thumb to pinky, proximal to distal.
JOINTS = (
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)

# Hand base frame: fingers along +z, palm normal +x, thumb side +y.
# Palm up in the world: base x -> world z, base z -> world x, base y -> world -y.
# That is a 180 degree turn about (1, 0, 1)/sqrt(2); quaternion (w, x, y, z).
PALM_UP_QUAT = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
HAND_POS = (0.0, 0.0, 0.5)

GRAVITY = 9.81
# The pen is a capsule lying along its local long axis. length is the total tip-to-tip length.
PEN = {"length": 0.140, "radius": 0.0045, "mass": 0.010}
FRICTION = 1.0  # static and dynamic, hand and pen, combine mode "average"

SIM_DT = 1 / 240
DECIMATION = 4  # 60 Hz control
CONTROL_DT = SIM_DT * DECIMATION
