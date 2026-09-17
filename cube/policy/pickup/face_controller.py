"""Pick the required support grasp directly, then reorient the whole cube before finger turns."""

import os
from inverted_controller import Controller as OrientedController


class Controller(OrientedController):
    def __init__(self, info):
        os.environ.setdefault("RUBIKS_FACE_DOWN", "0")
        super().__init__(info)
