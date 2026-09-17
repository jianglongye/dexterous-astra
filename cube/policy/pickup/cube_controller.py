"""Quarter-turn manipulation with physical pedestal regrasp between moves."""

from face_controller import Controller as FaceController


class Controller(FaceController):
    def __init__(self, info):
        super().__init__(info)
        self.moves = [
            quarter for move in self.moves for quarter in ([move[0] + "'"] * 2 if move.endswith("2") else [move])
        ]
        self.last_started_move = 0
        self.parking = False
        self.transferring = False
        self.park_log = -1
