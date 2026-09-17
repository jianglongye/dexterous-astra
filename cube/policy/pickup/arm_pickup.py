"""Select pickup yaw using the complete arm and the observed cube pose."""

import mujoco
import numpy as np
from arms import JOINTS


class PickupPlanner:
    def __init__(self, model_path, finger_model):
        self.m = mujoco.MjModel.from_binary_path(model_path)
        self.d = mujoco.MjData(self.m)
        self.joints = {side: np.array([self.m.joint(f"{side}_arm_{name}").id for name in JOINTS]) for side in "lr"}
        self.common = []
        self.finger_qadr = np.array(
            [
                self.m.joint(finger_model.joint(int(finger_model.actuator_trnid[i, 0])).name).qposadr[0]
                for i in range(20)
            ]
        )
        for j in range(finger_model.njnt):
            name = finger_model.joint(j).name
            if name in {"l_wrist", "r_wrist"}:
                continue
            target = self.m.joint(name).id
            width = {0: 7, 1: 4, 2: 1, 3: 1}[int(finger_model.jnt_type[j])]
            self.common.append((int(finger_model.jnt_qposadr[j]), int(self.m.jnt_qposadr[target]), width))

    def synchronize(self, controller, obs):
        """Copy observations into private arm data for geometric queries."""
        c, m, d = controller, self.m, self.d
        c.sync(obs)
        c.contact_points(obs)
        for source, target, width in self.common:
            d.qpos[target : target + width] = c.d.qpos[source : source + width]
        for side in "lr":
            d.qpos[m.jnt_qposadr[self.joints[side]]] = obs["arm_qpos"][side]
        mujoco.mj_forward(m, d)
        return d.qpos.copy()
