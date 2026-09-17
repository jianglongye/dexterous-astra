"""Quaternion helpers shared by the pickup controllers."""

import numpy as np
from baseline import quat, mul, blend


def conjugate(q):
    return np.asarray(q) * [1, -1, -1, -1]


def interpolate(a, b, u):
    delta = mul(b, conjugate(a))
    if delta[0] < 0:
        delta = -delta
    angle = 2 * np.arccos(np.clip(delta[0], -1, 1))
    axis = delta[1:] / max(np.linalg.norm(delta[1:]), 1e-9)
    return mul(quat(axis, angle * float(blend(0, 1, u))), a)
