"""Small box-constrained least-squares steps for redundant arm kinematics."""

import numpy as np


def box_quadratic(hessian, linear, lower, upper):
    """Minimize .5*x.H.x - b.x with an active set, keeping joint bounds exact."""
    x = np.clip(np.zeros_like(linear), lower, upper)
    active = np.zeros(len(x), dtype=int)
    for _ in range(40):
        free = np.flatnonzero(active == 0)
        fixed = np.flatnonzero(active != 0)
        proposal = x.copy()
        if len(free):
            rhs = linear[free] - hessian[np.ix_(free, fixed)] @ x[fixed]
            proposal[free] = np.linalg.solve(hessian[np.ix_(free, free)], rhs)
        direction = proposal - x
        steps = np.ones(len(x))
        above = proposal > upper + 1e-12
        below = proposal < lower - 1e-12
        steps[above] = (upper[above] - x[above]) / direction[above]
        steps[below] = (lower[below] - x[below]) / direction[below]
        blocking = int(np.argmin(steps))
        if steps[blocking] < 1:
            x += max(0.0, steps[blocking]) * direction
            active[blocking] = 1 if above[blocking] else -1
            x[blocking] = upper[blocking] if above[blocking] else lower[blocking]
            continue
        x = proposal
        gradient = hessian @ x - linear
        violation = gradient * active
        if np.max(violation) <= 1e-10:
            break
        active[int(np.argmax(violation))] = 0
    return np.clip(x, lower, upper)


def step(jacobian, error, q, limits, trust=0.08):
    lower, upper = limits[:, 0] + 0.02, limits[:, 1] - 0.02
    null = np.eye(len(q)) - np.linalg.pinv(jacobian) @ jacobian
    preference = np.zeros_like(q)
    if np.linalg.norm(error) > 1e-5:
        clearance_low = np.maximum(q - lower, 0.03)
        clearance_high = np.maximum(upper - q, 0.03)
        away = 0.002 * (1 / clearance_low**2 - 1 / clearance_high**2)
        preference = null @ (-0.005 * q + np.clip(away, -0.12, 0.12))
    damping = 0.0001
    hessian = jacobian.T @ jacobian + damping * np.eye(len(q))
    linear = jacobian.T @ error + damping * preference
    return box_quadratic(hessian, linear, np.maximum(lower - q, -trust), np.minimum(upper - q, trust))
