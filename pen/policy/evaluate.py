"""Evaluate the policy checkpoint on seeded randomized trials in Isaac Lab and record every control step.

Each trial seed fixes the grasp noise, pen yaw/offset, pen mass, and pen friction. A trial whose
randomized grasp does not settle validly is redrawn from the same seed stream (attempts are reported).
The policy acts deterministically (actor mean). score.py judges the recorded trajectories.
"""

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--run_cfg", required=True, help="env_cfg.json of the training run (state.json next to it)")
ap.add_argument("--out", required=True)
ap.add_argument("--seed0", type=int, required=True)
ap.add_argument("--trials", type=int, default=32)
ap.add_argument("--seconds", type=float, default=12.0)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_mul  # noqa: E402
from rsl_rl.modules import ActorCritic  # noqa: E402
from tensordict import TensorDict  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from pen_env import PenSpinCfg, PenSpinEnv  # noqa: E402
from spec import CONTROL_DT, FINGERS, FRICTION, JOINTS, PEN  # noqa: E402

# Task settings the policy observed in each training curriculum stage: (target turns, episode length for the time input).
STAGE_OBS = {0: (1e9, 10.0), 1: (1e9, 10.0), 2: (1e9, 10.0), 3: (3.0, 12.0)}

cfg = PenSpinCfg()
cfg.scene.num_envs = args.trials
cfg.sim.device = args.device
cfg.action_scale = json.loads(Path(args.run_cfg).read_text())["action_scale"]
env = PenSpinEnv(cfg)
dev = env.device
n = env.num_envs
gens = [torch.Generator(device="cpu").manual_seed(args.seed0 + i) for i in range(n)]


def draw(i):
    g = gens[i]
    u = lambda *s: torch.rand(*s, generator=g)
    q = (env.grasp_q.cpu() + cfg.joint_noise * (2 * u(22) - 1)).clamp(env.q_lo.cpu(), env.q_hi.cpu())
    pos = env.grasp_pen_pos.cpu().clone()
    pos[:2] += cfg.pen_xy_noise * (2 * u(2) - 1)
    pos[2] += 0.002
    yaw = math.radians(cfg.pen_yaw_noise_deg) * (2 * u(1) - 1)
    quat = torch.tensor([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    return q, pos, quat


# per-trial pen physics
view = env.pen.root_physx_view
masses = view.get_masses()
mats = view.get_material_properties()
for i in range(n):
    u = torch.rand(2, generator=gens[i])
    masses[i] = PEN["mass"] * (cfg.mass_range[0] + (cfg.mass_range[1] - cfg.mass_range[0]) * u[0])
    mats[i, :, :2] = FRICTION * (cfg.friction_range[0] + (cfg.friction_range[1] - cfg.friction_range[0]) * u[1])
all_cpu = torch.arange(n)
view.set_masses(masses, all_cpu)
view.set_material_properties(mats, all_cpu)

ids = torch.arange(n, device=dev)
attempts = np.zeros(n, dtype=int)
pending = np.ones(n, dtype=bool)
bank = {
    "q": torch.zeros(n, 22, device=dev),
    "targets": torch.zeros(n, 22, device=dev),
    "pos": torch.zeros(n, 3, device=dev),
    "quat": torch.zeros(n, 4, device=dev),
}
while pending.any() and attempts.max() < 20:
    qs, ps, rs = zip(*[draw(i) for i in range(n)])
    q = torch.stack(qs).to(dev)
    pos = torch.stack(ps).to(dev)
    quat = quat_mul(torch.stack(rs).to(dev), env.grasp_pen_quat.expand(n, 4))
    valid, q_s, p_s, qt_s, _ = env.settle(ids, q, pos, quat)
    v = valid.cpu().numpy()
    take = pending & v
    t = torch.as_tensor(take, device=dev)
    bank["q"][t], bank["targets"][t], bank["pos"][t], bank["quat"][t] = q_s[t], q[t], p_s[t], qt_s[t]
    attempts[pending] += 1
    pending &= ~v
if pending.any():
    print("[eval] trials without a valid settled grasp:", np.flatnonzero(pending).tolist(), flush=True)

# the curriculum stage and joint-target box this checkpoint was trained under (promotion applies afterwards)
ck_iter = torch.load(args.ckpt, map_location="cpu", weights_only=False)["iter"]
history = [
    rec
    for rec in json.loads(Path(args.run_cfg).with_name("state.json").read_text())["history"]
    if rec["iter"] <= ck_iter
]
stage, cfg.target_box = history[-1]["stage"], history[-1].get("target_box", 1.0)
cfg.target_turns, cfg.obs_episode_s = STAGE_OBS[stage]
cfg.episode_length_s = args.seconds + 1.0

# reset every env to its own settled grasp
env.bank = bank
env._reset_idx(ids)
env.sim.step(render=False)
env.scene.update(dt=0.0)
obs = env._get_observations()["policy"]

ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
td = TensorDict({"policy": obs}, batch_size=[n])
policy = ActorCritic(
    td,
    {"policy": ["policy"], "critic": ["policy"]},
    22,
    actor_obs_normalization=True,
    critic_obs_normalization=True,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
).to(dev)
policy.load_state_dict(ck["model_state_dict"])
policy.eval()

steps = int(args.seconds / CONTROL_DT)
_, _, axis0 = env._pen_state()
heading0 = torch.atan2(axis0[:, 1], axis0[:, 0]).cpu().numpy()
keys = (
    "pen_pos",
    "pen_quat",
    "pen_axis",
    "pen_linvel",
    "pen_angvel",
    "q",
    "qd",
    "targets",
    "finger_force",
    "palm_force",
    "action",
    "physx_penetration",
)
log = {k: [] for k in keys}
with torch.inference_mode():
    for s in range(steps):
        a = policy.act_inference(TensorDict({"policy": obs}, batch_size=[n]))
        o, _, term, trunc, _ = env.step(a)
        obs = o["policy"]
        pos, quat, axis = env._pen_state()
        per, palm = env._finger_forces()
        for k, v in zip(
            keys,
            (
                pos,
                quat,
                axis,
                env.pen.data.root_lin_vel_w,
                env.pen.data.root_ang_vel_w,
                env._q(),
                env._qd(),
                env.targets,
                per,
                palm,
                env.action,
                env.physx_penetration(),
            ),
        ):
            log[k].append(v.detach().cpu().numpy().astype(np.float32))
        if (term | trunc).any():
            print("[eval] unexpected environment reset at step", s, flush=True)
            break
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
arrays = {k: np.stack(v, 1) for k, v in log.items()}  # (trials, T, ...)
np.savez_compressed(
    out / "trajectories.npz",
    **arrays,
    ref_pos=bank["pos"].cpu().numpy(),
    heading0=heading0,
    q0=bank["q"].cpu().numpy(),
    pen_pos0=bank["pos"].cpu().numpy(),
    pen_quat0=bank["quat"].cpu().numpy(),
    q_lo=env.q_lo.cpu().numpy(),
    q_hi=env.q_hi.cpu().numpy(),
    masses=masses[:, 0].numpy(),
    friction=mats[:, 0, 0].numpy(),
    attempts=attempts,
    unsettled=pending,
)
meta = {
    "ckpt": args.ckpt,
    "target_box": cfg.target_box,
    "stage_obs": stage,
    "seed0": args.seed0,
    "trials": n,
    "joints": list(JOINTS),
    "fingers": list(FINGERS),
    "control_dt": CONTROL_DT,
    "steps": len(log["q"]),
    "settle_attempts_mean": float(attempts.mean()),
    "unsettled": int(pending.sum()),
}
(out / "meta.json").write_text(json.dumps(meta, indent=1))
print("[eval] wrote", out, flush=True)
app.close()
