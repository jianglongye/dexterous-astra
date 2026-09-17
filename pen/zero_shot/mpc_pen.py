"""Closed-loop non-RL planning: separate planning and continuously simulated actor.

The actor pauses simulation while awaiting a plan, then executes six real control
steps. Only the planning copies reset. Recorded actor frames have no state writes,
no learned checkpoint, no object forces and no artificial animation.
"""

import argparse, json, math, os, sys, time
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--role", choices=["actor", "planner"], required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--seed", type=int, default=3)
p.add_argument("--population", type=int, default=128)
p.add_argument("--generations", type=int, default=3)
p.add_argument("--seconds", type=float, default=12)
p.add_argument("--sign", type=int, default=1)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
app = AppLauncher(a).app
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sim"))
from pen_env import PenSpinCfg, PenSpinEnv
from spec import CONTROL_DT, SIM_DT, DECIMATION

torch.set_num_threads(3)
torch.manual_seed(a.seed)
cfg = PenSpinCfg()
cfg.scene.num_envs = 1 if a.role == "actor" else a.population
cfg.sim.device = a.device
cfg.mass_range = (1.0, 1.0)
cfg.friction_range = (1.0, 1.0)
env = PenSpinEnv(cfg)
dev = env.device
n = env.num_envs
a.out.mkdir(parents=True, exist_ok=True)
lo, hi = env.q_lo, env.q_hi
K = 12
EXEC = 6


def step(target):
    env.robot.set_joint_position_target(env._to_isaac(target))
    for _ in range(DECIMATION):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=SIM_DT)


def state():
    pen = env.pen.data.root_state_w[0].clone()
    pen[:3] -= env.scene.env_origins[0]
    return dict(q=env._q()[0].cpu().numpy(), qd=env._qd()[0].cpu().numpy(), pen=pen.cpu().numpy())


def restore(s):
    q = torch.tensor(s["q"], device=dev).expand(n, -1)
    qd = torch.tensor(s["qd"], device=dev).expand(n, -1)
    env.robot.write_joint_state_to_sim(env._to_isaac(q), env._to_isaac(qd))
    pen = torch.tensor(s["pen"], device=dev).expand(n, -1).clone()
    pen[:, :3] += env.scene.env_origins
    env.pen.write_root_state_to_sim(pen)
    env.scene.update(dt=0.0)


def heading():
    pos, quat, axis = env._pen_state()
    return torch.atan2(axis[:, 1], axis[:, 0]), pos, axis


def atomic_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data))
    temp.replace(path)


started = time.time()
if a.role == "planner":
    i = 0
    warm = None
    while not (a.out / "finished.json").exists():
        request = a.out / f"request-{i:04d}.npz"
        if not request.exists():
            time.sleep(0.1)
            continue
        s = dict(np.load(request))
        start = torch.tensor(s["targets"], device=dev)
        ref = torch.tensor(s["ref"], device=dev)
        net = float(s["net"])
        remaining = max(0, 4 * math.pi - net)
        mean = start.expand(2, -1).clone() if warm is None else warm.clone()
        std = torch.full_like(mean, 0.3)
        std[:, [1, 3, 6, 10, 14, 17, 19]] *= 0.5
        best = None
        for gen in range(a.generations):
            noise = torch.randn(n, 2, 22, device=dev)
            noise = 0.65 * noise + 0.35 * noise[:, :1]
            points = (mean + noise * std).clamp(lo, hi)
            points[0] = start
            if best is not None:
                points[1] = best[1]
            restore(s)
            prev, _, _ = heading()
            total = torch.zeros(n, device=dev)
            bad = torch.zeros(n, dtype=torch.bool, device=dev)
            pen_max = torch.zeros(n, device=dev)
            for knot in range(2):
                begin = start.expand(n, -1) if knot == 0 else points[:, knot - 1]
                for k in range(K):
                    step(begin + (points[:, knot] - begin) * (k + 1) / K)
                    h, pos, axis = heading()
                    total += a.sign * (torch.remainder(h - prev + math.pi, 2 * math.pi) - math.pi)
                    prev = h
                    bad |= (
                        (pos[:, 2] < ref[2] - 0.035)
                        | ((pos[:, :2] - ref[:2]).norm(dim=-1) > 0.07)
                        | (axis[:, 2].abs() > 0.7)
                    )
                    if k % 6 == 5:
                        pen_max = torch.maximum(pen_max, env.physx_penetration())
            per, _ = env._finger_forces()
            support = (per > 0.01).sum(-1)
            speed = env.pen.data.root_lin_vel_w.norm(dim=-1)
            spin = env.pen.data.root_ang_vel_w.norm(dim=-1)
            violation = torch.maximum((lo - env._q()).amax(-1), (env._q() - hi).amax(-1)).clamp(min=0)
            bad |= (pen_max > 0.002) | (violation > 0.05)
            score = (
                total.clamp(max=remaining) * 20
                + support * 0.15
                - speed * 5
                - spin * (0.4 if remaining < 0.05 else 0.04)
            )
            score -= (
                abs(axis[:, 2]) * 2 + pen_max * 500 + ((pos[:, :2] - ref[:2]).norm(dim=-1) - 0.035).clamp(min=0) * 30
            )
            score = torch.where(bad, score - 1000, score)
            order = score.argsort(descending=True)
            winner = int(order[0])
            if best is None or float(score[winner]) > best[0]:
                best = (float(score[winner]), points[winner].clone())
            elite = points[order[: max(8, n // 8)]]
            mean = 0.25 * mean + 0.75 * elite.mean(0)
            std = (0.25 * std + 0.75 * elite.std(0)).clamp(min=0.04)
        warm = best[1]
        atomic_json(a.out / f"response-{i:04d}.json", dict(points=warm.cpu().tolist(), score=best[0]))
        if i % 10 == 0:
            print(
                json.dumps(
                    dict(request=i, actor_turns=net / (2 * math.pi), score=best[0], wall_s=time.time() - started)
                ),
                flush=True,
            )
        i += 1
else:
    valid, *_ = env.settle(
        torch.arange(n, device=dev),
        env.grasp_q.expand(n, -1),
        env.grasp_pen_pos.expand(n, -1),
        env.grasp_pen_quat.expand(n, -1),
    )
    assert bool(valid[0]), "Actor initial grasp is invalid"
    targets = env.grasp_q.clone()
    ref = state()["pen"][:3].copy()
    prev, _, _ = heading()
    net = 0.0
    hold = 0
    dropped = False
    keys = ("q", "pen_pos", "pen_quat", "pen_axis", "pen_linvel", "pen_angvel", "finger_force", "physx_penetration")
    log = {k: [] for k in keys}
    peak = 0.0
    violation = 0.0
    max_tilt = 0.0

    def record():
        global prev, net, hold, dropped, peak, violation, max_tilt
        h, pos, axis = heading()
        net += a.sign * float(torch.remainder(h[0] - prev[0] + math.pi, 2 * math.pi) - math.pi)
        prev = h.clone()
        per, _ = env._finger_forces()
        pen = env.physx_penetration()
        peak = max(peak, float(pen[0]))
        violation = max(
            violation, float(torch.maximum((lo - env._q()[0]).max(), (env._q()[0] - hi).max()).clamp(min=0))
        )
        tilt = math.degrees(math.asin(min(1.0, abs(float(axis[0, 2])))))
        max_tilt = max(max_tilt, tilt)
        dropped |= bool(pos[0, 2] < ref[2] - 0.06 or (pos[0, :2] - torch.tensor(ref[:2], device=dev)).norm() > 0.15)
        still = bool(
            env.pen.data.root_lin_vel_w[0].norm() < 0.05
            and env.pen.data.root_ang_vel_w[0].norm() < 1
            and (per[0] > 0.01).any()
        )
        hold = hold + 1 if net >= 4 * math.pi and still else 0
        vals = (
            env._q(),
            pos,
            env.pen.data.root_quat_w,
            axis,
            env.pen.data.root_lin_vel_w,
            env.pen.data.root_ang_vel_w,
            per,
            pen,
        )
        for k, v in zip(keys, vals):
            log[k].append(v[0].detach().cpu().numpy().copy())

    record()
    for i in range(math.ceil(a.seconds * 60 / EXEC)):
        request = a.out / f"request-{i:04d}.npz"
        temp = request.with_suffix(".tmp.npz")
        np.savez(temp, **state(), targets=targets.cpu().numpy(), ref=ref, net=net)
        temp.replace(request)
        response = a.out / f"response-{i:04d}.json"
        wait = time.time()
        while not response.exists():
            if time.time() - wait > 180:
                raise TimeoutError("Planner response timed out")
            time.sleep(0.1)
        point = torch.tensor(json.loads(response.read_text())["points"][0], device=dev)
        begin = targets.clone()
        for k in range(EXEC):
            targets = begin + (point - begin) * (k + 1) / K
            step(targets[None])
            record()
            if dropped:
                break
        if i % 10 == 0:
            print(
                json.dumps(dict(step=i, turns=net / (2 * math.pi), penetration=peak, sim_s=(len(log["q"]) - 1) / 60)),
                flush=True,
            )
        if dropped or hold >= 60:
            break
    np.savez_compressed(a.out / "trajectory.npz", **{k: np.stack(v) for k, v in log.items()}, fps=60)
    report = dict(
        simulator="Isaac Lab / PhysX",
        robot="Sharpa Wave",
        controller="non-RL closed-loop sampled finger-keyframe planner",
        learned_checkpoint=None,
        state_resets_during_replay=0,
        object_actuation=False,
        seed=a.seed,
        sign=a.sign,
        net_turns=net / (2 * math.pi),
        dropped=dropped,
        penetration_m=peak,
        joint_violation_rad=violation,
        max_tilt_deg=max_tilt,
        target_turns=2,
        hold_s=hold / 60,
        passed=net >= 4 * math.pi
        and hold >= 60
        and not dropped
        and peak <= 0.002
        and violation <= 0.05
        and max_tilt < 45,
        frames=len(log["q"]),
        duration_s=(len(log["q"]) - 1) / 60,
        wall_s=time.time() - started,
        physics="Nominal original Sharpa USD and PhysX; no learned policy; planning time omitted from simulation replay.",
    )
    atomic_json(a.out / "report.json", report)
    atomic_json(a.out / "finished.json", report)
    print(json.dumps(report), flush=True)
app.close()
