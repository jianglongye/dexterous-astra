"""One-hand policy solve: scripted pickup, then only learned left-hand finger maneuvers.

The scripted pickup (pickup/) lifts the cube with the left hand and turns it so the first face to move is
on top; the right hand never touches the cube afterwards. Then the checkpoints act: face turns and rolls
that bring the next face up. Physics is never reset. The move sequence is the scramble's inverse.

    python policy/solve.py --out OUT --seed 5005 --scramble "U F L"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "policy/pickup", ROOT / "policy", ROOT / "sim"):
    sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import physics  # noqa: E402
import roll  # noqa: E402
import scene  # noqa: E402
import turn  # noqa: E402
from network import load_policy  # noqa: E402
from run import HOLD_S, LIMITS, load_controller  # noqa: E402

CHECKPOINTS = ROOT / "checkpoints"
TURN_POLICIES = ["turn1-P1.pt", "turn23-THc.pt"]  # turn 1, then turns 2 and 3
ROLL_POLICIES = ["roll1-RHc.pt", "roll2-RUc.pt"]  # both trained for top L -> target F; other pairs by cube symmetry
ROLL_TRAINED = ("L", "F")
SETTLE_S = 0.3  # hold between learned maneuvers, fingers keep their last command
PREFIX_S = 60.0  # scripted pickup time limit
PICKUP = ROOT / "policy/pickup"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def plan(scramble, first_up):
    """Roll to bring each move's face up, then turn it. Only the +90 degree (primed) turn was learned."""
    steps, up = [], first_up
    for move in scene.invert(scramble).split():
        face, suffix = move[0], move[1:]
        if suffix != "'":
            raise ValueError("The learned turn performs primed moves only")
        if face != up:
            steps.append(("roll", up, face))
            up = face
        steps.append(("turn", face, 1))
    return steps


def pickup_controller(sim, scramble, out):
    """The scripted two-arm pickup, configured through environment variables."""
    os.environ.update(
        RUBIKS_V2_PROFILE="ready",
        RUBIKS_V2_SMALL_PROFILE="phase",
        RUBIKS_SUPPORT_POSTURE=".03",
        RUBIKS_ARM_CONTACT_OFFSET="0",
        RUBIKS_CONTACT_LIFT=".005",
        RUBIKS_CONTACT_WRENCH="1",
        RUBIKS_ARC_GRASP=str(PICKUP / "data/natural-grasp.json"),
        RUBIKS_ARM_BENT_GRASP=str(PICKUP / "data/natural-support.json"),
    )
    os.environ.pop("RUBIKS_SUPPORT_MEASURED_RIGHT", None)
    # The support opening belongs to the four-finger grasp; it is installed once that grasp is reached.
    cfg = json.loads((PICKUP / "data/refinement.json").read_text())
    cfg["support_open_joints"] = None
    write_json(out / "refinement.json", cfg)
    os.environ["RUBIKS_V2_REFINEMENT"] = str((out / "refinement.json").resolve())
    return load_controller(PICKUP / "refine_motion.py", sim.controller_info(scramble, out, 720.0))


class Audit:
    """Solve checks on every recorded sample: continuity, intact cube, contact and joint limits, solved hold."""

    def __init__(self, sim):
        self.sim = sim
        self.home = scene.cubies(sim.m)[1]
        self.previous = None
        self.continuous = self.intact = True
        self.since = None
        self.worst = dict(penetration_m=sim.penetration(), joint_violation_rad=sim.joint_violation())
        self.last_time = None
        self.last_observation = None

    def update(self):
        s = self.sim
        if self.last_time == float(s.d.time):
            return self.last_observation
        obs = s.observe()
        self.last_time = float(s.d.time)
        self.last_observation = obs
        rots = obs["cubie_rot"]
        if self.previous is not None:
            jump = np.arccos(np.clip((np.einsum("nij,nij->n", self.previous, rots) - 1) / 2, -1, 1)).max()
            self.continuous &= bool(jump < 0.5)
        self.previous = rots.copy()
        if obs["misalign_deg"] < 20:
            slots = np.einsum("nij,nj->ni", scene.snap(rots)[0], self.home)
            self.intact &= len(np.unique(slots, axis=0)) == len(self.home) and scene.legal(obs["facelets"])
        for key, val in s.substep_worst.items():
            self.worst[key] = max(self.worst[key], val)
        solved = obs["facelets"] == scene.SOLVED and obs["misalign_deg"] < LIMITS["misalign_deg"] and s.held_in_hand()
        self.since = (float(s.d.time) if self.since is None else self.since) if solved else None
        return obs

    def report(self):
        s = self.sim
        obs = self.last_observation
        checks = dict(
            solved=self.since is not None and s.d.time - self.since >= HOLD_S - 1e-7,
            continuous=self.continuous,
            cube_intact=bool(self.intact),
            cube_actuators_zero=s.cube_actuators == 0,
            penetration_ok=self.worst["penetration_m"] <= LIMITS["penetration_m"],
            joint_limits_ok=self.worst["joint_violation_rad"] <= LIMITS["joint_violation_rad"],
            held_in_hand=s.held_in_hand(),
        )
        return dict(
            passed=all(checks.values()),
            checks=checks,
            **self.worst,
            sim_time=float(s.d.time),
            final_facelets=obs["facelets"],
            misalign_deg=obs["misalign_deg"],
        )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scramble", default="U F L")
    p.add_argument("--seed", type=int, default=5005, help="friction (+-2%%) and actuator strength (+-1%%) jitter")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    turn_policies = [load_policy(CHECKPOINTS / name) for name in TURN_POLICIES]
    roll_policies = [load_policy(CHECKPOINTS / name) for name in ROLL_POLICIES]
    sim, physics_info = physics.make_sim(a.scramble, a.out, a.seed)
    ctrl = pickup_controller(sim, a.scramble, a.out)
    audit = Audit(sim)
    trace = dict(
        qpos=[sim.d.qpos.copy()],
        time=[float(sim.d.time)],
        hand_qpos=[sim.d.qpos[sim.act_qadr].copy()],
        phase=[sim.phase],
    )

    def record_step():
        audit.update()
        if float(sim.d.time) - trace["time"][-1] < 0.02 - 1e-7:
            return
        trace["time"].append(float(sim.d.time))
        trace["qpos"].append(sim.d.qpos.copy())
        trace["hand_qpos"].append(sim.d.qpos[sim.act_qadr].copy())
        trace["phase"].append(sim.phase)

    error, maneuvers, steps, started = None, [], [], time.monotonic()
    try:
        # Scripted pickup until the first face is on top and the first turn would begin.
        refinement = json.loads((PICKUP / "data/refinement.json").read_text())
        while float(sim.d.time) < PREFIX_S:
            f = ctrl.fingers
            if f.phase in ("arc_relax", "arc_route") and f.mi == 0 and f.pickup_done:
                break
            obs = audit.update()
            if f.tripod.n == 4 and ctrl.refinement["support_open_joints"] is None:
                ctrl.refinement["support_open_joints"] = refinement["support_open_joints"]
            sim.apply(ctrl.act(float(sim.d.time), obs), 0.01)
            sim.step(10)
            record_step()
        else:
            raise RuntimeError("Scripted pickup did not reach the first face-up grip")
        if any(not n.startswith("l_") for n in physics.cube_contacts(sim)):
            raise RuntimeError("Cube touches something other than the left hand at hand-over")
        for side in "lr":
            sim.arms.filters[side].v[:] = 0
        # Hand-over: hold the measured finger posture (targets = joint positions), as in the training starts.
        sim.hand_target[:20] = sim.d.qpos[sim.act_qadr[:20]]
        sim.phase = "one_hand_settle"
        for _ in range(round(0.8 / 0.02)):
            sim.step(20)
            record_step()
        steps = plan(a.scramble, turn.up_face(sim))
        write_json(a.out / "plan.json", dict(steps=steps, handover_time=float(sim.d.time)))
        for index, step in enumerate(steps):
            kind = step[0]
            sim.phase = "one_hand_hold"
            if index:
                for _ in range(round(SETTLE_S / 0.02)):
                    sim.step(20)
                    record_step()
            else:
                sim.hand_target[:20] = sim.d.qpos[sim.act_qadr[:20]]
            if kind == "turn":
                turn_index = sum(1 for previous in steps[:index] if previous[0] == "turn")
                cp, policy = turn_policies[min(turn_index, len(turn_policies) - 1)]
                # Cubie-cubie contacts are redundant with the welds; they stay on only for the first turn.
                report, rows = turn.execute(
                    sim, cp, policy, step[2], after_step=record_step, internal_collisions=not turn_index
                )
            else:
                roll_index = sum(1 for previous in steps[:index] if previous[0] == "roll")
                cp, policy = roll_policies[min(roll_index, len(roll_policies) - 1)]
                relabel = roll.label_symmetry(*ROLL_TRAINED, step[1], step[2])
                report, rows = roll.execute(sim, cp, policy, *ROLL_TRAINED, relabel, after_step=record_step)
            report.update(index=index, step=list(step))
            maneuvers.append(report)
            write_json(a.out / f"maneuver-{index}-{kind}.json", dict(report=report, metrics=rows))
            if not report["success"]:
                raise RuntimeError(f"Learned {kind} {index} failed: {report.get('failure')}")
            if kind == "roll" and turn.up_face(sim) != step[2]:
                raise RuntimeError("Roll did not bring the planned face up")
        sim.phase = "one_hand_final_hold"
        end = float(sim.d.time) + HOLD_S + 0.5
        while float(sim.d.time) < end and not audit.report()["checks"]["solved"]:
            sim.step(20)
            record_step()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    audit.update()
    report = dict(
        audit.report(),
        error=error,
        maneuvers=maneuvers,
        wall_seconds=time.monotonic() - started,
        scramble=a.scramble,
        physics=physics_info,
    )
    right_touch = any("r_" in str(m.get("other_body_contacts")) for m in maneuvers)
    report["passed"] = bool(report["passed"] and not error and len(maneuvers) == len(steps) and not right_touch)
    write_json(a.out / "report.json", report)
    np.savez_compressed(a.out / "trajectory.npz", **{k: np.asarray(v) for k, v in trace.items()})
    print("RESULT", {k: v for k, v in report.items() if k not in ("maneuvers", "physics")}, flush=True)


if __name__ == "__main__":
    main()
