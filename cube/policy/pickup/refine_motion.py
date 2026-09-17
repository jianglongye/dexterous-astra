"""Small, independently configurable changes to the complete v2 solver."""

import json
import os
from pathlib import Path
from types import MethodType

import numpy as np

from small_motion import Controller as WorkingController
from baseline import blend, mat, mul, quat
from motion import smoothstep


def local_stroke_frame(fingers, obs):
    """A cube layer is geometrically periodic under quarter-turn regrasping."""
    fingers.arc_yaw -= fingers.regrasp_yaw_shift
    fingers.arc_frame = mul(fingers.fq, quat([0, 0, 1], fingers.arc_yaw))
    fingers.arc_rotation = mat(fingers.arc_frame)
    fingers.stroke_wrist = (
        obs["cube_pos"] + fingers.arc_rotation @ fingers.arc["pos"],
        mul(fingers.arc_frame, fingers.arc["quat"]),
    )


def reachable_continuation_shift(fingers, obs, preferred=None):
    """Use original private geometry to reject unreachable equivalent grasps."""
    import mujoco
    from arm_pickup import PickupPlanner
    from arm_planner import RegraspPlanner
    from arms import RANGES, descendants

    if not hasattr(fingers, "continuation_geometry"):
        fingers.continuation_geometry = PickupPlanner(fingers.info["arm_model_path"], fingers.m)
        geometry = fingers.continuation_geometry
        mujoco.mj_kinematics(geometry.m, geometry.d)
        geometry.motion_reference = {name: geometry.d.body(name).xpos.copy() for name in ["r_wrist", "r_arm_forearm"]}
    geometry = fingers.continuation_geometry
    original = geometry.synchronize(fingers, obs)
    m, d = geometry.m, geometry.d
    # The physical controller settles these fingers before sending the route.
    # This is a prediction in a private model, never a physical state reset.
    for i in range(20, 40):
        name = fingers.m.joint(int(fingers.m.actuator_trnid[i, 0])).name
        original[m.joint(name).qposadr[0]] = [0.15, 0.0, 0.3, 0.2][i % 4]
    choices = []
    for shift in np.arange(-2, 3) * np.pi / 2:
        frame = mul(fingers.fq, quat([0, 0, 1], fingers.arc_yaw - shift))
        rotation = mat(frame)
        q = mul(frame, fingers.arc["quat"])
        p = obs["cube_pos"] + rotation @ fingers.arc["pos"]
        outward = rotation[:, 2]
        tangent = np.cross([0.0, 0.0, 1.0], outward)
        if np.linalg.norm(tangent) < 0.2:
            tangent = np.array([1.0, 0.0, 0.0])
        tangent /= np.linalg.norm(tangent)
        target = p + fingers.regrasp_clearance * outward - fingers.regrasp_lateral * tangent
        choices.append(
            (
                1 - abs(np.dot(obs["wrist"]["r"][1], q)),
                float(shift),
                [target, p + fingers.regrasp_clearance * outward],
                q,
            )
        )
    details, feasible = [], []
    limit = getattr(fingers, "continued_route_limit", None)
    ordered = sorted(
        choices, key=lambda row: (preferred is not None and abs(row[1] - preferred) > 1e-8, row[0], abs(row[1]))
    )
    qadr = m.jnt_qposadr[geometry.joints["r"]]
    for _, shift, targets, q in ordered:
        d.qpos[:] = original
        mujoco.mj_forward(m, d)
        path = []
        for p in targets:
            planner = RegraspPlanner(m, geometry.joints["r"], descendants(m, "r_arm_upper"), RANGES)
            segment = planner.plan(d, m.body("r_wrist").id, p, q)
            if not segment:
                path = []
                break
            path.extend(segment)
            d.qpos[qadr] = segment[-1]
            mujoco.mj_forward(m, d)
        detail = {"shift_rad": shift, "reachable": bool(path)}
        details.append(detail)
        if path:
            peak, travel = 0.0, 0.0
            previous = original[qadr]
            for point in path:
                travel += float(np.linalg.norm(point - previous))
                samples = max(2, int(np.ceil(np.max(np.abs(point - previous)) / 0.06)) + 1)
                for u in np.linspace(0.0, 1.0, samples):
                    d.qpos[qadr] = previous + u * (point - previous)
                    mujoco.mj_kinematics(m, d)
                    peak = max(
                        peak,
                        *(
                            float(np.linalg.norm(d.body(name).xpos - origin))
                            for name, origin in geometry.motion_reference.items()
                        ),
                    )
                previous = point
            detail.update(peak_excursion_m=peak, joint_travel_rad=travel)
            feasible.append((peak, travel, shift))
            if limit is None or peak <= limit:
                print("CONTINUATION_GRASP", json.dumps(details), flush=True)
                return shift


def prepare_local_stroke(fingers, t, obs, new_move=False):
    continuing = new_move and getattr(fingers, "continues_grasp", lambda: False)()
    fingers.original_prepare_stroke(t, obs, new_move)
    if new_move:
        fingers.regrasp_continued_move = continuing
    if new_move or not hasattr(fingers, "regrasp_yaw_reference"):
        fingers.regrasp_yaw_reference = fingers.arc_yaw
    delta = fingers.arc_yaw - fingers.regrasp_yaw_reference
    center = getattr(fingers, "regrasp_yaw_center", 0.0)
    fingers.regrasp_yaw_shift = (
        0.0 if new_move else np.floor((delta - center + np.pi / 4 - 1e-10) / (np.pi / 2)) * (np.pi / 2)
    )
    if getattr(fingers, "face", "L") not in getattr(fingers, "regrasp_yaw_faces", "URFDLB"):
        fingers.regrasp_yaw_shift = 0.0
    elif continuing and getattr(fingers, "regrasp_continue_nearest", False):
        # A half turn keeps the support grasp. Choose the equivalent new pinch
        # nearest the released wrist instead of restarting its first-quarter yaw.
        fingers.regrasp_yaw_shift = reachable_continuation_shift(fingers, obs)
    elif (
        not new_move
        and getattr(fingers, "regrasp_continued_move", False)
        and getattr(fingers, "continued_route_limit", None) is not None
    ):
        fingers.regrasp_yaw_shift = reachable_continuation_shift(fingers, obs, fingers.regrasp_yaw_shift)
    elif getattr(fingers, "grasp_route_limit", None) is not None:
        fingers.regrasp_yaw_shift = reachable_continuation_shift(fingers, obs, fingers.regrasp_yaw_shift)
    local_stroke_frame(fingers, obs)
    print("LOCAL_REGRASP_YAW", fingers.mi, fingers.stroke_count, float(fingers.regrasp_yaw_shift), flush=True)


def configuration(path):
    config = dict(
        pickup_idle_offset_m=None,
        pickup_high_m=0.20,
        pedestal_raise_m=0.30,
        route_lateral_m=0.06,
        retreat_m=0.10,
        unused_relaxation=0.20,
        relax_route=False,
        route_ramp_s=0.0,
        wrap_turn_yaw=False,
        route_ramp_faces=list("URFDLB"),
        wrap_turn_faces=list("URFDLB"),
        wrap_yaw_center_deg=0.0,
        hold_waiting_wrist=False,
        continue_nearest_yaw=False,
        waiting_offset_m=[0.0, 0.0, 0.0],
        waiting_position_m=None,
        continued_route_limit_m=None,
        waiting_lateral_first=False,
        grasp_route_limit_m=None,
        waiting_hold_fingers=False,
        waiting_clearance_m=0.0,
        relax_ease_s=0.0,
        turn_rest_joints=None,
        support_rest_joints=None,
        balanced_pinch=False,
        balanced_pinch_faces=list("URFDLB"),
        reachable_waiting_wrist=False,
        support_alignment=None,
        support_open_joints=None,
        route_lateral_faces=None,
    )
    overrides = json.loads(Path(path).read_text())
    config.update(overrides)
    return config


def configure_turn_rest(fingers, target):
    """Choose resting geometry without changing either active turning finger."""
    q = np.asarray(target, float)
    fingers.open["r"][8:] = q


def configure_support_open(tripod, target):
    """Replace only the free opening target, retaining the loaded grasp."""
    if target is None or getattr(tripod, "refined_open_installed", False):
        return
    q = np.asarray(target, float)
    tripod.open = q.copy()
    tripod.refined_open_installed = True


class Controller(WorkingController):
    def __init__(self, info):
        super().__init__(info)
        self.refinement = configuration(os.environ["RUBIKS_V2_REFINEMENT"])
        self.original_rest = tuple(np.asarray(v).copy() for v in self.rest_wrist)
        self.pickup_rest = self.original_rest
        if self.refinement["pickup_idle_offset_m"] is not None:
            p, q = (np.asarray(v).copy() for v in info["wrist_init"]["r"])
            self.pickup_rest = (p + np.asarray(self.refinement["pickup_idle_offset_m"]), q)
        self.original_route_lateral = self.small["lateral_m"]
        self.small = dict(
            self.small, lateral_m=self.refinement["route_lateral_m"], retreat_m=self.refinement["retreat_m"]
        )
        self.config["unused_relaxation"] = self.refinement["unused_relaxation"]
        configure_turn_rest(self.fingers, self.refinement["turn_rest_joints"])
        configure_support_open(self.fingers.tripod, self.refinement["support_open_joints"])
        self.support_rest_progress = 0.0
        self.support_rest_time = None
        self.route_ramp = None
        self.waiting_hold = None
        self.waiting_final = None
        self.waiting_joints = None
        if self.refinement["wrap_turn_yaw"]:
            # Change only this controller instance, never the simulator or
            # original classes. Finger paths, turn goals and forces stay shared.
            f = self.fingers
            f.regrasp_yaw_faces = self.refinement["wrap_turn_faces"]
            f.regrasp_yaw_center = np.radians(self.refinement["wrap_yaw_center_deg"])
            f.regrasp_continue_nearest = self.refinement["continue_nearest_yaw"]
            f.regrasp_clearance = self.small["clearance_m"]
            f.regrasp_lateral = self.small["lateral_m"]
            f.grasp_route_limit = self.refinement["grasp_route_limit_m"]
            f.continued_route_limit = f.grasp_route_limit or self.refinement["continued_route_limit_m"]
            f.original_prepare_stroke = f.prepare_stroke
            f.prepare_stroke = MethodType(prepare_local_stroke, f)
        print("MOTION_REFINEMENT", json.dumps(self.refinement, sort_keys=True), flush=True)

    def relaxed_route_enabled(self):
        # The original controller already implements release, settle, travel,
        # and gradual restoration of the turning fingers before contact.
        return (
            self.refinement["relax_route"]
            and getattr(self.fingers, "bent_support", False)
            or super().relaxed_route_enabled()
        )

    def prepare_relaxed_route(self, t, obs):
        # The parent owns release checks, fixed wrists, support feedback, and
        # measured settling. Only ease its released right-finger targets.
        command = super().prepare_relaxed_route(t, obs)
        duration = self.refinement.get("relax_ease_s", 0.0)
        if command is None or duration <= 0:
            return command

    def _act(self, t, obs):
        f = self.fingers
        faces = self.refinement.get("route_lateral_faces")
        if faces is not None:
            face = f.moves[min(f.mi, len(f.moves) - 1)][0]
            lateral = self.refinement["route_lateral_m"] if face in faces else self.original_route_lateral
            self.small["lateral_m"] = lateral
            # The private grasp query and physical waypoint must agree.
            f.regrasp_lateral = lateral
        if self.refinement.get("support_open_joints") is not None:
            configure_support_open(f.tripod, self.refinement["support_open_joints"])
        before = f.phase
        # Restrict the nearby waiting pose to the first move. Later pedestal
        # placement and pickup retain their validated waiting-hand clearance.
        self.rest_wrist = self.pickup_rest if f.mi == 0 and not f.parking else self.original_rest
        command = super()._act(t, obs)
        phase = command.get("phase", f.phase)
        # V2's route override keys on f.phase, which remains arc_route during
        # the original arc_relax hold. Restore that hold before any wrist
        # command reaches the simulator, including the route-entry tick.
        if phase == "arc_relax":
            command["wrist"]["r"] = self.route_relaxation["pose"]
        elif (
            phase == "arc_route"
            and not f.parking
            and not f.inverting
            and f.pickup_done
            and self.relaxed_route_enabled()
            and not self.relaxed_route_ready()
        ):
            command["wrist"]["r"] = tuple(np.asarray(v).copy() for v in obs["wrist"]["r"])
        if phase != "arc_route" or not self.relaxed_route_enabled() or self.relaxed_route_ready():
            self.ramp_route(t, obs, command)
        self.shape_support_rest(t, obs, command)
        left_touching = any(
            body.startswith("l_") and other.startswith("cube/") and distance <= 0
            for body, other, distance in obs["contacts"]
        )
        if left_touching:
            return command
        # Key on the phase that produced the command, including its last tick.
        # All approach, loaded grasp, layer-turn and support commands are shared.
        if before == "tripod_high":
            p, q = command["wrist"]["l"]
            command["wrist"]["l"] = (p + np.array([0.0, 0.0, self.refinement["pickup_high_m"] - 0.20]), q)
        return command

    def shape_support_rest(self, t, obs, command):
        target = self.refinement.get("support_rest_joints")
        f = self.fingers
        previous = self.support_rest_time
        elapsed = 0.0 if previous is None else np.clip(t - previous, 0.0, 0.05)
        self.support_rest_time = t
        # The original straight pinky clears the pedestal during acquisition.
        # Restore it during placement rotation, before the subsequent descent.
        ready = f.pickup_done and not f.parking and obs["cube_pos"][2] > 0.28
        self.support_rest_progress += np.clip(float(ready) - self.support_rest_progress, -elapsed, elapsed)
        weight = smoothstep(self.support_rest_progress)
        command["hand"][16:20] = (1 - weight) * command["hand"][16:20] + weight * np.asarray(target)

    def ramp_route(self, t, obs, command):
        duration = self.refinement.get("route_ramp_s", 0.0)
        touching = any(
            body.startswith("r_") and other.startswith("cube/") and distance <= 0
            for body, other, distance in obs["contacts"]
        )
        if duration <= 0 or command.get("phase") != "arc_route" or touching or self.fingers.parking:
            self.route_ramp = None
            return
