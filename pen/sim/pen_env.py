"""Isaac Lab scene: a horizontal pen resting on the fingertips of a palm-up Sharpa Wave hand.

Physics is the official Sharpa USD (drives, armature, limits, convex-hull collision) plus a capsule pen.
No forces, pose writes, constraints, or actuators ever act on the pen after a reset.
Observations and actions are the ones the policy was trained with; rewards and terminations are not used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from spec import (
    CONTROL_DT,
    DECIMATION,
    FINGERS,
    FRICTION,
    GRAVITY,
    HAND_POS,
    HAND_USD,
    JOINTS,
    PALM_UP_QUAT,
    PEN,
    SIM_DT,
)

GRASP = json.loads(
    (Path(__file__).parent / "grasp.json").read_text()
)  # settled "gathered" fingertip grasp, nominal pen
CONTACT_N = 0.01  # N, a finger "touches" the pen above this force (the 10 g pen weighs 0.098 N)
SENSOR_BODIES = (
    ["right_hand_C_MC", "right_thumb_MC", "right_thumb_PP", "right_thumb_DP", "right_thumb_elastomer"]
    + [f"right_{f}_{part}" for f in ("index", "middle", "ring", "pinky") for part in ("PP", "MP", "DP", "elastomer")]
    + ["right_pinky_MC"]
)
N_OBS = 22 * 4 + 3 + 3 + 3 + 3 + 15 + 10 + 5


@configclass
class PenSpinCfg(DirectRLEnvCfg):
    decimation = DECIMATION
    episode_length_s = 10.0
    action_space = 22
    observation_space = N_OBS
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=SIM_DT,
        render_interval=DECIMATION,
        gravity=(0.0, 0.0, -GRAVITY),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=FRICTION, dynamic_friction=FRICTION, restitution=0.0
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2, gpu_max_rigid_contact_count=2**23, gpu_max_rigid_patch_count=2**21
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=0.6, replicate_physics=True)

    obs_episode_s: float = 0.0  # episode length used to normalise the time input (0: episode_length_s)
    target_box: float = 1.0  # joint targets limited to the starting grasp +/- target_box * half range
    # reset randomization
    joint_noise: float = 0.03
    pen_xy_noise: float = 0.002
    pen_yaw_noise_deg: float = 8.0
    mass_range: tuple = (0.85, 1.15)
    friction_range: tuple = (0.9, 1.1)
    settle_s: float = 1.0
    # control
    action_scale: float = 0.05  # rad per control step
    # task
    spin_sign: float = 1.0  # +1: counter-clockwise seen from above
    target_turns: float = 1e9  # finite -> net rotation is capped here and the hold flag turns on


class PenSpinEnv(DirectRLEnv):
    cfg: PenSpinCfg

    def __init__(self, cfg: PenSpinCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        n = self.num_envs
        self.jmap = torch.tensor([self.robot.joint_names.index(j) for j in JOINTS], device=dev)
        lim = self.robot.data.joint_pos_limits[0, self.jmap]
        self.q_lo, self.q_hi = lim[:, 0], lim[:, 1]
        names = self.robot.body_names
        self.tip_frames = torch.tensor([names.index(f"right_{f}_fingertip") for f in FINGERS], device=dev)
        self.grasp_q = torch.tensor(GRASP["q"], device=dev)
        self.grasp_pen_pos = torch.tensor(GRASP["pen_pos"], device=dev)  # world frame of the MuJoCo scene
        self.grasp_pen_quat = torch.tensor(GRASP["pen_quat"], device=dev)

        self.targets = torch.zeros(n, 22, device=dev)
        self.action = torch.zeros(n, 22, device=dev)
        self.ref_pos = torch.zeros(n, 3, device=dev)  # settled pen centre, env-local
        self.heading = torch.zeros(n, device=dev)
        self.progress = torch.zeros(n, device=dev)
        self.best = torch.zeros(n, device=dev)
        self.holding = torch.zeros(n, dtype=torch.bool, device=dev)
        self._randomize_pen_physics(torch.arange(n, device=dev))

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Hand",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(HAND_USD),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=True,
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=HAND_POS, rot=PALM_UP_QUAT),
            actuators={"hand": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
        )
        pen_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Pen",
            spawn=sim_utils.CapsuleCfg(
                radius=PEN["radius"],
                height=PEN["length"] - 2 * PEN["radius"],
                axis="X",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=1.0, solver_position_iteration_count=8
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=PEN["mass"]),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=FRICTION, dynamic_friction=FRICTION),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.9)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.16, 0.0, HAND_POS[2] + 0.2)),
        )
        self.robot = Articulation(robot_cfg)
        self.pen = RigidObject(pen_cfg)
        # PhysX filtered contact reporting needs one sensor per body.
        self.contact_sensors = []
        for body in SENSOR_BODIES:
            sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Hand/{body}",
                    filter_prim_paths_expr=["/World/envs/env_.*/Pen"],
                    track_contact_points=True,
                    max_contact_data_count_per_prim=16,
                )
            )
            self.scene.sensors[body] = sensor
            finger = next((k for k, f in enumerate(FINGERS) if body.startswith(f"right_{f}_")), -1)
            self.contact_sensors.append((finger, sensor))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["pen"] = self.pen
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light.func("/World/Light", light)

    def _randomize_pen_physics(self, env_ids):
        c = self.cfg
        view = self.pen.root_physx_view
        ids = env_ids.cpu()
        masses = view.get_masses()
        u = torch.rand(len(ids), 1)
        masses[ids] = PEN["mass"] * (c.mass_range[0] + (c.mass_range[1] - c.mass_range[0]) * u)
        view.set_masses(masses, ids)
        mats = view.get_material_properties()
        f = FRICTION * (c.friction_range[0] + (c.friction_range[1] - c.friction_range[0]) * torch.rand(len(ids), 1))
        mats[ids, :, 0] = f
        mats[ids, :, 1] = f
        view.set_material_properties(mats, ids)

    # ------------------------------------------------------------------ helpers
    def _to_isaac(self, x):
        out = torch.zeros(x.shape[0], 22, device=self.device)
        out[:, self.jmap] = x
        return out

    def _q(self):
        return self.robot.data.joint_pos[:, self.jmap]

    def _qd(self):
        return self.robot.data.joint_vel[:, self.jmap]

    def _pen_state(self):
        pos = self.pen.data.root_pos_w - self.scene.env_origins
        quat = self.pen.data.root_quat_w
        axis = quat_apply(quat, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(quat.shape[0], 3))
        return pos, quat, axis

    def _finger_forces(self):
        per = torch.zeros(self.num_envs, 5, device=self.device)
        palm = torch.zeros(self.num_envs, device=self.device)
        for finger, sensor in self.contact_sensors:
            f = sensor.data.force_matrix_w[:, 0, 0, :].norm(dim=-1)
            if finger >= 0:
                per[:, finger] += f
            else:
                palm += f
        return per, palm

    def physx_penetration(self):
        """Deepest PhysX-reported pen-hand contact penetration per env (m, >= 0)."""
        worst = torch.zeros(self.num_envs, device=self.device)
        for _, sensor in self.contact_sensors:
            _, _, _, sep, count, start = sensor.contact_physx_view.get_contact_data(dt=SIM_DT)
            count = count.reshape(self.num_envs, -1)[:, 0].long()
            start = start.reshape(self.num_envs, -1)[:, 0].long()
            total = int(count.sum())
            if total == 0:
                continue
            env_of = torch.repeat_interleave(torch.arange(self.num_envs, device=self.device), count)
            offs = torch.arange(total, device=self.device) - torch.repeat_interleave(
                torch.cumsum(count, 0) - count, count
            )
            vals = sep.reshape(-1)[start[env_of] + offs]
            worst.scatter_reduce_(0, env_of, (-vals).clamp(min=0), reduce="amax")
        return worst

    def _write_state(self, env_ids, q, targets, pen_pos, pen_quat):
        zeros = torch.zeros(len(env_ids), 22, device=self.device)
        self.robot.write_joint_state_to_sim(self._to_isaac(q), zeros, env_ids=env_ids)
        self.robot.set_joint_position_target(self._to_isaac(targets), env_ids=env_ids)
        root = torch.zeros(len(env_ids), 13, device=self.device)
        root[:, :3] = pen_pos + self.scene.env_origins[env_ids]
        root[:, 3:7] = pen_quat
        self.pen.write_root_state_to_sim(root, env_ids=env_ids)

    def settle(self, env_ids, q, pos, quat):
        """Hold the commanded grasp and let the pen come to rest under the simulated physics."""
        self._write_state(env_ids, q, q, pos, quat)
        for _ in range(int(self.cfg.settle_s / SIM_DT)):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=SIM_DT)
        p, qt, axis = self._pen_state()
        per, palm = self._finger_forces()
        tips = (per[env_ids] > CONTACT_N).sum(-1)
        v = self.pen.data.root_lin_vel_w[env_ids].norm(dim=-1)
        w = self.pen.data.root_ang_vel_w[env_ids].norm(dim=-1)
        tilt = torch.rad2deg(torch.asin(axis[env_ids, 2].abs().clamp(max=1)))
        checks = {
            "height": p[env_ids, 2] > self.grasp_pen_pos[2] - 0.015,
            "tilt": tilt < 10,
            "still": (v < 0.02) & (w < 0.5),
            "tips3": tips >= 3,
            "no_palm": palm[env_ids] < CONTACT_N,
        }
        valid = torch.stack(list(checks.values())).all(0)
        print(
            "[settle]",
            {k: round(c.float().mean().item(), 3) for k, c in checks.items()},
            "pen z",
            round(p[env_ids, 2].mean().item(), 4),
            "grasp z",
            round(self.grasp_pen_pos[2].item(), 4),
            "touch",
            [round(x, 2) for x in (per[env_ids] > CONTACT_N).float().mean(0).tolist()],
            "q err",
            round((self._q()[env_ids] - q).abs().max().item(), 3),
            flush=True,
        )
        return valid, self._q()[env_ids].clone(), p[env_ids].clone(), qt[env_ids].clone(), tips

    # ------------------------------------------------------------------ policy interface
    def _pre_physics_step(self, actions):
        self.action = actions.clamp(-1, 1)
        half = 0.5 * (self.q_hi - self.q_lo) * self.cfg.target_box
        lo, hi = torch.maximum(self.grasp_q - half, self.q_lo), torch.minimum(self.grasp_q + half, self.q_hi)
        self.targets = (self.targets + self.cfg.action_scale * self.action).clamp(lo, hi)
        self.robot.set_joint_position_target(self._to_isaac(self.targets))

    def _apply_action(self):
        pass

    def _get_observations(self):
        pos, quat, axis = self._pen_state()
        q, qd = self._q(), self._qd()
        per, _ = self._finger_forces()
        tips = self.robot.data.body_pos_w[:, self.tip_frames] - self.pen.data.root_pos_w[:, None]
        span = self.q_hi - self.q_lo
        remaining = ((self.cfg.target_turns * 2 * math.pi - self.best) / (6 * math.pi)).clamp(0, 1)
        # the pen is observed as an unoriented line (invariant to its 180 degree symmetry)
        ax, ay, az = axis.unbind(-1)
        pen_dir = torch.stack([ax * ax - ay * ay, 2 * ax * ay, az * az], -1)
        pen_tilt = torch.stack([10 * ax * az, 10 * ay * az], -1)
        obs = torch.cat(
            [
                2 * (q - self.q_lo) / span - 1,
                0.1 * qd,
                2 * (self.targets - self.q_lo) / span - 1,
                self.action,
                10 * (pos - self.ref_pos),
                pen_dir,
                0.5 * self.pen.data.root_lin_vel_w,
                0.2 * self.pen.data.root_ang_vel_w,
                10 * tips.reshape(-1, 15),
                (per > CONTACT_N).float(),
                per.clamp(max=2.0),
                remaining[:, None],
                self.holding.float()[:, None],
                pen_tilt,
                (self.episode_length_buf.float() * CONTROL_DT / (self.cfg.obs_episode_s or self.cfg.episode_length_s))[
                    :, None
                ],
            ],
            -1,
        )
        return {"policy": torch.nan_to_num(obs).clamp(-10, 10)}

    def _get_dones(self):
        """Track net rotation for the observations; environments only end at the time limit."""
        c = self.cfg
        _, _, axis = self._pen_state()
        heading = torch.atan2(axis[:, 1], axis[:, 0])
        dpsi = torch.remainder(heading - self.heading + math.pi, 2 * math.pi) - math.pi
        self.heading = heading
        self.progress += c.spin_sign * dpsi
        gain = (self.progress - self.best).clamp(min=0)
        target = c.target_turns * 2 * math.pi
        spinning = ~self.holding
        gain = torch.where(spinning, torch.minimum(gain, (target - self.best).clamp(min=0)), torch.zeros_like(gain))
        self.best = self.best + gain
        self.holding |= spinning & (self.best >= target - 1e-6)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(self.holding), time_out

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids):
        """Start environment i from its own settled grasp, row i of self.bank."""
        env_ids = torch.as_tensor(env_ids, device=self.device)
        super()._reset_idx(env_ids)
        b = {k: v[env_ids] for k, v in self.bank.items()}
        self._write_state(env_ids, b["q"], b["targets"], b["pos"], b["quat"])
        self.targets[env_ids] = b["targets"]
        self.action[env_ids] = 0
        self.ref_pos[env_ids] = b["pos"]
        axis = quat_apply(b["quat"], torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(len(env_ids), 3))
        self.heading[env_ids] = torch.atan2(axis[:, 1], axis[:, 0])
        for buf in (self.progress, self.best):
            buf[env_ids] = 0
        self.holding[env_ids] = False
