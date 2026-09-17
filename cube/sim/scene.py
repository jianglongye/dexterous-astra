# /// script
# requires-python = ">=3.10"
# dependencies = ["mujoco==3.13.0", "numpy", "imageio"]
# ///
"""MuJoCo cube + Wuji hands on generic seven-DOF arms and a fixed base.

uv run scene.py [--scramble "R U F'"] [--png still.png]
"""

import itertools
import os
import re
import urllib.request
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

MUJOCO_SHA = "6783903ffece5bcbc2f8c73ba44975df86f33b9e"
WUJI_SHA = "4c1073d0a3ad1daaf6546d219db751d8448d3888"
CUBE_URL = f"https://raw.githubusercontent.com/google-deepmind/mujoco/{MUJOCO_SHA}/model/cube/"
WUJI_URL = f"https://raw.githubusercontent.com/wuji-technology/wuji-description/{WUJI_SHA}/hand2/hand2_beta2/body/"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "dexterous-astra" / "rubiks-cube"

PITCH = 0.019  # cubie spacing (m); cube edge = 3 * PITCH
CUBE_POS = (0.0, 0.0, 0.2)  # cube centre, resting on the pedestal
PEDESTAL = 0.02  # pedestal half-width (m)
# Initial wrist poses: position and quaternion (w x y z). Only the legacy
# floating embodiment uses these as mocap targets; arms use joint motors.
WRISTS = {
    "l": ((-0.15, 0.07, 0.2), (0.70710678, 0.0, -0.70710678, 0.0)),
    "r": ((-0.15, -0.07, 0.2), (0.70710678, 0.0, -0.70710678, 0.0)),
}

# Face letter -> outward normal, and the (right, down) axes of its 3x3 sticker grid (Kociemba order).
FACES = "URFDLB"
NORMAL = {"U": (0, 0, 1), "R": (1, 0, 0), "F": (0, -1, 0), "D": (0, 0, -1), "L": (-1, 0, 0), "B": (0, 1, 0)}
GRID = {
    "U": ((1, 0, 0), (0, -1, 0)),
    "R": ((0, 1, 0), (0, 0, -1)),
    "F": ((1, 0, 0), (0, 0, -1)),
    "D": ((1, 0, 0), (0, 1, 0)),
    "L": ((0, -1, 0), (0, 0, -1)),
    "B": ((-1, 0, 0), (0, 0, -1)),
}
SOLVED = "".join(f * 9 for f in FACES)

VISUAL = mujoco.MjvOption()
VISUAL.geomgroup[2] = 0  # hide the hands' translucent collision meshes


def _fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def fetch_assets():
    """Download the pinned cube and hand models into the cache (once)."""
    cube = _fetch(CUBE_URL + "cube_3x3x3.xml", CACHE / f"cube-{MUJOCO_SHA[:8]}" / "cube_3x3x3.xml")
    for f in re.findall(r'file="([^"]+)"', cube.read_text()):
        _fetch(CUBE_URL + "assets/" + f, cube.parent / "assets" / f)
    hands = {}
    for side in ("left", "right"):
        xml = _fetch(WUJI_URL + f"mjcf/{side}.xml", CACHE / f"wuji-{WUJI_SHA[:8]}" / "mjcf" / f"{side}.xml")
        for f in re.findall(r'file="([^"]+)"', xml.read_text()):
            _fetch(WUJI_URL + f"meshes/{side}/{f}", xml.parent.parent / "meshes" / side / f)
        hands[side[0]] = xml
    return cube, hands


def _load(xml, parent, subdir):
    """Load a child spec with absolute asset paths and the parent's solver options."""
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.default.name = xml.stem  # an unnamed child default would save as class="", which fails to reload
    base = xml.parent / subdir
    for asset in list(spec.meshes) + list(spec.textures):
        if asset.file:
            asset.file = str((base / asset.file).resolve())
    spec.meshdir = spec.texturedir = ""
    for k in ("timestep", "tolerance", "iterations", "integrator", "jacobian"):
        setattr(spec.option, k, getattr(parent.option, k))
    spec.memory = parent.memory
    return spec


def _lookat(pos, target):
    f = np.subtract(target, pos) / np.linalg.norm(np.subtract(target, pos))
    x = np.cross(f, (0, 0, 1))
    x /= np.linalg.norm(x)
    return {"pos": pos, "xyaxes": [*x, *np.cross(x, f)]}


def build_spec(free_cube=True, embodiment="arms"):
    cube_xml, hand_xmls = fetch_assets()
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.modelname = "rubiks_cube_wuji"
    spec.option.timestep = 0.001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.memory = 256 << 20
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
    world = spec.worldbody
    world.add_light(pos=(0, 0, 1.5), dir=(0, 0, -1), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(1, 1, 0.01), rgba=(0.85, 0.85, 0.82, 1))
    h = CUBE_POS[2] - 1.5 * PITCH
    world.add_geom(
        name="pedestal",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(PEDESTAL, PEDESTAL, h / 2),
        pos=(0, 0, h / 2),
        rgba=(0.3, 0.3, 0.3, 1),
    )
    world.add_camera(name="closeup", **_lookat((0.22, -0.12, 0.32), CUBE_POS))
    overview = ((0.85, -1.15, 0.9), (-0.3, 0, 0.28)) if embodiment == "arms" else ((0.55, -0.45, 0.5), CUBE_POS)
    world.add_camera(name="overview", **_lookat(*overview))

    cube = _load(cube_xml, spec, "assets")
    for act in list(cube.actuators):  # the cube is passive: no face motors
        cube.delete(act)
    if free_cube:
        cube.body("core").add_freejoint(name="core")
    spec.attach(cube, prefix="cube/", frame=world.add_frame(pos=CUBE_POS))

    if embodiment == "arms":
        from arms import add_arm, add_base

        base = add_base(spec)
    for s, xml in hand_xmls.items():
        hand = _load(xml, spec, "../meshes/" + ("left" if s == "l" else "right"))
        pos, quat = WRISTS[s]
        if embodiment == "arms":
            frame = add_arm(spec, base, s, WRISTS[s])
            spec.attach(hand, prefix="", frame=frame)
            continue
        spec.attach(hand, prefix="", frame=world.add_frame(pos=pos, quat=quat))
        root = spec.body(f"{s}_wrist")
        root.add_freejoint(name=f"{s}_wrist")
        for b in [root, *root.find_all(mujoco.mjtObj.mjOBJ_BODY)]:
            b.gravcomp = 1  # stands in for the arm holding the hand up
        world.add_body(name=f"{s}_wrist_target", mocap=True, pos=pos, quat=quat)
        weld = spec.add_equality(
            type=mujoco.mjtEq.mjEQ_WELD,
            name1=f"{s}_wrist",
            name2=f"{s}_wrist_target",
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            solref=(0.01, 1),
        )
        weld.data = [0] * 10 + [1]  # anchor at the wrist origin (MjSpec's default is 1 m away), pose from qpos0
    return spec


# ---------------------------------------------------------------- cube state

CENTRES = ("pX", "nX", "pY", "nY", "pZ", "nZ")


def apply_detents(model, data, k=0.02):
    """Passive spring (N*m/rad) pulling each face centre to its nearest quarter turn."""
    for n in CENTRES:
        j = model.joint(f"cube/{n}")
        th = data.qpos[j.qposadr[0]]
        data.qfrc_applied[j.dofadr[0]] = -k * (th - np.round(th / (np.pi / 2)) * np.pi / 2)


def cubies(model):
    """Body ids of the 26 cubies and their home positions in cubie units."""
    core = model.body("cube/core").id
    ids = [b for b in range(model.nbody) if model.body_parentid[b] == core]
    home = np.array([model.geom_pos[model.body_geomadr[b]] / PITCH for b in ids]).round().astype(int)
    return ids, home


def cubie_rotations(model, data):
    ids, home = cubies(model)
    core = data.xmat[model.body("cube/core").id].reshape(3, 3)
    return np.array([core.T @ data.xmat[b].reshape(3, 3) for b in ids]), home


GROUP = np.array(
    [
        m
        for m in (
            np.eye(3)[list(p)] * s
            for p in itertools.permutations(range(3))
            for s in itertools.product((1, -1), repeat=3)
        )
        if np.linalg.det(m) > 0
    ]
)


def snap(rots):
    """Nearest of the 24 cube rotations for each cubie, and the angle to it in degrees."""
    score = np.einsum("gij,nij->ng", GROUP, rots)
    best = score.argmax(1)
    cos = (score[np.arange(len(rots)), best] - 1) / 2
    return GROUP[best].round().astype(int), np.degrees(np.arccos(np.clip(cos, -1, 1)))


def facelets(model, data):
    """Kociemba facelet string (URFDLB), plus the worst cubie misalignment in degrees."""
    rots, home = cubie_rotations(model, data)
    snaps, angles = snap(rots)
    out = ["?"] * 54
    for S, p in zip(snaps, home):
        slot = S @ p
        for k in np.flatnonzero(p):
            d0 = np.zeros(3, int)
            d0[k] = p[k]
            face = next(f for f in FACES if NORMAL[f] == tuple(S @ d0))
            right, down = GRID[face]
            r, c = int(np.dot(slot, down)) + 1, int(np.dot(slot, right)) + 1
            out[FACES.index(face) * 9 + r * 3 + c] = next(f for f in FACES if NORMAL[f] == tuple(d0))
    return "".join(out), float(angles.max())


# Kociemba facelet indices of the 8 corners and 12 edges (first index on U/D, or F/B for the middle-layer edges).
CORNERS = ((8, 9, 20), (6, 18, 38), (0, 36, 47), (2, 45, 11), (29, 26, 15), (27, 44, 24), (33, 53, 42), (35, 17, 51))
EDGES = (
    (5, 10),
    (7, 19),
    (3, 37),
    (1, 46),
    (32, 16),
    (28, 25),
    (30, 43),
    (34, 52),
    (23, 12),
    (21, 41),
    (50, 39),
    (48, 14),
)


def legal(fl):
    """True if the facelets are reachable by face turns: no twisted corner, flipped edge, or lone swap."""

    def parity(p):
        return sum(a > b for i, a in enumerate(p) for b in p[i + 1 :]) % 2

    try:
        cp = [[set(SOLVED[j] for j in c) for c in CORNERS].index(set(fl[j] for j in c)) for c in CORNERS]
        ep = [[set(SOLVED[j] for j in e) for e in EDGES].index(set(fl[j] for j in e)) for e in EDGES]
    except ValueError:
        return False
    twist = sum(next(k for k, j in enumerate(c) if fl[j] in "UD") for c in CORNERS)
    flip = sum(fl[e[0]] != SOLVED[EDGES[i][0]] for e, i in zip(EDGES, ep))
    return (
        sorted(cp) == list(range(8))
        and sorted(ep) == list(range(12))
        and twist % 3 == 0
        and flip % 2 == 0
        and parity(cp) == parity(ep)
    )


def _rotmat(axis, angle):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), angle)
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3).round()


def parse_moves(moves):
    out = []
    for tok in moves.split():
        face, suffix = tok[0], tok[1:]
        assert face in FACES and suffix in ("", "'", "2"), f"bad move {tok!r}"
        out.append((face, {"": -np.pi / 2, "'": np.pi / 2, "2": np.pi}[suffix]))
    return out


def invert(moves):
    inv = {"": "'", "'": "", "2": "2"}
    return " ".join(t[0] + inv[t[1:]] for t in reversed(moves.split()))


def apply_scramble(model, data, moves):
    """Set cubie joints to the scrambled configuration (initial state only)."""
    ids, home = cubies(model)
    rots = [np.eye(3) for _ in ids]
    for face, angle in parse_moves(moves):
        n = np.array(NORMAL[face])
        R = _rotmat(n, angle)
        for i, p in enumerate(home):
            if np.dot(rots[i] @ p, n) == 1:
                rots[i] = R @ rots[i]
    for b, R in zip(ids, rots):
        j = model.body_jntadr[b]
        adr = model.jnt_qposadr[j]
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            axis = model.jnt_axis[j]
            data.qpos[adr] = np.arctan2(
                np.dot(axis, [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]), np.trace(R) - 1
            )
        else:
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, R.flatten())
            data.qpos[adr : adr + 4] = q
    mujoco.mj_forward(model, data)
