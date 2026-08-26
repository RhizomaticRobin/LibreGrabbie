# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-agnostic pose + measurement layer for the sim-ready robotic hand.

Shared by the Isaac Lab / Kit front-end (``robot_hand_show.py``) and the native
Newton GL front-end (``robot_hand_show_newton.py``) so the two share **one source
of truth** for the actively-tuned pose constants (``OPPOSITION``/``POSES``/etc.).

This module imports neither Isaac Lab nor Newton. ``pxr`` is imported lazily inside
the geometry helpers, so importing this module is cheap and does not require USD.

The two runtime dependencies a front-end must supply:
  * a joint index/limits lookup (``JointLookup``) — how joint *names* map to DOF
    indices and where their authored limits live (Isaac Lab ``robot.joint_names``
    order, or Newton ``joint_qd_start`` order);
  * a body-pose reader (``body_pose(idx) -> (pos[3], rot[3,3])``) — how to read a
    body's live world transform (Isaac Lab's ``NewtonManager._state_0.body_q`` or
    Newton's native ``state.body_q``).
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

# --- static kinematic structure (from the converted asset's joint tree) ---------------
# The converter report confirms these joint/body name assignments.

THUMB = {"opp": "Revolute_5", "hinge": "Revolute_1", "lat": "Revolute_2",
         "mcp": "Revolute_3", "pip": "Revolute_4", "dip": "Revolute_6"}
THUMB_TIP = "Distal_Phalanx_Bone_V02_01"
# Finger chains A-D; which of A/C is index vs ring is resolved at runtime by lateral
# distance from the thumb base (B=middle and D=pinky are fixed by their CAD part names
# and used as a sanity check on the ordering).
CHAINS = {
    "A": {"abd": "Revolute_20", "mcp": "Revolute_7", "pip": "Revolute_11", "dip": "Revolute_12",
          "tip": "Distal_Phalanx_Bone_V02", "meta": "Metacarpal_Bone_V02_02"},
    "B": {"abd": "Revolute_21", "mcp": "Revolute_18", "pip": "Revolute_13", "dip": "Revolute_14",
          "tip": "Distal_Phalanx_Bone_V02_03", "meta": "Metacarpal_Bone_V02"},
    "C": {"abd": "Revolute_22", "mcp": "Revolute_15", "pip": "Revolute_16", "dip": "Revolute_17",
          "tip": "Distal_Phalanx_Bone_V02_04", "meta": "Metacarpal_Bone_V02_01"},
    "D": {"abd": "Revolute_19", "mcp": "Revolute_8", "pip": "Revolute_9", "dip": "Revolute_10",
          "tip": "Distal_Phalanx_Bone_V02_02", "meta": "Metacarpal_Bone_V02_pinky"},
}


# --- frac -> joint angle math ----------------------------------------------------------

class JointLookup(NamedTuple):
    """Name -> DOF index plus authored limits, for the frac-to-angle conversion."""

    col: dict[str, int]
    lower: np.ndarray
    upper: np.ndarray
    n_dof: int

    def angle(self, jname: str, frac: float) -> float:
        """Fraction -> joint angle.

        frac in [0,1] flexes toward the larger-|limit| side; signed frac in [-1,1]
        for two-sided joints (abductions, thumb opposition). 95% clamp.
        """
        i = self.col[jname]
        lo, hi = float(self.lower[i]), float(self.upper[i])
        if frac >= 0:
            end = hi if abs(hi) >= abs(lo) else lo
        else:
            end = lo if abs(hi) >= abs(lo) else hi
        return 0.95 * abs(frac) * end

    def angles(self, pose: dict[str, float]) -> np.ndarray:
        """A full DOF target vector (shape ``(n_dof,)``, float64). Front-ends cast/wrap."""
        t = np.zeros(self.n_dof, dtype=np.float64)
        for jname, frac in pose.items():
            t[self.col[jname]] = self.angle(jname, frac)
        return t


# --- pose builders (thumb/merge are pure; finger/touch_finger need finger_of) ----------

def merge(*parts: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in parts:
        out.update(p)
    return out


def thumb(*, opp: float = 0.0, hinge: float = 0.0, lat: float = 0.0,
          mcp: float = 0.0, pip: float = 0.0, dip: float = 0.0) -> dict[str, float]:
    # opposition base sign inverted: the asset's Revolute_5 rotates opposite to the
    # physical 3D-printed hand (verified manually), so negate to drive it correctly.
    return {THUMB["opp"]: -opp, THUMB["hinge"]: hinge, THUMB["lat"]: lat,
            THUMB["mcp"]: mcp, THUMB["pip"]: pip, THUMB["dip"]: dip}


def resolve_finger_of(meta_positions: dict[str, np.ndarray], thumb_base_pos: np.ndarray) -> dict[str, str]:
    """Map {index,middle,ring,pinky} -> chain key by metacarpal distance from the thumb base.

    B must come out as middle and D as pinky (their CAD part names fix them); anything
    else means the asset's joint tree changed and the constants below are stale.
    """
    by_dist = sorted(CHAINS, key=lambda k: float(np.linalg.norm(meta_positions[k] - thumb_base_pos)))
    finger_of = {"index": by_dist[0], "middle": by_dist[1], "ring": by_dist[2], "pinky": by_dist[3]}
    if finger_of["middle"] != "B" or finger_of["pinky"] != "D":
        raise RuntimeError(
            f"finger ordering sanity failed: {finger_of} (B must be middle, D pinky) — "
            "asset joint tree changed; re-derive the CHAINS map.")
    return finger_of


class Poses(NamedTuple):
    """The full pose battery bound to a resolved ``finger_of``."""

    finger: Callable
    thumb: Callable
    merge: Callable
    touch_finger: Callable
    OPPOSITION: dict
    CURL: float
    TUCK: dict
    POSES: list
    PROBE_POSES: list


def build_poses(finger_of: dict[str, str]) -> Poses:
    """Build the pose battery for a resolved ``finger_of``.

    The constants here are the result of ``--probe-grid`` tuning on the physical
    3D-printed hand; see the docstrings/comments preserved verbatim from the original.
    """

    def finger(fkey: str, *, abd: float = 0.0, curl: float = 0.0,
               mcp: float | None = None, pip: float | None = None,
               dip: float | None = None) -> dict[str, float]:
        c = CHAINS[finger_of[fkey]]
        return {
            c["abd"]: abd,
            c["mcp"]: mcp if mcp is not None else curl,
            c["pip"]: pip if pip is not None else curl,
            c["dip"]: dip if dip is not None else curl,
        }

    def touch_finger(fkey: str) -> dict[str, float]:
        return finger(fkey, curl=0.9, abd=0.9)

    # Re-tuned after reversing the opposition base to match the physical hand
    # (grid v8, corrected opposition): the thumb now swings forward to each curled
    # fingertip pad-first. Residual gaps index 48 / middle 40 / ring 47 / pinky 57 mm
    # are the joint-range floor — the thumb's forward hinge maxes at 125 deg. For the
    # pinky (farthest finger) the thumb is held LESS curled (tc=0.3): it is wrist-mounted
    # behind the fingertip plane, so curling lifts its tip up into that plane (closer),
    # while over-extending drops it away — tc=0.3 is the extended-but-reaching optimum.
    OPPOSITION = {
        "index": merge(thumb(opp=-1.0, hinge=0.6, lat=-0.9, mcp=0.65, pip=0.65, dip=0.65),
                       touch_finger("index")),
        "middle": merge(thumb(opp=-1.0, hinge=0.6, lat=-0.9, mcp=0.65, pip=0.65, dip=0.65),
                        touch_finger("middle")),
        "ring": merge(thumb(opp=-1.0, hinge=0.8, lat=-0.9, mcp=0.5, pip=0.5, dip=0.5),
                      touch_finger("ring")),
        "pinky": merge(thumb(opp=-1.0, hinge=1.0, lat=-0.9, mcp=0.3, pip=0.3, dip=0.3),
                       touch_finger("pinky")),
    }

    CURL = 0.92  # "closed" finger
    TUCK = merge(thumb(opp=-0.6, hinge=0.4, mcp=0.6, pip=0.6, dip=0.5))  # thumb over the palm

    POSES: list[tuple[str, dict]] = [
        ("open", {}),
        ("fist", merge(*(finger(f, curl=CURL) for f in ("index", "middle", "ring", "pinky")), TUCK)),
        ("open", {}),
        ("touch index", OPPOSITION["index"]),
        ("touch middle", OPPOSITION["middle"]),
        ("touch ring", OPPOSITION["ring"]),
        ("touch pinky", OPPOSITION["pinky"]),
        ("open", {}),
        ("peace", merge(finger("index", abd=0.9), finger("middle", abd=-0.9),
                        finger("ring", curl=CURL), finger("pinky", curl=CURL), TUCK)),
        ("vulcan", merge(finger("index", abd=0.9), finger("middle", abd=0.9),
                         finger("ring", abd=-0.9), finger("pinky", abd=-0.9),
                         thumb(opp=0.5))),
        ("the bird", merge(finger("index", curl=CURL), finger("ring", curl=CURL),
                           finger("pinky", curl=CURL), TUCK)),
        ("rock on", merge(finger("middle", curl=CURL), finger("ring", curl=CURL), TUCK)),
        # thumbs up / shaka: opp sign flipped to the index side to match the reversed
        # opposition base (these are extended-thumb gestures; on the pinky side the thumb
        # crosses through the palm). TUCK and vulcan read fine on their current sides.
        ("thumbs up", merge(*(finger(f, curl=CURL) for f in ("index", "middle", "ring", "pinky")),
                            thumb(opp=-0.8))),
        ("shaka", merge(finger("index", curl=CURL), finger("middle", curl=CURL),
                        finger("ring", curl=CURL), finger("pinky", abd=-0.9),
                        thumb(opp=-0.9))),
        ("ok sign", merge(OPPOSITION["index"],
                          finger("middle", abd=-0.4), finger("ring", abd=-0.7), finger("pinky", abd=-0.9))),
        ("open", {}),
    ]

    PROBE_POSES: list[tuple[str, dict]] = [
        ("open", {}),
        ("abd_plus", merge(*(finger(f, abd=0.9) for f in ("index", "middle", "ring", "pinky")))),
        ("abd_minus", merge(*(finger(f, abd=-0.9) for f in ("index", "middle", "ring", "pinky")))),
        ("opp_plus", thumb(opp=0.7)),
        ("opp_minus", thumb(opp=-0.7)),
        ("thumb_curl", thumb(hinge=0.5, mcp=0.6, pip=0.6, dip=0.5)),
        ("curl_all", merge(*(finger(f, curl=0.7) for f in ("index", "middle", "ring", "pinky")))),
        ("touch index", OPPOSITION["index"]),
        ("touch middle", OPPOSITION["middle"]),
        ("touch ring", OPPOSITION["ring"]),
        ("touch pinky", OPPOSITION["pinky"]),
    ]

    return Poses(finger=finger, thumb=thumb, merge=merge, touch_finger=touch_finger,
                 OPPOSITION=OPPOSITION, CURL=CURL, TUCK=TUCK, POSES=POSES, PROBE_POSES=PROBE_POSES)


# --- measurement layer (true fingertip + pad orientation) ------------------------------
# Pure geometry given a pxr stage and a body-pose reader. The stage may be Kit's live
# stage (Isaac Lab) or the on-disk USDA opened with Usd.Stage.Open (native Newton); the
# runtime pose always comes from the solver state via ``body_pose``.

def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def tip_offset(stage, root_prim_path: str, body_name: str) -> np.ndarray:
    """Farthest mesh vertex from the body origin in the distal body's local frame.

    The joint anchor sits near the origin; the fingertip is the far end of the mesh.
    """
    from pxr import Usd, UsdGeom

    body_prim = stage.GetPrimAtPath(f"{root_prim_path}/{body_name}")
    if not body_prim or not body_prim.IsValid():
        raise RuntimeError(f"prim for body {body_name!r} not found at {root_prim_path}/{body_name}")
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    best, best_d = np.zeros(3), -1.0
    for prim in Usd.PrimRange(body_prim):
        if prim.GetTypeName() != "Mesh":
            continue
        pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        rel, _ = xcache.ComputeRelativeTransform(prim, body_prim)
        m = np.array([[rel[i][j] for j in range(4)] for i in range(4)], dtype=float)
        pts_b = pts @ m[:3, :3] + m[3, :3]
        d = np.linalg.norm(pts_b, axis=1)
        k = int(d.argmax())
        if d[k] > best_d:
            best_d, best = float(d[k]), pts_b[k]
    return best


class HandProbe:
    """True-fingertip world positions and pad-direction measurement.

    Args:
        stage: pxr ``Usd.Stage`` carrying the hand prims (Kit live stage or on-disk USDA).
        root_prim_path: path prefix such that ``f"{root}/{body_name}"`` resolves a body prim.
        body_index: ``name -> body index`` matching the solver's body ordering.
        body_pose: ``idx -> (pos[3], rot[3,3])`` reading live body world transforms.
        joint_lookup: :class:`JointLookup` (for ``col``/``upper``/``lower`` in pad_local_of).
        tip_bodies: ``{"thumb", "A", "B", "C", "D"}`` -> body index.
        meta_bodies, thumb_base_idx, palm_idx: indices used by the probe prints.
    """

    DIP_OF = {"thumb": THUMB["dip"], **{k: CHAINS[k]["dip"] for k in CHAINS}}

    def __init__(self, stage, root_prim_path: str, body_index: Callable[[str], int],
                 body_pose: Callable[[int], tuple[np.ndarray, np.ndarray]],
                 joint_lookup: JointLookup, tip_bodies: dict, meta_bodies: dict,
                 thumb_base_idx: int, palm_idx: int) -> None:
        self.stage = stage
        self.root = root_prim_path
        self.body_index = body_index
        self.body_pose = body_pose
        self.jl = joint_lookup
        self.tip_bodies = tip_bodies
        self.meta_bodies = meta_bodies
        self.thumb_base_idx = thumb_base_idx
        self.palm_idx = palm_idx
        # Static mesh geometry — compute once.
        self.tip_local = {
            k: tip_offset(stage, root_prim_path, THUMB_TIP if k == "thumb" else CHAINS[k]["tip"])
            for k in tip_bodies
        }

    def tip_world(self, key: str) -> np.ndarray:
        pos, r = self.body_pose(self.tip_bodies[key])
        return pos + r @ self.tip_local[key]

    def joint_frame_world(self, jname: str) -> tuple[np.ndarray, np.ndarray]:
        """World (axis, anchor) of a revolute joint from its body0 live pose."""
        jp = self.stage.GetPrimAtPath(f"{self.root}/{jname}")
        b0_name = str(jp.GetRelationship("physics:body0").GetTargets()[0]).split("/")[-1]
        lp0 = np.array(jp.GetAttribute("physics:localPos0").Get(), dtype=float)
        lr0 = jp.GetAttribute("physics:localRot0").Get()
        rx, ry, rz = lr0.GetImaginary()
        r_local = quat_to_rot(rx, ry, rz, lr0.GetReal())
        p0, r0 = self.body_pose(self.body_index(b0_name))
        axis = r0 @ r_local @ np.array([0.0, 0.0, 1.0])
        return axis / np.linalg.norm(axis), p0 + r0 @ lp0

    def pad_local_of(self, key: str) -> np.ndarray:
        """Pad direction in the distal body frame: the side that leads when the digit's
        DIP flexes through its working range (the curl-concave side)."""
        jname = self.DIP_OF[key]
        i = self.jl.col[jname]
        sign = 1.0 if abs(float(self.jl.upper[i])) >= abs(float(self.jl.lower[i])) else -1.0
        axis, anchor = self.joint_frame_world(jname)
        v = sign * np.cross(axis, self.tip_world(key) - anchor)
        v /= np.linalg.norm(v)
        _, r = self.body_pose(self.tip_bodies[key])
        return r.T @ v
