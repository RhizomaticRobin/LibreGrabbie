"""Arm builders for the flex-hand grasp demo (flexpalm_arm_demo.py).

Each builder adds a fixed-base serial arm to an existing ``newton.ModelBuilder``
and returns an :class:`ArmSpec` describing how the demo should drive it. The demo
is otherwise arm-agnostic: the hand mount (a mechanical mate onto the flange face),
the cube/pedestal, the tactile sensor and the reach->grasp->lift schedule all key
off the spec.

Two arms:
  * ``franka`` - Franka FR3 via ``add_urdf`` (the original demo arm).
  * ``dorna``  - the user's Dorna 2 (5-DOF) from ``~/Documents/dorna 2 robot arm.usda``,
                 with hollow-6xxx-aluminum mass/inertia tuned to the real 5.5 kg arm
                 weight, and the USD articulation re-rooted at the base (see below).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import warp as wp
import newton
import newton.utils
from newton._src.geometry.inertia import compute_inertia_mesh


# Universal grasp pocket in the hand's rigid rest frame (flange o mount), derived from
# the working Franka grasp: pocket = inv(flange_ready_franka o mount_franka) . cube_franka.
# Placing the cube at (flange_ready o mount) . GRASP_POCKET reproduces the SAME
# hand-relative grasp on any arm, absorbing the per-flange mount azimuth difference.
GRASP_POCKET = (0.03545, 0.05977, 0.06315)


@dataclass
class ArmSpec:
    name: str
    flange: int          # body index the hand mounts on (the tool flange)
    base: int            # world-fixed root body
    arm_dofs: list       # joint_q / joint_target_pos dof indices, in j0..jN order
    init_q: np.ndarray   # ready pose (flange pointing down), len == len(arm_dofs)
    shape_lo: int        # [lo, hi) range of arm collider shapes (for collision filtering)
    shape_hi: int
    lift_dof: int        # dof to raise/lower the flange for approach & lift
    lift_sign: float     # sign so (+arm_lift * lift_sign) RAISES the flange
    lift_amt: float      # approach-clearance magnitude [rad] (small -> near-vertical, clean descent)
    lift_raise: float    # final lift magnitude [rad] (larger; cube is enclosed so tilt is ok)
    cube_pos: tuple      # default graspable-cube position [m] (the digit convergence)
    cube_half: float     # default cube half-extent [m]
    cube_density: float
    cam_off: tuple       # default GL camera offset from the cube
    seat_dz: float       # extra mount seat offset along the flange tool axis [m]
    mrz: float           # default palm spin about the wrist axis [deg]
    ped_hw: float        # pedestal top half-width [m] (narrow lets fingers cage under the cube)


# ----------------------------------------------------------------------------
# Franka FR3
# ----------------------------------------------------------------------------
_FRANKA_INIT_Q = [-3.68e-3, 2.39e-2, 3.68e-3, -2.368, -1.29e-4, 2.392, 0.7855]


def build_franka(builder) -> ArmSpec:
    urdf = str(newton.utils.download_asset("franka_emika_panda") / "urdf" / "fr3.urdf")
    lo = builder.shape_count
    builder.add_urdf(urdf, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                     floating=False, enable_self_collisions=False)
    hi = builder.shape_count
    for i in range(7):
        builder.joint_q[i] = _FRANKA_INIT_Q[i]
        builder.joint_target_pos[i] = _FRANKA_INIT_Q[i]
        builder.joint_target_ke[i] = 650.0          # panda_hydro gains (stable PD)
        builder.joint_target_kd[i] = 100.0
        builder.joint_target_mode[i] = int(newton.JointTargetMode.POSITION)
        builder.joint_effort_limit[i] = 80.0
        builder.joint_armature[i] = 0.1
    flange = next(i for i, lbl in enumerate(builder.body_label) if lbl.endswith("/fr3_link8"))
    base = next(i for i, lbl in enumerate(builder.body_label) if lbl.endswith("/fr3_link0"))
    return ArmSpec(
        name="franka", flange=flange, base=base, arm_dofs=list(range(7)),
        init_q=np.array(_FRANKA_INIT_Q, dtype=np.float32), shape_lo=lo, shape_hi=hi,
        lift_dof=1, lift_sign=-1.0, lift_amt=0.35, lift_raise=0.35,
        cube_pos=(0.40, 0.035, 0.19), cube_half=0.03, cube_density=150.0,
        cam_off=(0.38, -0.42, 0.26), seat_dz=0.0, mrz=0.0, ped_hw=0.0255,
    )


# ----------------------------------------------------------------------------
# Dorna 2  (5-DOF, hollow 6xxx aluminum, 500 mm reach)
# ----------------------------------------------------------------------------
# The Dorna 2 CAD conversion is NOT redistributed in this repo (third-party CAD);
# point DEXIGRAB_DORNA_USD at your own conversion to enable --arm dorna.
_DORNA_USD = os.environ.get("DEXIGRAB_DORNA_USD",
                            os.path.expanduser("~/Documents/dorna 2 robot arm.usda"))
# flange-down ready pose (j0..j4 [rad]); found by gridding the pitch dofs for toolZ=-Z
# with the flange reaching FORWARD (~0.39 m, clear of the base footprint) so the cube +
# pedestal stand free, and high enough (z~0.35) that the down-pointing hand clears the floor.
_DORNA_INIT_Q = np.array([0.0, 0.39, -1.16, -2.7, 0.0], dtype=np.float32)
_ALU_RHO = 2700.0          # 6xxx aluminum density [kg/m^3]
_DORNA_MASS = 5.5          # Dorna 2 Black/Blue arm weight [kg]
# manual joint travel (Black/Blue), total range per axis [rad]; applied symmetric
# about the ready pose (USD zero != manual zero, so we can't map the absolute frame;
# the scripted motion is tiny so this only needs to not clip the choreography).
_DORNA_RANGE = np.radians([355.0, 272.0, 284.0, 270.0, 720.0])

_DORNA_LINKS = ("link_5step", "link_4step", "link_3step", "link_2step", "link_1step", "basestep")


def _reroot_dorna(b, n_bodies_before, n_joints_before):
    """The Dorna USD's PhysicsRevolute chain imports rooted at the *flange*
    (link_5step welded to world, chain running down to basestep) -- inverted.
    Rewire the just-added articulation so it roots at ``basestep`` (world-fixed)
    with the chain basestep -> link_1 -> ... -> link_5 (flange), giving natural
    dof order dof0=j0(base yaw) .. dof4=j4(wrist). Returns (flange, base) body idx.

    Each reversed joint reuses the *opposite* original joint's anchor frames with
    parent/child swapped, so the link geometry is preserved exactly; only the
    kinematic root flips. Verified: dof k moves exactly its downstream links.
    """
    by = {nm: next(i for i, l in enumerate(b.body_label) if l.endswith("/" + nm)) for nm in _DORNA_LINKS}
    L = [by[nm] for nm in _DORNA_LINKS]   # L[0]=link_5(flange) .. L[5]=basestep
    J0 = n_joints_before                  # joint slots J0 (FIXED) .. J0+5 (revolutes)
    Xp = [wp.transform(*[float(x) for x in t]) for t in b.joint_X_p]
    Xc = [wp.transform(*[float(x) for x in t]) for t in b.joint_X_c]
    base_rest = wp.transform(*[float(x) for x in b.body_q[by["basestep"]]])
    # slot k (1..5) := reverse of original joint (6-k); reversed chain base->...->flange
    src = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
    pc = {1: (5, 4), 2: (4, 3), 3: (3, 2), 4: (2, 1), 5: (1, 0)}  # (parent,child) into L
    for k in range(1, 6):
        oj = J0 + src[k]
        b.joint_parent[J0 + k] = L[pc[k][0]]
        b.joint_child[J0 + k] = L[pc[k][1]]
        b.joint_X_p[J0 + k], b.joint_X_c[J0 + k] = Xc[oj], Xp[oj]   # swap anchors for reversal
    b.joint_parent[J0] = -1                     # FIXED slot -> world -> base
    b.joint_child[J0] = by["basestep"]
    b.joint_X_p[J0], b.joint_X_c[J0] = base_rest, wp.transform_identity()
    return by["link_5step"], by["basestep"]


def _apply_hollow_aluminum(b, bodies, target_mass):
    """Set mass/inertia of ``bodies`` from a hollow aluminum shell, with the wall
    thickness solved so the total equals ``target_mass``. Mesh inertia per body is
    integrated over its collider mesh in the body frame (shell wedge of +/- t)."""
    # gather one representative mesh per body (the highest-vertex collider/visual)
    per_body = {}
    for s in range(b.shape_count):
        src = b.shape_source[s]
        bod = b.shape_body[s]
        if src is None or bod not in bodies or not hasattr(src, "vertices"):
            continue
        nv = len(src.vertices)
        if bod not in per_body or nv > per_body[bod][0]:
            per_body[bod] = (nv, s)

    def body_local(s):
        src = b.shape_source[s]
        sc = np.asarray(b.shape_scale[s][:3], dtype=np.float64)
        V = np.asarray(src.vertices, dtype=np.float64) * sc
        tf = b.shape_transform[s]
        p = np.array([float(x) for x in tf[:3]]); q = wp.quat(*[float(x) for x in tf[3:7]])
        Vb = np.array([p + np.array([float(z) for z in wp.quat_rotate(q, wp.vec3(*v))]) for v in V])
        return Vb, np.asarray(src.indices, dtype=np.int64)

    # solve thickness: shell mass ~ linear in t for thin walls
    t0 = 0.002
    m0 = 0.0
    cache = {}
    for bod, (_, s) in per_body.items():
        V, I = body_local(s); cache[bod] = (V, I)
        mass, _, _, _ = compute_inertia_mesh(_ALU_RHO, V, I, is_solid=False, thickness=t0)
        m0 += mass
    t = t0 * target_mass / m0
    total = 0.0
    for bod, (V, I) in cache.items():
        mass, com, Imat, _ = compute_inertia_mesh(_ALU_RHO, V, I, is_solid=False, thickness=t)
        Inp = np.array([[float(Imat[r, c]) for c in range(3)] for r in range(3)], dtype=np.float64)
        b.body_mass[bod] = float(mass)
        b.body_inv_mass[bod] = 1.0 / float(mass)
        b.body_com[bod] = wp.vec3(*[float(x) for x in com])
        b.body_inertia[bod] = wp.mat33(*Inp.flatten().tolist())
        b.body_inv_inertia[bod] = wp.mat33(*np.linalg.inv(Inp).flatten().tolist())
        total += float(mass)
    return t, total


def build_dorna(builder, verbose=False) -> ArmSpec:
    if not os.path.exists(_DORNA_USD):
        raise FileNotFoundError(f"Dorna USD not found: {_DORNA_USD}")
    nb, nj = builder.body_count, len(builder.joint_type)
    lo = builder.shape_count
    # skip_mesh_approximation: keep the FULL-resolution CAD meshes (~8k verts/link) instead of
    # the default 64-vertex convex-hull colliders, so the arm renders smooth. The arm is a
    # filtered kinematic positioner (its collisions vs hand/cube/ground are disabled below and
    # self-collisions are off), so the non-convex meshes are render-only -- no collision cost.
    builder.add_usd(_DORNA_USD, floating=False, verbose=False,
                    skip_mesh_approximation=True, enable_self_collisions=False)
    hi = builder.shape_count
    flange, base = _reroot_dorna(builder, nb, nj)
    arm_bodies = set(range(nb, builder.body_count))
    t, total = _apply_hollow_aluminum(builder, arm_bodies, _DORNA_MASS)
    if verbose:
        print(f"[dorna] hollow-aluminum wall t={t*1000:.1f}mm -> total arm mass {total:.2f} kg", flush=True)
    # arm is added right after the ground plane -> its 5 revolute dofs are 0..4,
    # in natural order (dof0=j0 base .. dof4=j4 wrist) after the re-root.
    arm_dofs = list(range(5))
    for d, i in enumerate(arm_dofs):
        builder.joint_q[i] = float(_DORNA_INIT_Q[d])
        builder.joint_target_pos[i] = float(_DORNA_INIT_Q[d])
        builder.joint_target_ke[i] = 450.0
        builder.joint_target_kd[i] = 45.0
        builder.joint_target_mode[i] = int(newton.JointTargetMode.POSITION)
        builder.joint_effort_limit[i] = 60.0
        builder.joint_armature[i] = 0.05
        builder.joint_limit_lower[i] = float(_DORNA_INIT_Q[d] - 0.5 * _DORNA_RANGE[d])
        builder.joint_limit_upper[i] = float(_DORNA_INIT_Q[d] + 0.5 * _DORNA_RANGE[d])
    return ArmSpec(
        name="dorna", flange=flange, base=base, arm_dofs=arm_dofs,
        init_q=_DORNA_INIT_Q.copy(), shape_lo=lo, shape_hi=hi,
        lift_dof=2, lift_sign=-1.0, lift_amt=0.15, lift_raise=0.5,
        cube_pos=(-0.02, -0.492, 0.221), cube_half=0.03, cube_density=150.0,
        cam_off=(0.42, 0.30, 0.26), seat_dz=0.0, mrz=0.0, ped_hw=0.0255,
    )


def build_arm(builder, name, verbose=False) -> ArmSpec:
    if name == "franka":
        return build_franka(builder)
    if name == "dorna":
        return build_dorna(builder, verbose=verbose)
    raise ValueError(f"unknown arm '{name}' (expected 'franka' or 'dorna')")


def flange_world_at_ready(name):
    """Build the arm ALONE in a throwaway model and return (flange_world_transform,
    spec) at the ready (init_q) pose. Used to place the graspable cube in the hand's
    rigid rest frame (flange o mount) so the grasp geometry transfers between arms."""
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    spec = build_arm(b, name)
    m = b.finalize()
    st = m.state()
    newton.eval_fk(m, m.joint_q, m.joint_qd, st)
    fl = st.body_q.numpy()[spec.flange]
    return wp.transform(wp.vec3(*[float(x) for x in fl[:3]]), wp.quat(*[float(x) for x in fl[3:7]])), spec

