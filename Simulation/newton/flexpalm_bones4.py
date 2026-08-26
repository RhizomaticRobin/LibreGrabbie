# Semi-flexible robot palm (Newton + MuJoCo): the 4 big palm bones + the 4 small
# Palm_bone1 sub-bones, taken as individual USD mesh components, built as clean
# `add_link` bodies at their own centroids and wired together with the restricted-
# spherical-joint "squares" structure proven in example_joint_squares.py.
#
#   * Big bones are long rails along X stacked across the palm in Y. Each is held
#     at BOTH ends: a +X connector "rung" at the finger end, and the real small
#     wrist bones at the -X end. So a big bone cannot pivot/shift on one point.
#   * The 4 small wrist bones (~20 mm cubes) are NOT long rails, so they get a
#     single restricted joint per neighbour (no both-ends needed) following the
#     user's hand-picked connectivity (EDGES). They ARE the -X connective tissue.
#   * The structure is a spanning TREE rooted at the frozen pinky bone (a wrist
#     "caterpillar": pinky-S03-S04-{ring,S00}, S00-{mid,S01}, S01-ptr) plus the
#     three +X big-bone rungs as equality-CONNECT loop closures. Closed loops ->
#     SolverMuJoCo. Cone limit + centering spring are NATIVE D6 (implicit/stable).
#
# All joint anchors are placed at the real world contact point and expressed in
# BOTH bodies' local frames so they coincide at rest -> no cramming, no shift.
#
#   python flexpalm_bones4.py --check       # headless numeric verification
#   python flexpalm_bones4.py --viewer gl   # GL window for visual audit
import os
import sys
import math
import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
import newton_compat  # noqa: F401  (1.2->1.4 equality API shim)

from pxr import Usd, UsdGeom, UsdPhysics, Gf

# the real pose battery (backend-agnostic; imports only numpy). Prefer the vendored copy
# sitting next to this file (self-contained); fall back to the IsaacLab demos copy.
_LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _LOCAL_DIR not in sys.path:
    sys.path.insert(0, _LOCAL_DIR)
if not os.path.exists(os.path.join(_LOCAL_DIR, "robot_hand_poses.py")):
    _POSES_DIR = os.path.join(os.environ.get("RESEARCH_ROOT", os.path.expanduser("~/research")),
                              "robotics/sim/Robotics/IsaacLab/scripts/demos")
    if _POSES_DIR not in sys.path:
        sys.path.insert(0, _POSES_DIR)
import robot_hand_poses as rhp   # noqa: E402  (THUMB/CHAINS/JointLookup/build_poses/POSES)

# Assets ship in ../assets of the DexiGrab repo; override with DEXIGRAB_ASSETS.
_ASSETS = os.environ.get("DEXIGRAB_ASSETS") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "assets")
USD = os.path.join(_ASSETS, "robot_hand_flexpalm.usda")
# The working copy's palmStrips variant is "deformable", which drops the palm
# cover strips (Palm1/Palm2_2/Palm3_2/Palm_pinky/palm_bone_flex) and loses their
# scene placement. The untouched ORIGINAL has palmStrips="rigid" with the strips
# placed in world space; we read their geometry from there (read-only) and weld
# each over its bone as the outer cover.
ORIG_USD = os.path.join(_ASSETS, "robot_hand_simready.usda")
PALM_ROOT = "/Robotic_Hand_V5_simulacra/Palm_rigid"
DEFORM_ROOT = PALM_ROOT + "/Palm_Deformable"

# outer cover strips (in ORIG_USD) -> the bone each one lays over (by Y position)
PALM_STRIPS = {
    "Palm_pinky": "pinky", "Palm3_2": "ring", "Palm2_2": "mid", "Palm1": "ptr",
    "palm_bone_flex": "S00", "palm_bone_flex_01": "S04",
}

# Shell2: the big outer bottom-chassis shell, dropped from the original CAD export
# and recovered from Onshape. It lives in a separate re-export; its geometry refs
# don't resolve in plain USD, so we compose it as (base Shell2 Xform transform) x
# (part-file local mesh) -- validated to reproduce Shell1's known placement exactly.
SHELL2_DIR = os.path.join(_ASSETS, "Robotic_Hand_V5_simulacra_with_missing_shell.zip")
SHELL2_BASE = SHELL2_DIR + "/Robotic_Hand_V5_simulacra_base.usd"
SHELL2_PART = SHELL2_DIR + "/parts/Shell2_JFD.usd"
SHELL2_XFORM = "/World/Robotic_Hand_V5_simulacra/Palm_rigid/Shell2"

# ---- restricted-joint constants (cone geometry from example_joint_squares.py;
# spring/damping scaled ~100x softer for these small, light bones, tuned live in
# the viewer to a soft, semi-flexible feel) ----
THETA_MAX = math.radians(35.0)   # per-cone angular limit
K_SPRING = 2.0                   # centering spring (biases joint back to 0)
DAMPING = 0.25                   # joint drive damping
LIMIT_KE = 5.0e3                 # stiffness of the +/-THETA_MAX wall (firm)
LIMIT_KD = 2.0e0
# kamino bushings: SolverKamino rejects equality constraints AND gimbal-config D6 (3 angular/0
# linear). So under `kamino`, every closed loop becomes a stiff 6-DOF D6 (3 linear + 3 angular).
# Kamino integrates these PD springs implicitly (backward-Euler) -> unconditionally stable for
# any stiffness/dt, which is the whole point of the port. Phase-3-tunable.
BUSHING_LIN_KE = 2.0e4           # linear pin stiffness [N/m] (holds a loop-closure point together)
BUSHING_LIN_KD = 1.0e2           # linear pin damping
BUSHING_ANG_KE = 5.0e2           # shell-hinge angular spring [N*m/rad] (scaled by torquescale)
BUSHING_ANG_KD = 5.0e0
# Shell1<->Shell2 ball-joint center (build-frame world, lift=0), CALIBRATED by matching
# the prior full-CoACD-collision shell motion (calibrate_shell_ball.py --optimize; converged
# Nelder-Mead, shell-trajectory match 6.6mm vs the raw geometric sphere fit's 54mm).
# Shell1<->Shell2 weld, CALIBRATED (5D, batched: SHELL_REG=1.0 calibrate_shell_ball.py --batched
# --pop 4096, ~41k candidates). The pivot is PINNED to the shells' fitted mating-sphere center (where
# they physically articulate) via a geometric regularizer, then torquescale + solref are fit to the
# prior full-CoACD-collision shell trajectory: pos 15mm / rot 13.5deg vs the reference's ~38mm drift.
# (The unconstrained dynamic optimum reached 5.9mm/5.5deg but floated the pivot ~85mm off the surface;
# we chose the physically-placed pivot -- still far better than a free ball's 27deg rotation gap.)
SHELL_BALL_CENTER = (-0.1206, -0.0298, -0.0082)
SHELL_WELD_TORQUESCALE = 0.500   # rotational-residual weight of the weld
SHELL_WELD_SOLREF_TC = 0.0993    # weld constraint compliance time-constant [s]

# Big bones: long rails, increasing-Y order (pinky -> pointer). pinky is frozen.
BIG = {
    "pinky": "Base_Bone_12_V02_pinky",
    "ring":  "Base_Bone_12_V02_ringfing",
    "mid":   "Base_Bone_12_V02_middlefinger",
    "ptr":   "Base_Bone_12_V02_pointerfinger_and_thumb_attachment",
}
# Small wrist bones (Palm_bone1/*). _02 is degenerate (17 verts, 2 mm) -> skipped.
SMALL = {
    "S03": "Palm_bone1/Palm_bone1_03",
    "S04": "Palm_bone1/Palm_bone1_04",
    "S00": "Palm_bone1/Palm_bone1",
    "S01": "Palm_bone1/Palm_bone1_01",
}
BIG_ORDER = ["pinky", "ring", "mid", "ptr"]

# Spanning TREE (parent, child) rooted at the frozen pinky -- the user's EDGES
# graph laid out parents-before-children for MuJoCo. Big<->small edges anchor at
# the big bone's -X tip (firm wrist hold); small<->small edges at the midpoint.
TREE_EDGES = [
    ("pinky", "S03"),   # frozen pinky -> its wrist bone
    ("S03", "S04"),
    ("S04", "ring"),
    ("S04", "S00"),
    ("S00", "mid"),
    ("S00", "S01"),
    ("S01", "ptr"),
]
# +X big-bone rungs (finger end). Each is a connector body joined to bone_i as a
# tree leaf and CONNECT-closed to bone_{i+1} -> holds the big bones' finger ends.
PLUS_PAIRS = [("pinky", "ring"), ("ring", "mid"), ("mid", "ptr")]

# ---- Fingers: each chain is its real USD PhysicsRevoluteJoint sequence, in
# chain order (root first). The root joint's USD body0 is the monolithic
# Palm_rigid; we re-anchor it onto the mapped flexible big bone so the finger
# RIDES that bone. All joint frames/limits/drive are read from the USD at build
# time (transcription, not guesswork). Drive stiffness/damping are authored
# per-degree (USD convention) -> converted to per-radian for Newton.
DEG2RAD = math.pi / 180.0
FINGER_CHAINS = {
    "thumb":   {"bone": "ptr",   "joints": ["Revolute_5", "Revolute_1", "Revolute_2",
                                            "Revolute_3", "Revolute_4", "Revolute_6"]},
    "pinky":   {"bone": "pinky", "joints": ["Revolute_19", "Revolute_8", "Revolute_9", "Revolute_10"]},
    "ring":    {"bone": "ring",  "joints": ["Revolute_20", "Revolute_7", "Revolute_11", "Revolute_12"]},
    "middle":  {"bone": "mid",   "joints": ["Revolute_21", "Revolute_18", "Revolute_13", "Revolute_14"]},
    "pointer": {"bone": "ptr",   "joints": ["Revolute_22", "Revolute_15", "Revolute_16", "Revolute_17"]},
}
HAND_ROOT = "/Robotic_Hand_V5_simulacra"

# Two SEPARATE rigid shell groups, one per side of the hand:
#  - Shell1 group (Shell1 + the two driver-side covers + the back plate) -> all
#    welded to the PINKY-side bone, so they are mutually rigid (fixed to each
#    other) on the non-thumb side.
#  - Shell2 (below) -> the thumb-motor side (ptr bone). Kept separate.
SHELL2_ANCHOR_BONE = "ptr"           # thumb-motor side
# Grouped by the part's actual side (Y position): the thumb-side cover joins Shell2
# on ptr (rigid together); the pinky-side parts + back plate go on pinky.
SHELL_BONE = {
    "driver_side_palm_cover3": "ptr",     # thumb/pointer side (y +0.018), with Shell2
    "Shell1": "pinky",                    # main back shell
    "cover_back": "pinky",                # back plate
    "driver_side_palm_cover2": "pinky",   # pinky side (y -0.075)
}


def extract_mesh(stage, xcache, prim_path):
    """Return (verts_world [N,3] f32, tri_indices [M*3] i32) for a Mesh prim,
    triangulating polygon faces. Resolves USD references/overs automatically."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise KeyError(f"prim not found: {prim_path}")
    # the prim may be an Xform wrapping a Mesh of the same name; find the Mesh
    mesh_prim = prim if prim.GetTypeName() == "Mesh" else None
    if mesh_prim is None:
        for p in Usd.PrimRange(prim):
            if p.GetTypeName() == "Mesh":
                mesh_prim = p
                break
    if mesh_prim is None:
        raise KeyError(f"no Mesh under {prim_path}")

    mesh = UsdGeom.Mesh(mesh_prim)
    pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)

    M = np.array(xcache.GetLocalToWorldTransform(mesh_prim), dtype=np.float64).reshape(4, 4)
    homog = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    world = (homog @ M)[:, :3]

    tris = []
    o = 0
    for c in counts:
        c = int(c)
        for k in range(1, c - 1):
            tris.append((idx[o], idx[o + k], idx[o + k + 1]))
        o += c
    return world.astype(np.float32), np.asarray(tris, dtype=np.int32).reshape(-1)


@wp.kernel
def _curl_kernel(t: wp.float32, omega: wp.float32,
                 dof_idx: wp.array(dtype=wp.int32),
                 curl_target: wp.array(dtype=wp.float32),
                 phase: wp.array(dtype=wp.float32),
                 out_target: wp.array(dtype=wp.float32)):
    k = wp.tid()
    # smooth open(0) -> fist(1) -> open(0) cycle, staggered per finger by phase
    frac = 0.5 * (1.0 - wp.cos(omega * t - phase[k]))
    out_target[dof_idx[k]] = frac * curl_target[k]


def gf_to_pos_quat(M):
    """Gf.Matrix4d -> (np[3] pos, wp.quat xyzw)."""
    t = M.ExtractTranslation()
    q = M.ExtractRotationQuat()          # Gf: real=w, imaginary=(x,y,z)
    im = q.GetImaginary()
    return (np.array([t[0], t[1], t[2]], dtype=np.float64),
            wp.quat(float(im[0]), float(im[1]), float(im[2]), float(q.GetReal())))


def gf_make(pos, quat):
    """(pos, Gf.Quat*) -> Gf.Matrix4d (rotation then translation, row convention)."""
    M = Gf.Matrix4d(1.0)
    if quat is not None:
        M.SetRotateOnly(Gf.Quatd(quat.GetReal(), Gf.Vec3d(*[float(x) for x in quat.GetImaginary()])))
    M.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    return M


_COACD_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".coacd_cache")


def _sphere_ls(M):
    """Least-squares sphere fit -> (center, radius)."""
    M = np.asarray(M, np.float64)
    A = np.c_[2 * M, np.ones(len(M))]; b = (M ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    return c, float(np.sqrt(max(sol[3] + c @ c, 1e-12)))


def _fit_ball_centers(P1, P2, thr=0.004):
    """Fit the ball-and-socket center of rotation from two shell point clouds. Returns
    (combined_center, shell1_center, shell2_center, r1, r2): each shell's OWN spherical
    mating surface center (fit to its mating-region verts), plus the combined fit. The
    two per-shell centers should coincide for a true ball-and-socket; the gap reveals
    placement error."""
    P1 = np.asarray(P1, np.float32); P2 = np.asarray(P2, np.float32)

    def nn(A, B):
        out = np.empty(len(A), np.float32)
        for i in range(0, len(A), 256):
            blk = A[i:i + 256]
            out[i:i + 256] = np.sqrt(((blk[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
        return out
    m1 = P1[nn(P1, P2) < thr]; m2 = P2[nn(P2, P1) < thr]
    if len(m1) < 12 or len(m2) < 12:
        m1, m2 = P1, P2
    c1, r1 = _sphere_ls(m1); c2, r2 = _sphere_ls(m2)
    cc, _ = _sphere_ls(np.vstack([m1, m2]))
    return cc, c1, c2, r1, r2


def convex_decompose(verts, tris, threshold=0.05):
    """Collision-aware convex decomposition (CoACD) of a (possibly concave) mesh
    into a list of convex (verts[f32], tri_indices[i32]) pieces. Cached to disk by
    geometry hash so relaunches don't re-run CoACD. Used to give the shells real
    wall-vs-wall collision under MuJoCo (which only collides convex geoms)."""
    import hashlib
    import pickle
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(tris, dtype=np.int32).reshape(-1, 3)
    key = hashlib.md5(v.tobytes() + f.tobytes() + str(threshold).encode()).hexdigest()[:16]
    os.makedirs(_COACD_CACHE, exist_ok=True)
    cache = os.path.join(_COACD_CACHE, f"{key}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    import io
    import contextlib
    import coacd
    try:
        coacd.set_log_level("error")
    except Exception:
        pass
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        raw = coacd.run_coacd(coacd.Mesh(v, f), threshold=threshold)
    parts = [(np.asarray(pv, dtype=np.float32), np.asarray(pf, dtype=np.int32).reshape(-1)) for pv, pf in raw]
    with open(cache, "wb") as fh:
        pickle.dump(parts, fh)
    return parts


def extract_mesh_local(stage, xcache, xform_path):
    """Mesh verts expressed in the RigidBody Xform's own frame (so USD joint
    localPos/localRot, which are relative to that Xform, apply directly), plus
    the Xform world transform (pos, quat) to place the body."""
    xform_prim = stage.GetPrimAtPath(xform_path)
    Mx = xcache.GetLocalToWorldTransform(xform_prim)         # Xform -> world
    world, tris = extract_mesh(stage, xcache, xform_path)    # verts in world
    Minv = np.array(Mx.GetInverse(), dtype=np.float64).reshape(4, 4)
    homog = np.concatenate([world.astype(np.float64), np.ones((len(world), 1))], axis=1)
    local = (homog @ Minv)[:, :3]
    pos, quat = gf_to_pos_quat(Mx)
    return local.astype(np.float32), tris, pos, quat


class Example:
    def __init__(self, viewer, args, ext_builder=None, attach_body=-1,
                 attach_xform=None, finalize=True):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args
        # embed support: attach the hand to a parent body (e.g. an arm flange)
        # instead of freezing it to the world, and optionally skip finalize/solver
        # so a caller can add more (an arm, an object) to the same builder.
        self.attach_body = attach_body
        self.attach_xform = attach_xform
        self._do_finalize = finalize
        self.k_spring = float(getattr(args, "kspring", K_SPRING))
        self.damping = float(getattr(args, "damping", DAMPING))
        self.theta_max = math.radians(float(getattr(args, "theta", math.degrees(THETA_MAX))))
        # lift the whole hand +Z so it clears the ground plane (shell reaches z=-0.069)
        self.lift = np.array([0.0, 0.0, float(getattr(args, "lift", 0.15))], dtype=np.float64)
        # collision-hull fidelity: newton.Mesh defaults to a coarse 64-vertex convex
        # hull; raise it so the collision geometry tracks the real meshes.
        self.hullverts = int(getattr(args, "hullverts", 256))
        # simple_shells: collide each back/bottom shell (Shell1, Shell2, the driver-side
        # cover) as ONE convex hull instead of a CoACD decomposition (~70 convex parts ->
        # ~5). The shells are the outer chassis, away from the palm-cup grasp surface, so
        # the cube never enters them -> a solid hull loses no task-relevant fidelity, and
        # it also smooths the rough recovered-mesh Shell2. Fingers + palm bones keep full
        # collision (grasp + tactile), and the visual meshes are unchanged. This slashes
        # the cube<->hand contact-feature count so the batched mujoco-warp solve stays
        # stable at high world counts. Mechanism (D6 palm springs, closed loops) untouched.
        self.simple_shells = bool(getattr(args, "simple_shells", False))
        # shells_jointed (preferred over simple_shells): keep the back/bottom shells as
        # FULL-resolution visual meshes (topology preserved -- no convex-blob hack) but
        # take them OUT of collision entirely, and connect Shell1->Shell2 with an explicit
        # spring-centered spherical (D6) joint -- the chassis hinge the design implied. The
        # shells are the outer chassis (never touch the cube; already self-filtered), so
        # collision was only ever a liability; jointing them makes the connection explicit
        # and lets the contact solver ignore them -> stable at high world counts, no
        # topology loss. Supersedes simple_shells; implies non-colliding shells.
        self.shells_jointed = bool(getattr(args, "shells_jointed", False))
        if self.shells_jointed:
            self.simple_shells = False
        # defer_shell_weld: build the shells but DON'T add the Shell1<->Shell2 weld here;
        # instead expose relpose/anchor/body indices so a caller can add per-world welds
        # AFTER replicate (used to batch-evaluate many weld params in parallel worlds).
        self.defer_shell_weld = bool(getattr(args, "defer_shell_weld", False))
        # kamino: build the closed loops as stiff 6-DOF D6 bushings (no equality constraints,
        # no gimbal-D6) so the model runs under SolverKamino. Default OFF -> exact MuJoCo path.
        # NOTE: blocked at the Newton->kamino bridge (can't represent closed loops); kept inert.
        self.kamino = bool(getattr(args, "kamino", False))
        # soft_loops: skip the equality loop closures (CONNECT + WELD) entirely; the env closes
        # them each substep with explicit penalty-spring FORCES on body_f (no equality SOLVE).
        # Tests "is the equality solve the NaN source?" while staying on SolverMuJoCo. The anchor
        # data the env needs is already exposed via self.loop_specs + self.shell_weld_data.
        self.soft_loops = bool(getattr(args, "soft_loops", False))
        self._shell1_world = None

        # ---- extract all 8 bone meshes (world space) ----
        stage = Usd.Stage.Open(USD)
        xc = UsdGeom.XformCache()
        self.verts, self.tris, self.cen, self.xmin, self.xmax = {}, {}, {}, {}, {}
        for key, name in {**BIG, **SMALL}.items():
            path = f"{PALM_ROOT}/{name}" if "/" in name else f"{PALM_ROOT}/{name}/{name}"
            v, t = extract_mesh(stage, xc, path)
            self.verts[key] = v
            self.tris[key] = t
            self.cen[key] = v.mean(axis=0)
            self.xmin[key] = float(v[:, 0].min())
            self.xmax[key] = float(v[:, 0].max())

        if ext_builder is None:
            builder = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
            builder.add_ground_plane()
        else:
            builder = ext_builder      # caller owns registration + ground/arm
        self.builder = builder
        Cfg = newton.ModelBuilder.JointDofConfig

        # Rigid parts: collidable, but all in the SAME negative collision group so
        # the overlapping/welded hand parts never self-collide (they'd explode),
        # while still colliding with the ground and external objects. (Newton:
        # same negative group -> mutually filtered; collides with everything else.)
        HAND_GROUP = -1
        bone_cfg = newton.ModelBuilder.ShapeConfig()
        bone_cfg.density = 1000.0
        bone_cfg.has_shape_collision = True
        bone_cfg.collision_group = HAND_GROUP
        # Connector "rung" proxies: keep for the kinematics (mass) but make them
        # INVISIBLE and non-colliding so they don't clutter the view.
        link_cfg = newton.ModelBuilder.ShapeConfig()
        link_cfg.density = 150.0          # light connector proxies (penalty springs are stable here
        link_cfg.has_shape_collision = False   # once the body_f wrench convention is correct)
        link_cfg.is_visible = False

        COLOR = {"pinky": (0.85, 0.30, 0.30), "ring": (0.30, 0.75, 0.45),
                 "mid": (0.35, 0.55, 0.90), "ptr": (0.90, 0.75, 0.25),
                 "S03": (0.70, 0.55, 0.85), "S04": (0.55, 0.70, 0.85),
                 "S00": (0.85, 0.60, 0.70), "S01": (0.75, 0.75, 0.55)}

        # ---- bone bodies: clean add_link at each centroid, real mesh attached ----
        self.body = {}        # key -> body index
        self.origin = {}      # body index -> world origin (np[3]) of the body frame
        self.coll_shapes = [] # collidable hand shape indices (for self-collision filtering)
        self.bone_shape = {}  # bone key -> collision shape index
        self.fbody_shape = {} # finger body name -> collision shape index
        self.finger_shapes = {}  # finger name -> [collision shape indices]
        for key in {**BIG, **SMALL}:
            c = self.cen[key]
            b = builder.add_link(
                xform=wp.transform(p=wp.vec3(*(c + self.lift)), q=wp.quat_identity()),
                label=f"bone_{key}")
            m = newton.Mesh(self.verts[key] - c.astype(np.float32), self.tris[key],
                            compute_inertia=True, color=COLOR[key], maxhullvert=self.hullverts)
            sidx = builder.add_shape_mesh(b, mesh=m, cfg=bone_cfg)
            self.coll_shapes.append(sidx)
            self.bone_shape[key] = sidx
            self.body[key] = b
            self.origin[b] = np.asarray(c, dtype=np.float64)

        def lin_pin(a):    # stiff linear spring axis (the bushing "pin")
            return Cfg(axis=a, target_pos=0.0, target_ke=BUSHING_LIN_KE, target_kd=BUSHING_LIN_KD)

        def rball(parent_b, child_b, anchor_w, label):
            """restricted spherical joint at world point anchor_w, expressed in
            both bodies' (identity-rotation) local frames so it coincides at rest.
            Under `kamino`: also add 3 stiff linear axes -> a 6-DOF D6 bushing (avoids the
            rejected gimbal config); the linear springs pin the joint point, the angular axes
            keep the cone+centering spring. Under MuJoCo: angular-only (the original ball)."""
            def ax(a):
                return Cfg(axis=a, limit_lower=-self.theta_max, limit_upper=self.theta_max,
                           limit_ke=LIMIT_KE, limit_kd=LIMIT_KD,
                           target_pos=0.0, target_ke=self.k_spring, target_kd=self.damping)
            p_local = anchor_w - self.origin[parent_b]
            c_local = anchor_w - self.origin[child_b]
            lin_axes = ([lin_pin(wp.vec3(1.0, 0.0, 0.0)), lin_pin(wp.vec3(0.0, 1.0, 0.0)),
                         lin_pin(wp.vec3(0.0, 0.0, 1.0))] if self.kamino else None)
            return builder.add_joint_d6(
                parent=parent_b, child=child_b,
                linear_axes=lin_axes,
                angular_axes=[ax(wp.vec3(1.0, 0.0, 0.0)), ax(wp.vec3(0.0, 1.0, 0.0)),
                              ax(wp.vec3(0.0, 0.0, 1.0))],
                parent_xform=wp.transform(p=wp.vec3(*p_local), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(*c_local), q=wp.quat_identity()),
                label=label)

        def edge_anchor(a, b):
            """world anchor for a tree edge: big<->small uses the big bone's -X
            tip (firm wrist hold); small<->small uses the centroid midpoint."""
            ab = [k for k in (a, b) if k in BIG]
            if ab:                       # big involved -> anchor at big -X tip
                g = ab[0]
                c = self.cen[g]
                return np.array([self.xmin[g], float(c[1]), float(c[2])], dtype=np.float64)
            return 0.5 * (self.cen[a].astype(np.float64) + self.cen[b].astype(np.float64))

        # ---- tree joints (parents before children) ----
        # Root the hand: either frozen to the world (standalone) or fixed to a
        # parent body such as an arm flange (embed). The child anchor is the pinky
        # bone's body origin (its centroid). attach_xform places that origin in the
        # parent frame; default keeps the standalone world pose (centroid + lift).
        cpin = self.cen["pinky"]
        if self.attach_body == -1:
            root_xform = wp.transform(p=wp.vec3(*(cpin + self.lift)), q=wp.quat_identity())
            root_label = "freeze_pinky"
        else:
            root_xform = self.attach_xform if self.attach_xform is not None else \
                wp.transform(p=wp.vec3(*(cpin + self.lift)), q=wp.quat_identity())
            root_label = "mount_pinky"
        joints = [builder.add_joint_fixed(
            parent=self.attach_body, child=self.body["pinky"],
            parent_xform=root_xform,
            child_xform=wp.transform(q=wp.quat_identity()), label=root_label)]

        self.corners = []     # (parent_body, child_body) per restricted joint, for bend
        for (a, b) in TREE_EDGES:
            joints.append(rball(self.body[a], self.body[b], edge_anchor(a, b), f"jt_{a}_{b}"))
            self.corners.append((self.body[a], self.body[b]))

        # +X connector rungs: tree leaf bone_a -> connector, loop-close to bone_b
        self.plus_link = {}
        for (a, b) in PLUS_PAIRS:
            ta = np.array([self.xmax[a], float(self.cen[a][1]), float(self.cen[a][2])])
            tb = np.array([self.xmax[b], float(self.cen[b][1]), float(self.cen[b][2])])
            p = 0.5 * (ta + tb)
            lk = builder.add_link(xform=wp.transform(p=wp.vec3(*(p + self.lift)), q=wp.quat_identity()),
                                  label=f"plus_{a}_{b}")
            builder.add_shape_box(lk, hx=0.004, hy=max(0.5 * float(np.linalg.norm(tb - ta)), 0.004),
                                  hz=0.004, cfg=link_cfg)
            self.body[f"plus_{a}_{b}"] = lk
            self.origin[lk] = p
            self.plus_link[(a, b)] = (lk, ta, tb)
            joints.append(rball(self.body[a], lk, ta, f"jt_{a}_plus_{b}"))   # leaf
            self.corners.append((self.body[a], lk))

        # ===== Fingers: transcribe the real USD revolute chains; re-anchor each
        # finger ROOT from the monolithic Palm_rigid onto its flexible big bone =====
        Mpr = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(PALM_ROOT))   # Palm_rigid->world
        finger_cfg = newton.ModelBuilder.ShapeConfig()
        finger_cfg.density = 1000.0
        finger_cfg.has_shape_collision = True
        finger_cfg.collision_group = HAND_GROUP
        fke = float(getattr(args, "fingerke", 1.0))
        K_DEG2RAD = 180.0 / math.pi   # USD drive stiffness is per-degree -> per-radian

        self.finger_body = {}
        self.finger_local = {}        # finger body name -> local collision verts (for tactile pads)
        self.fingertips = []          # distal body indices (droop measurement)
        self.curl_specs = []          # (joint_idx, curl_target_rad, finger_phase) for flexion joints
        self.jname_to_jid = {}        # bare USD joint name (Revolute_N) -> Newton joint index
        FINGER_PHASE = {"index": 0.0, "middle": 0.45, "ring": 0.9, "pinky": 1.35,
                        "pointer": 0.0, "thumb": 1.8}
        AXmap = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}

        def gfq_wp(q):
            im = q.GetImaginary()
            return wp.quat(float(im[0]), float(im[1]), float(im[2]), float(q.GetReal()))

        def build_fbody(name):
            if name in self.finger_body:
                return self.finger_body[name]
            local, tris, pos, quat = extract_mesh_local(stage, xc, f"{HAND_ROOT}/{name}")
            b = builder.add_link(xform=wp.transform(p=wp.vec3(*(pos + self.lift)), q=quat),
                                 label=f"f_{name}")
            m = newton.Mesh(local, tris, compute_inertia=True, color=(0.72, 0.72, 0.75), maxhullvert=self.hullverts)
            sidx = builder.add_shape_mesh(b, mesh=m, cfg=finger_cfg)
            self.coll_shapes.append(sidx)
            self.fbody_shape[name] = sidx
            self.finger_body[name] = b
            self.finger_local[name] = local          # body-local verts for tactile pad specs
            self.origin[b] = np.asarray(pos, dtype=np.float64)
            return b

        for fname, info in FINGER_CHAINS.items():
            last_child = None
            for ji, jname in enumerate(info["joints"]):
                jp = stage.GetPrimAtPath(f"{HAND_ROOT}/{jname}")
                J = UsdPhysics.RevoluteJoint(jp)
                b0 = J.GetBody0Rel().GetTargets()[0].name
                b1 = J.GetBody1Rel().GetTargets()[0].name
                lp0 = J.GetLocalPos0Attr().Get(); lr0 = J.GetLocalRot0Attr().Get()
                lp1 = J.GetLocalPos1Attr().Get(); lr1 = J.GetLocalRot1Attr().Get()
                lo = math.radians(float(J.GetLowerLimitAttr().Get()))
                hi = math.radians(float(J.GetUpperLimitAttr().Get()))
                drv = UsdPhysics.DriveAPI.Get(jp, "angular")
                ke = float(drv.GetStiffnessAttr().Get()) * K_DEG2RAD * fke
                kd = float(drv.GetDampingAttr().Get()) * K_DEG2RAD * fke
                eff = float(drv.GetMaxForceAttr().Get())
                axis = wp.vec3(*AXmap[str(J.GetAxisAttr().Get())])

                child_b = build_fbody(b1)
                self.finger_shapes.setdefault(fname, []).append(self.fbody_shape[b1])
                child_xf = wp.transform(p=wp.vec3(float(lp1[0]), float(lp1[1]), float(lp1[2])),
                                        q=gfq_wp(lr1))
                if ji == 0:
                    # ROOT: parent is the mapped big bone (centroid frame, identity rot)
                    parent_b = self.body[info["bone"]]
                    Mjw = gf_make([lp0[0], lp0[1], lp0[2]], lr0) * Mpr   # joint frame -> world
                    wpos, wquat = gf_to_pos_quat(Mjw)
                    pxf = wp.transform(p=wp.vec3(*(wpos - self.origin[parent_b])), q=wquat)
                else:
                    parent_b = build_fbody(b0)
                    pxf = wp.transform(p=wp.vec3(float(lp0[0]), float(lp0[1]), float(lp0[2])),
                                       q=gfq_wp(lr0))

                jid = builder.add_joint_revolute(
                    parent=parent_b, child=child_b, parent_xform=pxf, child_xform=child_xf,
                    axis=axis, target_pos=0.0, target_ke=ke, target_kd=kd,
                    limit_lower=lo, limit_upper=hi, limit_ke=1.0e3, limit_kd=1.0e1,
                    effort_limit=eff, label=f"{fname}_{jname}")
                joints.append(jid)
                self.jname_to_jid[jname] = jid          # for the real pose battery
                # flexion joint = one-sided range (one end ~0); curl toward the far end.
                # two-sided joints (abduction roots, thumb opposition) are left at 0.
                if min(abs(lo), abs(hi)) < math.radians(5.0):
                    ct = hi if abs(hi) > abs(lo) else lo
                    self.curl_specs.append((jid, 0.85 * ct, FINGER_PHASE.get(fname, 0.0)))
                last_child = child_b
            self.fingertips.append(last_child)

        # ===== rigid back-of-hand shell + covers: each welded to its palm bone =====
        shell_cfg = newton.ModelBuilder.ShapeConfig()         # single convex (visual+collision)
        shell_cfg.density = 1000.0
        shell_cfg.has_shape_collision = True
        shell_cfg.collision_group = HAND_GROUP
        shell_vis_cfg = newton.ModelBuilder.ShapeConfig()     # visual-only full mesh (decomposed shells)
        shell_vis_cfg.density = 1000.0
        shell_vis_cfg.has_shape_collision = False
        shell_part_cfg = newton.ModelBuilder.ShapeConfig()    # invisible convex collision parts
        shell_part_cfg.density = 0.0
        shell_part_cfg.has_shape_collision = True
        shell_part_cfg.is_visible = False
        shell_part_cfg.collision_group = HAND_GROUP
        # Concave shells get a real convex DECOMPOSITION so their walls collide
        # (MuJoCo only collides convex geoms; a single hull is a solid blob).
        DECOMPOSE = {"Shell1", "driver_side_palm_cover3"}
        self.shell_body = {}
        self.shell_coll = {}         # shell name -> [collision shape indices]
        for nm, bone in SHELL_BONE.items():
            local, tris, pos, quat = extract_mesh_local(stage, xc, f"{PALM_ROOT}/{nm}")
            b = builder.add_link(xform=wp.transform(p=wp.vec3(*(pos + self.lift)), q=quat),
                                 label=f"shell_{nm}")
            if nm == "Shell1" and self.shells_jointed:
                self._shell1_body = b
                self._shell1_pos = np.asarray(pos, np.float64) + self.lift
                self._shell1_quat = quat
                R1 = np.array(wp.quat_to_matrix(wp.quat(*[float(x) for x in quat]))).reshape(3, 3)
                self._shell1_wv = (np.asarray(local, np.float64) @ R1.T) + self._shell1_pos
            if self.shells_jointed:
                # full-resolution VISUAL mesh only -- no collision (chassis, off the grasp
                # surface); topology preserved, zero contact-feature cost.
                vm = newton.Mesh(local, tris, compute_inertia=True, color=(0.45, 0.48, 0.55),
                                 maxhullvert=self.hullverts)
                builder.add_shape_mesh(b, mesh=vm, cfg=shell_vis_cfg)
                self.shell_coll[nm] = []
            elif nm in DECOMPOSE and not self.simple_shells:
                vm = newton.Mesh(local, tris, compute_inertia=True, color=(0.45, 0.48, 0.55),
                                 maxhullvert=self.hullverts)
                builder.add_shape_mesh(b, mesh=vm, cfg=shell_vis_cfg)     # visual only
                plist = []
                for pv, pf in convex_decompose(local, tris):
                    pm = newton.Mesh(pv, pf, compute_inertia=False, maxhullvert=self.hullverts)
                    s = builder.add_shape_mesh(b, mesh=pm, cfg=shell_part_cfg)
                    self.coll_shapes.append(s)
                    plist.append(s)
                self.shell_coll[nm] = plist
                print(f"[INFO] {nm}: {len(plist)} convex collision parts", flush=True)
            else:
                m = newton.Mesh(local, tris, compute_inertia=True, color=(0.45, 0.48, 0.55),
                                maxhullvert=self.hullverts)
                s = builder.add_shape_mesh(b, mesh=m, cfg=shell_cfg)
                self.coll_shapes.append(s)
                self.shell_coll[nm] = [s]
            self.shell_body[nm] = b
            # weld to its bone (bones are at centroid, identity rot) preserving world
            # pose; lift cancels in the relative xform so it stays consistent
            bone_b = self.body[bone]
            joints.append(builder.add_joint_fixed(
                parent=bone_b, child=b,
                parent_xform=wp.transform(p=wp.vec3(*(pos - self.origin[bone_b])), q=quat),
                child_xform=wp.transform(q=wp.quat_identity()),
                label=f"weld_{nm}_{bone}"))

        # ===== palm cover strips (read from the ORIGINAL USD, which still places
        # them) welded over their bone -> the outer palm cover the variant dropped =====
        self.strip_body = {}
        if os.path.exists(ORIG_USD):
            ostage = Usd.Stage.Open(ORIG_USD)
            oxc = UsdGeom.XformCache()
            strip_cfg = newton.ModelBuilder.ShapeConfig()
            strip_cfg.density = 1000.0
            strip_cfg.has_shape_collision = True
            strip_cfg.collision_group = HAND_GROUP
            for nm, bone in PALM_STRIPS.items():
                spath = f"{DEFORM_ROOT}/{nm}"
                if not ostage.GetPrimAtPath(spath).IsValid():
                    continue
                local, tris, pos, quat = extract_mesh_local(ostage, oxc, spath)
                if len(tris) == 0:
                    continue
                b = builder.add_link(xform=wp.transform(p=wp.vec3(*(pos + self.lift)), q=quat),
                                     label=f"strip_{nm}")
                m = newton.Mesh(local, tris, compute_inertia=True, color=(0.80, 0.80, 0.83), maxhullvert=self.hullverts)
                self.coll_shapes.append(builder.add_shape_mesh(b, mesh=m, cfg=strip_cfg))
                self.strip_body[nm] = b
                bone_b = self.body[bone]
                joints.append(builder.add_joint_fixed(
                    parent=bone_b, child=b,
                    parent_xform=wp.transform(p=wp.vec3(*(pos - self.origin[bone_b])), q=quat),
                    child_xform=wp.transform(q=wp.quat_identity()),
                    label=f"weld_{nm}_{bone}"))

        # ===== Shell2: recovered outer bottom-chassis shell, composed from the
        # re-export (base Xform transform x part-file local mesh) and FROZEN
        # (welded to the frozen pinky) as the static outer chassis =====
        if os.path.exists(SHELL2_BASE) and os.path.exists(SHELL2_PART):
            bstage = Usd.Stage.Open(SHELL2_BASE)
            bxc = UsdGeom.XformCache()
            T2 = np.array(bxc.GetLocalToWorldTransform(bstage.GetPrimAtPath(SHELL2_XFORM)),
                          dtype=np.float64).reshape(4, 4)
            pstage = Usd.Stage.Open(SHELL2_PART)
            pxc = UsdGeom.XformCache()
            mp = next(p for p in pstage.Traverse() if p.GetTypeName() == "Mesh")
            mg = UsdGeom.Mesh(mp)
            pts = np.asarray(mg.GetPointsAttr().Get(), dtype=np.float64)
            counts = np.asarray(mg.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
            idx = np.asarray(mg.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
            Mp = np.array(pxc.GetLocalToWorldTransform(mp), dtype=np.float64).reshape(4, 4)
            plocal = (np.concatenate([pts, np.ones((len(pts), 1))], 1) @ Mp)[:, :3]
            world = (np.concatenate([plocal, np.ones((len(plocal), 1))], 1) @ T2)[:, :3].astype(np.float32)
            tris = []
            o = 0
            for c in counts:
                c = int(c)
                for k in range(1, c - 1):
                    tris.append((idx[o], idx[o + k], idx[o + k + 1]))
                o += c
            tris = np.asarray(tris, dtype=np.int32).reshape(-1)
            cen = world.mean(axis=0)
            b = builder.add_link(xform=wp.transform(p=wp.vec3(*(cen + self.lift)), q=wp.quat_identity()),
                                 label="shell2")
            local2 = world - cen
            # same PETG-gray material as the other shells (the recovered Shell2 part
            # came in with a different material; match it to everything else)
            vm = newton.Mesh(local2, tris, compute_inertia=True, color=(0.45, 0.48, 0.55),
                             maxhullvert=self.hullverts)
            builder.add_shape_mesh(b, mesh=vm, cfg=shell_vis_cfg)         # visual only
            plist = []
            if self.shells_jointed:
                pass   # visual mesh only (added above); no collision
            elif self.simple_shells:
                # ONE convex hull (mujoco-warp convexifies the mesh) -> smooths Shell2's
                # rough recovered geometry and collapses its 17 parts to 1.
                pm = newton.Mesh(local2, tris, compute_inertia=False, maxhullvert=self.hullverts)
                s = builder.add_shape_mesh(b, mesh=pm, cfg=shell_part_cfg)
                self.coll_shapes.append(s); plist.append(s)
            else:
                for pv, pf in convex_decompose(local2, tris):
                    pm = newton.Mesh(pv, pf, compute_inertia=False, maxhullvert=self.hullverts)
                    s = builder.add_shape_mesh(b, mesh=pm, cfg=shell_part_cfg)
                    self.coll_shapes.append(s)
                    plist.append(s)
            self.shell_coll["Shell2"] = plist
            print(f"[INFO] Shell2: {len(plist)} convex collision parts", flush=True)
            self.shell_body["Shell2"] = b
            self._shell2_body = b
            self._shell2_pos = np.asarray(cen, np.float64) + self.lift     # body pos (identity rot)
            self._shell2_wv = np.asarray(world, np.float64) + self.lift    # world verts (with lift)
            # Shell2 stays anchored to the thumb-motor (ptr) side -- its real side-wall
            # connection. The explicit Shell1<->Shell2 spherical (ball) coupling is added as
            # a loop-closure equality below, so Shell2 keeps this anchor AND gains the hinge.
            anchor_b = self.body[SHELL2_ANCHOR_BONE]
            joints.append(builder.add_joint_fixed(
                parent=anchor_b, child=b,
                parent_xform=wp.transform(p=wp.vec3(*(cen - self.origin[anchor_b])), q=wp.quat_identity()),
                child_xform=wp.transform(q=wp.quat_identity()), label=f"weld_Shell2_{SHELL2_ANCHOR_BONE}"))

        # MuJoCo: articulate the tree now (loop closures below are EQUALITY constraints, not
        # joints). kamino: defer until the loop-closure + weld D6 joints are built, so they're
        # part of the articulation (an orphaned joint mismatches kamino's joint-model size).
        if not self.kamino:
            builder.add_articulation(joints, label="hand")

        # ---- loop closures: +X connector far end <-> next bone +X tip ----
        self.loop_specs = []
        for (a, b) in PLUS_PAIRS:
            lk, ta, tb = self.plus_link[(a, b)]
            anchor_link = tb - self.origin[lk]               # connector local
            anchor_bone = tb - self.origin[self.body[b]]     # next-bone local
            if self.kamino:
                # ball-and-socket loop closure as a 6-DOF D6: 3 stiff linear (pin the point) +
                # 3 FREE angular (rotation unconstrained, like CONNECT). Both bodies are
                # identity-rotation, so p-only joint frames coincide at the closure point.
                free = lambda a_: Cfg(axis=a_, target_ke=0.0, target_kd=0.0)
                joints.append(builder.add_joint_d6(
                    parent=lk, child=self.body[b],
                    linear_axes=[lin_pin(wp.vec3(1.0, 0.0, 0.0)), lin_pin(wp.vec3(0.0, 1.0, 0.0)),
                                 lin_pin(wp.vec3(0.0, 0.0, 1.0))],
                    angular_axes=[free(wp.vec3(1.0, 0.0, 0.0)), free(wp.vec3(0.0, 1.0, 0.0)),
                                  free(wp.vec3(0.0, 0.0, 1.0))],
                    parent_xform=wp.transform(p=wp.vec3(*anchor_link), q=wp.quat_identity()),
                    child_xform=wp.transform(p=wp.vec3(*anchor_bone), q=wp.quat_identity()),
                    label=f"loop_plus_{a}_{b}"))
            elif self.soft_loops:
                pass   # closed by an env-applied penalty spring (loop_specs has the anchors)
            else:
                builder.add_equality_constraint_connect(
                    body1=lk, body2=self.body[b], anchor=wp.vec3(*anchor_link),
                    label=f"close_plus_{a}_{b}",
                    custom_attributes={"mujoco:eq_solref": [0.005, 1.0]})
            self.loop_specs.append((lk, anchor_link, self.body[b], anchor_bone))
        self._hand_joints = joints   # for the deferred kamino add_articulation

        # ---- explicit Shell1<->Shell2 weld (the chassis hinge) ----
        # Both shells stay anchored to their own side (Shell1->pinky, Shell2->ptr); this
        # closes the loop with a compliant WELD at the center of the spherical surfaces they
        # mate on (fitted from the meshes, then calibrated), so the two halves pivot about the
        # ball center as the palm flexes -- the design's implicit ball joint, explicit.
        if self.shells_jointed and getattr(self, "_shell1_wv", None) is not None and self.shell_body.get("Shell2") is not None:
            C, C1, C2, r1, r2 = _fit_ball_centers(self._shell1_wv, self._shell2_wv)
            self.fit_centers = {"combined": np.asarray(C, np.float64),
                                "shell1": np.asarray(C1, np.float64),
                                "shell2": np.asarray(C2, np.float64)}
            # CALIBRATED weld: the pivot is PINNED to the fitted mating-sphere center (the cyan/
            # yellow markers -- where the shells physically articulate), and the weld torquescale +
            # solref were converged by a BATCHED GPU evolutionary search (~41k candidates, geometric
            # regularizer ON) to match the prior FULL-CoACD-collision config's shell-motion
            # trajectory: tracks that reference's Shell2-in-Shell1 path to 15mm pos / 13.5deg rot
            # (the reference itself drifts ~38mm). See calibrate_shell_ball.py --batched (SHELL_REG=1).
            C = np.array(SHELL_BALL_CENTER, np.float64) + self.lift
            ov = getattr(self.args, "shell_ball_center", None)   # explicit override (build-frame world)
            if ov is not None:
                C = np.asarray(ov, np.float64)
            # WELD (not a free CONNECT): constrains BOTH the pivot position (at the
            # calibrated center) AND the relative orientation, with `torquescale` weighting
            # rotation and `eq_solref` the compliance. A free ball pivots too loosely vs the
            # prior collision (27deg residual); the weld's rotational stiffness closes that.
            # torquescale + solref are calibrated alongside the center (calibrate_shell_ball.py).
            ts = float(getattr(self.args, "shell_torquescale", SHELL_WELD_TORQUESCALE))
            tc = float(getattr(self.args, "shell_solref_tc", SHELL_WELD_SOLREF_TC))
            T1 = wp.transform(wp.vec3(*self._shell1_pos), wp.quat(*[float(x) for x in self._shell1_quat]))
            T2 = wp.transform(wp.vec3(*self._shell2_pos), wp.quat_identity())
            relpose = wp.transform_multiply(wp.transform_inverse(T1), T2)   # Shell2 rel Shell1 (rest)
            # data for per-world weld addition after replicate (batched calibration)
            self.shell_weld_data = {"body1": int(self._shell1_body), "body2": int(self._shell2_body),
                                    "relpose": relpose, "shell2_pos": np.asarray(self._shell2_pos, np.float64),
                                    "center": np.asarray(C, np.float64), "ts": ts, "tc": tc}
            if not self.defer_shell_weld and self.kamino:
                # 6-DOF D6 bushing replacing the weld: linear pin (3) + angular spring (3) holding
                # the rest relative orientation. Both joint frames are placed at the SAME world
                # transform (ball center C, identity orientation) at build, so all 6 coords are 0
                # at rest -> the springs hold the current relative pose. Angular ke scaled by the
                # calibrated torquescale (the rotational/translational stiffness ratio).
                Tj = wp.transform(wp.vec3(*C), wp.quat_identity())
                pxf = wp.transform_multiply(wp.transform_inverse(T1), Tj)
                cxf = wp.transform_multiply(wp.transform_inverse(T2), Tj)
                ang_ke = BUSHING_ANG_KE * ts
                ang = lambda a_: Cfg(axis=a_, target_pos=0.0, target_ke=ang_ke, target_kd=BUSHING_ANG_KD)
                self._hand_joints.append(builder.add_joint_d6(
                    parent=self._shell1_body, child=self._shell2_body,
                    linear_axes=[lin_pin(wp.vec3(1.0, 0.0, 0.0)), lin_pin(wp.vec3(0.0, 1.0, 0.0)),
                                 lin_pin(wp.vec3(0.0, 0.0, 1.0))],
                    angular_axes=[ang(wp.vec3(1.0, 0.0, 0.0)), ang(wp.vec3(0.0, 1.0, 0.0)),
                                  ang(wp.vec3(0.0, 0.0, 1.0))],
                    parent_xform=pxf, child_xform=cxf, label="bushing_Shell1_Shell2"))
            elif not self.defer_shell_weld and self.soft_loops:
                pass   # shell hinge applied as an env penalty spring (shell_weld_data has the anchors)
            elif not self.defer_shell_weld:
                anchor2 = C - self._shell2_pos              # ball center in Shell2's (identity-rot) frame
                builder.add_equality_constraint_weld(
                    body1=self._shell1_body, body2=self._shell2_body, anchor=wp.vec3(*anchor2),
                    torquescale=ts, relpose=relpose, label="weld_Shell1_Shell2",
                    custom_attributes={"mujoco:eq_solref": [tc, 1.0]})
            # expose marker specs for debug viz: (body_index, offset in that body's frame,
            # RGB, name). Stored body-LOCAL so the viewer transforms by the live pose (the
            # hand gets mounted/posed, so build-frame world points wouldn't line up).
            # R1s = Shell1's OWN build rotation (compute explicitly; don't rely on the leaked
            # shell-bone loop variable, which is the last bone's, not Shell1's).
            R1s = np.array(wp.quat_to_matrix(wp.quat(*[float(x) for x in self._shell1_quat]))).reshape(3, 3)
            self.shell_marker_specs = [
                (self._shell1_body, (R1s.T @ (C - self._shell1_pos)).astype(np.float32), (1.0, 0.0, 1.0), "joint(placed)"),
                (self._shell1_body, (R1s.T @ (C1 - self._shell1_pos)).astype(np.float32), (0.0, 1.0, 1.0), "shell1_center"),
                (self._shell2_body, (C2 - self._shell2_pos).astype(np.float32), (1.0, 1.0, 0.0), "shell2_center"),
            ]
            print(f"[INFO] Shell1<->Shell2 ball: placed {C.round(4)} | shell1-sphere {C1.round(4)} "
                  f"(r={r1:.3f}) | shell2-sphere {C2.round(4)} (r={r2:.3f}) | gap {np.linalg.norm(C1-C2)*1000:.1f}mm",
                  flush=True)

        # kamino: articulate the whole hand now that the loop-closure (+ weld) D6 joints exist,
        # so they belong to the articulation and kamino's joint-model size matches the model.
        if self.kamino:
            builder.add_articulation(self._hand_joints, label="hand")

        # Filter intra-hand shape pairs so the hand doesn't self-collide (the
        # SolverMuJoCo path collides inside MuJoCo and ignores collision groups);
        # external collision (ground / objects) is unaffected. EXCEPTION: let the
        # outer shell (Shell2) collide with the inner shell group, so the two palm
        # halves physically stop against each other as the palm flexes.
        allow = set()

        def allow_pairs(A, B):
            for a in A:
                for b in B:
                    if a != b:
                        allow.add(frozenset((a, b)))

        # finger <-> finger (different fingers collide; intra-finger stays filtered)
        fnames = list(self.finger_shapes)
        for ia in range(len(fnames)):
            for ib in range(ia + 1, len(fnames)):
                allow_pairs(self.finger_shapes[fnames[ia]], self.finger_shapes[fnames[ib]])
        # thumb-side shell (Shell2 + cover3, both on ptr) <-> Shell1 (on pinky): all
        # convex-decomposed, so the real walls collide cleanly as the palm flexes
        allow_pairs(self.shell_coll.get("driver_side_palm_cover3", []), self.shell_coll.get("Shell1", []))
        allow_pairs(self.shell_coll.get("Shell2", []), self.shell_coll.get("Shell1", []))
        # pointer finger <-> Shell2 (thumb side); pinky finger <-> Shell1 (pinky side)
        allow_pairs(self.finger_shapes.get("pointer", []), self.shell_coll.get("Shell2", []))
        allow_pairs(self.finger_shapes.get("pinky", []), self.shell_coll.get("Shell1", []))
        # small wrist bones <-> back plate (cover_back)
        small_shapes = [self.bone_shape[k] for k in ("S00", "S01", "S03", "S04") if k in self.bone_shape]
        allow_pairs(small_shapes, self.shell_coll.get("cover_back", []))
        for i in range(len(self.coll_shapes)):
            for j in range(i + 1, len(self.coll_shapes)):
                a, b = self.coll_shapes[i], self.coll_shapes[j]
                if frozenset((a, b)) in allow:
                    continue
                builder.add_shape_collision_filter_pair(a, b)

        if self._do_finalize:
            builder.color()
            self.model = builder.finalize()
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
            self.solver = newton.solvers.SolverMuJoCo(self.model, iterations=100, ls_iterations=20)
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.control = self.model.control()
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
            self.contacts = self.model.contacts()
            self.setup_drive()
            if self.viewer is not None:
                self.viewer.set_model(self.model)
            self.graph = None
            self.capture()
        # embed mode (finalize=False): caller finalizes, then calls bind()+setup_drive()

    def bind(self, model, solver, state_0, state_1, control, contacts):
        """Embed mode: wire this hand to a model the CALLER finalized (e.g. with an
        arm + object in the same builder). Call setup_drive() afterward."""
        self.model, self.solver = model, solver
        self.state_0, self.state_1 = state_0, state_1
        self.control, self.contacts = control, contacts
        self.graph = None

    def setup_drive(self):
        """Post-finalize: map flexion joints to DOF slots and build the pose battery.
        Needs self.model populated and self.state_0 at the rest pose (eval_fk'd)."""
        args = self.args
        qd_start = self.model.joint_qd_start.numpy()
        self.n_curl = len(self.curl_specs)
        if self.n_curl:
            self.curl_dof = wp.array(np.array([qd_start[j] for (j, _, _) in self.curl_specs],
                                              dtype=np.int32), dtype=wp.int32)
            self.curl_target = wp.array(np.array([t for (_, t, _) in self.curl_specs],
                                                 dtype=np.float32), dtype=wp.float32)
            self.curl_phase = wp.array(np.array([p for (_, _, p) in self.curl_specs],
                                                dtype=np.float32), dtype=wp.float32)
        self.curl_omega = float(getattr(args, "curl_omega", 1.2))
        self.do_curl = not bool(getattr(args, "no_curl", False))
        # ---- real pose battery (robot_hand_poses): build the show timeline ----
        self.use_poses = False
        if not bool(getattr(args, "no_curl", False)):
            try:
                col = {jn: int(qd_start[jid]) for jn, jid in self.jname_to_jid.items()}
                jl = rhp.JointLookup(
                    col=col,
                    lower=self.model.joint_limit_lower.numpy().astype(np.float64),
                    upper=self.model.joint_limit_upper.numpy().astype(np.float64),
                    n_dof=int(self.model.joint_dof_count))
                meta_pos = {k: self.body_pos(self.finger_body[rhp.CHAINS[k]["meta"]]) for k in rhp.CHAINS}
                thumb_base = self.body_pos(self.finger_body["Group_1"])
                finger_of = rhp.resolve_finger_of(meta_pos, thumb_base)
                poses = rhp.build_poses(finger_of)
                self.pose_names = [nm for nm, _ in poses.POSES]
                self.pose_T = [jl.angles(p).astype(np.float32) for _, p in poses.POSES]
                self.blend_s = float(getattr(args, "blend", 0.9))
                self.hold_s = float(getattr(args, "hold", 1.6))
                self.use_poses = True
                print(f"[INFO] pose battery: {len(self.pose_T)} poses, chains {finger_of}", flush=True)
            except Exception as e:
                print(f"[WARN] pose battery unavailable ({e}); falling back to curl cycle", flush=True)

    def _drive(self):
        # write the time-varying joint targets into control.joint_target_pos in place
        if self.control.joint_target_pos is None:
            return
        if self.use_poses:
            # cycle the real pose battery: blend (smoothstep) then hold, looping
            n = len(self.pose_T)
            period = self.blend_s + self.hold_s
            i = int(self.sim_time / period)
            local = self.sim_time - i * period
            cur, prev = self.pose_T[i % n], self.pose_T[(i - 1) % n]
            f = min(local / self.blend_s, 1.0) if self.blend_s > 0 else 1.0
            f = f * f * (3.0 - 2.0 * f)                      # smoothstep
            target = (1.0 - f) * prev + f * cur
            self.control.joint_target_pos.assign(np.ascontiguousarray(target, dtype=np.float32))
        elif self.do_curl and self.n_curl > 0:
            wp.launch(_curl_kernel, dim=self.n_curl, inputs=[
                float(self.sim_time), float(self.curl_omega),
                self.curl_dof, self.curl_target, self.curl_phase, self.control.joint_target_pos])

    def current_pose_name(self):
        if not self.use_poses:
            return "curl"
        period = self.blend_s + self.hold_s
        return self.pose_names[int(self.sim_time / period) % len(self.pose_names)]

    def capture(self):
        if wp.get_device().is_cuda:
            try:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph
            except Exception:
                self.graph = None
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self._drive()                 # update finger curl targets for this frame
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    # ---- measurement helpers ----
    def body_x_axis(self, idx):
        q = self.state_0.body_q.numpy()[idx]
        quat = wp.quat(float(q[3]), float(q[4]), float(q[5]), float(q[6]))
        d = wp.quat_rotate(quat, wp.vec3(1.0, 0.0, 0.0))
        return np.array([float(d[0]), float(d[1]), float(d[2])])

    def corner_bend_deg(self, k):
        p, c = self.corners[k]
        a, b = self.body_x_axis(p), self.body_x_axis(c)
        return math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0))))

    def body_pos(self, idx):
        return self.state_0.body_q.numpy()[idx][:3].copy()

    def loop_gap(self):
        bq = self.state_0.body_q.numpy()
        worst = 0.0
        for (lk, la, bn, ba) in self.loop_specs:
            ql = bq[lk]
            p1 = wp.transform_point(wp.transform(wp.vec3(*ql[:3]), wp.quat(*ql[3:])), wp.vec3(*la))
            qn = bq[bn]
            p2 = wp.transform_point(wp.transform(wp.vec3(*qn[:3]), wp.quat(*qn[3:])), wp.vec3(*ba))
            worst = max(worst, float(wp.length(p1 - p2)))
        return worst

    def max_ang_speed(self, bodies=None):
        qd = self.state_0.body_qd.numpy()[:, 3:6]
        if bodies is not None:
            qd = qd[bodies]
        return float(np.max(np.linalg.norm(qd, axis=1)))


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--check", action="store_true",
                        help="Headless numeric verification instead of the GUI.")
    parser.add_argument("--kick", type=float, default=0.5,
                        help="rad/s perturbation on every joint DOF.")
    parser.add_argument("--kspring", type=float, default=K_SPRING,
                        help="centering-spring stiffness per cone axis.")
    parser.add_argument("--damping", type=float, default=DAMPING,
                        help="joint drive damping per cone axis.")
    parser.add_argument("--theta", type=float, default=math.degrees(THETA_MAX),
                        help="per-cone angular limit in degrees.")
    parser.add_argument("--fingerke", type=float, default=1.0,
                        help="scale on the transcribed finger drive stiffness/damping.")
    parser.add_argument("--lift", type=float, default=0.15,
                        help="lift the whole hand +Z (m) so it clears the ground plane.")
    parser.add_argument("--curl-omega", dest="curl_omega", type=float, default=1.2,
                        help="fallback curl-cycle angular frequency [rad/s] (if poses unavailable).")
    parser.add_argument("--no-curl", dest="no_curl", action="store_true",
                        help="disable all finger animation (hold extended).")
    parser.add_argument("--hold", type=float, default=1.6, help="seconds to hold each pose.")
    parser.add_argument("--blend", type=float, default=0.9, help="seconds to blend between poses.")
    parser.add_argument("--hullverts", type=int, default=256,
                        help="max vertices per collision convex hull (64=coarse default).")
    args, _ = parser.parse_known_args()

    if args.check:
        wp.init()
        args.no_curl = True          # measure settling without the curl animation
        viewer = newton.viewer.ViewerNull(num_frames=400)
        ex = Example(viewer, args)
        ex.graph = None
        tmax = math.degrees(ex.theta_max)
        nk = len(ex.corners)
        all_bones = [ex.body[k] for k in {**BIG, **SMALL}]
        p0 = {k: ex.body_pos(ex.body[k]) for k in {**BIG, **SMALL}}
        tip0 = [ex.body_pos(t) for t in ex.fingertips]

        KICK = float(args.kick)
        jqd = ex.state_0.joint_qd.numpy().copy()
        jqd[:] = KICK
        ex.state_0.joint_qd.assign(jqd)
        print(f"[check] kicked {jqd.size} joint DOFs at {KICK} rad/s")

        N = 600
        traj = np.zeros((N, nk)); gaps = np.zeros(N); speed = np.zeros(N)
        for f in range(N):
            ex.step()
            for k in range(nk):
                traj[f, k] = ex.corner_bend_deg(k)
            gaps[f] = ex.loop_gap()
            speed[f] = ex.max_ang_speed(all_bones)

        drift = {k: float(np.linalg.norm(ex.body_pos(ex.body[k]) - p0[k])) for k in {**BIG, **SMALL}}
        peak = traj.max(axis=0); rest = traj[-100:].mean(axis=0)
        finite = np.all(np.isfinite(ex.state_0.body_q.numpy()))

        print(f"[check] THETA_MAX={tmax:.1f} deg  bodies={ex.model.body_count}  "
              f"corners={nk}  eq_constraints={ex.model.equality_constraint_count}")
        print("[check] bone centroid drift from rest (mm): "
              + "  ".join(f"{k}:{drift[k]*1e3:.1f}" for k in {**BIG, **SMALL}))
        print(f"[check] peak corner bend (under kick) = {peak.max():.2f} deg  (cone {tmax:.1f})")
        print(f"[check] resting corner bend after spring-back = {rest.max():.3f} deg")
        print(f"[check] loop closure gap: rest {gaps[-100:].mean()*1e3:.2f} mm  peak {gaps.max()*1e3:.2f} mm")
        print(f"[check] settle: max bone angular speed = {speed[-100:].max():.4f} rad/s")
        tipdrift = [float(np.linalg.norm(ex.body_pos(t) - tip0[i])) for i, t in enumerate(ex.fingertips)]
        print(f"[check] fingertip drift from rest (mm): "
              + "  ".join(f"{n}:{tipdrift[i]*1e3:.1f}" for i, n in enumerate(FINGER_CHAINS)))

        settled = speed[-100:].max() < 1.0
        within = peak.max() < tmax + 5.0
        flexed = peak.max() > 2.0
        springback = rest.max() < 1.0
        loop_ok = gaps[-100:].mean() < 0.002
        holds = max(drift.values()) < 0.010
        print(f"[check] finite: {'PASS' if finite else 'FAIL'} | "
              f"holds(<10mm): {'PASS' if holds else 'FAIL'} | "
              f"loops closed(<2mm): {'PASS' if loop_ok else 'FAIL'} | "
              f"settles: {'PASS' if settled else 'FAIL'} | "
              f"flexed(>2deg): {'PASS' if flexed else 'FAIL'} | "
              f"cone holds(<{tmax+5:.0f}): {'PASS' if within else 'FAIL'} | "
              f"spring-back(<1deg): {'PASS' if springback else 'FAIL'}")
    else:
        viewer, args = newton.examples.init(parser)
        newton.examples.run(Example(viewer, args), args)
