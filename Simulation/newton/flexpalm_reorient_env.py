"""Allegro-style in-hand cube REORIENTATION RL environment, driven by the flexible
3D-printed flexpalm hand + its 15-pad tactile sensor. Pure Newton + SolverMuJoCo,
GPU-batched over N worlds (one model, one solver), `isaac` conda env.

The hand is fixed palm-up; a free cube rests in the cupped fingers; the policy moves
the 22 finger DOFs (PD position targets) to rotate the cube toward a randomly sampled
target orientation. Reward, goal sampling, success/consecutive-success and drop
termination mirror IsaacLab's `InHandManipulationEnv`. Observations include the 15
per-pad tactile normal forces (the hand's sensors), as the user requested.

Build/scaling validated empirically: the heavy closed-loop CoACD hand replicates into
N worlds via `ModelBuilder.replicate` (mesh-shared), zero global DOFs, finite stepping,
~6k env-steps/s at N=256 on an RTX 5090.

  python flexpalm_reorient_env.py --check                  # headless gate (small N)
  (training: see train_reorient.py)
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import warp as wp
import newton
import newton_compat  # noqa: F401  (1.2->1.4 equality API shim)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flexpalm_bones4 as fb                          # noqa: E402  (the hand)
from hand_taxels_newton import HandTaxelSensor         # noqa: E402  (tactile, batched)

# ---- tunables (validated palm-up cup geometry) ----
MOUNT_QUAT = (0.0, 0.0, 0.0, 1.0)        # identity -> fingers curl UP (palm-up cup)
MOUNT_Z = 0.20                            # pinky-root height; palm ~0.19, cup tips ~0.27
N_FINGER = 22                             # actuated finger revolute DOFs
N_PADS = 15                               # tactile pads
CUPPED_FRAC = 0.45                        # default finger curl (open->fist blend) cradling the cube
ACT_DELTA = float(os.environ.get("ACT_DELTA", "0.7"))    # max per-joint target deviation from cupped [rad]
                                                          # (0.7 gives the fingers authority to roll the cube)
VEL_CLAMP = 40.0                          # watchdog: clamp joint vel [rad/s] each substep (>> nominal; only catches runaway)
BVEL_CLAMP = 40.0                         # watchdog: clamp body spatial vel each substep (arrests blowups -> finite, before NaN)
CUBE_HALF = 0.025                         # 5 cm cube
CUBE_POS = (0.025, 0.10, 0.225)           # rest in the cup (just above the palm)
CUBE_DENSITY = 150.0
FALL_DIST = 0.12                          # cube this far from the in-hand ref -> dropped
SUCCESS_TOL = 0.4                         # rad; rot_dist <= tol counts as a success
NAN_PENALTY = -50.0                       # reward for a diverged world (terminate + reset it)
ROT_SHAPE_K = 10.0                        # dense reward per rad the cube turns TOWARD the goal/step
AT_GOAL_BONUS = 3.0                        # per-step reward for being within tolerance (encourages staying)
STAB_K = 8.0                               # bonus for holding the cube STILL at the goal (kills jitter -> real holds)
HOLD_STEPS = 10                           # consecutive in-tolerance steps to "achieve" (hold) a goal
HOLD_BONUS = 50.0                          # bonus for achieving (holding) a goal -> resample + curriculum
# Goal curriculum: start with goals near the current orientation (easy wins bootstrap the
# manipulation primitive past the "just hold" local optimum), widen toward any-axis as success rises.
GOAL_ANGLE_START = 0.5                     # rad; initial goal difficulty
GOAL_ANGLE_MAX = math.pi                   # rad; full any-orientation
CURRIC_THRESH = 0.6                        # widen only when very good at the current level (solid
                                           # per-level mastery before harder goals; self-regulating)
CURRIC_STEP = 0.003                        # rad/step widening rate when above threshold
VEL_OBS_SCALE = 0.2
ACT_MOVING_AVG = float(os.environ.get("ACT_MAVG", "0.3"))    # low-pass on targets (LOWER = smoother -> less cube jitter)
OBS_DIM = N_FINGER * 2 + 3 + 4 + 3 + 3 + 4 + 4 + N_PADS + N_FINGER   # = 102


# ---------------------------------------------------------------------------
# torch quaternion helpers (xyzw, matching Newton free-joint quat order)
# ---------------------------------------------------------------------------
def quat_mul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


def quat_conj(q):
    return q * torch.tensor([-1.0, -1.0, -1.0, 1.0], device=q.device)


def quat_norm(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def rand_quat(n, device):
    """Uniform random unit quaternion (xyzw), Shepperd / Marsaglia method."""
    u = torch.rand(n, 3, device=device)
    q = torch.stack([
        torch.sqrt(1 - u[:, 0]) * torch.sin(2 * math.pi * u[:, 1]),
        torch.sqrt(1 - u[:, 0]) * torch.cos(2 * math.pi * u[:, 1]),
        torch.sqrt(u[:, 0]) * torch.sin(2 * math.pi * u[:, 2]),
        torch.sqrt(u[:, 0]) * torch.cos(2 * math.pi * u[:, 2]),
    ], dim=-1)
    return quat_norm(q)


def rand_quat_near(base, max_angle, device, min_angle=0.0):
    """Random quat (xyzw) `[min_angle, max_angle]` rad from `base` [n,4]: base rotated by a random
    axis and angle ~ U(min_angle, max_angle). Curriculum goals start easy (small max_angle) and widen.
    `min_angle` (set to the success tolerance by callers) guarantees a fresh goal is NEVER already
    inside tolerance at spawn -- otherwise ~80% of goals sampled in U(0, 0.5) rad sit within the 0.4 rad
    tolerance from step 0, and a passive cube racks up "held" goals with zero skill (consecutive_successes
    reads ~0.25 at init). With min_angle=tol every goal must be rotated INTO tolerance to be solved."""
    n = base.shape[0]
    axis = torch.randn(n, 3, device=device)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    lo = min(min_angle, max_angle)                       # guard: never invert the band
    ang = lo + (max_angle - lo) * torch.rand(n, device=device)
    half = 0.5 * ang
    dq = torch.cat([axis * torch.sin(half).unsqueeze(-1), torch.cos(half).unsqueeze(-1)], dim=-1)  # xyzw
    return quat_norm(quat_mul(dq, base))


def rotation_distance(qa, qb):
    """Angle [rad] between two xyzw quats: 2*asin(|vec(qa*conj(qb))|)."""
    d = quat_mul(qa, quat_conj(qb))
    return 2.0 * torch.asin(torch.clamp(d[..., :3].norm(dim=-1), max=1.0))


# Velocity watchdog as WARP kernels (not torch.clamp_): keeps the whole substep loop
# Warp-only so it can be captured into a CUDA graph. Clamps in-place each substep.
@wp.kernel
def _clamp_f32_k(a: wp.array(dtype=wp.float32), c: wp.float32):
    i = wp.tid()
    a[i] = wp.clamp(a[i], -c, c)


@wp.kernel
def _clamp_spatial_k(a: wp.array(dtype=wp.spatial_vectorf), c: wp.float32):
    i = wp.tid()
    v = a[i]
    a[i] = wp.spatial_vector(wp.clamp(v[0], -c, c), wp.clamp(v[1], -c, c), wp.clamp(v[2], -c, c),
                             wp.clamp(v[3], -c, c), wp.clamp(v[4], -c, c), wp.clamp(v[5], -c, c))


# Penalty-spring loop closures (solver='mujoco' + soft_loops): close each kinematic loop with an
# explicit spring FORCE between two body-local anchor points, added to body_f each substep instead
# of an equality constraint. Removes the equality SOLVE (the suspected NaN source) while staying on
# SolverMuJoCo. body_f is the per-body spatial wrench [torque(top), force(bottom)] about COM, world.
LOOP_SPRING_KE = float(os.environ.get("LOOP_KE", "1.0e4"))   # closure spring stiffness [N/m]
LOOP_SPRING_KD = float(os.environ.get("LOOP_KD", "1.0e2"))   # closure spring damping
LOOP_SPRING_FMAX = float(os.environ.get("LOOP_FMAX", "300")) # per-spring force clamp [N]


@wp.kernel
def _loop_spring_k(body_q: wp.array(dtype=wp.transform),
                   body_qd: wp.array(dtype=wp.spatial_vector),
                   body_com: wp.array(dtype=wp.vec3),
                   nbody: wp.int32, n_springs: wp.int32,
                   s_b1: wp.array(dtype=wp.int32), s_b2: wp.array(dtype=wp.int32),
                   s_a1: wp.array(dtype=wp.vec3), s_a2: wp.array(dtype=wp.vec3),
                   s_k: wp.array(dtype=wp.float32), s_d: wp.array(dtype=wp.float32),
                   fmax: wp.float32,
                   body_f: wp.array(dtype=wp.spatial_vector)):
    tid = wp.tid()
    w = tid // n_springs
    s = tid - w * n_springs
    b1 = w * nbody + s_b1[s]
    b2 = w * nbody + s_b2[s]
    T1 = body_q[b1]; T2 = body_q[b2]
    p1 = wp.transform_point(T1, s_a1[s])           # closure point on body1, world
    p2 = wp.transform_point(T2, s_a2[s])           # closure point on body2, world
    c1 = wp.transform_point(T1, body_com[b1])      # COM world
    c2 = wp.transform_point(T2, body_com[b2])
    qd1 = body_qd[b1]; qd2 = body_qd[b2]
    # body_qd is [linear(top), angular(bottom)]; point vel = v_com + omega x (p - com)
    pv1 = wp.spatial_top(qd1) + wp.cross(wp.spatial_bottom(qd1), p1 - c1)
    pv2 = wp.spatial_top(qd2) + wp.cross(wp.spatial_bottom(qd2), p2 - c2)
    F = s_k[s] * (p2 - p1) + s_d[s] * (pv2 - pv1)  # pull p1 toward p2
    fn = wp.length(F)
    if fn > fmax:
        F = F * (fmax / fn)
    # body_f convention is [force(top 0:3), torque(bottom 3:6)] (apply_mjc_body_f_kernel),
    # wrench about COM in world. Force +F at p1 on body1, -F at p2 on body2.
    wp.atomic_add(body_f, b1, wp.spatial_vector(F, wp.cross(p1 - c1, F)))
    wp.atomic_add(body_f, b2, wp.spatial_vector(-F, wp.cross(p2 - c2, -F)))


class _StubTactile:
    """Phase-1 placeholder for the tactile sensor under solver='kamino': the real
    HandTaxelSensor is a mujoco-warp contact rasterizer (needs mjw_data) and is rewritten
    over Newton-generic contacts in Phase 2. Returns zero pad forces so obs stays 102-dim."""
    def __init__(self, num_worlds, device, n_pads=15):
        self._z = torch.zeros(num_worlds, n_pads, device=device)

    def update(self):
        pass

    def pad_totals_torch(self):
        return self._z


class FlexPalmReorientEnv:
    """Batched-Newton in-hand reorientation VecEnv. Implements the rsl_rl VecEnv
    protocol; `reset()` returns a {"policy": obs} dict so it also feeds CfCPPO with
    no adapter."""

    def __init__(self, num_envs=256, device="cuda", substeps=8, episode_s=10.0,
                 njmax=384, nconmax=320, seed=0, hand_args=None, verbose=True,
                 collision_method="convex_hull", share_meshes=False,
                 contacts_per_world=600, tripairs_per_world=80_000, weld_params=None,
                 use_graph=True, solver="mujoco", soft_loops=False, cube_half=None, reward_scale=1.0,
                 drop_penalty=50.0):
        # reward_scale: multiply the per-step reward. The shaping+bonus reward is large (~100/step ->
        # returns in the thousands -> value loss ~5e3), which blows up big-net/full-batch PPO value
        # gradients (NaN). Advantage-normalized PPO is scale-invariant, so scaling down (e.g. 0.1)
        # stabilizes the value fit without changing behavior. Default 1.0 (MLP baseline unchanged).
        self._reward_scale = float(reward_scale)
        # Drop-penalty curriculum knob. At the default 50 every episode's return
        # is dominated by the terminal drop, so with a policy that always drops
        # (e.g. a BC-cloned gait) the advantage signal can't see reach QUALITY —
        # per-episode shaping differences (~±2 unscaled) drown under the constant
        # -50. Phase-1 finetunes run ~10 so reach-vs-drop trade-offs carry
        # gradient; restore 50 once drops are rare.
        self._drop_penalty = float(drop_penalty)
        # cube_half: box half-extent [m] (default 0.025 = 5cm cube). Larger -> easier for this
        # hand's grasp span to get purchase. The rest height is raised so the bigger cube still
        # seats on the palm (bottom face ~unchanged).
        self._cube_half = float(cube_half) if cube_half else CUBE_HALF
        self._cube_pos = (CUBE_POS[0], CUBE_POS[1], CUBE_POS[2] + (self._cube_half - CUBE_HALF))
        # solver: "mujoco" (default, equality-constrained) or "kamino" (blocked at the bridge).
        # soft_loops (mujoco only): replace the 4 equality loop closures with explicit penalty
        # springs on body_f -> removes the equality SOLVE (the suspected NaN source), tactile/graph
        # intact. This is the active NaN-debug path.
        self._solver_kind = str(solver).lower()
        self._kamino = self._solver_kind == "kamino"
        self._soft_loops = bool(soft_loops) and not self._kamino
        # weld_params: optional [num_envs, 5] array (cx,cy,cz,torquescale,solref_tc) giving a
        # DIFFERENT Shell1<->Shell2 weld per world -> batch-evaluate many calibration candidates
        # in parallel (one rollout scores all worlds). None = homogeneous (proto weld).
        self._weld_params = None if weld_params is None else np.asarray(weld_params, np.float64)
        # CUDA-graph capture of the substep physics loop (the ~90% of per-step launches).
        # Captured once then replayed; eager fallback (LOUD) if capture fails.
        self._use_graph = bool(use_graph)
        self._graph = None
        self._graph_ok = None        # None=untried, True=captured, False=eager fallback
        self._warmed = False         # one eager warm-up step before capture (triggers lazy allocs)
        self.verbose = bool(verbose)
        self._contacts_per_world = int(contacts_per_world)
        self._tripairs_per_world = int(tripairs_per_world)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.substeps = int(substeps)
        self.fps = 100
        self.dt = (1.0 / self.fps) / self.substeps
        self.num_actions = N_FINGER
        self.num_obs = OBS_DIM
        self.max_episode_length = int(episode_s * self.fps)
        torch.manual_seed(seed)

        args = hand_args or argparse.Namespace()
        # hullverts=64 collision hulls: +9% speed, no grasp/tactile loss (held 98%). The
        # eval viewer overrides to 256 for crisper visuals (N=1, speed irrelevant there).
        # shells_jointed: full-visual non-colliding shells + explicit Shell1<->Shell2 spherical
        # joint (topology-preserving alternative to the simple_shells convex hulls).
        for k, v in dict(kspring=2.0, damping=0.25, theta=35.0, fingerke=0.25,
                         hullverts=64, lift=0.0, hold=1.6, blend=0.9, shells_jointed=True).items():
            setattr(args, k, getattr(args, k, v))
        if self._kamino:
            args.kamino = True                           # build closed loops as 6-DOF D6 bushings
            if self._weld_params is not None:
                raise ValueError("weld_params (per-world MuJoCo welds) is not supported with solver='kamino'")
        if self._soft_loops:
            args.soft_loops = True                        # hand skips equality; env applies springs
            if self._weld_params is not None:
                raise ValueError("weld_params is not supported with soft_loops=True")
        if self._weld_params is not None:
            args.defer_shell_weld = True                 # add per-world welds after replicate

        _SolverCls = newton.solvers.SolverKamino if self._kamino else newton.solvers.SolverMuJoCo
        # ---- proto: world-fixed base + palm-up hand + free cube ----
        proto = newton.ModelBuilder()
        _SolverCls.register_custom_attributes(proto)
        base = proto.add_body(label="mount")
        proto.add_joint_fixed(parent=-1, child=base,
                              parent_xform=wp.transform(q=wp.quat_identity()),
                              child_xform=wp.transform(q=wp.quat_identity()), label="fix_mount")
        mount_xf = wp.transform(wp.vec3(0.0, 0.0, MOUNT_Z), wp.quat(*MOUNT_QUAT))
        self.hand = fb.Example(None, args, ext_builder=proto, attach_body=base,
                               attach_xform=mount_xf, finalize=False)
        # free cube on top of the cup
        cube_body = proto.add_link(xform=wp.transform(wp.vec3(*self._cube_pos), wp.quat_identity()), label="cube")
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.density = CUBE_DENSITY; cfg.has_shape_collision = True; cfg.collision_group = 3
        self.cube_shape = proto.add_shape_box(cube_body, hx=self._cube_half, hy=self._cube_half, hz=self._cube_half, cfg=cfg)
        cj = proto.add_joint_free(parent=-1, child=cube_body,
                                  parent_xform=wp.transform(q=wp.quat_identity()), label="free_cube")
        proto.add_articulation([cj], label="cube")       # free joints must belong to an articulation
        cube_local_body = cube_body                      # body index within one world

        # share_meshes=True simplifies + shares collision meshes by reference across worlds
        # (memory lever) BEFORE replicate. DISABLED by default: sharing one mesh across N
        # worlds appears to introduce a per-world collision hazard that destabilizes the
        # batched contact solve at high world counts. share_meshes=False matches the proven
        # single-world demos (raw per-world CoACD meshes, hullverts=256).
        if share_meshes:
            proto.approximate_meshes(collision_method, keep_visual_shapes=True)

        # ---- replicate into N worlds, finalize, solver ----
        scene = newton.ModelBuilder()
        _SolverCls.register_custom_attributes(scene)
        if self._weld_params is None:
            scene.replicate(proto, world_count=self.num_envs)             # homogeneous (fast path)
        else:
            # heterogeneous: build each world by hand so its Shell1<->Shell2 weld (one
            # calibration candidate) is added INSIDE that world's context (else it lands on
            # the global world -1 and finalize rejects it). One rollout then scores all worlds.
            wd = self.hand.shell_weld_data
            s2pos = wd["shell2_pos"]; relpose = wd["relpose"]; b1l, b2l = wd["body1"], wd["body2"]
            for w in range(self.num_envs):
                off = scene.body_count
                scene.begin_world()
                scene.add_builder(proto)
                cx, cy, cz, ts, tc = (float(x) for x in self._weld_params[w])
                scene.add_equality_constraint_weld(
                    body1=off + b1l, body2=off + b2l, anchor=wp.vec3(*(np.array([cx, cy, cz]) - s2pos)),
                    torquescale=ts, relpose=relpose, label=f"weld_S1S2_w{w}",
                    custom_attributes={"mujoco:eq_solref": [tc, 1.0]})
                scene.end_world()
        self.model = scene.finalize(device=str(self.device))
        assert self.model.world_count == self.num_envs
        if self._kamino:
            # maximal-coordinate P-ADMM; use_collision_detector=False -> consume the Newton
            # CollisionPipeline contacts we pass to step(). use_fk_solver=False: kamino's FK
            # solver can't classify our 6-DOF D6 bushing dofs ("Unknown joint dof type"), and we
            # reset body poses via newton.eval_fk ourselves, so its internal FK isn't needed.
            kcfg = newton.solvers.SolverKamino.Config(use_fk_solver=False)
            self.solver = newton.solvers.SolverKamino(self.model, kcfg)
            assert int(self.model.equality_constraint_count) == 0, \
                f"kamino build still has {self.model.equality_constraint_count} equality constraints"
        else:
            self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=njmax, nconmax=nconmax,
                                                      iterations=100, ls_iterations=20)
        # Collision pipeline with buffers SCALED TO WORLD COUNT. The default
        # max_triangle_pairs (1M) overflows above ~16 worlds for this many-part hand
        # (~40k triangle pairs/world), silently corrupting the contact solve -> NaN.
        # Sizing it per-world is the actual fix for the high-N instability.
        from newton._src.sim.collide import CollisionPipeline
        # Buffers sized to MEASURED simple_shells usage (~214 contacts/world peak) with margin.
        # Undersizing silently drops contacts -> penetration/NaN; oversizing wastes VRAM and
        # caps how high num_envs can scale. (With full CoACD shells these need to be ~10x larger.)
        per_world_contacts = int(getattr(self, "_contacts_per_world", 600))
        per_world_tripairs = int(getattr(self, "_tripairs_per_world", 80_000))
        tri_pairs = max(1_000_000, per_world_tripairs * self.num_envs)
        rigid_max = max(int(self.model.rigid_contact_max), per_world_contacts * self.num_envs)
        self._pipeline = CollisionPipeline(self.model, broad_phase="explicit",
                                           max_triangle_pairs=tri_pairs, rigid_contact_max=rigid_max)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts(collision_pipeline=self._pipeline)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ---- pose battery (bind the hand to world 0 of the finalized model) ----
        self.hand.bind(self.model, self.solver, self.state_0, self.state_1, self.control, self.contacts)
        self.hand.setup_drive()
        nm = self.hand.pose_names
        qd_start = self.model.joint_qd_start.numpy()
        q_start = self.model.joint_coord_start.numpy() if hasattr(self.model, "joint_coord_start") else qd_start
        # world-0 (== local) finger dof slots, in Revolute_1..N_FINGER order
        fj = [(int(jn.split("_")[1]), jid) for jn, jid in self.hand.jname_to_jid.items() if jn.startswith("Revolute_")]
        fj.sort()
        local_finger_dof = np.array([int(qd_start[jid]) for _, jid in fj], dtype=np.int64)   # (22,)
        open_vec = self.hand.pose_T[nm.index("open")]
        fist_vec = self.hand.pose_T[nm.index("fist")]
        self.finger_open = torch.tensor(open_vec[local_finger_dof], device=self.device, dtype=torch.float32)
        self.finger_fist = torch.tensor(fist_vec[local_finger_dof], device=self.device, dtype=torch.float32)
        self.finger_cupped = (1 - CUPPED_FRAC) * self.finger_open + CUPPED_FRAC * self.finger_fist
        lo = self.model.joint_limit_lower.numpy(); hi = self.model.joint_limit_upper.numpy()
        self.finger_lo = torch.tensor(lo[local_finger_dof], device=self.device, dtype=torch.float32)
        self.finger_hi = torch.tensor(hi[local_finger_dof], device=self.device, dtype=torch.float32)

        # ---- per-world flat index tensors (CSR per-world contiguous blocks) ----
        cws = self.model.joint_coord_world_start.numpy()
        dws = self.model.joint_dof_world_start.numpy()
        bws = self.model.body_world_start.numpy()
        w = np.arange(self.num_envs)
        self.finger_dof_idx = torch.tensor(dws[w, None] + local_finger_dof[None, :], device=self.device)   # [N,22]
        self.finger_q_idx = self.finger_dof_idx                                  # revolute: coord==dof slot
        # cube free-joint coords = the LAST 7 coords of each world's block (free cube
        # was added to the proto after the hand). Verified by the --check cube-pose readout.
        coords_per_world = int(cws[1] - cws[0])
        self.dofs_per_world = int(dws[1] - dws[0])
        assert (np.diff(dws[:self.num_envs + 1]) == self.dofs_per_world).all(), "non-uniform per-world dofs"
        self.cube_q_idx = torch.tensor(cws[w, None] + (coords_per_world - 7) + np.arange(7)[None, :], device=self.device)  # [N,7]
        self.cube_body_idx = torch.tensor(bws[w] + cube_local_body, device=self.device)                    # [N]

        # zero-copy torch views of the live sim arrays
        self.jq = wp.to_torch(self.state_0.joint_q)
        self.jqd = wp.to_torch(self.state_0.joint_qd)
        self.tgt = wp.to_torch(self.control.joint_target_pos)
        self.bq = wp.to_torch(self.state_0.body_q)          # [body,7] pos+quat(xyzw)
        self.bqd = wp.to_torch(self.state_0.body_qd)        # [body,6] spatial [ang3, lin3]
        if self._soft_loops:
            self._build_loop_springs()

        # ---- tactile (batched: pad_total [N,15]) ----
        if self._kamino:
            self.sensor = _StubTactile(self.num_envs, self.device)   # Phase 1: no mjw_data under kamino
        else:
            self.sensor = HandTaxelSensor(self.hand, self.solver, self.model)

        # ---- RL buffers ----
        N = self.num_envs
        self.goal_quat = rand_quat(N, self.device)
        self.prev_targets = self.finger_cupped.repeat(N, 1)            # [N,22]
        self.actions = torch.zeros(N, N_FINGER, device=self.device)
        self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=self.device)
        self.reset_buf = torch.ones(N, dtype=torch.bool, device=self.device)
        self.successes = torch.zeros(N, device=self.device)
        self.prev_rot_dist = torch.full((N,), math.pi, device=self.device)   # for dense d(rot_dist) shaping
        self.goal_angle = GOAL_ANGLE_START     # curriculum difficulty (rad), widens with success
        self._succ_ema = 0.0                    # running hold/achieve fraction driving the curriculum
        self.hold_counter = torch.zeros(N, device=self.device)   # consecutive in-tolerance steps
        self.in_hand_pos = torch.tensor(self._cube_pos, device=self.device).repeat(N, 1)   # reference cube position
        self.obs_buf = torch.zeros(N, OBS_DIM, device=self.device)
        # Named slices of the obs concatenation in _compute_obs (simbridge sim_obs reads this).
        self.obs_slices = {"finger_q": (0, 22), "finger_qd": (22, 44),
                           "cube_pos_err": (44, 47), "cube_quat": (47, 51),
                           "cube_linvel": (51, 54), "cube_angvel": (54, 57),
                           "goal_quat": (57, 61), "rel_quat": (61, 65),
                           "tactile": (65, 80), "prev_actions": (80, 102)}
        self.cfg = argparse.Namespace(success_tolerance=SUCCESS_TOL)
        if verbose:
            print(f"[reorient] N={N} obs={OBS_DIM} act={N_FINGER} bodies/world={self.model.body_count//N} "
                  f"dofs/world={coords_per_world} max_ep={self.max_episode_length}", flush=True)
        self.reset()

    # ---- helpers ----
    def _cube_pose(self):
        bq = self.bq[self.cube_body_idx]            # [N,7]
        return bq[:, :3], quat_norm(bq[:, 3:7])

    def _build_loop_springs(self):
        """Build per-world penalty-spring specs (b1,b2,anchors,k,d) replacing the 4 equality
        loop closures, from the hand's exposed loop_specs + shell_weld_data."""
        h = self.hand
        dev = str(self.device)
        self._spr_nbody = self.model.body_count // self.num_envs
        b1, b2, a1, a2 = [], [], [], []
        for (lk, a_link, bb, a_bone) in h.loop_specs:            # 3 +X loop closures
            b1.append(int(lk)); b2.append(int(bb))
            a1.append(np.asarray(a_link, np.float32)); a2.append(np.asarray(a_bone, np.float32))
        wd = getattr(h, "shell_weld_data", None)                 # the shell hinge (point spring)
        if wd is not None:
            R1 = np.array(wp.quat_to_matrix(wp.quat(*[float(x) for x in h._shell1_quat]))).reshape(3, 3)
            C = np.asarray(wd["center"], np.float64)
            b1.append(int(wd["body1"])); b2.append(int(wd["body2"]))
            a1.append((R1.T @ (C - np.asarray(h._shell1_pos, np.float64))).astype(np.float32))
            a2.append((C - np.asarray(wd["shell2_pos"], np.float64)).astype(np.float32))
        self._spr_n = len(b1)
        self._spr_b1 = wp.array(np.array(b1, np.int32), dtype=wp.int32, device=dev)
        self._spr_b2 = wp.array(np.array(b2, np.int32), dtype=wp.int32, device=dev)
        self._spr_a1 = wp.array(np.stack(a1), dtype=wp.vec3, device=dev)
        self._spr_a2 = wp.array(np.stack(a2), dtype=wp.vec3, device=dev)
        self._spr_k = wp.array(np.full(self._spr_n, LOOP_SPRING_KE, np.float32), dtype=wp.float32, device=dev)
        self._spr_d = wp.array(np.full(self._spr_n, LOOP_SPRING_KD, np.float32), dtype=wp.float32, device=dev)
        self._body_com = self.model.body_com
        if self.verbose:
            print(f"[soft_loops] {self._spr_n} penalty springs/world (k={LOOP_SPRING_KE:.0e}, "
                  f"d={LOOP_SPRING_KD:.0e}, fmax={LOOP_SPRING_FMAX:.0f})", flush=True)

    def _apply_loop_springs(self):
        wp.launch(_loop_spring_k, dim=self.num_envs * self._spr_n,
                  inputs=[self.state_0.body_q, self.state_0.body_qd, self._body_com,
                          self._spr_nbody, self._spr_n, self._spr_b1, self._spr_b2,
                          self._spr_a1, self._spr_a2, self._spr_k, self._spr_d,
                          float(LOOP_SPRING_FMAX), self.state_0.body_f])

    def _substep_loop(self):
        """The pure physics substep loop — Warp-only (no host syncs, no torch), so it can be
        captured into a CUDA graph and replayed. The velocity watchdog is a Warp kernel."""
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            if self._soft_loops:
                self._apply_loop_springs()           # penalty loop closures -> body_f (before step)
            self.model.collide(self.state_0, self.contacts, collision_pipeline=self._pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            # watchdog: arrest a diverging world before its velocity overflows to inf/NaN.
            # The ceiling is far above any nominal motion, so well-behaved worlds are
            # untouched (no quality loss); only a blowup gets clamped to finite, giving the
            # solver a chance to recover instead of cascading the whole batch to NaN.
            wp.launch(_clamp_f32_k, dim=self.state_0.joint_qd.shape[0],
                      inputs=[self.state_0.joint_qd, float(VEL_CLAMP)])
            wp.launch(_clamp_spatial_k, dim=self.state_0.body_qd.shape[0],
                      inputs=[self.state_0.body_qd, float(BVEL_CLAMP)])

    def _refresh_views(self):
        # state refs are stable after capture (even substeps -> state_0 ends on its start buffer).
        self.jq = wp.to_torch(self.state_0.joint_q)
        self.jqd = wp.to_torch(self.state_0.joint_qd)
        self.bq = wp.to_torch(self.state_0.body_q)
        self.bqd = wp.to_torch(self.state_0.body_qd)

    def _graph_warn(self, exc):
        bar = "!" * 74
        print(f"\n{bar}\n[GRAPH CAPTURE FAILED] _step_sim is running EAGER (slower, more launch "
              f"overhead).\n  reason: {type(exc).__name__}: {exc}\n  -> pass use_graph=False to "
              f"silence, or fix the capture-breaking op above.\n{bar}\n", flush=True)

    def _step_sim(self):
        # Double-buffer ping-pong + CUDA graph only lines up for EVEN substeps (the graph's
        # input buffer == its output buffer); odd substeps run eager (loud, once).
        if not self._use_graph or self.substeps % 2 == 1:
            if self._use_graph and self._graph_ok is None and self.substeps % 2 == 1:
                self._graph_ok = False
                self._graph_warn(RuntimeError(f"substeps={self.substeps} is odd; graph needs even substeps"))
            self._substep_loop(); self._refresh_views(); return
        if self._graph is None and self._graph_ok is not False:
            if not self._warmed:                      # one eager step first to trigger lazy allocations
                self._warmed = True
                self._substep_loop(); self._refresh_views(); return
            try:
                with wp.ScopedCapture() as cap:
                    self._substep_loop()
                self._graph = cap.graph; self._graph_ok = True
                if self.verbose:
                    print(f"[graph] captured _step_sim (substeps={self.substeps}) -> replaying", flush=True)
            except Exception as exc:                  # capture-breaking op -> LOUD eager fallback
                self._graph_ok = False
                self._graph_warn(exc)
        if self._graph_ok:
            wp.capture_launch(self._graph)
            self._refresh_views(); return
        self._substep_loop(); self._refresh_views()

    def _apply_action(self, actions):
        # Residual around the cupped hold pose: a=0 holds the cube (stable), a=+/-1
        # reaches +/- half the joint range. Centering on the hold pose (vs absolute
        # scale-to-limits) keeps the stiff closed-loop hand stable and is far more
        # sample-efficient than starting every action from joint mid-range.
        a = torch.clamp(actions, -1.0, 1.0)
        self.actions = a
        tgt = self.finger_cupped + a * ACT_DELTA
        tgt = torch.clamp(tgt, self.finger_lo, self.finger_hi)
        tgt = ACT_MOVING_AVG * tgt + (1.0 - ACT_MOVING_AVG) * self.prev_targets
        self.prev_targets = tgt
        self.tgt[self.finger_dof_idx] = tgt

    def _compute_obs(self):
        cube_pos, cube_rot = self._cube_pose()
        finger_q = self.jq[self.finger_q_idx]                       # [N,22]
        finger_qd = self.jqd[self.finger_dof_idx]
        unscaled = 2.0 * (finger_q - self.finger_lo) / (self.finger_hi - self.finger_lo).clamp_min(1e-6) - 1.0
        cube_w = self.bqd[self.cube_body_idx][:, :3]                # angvel
        cube_v = self.bqd[self.cube_body_idx][:, 3:6]               # linvel
        tactile = self.sensor.pad_totals_torch().to(self.device)    # [N,15]
        rel = quat_mul(cube_rot, quat_conj(self.goal_quat))
        self.obs_buf = torch.cat([
            unscaled, VEL_OBS_SCALE * finger_qd,
            cube_pos - self.in_hand_pos, cube_rot,
            cube_v, VEL_OBS_SCALE * cube_w,
            self.goal_quat, rel, tactile, self.actions,
        ], dim=-1)
        return self.obs_buf

    def _compute_reward_done(self):
        cube_pos, cube_rot = self._cube_pose()
        rot_dist = rotation_distance(cube_rot, self.goal_quat)
        goal_dist = (cube_pos - self.in_hand_pos).norm(dim=-1)
        dropped = goal_dist >= FALL_DIST
        # DENSE shaping: reward every radian the cube turns TOWARD the goal this step. Potential-
        # based (prev - cur), so it telescopes to (initial - final) over an episode -> no oscillation
        # exploit; gives a real gradient toward rotating (the weak 1/rot_dist term alone did not).
        # Zeroed on drops (the cube leaving the hand is not "progress").
        shaping = ROT_SHAPE_K * (self.prev_rot_dist - rot_dist)
        shaping = torch.where(dropped, torch.zeros_like(shaping), shaping)
        # REACH-AND-HOLD (mastery): being within tolerance gives a dense per-step bonus; a goal is
        # only "achieved" (-> resample + curriculum) once the cube is HELD there for HOLD_STEPS
        # consecutive steps. This makes the policy STABILIZE at the goal instead of merely swinging
        # through it (the single-step-touch resample capped success at ~10% and never taught holding).
        at_goal = rot_dist.abs() <= self.cfg.success_tolerance
        self.hold_counter = torch.where(at_goal, self.hold_counter + 1, torch.zeros_like(self.hold_counter))
        achieved = self.hold_counter >= HOLD_STEPS
        # stability: when at the goal, reward keeping the cube STILL (low angular speed) -> the cube
        # settles instead of jittering in/out of tolerance, so at-goal hovering becomes real holds.
        cube_angvel = self.bqd[self.cube_body_idx][:, 3:6]          # true angular vel (body_qd bottom)
        stability = STAB_K * at_goal.float() * torch.exp(-2.0 * cube_angvel.norm(dim=-1))
        # steep DEEP-PROXIMITY term: strongly reward driving rot_dist toward 0 (not just inside the
        # tolerance) so the cube settles at the CENTER of the goal, well clear of the boundary it
        # was parking on (rot_dist ~0.44 at a 0.4 tol -> constant boundary crossing, no holds).
        deep = 6.0 * torch.exp(-6.0 * rot_dist.abs())
        rew = (shaping + 1.0 / (rot_dist.abs() + 0.1) + deep + AT_GOAL_BONUS * at_goal.float() + stability
               + HOLD_BONUS * achieved.float() - 10.0 * goal_dist - 2e-4 * (self.actions ** 2).sum(-1))
        rew = rew - self._drop_penalty * dropped.float()
        self.successes += achieved.float()                 # count mastered (held) goals
        self.prev_rot_dist = rot_dist                      # carry for next step's shaping
        # curriculum keyed on the AT-GOAL (proximity) rate -> coverage progresses toward any-axis
        self._succ_ema = 0.99 * self._succ_ema + 0.01 * float(at_goal.float().mean())
        if self._succ_ema > CURRIC_THRESH and self.goal_angle < GOAL_ANGLE_MAX:
            self.goal_angle = min(GOAL_ANGLE_MAX, self.goal_angle + CURRIC_STEP)
        # on a held goal: resample near the current orientation (curriculum), reset hold + shaping
        if achieved.any():
            ids = achieved.nonzero(as_tuple=False).flatten()
            self.goal_quat[ids] = rand_quat_near(cube_rot[ids], self.goal_angle, self.device,
                                                 min_angle=self.cfg.success_tolerance)
            self.prev_rot_dist[ids] = rotation_distance(cube_rot[ids], self.goal_quat[ids])
            self.hold_counter[ids] = 0
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        done = dropped | timeout
        rew = rew * self._reward_scale                      # keep returns O(10s) for stable value fit
        # report BOTH the per-step at-goal rate and the hold/achieve rate (mastery)
        return rew, done, dropped, timeout, rot_dist, at_goal, achieved

    # ---- VecEnv API ----
    def get_observations(self):
        return {"policy": self.obs_buf}

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return {"policy": self.obs_buf}
        n = len(env_ids)
        # cube: seated pose + small noise, near-upright + small rot noise
        pos = self.in_hand_pos[env_ids] + 0.005 * (torch.rand(n, 3, device=self.device) - 0.5)
        rot = quat_norm(torch.cat([0.1 * (torch.rand(n, 3, device=self.device) - 0.5),
                                   torch.ones(n, 1, device=self.device)], dim=-1))
        self.jq[self.cube_q_idx[env_ids]] = torch.cat([pos, rot], dim=-1)
        # fingers -> cupped default
        self.jq[self.finger_q_idx[env_ids]] = self.finger_cupped.unsqueeze(0)
        self.prev_targets[env_ids] = self.finger_cupped
        # zero velocities for those worlds (whole per-world qd block, vectorized)
        self.jqd.view(self.num_envs, self.dofs_per_world)[env_ids] = 0.0
        self.goal_quat[env_ids] = rand_quat_near(rot, self.goal_angle, self.device,   # curriculum goal
                                                 min_angle=self.cfg.success_tolerance)  # never spawn pre-solved
        self.episode_length_buf[env_ids] = 0
        self.successes[env_ids] = 0.0
        self.hold_counter[env_ids] = 0.0
        # Clear the SOLVER's per-world dynamic state for the reset worlds. Writing only
        # Newton State leaves mjw_data's qvel/qacc/qacc_warmstart stale; if a world had
        # diverged (NaN), the carried-over warmstart re-corrupts it on the next step and
        # NaN worlds never recover (and accumulate at high world counts). qpos re-syncs
        # from the finite Newton joint_q on the next solver.step.
        if not self._kamino:
            d = self.solver.mjw_data
            for f in ("qvel", "qacc", "qacc_warmstart", "qfrc_applied"):
                arr = getattr(d, f, None)
                if arr is not None:
                    wp.to_torch(arr)[env_ids] = 0.0
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        if self._kamino:
            # kamino has no mjw_data; reinit its per-world warmstart for the reset worlds so a
            # previously-diverged world starts clean (mirrors IsaacLab's kamino_manager.step()).
            if getattr(self, "_kreset_mask", None) is None:
                self._kreset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            m = self._kreset_mask; m.zero_(); m[env_ids] = True
            self.solver.reset(self.state_0, world_mask=wp.from_torch(m))
        self.bq = wp.to_torch(self.state_0.body_q)
        # shaping baseline for the reset worlds: rot_dist from the new cube pose to the new goal
        cube_pos, cube_rot = self._cube_pose()
        self.prev_rot_dist[env_ids] = rotation_distance(cube_rot[env_ids], self.goal_quat[env_ids])
        self.sensor.update()
        self._compute_obs()
        # just-reset worlds have stale (possibly NaN) mjw_data until the next solver.step
        # re-syncs; sanitize so the policy never sees NaN.
        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=0.0, neginf=0.0)
        return {"policy": self.obs_buf}

    def step(self, actions):
        self._apply_action(actions.to(self.device))
        self._step_sim()
        self.sensor.update()
        self.episode_length_buf += 1
        rew, done, dropped, timeout, rot_dist, at_goal, achieved = self._compute_reward_done()
        # NaN guard: the stiff closed-loop hand can rarely diverge in an individual
        # world under adversarial exploration (failure is confined per-world). Detect over
        # the FULL per-world body state (not just fingers) so a diverged cube/shell is caught
        # too, then treat the world as a drop: terminate + reset it, penalize it (NAN_PENALTY),
        # and sanitize what the policy sees. One bad world never poisons the batch.
        finger_q = self.jq[self.finger_q_idx]
        nb = self.bq.shape[0] // self.num_envs
        body_finite = torch.isfinite(self.bq.view(self.num_envs, nb, 7)).all(dim=(1, 2))
        nan_env = ~(body_finite & torch.isfinite(finger_q).all(dim=1) & torch.isfinite(rew))
        if bool(nan_env.any()):
            done = done | nan_env
            rew = torch.where(nan_env, torch.full_like(rew, NAN_PENALTY), rew)
        extras = {"time_outs": timeout & ~nan_env, "log": {
            "rot_dist": rot_dist[torch.isfinite(rot_dist)].mean(), "success_rate": at_goal.float().mean(),
            "hold_rate": achieved.float().mean(), "consecutive_successes": self.successes.mean(),
            "dropped": dropped.float().mean(), "nan_envs": nan_env.float().mean(),
            "goal_angle": torch.tensor(self.goal_angle, device=self.device)}}
        self._compute_obs()
        done_ids = done.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            self.reset(done_ids)
        rew = torch.nan_to_num(rew, nan=NAN_PENALTY, posinf=0.0, neginf=NAN_PENALTY)
        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=0.0, neginf=0.0)
        # bool dones (CfCPPO does torch.where(dones, ...); the MLP trainer casts into its float buffer)
        return {"policy": self.obs_buf}, rew, done, extras


def _check(num_envs=4, steps=200):
    import time
    wp.init()
    env = FlexPalmReorientEnv(num_envs=num_envs)
    obs = env.reset()["policy"]
    assert obs.shape == (num_envs, OBS_DIM), obs.shape
    print(f"[check] obs {tuple(obs.shape)} finite={bool(torch.isfinite(obs).all())}")
    cp0, _ = env._cube_pose()
    print(f"[check] cube rest pos (env0) {cp0[0].cpu().numpy().round(3)}  (target {CUBE_POS})")
    peak_tac = torch.zeros(num_envs, N_PADS, device=env.device)
    t0 = time.time()
    for i in range(steps):
        act = 0.3 * torch.randn(num_envs, N_FINGER, device=env.device)
        obs, rew, done, extras = env.step(act)
        peak_tac = torch.maximum(peak_tac, env.sensor.pad_totals_torch())
        if i % 50 == 0:
            fin = bool(torch.isfinite(obs["policy"]).all() and torch.isfinite(rew).all())
            print(f"  step {i:3d}: rew {rew.mean().item():+.2f}  rot_dist {extras['log']['rot_dist'].item():.2f}  "
                  f"dropped {extras['log']['dropped'].item():.2f}  finite={fin}")
    dt = time.time() - t0
    cube_now, _ = env._cube_pose()
    held = ((cube_now - env.in_hand_pos).norm(dim=-1) < FALL_DIST).float().mean().item()
    tac_fire = (peak_tac.max(dim=1).values > 0.02).float().mean().item()
    print(f"[check] {steps} steps in {dt:.1f}s ({num_envs*steps/dt:.0f} env-steps/s)")
    print(f"[check] tactile fired (>0.02N on >=1 pad) in {tac_fire*100:.0f}% of envs; peak pad max {peak_tac.max().item():.2f} N")
    print(f"[check] cube still in-hand after random actions: {held*100:.0f}% of envs")
    ok = bool(torch.isfinite(obs["policy"]).all()) and tac_fire > 0.5
    print(f"[check] RESULT: {'PASS' if ok else 'FAIL'} (finite + tactile firing)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=200)
    a = p.parse_args()
    if a.check:
        _check(a.num_envs, a.steps)
    else:
        print("use --check (training: train_reorient.py)")
