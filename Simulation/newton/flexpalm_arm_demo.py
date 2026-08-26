# Demo: the flexible robot hand bolted to a Franka FR3 arm flange, reaching down to
# grasp an object, with the tactile-sensing capability (from robot_hand_tactile.py /
# _hand_taxels.py) shown as an IN-VIEWER fingertip heatmap (fingertips light up by
# contact force). Pure Newton + SolverMuJoCo, `isaac` env.
#
# Launch needs the cu13 NVRTC on the loader path (tactile kernel JIT):
#   LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib"
#
#   python flexpalm_arm_demo.py --check          # headless: finalize + grasp + tactile fires
#   python flexpalm_arm_demo.py --viewer gl      # GL window
import os
import sys
import math
import numpy as np
import warp as wp
import newton
import newton.examples
import newton.utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flexpalm_bones4 as fb                       # noqa: E402  (the hand)
from hand_taxels_newton import HandTaxelSensor      # noqa: E402  (tactile)
import arms                                          # noqa: E402  (franka / dorna arm builders)

ARM_DOFS = 7
# Franka "ready" pose (panda_hydro): hand presented forward/down.
INIT_Q = [-3.68e-3, 2.39e-2, 3.68e-3, -2.368, -1.29e-4, 2.392, 0.7855]
# cover_back (the backplate) plate geometry in the hand-root (pinky) frame, measured
# by PCA over its collision mesh (outward normal = thin axis; center = mesh centroid).
COVER_BACK_NORMAL = (-0.992, 0.121, -0.048)
COVER_BACK_CENTER = (-0.081, 0.0396, -0.0428)


def _q2R(q):
    """quat xyzw -> 3x3 rotation."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float64)


def _R2q(R):
    """3x3 rotation -> quat xyzw."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s; x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return (x, y, z, w)


def _flange_rotation(urdf, base_pose, init_q):
    """Pre-pass: build the arm alone and return the flange's world rotation (3x3)
    at the init pose, so we can analytically orient the mounted hand."""
    ab = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(ab)
    ab.add_urdf(urdf, xform=base_pose, floating=False, enable_self_collisions=False)
    for i in range(ARM_DOFS):
        ab.joint_q[i] = init_q[i]
    am = ab.finalize()
    st = am.state()
    newton.eval_fk(am, am.joint_q, am.joint_qd, st)
    fi = next(i for i, l in enumerate(am.body_label) if l.endswith("/fr3_link8"))
    return _q2R(am.state().body_q.numpy()[fi][3:7] if False else st.body_q.numpy()[fi][3:7])


def _heat(base, frac):
    """Light a pad up by contact: keep its natural color at 0 force, ramp through
    yellow to red as frac->1 (so untouched fingers look normal, pressed ones glow)."""
    f = float(np.clip(frac, 0.0, 1.0))
    base = np.asarray(base, dtype=np.float32)
    if f <= 1e-3:
        return base
    yellow = np.array([1.0, 0.85, 0.0], dtype=np.float32)
    red = np.array([1.0, 0.05, 0.0], dtype=np.float32)
    if f < 0.5:
        return base + (yellow - base) * (f / 0.5)
    return yellow + (red - yellow) * ((f - 0.5) / 0.5)


class Demo:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = int(getattr(args, "substeps", 4))   # fewer -> faster wall-clock
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.args = args
        self.force_max = float(getattr(args, "force_max", 3.0))

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_ground_plane()

        # ---- arm (fixed base): franka FR3 or the user's Dorna 2 ----
        # arms.build_arm sets the ready pose, PD gains, joint limits and (for the
        # Dorna) re-roots the USD articulation at the base + applies hollow-6xxx-
        # aluminum mass/inertia tuned to the real 5.5 kg arm weight.
        spec = arms.build_arm(builder, str(getattr(args, "arm", "franka")), verbose=bool(getattr(args, "check", False)))
        self.spec = spec
        self._arm_shape_lo = spec.shape_lo
        self._arm_shape_hi = spec.shape_hi
        flange = spec.flange
        self.flange = flange
        # Spawn the arm already in the RAISED settle pose (lift applied to the lift
        # dof) so the OPEN hand starts clear ABOVE the cube. Otherwise the arm spawns
        # at the grasp pose with the spreading fingers intersecting the cube -> at the
        # first step the contact solver flings it off the pedestal before the grasp.
        _lift0 = float(args.lift_amt if getattr(args, "lift_amt", None) is not None else spec.lift_amt)
        _ldof = spec.arm_dofs[spec.lift_dof]
        _raised = float(spec.init_q[spec.lift_dof] + spec.lift_sign * _lift0)
        builder.joint_q[_ldof] = _raised
        builder.joint_target_pos[_ldof] = _raised

        # ---- the flexible hand, welded to the flange ----
        # MOUNT seats the hand on the flange; tune with --m* flags in the viewer.
        # MECHANICAL MATE of cover_back (the backplate) onto the flange face. Computed
        # purely in the flange-local frame from cover_back's measured plate geometry in
        # the hand-root (pinky) frame -> the plate seats FLAT (its normal anti-parallel
        # to the flange face) and CENTERED (plate center on the flange axis). The fingers
        # then extend along the flange tool axis (down in this arm pose). mrz spins the
        # hand about the tool axis (palm orientation); mrx/mry are tiny extra tilt.
        n_p = np.array(COVER_BACK_NORMAL, dtype=np.float64); n_p /= np.linalg.norm(n_p)
        c_p = np.array(COVER_BACK_CENTER, dtype=np.float64)
        tgt = np.array([0.0, 0.0, -1.0])                      # flange-local -Z (into the flange)
        d = float(np.clip(n_p @ tgt, -1.0, 1.0))
        ax = np.cross(n_p, tgt); axn = float(np.linalg.norm(ax))
        if axn < 1e-9:
            q1 = wp.quat_identity() if d > 0 else wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi)
        else:
            q1 = wp.quat_from_axis_angle(wp.vec3(*(ax / axn)), math.acos(d))
        mrz = args.mrz if getattr(args, "mrz", None) is not None else spec.mrz
        rz = math.radians(float(mrz))
        q = wp.mul(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, -1.0), rz), q1)
        rx = math.radians(float(getattr(args, "mrx", 0.0))); ry = math.radians(float(getattr(args, "mry", 0.0)))
        if rx:
            q = wp.mul(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), rx), q)
        if ry:
            q = wp.mul(wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), ry), q)
        qc = np.array([float(x) for x in wp.quat_rotate(q, wp.vec3(*c_p))])
        z_seat = 0.007 + float(spec.seat_dz)
        t = np.array([float(getattr(args, "mdx", 0.0)), float(getattr(args, "mdy", 0.0)),
                      z_seat + float(getattr(args, "mdz", 0.0))], dtype=np.float64) - qc
        mount = wp.transform(wp.vec3(*t), q)
        self.mount = mount   # hand-relative-to-flange (rigid); hand rest frame = flange o mount
        args.lift = 0.0      # no world-clearance lift; the arm holds the hand up
        self.hand = fb.Example(None, args, ext_builder=builder, attach_body=flange,
                               attach_xform=mount, finalize=False)

        # ---- graspable object: a FREE cube on a thin fixed pedestal, placed in the
        # hand's CLOSING ARC -- the pocket the down-pointing fingers curl into as they
        # close (measured: fingers sweep from z~0.06 open up to z~0.15 closed, toward
        # +Y/-X). The open hand clears the cube; closing wraps it -> multi-pad contact.
        # A free cube keeps grasp forces bounded (it can shift) so the demo stays stable.
        self.cube_half = float(args.cube if getattr(args, "cube", None) is not None else spec.cube_half)
        # Place the cube in the hand's rigid rest frame (flange o mount) at the universal
        # GRASP_POCKET derived from the working Franka grasp. This reproduces the same
        # hand-relative grasp on any arm (absorbing the per-flange mount-azimuth difference),
        # so the open hand descends cleanly AROUND the cube instead of swallowing it.
        flange_ready, _ = arms.flange_world_at_ready(str(getattr(args, "arm", "franka")))
        hand_rest = wp.transform_multiply(flange_ready, self.mount)
        pk = np.array([float(x) for x in wp.transform_point(hand_rest, wp.vec3(*arms.GRASP_POCKET))])
        cube_pos = (float(args.cubex if getattr(args, "cubex", None) is not None else pk[0]),
                    float(args.cubey if getattr(args, "cubey", None) is not None else pk[1]),
                    float(args.cubez if getattr(args, "cubez", None) is not None else pk[2]))
        # fixed pedestal: ground up to just under the cube (same group as the cube
        # so it supports it). Filtered against the hand below, so it never fights fingers.
        # Its top is ~as wide as the cube so a 6 cm cube sits stably (a narrow post lets
        # the cube tip off under the slightest nudge from the closing hand).
        cz_bot = cube_pos[2] - self.cube_half
        ped_hw = float(args.ped_hw if getattr(args, "ped_hw", None) is not None else spec.ped_hw)
        ped_cfg = newton.ModelBuilder.ShapeConfig()
        ped_cfg.density = 500.0; ped_cfg.has_shape_collision = True; ped_cfg.collision_group = 7
        ped_body = builder.add_link(xform=wp.transform(
            wp.vec3(cube_pos[0], cube_pos[1], 0.5 * cz_bot), wp.quat_identity()), label="pedestal")
        self.ped_shape = builder.add_shape_box(ped_body, hx=ped_hw, hy=ped_hw,
                                               hz=0.5 * cz_bot, cfg=ped_cfg)
        pj = builder.add_joint_fixed(parent=-1, child=ped_body,
                                     parent_xform=wp.transform(wp.vec3(cube_pos[0], cube_pos[1], 0.5 * cz_bot),
                                                               wp.quat_identity()),
                                     child_xform=wp.transform(q=wp.quat_identity()), label="fix_ped")
        builder.add_articulation([pj], label="pedestal")
        # free cube on top
        self.cube_body = builder.add_link(
            xform=wp.transform(wp.vec3(*cube_pos), wp.quat_identity()), label="cube")
        cube_cfg = newton.ModelBuilder.ShapeConfig()
        cube_cfg.density = float(args.cube_density if getattr(args, "cube_density", None) is not None
                                 else spec.cube_density); cube_cfg.has_shape_collision = True
        cube_cfg.collision_group = 7        # collides with hand(-1) and pedestal
        self.cube_shape = builder.add_shape_box(self.cube_body, hx=self.cube_half, hy=self.cube_half,
                                                hz=self.cube_half, cfg=cube_cfg)
        # parent_xform = IDENTITY: the free joint's coords auto-init from the body's
        # add-xform (cube_pos), so a cube_pos parent_xform would DOUBLE the position.
        cj = builder.add_joint_free(parent=-1, child=self.cube_body,
                                    parent_xform=wp.transform(q=wp.quat_identity()),
                                    label="free_cube")
        builder.add_articulation([cj], label="cube")

        # The arm is a kinematic positioner: filter it against the hand + cube so the
        # flange/links never collide with the hand they carry (that was the NaN), and
        # the arm doesn't bump the object. The HAND<->cube pair stays active (the grasp).
        arm_shapes = list(range(self._arm_shape_lo, self._arm_shape_hi))
        non_arm = list(self.hand.coll_shapes) + [self.cube_shape]
        if self.ped_shape is not None:
            non_arm.append(self.ped_shape)
        for a in arm_shapes:
            for b in non_arm:
                builder.add_shape_collision_filter_pair(a, b)
        # pedestal (if present) only supports the cube -> don't let it fight the hand
        if self.ped_shape is not None:
            for b in self.hand.coll_shapes:
                builder.add_shape_collision_filter_pair(self.ped_shape, b)

        builder.color()
        self.model = builder.finalize(device="cuda")
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        # implicitfast integrator is required for the stiff arm PD carrying the heavy
        # hand (explicit euler diverges). Bigger contact buffers: arm + hand + cube
        # produce many more simultaneous contacts than the standalone hand. The
        # njmax/nconmax sizing is for mujoco-warp 3.8's convex narrowphase: with the
        # full-CoACD shells (~108 hulls) + arm + cube, the 1.2-era 400/256 overflows
        # its candidate buffers and poisons the CUDA context (illegal memory access
        # at the next module load). 16384/4096 is measured-safe on newton 1.4.
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, iterations=int(getattr(args, "iters", 70)),
            ls_iterations=10, njmax=16384, nconmax=4096)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.model.contacts()

        # wire the hand to the finalized model + build its pose battery
        self.hand.bind(self.model, self.solver, self.state_0, self.state_1, self.control, self.contacts)
        self.hand.setup_drive()

        # ---- arm DOF slots in joint_target_pos (the arm is added first) ----
        self.arm_dof = list(spec.arm_dofs)
        self.arm_init = np.array(spec.init_q, dtype=np.float32)

        # ---- hand grasp endpoints from the pose battery (open <-> fist) ----
        names = self.hand.pose_names
        self.open_vec = self.hand.pose_T[names.index("open")].copy()
        self.fist_vec = self.hand.pose_T[names.index("fist")].copy()

        # ---- tactile sensor + heatmap tables ----
        self.sensor = HandTaxelSensor(self.hand, self.solver, self.model)
        self.pad_shape = np.array([self.hand.fbody_shape[p.link] for p in self.sensor.pads], dtype=np.int64)
        self.base_color = self.model.shape_color.numpy().copy()
        self.base_color[self.cube_shape] = (0.95, 0.45, 0.10)    # the graspable cube: orange
        if self.ped_shape is not None:
            self.base_color[self.ped_shape] = (0.22, 0.22, 0.25)  # pedestal: dark, unobtrusive
        self.model.shape_color.assign(self.base_color)

        # set by __main__ in interactive mode (render() ticks it); --check's bare
        # step() loops never render, so no session is created there
        self.bridge = None

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            # frame the camera on the hand + cube (not the whole model -> arm base)
            tgt = np.array([self.body_pos(self.cube_body)[0], self.body_pos(self.cube_body)[1],
                            self.body_pos(self.cube_body)[2]])
            eye = tgt + np.array([float(args.camx if getattr(args, "camx", None) is not None else spec.cam_off[0]),
                                  float(args.camy if getattr(args, "camy", None) is not None else spec.cam_off[1]),
                                  float(args.camz if getattr(args, "camz", None) is not None else spec.cam_off[2])])
            dirv = tgt - eye
            h = float(np.hypot(dirv[0], dirv[1]))
            try:
                self.viewer.set_camera(pos=wp.vec3(*eye.tolist()),
                                       pitch=float(np.degrees(np.arctan2(dirv[2], h))),
                                       yaw=float(np.degrees(np.arctan2(dirv[1], dirv[0]))))
            except Exception as exc:
                print(f"[WARN] set_camera failed: {exc}", flush=True)
        self.graph = None

    # ---- scheduled reach-grasp-lift ----
    def _schedule(self):
        """Return (arm_target[7], grasp_frac). Arm holds the ready pose presenting the
        hand at the cube; the hand opens, then closes the grasp and holds. (Arm reach
        motion is layered on once the grasp+tactile is solid.)"""
        t = self.sim_time
        a = self.arm_init.copy()
        gmax = float(getattr(self.args, "grasp_max", 0.85))   # firm close
        t_settle = float(getattr(self.args, "settle", 0.2))   # brief settle, raised+open
        t_app = float(getattr(self.args, "approach_dur", 0.8))  # descend to the cube
        tc = float(getattr(self.args, "close_dur", 0.9))      # close duration (slow enough to cage, not fling)
        t_hold = float(getattr(self.args, "hold_dur", 0.3))   # hold grip before lifting
        t_lift = float(getattr(self.args, "lift_dur", 1.0))   # lift duration
        lift_amt = float(self.args.lift_amt if getattr(self.args, "lift_amt", None) is not None
                         else self.spec.lift_amt)             # APPROACH clearance (small -> near-vertical descent)
        lift_raise = float(getattr(self.spec, "lift_raise", lift_amt))  # final raise (cube enclosed -> tilt ok)

        def smooth(x):
            x = min(max(x, 0.0), 1.0)
            return x * x * (3.0 - 2.0 * x)

        t_app_end = t_settle + t_app
        t_close_end = t_app_end + tc
        t_lift_start = t_close_end + t_hold
        # arm_lift: shoulder raise above the grasp pose. Start RAISED+open, descend to the
        # cube (approach), close, hold, then raise again to pick it up.
        if t < t_settle:                    # poised above the cube, hand open
            arm_lift = lift_amt; g = 0.0
        elif t < t_app_end:                 # APPROACH: lower the hand onto the cube
            arm_lift = lift_amt * (1.0 - smooth((t - t_settle) / t_app)); g = 0.0
        elif t < t_close_end:               # GRASP: close firmly on the cube
            arm_lift = 0.0; g = min((t - t_app_end) / tc, 1.0) * gmax
        elif t < t_lift_start:              # hold the grip
            arm_lift = 0.0; g = gmax
        else:                               # LIFT: raise the cube off the pedestal
            arm_lift = lift_raise * smooth((t - t_lift_start) / t_lift); g = gmax
        ld = self.spec.lift_dof
        a[ld] = self.arm_init[ld] + self.spec.lift_sign * arm_lift
        return a, g

    def _drive(self):
        if self.control.joint_target_pos is None:
            return
        arm_t, g = self._schedule()
        gg = g * g * (3.0 - 2.0 * g)
        target = (1.0 - gg) * self.open_vec + gg * self.fist_vec   # hand dofs (arm dofs are 0 here)
        target[self.arm_dof] = arm_t                                # overwrite arm dofs
        self.control.joint_target_pos.assign(np.ascontiguousarray(target, dtype=np.float32))

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _recolor(self):
        self.sensor.update()
        f = self.sensor.pad_totals_np()                  # (P,) normal force per pad [N]
        cols = self.base_color.copy()
        for i, shp in enumerate(self.pad_shape):
            cols[shp] = _heat(self.base_color[shp], f[i] / self.force_max)
        self.model.shape_color.assign(cols)

    def step(self):
        self._drive()
        self.simulate()
        self._recolor()
        self.sim_time += self.frame_dt

    def render(self):
        if self.bridge is not None:
            self.bridge.tick()   # examples.run calls render() every iteration, paused or not
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    # ---- helpers ----
    def body_pos(self, idx):
        return self.state_0.body_q.numpy()[idx][:3].copy()


def _add_demo_args(parser):
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--arm", choices=["franka", "dorna"], default="franka",
                        help="which arm carries the hand (default: franka)")
    parser.add_argument("--lift-amt", dest="lift_amt", type=float, default=None,
                        help="lift-dof delta [rad] (default: per-arm)")
    # mr* are EXTRA fine-tune spins on top of the analytic fingers-down mount (0 = none).
    parser.add_argument("--mrx", type=float, default=0.0); parser.add_argument("--mry", type=float, default=0.0)
    parser.add_argument("--mrz", type=float, default=None, help="palm spin about wrist axis [deg] (default: per-arm)")
    # md* nudge the seated plate off the flange axis/face (0 = mechanical-mate default).
    parser.add_argument("--mdx", type=float, default=0.0); parser.add_argument("--mdy", type=float, default=0.0)
    parser.add_argument("--mdz", type=float, default=0.0)
    parser.add_argument("--cube", type=float, default=None, help="cube half-extent [m] (default: per-arm)")
    parser.add_argument("--cube-density", dest="cube_density", type=float, default=None)
    parser.add_argument("--ped-hw", dest="ped_hw", type=float, default=None, help="pedestal half-width [m]")
    # default cube pos = the fist convergence (measured): where the 4 fingertips meet
    # when closed, so the closing fingers squeeze it from all sides (a small/heavy cube
    # there is trapped rather than swept past). Per-arm default; override with --cube[xyz].
    parser.add_argument("--cubex", type=float, default=None); parser.add_argument("--cubey", type=float, default=None)
    parser.add_argument("--cubez", type=float, default=None)
    parser.add_argument("--camx", type=float, default=None)
    parser.add_argument("--camy", type=float, default=None)
    parser.add_argument("--camz", type=float, default=None)
    parser.add_argument("--force-max", dest="force_max", type=float, default=0.6, help="heatmap saturation force [N]")
    # hand args (mirror flexpalm_bones4 so fb.Example reads them)
    parser.add_argument("--kspring", type=float, default=2.0); parser.add_argument("--damping", type=float, default=0.25)
    parser.add_argument("--theta", type=float, default=35.0); parser.add_argument("--fingerke", type=float, default=0.4)
    parser.add_argument("--grasp-max", dest="grasp_max", type=float, default=0.85, help="max grasp close fraction")
    parser.add_argument("--hullverts", type=int, default=256)
    parser.add_argument("--hold", type=float, default=1.6); parser.add_argument("--blend", type=float, default=0.9)
    parser.add_argument("--curl-omega", dest="curl_omega", type=float, default=1.2)
    parser.add_argument("--no-curl", dest="no_curl", action="store_true")
    return parser


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    _add_demo_args(parser)
    args, _ = parser.parse_known_args()

    if args.check:
        wp.init()
        N = int(getattr(args, "frames", 1400))   # long enough to catch slow drift
        # ---- honesty gate: CLEAN OPEN DESCENT. Run the approach with the hand forced
        # OPEN (grasp_max=0) and require the cube to barely move -- i.e. the open hand
        # descends AROUND the cube rather than swallowing/knocking it. A cube that gets
        # displaced here was trapped inside the shell, so a later "lift" is fake.
        import copy as _copy
        oargs = _copy.copy(args); oargs.grasp_max = 0.0
        oex = Demo(newton.viewer.ViewerNull(num_frames=200), oargs)
        c0 = oex.state_0.body_q.numpy()[oex.cube_body][:3].copy()
        f_close = int((float(getattr(oargs, "settle", 0.2)) + float(getattr(oargs, "approach_dur", 0.8))) * oex.fps)
        for f in range(f_close + 5):
            oex.step()
        descent_disp = float(np.linalg.norm(oex.state_0.body_q.numpy()[oex.cube_body][:3] - c0))
        clean_descent = descent_disp < 0.015
        print(f"[check] open-hand descent disturbed cube by {descent_disp*1000:.0f} mm  "
              f"(clean-approach:{'PASS' if clean_descent else 'FAIL -- cube swallowed/knocked'})")
        del oex

        ex = Demo(newton.viewer.ViewerNull(num_frames=N + 10), args)
        peak = np.zeros(len(ex.sensor.pads))
        finite = True
        # frame where the LIFT phase begins (settle+approach+close+hold), so we can
        # compare the cube height grasped-on-pedestal vs after the lift completes.
        a = ex.args
        t_lift_start = (float(getattr(a, "settle", 0.2)) + float(getattr(a, "approach_dur", 0.8))
                        + float(getattr(a, "close_dur", 0.9)) + float(getattr(a, "hold_dur", 0.3)))
        f_prelift = int(t_lift_start * ex.fps)
        z_prelift = None
        for f in range(N):
            ex.step()
            peak = np.maximum(peak, ex.sensor.pad_totals_np())
            if f == f_prelift:
                z_prelift = float(ex.state_0.body_q.numpy()[ex.cube_body][2])
            if f % 50 == 0:
                finite = finite and bool(np.all(np.isfinite(ex.state_0.body_q.numpy())))
        z_final = float(ex.state_0.body_q.numpy()[ex.cube_body][2])
        lift = (z_final - z_prelift) if z_prelift is not None else float("nan")
        names = [p.name for p in ex.sensor.pads]
        print(f"[check] bodies={ex.model.body_count} dofs={ex.model.joint_dof_count} "
              f"eq={ex.model.equality_constraint_count} pads={len(ex.sensor.pads)}")
        order = np.argsort(peak)[::-1]
        print("[check] peak per-pad normal force [N]: " +
              "  ".join(f"{names[i]}:{peak[i]:.2f}" for i in order[:6]))
        n_fired = int((peak > 0.05).sum())
        thumb_fired = any(peak[i] > 0.05 for i, p in enumerate(ex.sensor.pads) if "thumb" in p.name)
        print(f"[check] cube z: pre-lift={z_prelift:.3f} -> final={z_final:.3f}  (lifted {lift*1000:+.0f} mm)")
        lifted = lift > 0.02
        real_grasp = clean_descent and lifted and n_fired >= 2 and finite
        print(f"[check] finite:{'PASS' if finite else 'FAIL'} | "
              f"tactile fired on {n_fired} pads (thumb:{'yes' if thumb_fired else 'no'}) | "
              f"grasp-contact(>=2 pads):{'PASS' if n_fired >= 2 else 'FAIL'} | "
              f"cube-lifted(>2cm):{'PASS' if lifted else 'FAIL'}")
        print(f"[check] REAL GRASP (clean approach + contact + lift): {'PASS' if real_grasp else 'FAIL'}")
    else:
        viewer, args = newton.examples.init(parser)
        # auto-run: the GL viewer starts paused by default, which would freeze the
        # hand at its t=0 open pose so the reach+grasp never plays. Start unpaused.
        if hasattr(viewer, "_paused"):
            viewer._paused = False
        demo = Demo(viewer, args)
        try:  # optional live-inspection bridge (private tool; not needed to run)
            from simbridge import Bridge
            demo.bridge = Bridge.attach(model=demo.model, state_fn=lambda: demo.state_0,
                                        viewer=viewer, name="flexpalm-arm",
                                        extras={"contacts": demo.contacts})
        except ImportError:
            pass
        newton.examples.run(demo, args)
