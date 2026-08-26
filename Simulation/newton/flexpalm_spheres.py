# Semi-flexible palm via SPHERE-PROXY cone joints (Newton + MuJoCo).
#
# Each split palm bone gets a small sphere body welded rigidly to it at the
# bone's centroid (a clean connection point). The spheres are coupled into the
# user's lattice with NATIVE add_joint_d6 cone joints (3 angular axes each
# limited to +/-THETA_MAX with a centering spring) -- the real restricted
# spherical joint, solved IMPLICITLY by MuJoCo so it is unconditionally stable
# (unlike an external body_f penalty). 3 loop edges close via equality CONNECT.
# Pinky base + shells stay the frozen anchor (Palm_rigid).
#
#   python flexpalm_spheres.py --check       # headless numeric check
#   python flexpalm_spheres.py --viewer gl   # GL window
import os, math, argparse
import numpy as np, warp as wp
import newton, newton.examples
import newton_compat  # noqa: F401  (1.2->1.4 equality API shim)
from newton.solvers import SolverMuJoCo
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from pxr import Usd, UsdGeom, Gf

# Assets ship in ../assets of the DexiGrab repo; override with DEXIGRAB_ASSETS.
_ASSETS = os.environ.get("DEXIGRAB_ASSETS") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "assets")
USD = os.path.join(_ASSETS, "robot_hand_flexpalm.usda")
PR  = "/Robotic_Hand_V5_simulacra/Palm_rigid"

# bone id -> body-label suffix ; "pinky" = the frozen Palm_rigid anchor
BONE = {
    "ring": "Base_Bone_12_V02_ringfing", "mid": "Base_Bone_12_V02_middlefinger",
    "ptr": "Base_Bone_12_V02_pointerfinger_and_thumb_attachment",
    "S00": "Palm_bone1/Palm_bone1", "S01": "Palm_bone1/Palm_bone1_01",
    "S03": "Palm_bone1/Palm_bone1_03", "S04": "Palm_bone1/Palm_bone1_04",
    "pinky": "Palm_rigid",
}
FLEX = ["ring", "mid", "ptr", "S00", "S01", "S03", "S04"]
# spanning tree (parent, child) rooted at frozen pinky, + loop closures
TREE = [("pinky","ring"),("ring","S04"),("S04","S00"),("S00","S01"),
        ("S01","ptr"),("ptr","mid"),("S04","S03")]
LOOPS = [("mid","ring"),("mid","S00"),("pinky","S03")]

# Native (implicit) joints -> unconditionally stable, so springs can be stiff.
THETA_MAX = math.radians(15.0)   # cone half-angle
K_SPRING  = 5.0e1                # centering spring [N*m/rad]
DAMPING   = 2.0e0                # joint damping
LIMIT_KE  = 5.0e2                # cone-wall stiffness
LIMIT_KD  = 5.0e0
SPH_R     = 0.006                # connector sphere radius
SPH_M     = 0.005                # connector sphere mass
EQ_TIMECONST = 5.0e-3            # weld/loop equality stiffness (>= 2*dt); default 0.02 too soft


def mesh_centroid(stage, primpath, xc):
    prim = stage.GetPrimAtPath(primpath); pts = []
    for d in Usd.PrimRange(prim):
        if d.GetTypeName() == "Mesh":
            mat = xc.GetLocalToWorldTransform(d)
            for v in UsdGeom.Mesh(d).GetPointsAttr().Get() or []:
                w = mat.Transform(Gf.Vec3d(*v)); pts.append((w[0], w[1], w[2]))
    return np.array(pts).mean(0)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer; self.args = args
        self.frame_dt = 1.0 / 120.0; self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps; self.sim_time = 0.0

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(USD, schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])
        labels = list(builder.body_label)

        def bidx(suffix):
            for i, l in enumerate(labels):
                if l.endswith("/" + suffix) or l.endswith(suffix): return i
            raise KeyError(suffix)
        bone_body = {k: bidx(v) for k, v in BONE.items()}

        # world centroids of every lattice node (clean connection points)
        stage = Usd.Stage.Open(USD); xc = UsdGeom.XformCache()
        cen = {k: mesh_centroid(stage, PR + ("" if BONE[k] == "Palm_rigid" else "/" + BONE[k]), xc)
               for k in BONE}

        sphere_cfg = newton.ModelBuilder.ShapeConfig()
        sphere_cfg.density = 5000.0              # gives sphere mass + valid inertia
        sphere_cfg.has_shape_collision = False   # connectors don't collide

        art_joints = []   # joints that form the sphere-lattice articulation

        # frozen lattice root: an anchor sphere fixed to the world at the pinky
        # point (Palm_rigid is itself FixedBase to world there, so equivalent).
        pa = builder.add_link(xform=wp.transform(wp.vec3(*cen["pinky"]), wp.quat_identity()),
                              label="sph_pinky_anchor")
        sphere_cfg0 = newton.ModelBuilder.ShapeConfig()
        sphere_cfg0.density = 0.0; sphere_cfg0.has_shape_collision = False
        builder.add_shape_sphere(pa, radius=SPH_R, cfg=sphere_cfg0)
        art_joints.append(builder.add_joint_fixed(
            parent=-1, child=pa,
            parent_xform=wp.transform(wp.vec3(*cen["pinky"]), wp.quat_identity()),
            child_xform=wp.transform(q=wp.quat_identity()), label="freeze_pinky_anchor"))

        # one sphere body per flex bone, at the bone centroid, welded to the bone.
        # add_link (NOT add_body) so it gets no auto free joint -- only our D6.
        self.node_body = {"pinky": pa}
        for k in FLEX:
            c = cen[k]
            s = builder.add_link(xform=wp.transform(wp.vec3(*c), wp.quat_identity()),
                                 label=f"sph_{k}")
            builder.add_shape_sphere(s, radius=SPH_R, cfg=sphere_cfg)
            # rigid weld: sphere pose relative to the bone body
            tb = builder.body_q[bone_body[k]]
            pb = np.array([float(tb[j]) for j in range(3)])
            qb = wp.quat(float(tb[3]),float(tb[4]),float(tb[5]),float(tb[6]))
            qbi = wp.quat_inverse(qb)
            relpose = wp.transform(wp.quat_rotate(qbi, wp.vec3(*(c - pb))), qbi)  # sphere has identity orient
            builder.add_equality_constraint_weld(body1=bone_body[k], body2=s,
                anchor=wp.vec3(0.0,0.0,0.0), relpose=relpose, torquescale=1.0, label=f"weld_{k}")
            self.node_body[k] = s

        Cfg = newton.ModelBuilder.JointDofConfig
        def restricted_axes():
            def ax(a):
                return Cfg(axis=a, limit_lower=-THETA_MAX, limit_upper=THETA_MAX,
                           limit_ke=LIMIT_KE, limit_kd=LIMIT_KD,
                           target_pos=0.0, target_ke=K_SPRING, target_kd=DAMPING)
            return [ax(wp.vec3(1.,0.,0.)), ax(wp.vec3(0.,1.,0.)), ax(wp.vec3(0.,0.,1.))]

        def local_pt(body_idx, world_pt):
            t = builder.body_q[body_idx]
            p = wp.vec3(float(t[0]),float(t[1]),float(t[2]))
            q = wp.quat(float(t[3]),float(t[4]),float(t[5]),float(t[6]))
            return wp.quat_rotate(wp.quat_inverse(q), wp.vec3(*world_pt) - p)

        # native D6 cone joints along the spanning tree (anchor at edge midpoint)
        for a, b in TREE:
            ia, ib = self.node_body[a], self.node_body[b]
            mid = 0.5 * (cen[a] + cen[b])
            art_joints.append(builder.add_joint_d6(parent=ia, child=ib, angular_axes=restricted_axes(),
                parent_xform=wp.transform(local_pt(ia, mid), wp.quat_identity()),
                child_xform=wp.transform(local_pt(ib, mid), wp.quat_identity()),
                label=f"cone_{a}_{b}"))
        builder.add_articulation(art_joints, label="palm_lattice")
        # loop closures: equality CONNECT at edge midpoint
        for a, b in LOOPS:
            ia, ib = self.node_body[a], self.node_body[b]
            mid = 0.5 * (cen[a] + cen[b])
            builder.add_equality_constraint_connect(body1=ia, body2=ib,
                anchor=local_pt(ia, mid), label=f"loop_{a}_{b}")

        self.model = builder.finalize(device="cuda")
        self.n_bodies = self.model.body_count
        self.bone_body = bone_body; self.cen = cen
        self.solver = SolverMuJoCo(self.model, integrator="implicitfast", solver="newton",
                                   cone="elliptic", impratio=10.0, iterations=100,
                                   ls_iterations=50, njmax=400, nconmax=200)
        # stiffen the weld/loop equality compliance (default timeconst 0.02 is far
        # too soft -> bones lag their spheres). Must stay >= 2*dt for stability.
        for mdl in (getattr(self.solver, "mjw_model", None), getattr(self.solver, "mj_model", None)):
            sr = getattr(mdl, "eq_solref", None) if mdl is not None else None
            if sr is None: continue
            if hasattr(sr, "numpy"):
                a = sr.numpy(); a[..., 0] = EQ_TIMECONST; a[..., 1] = 1.0; sr.assign(a)
            else:
                sr[..., 0] = EQ_TIMECONST; sr[..., 1] = 1.0

        self.state_0 = self.model.state(); self.state_1 = self.model.state()
        self.control = self.model.control(); self.contacts = self.model.contacts()
        self.viewer.set_model(self.model)
        self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate(); self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time); self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0); self.viewer.end_frame()

    def bone_pos(self, k):
        return self.state_0.body_q.numpy()[self.bone_body[k]][:3]

    def cone_bend_deg(self, a, b):
        q = self.state_0.body_q.numpy()
        def zaxis(k):
            t = q[self.node_body[k]]; quat = wp.quat(float(t[3]),float(t[4]),float(t[5]),float(t[6]))
            d = wp.quat_rotate(quat, wp.vec3(0.,0.,1.)); return np.array([float(d[0]),float(d[1]),float(d[2])])
        return math.degrees(math.acos(float(np.clip(zaxis(a) @ zaxis(b), -1, 1))))

    def max_ang_speed(self):
        qd = self.state_0.body_qd.numpy()[:, 3:6]
        return float(np.max(np.linalg.norm(qd, axis=1)))


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--check", action="store_true")
    args, _ = parser.parse_known_args()
    if args.check:
        wp.init()
        ex = Example(newton.viewer.ViewerNull(num_frames=300), args)
        p0 = {k: ex.bone_pos(k).copy() for k in BONE}
        for f in range(300): ex.step()
        speed = ex.max_ang_speed()
        drift = {k: float(np.linalg.norm(ex.bone_pos(k) - p0[k])) for k in BONE}
        finite = np.all(np.isfinite(ex.state_0.body_q.numpy()))
        print(f"[check] bodies={ex.n_bodies}  spheres={len(FLEX)}  tree_cone_joints={len(TREE)}  loop_eq={len(LOOPS)}")
        print(f"[check] settle: max body angular speed = {speed:.4f} rad/s")
        print("[check] bone drift (mm): " + "  ".join(f"{k}:{drift[k]*1e3:.1f}" for k in BONE))
        within = all(ex.cone_bend_deg(a,b) < math.degrees(THETA_MAX)+5 for a,b in TREE)
        print(f"[check] finite:{'PASS' if finite else 'FAIL'} | settles:{'PASS' if speed<2 else 'FAIL'} | "
              f"holds(<30mm):{'PASS' if max(drift.values())<0.03 else 'FAIL'} | cones within limit:{'PASS' if within else 'FAIL'}")
    else:
        viewer, args = newton.examples.init(parser)
        newton.examples.run(Example(viewer, args), args)
