# Semi-flexible palm for the real robot hand (Newton + MuJoCo), NATIVE version.
# The 7 split palm bones are coupled into a lattice with native MuJoCo WELD
# equality constraints (6-DOF soft springs the solver integrates IMPLICITLY ->
# unconditionally stable, unlike external body_f penalties). Each weld holds a
# bone at its rest pose relative to its neighbour; the weld compliance is the
# "give" -> a semi-flexible palm that springs back, anchored at the frozen pinky
# base + shells.
#
#   python flexpalm.py --check       # headless numeric check
#   python flexpalm.py --viewer gl   # GL window
import os, math, argparse
import numpy as np, warp as wp
import newton, newton.examples
import newton_compat  # noqa: F401  (1.2->1.4 equality API shim)
from newton.solvers import SolverMuJoCo
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

# Assets ship in ../assets of the DexiGrab repo; override with DEXIGRAB_ASSETS.
_ASSETS = os.environ.get("DEXIGRAB_ASSETS") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "assets")
USD = os.path.join(_ASSETS, "robot_hand_flexpalm.usda")

BONE = {
    "ring": "Base_Bone_12_V02_ringfing", "mid": "Base_Bone_12_V02_middlefinger",
    "ptr": "Base_Bone_12_V02_pointerfinger_and_thumb_attachment",
    "S00": "Palm_bone1/Palm_bone1", "S01": "Palm_bone1/Palm_bone1_01",
    "S03": "Palm_bone1/Palm_bone1_03", "S04": "Palm_bone1/Palm_bone1_04",
    "pinky": "Palm_rigid",   # frozen anchor body
}
EDGES = [("mid","ptr"),("mid","ring"),("mid","S00"),("pinky","ring"),("pinky","S03"),
         ("ptr","S01"),("ring","S04"),("S00","S04"),("S03","S04"),("S01","S00")]
TORQUESCALE = 1.0   # angular weld weight


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer; self.args = args
        self.frame_dt = 1.0 / 120.0; self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps; self.sim_time = 0.0

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(USD, schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])
        labels = list(builder.body_label)

        def bidx(suffix):
            for i, l in enumerate(labels):
                if l.endswith("/" + suffix) or l.endswith(suffix): return i
            raise KeyError(suffix)
        self.bid = {k: bidx(v) for k, v in BONE.items()}

        bq = builder.body_q
        for a, b in EDGES:
            ia, ib = self.bid[a], self.bid[b]
            ta, tb = bq[ia], bq[ib]
            pa = np.array([float(ta[0]), float(ta[1]), float(ta[2])])
            pb = np.array([float(tb[0]), float(tb[1]), float(tb[2])])
            qa = wp.quat(float(ta[3]),float(ta[4]),float(ta[5]),float(ta[6]))
            qb = wp.quat(float(tb[3]),float(tb[4]),float(tb[5]),float(tb[6]))
            qai = wp.quat_inverse(qa)
            relpose = wp.transform(wp.quat_rotate(qai, wp.vec3(*(pb - pa))), wp.mul(qai, qb))
            builder.add_equality_constraint_weld(
                body1=ia, body2=ib, anchor=wp.vec3(0.0, 0.0, 0.0),
                relpose=relpose, torquescale=TORQUESCALE, label=f"weld_{a}_{b}")

        self.model = builder.finalize(device="cuda")
        self.n_bodies = self.model.body_count
        self.n_eq = self.model.equality_constraint_count
        self.solver = SolverMuJoCo(self.model, integrator="implicitfast", solver="newton",
                                   cone="elliptic", impratio=10.0, iterations=100,
                                   ls_iterations=50, njmax=400, nconmax=200)
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

    def body_pos(self, i):
        return self.state_0.body_q.numpy()[i][:3]

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
        lat = list(BONE)
        p0 = {k: ex.body_pos(ex.bid[k]).copy() for k in lat}
        N = 300
        for f in range(N): ex.step()
        speed = ex.max_ang_speed()
        drift = {k: float(np.linalg.norm(ex.body_pos(ex.bid[k]) - p0[k])) for k in lat}
        finite = np.all(np.isfinite(ex.state_0.body_q.numpy()))
        print(f"[check] bodies={ex.n_bodies}  weld_constraints={ex.n_eq}  edges={len(EDGES)}")
        print(f"[check] settle: max body angular speed = {speed:.4f} rad/s")
        print("[check] bone drift from start (mm): " + "  ".join(f"{k}:{drift[k]*1e3:.1f}" for k in lat))
        ok = finite and speed < 2.0 and max(drift.values()) < 0.03
        print(f"[check] finite:{'PASS' if finite else 'FAIL'} | holds(<30mm):{'PASS' if max(drift.values())<0.03 else 'FAIL'} | settles:{'PASS' if speed<2 else 'FAIL'}")
    else:
        viewer, args = newton.examples.init(parser)
        newton.examples.run(Example(viewer, args), args)
