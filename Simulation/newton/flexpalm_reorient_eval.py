"""GL viewer for the flexpalm in-hand reorientation env: watch the palm-up hand cradle
the cube and articulate its fingers, with the live per-pad tactile heatmap and the cube
tinting green as it nears the goal orientation. Single world, full-res collision hulls.

  cd $RESEARCH_ROOT/robotics/sim/newton-cube-chains/flexpalm_hand
  unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib"
  DISPLAY=:1 python flexpalm_reorient_eval.py --viewer gl              # scripted finger manipulation
  DISPLAY=:1 python flexpalm_reorient_eval.py --viewer gl --policy ~/data/logdir/flexpalm_reorient/policy.pt
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
import newton.examples

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flexpalm_reorient_env as RE          # noqa: E402
from flexpalm_arm_demo import _heat          # noqa: E402  (pad-force -> heat color)
try:  # optional live-inspection bridge (private tool; not needed to run)
    from simbridge import Bridge             # noqa: E402
except ImportError:
    Bridge = None


class ReorientEval:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.force_max = float(getattr(args, "force_max", 3.0))
        ha = argparse.Namespace(hullverts=int(getattr(args, "hullverts", 256)))   # crisp visuals
        self.env = RE.FlexPalmReorientEnv(num_envs=1, substeps=int(getattr(args, "substeps", 8)),
                                          hand_args=ha, contacts_per_world=1500, verbose=True)
        self.model = self.env.model
        # action baseline: a=0 holds the cupped pose; we wiggle around it to manipulate
        self.a_amp = float(getattr(args, "amp", 0.8))
        self.t = 0.0
        # policy (optional)
        self.policy = None
        pp = getattr(args, "policy", None)
        if pp and os.path.exists(pp):
            from train_reorient import ActorCritic
            self.policy = ActorCritic(self.env.num_obs, self.env.num_actions).to(self.env.device)
            self.policy.load_state_dict(torch.load(pp, map_location=self.env.device)); self.policy.eval()
            print(f"[eval] loaded policy {pp}", flush=True)
        else:
            print("[eval] no policy -> scripted finger manipulation (per-finger sinusoid)", flush=True)

        # tactile pad shapes + base colors (for the heatmap recolor)
        self.sensor = self.env.sensor
        self.pad_shape = np.array([self.env.hand.fbody_shape[p.link] for p in self.sensor.pads], dtype=np.int64)
        self.base_color = self.model.shape_color.numpy().copy()
        self.base_color[self.env.cube_shape] = (0.95, 0.45, 0.10)   # cube: orange
        self.model.shape_color.assign(self.base_color)

        self.obs = self.env.reset()["policy"]
        viewer.set_model(self.model)
        # camera framed on the cube (palm-up cup)
        tgt = np.array(RE.CUBE_POS)
        eye = tgt + np.array([0.34, -0.34, 0.18])
        d = tgt - eye; h = float(np.hypot(d[0], d[1]))
        try:
            viewer.set_camera(pos=wp.vec3(*eye.tolist()),
                              pitch=float(np.degrees(np.arctan2(d[2], h))),
                              yaw=float(np.degrees(np.arctan2(d[1], d[0]))))
        except Exception as exc:
            print(f"[eval] set_camera failed: {exc}", flush=True)
        if getattr(self.env.hand, "shell_marker_specs", None):
            print("[eval] markers: MAGENTA = placed ball joint | CYAN = Shell1 sphere center | "
                  "YELLOW = Shell2 sphere center", flush=True)
        # live awareness/control (screenshots, state, reward viz, pause/step/exec)
        self.bridge = (Bridge.attach(env=self.env, viewer=viewer, name="flexpalm-eval")
                       if Bridge is not None else None)

    def step(self):
        if self.policy is not None:
            with torch.no_grad():
                a = self.policy.dist(self.obs).mean
        else:
            # per-finger sinusoid around the cupped hold -> visibly articulates + rotates the cube
            ph = self.t * 2.2
            k = torch.arange(self.env.num_actions, device=self.env.device).float()
            a = self.a_amp * torch.sin(ph + 0.7 * k).unsqueeze(0)
        self.obs, rew, done, ex = self.env.step(a)
        self.bridge.report_step(rew=rew, done=done, extras=ex)
        self.obs = self.obs["policy"]
        self.t += 1.0 / self.env.fps
        self._recolor(float(ex["log"]["rot_dist"]))

    def _recolor(self, rot_dist):
        self.sensor.update()
        f = self.sensor.pad_totals_torch()[0].cpu().numpy()    # world 0 pad forces (15,)
        cols = self.base_color.copy()
        for i, shp in enumerate(self.pad_shape):
            cols[shp] = _heat(self.base_color[shp], f[i] / self.force_max)
        # cube -> green as orientation nears the goal (rot_dist: ~pi far -> 0 aligned)
        align = max(0.0, 1.0 - rot_dist / math.pi)
        cols[self.env.cube_shape] = (0.95 * (1 - align), 0.30 + 0.65 * align, 0.10)
        self.model.shape_color.assign(cols)

    def _draw_markers(self):
        specs = getattr(self.env.hand, "shell_marker_specs", None)
        if not specs:
            return
        bq = self.env.state_0.body_q.numpy()
        for body_idx, local, color, name in specs:
            p = bq[body_idx]
            q = wp.quat(*[float(x) for x in p[3:7]])
            w = np.array(p[:3]) + np.array([float(x) for x in wp.quat_rotate(q, wp.vec3(*[float(x) for x in local]))])
            self.viewer.log_points(name, wp.array([w], dtype=wp.vec3),
                                   radii=wp.array([0.007], dtype=wp.float32),
                                   colors=wp.array([color], dtype=wp.vec3))

    def render(self):
        self.bridge.tick()   # examples.run calls render() every iteration, paused or not
        self.viewer.begin_frame(self.env.sim_time if hasattr(self.env, "sim_time") else self.t)
        self.viewer.log_state(self.env.state_0)
        self._draw_markers()
        self.viewer.end_frame()


def _add_args(p):
    p.add_argument("--policy", type=str, default=None)
    p.add_argument("--substeps", type=int, default=8)
    p.add_argument("--hullverts", type=int, default=256)
    p.add_argument("--amp", type=float, default=0.8, help="scripted finger wiggle amplitude")
    p.add_argument("--force-max", dest="force_max", type=float, default=3.0)
    return p


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    _add_args(parser)
    viewer, args = newton.examples.init(parser)
    if hasattr(viewer, "_paused"):
        viewer._paused = False
    newton.examples.run(ReorientEval(viewer, args), args)
