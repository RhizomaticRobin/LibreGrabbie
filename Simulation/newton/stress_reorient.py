"""Stress-test harness for the closed-loop flexpalm hand under SolverMuJoCo.

The hand's convex shell + underactuated spring-coupled palm + 3 closed-loop equality
constraints produce stiff, near-singular configurations that the contact/constraint
solver can drive to NaN under stress (seen single-world in the GUI: tug the model and
the bodies vanish + framerate drops). This localizes WHERE/WHEN divergence starts so we
can make the solver robust on those configs without changing the hand (no quality loss).

It steps substep-by-substep, applies stressors, and on the first non-finite value reports
the offending body/joint, the substep, and the run-up trajectory (was velocity/force
exploding -> integrator stiffness, or a sudden jump -> constraint singularity?).

Stressors (mimic RL exploration + the GUI tug):
  tug    : random external spatial wrench (N*m / N) on every body each substep
  slew   : max-rate finger target slewing (snap to random +/- limits each step)
  squeeze: drive all fingers fully closed (crush the cube)
  all    : tug + slew

  python stress_reorient.py --mode tug --mag 5 --num-envs 1 --steps 300
  python stress_reorient.py --mode all --num-envs 16 --bisect          # find the threshold
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flexpalm_reorient_env as E   # noqa: E402


def _refresh(env):
    env.jq = wp.to_torch(env.state_0.joint_q)
    env.jqd = wp.to_torch(env.state_0.joint_qd)
    env.bq = wp.to_torch(env.state_0.body_q)
    env.bqd = wp.to_torch(env.state_0.body_qd)


def _first_nonfinite(env):
    """Return (array_name, flat_index, world, body_or_dof) of the first non-finite, else None."""
    bpw = env.model.body_count // env.num_envs
    dpw = env.dofs_per_world
    cpw = env.jq.numel() // env.num_envs
    for name, t, per in (("body_qd", env.bqd, bpw), ("body_q", env.bq, bpw),
                         ("joint_qd", env.jqd, dpw), ("joint_q", env.jq, cpw)):
        mask = ~torch.isfinite(t)
        if mask.any():
            idx = mask.nonzero()[0]
            row = int(idx[0])
            world = row // per
            local = row % per
            lbl = ""
            if name.startswith("body"):
                lbl = env.model.body_label[row].split("/")[-1]
            return name, idx.tolist(), world, local, lbl
    return None


def run(num_envs, mode, mag, steps, solver_kw, report=True):
    wp.init()
    env = E.FlexPalmReorientEnv(num_envs=num_envs, verbose=False, **solver_kw)
    env.reset()
    sub = env.substeps
    a_slew = torch.zeros(num_envs, 22, device=env.device)
    a_cup = 2 * (env.finger_cupped - env.finger_lo) / (env.finger_hi - env.finger_lo).clamp_min(1e-6) - 1
    hist = []   # (frame, max|jqd|, max|bqd|, n_contact, max_force)
    for f in range(steps):
        if mode in ("slew", "all"):
            a_slew = torch.sign(torch.randn(num_envs, 22, device=env.device))   # snap to +/- limits
        elif mode == "squeeze":
            a_slew = torch.ones(num_envs, 22, device=env.device)                # crush
        else:
            a_slew = a_cup.unsqueeze(0).repeat(num_envs, 1) + 0.3 * torch.randn(num_envs, 22, device=env.device)
        env._apply_action(a_slew)
        for s in range(sub):
            env.state_0.clear_forces()
            if mode in ("tug", "all"):
                bf = wp.to_torch(env.state_0.body_f)
                bf += mag * torch.randn_like(bf)
            env.model.collide(env.state_0, env.contacts, collision_pipeline=env._pipeline)
            env.solver.step(env.state_0, env.state_1, env.control, env.contacts, env.dt)
            env.state_0, env.state_1 = env.state_1, env.state_0
            _refresh(env)
            fn = _first_nonfinite(env)
            if fn is not None:
                env.sensor.update()
                maxf = float(env.sensor.pad_totals_torch().max())
                if report:
                    print(f"[stress] FIRST NaN at frame {f} substep {s}: {fn[0]} world={fn[2]} "
                          f"local_idx={fn[3]} body='{fn[4]}'")
                    print(f"[stress] run-up (last frames) max|jointvel| / max|bodyvel| / contacts / maxpadforce:")
                    for h in hist[-6:]:
                        print("          f%-4d |jqd|=%.1f |bqd|=%.1f ncon=%d maxF=%.1fN" % h)
                return dict(frame=f, substep=s, **{"arr": fn[0], "world": fn[2], "body": fn[4]}, hist=hist)
        # per-frame run-up tracking
        _refresh(env)
        ncon = int(env.contacts.rigid_contact_count.numpy()[0]) if hasattr(env.contacts, "rigid_contact_count") else -1
        env.sensor.update()
        hist.append((f, float(env.jqd.abs().max()), float(env.bqd.abs().max()), ncon,
                     float(env.sensor.pad_totals_torch().max())))
    if report:
        print(f"[stress] SURVIVED {steps} frames, no NaN. "
              f"peak |jqd|={max(h[1] for h in hist):.1f} |bqd|={max(h[2] for h in hist):.1f} "
              f"maxF={max(h[4] for h in hist):.1f}N")
    return dict(frame=None, hist=hist)


def bisect_threshold(num_envs, mode, steps, solver_kw):
    """Binary-search the stress magnitude (tug wrench) at which NaN first appears."""
    lo, hi = 0.0, 50.0
    for _ in range(7):
        mid = 0.5 * (lo + hi)
        r = run(num_envs, mode, mid, steps, solver_kw, report=False)
        survived = r["frame"] is None
        print(f"[bisect] mag={mid:.2f} -> {'survived' if survived else 'NaN@f%d' % r['frame']}")
        if survived:
            lo = mid
        else:
            hi = mid
    print(f"[bisect] NaN threshold ~ tug magnitude {hi:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["tug", "slew", "squeeze", "all"], default="all")
    p.add_argument("--mag", type=float, default=5.0, help="external tug wrench magnitude")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--substeps", type=int, default=8)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--ls-iters", type=int, default=20)
    p.add_argument("--bisect", action="store_true")
    a = p.parse_args()
    skw = dict(substeps=a.substeps)
    if a.bisect:
        bisect_threshold(a.num_envs, a.mode, a.steps, skw)
    else:
        run(a.num_envs, a.mode, a.mag, a.steps, skw)
