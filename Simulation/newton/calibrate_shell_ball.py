"""Calibrate the Shell1<->Shell2 ball-joint center so the (fast, non-colliding)
shells_jointed config reproduces the shell motion of the prior HIGH-FIDELITY config
(full CoACD collision shells -- slower + occasionally NaN, but the fidelity reference).

For a fixed, aggressive scripted trajectory (single world), we record Shell2's pose in
Shell1's frame over time. The reference is the collision config; each candidate ball
center is scored by how closely its relative-pose trajectory matches the reference.
Reports the drift magnitude (is the center dynamically meaningful?) and the best center.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch, warp as wp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flexpalm_reorient_env as E


def relpose_traj(hand_args, frames=140, amp=1.0, substeps=8):
    env = E.FlexPalmReorientEnv(num_envs=1, substeps=substeps, hand_args=hand_args, verbose=False,
                                contacts_per_world=2500, tripairs_per_world=300_000)
    env.reset()
    s1 = env.hand.shell_body["Shell1"]; s2 = env.hand.shell_body["Shell2"]
    k = torch.arange(env.num_actions, device=env.device).float()
    traj = []; nan = 0
    for f in range(frames):
        a = amp * torch.sin(f * 0.22 + 0.7 * k).unsqueeze(0)     # deterministic palm-flexing wiggle
        obs, rew, done, ex = env.step(a)
        if not torch.isfinite(obs["policy"]).all():
            nan += 1
        bq = env.state_0.body_q.numpy()
        T1 = wp.transform(wp.vec3(*[float(x) for x in bq[s1][:3]]), wp.quat(*[float(x) for x in bq[s1][3:7]]))
        T2 = wp.transform(wp.vec3(*[float(x) for x in bq[s2][:3]]), wp.quat(*[float(x) for x in bq[s2][3:7]]))
        rel = wp.transform_multiply(wp.transform_inverse(T1), T2)
        traj.append([float(x) for x in rel])                     # pos3 + quat4(xyzw)
    cp, _ = env._cube_pose()
    held = float(((cp - env.in_hand_pos).norm(dim=-1)[0] < E.FALL_DIST).item())
    fc = getattr(env.hand, "fit_centers", None)
    del env; torch.cuda.empty_cache()
    return np.array(traj), held, nan, fc


def quat_angle_diff(qa, qb):                                     # geodesic angle [rad] between xyzw quats, per row
    d = np.abs((qa * qb).sum(1).clip(-1, 1))
    return 2 * np.arccos(d)


def main():
    wp.init()
    # reference: prior full-CoACD collision shells
    ref_ha = argparse.Namespace(shells_jointed=False, simple_shells=False, hullverts=64)
    ref, ref_held, ref_nan, _ = relpose_traj(ref_ha)
    # how much do the shells actually move relative to each other in the reference?
    dpos = np.linalg.norm(ref[:, :3] - ref[0, :3], axis=1)
    dang = quat_angle_diff(ref[:, 3:7], np.tile(ref[0, 3:7], (len(ref), 1)))
    print(f"[ref] CoACD-collision: held={ref_held:.0f} nan={ref_nan} | shell rel drift: "
          f"pos max {dpos.max()*1000:.2f}mm, rot max {np.degrees(dang.max()):.2f}deg", flush=True)

    # candidate centers: the fitted ones + midpoint + a small grid around the combined fit
    _, _, _, fc = relpose_traj(argparse.Namespace(shells_jointed=True, hullverts=64), frames=2)
    cands = {"combined": fc["combined"], "shell1": fc["shell1"], "shell2": fc["shell2"],
             "mid(s1,s2)": 0.5 * (fc["shell1"] + fc["shell2"])}
    base = fc["combined"]
    for dz in (-0.012, -0.006, 0.006, 0.012):
        cands[f"comb+z{dz:+.3f}"] = base + np.array([0, 0, dz])
    for dy in (-0.008, 0.008):
        cands[f"comb+y{dy:+.3f}"] = base + np.array([0, dy, 0])

    print(f"[ref] frames={len(ref)}", flush=True)
    results = []
    for name, c in cands.items():
        ha = argparse.Namespace(shells_jointed=True, hullverts=64,
                                shell_ball_center=tuple(float(x) for x in c))
        traj, held, nan, _ = relpose_traj(ha)
        n = min(len(traj), len(ref))
        pos_err = np.linalg.norm(traj[:n, :3] - ref[:n, :3], axis=1).mean()
        rot_err = quat_angle_diff(traj[:n, 3:7], ref[:n, 3:7]).mean()
        score = pos_err * 1000 + np.degrees(rot_err)             # mm + deg
        results.append((name, c, score, pos_err * 1000, np.degrees(rot_err), held, nan))
    results.sort(key=lambda r: r[2])
    print("\n  center                     score   pos_err(mm) rot_err(deg) held nan", flush=True)
    for name, c, score, pe, re, held, nan in results:
        print("  %-26s %6.2f   %8.3f   %8.3f   %.0f  %d   @%s" %
              (name, score, pe, re, held, nan, np.round(c, 4)), flush=True)
    best = results[0]
    print(f"\n[best] {best[0]} @ {np.round(best[1],4)}  (score {best[2]:.2f})", flush=True)


def optimize():
    """Converge on the ball center (x,y,z) that best matches the CoACD-collision reference
    shell-motion trajectory, via Nelder-Mead. Objective = mean shell-position-trajectory
    error [mm] (+ small rotation term to break ties) with a stability penalty."""
    from scipy.optimize import minimize
    wp.init()
    ref, rh, rn, _ = relpose_traj(argparse.Namespace(shells_jointed=False, simple_shells=False, hullverts=64))
    dpos = np.linalg.norm(ref[:, :3] - ref[0, :3], axis=1)
    print(f"[ref] CoACD drift pos_max {dpos.max()*1000:.2f}mm | held {rh:.0f} nan {rn}", flush=True)
    _, _, _, fc = relpose_traj(argparse.Namespace(shells_jointed=True, hullverts=64), frames=2)
    x0 = np.asarray(fc["combined"], float) + np.array([0.0, 0.0, 0.006])
    evals = {"n": 0}

    def obj(c):
        evals["n"] += 1
        tr, held, nan, _ = relpose_traj(argparse.Namespace(shells_jointed=True, hullverts=64,
                                                            shell_ball_center=tuple(float(x) for x in c)))
        nn = min(len(tr), len(ref))
        pe = np.linalg.norm(tr[:nn, :3] - ref[:nn, :3], axis=1).mean() * 1000.0
        re = np.degrees(quat_angle_diff(tr[:nn, 3:7], ref[:nn, 3:7]).mean())
        score = pe + 0.1 * re + 1000.0 * (1.0 - held) + 200.0 * nan
        print(f"  eval {evals['n']:2d} @ {np.round(c,4)} -> pos {pe:.2f}mm rot {re:.1f}deg held {held:.0f} score {score:.2f}", flush=True)
        return score

    init = np.array([x0, x0 + [0.006, 0, 0], x0 + [0, 0.006, 0], x0 + [0, 0, 0.006]])
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"initial_simplex": init, "xatol": 3e-4, "fatol": 0.3, "maxiter": 80})
    print(f"\n[CONVERGED] center = {np.round(res.x,4)} | score {res.fun:.2f} | {evals['n']} evals", flush=True)
    return res.x


def optimize5():
    """Converge the WELD's center(3) + torquescale + solref-timeconst (5 params) to match
    the CoACD-collision shell trajectory in BOTH position and orientation. The weld's
    rotational stiffness can close the ~27deg residual a free ball leaves."""
    from scipy.optimize import minimize
    import flexpalm_bones4 as fb
    wp.init()
    ref, rh, rn, _ = relpose_traj(argparse.Namespace(shells_jointed=False, simple_shells=False, hullverts=64))
    dpos = np.linalg.norm(ref[:, :3] - ref[0, :3], axis=1)
    dang = quat_angle_diff(ref[:, 3:7], np.tile(ref[0, 3:7], (len(ref), 1)))
    print(f"[ref] CoACD drift pos {dpos.max()*1000:.2f}mm rot {np.degrees(dang.max()):.2f}deg | held {rh:.0f} nan {rn}", flush=True)
    c0 = np.array(fb.SHELL_BALL_CENTER, float)
    x0 = np.array([c0[0], c0[1], c0[2], 0.0, -1.8])          # log10(torquescale)=0 (1.0), log10(tc)=-1.8 (~0.016)
    ev = {"n": 0}

    def unpack(x):
        return x[:3], float(10 ** x[3]), float(10 ** np.clip(x[4], -2.7, -1.0))

    def obj(x):
        ev["n"] += 1
        c, ts, tc = unpack(x)
        tr, held, nan, _ = relpose_traj(argparse.Namespace(
            shells_jointed=True, hullverts=64, shell_ball_center=tuple(float(v) for v in c),
            shell_torquescale=ts, shell_solref_tc=tc))
        nn = min(len(tr), len(ref))
        pe = np.linalg.norm(tr[:nn, :3] - ref[:nn, :3], axis=1).mean() * 1000.0
        re = np.degrees(quat_angle_diff(tr[:nn, 3:7], ref[:nn, 3:7]).mean())
        score = pe + re + 1000.0 * (1.0 - held) + 200.0 * nan
        print(f"  ev {ev['n']:3d} c={np.round(c,4)} ts={ts:.2f} tc={tc:.4f} -> pos {pe:.2f} rot {re:.1f} held {held:.0f} score {score:.2f}", flush=True)
        return score

    d = [0.006, 0.006, 0.006, 0.6, 0.4]
    init = np.array([x0] + [x0 + np.eye(5)[i] * d[i] for i in range(5)])
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"initial_simplex": init, "xatol": 2e-4, "fatol": 0.25, "maxiter": 160})
    c, ts, tc = unpack(res.x)
    print(f"\n[CONVERGED5] center={np.round(c,4)} torquescale={ts:.3f} solref_tc={tc:.4f} | score {res.fun:.2f} | {ev['n']} evals", flush=True)
    return c, ts, tc


def _qrot(q, v):                       # rotate vec v by xyzw quat q (torch, batched)
    xyz = q[..., :3]; w = q[..., 3:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def batched(pop=4096, gens=12, amp=1.0, frames=140, n_elite=24):
    """BATCHED calibration: each world gets a different weld (center, torquescale, solref);
    one rollout scores the whole population in parallel. Evolutionary search (sample ->
    eval all-at-once -> keep elites -> resample tighter) over center(3)+log(ts)+log(tc).
    Cranks the 5090: thousands of candidates per generation."""
    import torch
    import flexpalm_bones4 as fb
    wp.init()
    ref, rh, rn, _ = relpose_traj(argparse.Namespace(shells_jointed=False, simple_shells=False, hullverts=64), frames=frames, amp=amp)
    F = len(ref)
    ref_pos = torch.tensor(ref[:, :3], device="cuda", dtype=torch.float32)
    ref_quat = torch.tensor(ref[:, 3:7], device="cuda", dtype=torch.float32)
    dpos = np.linalg.norm(ref[:, :3] - ref[0, :3], axis=1)
    print(f"[ref] CoACD drift pos {dpos.max()*1000:.2f}mm | held {rh:.0f} nan {rn} | frames {F}", flush=True)

    # geometric sphere-fit center (where the two shells physically mate) -> optional
    # regularization so the dynamically-fit pivot stays ON the mating surface.
    reg = float(os.environ.get("SHELL_REG", "0.0"))          # penalty weight (mm score / mm deviation)
    c_geom = np.array([-0.12055, -0.0293, -0.0092])          # avg(shell1,shell2) sphere centers
    c0 = c_geom.copy() if reg > 0 else np.array(fb.SHELL_BALL_CENTER)
    rng = np.random.default_rng(0)
    # broad initial search (the optimum moved a lot last time); shells span ~x[-0.19,-0.06] etc.
    csd0 = np.array([0.02, 0.02, 0.02]) if reg > 0 else np.array([0.05, 0.06, 0.04])
    cm, csd = c0.copy(), csd0; tsm, tssd, tcm, tcsd = 0.0, 1.0, -1.6, 0.7
    if reg > 0:
        print(f"[reg] geometric-center regularization ON: weight {reg}, anchor {c_geom}", flush=True)
    k = torch.arange(22, device="cuda").float()
    best = None
    for g in range(gens):
        c = cm + rng.normal(0, csd, (pop, 3))
        lts = rng.normal(tsm, tssd, pop); ltc = rng.normal(tcm, tcsd, pop)
        ts = np.clip(10 ** lts, 0.02, 50.0); tc = np.clip(10 ** ltc, 0.003, 0.12)
        params = np.column_stack([c, ts, tc])
        if best is not None:                                 # always carry the elites
            params[:len(best)] = best
        import flexpalm_reorient_env as RE
        env = RE.FlexPalmReorientEnv(num_envs=pop, substeps=8, weld_params=params, verbose=False,
                                     contacts_per_world=500, tripairs_per_world=30_000)
        nbody = env.model.body_count // pop
        i1 = torch.tensor(np.arange(pop) * nbody + env.hand.shell_body["Shell1"], device="cuda")
        i2 = torch.tensor(np.arange(pop) * nbody + env.hand.shell_body["Shell2"], device="cuda")
        env.reset()
        rpos = torch.zeros(pop, F, 3, device="cuda"); rquat = torch.zeros(pop, F, 4, device="cuda")
        for f in range(F):
            a = (amp * torch.sin(f * 0.22 + 0.7 * k)).unsqueeze(0).repeat(pop, 1)
            env._apply_action(a); env._step_sim()
            bq = env.bq
            p1 = bq[i1][:, :3]; q1 = RE.quat_norm(bq[i1][:, 3:7]); p2 = bq[i2][:, :3]; q2 = RE.quat_norm(bq[i2][:, 3:7])
            q1c = RE.quat_conj(q1)
            rpos[:, f] = _qrot(q1c, p2 - p1); rquat[:, f] = RE.quat_mul(q1c, q2)
        pe = (rpos - ref_pos).norm(dim=-1).mean(1) * 1000.0
        dot = (rquat * ref_quat).sum(-1).abs().clamp(max=1.0)
        re = torch.rad2deg(2 * torch.arccos(dot)).mean(1)
        score = pe + re
        if reg > 0:                                          # keep the pivot near the mating surface
            dev = torch.tensor(np.linalg.norm(params[:, :3] - c_geom, axis=1) * 1000.0, device="cuda", dtype=torch.float32)
            score = score + reg * dev
        score[~torch.isfinite(score)] = 1e6
        order = torch.argsort(score)
        elite = params[order[:n_elite].cpu().numpy()]
        best = elite
        b = order[0].item()
        print(f"[gen {g}] best score {score[b]:.2f} (pos {pe[b]:.2f}mm rot {re[b]:.1f}deg) @ c={np.round(params[b,:3],4)} ts={params[b,3]:.2f} tc={params[b,4]:.4f} | pop {pop}", flush=True)
        # tighten around elite centroid
        cm = elite[:, :3].mean(0); csd *= 0.55
        tsm = np.log10(np.clip(elite[:, 3], 0.02, 30)).mean(); tssd *= 0.6
        tcm = np.log10(np.clip(elite[:, 4], 0.003, 0.06)).mean(); tcsd *= 0.6
        del env; torch.cuda.empty_cache()
    bp = best[0]
    print(f"\n[BATCHED BEST] center={np.round(bp[:3],4)} torquescale={bp[3]:.3f} solref_tc={bp[4]:.4f}", flush=True)
    return bp


def _argval(flag, default):
    import sys
    return int(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


if __name__ == "__main__":
    import sys
    if "--batched" in sys.argv:
        batched(pop=_argval("--pop", 4096), gens=_argval("--gens", 12))
    elif "--optimize5" in sys.argv:
        optimize5()
    elif "--optimize" in sys.argv:
        optimize()
    else:
        main()
