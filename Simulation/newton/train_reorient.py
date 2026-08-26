"""PPO trainer for the flexpalm in-hand cube-reorientation env (FlexPalmReorientEnv).

Self-contained GPU-resident PPO (actor-critic MLP) so it runs with zero RL-library
version friction; the env also implements the rsl_rl VecEnv protocol and returns a
{"policy": obs} dict from reset(), so it drops into rsl_rl's OnPolicyRunner or the
liquid-NN CfCPPO with no adapter when you want those (see README_reorient.md).

  # lease the GPU for the run (single RTX 5090 is shared):
  # --sm-heavy: PPO training saturates SM (singleton vs other heavy jobs; light sessions co-admit)
  JOB=$(~/tools/gpu-queue acquire --sm-heavy --seconds 3600 --vram 8000 --conda-env isaac --label "flexpalm reorient PPO")
  trap '~/tools/gpu-queue release "$JOB"' EXIT
  unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib"
  python train_reorient.py --num-envs 256 --iters 1000
  python train_reorient.py --num-envs 4 --iters 30 --smoke      # quick pipeline check
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flexpalm_reorient_env as E   # noqa: E402


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(512, 512, 256)):
        super().__init__()
        def mlp(out):
            layers, last = [], obs_dim
            for h in hidden:
                layers += [nn.Linear(last, h), nn.ELU()]
                last = h
            layers += [nn.Linear(last, out)]
            return nn.Sequential(*layers)
        self.actor = mlp(act_dim)
        self.critic = mlp(1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, obs):
        mean = self.actor(obs)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def train(num_envs=256, iters=1000, n_steps=32, epochs=5, minibatches=4, gamma=0.99,
          lam=0.95, clip=0.2, lr=5e-4, ent_coef=0.01, vf_coef=1.0, max_grad=1.0,
          substeps=8, smoke=False, log_dir=None, tripairs_per_world=80_000, contacts_per_world=600,
          solver="mujoco", soft_loops=False, cube_half=None):
    wp.init()
    dev = torch.device("cuda")
    env = E.FlexPalmReorientEnv(num_envs=num_envs, substeps=substeps,
                                tripairs_per_world=tripairs_per_world, contacts_per_world=contacts_per_world,
                                solver=solver, soft_loops=soft_loops, cube_half=cube_half)
    obs_dim, act_dim = env.num_obs, env.num_actions
    ac = ActorCritic(obs_dim, act_dim).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    N, T = num_envs, n_steps

    # rollout buffers
    b_obs = torch.zeros(T, N, obs_dim, device=dev)
    b_act = torch.zeros(T, N, act_dim, device=dev)
    b_logp = torch.zeros(T, N, device=dev)
    b_val = torch.zeros(T, N, device=dev)
    b_rew = torch.zeros(T, N, device=dev)
    b_done = torch.zeros(T, N, device=dev)

    obs = env.reset()["policy"]
    t0 = time.time()
    skipped_total = 0
    r_hist = []; max_nan = 0.0
    for it in range(iters):
        ep_rew = 0.0; succ = 0.0; hold = 0.0; rotd = 0.0; drop = 0.0; nanf = 0.0
        for t in range(T):
            with torch.no_grad():
                d = ac.dist(obs); a = d.sample()
                lp = d.log_prob(a).sum(-1); v = ac.value(obs)
            b_obs[t] = obs; b_act[t] = a; b_logp[t] = lp; b_val[t] = v
            obs, rew, done, extras = env.step(a)
            obs = obs["policy"]
            b_rew[t] = rew; b_done[t] = done
            ep_rew += rew.mean().item()
            succ += extras["log"]["success_rate"].item()
            hold += extras["log"]["hold_rate"].item()
            rotd += extras["log"]["rot_dist"].item()
            drop += extras["log"]["dropped"].item()
            nanf += extras["log"]["nan_envs"].item()
        with torch.no_grad():
            last_v = ac.value(obs)
        # GAE
        adv = torch.zeros_like(b_rew); gae = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nonterm = 1.0 - b_done[t]
            nextv = last_v if t == T - 1 else b_val[t + 1]
            delta = b_rew[t] + gamma * nextv * nonterm - b_val[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + b_val
        # sanitize before the update: a NaN slipping through the env guard (e.g. an exploded
        # value bootstrap) must not reach the optimizer. nan_to_num + finite-guarded step below
        # together guarantee one diverged world can never collapse the policy.
        adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
        ret = torch.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # flatten + PPO update
        f_obs = b_obs.reshape(T * N, obs_dim); f_act = b_act.reshape(T * N, act_dim)
        f_logp = b_logp.reshape(-1); f_adv = adv.reshape(-1); f_ret = ret.reshape(-1)
        bs = T * N; mb = bs // minibatches
        skipped = 0
        for _ in range(epochs):
            idx = torch.randperm(bs, device=dev)
            for s in range(0, bs, mb):
                j = idx[s:s + mb]
                d = ac.dist(f_obs[j]); lp = d.log_prob(f_act[j]).sum(-1)
                ratio = (lp - f_logp[j]).exp()
                a_j = f_adv[j]
                pl = -torch.min(ratio * a_j, torch.clamp(ratio, 1 - clip, 1 + clip) * a_j).mean()
                vl = (ac.value(f_obs[j]) - f_ret[j]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pl + vf_coef * vl - ent_coef * ent
                opt.zero_grad(); loss.backward()
                gn = nn.utils.clip_grad_norm_(ac.parameters(), max_grad)
                # ANTI-COLLAPSE: never apply a non-finite update. If the loss or its gradient
                # is NaN/Inf (one poisoned transition that slipped past the env guard), drop
                # this minibatch — the weights stay intact instead of being overwritten by NaN.
                if torch.isfinite(loss) and torch.isfinite(gn):
                    opt.step()
                else:
                    skipped += 1
        skipped_total += skipped
        r_hist.append(ep_rew / T); max_nan = max(max_nan, nanf / T)

        if it % 5 == 0 or it == iters - 1:
            sps = (it + 1) * T * N / (time.time() - t0)
            warn = f" | NaN-worlds {nanf/T:.3%}" + (f" SKIPPED {skipped} updates" if skipped else "")
            ga = extras["log"].get("goal_angle")
            gastr = f" | goal {float(ga):.2f}rad" if ga is not None else ""
            print(f"[train] it {it:4d} | rew/step {ep_rew/T:+7.2f} | atgoal {succ/T:.3f} | "
                  f"hold {hold/T:.3f} | rot_dist {rotd/T:.2f} | dropped {drop/T:.2f}{gastr} | "
                  f"{sps:.0f} env-steps/s{warn}", flush=True)
        if log_dir and (it % 100 == 0 or it == iters - 1):
            os.makedirs(log_dir, exist_ok=True)
            torch.save(ac.state_dict(), os.path.join(log_dir, "policy.pt"))
    if log_dir:
        torch.save(ac.state_dict(), os.path.join(log_dir, "policy.pt"))
        print(f"[train] saved policy to {log_dir}/policy.pt", flush=True)
    sps = iters * T * N / (time.time() - t0)
    vram = torch.cuda.max_memory_allocated() / 1e9
    healthy = bool(np.isfinite(r_hist[-1]) and r_hist[-1] > r_hist[0] - 1.0 and max_nan < 0.15)
    print(f"[SUMMARY] N={N} substeps={substeps} rew {r_hist[0]:+.2f}->{r_hist[-1]:+.2f} "
          f"max_nan {max_nan:.2%} skipped {skipped_total} sps {sps:.0f} vram {vram:.1f}GB "
          f"-> {'HEALTHY' if healthy else 'BROKEN'}", flush=True)
    return ac


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # WINNER config: N=8192, substeps=8, soft_loops (penalty-spring loop closures, NO equality).
    # Removing the equality SOLVE eliminates the NaN (equality crept to 28% over a long run;
    # soft_loops holds 0.00%) AND runs ~3x faster (47k vs 16k env-steps/s, no constraint solve).
    # soft_loops needs substeps=8: substeps=4's larger dt blows up the explicit springs.
    p.add_argument("--num-envs", type=int, default=8192)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--n-steps", type=int, default=24)
    p.add_argument("--substeps", type=int, default=8)
    p.add_argument("--tripairs-per-world", type=int, default=15_000)
    p.add_argument("--contacts-per-world", type=int, default=500)
    p.add_argument("--solver", type=str, default="mujoco")
    p.add_argument("--soft-loops", dest="soft_loops", action="store_true", default=True,
                   help="penalty-spring loop closures, no equality (default; the NaN fix)")
    p.add_argument("--hard-loops", dest="soft_loops", action="store_false",
                   help="use the original MuJoCo equality loop closures (NaN-prone)")
    p.add_argument("--cube-half", type=float, default=None, help="cube half-extent [m] (default 0.025)")
    p.add_argument("--smoke", action="store_true", help="short pipeline-validation run")
    p.add_argument("--log-dir", type=str, default=os.path.join("logs", "flexpalm_reorient"))
    a = p.parse_args()
    if a.smoke:
        a.iters = min(a.iters, 30)
    train(num_envs=a.num_envs, iters=a.iters, n_steps=a.n_steps, substeps=a.substeps,
          smoke=a.smoke, log_dir=a.log_dir,
          tripairs_per_world=a.tripairs_per_world, contacts_per_world=a.contacts_per_world,
          solver=a.solver, soft_loops=a.soft_loops, cube_half=a.cube_half)
