# Simulation

The DEXIGRAB physics-simulation line: the hand rebuilt as a full articulated model in
[Newton](https://github.com/newton-physics/newton) (NVIDIA's GPU physics engine, MuJoCo-Warp
solver) — flexible palm mechanism and all — with per-pad tactile sensing, an arm-mounted
grasp demo, and a GPU-batched in-hand reorientation RL environment.

![reach -> whole-hand grasp -> lift, palm-side view with the 15 tactile pads](media/grasp_demo.gif)

*The arm-mounted demo: reach, whole-hand grasp, lift — recorded straight from `newton/flexpalm_arm_demo.py --arm franka --viewer gl` (full quality: [media/grasp_demo.mp4](media/grasp_demo.mp4)).*

This is the simulation work the main README used to promise as "upcoming". It's here now,
including the parts that didn't go the way I wanted (see *Honest status* below).

## What's in here

| Path | What it is |
|---|---|
| [`newton/`](newton/) | All code. Start with [`newton/README.md`](newton/README.md) (hand model + arm grasp demo) and [`newton/README_reorient.md`](newton/README_reorient.md) (batched RL env + the solver-stability findings). |
| [`assets/`](assets/) | The hand CAD as USD (same geometry as `Hardware/`, exported for sim) + the Shell2 recovery re-export. |

Highlights:

- **The hand as a real mechanism, not a rigid prop**: 8 spherical-jointed palm bones with
  tuned spring/damping (the flexible TPU palm), 3 closed kinematic loops, 22 finger DOFs,
  convex-decomposed self-colliding shells.
- **`flexpalm_arm_demo.py`** — reach → whole-hand grasp → lift of a cube on a Franka FR3
  flange, with a 15-pad tactile heatmap. Its `--check` gate only passes a **real grasp**:
  clean approach (the open hand must descend *around* the cube, not through it) + tactile
  contact on ≥2 pads + a >2 cm lift. Lift-only checks are a trap — they green-light a cube
  wedged inside the shell.
- **`flexpalm_reorient_env.py`** — Allegro-style in-hand cube reorientation, GPU-batched
  (one model, one solver, N worlds), 102-D obs including the 15 tactile pad forces,
  validated at **N=8192 with 0.0% solver divergence at ~47k env-steps/s** on one RTX 5090.
- **`train_reorient.py`** — self-contained GPU PPO on that env (rsl_rl `VecEnv`-compatible).
- **`stress_reorient.py`** — the NaN-localization harness that found the solver failure
  modes documented in `README_reorient.md`.

## Requirements

An NVIDIA GPU + a Python env with `newton` (verified on **1.4.0**; the 1.2-era equality API
this was written against is restored by the tiny [`newton/newton_compat.py`](newton/newton_compat.py)
shim), `warp-lang`, `torch`, `usd-core`, `numpy`. Verified on an RTX 5090.

## Honest status

- **What stands**: the mechanism sim, the tactile sensor, the honest-gated grasp demo, and a
  batched env that is *stable at scale* — getting there took root-causing a real solver
  failure (the batched contact + equality solve diverged to NaN in a growing fraction of
  worlds past N≈8; the full forensics, the fix, and the measured tables are in
  [`newton/README_reorient.md`](newton/README_reorient.md)).
- **What didn't work**: dexterous reorientation *policies*. Training runs stall for reasons
  that trace to contact fidelity, not hyperparameters — so no pretrained models are
  published, and I'm not going to ship a policy that only works in a sim I don't trust.
  That result is what pushed this project down the stack, into engine-level work on contact
  physics faithful enough that soft-hand policies trained in sim mean something.

Same licenses as the rest of the repo (GPL-3.0 code; the CAD-derived assets follow the
hardware license).
