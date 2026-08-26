# flexpalm in-hand cube reorientation — batched-Newton RL env

An **Allegro-style in-hand reorientation task** driven by the flexible flexpalm hand **+ its
15-pad tactile sensor**: the hand is fixed palm-up, cradles a free cube, and the policy moves the
22 finger DOFs (PD position targets) to rotate the cube to a randomly sampled target orientation.
Pure Newton + `SolverMuJoCo`, **GPU-batched over N worlds** in one model/one solver, `isaac` env.

Mirrors IsaacLab's `InHandManipulationEnv` (random-quat goal, rotation-distance reward,
success / consecutive-success goal resampling, drop termination) but with our underactuated
closed-loop hand instead of a fully-actuated Allegro/Shadow hand.

## Run

```bash
cd Simulation/newton
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib"   # tactile NVRTC JIT
python flexpalm_reorient_env.py --check --num-envs 64        # env gate (finite, cube seats, tactile fires)
python stress_reorient.py --mode all --num-envs 1            # solver stress / NaN-localization harness
python train_reorient.py --num-envs 256 --iters 2000         # self-contained GPU PPO
```

## Files
| File | Role |
|---|---|
| `flexpalm_reorient_env.py` | `FlexPalmReorientEnv` — batched build, palm-up cube seating, 22-DOF residual action, 102-dim obs (incl. 15 tactile pad forces), rotation-distance reward, goal resampling, drop termination, NaN-guarded rsl_rl-style VecEnv. `--check` gate. |
| `train_reorient.py` | Self-contained GPU PPO (actor-critic MLP). Env also satisfies the rsl_rl `VecEnv` protocol + returns `{"policy": obs}` from `reset()`, so it drops into rsl_rl `OnPolicyRunner` or liquid-NN `CfCPPO` with no adapter. |
| `stress_reorient.py` | Stress/NaN-localization harness — applies external "tug" wrenches + max-rate finger slew, steps substep-by-substep, reports the first non-finite body/joint + run-up. |
| `hand_taxels_newton.py` | `HandTaxelSensor` — now **batched** (`pad_total[N,15]`, indexed by `con_worldid`); N=1 keeps the original single-world behavior. |
| `flexpalm_bones4.py` | The hand builder (unchanged mechanism); added `simple_shells` collision option (see below). |

## Observation / action / reward

- **obs (102)**: finger pos (unscaled) 22 · finger vel×0.2 22 · cube pos3/quat4/linvel3/angvel3 13 ·
  goal quat 4 · relative quat 4 · **tactile pad forces 15** · prev action 22.
- **action (22)** ∈ [-1,1]: a *residual around the cupped hold pose* (a=0 holds the cube; a=±1 →
  ±0.35 rad), so the stiff hand stays stable and learning starts from a working grasp.
- **reward**: `1/(|rot_dist|+0.1) − 10·goal_dist − 2e-4·Σa² + 250·[rot_dist≤tol] − 50·[dropped]`,
  `rot_dist = 2·asin(‖vec(q_obj·conj(q_goal))‖)`. Consecutive success resamples the goal without
  ending the episode; terminate on drop or timeout.

## The real NaN root cause: the equality SOLVE (use `soft_loops`)

**The definitive fix.** The mitigations below (`shells_jointed`, buffer sizing, substeps, the
NaN-reset guard) reduced but never eliminated the divergence — a long N=8192 run still crept to
~28–31% NaN and collapsed reward. The actual root cause is the **4 MuJoCo equality constraints**
(3 loop closures + shell weld): the equality solve ill-conditions under contact + exploration and
diverges. **`soft_loops=True`** removes all 4 and closes the loops with explicit **penalty-spring
forces** on `body_f` each substep (a Warp kernel between body-local anchors), staying on
SolverMuJoCo (tactile + CUDA graph intact):

| config (N=8192) | env-steps/s | long-run NaN | reward |
|---|---|---|---|
| equality, substeps=4 | 16.8k | **collapsed → 31%** | −4 → −21 |
| equality, substeps=8 | 16k | **crept → 28%** | flat/creep |
| **`soft_loops`, substeps=8** | **47k** | **0.00% (flat, 200 iters)** | −4.7 → −3.2 ✓ |

Removing the equality solve **both eliminates the NaN and runs ~3× faster** (no constraint solve).
Requires **substeps=8**: at substeps=4 the larger dt blows up the explicit springs (NaN → 62%) — an
explicit-integration limit on the light connector proxies. This is now the **trainer default**
(`--soft-loops`, on by default; `--hard-loops` selects the old equality path). Gains
`LOOP_SPRING_KE/KD/FMAX` (1e4 / 1e2 / 300). Convention note: Newton `body_f` is `[force, torque]`
and `body_qd` is `[linear, angular]` — get either wrong and the wrench pumps energy and blows up.

*(A `solver="kamino"` path was attempted — maximal-coordinate, implicit springs — but is **blocked**:
the Newton→kamino `from_newton` bridge can't represent closed loops. Scaffolding kept but inert.)*

## The high-N stability fix (earlier mitigations — superseded by soft_loops as the cure)

The hand is **underactuated + closed-loop + shelled** — unlike the fully-actuated tree-hands
(Allegro/Shadow) the in-hand task was built for. Replicated naively, mujoco-warp's **batched
contact solve diverged to NaN in a growing fraction of worlds above ~8** (N=8 clean, N=256 ~50%).
Ruled out by stress-testing: buffer sizing (real but secondary bugs, fixed), solver config
(Allegro's settings were *worse*), a velocity watchdog (divergence is instantaneous, not a
runaway), and `SolverKamino` (rejects equality constraints → can't run our closed loops). The hand
*alone* was stable at N=256; **the cube vs the hand's ~108 convex collision parts** was the trigger.

**Fix — `shells_jointed` (default ON for this env):** keep the back/bottom shells as **full-
resolution VISUAL meshes** (topology preserved — no convex-blob hack) but take them **out of
collision entirely**, and connect **Shell1↔Shell2 with one explicit equality constraint** — the
chassis hinge the design implied, made explicit so the contact solver never has to mediate the
shells. The shells are the outer chassis (never touch the cube, already self-filtered), so collision
was only ever a liability. **Fingers + palm bones keep full collision (grasp + tactile), visuals are
unchanged, and the underactuated mechanism — D6 spring palm bones, 3 closed loops, 22 finger
revolutes — is untouched.** (`simple_shells`, collapsing each shell to one convex hull, is the
earlier superseded variant; still available.)

**The joint is a CALIBRATED compliant WELD, not a free ball.** A free `connect` (ball-and-socket)
pivots too loosely — it leaves a **27° rotation gap** vs how the shells actually move under the
prior full-CoACD collision. So Shell1↔Shell2 is a `weld` (constrains position **and** orientation,
`torquescale` weighting rotation, `eq_solref` the compliance), and its 5 free parameters — ball
**center (3)** + **torquescale** + **solref time-constant** — are fit to match the reference
CoACD-collision shell trajectory. The fit is done **batched on the GPU**
(`calibrate_shell_ball.py --batched`): each of N worlds gets a *different* candidate weld
(heterogeneous `begin_world`/`add_builder` build), one rollout scores the whole population, and an
evolutionary loop tightens around the elites — **~49k candidates (pop 4096 × 12 gens)** in minutes.

The pivot is **pinned to the shells' fitted mating-sphere center** (where they physically
articulate) by a geometric regularizer (`SHELL_REG=1.0`), then torquescale + solref are fit to the
trajectory: center `(-0.1206, -0.0298, -0.0082)`, torquescale `0.50`, solref_tc `0.099` →
**pos 15 mm / rot 13.5°** against the reference's own ~38 mm shell drift. (Letting the center float
free reaches **5.9 mm / 5.5°** but parks the pivot ~85 mm off the mating surface; we chose the
physically-placed pivot — still far better than the free ball's 27° rotation gap. Drop `SHELL_REG`
to recover the unconstrained fit.)

**Substeps depends on the weld stiffness.** A *stiff* weld (the unconstrained `torquescale 1.54`
fit) needs `substeps=8`: at `substeps=4` it leaves ~8% of worlds diverging under a worst-case probe,
masked by the NaN guard. The **chosen geometric weld is softer (`torquescale 0.50`)** and tolerates
`substeps=4` — only ~0.1% diverge worst-case at N=8192 (5/8192), and PPO trains healthy. The env
class still **defaults to `substeps=8`** (conservative for `--check`/eval and any future stiffer
weld), but **the trainer defaults to `substeps=4`** (the validated, 2×-faster training config).

Also kept as defenses: contact/triangle-pair buffers sized to world count, a NaN-reset guard
(diverged world → `NAN_PENALTY` + reset, logged), a finite-guarded optimizer step (a non-finite
loss/grad never reaches the weights — can't collapse the policy), and a velocity watchdog.

## Scaling + substeps frontier (RTX 5090, short-PPO sweep, geometric weld)

`HEALTHY` = reward improves + max-NaN < 15% + no collapse; `BROKEN` = reward collapsed / NaN runaway.

| N | substeps | tri-pairs/world | max NaN% | env-steps/s | VRAM | verdict |
|---|---|---|---|---|---|---|
| 512 | **2** | 80k | 30.5% | 5323 | — | **BROKEN** (reward → −22) |
| 512 | 4 | 80k | 1.2% | 2749 | — | HEALTHY |
| 512 | 6 | 80k | 1.4% | 2189 | — | HEALTHY |
| 512 | 8 | 80k | 0.0% | 2058 | — | HEALTHY (cleanest) |
| 2048 | 4 | 30k | 2.3% | 5809 | — | HEALTHY |
| 4096 | 4 | 30k | 2.6% | 7207 | 10.8 GB | HEALTHY |
| **8192** | **4** | **15k** | **~2%** | **11083** | **12.4 GB** | **HEALTHY ← default** |
| 4096 | 8 | 30k | 0.03% | 3905 | — | HEALTHY |
| 8192 | 8 | 15k | 0.07% | 5719 | — | HEALTHY |

**Read-off:** substeps floor is **4** (2 collapses); N scales to **8192** with no stability cost
(per-world divergence is N-independent — NaN% stays ~2% regardless of N) and throughput keeps rising
(11k env-steps/s at N=8192/s4). N=8192 fits in **12.4 GB** of 32 GB, but **N=16384 OOM'd** even at
8k tri-pairs/world (one internal buffer alone wanted 11.4 GB) — so 8192 is the practical ceiling at
these buffer sizes; past it needs a different broad-phase strategy (self-pair exclusion), not just a
leaner buffer. (**Superseded:** this frontier was measured on the *equality* path, whose s4 "winner"
actually collapses long-run — see the soft_loops section above; the real winner is `soft_loops`,
substeps=8, ~47k env-steps/s, 0% NaN.) **Equality-path winner: N=8192, substeps=4** — ~2× the substeps=8 throughput, NaN held ~2% by the
guards, training healthy.

**CUDA-graph capture (default on, `use_graph=True`).** The substep physics loop (`_step_sim`:
collide → solver.step → swap → Warp-kernel velocity watchdog — ~90% of per-step launches) is
captured once into a `wp.ScopedCapture` graph and replayed, killing per-launch + Python-loop
overhead. At N=8192/s4: **11.4k → 16.6k env-steps/s (1.46×)**, correct (graph-vs-eager divergence ≤
the eager-vs-eager atomic-nondeterminism baseline) and NaN-rate-neutral. Capturable because the loop
is Warp-only (the watchdog is a Warp kernel, not `torch.clamp_`); reward/done/NaN-guard/reset stay
eager (host syncs, ~10% of launches). Requires **even substeps** (the double-buffer ping-pong lines
up); odd substeps or any capture failure fall back to eager with a **loud banner** — never silent.

## Verified (RTX 5090, isaac, Newton 1.2.0 / mujoco-warp 3.8)

- `--check` N=64: finite, cube seats palm-up at the cup, tactile fires (all envs), cube held under
  random actions, REAL-grasp gate PASS.
- **Stability (`shells_jointed` weld, substeps=8):** N=256 `--check` PASS (finite, peak pad force
  finite — no masked NaN, cube held 98%); worst-case stress probe (uniform-random action/step)
  **0.4% diverged at N=256, 0.2% at N=512** (vs `simple_shells` 2.7%), handled by the reset guard.
  `simple_shells` alone: **N=16…2048 → 0% NaN** under the gentler `--check` actions, held 97–100%.
- **Throughput** (substeps=8): ~1,400 env-steps/s at N=256; scales with N. VRAM ~3 GB at N=256,
  ~14.6 GB at N=2048 (`simple_shells`); the triangle-pair broad-phase buffer is the VRAM bottleneck
  (`_tripairs_per_world` — lean 30k fits N=4096). The batched weld calibration runs pop=4096 at
  ~17 GB.
- **Profiling** (N=2048): the step is ~100% physics — **collision detection ~77%, solver ~23%**,
  everything else (tactile, obs, reward, reset) <0.5% combined. Both run per substep, so substeps is
  the main speed knob — but the calibrated weld needs 8 for stability (see above), so substeps is not
  free here. The remaining big lever is cutting collision broad-phase work (excluding hand-self
  candidate pairs / lower-vert finger hulls).
- **Training: PPO learns** — in the first iterations the drop rate falls and reward rises as the
  policy learns to hold + manipulate the cube. Full reorientation (success rate up) needs long runs,
  as for any in-hand task.
