# flexpalm_hand — flexible robot hand on a Franka *or* Dorna 2 arm (Newton)

A semi-flexible 3D-printed robot hand (8 spherical-jointed palm bones, 5 fingers, convex-
decomposed shell with self-collision) **mounted on a robot arm flange**, performing a
**reach → whole-hand grasp → lift** of a cube, with **per-pad tactile sensing shown as an
in-viewer fingertip heatmap**. Pure Newton + `SolverMuJoCo`, runs in the `isaac` conda env.

Two arms, selected with `--arm` (same hand, mount, grasp logic and tactile for both):
- **`franka`** (default) — Franka FR3 via `add_urdf`.
- **`dorna`** — a **Dorna 2** (5-DOF) with hollow-6xxx-aluminum mass/inertia tuned to the
  real **5.5 kg** arm weight. The Dorna CAD conversion is **not redistributed** in this repo
  (third-party CAD); point `DEXIGRAB_DORNA_USD` at your own USD conversion to enable it.

## Run

```bash
cd Simulation/newton
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"   # tactile kernel NVRTC JIT
# headless gate: finalize + reach/grasp/lift, asserts a REAL grasp (clean approach + contact + lift)
python flexpalm_arm_demo.py --arm franka --check
# interactive GL viewer. The viewer steps the full schedule.
python flexpalm_arm_demo.py --arm franka --viewer gl
```

The viewer steps a heavy model (56 bodies, MuJoCo solver + per-frame tactile) at ~0.3x
real-time, so the reach→grasp→lift sequence plays out over a few seconds of wall-clock after
the window opens, then holds the lifted cube.

## Files

| File | Role |
|---|---|
| `flexpalm_arm_demo.py` | **The demo.** Builds the arm (`--arm`) + welds the hand to the flange + a graspable cube on a pedestal; scripted reach/grasp/lift schedule; tactile heatmap recolor; `--check` honest gate. `--viewer gl`. |
| `arms.py` | **Arm builders.** `build_franka` / `build_dorna` → an `ArmSpec` (flange, dofs, ready pose, lift dofs, cube/pedestal defaults). For the Dorna: imports the USD, **re-roots** the articulation at the base (it imports rooted at the flange — inverted), and applies **hollow-aluminum mass/inertia** (`compute_inertia_mesh`, ρ=2700, wall thickness solved for 5.5 kg total). `GRASP_POCKET` + `flange_world_at_ready` place the cube in the hand's rigid rest frame so the grasp transfers between arms. |
| `flexpalm_bones4.py` | The hand builder (`Example`). Body/joint/shell/strip wiring, CoACD self-collision, pose battery. Embeddable: `Example(viewer, args, ext_builder=, attach_body=, attach_xform=, finalize=False)`. Standalone `--viewer gl`/`--check` still work. |
| `hand_taxels_newton.py` | `HandTaxelSensor` — standalone-Newton port of the IsaacLab `_hand_taxels.py` taxel sensor; rasterizes MJWarp per-contact data into per-pad normal force. |
| `franka_show.py` | Helper: spawn the Isaac Franka (Panda) arm in the Kit/PhysX viewer to inspect the flange (run from an IsaacLab checkout: `./isaaclab.sh -p franka_show.py`). |
| `robot_hand_poses.py` | **Vendored** pose battery (`CHAINS`, `JointLookup`, `build_poses`, ...; numpy-only). Copy of the IsaacLab demos file, kept here so the demo is self-contained. `flexpalm_bones4` prefers this local copy. |
| `flexpalm.py`, `flexpalm_spheres.py`, `probe_*.py` | Earlier scaffolding / geometry probes (kept for reference). |

## Assets

- The hand geometry ships in [`../assets/`](../assets): `robot_hand_flexpalm.usda` (working
  copy the builder loads), `robot_hand_simready.usda` (untouched original — the palm cover
  strips are read from here), and the Shell2 recovery re-export. Override the asset root with
  `DEXIGRAB_ASSETS`. The Dorna 2 arm USD is **not** included (see above).

## How the demo works

- **Mount (mechanical mate):** the hand's `cover_back` backplate (at the hand's −X wrist end,
  measured plate normal/center baked into `COVER_BACK_NORMAL`/`COVER_BACK_CENTER`) is seated
  **flat and centered** on the flange face, fingers extending down the tool axis. Computed
  purely in the flange-local frame, so it's independent of the arm pose.
- **Reach → grasp → lift:** arm starts poised above, descends onto the cube, the fingers +
  thumb close on it at the digit **convergence point**, then the shoulder (joint 1) raises to
  lift the cube off the pedestal. A free cube + pedestal keeps grasp forces bounded (stable).
- **Tactile:** `HandTaxelSensor.update()` each frame; `model.shape_color` is rewritten so each
  pad ramps from its base color → yellow → red by contact force (`--force-max` is the
  saturation force).

## Key CLI knobs

`--cube <half_m>` `--cubex/--cubey/--cubez` (cube size/pos — default 6 cm at the finger/thumb
convergence so the **thumb is required**), `--cube-density`, `--grasp-max` (close fraction /
grip firmness), `--force-max` (heatmap saturation N), `--mrz` (palm spin about the wrist axis),
`--mdx/--mdy/--mdz` (seat nudge), `--substeps`/`--iters` (sim speed vs accuracy).

Schedule timing is in `Example._schedule` via `getattr` (settle / approach_dur / close_dur /
hold_dur / lift_dur / lift_amt) — edit there to retune the choreography.

## How the cube is placed (and why the grasp transfers between arms)

The mount mate's shortest-arc rotation lands the hand at a **different roll about the tool axis**
on each flange, so a cube placed at a fixed *flange-local* offset ends up in a different spot
relative to the *hand* — on the Dorna it lands under the palm and gets swallowed. Instead the cube
is placed in the hand's **rigid rest frame** `flange ∘ mount` at a universal `GRASP_POCKET`
derived from the working Franka grasp. That reproduces the same hand-relative grasp on any arm.

For the Dorna, lifting via a single revolute (`dof2`) rotates the hand as it moves, so the
**approach** uses a *small* `lift_amt` (near-vertical, so the open hand descends cleanly **around**
the cube) and the **final lift** uses a larger `lift_raise` (the cube is enclosed by then, so the
tilt is harmless). This is the difference between a real grasp and a cube wedged inside the shell.

## Verified — honest `--check` gate (RTX 5090, `isaac` env, Newton 1.2.0)

`--check` asserts a **REAL GRASP = clean approach + contact + lift**, where:
- **clean approach**: with the hand forced *open*, the descending hand disturbs the resting cube
  by <1.5 cm (i.e. it descends *around* the cube, not *through* it — guards against the cube being
  swallowed by / wedged inside the shell, which a lift-only check cannot see);
- **contact**: tactile fires on ≥2 pads; **lift**: the cube rises >2 cm from grasp to end.

| Arm | mass | approach disturbance | pads (thumb) | cube lifted | REAL GRASP |
|---|---|---|---|---|---|
| `franka` | — | 0 mm | 6 (yes) | +160 mm | **PASS** |
| `dorna` | 5.50 kg (hollow Al, 2.9 mm wall) | 8 mm | 9 (yes) | +129 mm | **PASS** |
