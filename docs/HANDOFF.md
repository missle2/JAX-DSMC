# Handoff — state of play (rewritten 2026-10-21)

For someone (human or agent) picking the project up cold. It says what works, what is in progress,
which decisions stand and **why** (with the measurement that settled each), which invariants and
traps will bite, and how to run things. The play-by-play history was removed on 2026-10-21; where a
decision came from a long episode, the episode survives as one or two lines of evidence. The full
measurement ledgers are `docs/PSEUDOCODE.md` (sections 17-23), `docs/ADJOINT_BRANCH_REVIEW.md`,
`docs/math/shape_validation.tex` (the validation deck) and the probes in `scratch/`.

Working rules that explain the shape of everything below: **gates are measured before the code they
constrain** (twice a gate bundled into the implementing agent's task changed the design it was
meant to check); **a confident numerical claim gets a probe, not an argument** (several of this
project's confident conclusions were reversed by measurement); **never trust a Monte Carlo number
without seeds and a standard error**.

## Summary

**What works.**
- A SPARTA-style DSMC forward solver in JAX (NTC + VSS, f64, 1-3 D, periodic / specular / diffuse /
  inflow / outflow, multi-device x-slab decomposition). Validated against Bird/Alsmeyer shock
  thickness, Taylor-Green and Couette viscosity (Chapman-Enskog), BKW relaxation, two-sided-inflow
  Rankine-Hugoniot, and (with embedded bodies) SPARTA argon circle / sphere runs.
- Embedded bodies on the Cartesian grid: cut and split cells, polygon / analytic / **C1 level-set**
  surfaces, three trackers (stencil, SPARTA cell walk, **lane** = the examples' default).
- Pathwise AD adjoint, exact against hand-coded [P1]/[P2] recursions (oracles) and fixed-key FD.
- **The production gradient: the particle-carried ensemble adjoint** ("route b", decided
  2026-09-15), with the virtual-partner collision pull-back and the per-cell guarded cubic-in-v
  field fit (`ensemble_backward.CELL_BASIS = "cell:cubicv:adapt1.5:guard32:align0.9:both"` since
  2026-10-19), plus the shape adjoint (`jax_dsmc/shape.py`). Validated against CRN finite
  differences on inlet shock, Couette walls, VSS parameters, circle / sphere radius, `cos 2 theta`,
  and the Kn 0.5 squash (-0.7 +- 0.8 % vs paired FD, 16 seeds).
- A level-set shape optimiser (`examples/mesh/optimize.py`) and a cluster workflow (CURC Alpine).
  Best runs reach 0.20 (Kn 0.5) and 0.10 (M 10) of the circle's drag at fixed area.

**Production path.** forward `simulate.run(capture=True, mesh=...)` -> `ensemble_device.backward_virtual`
(value columns) -> `shape.gradient` (replay VJP + terms) -> `examples/ensemble_mesh.shape_terms` ->
`examples/mesh/optimize.py` (Adam as an optax chain, `--optimizer ox_adam`, the default since
2026-10-21) -> cluster via `examples/mesh/cluster/`.

**In progress (2026-10-21).** The optimiser comparison. Done (`optrules`, `optrules2`,
`optrules1seed`): Adam is robust even with one seed per iterate and stays the reference; the reworked
trust region (`tr`) is fastest early but parks at its radius floor; L-BFGS curvature pairs are seed
noise; MMA has a gradient-scaling problem and fragments shapes at one seed. Running (`optax1seed`):
optax step rules that change Adam's step SIZE - a cosine step schedule, schedule-free Adam, Prodigy -
plus AdEMAMix and the optax Adam control (section 2.6). Two fixes from these runs landed 2026-10-21: the warm start's
sub-cell collide state and the tracker's post-loop `inside_body` query (section 4.3).

**Known open problems.**
1. Optimised shapes grow **sub-DSMC-cell roughness**: the level set has ~400 boundary degrees of
   freedom, 0.5 m DSMC cells resolve ~50 of them. Step smoothing (`--smooth 8`) hides it, it does
   not remove it. A curvature penalty was proposed, not built.
2. The Kn 0.5 adjoint-vs-FD gap is closed only to about -1 % +- 1 %, and the **optimum is
   ill-conditioned** (squash `s*` moves 0.84 -> 0.62 between fits); thin bodies (< ~2 cells) stay
   about -4 % in every fit.
3. Translation-type controls (x/v part of a moved flat wall) are unresolved; controls are normal
   displacements only.
4. Free-molecular M 10 optimum is bounded by the box (the body slides into the keep-out).
5. Diffuse-wall circle gate (GD) reads 9 % below the precise FD (z -2.3) - not closed.
6. Multi-device scaling has never run on real multi-GPU; 3-D shape optimisation and hierarchical
   grids are not started (`docs/PLAN.md`).

---

## 1. Where things stand

### 1.1 Components

| component | where | state |
|---|---|---|
| Collision (NTC, VSS, thermal majorant) | `jax_dsmc/collide/` | validated (BKW, Taylor-Green, shocks) |
| Transport, BCs, inflow | `move.py`, `boundary.py`, `inflow.py` | validated; `inflow_shortfall` must be 0 |
| Adaptive attempt capacity | `simulate.run_adaptive` | host-side; not inside a gradient |
| Embedded surfaces | `jax_dsmc/mesh/` | gated vs SPARTA (geometry, forward); level set gated (LS0-LS3) |
| Pathwise + score surrogates | `adjoint.py` | oracles exact; score routes not used for production |
| Ensemble adjoint | `ensemble_backward.py` (host reference), `ensemble_device.py` (JIT), `cell_fit.py` | production |
| Shape adjoint | `shape.py`, `wall_adjoint.py` | production; terms listed in 2.4 |
| Optimiser | `examples/mesh/optimize.py`, `optimizers.py`, `optimize_squash.py` | running on the cluster |
| Multi-device | `decomp.py` (works), `parallel.py` (kept as the negative result) | not benchmarked on real GPUs |
| Fast SPARTA-faithful forward | branch `sparta-match` only | never merge into main |

### 1.2 Numbers worth remembering (details in the example READMEs and the validation deck)

| gate | result |
|---|---|
| BKW rate bias, N = 2.56e6, 24 seeds | -0.08 +- 0.12 %, sd ~ N^-0.500 |
| Bird normal shock L/lambda_1 (D = 1/2/3) | 5.98 / 6.12 / 6.18 vs 6.2; vs Alsmeyer 4.8: 4.53 / 4.64 / 4.68 |
| 2-D Taylor-Green viscosity | +0.38 %, R^2 = 0.9994 (exact NS solution, no free parameter) |
| Couette | gate is the Kn -> 0 intercept, with dx -> 0 correction, consistent with Chapman-Enskog; live numbers in `examples/couette/RESULTS.md` |
| SPARTA argon circle duplicate | drag +0.3 % (z = 1.4), surface collision rate 139 vs 140.5 per step, fields agree |
| Free-molecular circle Cd, free box | 12.245 +- 0.036 vs closed form (R 1.5), `scratch/freebox/` |
| Radius gradient, analytic circle | 1.44 +- 0.09 e-20 vs CRN FD 1.40 +- 0.01 (z 0.5) |
| Sphere radius gradient (LS3) | z = -1.75 / -0.91 vs FD (7 seeds) |
| Kn 0.5 squash, shipped fit | -0.7 +- 0.8 % vs paired FD (16 seeds, 300 captured steps, 2026-10-12) |

---

## 2. Decisions that stand, with reasons

### 2.1 Forward solver

- **Stochastic rounding of the attempt count plus a `log P(n given f)` score term** (spec 8.1). The
  deterministic remainder carry froze the path from `dt`, `fnum`, `cell_volume` to the collision
  rate: measured `jax.grad = [0, 0, 0]` against a healthy `||dJ/dv|| = 9.5`.
- **Stale-max NTC (dsmcFoam+ rule), floored by an a-priori thermal majorant** (`ntc.thermal_majorant`,
  `vremax_init`, `K_EV = 5`, `F_UP = 2`; spec 7.2.3, 17.5). SPARTA's own seed measured a 1e-2
  violation rate (min margin 0.62) on a counter-streaming IC; the thermal floor gave zero violations
  across six IC x transient cases, min margin 2.15. Collision counts are `Sigma`-invariant. **Do not
  "simplify" `Sigma` back to `cstate.vremax`**: removing the floor fails no test loudly.
- `EPS_REL_VR2 = 1e-8` stays (spec 17.4): raising it to 1e-6 was measured and rejected.
- **Move, then collide** (SPARTA order), transport term outside `B_k`; [P3] collides first. Pinned
  by Oracle 3 (1.3e-16 vs 9.8e-2) and `test_production_step_moves_before_it_collides`.
- **`A_max` is measured, not guessed** (`simulate.run_adaptive` grows it on `attempts_clipped` and
  re-runs the chunk). Every a-priori rule failed: the cold count was 5.4x too small, and a
  converging wave reached density ratio 11, not Rankine-Hugoniot's 4. Differentiated runs need the
  capacity settled first (host control flow cannot sit inside `jax.grad`).
- **Scale the cell count with N** (`--particles-per-cell`, default 1000): `A_max` is per cell, so
  pinned cells made 640k particles 39x slower (26.4 s vs 0.68 s). Below ~3e5 particles the CPU beats
  the GPU; above ~6e5 the GPU wins 3-4x. Rates agree to every digit across backends.
- `GridSpec.transverse` carries the suppressed extents so `cell_volume` is a volume at D = 1, 2.
- **Diffuse walls**: flux-weighted half-Maxwellian, closed-form hit time on a static mesh (the
  `K_bounce` sub-move scan was 1.65 s vs > 7 min with identical results). Corner handling: resolve
  the first face in time, then a statically gated second pass (applying faces in turn heated a cold
  gas to 350-386 K). Residual: two thermal faces on different axes leave a 3.3e-4 recrossing rate;
  exactly zero for Couette and shock geometries. The ordering in `apply_faces` is load-bearing.
- **Free-space box** (`forward.build(boundary="free")`, 2026-10-15): every face injects the drifting
  Maxwellian's inward flux. Why: the specular channel holds a half-range stream (free-molecular Cd
  17.2-17.7 vs 12.22) and at Kn 0.5 + M 10 chokes for D/H >= 0.3. Not retrofitted to the older
  `optc2` / `optckn` / `optab` / `squash` runs.
- **Output is fields, not particles** (`viz.FieldMovie`; 50 KB/frame vs 1.7 MB at N = 1e6).
- **Multi-device**: x-slab decomposition (`decomp.py`). Sharding the particle axis does not
  parallelise DSMC (global `argsort` replicates the table; collide 6.8 / 13.0 / 25.2 ms at 1/2/4
  devices) - `parallel.py` stays as the measured negative result. Re costs `Re^D` in cells: 4 x H200
  reach Re ~ 1500 in 2-D, ~66 in 3-D.

### 2.2 Which gradient estimator

- **Pathwise AD is exact** for parameters that reach `J` through the reparameterised sampler (`temp`,
  `vstream`, `alpha`, wall `T`, wall velocity): fixed-key FD to 1e-11, Couette closed form. Parameters
  that act only through decisions (`diam`, `omega`, `tref`, `nrho`, accommodation) read **exactly 0**
  pathwise and need a score or the ensemble adjoint.
- **Score-function / DiCE routes are dead for production** (measured 2026-09-10..15): the per-particle
  attribution is biased low and the bias grows with trajectory length (FD/AD = 1.13, 1.32, 1.57,
  1.95, 2.82 at M = 2, 5, 10, 20, 40; exact with collisions off), the trajectory score's SE grows as
  M^2.8, count scores for inlet density are 200-1300x the FD noise, and the shock density-ratio
  gradient has the wrong sign (`docs/ATTRIBUTION.md`, review sections 4 and 5). Terminal-label and
  trajectory scores remain in `adjoint.py`; `baseline="mean"` is rejected (biased).
- **Production: the particle-carried ensemble adjoint, route (b)** (2026-09-15). Carry the value `lambda`
  on particles, use the realised forward pairs as the collision quadrature, interpolate only one
  "ghost" per collided particle. Verified vs fixed-count FD to 0.5 % at 240 seeds; P3's
  retained-frame route (a) has a ~1 % reconstruction bias even at full retention. On the real solver:
  inlet temperature and density vs 384-seed FD (density at 0.43x the FD noise). Pathwise-inside-ensemble
  was measured and rejected (biased, noisier).
- Rules learned the hard way: the ghost interpolant must be at least quadratic (affine: -8 % at 30
  particles/cell); record **every** wall bounce and newborn hit individually (no "last event x count"
  shortcuts); the P3 paper's Test-3 value 0.1517 is not a target (the transcribed forward differs at
  the inlet; its own gradient is 0.144-0.147).
- **Collision pull-back**: the *sequential* form (`configure("sequential")`, ghosts on the current
  chained field) is the exact reference but its noise blows up above ~0.5 collisions/particle/step.
  The shipped default is the **virtual-partner substepped** pull-back, `VIRTUAL = (4, 8)`,
  `VIRTUAL_NODES = "staged"` (2026-09-16): unbiased to < 1 % at S = 4 (residual O(dt/S): +1.8 / +0.7 /
  +0.1 % at S = 2/4/8), per-seed sd 5-56x below the sequential form's. Fit nodes **must** be the NTC
  step's staged intermediate states (end-of-step nodes are -6 % biased, no substep count fixes it).
  The partner count K is a *stability* knob on long horizons (K = 2 blew up 1e5x over 300 steps at 40
  particles/cell; the device path refuses a backward whose rms lambda grows > 100x). Collisionless
  runs use `--substeps 1 --partners 2` (partners only cost) - this is a *setting*, not a code shortcut.
- Variance facts (2026-09-05, score route, still instructive): `sd ~ N^-0.5` verified for four parameters;
  noise is mostly pathwise (3-5x the score half), so a baseline caps at 1.02-1.07x and **more particles
  is the lever**; a spatial window costs signal (up to 6.3x SNR), not variance; at [P2]'s 10 % FD step the
  AD/FD gap is truncation.
- **Never verify an `alpha` gradient near 1.4**: `dT_y/d(alpha)` is stationary at `alpha = sqrt(2)` and
  argon's tabulated 1.40 sits 1 % away. Helium is the best-conditioned gas
  (`species_data.best_for_alpha_study()`); even there the FD check is low-power.

### 2.3 The backward's field fit (`jax_dsmc/cell_fit.py`)

Default `CELL_BASIS = "cell:cubicv:adapt1.5:guard32:align0.9:both"`: ONE least-squares fit per
collision cell (quadratic in x, cubic in v), order capped by node count (`adapt1.5`) and per query by
leverage (`guard32`), with surface normals splitting ahead / behind populations and complex cells
(resultant < 0.9) left two-sided (`both`). History of why:

- 2026-10-11: per-cell fits replaced the per-query kNN fit (`"quadratic_full"`, kept for parity tests
  and old artifact names). A Kn 0.5 squash seed went 985 s -> 95 s (backward 897 -> 13 s). Unguarded
  per-cell quadratics were unstable (seed 11 +97 %, seed 10 -40 %); `adapt1.5` fixed it; the shipped
  one-sided per-query backward was itself unstable on optc iterates (lambda rms ~150x).
- 2026-10-12 (`gapgate`): cubic-in-v closed the Kn 0.5 gap: quadratic -3.7 +- 0.8 %, cubic -0.7 +- 0.8 %
  vs paired FD (paired +3.0 +- 0.2 %). The older "7-9 % gap" was FD noise (64-seed FD: h = 0.1 and 0.2
  agree to 0.14 +- 0.67 %); horizon (1000 vs 300 captured steps) adds +2.1 % and is additive-ish.
- 2026-10-18/19 (`thingate`, `guardgate`): the cubic **blows up on thin bodies at M 10** (squash M 10
  dD/ds -19.6 vs FD +1.48; optm nose spikes, lambda up to 380). Cause: velocity-space *extrapolation* at
  post-collision queries in the empty gap between the beam and wall-reflected gas (leverage ~1e3 for the
  cubic vs ~30 for the quadratic). `guard32` fixes it (max lambda <= 2e-3, blunt squash unchanged,
  < 0.6 % of queries lowered); guard thresholds 2 / 8 hurt the sparse inlet gates; ridge cures but is
  -4 %. Spline fits were judged worse (no data in the gap, 5-D, noise).
- Residuals after the guard: thinnest squash (0.58 m) -3.5..-4.2 % in *every* fit including the quadratic;
  optm iterate 58 nose direction -3.8 +- 0.6 % (quadratic +0.7 +- 1.1 %).
- `value_class="hit_ahead"` (stratified fits, 2026-09-22) fixed the specular capsule but is **biased by
  construction** (drops `grad P (lam1 - lam0)`: mode2 +40 %); it conditions on the future, is unrepairable,
  and now warns. The default everywhere is `"none"`. Note `shape.gradient` is jitted: a new string option
  must be added to `static_argnames`.
- Rejected: the exact-angle `rb` option (+0.48 +- 0.22 %, +13 s/seed; the forward noise dominates), the
  Delta-t-linearised and end-of-step pull-backs (-10 %, -23 %), skipping the fits when attempts are zero
  (rule below).
- **Rule: no runtime collisionless shortcuts.** Collisionlessness is a property of the setup; tests and
  smoke runs exercise the real collisional forward and backward so lowering Kn changes no code.
- A resumed optimiser run keeps its recorded basis (`history.json` `backward_basis`; `PRE_CUBIC_BASIS`
  for old histories); asking for another raises.

### 2.4 Embedded surfaces and the shape gradient

- **Direction: SPARTA-style Cartesian cut / split cells** (not dsmcFoam+ polyhedra: tet-walking is
  hostile to static shapes). Split cells get sub-cells (`GridSpec.n_sub`, `grid.collision_cell`; 2026-09-23).
  Hierarchical grids are deliberately not built; mesh tables are per leaf cell so adding them is
  contained (`docs/PLAN.md`).
- **Controls are normal displacements of curved surfaces only** (2026-09-20). None of four `grad_x lambda`
  estimators resolves the x/v part of a translated flat wall (exact +0.003 on the shear probe; pooled /
  along-flight read 0 with understated SEs). Radius-type controls are FD-consistent only because the tally
  term is exact. Do not propose tangential translations of flat walls.
- **Representation: C1 level set** (`mesh/levelset.py`, uniform cubic B-spline coefficients, single surface
  id), not the 50-gon (its vertex edge-band term is representation-induced) and not density/porous
  bodies (no kinetic meaning for an impermeable wall). Hit search = sampled bracket + bisection + one
  Newton step with the bracket under `stop_gradient` - **the bisection has zero derivative, the Newton
  step carries `d t/d coef`**.
- **Terms of the shape gradient** (`shape.py`, assembled in `ensemble_mesh.shape_terms`): pathwise VJP of
  the replayed wall map; exact tally term; diffuse normal term in the *hybrid* form (the hit's own reward
  pathwise through the reparameterised sampler, value-to-go after the hit on the score; seed sd 3.1 ->
  0.77 e-21); silhouette band term for diffuse walls (count ONE passing per trajectory - counting every
  particle-step in the band over-estimated 5x); window-edge timing term (10-step band default; the 0.2-step
  band's apparent gain was a noise fluke); hit/collision order term (+0.4 % at Kn 0.5); cut-cell volume term
  (-2.9 % at Kn 0.5, gated by volume scaling); along-flight identity with `L* lambda` (fit agrees with T + L*
  to 0.8 %, T alone off by 3.2 %). Default variant `along_pool_plane` (pooled one-sided fits with the
  half-space cut that stops a pool reaching across a thin body: `one_pool` was -19 % at s = 0.35).
  `grazing = eps` floors `1/(v.n)`: it is the *variance* lever (sd ratio scales with it), not the bias;
  0.05 is the default of the cluster jobs.
- **Evidence for the gradient** (details: validation deck): analytic circle radius z 0.5; `cos 2 theta`
  mode z +0.7 / -0.2 (4 seeds); sphere z -1.75 / -0.91; specular capsule 1.356 +- 0.011 vs 1.35 (hit-ahead
  era; now `none`); plate (GP) 1.191 +- 0.068 vs 1.169 +- 0.096; free-molecular squash within ~2 % at
  s = 0.35 / 0.6 / 1; free-box circle radius gradient -0.9 +- 3.6 % vs FD.
  - Compare against the **truncated-horizon FD** and report the horizon separately: on the collisionless
    plane the estimator equals the truncated FD (+0.003 vs +0.0017 +- 0.0026) while the full FD differs by
    the horizon (H = 72 captured steps recovers it).
  - Free-molecular residual -1.5..-2 % is mostly horizon (300 -> 600 captured steps: +0.8 / +1.1 %, y-wall
    re-hits) and not grazing, FD curvature or the tally; unlocated remainder ~ -1 %.
  - Pair every estimator term against the CRN FD **seed by seed** before touching the formulation (that
    decomposition located the hit-ahead bias in one table).
- **Trackers**: `lane` (SPARTA's move loop: pass 0 over all particles, later passes on a compacted
  buffer; `K_BOUNCE = 32`) is the default of the mesh examples, `SimStatic.tracker` itself defaults to
  `"stencil"`. Particles still stuck after the bounce budget are deleted and counted
  (`Diagnostics.surface_stuck`, SPARTA MAXSTUCK); optimised shapes grow concave windward notches
  (curvature to -10) where 5-7 specular bounces happen in one step, which killed a bounded tracker.
  `inside_body` (killed and counted) is the catch-all and must be 0 for a valid run; `inflow_shortfall`
  likewise. Replay check: `replay_error` is relative round-off (~1e-12 on concave shapes), a flipped
  outcome reports 1, runners allow `REPLAY_TOL = 1e-9`.
- **Conventions that must not drift**: element normals point **into the flow** (SPARTA
  `z x (p2 - p1)`); surface ids `2D + e` after the `2D` box faces, an analytic / level-set body is id
  `2D`; `WallParams` row `2D + model` carries `diffuse T acc` (`acc = 0` = specular); `GridSpec.volume`
  carries cut volumes (NTC, moments, fill see them with no other change); SPARTA's `compute grid temp` is
  the un-centred `(m/3k)<v^2>` (`temp_kinetic`); the cell walk agrees with the stencil to round-off,
  not bitwise (it advances to every cell face); the diagnostics `stencil_overrun` and `walk_exhausted`
  must be 0.

### 2.5 Performance

**Forward** (SPARTA circle deck, argon, 1000 steps from empty, this laptop, 2026-10-01; harness
`scratch/perf/bench_forward.py`, SPARTA serial 0.52 s):

| 1000 steps | main code, 1 CPU core, polygon / level set | GPU (RTX 3050 Ti, WSL flags), polygon / level set |
|---|---|---|
| 2026-09-28 | 67.6 / 98.3 s | 15.1 / 20.8 s |
| 2026-10-01 | **6.3 / 11.6 s** (12x / 22x SPARTA) | **4.5 / 6.8 s** |

The non-differentiable SPARTA-faithful port on branch `sparta-match` runs 1.30 s on one core (2.5x SPARTA):
the floor, set by XLA per-op overhead and lock-step padding, not extra work. What changed (each gated by
`scratch/perf/noise_gate.py`, |z| < 3, all < 1): Halley inverse-CDF inflow speed; level-set search with block
cull + safeguarded Newton; compacted rows for polygon search and inside check; on XLA:CPU a
SPARTA-structured pass 0 (`wall_adjoint._first_pass_routed`; GPU keeps `_first_pass_dense`, since the lane's
gathers cost more launches there); NTC without scratch-row copies; `SimStatic.sort = "partition"` (carried
partition sort; same cells, different within-cell order, so seed-level draws differ - tested in
distribution); `inflow.revive_at`.
- **Remaining (2026-10-02 study, `scratch/xla_ops/`)**: ~70 % of a CPU step is ~100 large memory-bound
  passes; *passes* matter, not op count (removing 685 of ~2200 tiny ops was time-neutral in an interleaved
  A/B and was reverted; the host tracer inflates tiny ops). Bounded options total ~20 %
  (6.3 -> ~5 s per 1000 steps); < 5x SPARTA needs the tracker-bookkeeping rewrite. **Paused by the user
  2026-10-02** (risk to verified behaviour). The GPU step is launch-bound (5.3 ms wall, 2.1 ms kernels).
- Use `scratch/xla_ops/ab.sh` / `blocks.py` (interleaved untraced A/B); single 100-step blocks vary +-10 %.
  `jax_dsmc/fusion.py` (`materialise`) holds the one XLA:CPU fusion workaround.

**Adjoint** (one optimiser seed, optc settings):

| date | seed wall time | change |
|---|---|---|
| 2026-10-01 | 99 s | forward work above |
| 2026-10-02 | 64 s | two value backwards as one two-column sweep (`ensemble_device` value columns + unrolled Cholesky `_spd_solve`) |
| 2026-10-15 | 38 s | per-cell fit: value backwards 30.3 -> 2.8 s; shape gradient 26.9 s now 70 % of the seed (launch-bound, ~6950 kernels/step) |
| 2026-10-19 | (H200 2g.35gb slice) warm-up 53-58 s, guarded backward 40 s (unguarded 16), shape 28 s, capture + replay 12 s | guard costs time; leverage via cell tables |
| 2026-10-22 | (laptop GPU, reduced size: fnum x 16, 80 steps) seed 36.1 -> 24.7 s: backward 12.2 -> 9.2 s, shape 10.1 -> 3.8 s | design matrix built from gathered factors (`cell_fit.features`, bitwise equal: the column stack was ~25 % of the backward); traced replay at the record's sizes (`ensemble_mesh.replay_static`: the shape VJP transposed 32 passes over a lane of N/32, ~85 % of the shape step; gradient within 1e-17). Then shape 3.8 -> 2.2 s: a lean record's post-collision velocities from one scatter-max of the accepted writes instead of a 256-slot replay (`ntc.post_collision_velocities`, bitwise; ~1 750 launches a step, also in the backward), and 4 captured steps at a time through `vmap` (`shape._batched_scan`, `shape_batch`; each runtime branch taken once per batch, `jax_dsmc/step_axis.py`; 331 k -> 63 k kernels, gradient within 2e-15). Production numbers: `speedprof` |

- **Warm start** (`--restart-steps R`, 2026-10-19): each seed restarts from its previous iterate's final gas,
  particles inside the new body removed. After a drag-halving shape change the carried gas is within
  3.0 +- 0.5 % of a cold run in 50 steps and within noise after; coupling across iterates as good as cold
  (5.6 vs 6.1 % spread); unbiased (+0.4 +- 1.9 %). R = 500 replaces 2800 warm-up steps. M 10 S4:
  516 -> 307 s/iterate. Default 0 (cold), set by the `optrules2`/`optrules1seed` groups.
- Basis is not the cost driver for kNN fits (assembly is basis-independent; affine only 1.37x cheaper and
  moves lambda 73 %). f64 runs at 1/64 rate on consumer/RTX GPUs vs 1/2 on A100/H200: collisional adjoint
  runs belong on A100/H200. XLA command buffers (CUDA graphs) are on on the cluster since 2026-10-22
  (`cluster/env.sh` used to export the laptop's WSL guard flag; run_locked.sh keeps it locally); the
  `speedprof` group times off / default / + WHILE, CONDITIONAL (`scratch/shapefast/cluster_job.sh`).
- **Where a seed's time goes** (2026-10-22, `scratch/shapefast/`, kernel profile mapped to source lines):
  the shape step is launch-bound (~4000 kernels per captured step, 0.9 s of GPU work in a 10 s call) and
  was mostly the transpose of the traced replay's fixed 32 bounce passes over its oversized lane -- now
  cut to the record's needs (`replay_static`; sizes only grow within a process, since each new size
  recompiles the shape VJP; the history's `replay_sizes` shows them per iterate; a cut size the record
  outgrows fails the replay check and reruns at the forward's sizes, so a miss costs time, never
  correctness). Batching the steps does not make the full sizes cheap: 32 passes at 4 steps a time
  took 3.8 s against 2.2 s at the record's 8. The backward is compute-bound: the guard's per-query leverage is still
  ~30 % of it (`cell_fit._leverage_tables`); summing the shape step's six pull-backs into one saved
  3 % (not adopted); the cut-volume term costs 0.1 s.
- Memory: full adjoint 64 kB per slot per captured step (lean record, 2026-09-24); the fine circle
  (240 k slots x 300 steps) needs > 18.7 GB; the coarse `optc` fits a 24 GB RTX MIG slice.
- Recompiles: `npc_max` (backward's per-cell table width) is rounded to `ensemble_mesh.NPC_BUCKET = 64` and
  is static; each new bucket recompiles backward and shape gradient (~1200 s on H200). A larger bucket or
  a configured floor trades padding for fewer compiles.
- Test-suite performance (CPU default, XLA persistent cache `.pytest_cache/jax`): see section 5.

### 2.6 The shape optimiser (`examples/mesh/optimize.py`)

- **Step**: coefficient gradient smoothed on the coefficient grid (Gaussian of `--smooth` knots; the H1
  gradient), restricted to a band, `phi <- phi - eta g` with the largest knot moving exactly `--step`
  (0.05 m; normalised, so late iterates dither without decay). `reinitialise` every iterate (it must
  project the band knots all the way: stopping short polluted the surface cloud and made spurious zero
  crossings at iterate 2). Adam's elementwise normalisation discards H1 smoothing, so Adam's step is
  re-smoothed (otherwise overhangs -> `wall_recrossed`).
- **Area**: `--area-mode tangent` (exact area): gradient and step projected off `grad A` (Lagrange
  multiplier per iterate), a uniform shift restores the O(step^2) remainder (0.5 mm vs 1-2 cm per Adam
  iterate in `projection`). `lagrangian` is **void** (multiplier wound up ~400x the price, area 4-220 % of
  A0 with a fixed-size step); `penalty` is built but unused. `projection` remains the code default.
- **Smoothing**: `--smooth 8` (50 cm) chosen 2026-10-15 (`optab` round 2: smoothest shapes, best adjoint
  check - observed / predicted step 0.88 vs 0.45 in the projection era). A coarser level set equals
  smoothing at equal width (no gain).
- **Box**: `--case circle_long` (x in [-10, 20] m, centred on the body, `fnum x2`, 120 k slots, warm-up 2800
  steps; 90 k slots overflowed -> `inflow_shortfall`), `--keepout 1.0`, `--boundary free` for the `optm`
  family.
- **Stops**: converged when the last 5 objectives lie within 0.125 % of their mean (`window_converged`),
  cap `study.MAX_ITERATIONS = 300`. A constant-step Adam/GD random-walks near the optimum, so these stops
  are at the noise level. `optimize.py` ends with `os._exit(0)` (converged M 10 jobs hung at interpreter
  teardown).
- **Defaults (2026-10-21).** `default_config()` is `optimizer="ox_adam"`, `area_mode="tangent"`,
  `step=0.05`, `variant="along_pool_plane"`, `timing_band=10`; but still `smooth=2`, `restart_steps=0`,
  `seeds_per_iterate=2`, which are NOT the production settings. The cluster groups (`cluster/study.py`)
  pass the production flags: `--step 0.05 --smooth 8 --band 1.0 --keepout 1.0 --area-mode tangent
  --boundary free --case circle_long --fnum-scale 2 --warmup 2800 --grazing 0.05 --levelset-h 0.0625
  --levelset-samples 16` plus `--restart-steps 500` from `optrules2` on (the old `opt`/`optc` groups pin
  `--area-mode projection`). `--optimizer` accepts `gd`, `adam` (numpy), `tr`, `lbfgs`, `mma`
  (`optimizers.RULES`) and the optax arms `ox_adam`, `ox_ademamix`, `ox_adamcos`, `ox_sfadam`,
  `ox_prodigy` (`optax_rules.ARMS`; optax imported only for these); all but `gd`/`adam` need
  `--area-mode tangent`.
- **optax (2026-10-21, `examples/mesh/optax_rules.py`, `scratch/optax/`).** Our Adam core equals
  `optax.scale_by_adam(eps=1e-30, eps_root=0)` and our L-BFGS two-loop equals `optax.scale_by_lbfgs`
  (1e-16); the whole shipped Adam step (smooth -> Adam -> smooth -> area tangent -> max-norm `step`) as an
  optax chain (`ox_adam`) reproduces `adam` row for row. optax's default `eps = 1e-8` is ~1 % of a Kn 0.5
  coefficient gradient - a drop-in `optax.adam` is NOT our Adam. The optimiser step costs ~60-90 ms
  against 90-500 s per DSMC iterate: speed is irrelevant, optax buys flexibility. **The max-norm rescale
  to `step` erases any rule's step magnitude** (schedules, learning-rate-free estimates, AMSGrad/RAdam
  denominators): only direction changes survive it, so magnitude rules need a schedule on `step`
  (`ox_adamcos`), an averaged wrapper (`ox_sfadam`) or a capped learning rate (`ox_prodigy`, whose
  estimate starts at `0.1 step` and can only grow).
- **Optimiser comparison** (optrules, S = 4, best J/J0 at Kn 0.5 deck speed / Kn 0.5 + M 10; 2026-10-20):
  Adam 0.203 / 0.105; trust region 0.311 / 0.222; L-BFGS 0.80 / 0.41; MMA 0.65 / 0.45.
  - Adam: fixed step + momentum is noise-robust; overshoots near the optimum (M 10 rises after iterate 128).
    **Kept unchanged as the reference** (user decision).
  - `tr`: fastest per iterate (0.35 vs Adam 0.61 at iterate 25), but the radius random-walked to its floor
    (x1.5 / x0.5 on a rho centred at 0.5) and a rejection at the floor retried the identical trial on the
    identical seeds forever - the "converged" flag was fake. **Rework 2026-10-20** (user-chosen): rho
    pooled over 5 trials at the radius from paired per-seed differences, radius x2 / x0.5 only beyond rho's
    error, rejection only beyond the noise, a floor rejection re-evaluates the base on new seeds
    (`seed_shift`, `optimizers.judge`). Partial results 2026-10-21: still parks at the floor in 58-86 % of
    iterates; at M 10 the pooled rho is significantly negative near J/J0 0.24; S1 `tr` and L-BFGS drift up
    after iterate ~50.
  - L-BFGS: `y` is seed noise (cos(s, y) ~ 1e-3); steps move 1-3 coefficients. Kept as a reference.
  - MMA: |g| ~ 1e-6 per coefficient is far below Svanberg's 1e-5/span regulariser at Kn 0.5; no
    acceptance at M 10. No GCMMA (forward too expensive; user decision).
  - One seed per iterate: S1 Adam tracks S4 Adam per iterate at 4x lower cost per iterate - gradient noise is
    not Adam's limit. Judge rules by noise robustness first (3-D gradients will be noisier). Final S1 reading
    (2026-10-21): S1 Adam at M 10 reaches ~0.10-0.11 J0 at ~iterate 150 then climbs to ~0.19 by 330, the
    S4 overshoot again - Adam's limit is its fixed step SIZE (too small early: `tr` was 2x faster to 0.35;
    too large late), which is what `optax1seed` tests. S1 MMA fragments (15 pieces at M 10) and roughens
    (total turning 25 at Kn 0.5). Figures `optrules_noise*.png`, movies `figures/movie/optrules_*_noise.mp4`.
- **Regime findings** (`optm`): Kn 0.5: 0.21 of circle drag, flat since ~140; the observed dD vs adjoint
  slope S/(S + rho) with rho 4.8 is noise dilution, not bias. Free-molecular M 10: a wedge whose flat base
  receives no hits, so area migrates into the shadow and the body slides downstream into the keep-out
  (iterate 254, then drag rises 0.104 -> 0.15) - the box bounds this optimum; enlarge it or constrain the
  base before quoting the wedge. Kn 0.5 + M 10: nose gradient spikes 70-14000x, Adam's momentum carries a
  spike's direction 5-25 iterates; the nose grows a sting < 2 cells thick (the thin-body regime fixed by
  `guard32`; re-check after the guard).
- **Roughness** (2026-10-21, `scratch/optrules/roughness.py`, `reinit_drift.py`, `noise_shapes.py`;
  figures `examples/mesh/figures/optc/optrules_roughness.png`, `optrules_noise_shapes.png`): see open
  problem 1 in the summary.

### 2.7 Cluster and process

- **Cluster = CURC Alpine** (`examples/mesh/cluster/`: `study.py` job matrix by group, `submit_curc.sh`,
  `job.sbatch`, `env.sh`, `preflight.sh`, `pull_results.sh`, `collect.py`, `check_env.*`). JAX from a CUDA
  venv at `/projects/$USER/jax-env` (jax 0.6.2; the MI100 `jax` module is 0.4.7 and the code needs >= 0.4.36).
  `submit_curc.sh` and `env.sh` default to `GPU_KIND=a100` (the default since 2026-10-21; jobs that need
  more than 40 GB or speed go to `GPU_KIND=h200`, capped at 4 concurrent jobs); recent groups used `h200` (the
  default there is the 2g.35gb MIG slice; `grid_scale 2` needs `GRES_TYPE=h200`); `rtx` = RTX PRO 6000 MIG
  slices (24 GB, one per job).
- **Since 2026-10-21 run data is not in git.** `examples/mesh/results/optimize*/`, `cluster/logs/`,
  `bad_nodes.txt`, `scratch/*/cluster/`, warm starts and `opt_state.npz` are gitignored and move **only**
  by `bash examples/mesh/cluster/pull_results.sh` (`DRY=1` to list, `WARM=1` for warm starts, optional path
  arguments; one ssh connection, pull only, `--update`). The cluster checkout only `git pull`s. One-time
  adoption of the untracking on the cluster: `adopt_untracked.sh`. Git carries code, docs, figures,
  analyses. (Reason: tracking re-committed every history on every pull; `.git` reached 2.1 GB.)
- **GPU node failure (2026-09-22)**: all `ah200` jobs landed on one node whose driver refuses `cuInit`;
  JAX fell back to CPU silently (6900 s vs ~1800 s per seed; results valid). Now `env.sh` pins
  `JAX_PLATFORMS`, `job.sbatch` runs `check_env --quick`, failing nodes go to `bad_nodes.txt`, which
  `submit_curc.sh` excludes.
- **Preflight policy** (user, 2026-10-17): no long local toy runs before handing over cluster commands.
  `CHECK=1` resolution of each job plus `bash -n` is enough; `submit_curc.sh preflight <group>` runs the
  heavy version on the cluster as an option. Preflight toys cannot catch slot limits or memory.
- Every cluster command resumes from its artifact; a job that hits its time limit is resubmitted with the
  same `submit_curc.sh <group>` line. Groups: `sphere mode2 opt optc squash squashkn squashkn2 biasgate
  cellgate gapgate thingate guardgate shapeprof optckn optc2 optab optm optrules optrules2 optrules1seed`.
- **This laptop is shared and weak**: at most 1-2 JAX CPU processes at once (four concurrent host pilots
  stalled everything for 3.5 h on 2026-09-16); never chain multi-hour GPU jobs unattended (2026-09-20: the
  machine was unusable for 5+ h); benchmarks alone on the machine (a concurrent GPU job produced a phantom
  36x "cliff"). `/home/aakash/jax_dsmc/run_locked.sh gpu|cpu CMD...` (outside the repo) serialises JAX work
  (one GPU job, two CPU slots; sets the WSL flags).
- Math documentation: `docs/math/` - derivation deck (current estimator only), code twin (same frames with
  code references; edit `shape_derivation_body.tex` only) and the validation deck (full history; new
  findings go there). Build with explicit paths, not `cd`.

---

## 3. Invariants you must not break

These make `jax.grad` and the ensemble adjoint reproduce the published algorithms; violating one gives
wrong gradients, not an error.

1. **The RNG contract (spec 1).** Keys derive only from `(base_key, step, STAGE_ID)` via `fold_in`, never
   from traced state, and are recomputed rather than carried: common random numbers for FD, bit-identical
   replay under `jax.checkpoint`, no gradient path through the RNG. One batched draw per stage, consumed as
   the scan's `xs` - **no RNG inside the attempt-slot scan** (hoisting it cut compile 17 s -> 2.4 s and the
   whole suite 26 min -> 4; `tests/test_perf.py` asserts it structurally).
2. **Frozen decisions (spec 0.5).** Cell membership, sort permutation, attempt count, pair choice,
   accept/reject, face crossing are integers and carry no gradient by design; their sensitivity travels by
   score terms or the ensemble adjoint.
3. **Static shapes.** Loops are `scan` / `fori_loop` with static bounds (`n_steps`, `A_max`, `K_bounce`);
   data-dependent counts use a static capacity plus an `alive` mask. The two sanctioned `while_loop`s are
   never differentiated through: the lane tracker's forward (the traced replay uses `fori_loop`) and the
   level-set hit search (under `stop_gradient`).
4. **No nested function definitions (spec 0.6).** Scan bodies are module-level functions bound with
   `functools.partial`, so they are independently testable and their captured state is explicit. (This is
   the rule `jax_dsmc/mesh/walk.py` cites as "HANDOFF section 3".)
5. **The mesh interface is narrow (spec 4)**: `cell_of`, `cell_volume`, `cell_bounds`, `apply_bc`.
   Cell-index arithmetic must not leak into `collide` or `sample`. New `jax_dsmc.mesh` tables are laid out
   per leaf cell, never by `grid.n` arithmetic.
6. **f64 throughout** (`jax_dsmc` enables `jax_enable_x64` on import); conservation tests assert ~1e-14.
7. **No module may take a stdlib module's name** (`types.py` once shadowed `types` whenever the cwd was
   `jax_dsmc/`; the module is now `state.py`).
8. **Main stays differentiable.** Anything that would break `jax.grad` of the main forward lives on a
   branch (`sparta-match`), never in main; port only differentiability-preserving ideas.
9. **Every captured event is recorded individually** (wall bounces, newborn hits), and **the collisional
   code path is always exercised** (no "skip when zero" shortcuts, section 2.3).
10. **Thermal majorant floor, mesh conventions, tracker diagnostics at zero** - see 2.1 and 2.4.
11. A flip guard (`Diagnostics.decision_checksum`) must cover every discrete decision and **weight none by
    zero** (see 4.2).

---

## 4. Traps (each cost real time)

### 4.1 Statistics

- **Seed counts.** BKW's fitted rate has ~13 % sd at N = 1e4: 1, 5 and 8 seeds each showed a confident wrong
  trend (+5.78 % at 8 seeds, z 3.4; 16 seeds: +3.4 +- 1.3 % with the dt trend gone). Use >= 16 seeds,
  quote the standard error, and use independent estimators (`ols_full`, `ols_half`, `ratio`).
  `examples/couette` raises on `seeds < 2` and sets warm-up in diffusion times (`L^2/nu`), not steps; its
  gate is the `Kn -> 0` intercept (the "0.24 % from Chapman-Enskog" claim was retracted: one draw of a
  quantity with several per cent scatter, then misattributed four times). A 2-seed SE is not an error bar;
  use >= 4 for anything quoted. Numbers live in `examples/couette/RESULTS.md`, regenerated from logs.
- **"Flat" from an under-powered study** hid a 1.35 % cell-size bias (dx/lambda 0.6 / 0.3 / 0.15 read
  0.935 / 0.973 / 0.983 once ensembles were used).
- **Which mean free path** (spec 21.1): three definitions differ by up to 60 %; Bird's (4.52) is
  `[(alpha+1)(alpha+2)/(6 alpha)] lambda_HS(d_ref)`, a function of `alpha` only. A wrong factor gave a
  24 % false discrepancy that took a day. Reproducing one quoted number is not verification.
- **Noise floors** (turbulence): on a scalar subtract the constant floor `1.5 v_th^2/npc`
  (`validate.energy_history`); on a **spectrum never subtract it** (survivorship sets the slope: -0.71 vs
  -2) - lower it (`--ensemble M` = M x npc). The floor is usually not the binding limit: a single-mode IC
  combs `E(k)` (factor 12; use `--ic broadband`), Re = 61 puts dissipation at k ~ 10 (check
  `validate.dissipation_wavenumber`, `Re ~ Ma`), and a 1-D shell holds two modes so `E(k)` has ~100 %
  scatter whatever N (use `--ic-ensemble R`, quote `fit_slope`'s error bar). Fit windows must be chosen
  independent of the data (`validate.clean_window`: contiguous from k = 1, above the spectral peak).
- **Benchmarks at small size are noise** (ratios 1.02 / 1.40 / 1.69 / 0.14 for one configuration); the box
  varies ~1.6x run-to-run. Per-test pytest durations are not cost attribution (a shared XLA compile is
  billed to whichever test reaches it first).
- **A Python float argument recompiles per value** (150x slowdown in a 13-point sweep); wrap sweeps in
  `jnp.asarray`.

### 4.2 Numerics and AD

- **Silent failures** (no error, no NaN, no diagnostic): the zero collision-count gradients (exactly
  `[0, 0, 0]`); empty cells indexing a neighbour (`randint(key, shape, 0, 0)` returns 0 - 176/300 runs broke
  momentum; fixed with a clamp and a scratch row); `inflow.inject` dropping entrants when dead slots ran
  out (-46 % density ratio at Mach 4); `A_max` clipping (a hand-picked 64 against a peak of 73 ran with
  `attempts_clipped = 9` every step); per-device capacity / migration overflow in `decomp` (lost or duplicated
  particles while reporting zero overflow; invisible in 1-D). **Test by counting.**
- **FD at frozen randomness is meaningless unless no decision flips.** At fixed randomness `J` is a step
  function in any parameter feeding a discrete decision (FD read 0.0 on a broken gradient). The flip guard
  needed the specular boundary fold index and `scatter`'s degenerate-frame branch, and had weighted index 0
  by `mod(0*phi, 1) = 0`, dropping all of cell 0 (FD 346 vs true 127 with the guard silent). Diagnosis route
  that worked: scan `J(theta)` finely to prove a jump, bisect on `n_steps`, diff every intermediate array.
  Spec 13.4 mandates both FD modes.
- **Where branches are guarded**, make the denominator safe *first* (`d_safe = where(d > eps, d, 1)`);
  `where` alone still yields NaN gradients. `safe.py` is the highest-risk module.
- **Oracle 1 cannot run against `vss.scatter`**: [P1]'s `A` is rank 4 (lab-frame sigma); SPARTA samples the
  deflection in the `v_r` frame, giving a rank-6 pair Jacobian (53-81 % different at Maxwell molecules).
  Oracle 1 runs [P1]'s own lab-frame map. Which reparameterisation has lower gradient variance is
  unmeasured (U10).
- **Oracle 3 needs both `x_M` and `v_M` in the objective**, else `alpha == 0` and it degenerates into Oracle 1
  while still passing. The oracle alone does not pin production: `test_production_step_moves_before_it_collides`
  does.
- **Reverse-pass cost** of plain AD is 3-5x forward (the "17x" was a `vmap` artifact; `collide` is 90 % of the
  forward). AD memory ~4300 bytes/particle at `A_max` 256 (~30x forward) - the cap on AD-route runs; the
  ensemble adjoint's records are the production budget (2.5).
- **Forward random stream changed on the adjoint branch** (per-slot `fold_in` in `ntc.collide`, inverse-CDF
  inlet sampling, one draw per wall pass). Artifacts older than 2026-09-15 are statistically, not bitwise,
  reproducible. `examples/gradient_bench/provenance.py` re-evaluates seed 0 of every stored artifact against
  the current code; run it before trusting a figure. Artifact names carry the estimator (`_virt{S}x{K}st`,
  basis tag from `ensemble_backward.basis_tag`; untagged = sequential / per-query).
- **`terminal_objective` / `windowed_objective` default to `score_mode="trajectory"`**, `dice_reward` to
  `"paper"`; pin `"paper"` explicitly in any example whose numbers are documented.
- A pair of weak-signal checks is not a verification (helium alpha: `abs(FD)/SE = 0.57`).

### 4.3 Mesh, tracker, shape

- **Reward-to-go**: a hit's score weight must include the hit's *own* reward (from its own draw onward) or
  the windowed-shear gate reads 40x too small.
- **Translation controls** are unresolved (2.4); compare shape estimators against *truncated* FD.
- **Warm start and sub-cells (2026-10-21)**: `examples/ensemble_mesh._carried` must keep only the Cartesian
  cells of the carried `CollideState` (it was sized for the previous mesh's sub-cells).
- **`inside_body` after the bounce budget (2026-10-21)**: the tracker's post-loop "unresolved?" query sampled
  the whole reach (spacing h/2) and missed a sliver re-entry after the last allowed bounce (32 specular
  bounces along the M 10 nose); it now samples the remaining move (`wall_adjoint._lane_distance`). An
  earlier `inside_body` source was the level-set search starting ON the surface after a reflection
  (`levelset.hit_body` now counts a leaving on-surface start as phi = +0).
- **Cluster inflow slots**: optimised bodies change the box's filling; capacity too small -> `inflow_shortfall`
  (Invalid DSMC trajectory) mid-run, which preflight toys cannot catch.
- **`jax.clear_caches()` between gates** in one process: `ensemble_couette` re-jits a nested `score` per seed
  and gates in one process OOM. `cell_fit.TRACE` needs `JAX_PLATFORMS=cuda,cpu` (a `debug.callback`).
- Laptop GPU: 4 GB (RTX 3050 Ti): `fnum` 1 Kn 0.5 squash does not fit, M 300 is the max; one `shape_terms` per
  process; chain timings belong on the cluster (`shapeprof` group). A WSL `CUDA_ERROR_UNKNOWN` killed one batch.
- **The WSL CUDA-graph crash**: XLA command buffers crash in the WSL driver on the polygon-mesh step with
  an inflow source. Run with `XLA_FLAGS=--xla_gpu_enable_command_buffer=` (+
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`); `forward.py`, `ensemble_mesh.py`, `optimize.py`,
  `optimize_squash.py` set it themselves.

### 4.4 Optimiser

- Never accept a `tr` "converged" flag without checking for repeated identical rows.
- Judge a rule by seed-noise robustness first; do not conclude from 2 seeds (the restitution test flipped).
- `lagrangian` area mode is void; use `tangent`.
- Optimised shapes: check both `inside_body` and `surface_stuck` per run (section 2.4) and the keep-out
  against the box (the M 10 wedge).

### 4.5 Process

- **A gate runs BEFORE the deliverable it gates**, as separate standalone work: U1 and U3 were bundled into
  the agents implementing `safe.py` / `ntc.py`, and U3's answer changed `ntc.py`'s design.
- A subagent study can be correct and still recommend the wrong action if it measured a conditional peak
  rather than a probability-weighted quantity - re-read the gate, not just the recommendation.
- **Verify that a described fix is wired**: three fixes were explained in prose while the pseudocode still had
  the defect (`logP` computed and dropped). A named check per fix caught it; spot-checking did not.
- Never `pkill -f <script>` / `pgrep -f` from an agent shell (the pattern matches the agent's own shell and
  kills it; a `pgrep -fc` wait loop never ends). Wait on a PID or `/proc/<pid>` or a log marker.
- Batch edits that abort before writing lose everything applied earlier in the batch; write per edit.
- Parallel agents hit usage limits repeatedly; sequence by dependency and keep parallel width modest.
- Background jobs die with the session: relaunch with `nohup` and resume artifacts.

---

## 5. How to run things

All from the repository root; the environment is the venv described in `README.md`.

```bash
pytest -q                          # CPU, 4 host devices, XLA cache in .pytest_cache/jax
pytest --gpu -q                    # on the accelerator: before a milestone, not in the edit loop
pytest tests/test_perf.py tests/test_perf_budget.py            # structure + XLA cost budgets
python3 -m tests.test_perf_budget --print-baseline             # refresh budgets (per backend)
python3 -m tests.test_perf_budget --compile-time               # report, not a gate
python3 docs/phase0_verification.py [--milestone M3]           # structural checks + gap register
ruff check . && ruff format --check .
```

- **Why the suite runs on CPU**: every test is a tiny correctness case where GPU compilation dominates, and
  `tests/test_decomp.py` needs >= 2 devices (four forced host devices under the CPU default). Measured
  599 s (GPU) vs 329 s (CPU) for identical tests (2026-09, since trimmed; do not trust old totals).
- **Perf budgets** (`tests/test_perf_budget.py`) gate XLA's `cost_analysis` / `memory_analysis` (flops,
  bytes, peak temp) of the compiled program - bit-identical across runs, per-backend baselines. Wall-clock
  and compile-time are not gated (shared hardware, in-process caches).
- **Unverified-claims register** (`docs/phase0_verification.py`, spec 19): open rows are U4 (`A_max` sizing
  from the inflated `Sigma`), U7 (non-uniform `fnum` mass guard), U8, U9, U10; U2 and U11 are settled (score
  route biased; ensemble adjoint is the answer); U1, U3, U5, U6 closed. U4 / U6 / U7 are the silent-failure
  class and blocking. A row is deleted only by moving its evidence into the spec.
- **Shape gradient locally** (GPU; one process at a time):
  `python3 -m examples.ensemble_mesh --body levelset --parameter radius|mode2 --substeps 1 --partners 2`;
  per seed ~515 s (level set, all four variants + FD) on the laptop. Collisional cases use the default
  `--substeps 4 --partners 8`. `--backward-basis` selects the field fit (default `CELL_BASIS`).
- **Optimiser locally**: `python3 -m examples.mesh.optimize --iterations K --seeds-per-iterate S [--optimizer
  ox_adam] --tag T` (resumable under `examples/mesh/results/optimize/<tag>/`; `--plot` draws the figure;
  `examples.mesh.movie --tag T` makes the GIF/MP4). Production flags: section 2.6.
- **Cluster**: see `examples/mesh/cluster/README.md` and `submit_curc.sh` header. Typical session:
  `git pull` on the cluster; `[GPU_KIND=h200|rtx] bash examples/mesh/cluster/submit_curc.sh <group>` (default A100) on a login
  node (`module load slurm/alpine`); later `bash examples/mesh/cluster/pull_results.sh` on the laptop, then
  the group's analysis script in `scratch/<group>/`.
- **Gradient estimator artifacts / provenance**: `python3 -m examples.gradient_bench.report`;
  `examples/gradient_bench/provenance.py`.
- **Math decks**: `docs/math/README.md`.

---

## 6. Open problems and next steps

Ordered roughly by value.

1. **Finish the optimiser comparison** (`optax1seed` running; `optrules*` done): decide the rule for
   noisier 3-D gradients. Step-size control is the open lever (`ox_adamcos`, `ox_sfadam`, `ox_prodigy`);
   not built: a curvature penalty or shape filter (roughness), momentum reset on gradient spikes.
2. **Roughness / design-space size**: ~400 boundary DOF vs ~50 resolvable by 0.5 m cells.
   `scratch/optrules/roughness.py` measures total |turning| per run; `reinit_drift.py` tests whether
   reinitialisation alone roughens a shape. Candidates: coarser control (`--levelset-h`), wider `--smooth`,
   a curvature penalty.
3. **Thin-body residuals** (-4 % at 0.58 m, optm nose -3.8 %): unexplained; hypotheses not yet tested beyond
   the fit order.
4. **Enlarge the free-molecular M 10 box** (or constrain the base) before quoting the wedge.
   2026-10-23 (`scratch/optax/`, figures `optc/optax_*.png`, movies `movie/optax_{all,present}_*.mp4`):
   every Adam-type one-seed run at Kn 0.5 M 10 reached ~0.11-0.12 J0 by iterates 60-230 and then
   slid ~10 m downstream (centroid x 5 -> 16) until its base flattened on the outflow face, drag
   back to ~0.2-0.25 J0 -- a systematic pull (the four Adam variants drift alike, the one-seed scatter
   is ~0.004 J0), most likely the bow shock's interaction with the inflow face.  `--fix-centroid`
   is built (a mirror-symmetry option was dropped: L/D and heat-flux objectives need asymmetric
   shapes); the `optcm` group tests it with Adam, Adam + cosine, Prodigy and `ox_prodigycos`.
   Verdict of the optax arms: Adam (= ox_adam) remains the best at Kn 0.5 M 0.3 (0.20 J0); the
   cosine step is the fastest early (M 10: 0.12 J0 by iterate 60); AdEMAMix ~ Adam; schedule-free
   Adam slow (0.25); Prodigy unstable (its step grows: 0.21 -> 0.34 J0); L-BFGS / MMA stall at
   0.6-0.7 J0 at M 0.3.  Four seeds per iterate change Adam's per-iterate path little (drag, step
   consistency and drift all alike): the step's direction is not noise-limited.
5. **Cluster speed**: `speedprof` (XLA command buffers on native-Linux GPUs -- on in env.sh since
   2026-10-22, unmeasured --, `shape_batch` 1 / 4 / 12 with peak memory, and the 2026-10-22
   shape-step speed-ups at production size); set the `shape_batch` default from it (4 is a guess: ~1 GB per
   extra step at M 10 sizes, scaled from the laptop -- tight next to the ~18 GB runs on a 24 GB RTX). Next candidates: the guard's leverage (~30 % of the
   backward; exact skip at the fit's own nodes, where the leverage is <= 1), and particle slots
   (195 k slots for ~131 k live particles at M 10: every per-slot pass pays for the empty third).
6. **Retrofit `--boundary free`** to the older studies if they are to be compared (`optc2`, `optckn`, `optab`,
   `squash`).
7. 3-D shape optimisation (sphere gradient is gated; no optimiser run), hierarchical grids, real multi-GPU
   scaling numbers, thermal-wall epsilon adjoint: `docs/PLAN.md`.

---

## 7. Superseded - do not rely on

- Score-function / DiCE gradients as production (2.2); the sequential and Delta-t-linearised pull-backs as
  defaults (2.2); the per-query kNN `quadratic_full` fit as default (2.3, 2026-10-11); the unguarded cubic
  (2026-10-19).
- `value_class="hit_ahead"` (2026-10-02: biased by construction); the "stratified fit fixed the capsule" story
  is true of that one case only.
- The "Kn 0.5 adjoint is 7-9 % below FD" reading (2026-10-03): FD noise (2026-10-12).
- Augmented-Lagrangian area mode (`_al` runs, 2026-10-13: void); the projection-era optimiser runs (`optc`,
  `optckn` before `_tan`) for any claim about drag.
- Polygon (50-gon) shapes and any vertex edge-band term; translation controls (2026-09-20).
- A line search in the optimiser (the optimiser as built has a fixed largest displacement, reinitialises every
  iterate, and holds area by the tangent mode / a uniform phi shift).
- Split cells being "refused" (since 2026-09-23 they get sub-cells; `SplitCellError` remains available via
  `split_check`); the stencil tracker as the examples' default (the lane tracker is).
- "SPARTA's `reset_vremax()` is only called from `init()`" - false: it is also called in
  `Collide::collisions()` (`collide.cpp:396`, review 2026-09-14).
- Old test counts, wall-clock suite times, and `pytest -m perf --benchmark` (there is no such marker run;
  budgets are `tests/test_perf_budget.py`).
- Phase-0 revision/wave planning ("the next piece of work is M1"): all of M1-M2 and the M4 forward are done.
