# Review of `fix/paper-adjoints` (`bafec3d..d5f6856`) and benchmark of its gradients

> **Status (cleaned up 2026-10-21).** Written 2026-09-14/15 as a review of the 16 commits of
> `fix/paper-adjoints` (documented in [ADJOINT_CORRECTIONS.md](ADJOINT_CORRECTIONS.md) and
> [FORWARD_JIT_IMPLEMENTATION.md](FORWARD_JIT_IMPLEMENTATION.md)), then extended through
> 2026-10-04 as the running measurement record of the ensemble adjoint, the cut-cell forward and the
> shape adjoint (sections 5 to 7). The branch is merged into `main`. Section numbers are cited from
> code, tests, PSEUDOCODE, PLAN and example READMEs, so they are kept; the content has been condensed
> to conclusions, measured numbers and the traps found. Tables generated from artifacts are
> reproduced by `python3 -m examples.gradient_bench.report` (sections 4.8 to 5.12, 6) and
> `python3 examples/mesh/report.py` (7.2, 7.6 gates); `examples/mesh/review_section.py` writes the
> full 7.1 to 7.4 tables (this file now holds only their conclusions).
>
> **What still holds (summary).**
>
> 1. **Production gradient = particle-carried ensemble adjoint, route (b)** (decided 2026-09-15):
>    carry the adjoint value on each particle, use the *realised* forward pairs as the collision
>    quadrature, interpolate one ghost per collided particle, score sources and walls as kernels.
>    Every validated control agrees with CRN finite differences within noise (section 5.7), at
>    0.5 to 3.8 times the FD's noise (density 0.4 to 0.8x). Score-function routes are dead
>    (4.8, 4.9): terminal-label is biased by attribution ([ATTRIBUTION.md](ATTRIBUTION.md));
>    trajectory score error grows as `M^2.8`; inlet density is 185 to 1300x noisier than FD.
> 2. **Collision pull-back:** the shipped default is the virtual-partner staged form
>    `ensemble_backward.VIRTUAL = (4, 8)`, `VIRTUAL_NODES = "staged"` (5 to 56x lower per-seed sd
>    than the realised-pair forms, under 1 % residual); the sequential realised-pair form
>    (`SEQUENTIAL = True`, `configure("sequential")`) is the exact reference and is good to about
>    0.2 collisions/particle/step; the Delta-t-linearised form is biased (-9 % at 0.16). Virtual
>    partners need `K >= 8` for stability on long horizons (5.12).
> 3. **Ghost field fit:** the quadratic ghost basis (`BASIS = "quadratic_full"`) is required (affine
>    was -8 %). Since 2026-10-11 the default field fit of `device_backward` is the per-cell
>    polynomial `ensemble_backward.CELL_BASIS = "cell:cubicv:adapt1.5:guard32:align0.9:both"`
>    (`jax_dsmc/cell_fit.py`, 10x faster seeds; switched to cubicv 2026-10-12 and guarded
>    2026-10-19; PSEUDOCODE 9.7.2). Sections 5.4 to 5.12 were measured with the kNN
>    `quadratic_full` fit (`--backward-basis quadratic_full` reproduces them); the per-cell fit
>    matched the stored kNN artifacts to 0.2 % to 2 % on the unit gates.
> 4. **Walls:** the chronological earliest-hit tracker with the conditional-time tangent
>    (`wall_adjoint`) is correct and necessary for position observables (4.2); relaxed accommodation
>    carries no Bernoulli score; the same-trajectory mean baseline is biased and rejected (4.10);
>    the certified majorant is the right default for gradient runs (4.3, 4.11).
> 5. **Shape adjoint:** the hit-set (tally) term is exact; the fitted value gradients `grad_x
>    lambda`, `grad_v lambda` next to a surface are the weak link (7.6); shape controls are
>    restricted to **normal displacements of curved surfaces** on a **C1 level set** (decision
>    2026-09-20, 7.6/7.7); the diffuse silhouette term, grazing cut-off, hybrid normal term and
>    cut-volume term are built (7.8, 7.11); tracking on optimised shapes needed the lane tracker
>    (7.10).
>
> **Superseded since the original text** (details at each section):
> the "what is left" list of section 2 (all built or closed); the `hit_ahead` stratified fit as a
> default (7.9, biased by construction, `value_class="none"` is the default since 2026-10-02);
> the two-sided and unpooled fits of 7.6 (default estimator is now `along_pool_plane`); the
> kNN ghost fit as the default (item 3); the polygon vertex edge-band score (never built); the
> 198-job validation matrix (never launched); CPU-era wall-path timings (4.4).
>
> Current decisions with their reasons are collected in `docs/HANDOFF.md` section 2 (2.2 estimator,
> 2.3 field fit, 2.4 shape gradient, 2.5 performance, 2.6 optimiser).
>
> **Reproducibility.** The tree at review start passed 541 CPU tests (436 s) with 17 `ruff` errors;
> both were green after the section 3 fixes. The numbers here come from `scratch/bench/`,
> `examples/gradient_bench/results/`, `examples/mesh/results/`; re-run the relevant study before
> quoting a number against later code (the per-slot RNG change in B6 means pre-branch artifacts
> are not bit-reproducible). `examples/gradient_bench/provenance.py` re-evaluates stored rows.

## 1. Verdicts on the changes

### 1.1 Founded and correct (keep)

| Change | Verdict and evidence |
|---|---|
| Conditional-time wall estimator (`wall_adjoint.conditional_time`) | **Correct, a genuine extension of P3.** For a realised history `h`, `E[J] = Sum_h P_h(theta) E_u[J_h(tau(u,theta),theta)]`, so `dE[J] = E[grad log P_h * J + d_theta J_h]` with `tau` differentiated conditionally on its interval. P3 scores the branch but differentiates an unconditional `tau`, dropping a term for position observables (4.2). `tests/test_stochastic_reference.py` checks the one-particle quadrature value `2 Phi(0.5) - 1`. |
| Chronological earliest-hit tracking (specular/diffuse/periodic/absorbing in one loop) | Correct; agrees with the old fold path to 2e-16 in D = 1, 2, 3 where that was valid; resolves recrossings the old path missed (4.1). |
| Relaxed accommodation has no Bernoulli score | Correct; closes the double-counting defect. |
| Rejecting `baseline="mean"` | Correct: `E[b Sum_i w_i grad L_i] = Cov(mean phi, ...) != 0`. The main repo's `examples/shock_adjoint` ran with it, so its `diam`/`omega` numbers carry the bias (4.10). |
| `trajectory_surrogate`, `score_mode="trajectory"` | Mathematically correct once-per-event likelihood; exact-enumeration oracle is sound. Variance grows with the number of scored events (4.9): a reference, not production. |
| `flux_quantile` (implicit inverse-CDF JVP), `flux_integral` | Correct; `flux_integral` reduces to Bird eq. 4.22; fixes the 31 %-low rejection-sampler derivative. |
| `face_area` includes `grid.transverse` | Correct (all inflow examples use `transverse = 1`, so no validated result moved). |
| Tracked newborns, candidate compaction, `InjectionDiagnostics`; `decomp` reductions and `wrong_owner`; block checkpointing | Correct, tested; close the corresponding REVIEW_2026-09-14 items. |
| Certified majorant (`ntc.certified_majorant`) | **Correct and cheaper than the thermal heuristic at production occupancies** (the plan-time expectation of 2x looser was wrong). A true bound; still `stop_gradient`-ed (4.3, 4.11). |
| P3 host and JIT ensemble inflow (`ensemble_inflow*.py`) | Faithful transcription of P3 section 5 (term-by-term check: `rho = N_k^j/(N0 abs(Omega_j))`, `a = N dt mu/2` with stochastic rounding, the four-term MC correction `main.tex:1474-1492`, fixed normalisation `N0 + M(N_L+N_R)`, source samples keep absorbed births with value 0, weighted local-affine reconstruction); host and device consistent. |

### 1.2 Unfounded, over-claimed, or not worth building on

1. **"`ensemble_inflow` replaces trajectory AD as the supported inlet-gradient method."** It replicates
   P3's own model (constant Maxwell kernel, disjoint pairs, collision-first, fixed birth counts) and
   cannot return a gradient of *this* solver. Making one is what section 5 did.
2. **Integer-support machinery** (`rounded_weak_derivative`, the `count_support_boundary` diagnostic,
   `require_valid(gradients=True)` refusing runs). `E[J]` is piecewise linear in a stochastically
   rounded mean count with kinks at integers, so the interior score is unbiased almost everywhere and
   integer means are a measure-zero set. A harmless diagnostic, not a gate.
3. **The 198-job x 2048-seed `validation_jobs.json`** was untargeted; the deciding quantity is the
   variance ratio AD/FD, which section 4 establishes.
4. **Scaled P3 agreement numbers** (inlet N0 = 1000 to 4000 against 2e6, 8 Couette seeds) are "not
   inconsistent", not validations; 4.12 re-runs them at scale.
5. **Silent default change:** `terminal_objective`/`windowed_objective` moved to
   `score_mode="trajectory"` while `run`/`dice_reward` stayed `"paper"`, silently changing
   `examples/shock_adjoint`. Pinned (B9).

The branch's own pilot artifacts already showed the issue: trajectory score with `b = 0` had SE 135x
the FD's (32 seeds), 7.4x with a pilot baseline (1330 seeds, z about 1.7); both confirmatory files
predate the implementation fingerprint and cannot be resumed.

## 2. What was left at review time (2026-09-15) and how it closed

| Item (original order) | Outcome |
|---|---|
| 1. Production estimator; VSS-kernel four-term backward, retained-frame capture, transport-before-collision ordering | Route (b): `simulate.run(capture=True)`, realised pairs as the collision measure (kernel never appears explicitly; kernel-parameter derivatives use `d_theta log q` on accepted pairs), flipped operator order (5.4 to 5.8). |
| 2. Variance reduction for the remaining score terms | Moot for the ensemble adjoint; its variance control is the virtual pull-back (5.12) and windowed quasi-steady objectives (6). |
| 3. Hard cell-membership sensitivity | Handled by construction (carried values, no interpolation); cut-volume part for shapes (7.11). |
| 4. Windowed surface objectives | `simulate.run(emit_surface=)`, value-form per-step and per-hit rewards (7.6). |
| 5. Geometry interface | `jax_dsmc/mesh/`, SPARTA-style cut and split cells (7.1 to 7.5). |
| 6. Multi-D validation of the chronological wall estimator | Tracking: F3 in D = 1, 2, 3 (4.1); gradients: 2-D Couette and plane gates (7.6). |
| 7. GPU compile stall of the tiny JAX P3 case | Unresolved, not re-measured (ADJOINT_CORRECTIONS). |
| 8. Wall-path cost | Fixed (B3; 4.4, 4.7). |

## 3. Defects found and inefficiencies

| id | where | what | status |
|---|---|---|---|
| B1 | `inflow.inject(track=True)` | called `wall_adjoint.advance` without `epsilon`/`acc_delta`: entrants got hard accommodation and no regularisation on their birth step; their birth-step wall score was dropped in `paper` mode | fixed; `_step` forwards both; 3 tests in `tests/test_inflow.py` |
| B2 | `simulate._step`, `decomp.decomp_step` | `sample.moments` computed twice per step | fixed (only the cell temperature precedes the collisions) |
| B3 | `wall_adjoint._advance_impl` | a fresh diffuse sample drawn per particle, face and bounce (`2D K_bounce` N-sized draws per step) | fixed: one draw per pass (`standard_diffuse_draws`), empty passes skipped (4.4, 4.7) |
| B4 | `_advance_impl` | with `epsilon > 0` a periodic wrap is scored as an event whenever any DIFFUSE face exists: pure variance for periodic-invariant observables | documented; not re-verified |
| B5 | `ensemble_inflow*.py` | partner sampling with self: `O(1/R)` bias in the collision correction | fixed; `partner_indices` factored out and tested |
| B6 | `ntc.collide` per-slot `fold_in` streams | every example's forward stream changed: BKW/shock/Couette/variance artifacts not bit-reproducible | recorded; forward gates re-run (4.5, 4.6) |
| B7 | `wall_adjoint`, `simulate.run`, `ntc.collide`, `rng`, `ensemble_inflow_jax` | nested functions/lambdas as scan bodies against the project rule (module-level bodies + `functools.partial`) | not re-verified |
| B8 | repo | `ruff` red (17 errors, 15 unformatted files) | fixed |
| B9 | `examples/shock_adjoint` | estimator changed silently by the library default | fixed; `evaluate(..., score_mode="paper")` pinned |

Checked and correct: `flux_integral` against Bird 4.22; `first_hit` tie rule; the `conditional_time`
interval bounds (last hit to next hit; absorbed to +inf; no hit to -inf); `merge_accumulators`;
`inject` compaction and the shortfall identity; `local_affine` padding masks; `transverse = 1` in
every inflow example.

## 4. Benchmark

### 4.1 Forward gates

**F3, deterministic tracking versus the legacy fold path** (`scratch/bench/f3_wall_invariance.py`):
DIFFUSE faces with `acc = 0` draw no random number, so the two must agree exactly. D = 1, 2, 3 x 3
seeds x 4096 particles: at `dt = 0.04` (about 15 % hit once) max |dx| = 2.2e-16, dv = 0; at `dt =
0.25` (multi-bounce, |v| about 5 gaps/step) the old path left 211 to 234 recrossings per 4096, the
new one 0 to 1 (D <= 2) and 23 to 32 in D = 3 (4-pass budget; later removed by the lane tracker,
7.10). The tracker is the fold path where that was valid and resolves what it could not.

**F1/F2:** see 4.5, 4.6.

### 4.2 Is the conditional-time tangent needed? (`scratch/bench/probe_conditional_time.py`)

One particle at `x0 = 0.89`, `v = 1`, `dt = 0.1`, `eps = 0.02`, wall at 1; 12 keys x 65 536 draws.

| observable | exact | conditional (shipped) | unconditional | pathwise only |
|---|---:|---:|---:|---:|
| position, `acc = 0` | +0.3829 | +0.401 +- 0.061 (z +0.3) | **+0.530 +- 0.060 (z +2.5)** | +0.3824 +- 0.0015 |
| `v_x`, `acc = 1` | -39.666 | -39.686 +- 0.034 | -39.686 +- 0.034 | **0 (no path)** |

(i) With the unconditional tangent the position derivative is 38 % high. (ii) For a velocity
observable the two coincide and the score is indispensable. (iii) For an observable continuous across
the event (any position functional) the score has zero mean and costs **40x the standard error**
(0.061 versus 0.0015); a selective score or a control variate from the continuous part of the reward
is the obvious variance win and is not built.

### 4.3 Certified versus thermal majorant, static (`scratch/bench/b5_majorant_ratio.py`)

Equilibrium argon, 64 cells. `Sigma_cert/Sigma_thermal` at `npc = 16/64/256`: omega 0.81 gives
0.49/0.64/0.84, omega 0.5 gives 0.47/0.96/1.96, omega 1 gives 0.50. Mean acceptance `<q>/Sigma_cert`
at omega 0.81: 0.57/0.43/0.33 (thermal: 0.28). For VSS argon at production occupancy the certified
bound proposes fewer attempts, is a true bound, grows like `npc^(1-omega)` and overtakes the
heuristic only for `omega <= 0.5` at `npc >= 64`. Dynamic effect: 4.11.

### 4.4 Wall-path cost, CPU (`scratch/bench/b6_wall_cost.py`, N = 20 000, 8 cells/axis, 20 steps)

The branch head cost 2.2x the fold path's forward time and 1.45x its reverse time at D = 1, and
doubled again from `K_bounce` 3 to 8, because of the `2D K_bounce` per-face draws (B3). One draw per
pass plus skipping passes in which no particle reaches a face (a `lax.cond`; draws sit inside the
taken branch) brought the branch to **0.75x / 0.75x** of the fold path at D = 1 (forward 10.8 versus
14.5 ms/step, reverse 68.6 versus 91.6) and removed the `K` dependence of the forward; the reverse
pass still retains residuals per pass (`K = 8`: +35 % temporary bytes). The branch-head timing table
is dropped: that code no longer exists.

### 4.5 F1, Couette viscosity gate on the branch (`scratch/bench/F1_couette_gate.log`)

`python3 -m examples.couette --study gate --seeds 4` (100 cells, npc 400, dx/lambda = 0.30, Kn = 1/30,
3 diffusion times of warm-up, 20 000 sampling steps). Branch `mu_meas/mu_CE` 0.9656 +- 0.0061 (4 seeds)
versus main 0.9747 +- 0.0030 (6 seeds): difference -0.0091 +- 0.0068 (|z| = 1.3). The branch
reproduces the gate within noise; the extrapolated combined gate of the main repo (1.0105 +- 0.0103)
was not repeated.

### 4.6 F2, inflow-driven Mach-2 shock (`python3 -m examples.shock.inflow_report`)

60 lambda domain, 20 000 warm-up + 10 000 sampling steps (exercises `flux_quantile`, `face_area`,
tracked newborns, compaction). All six Rankine-Hugoniot moments within 0.5 % (n upstream -0.11 %,
T upstream -0.48 %, u +0.03 %, n downstream -0.49 %, T -0.30 %, u +0.36 %), mass flux conserved to
0.06 %, shortfall 0, clipped 0, shock at x/L = 0.720, identical to main.

### 4.7 Wall-path cost, GPU (RTX 3050, N = 40 000, 100 steps; branch after B2/B3)

Chronological tracker versus fold path (forward/reverse ms/step): D = 1 at parity (5.7 versus 5.8,
40 versus 39); D = 2 +26 %/+18 %; D = 3 +9 %/+14 %; +10 to 12 % reverse temporary memory; raising
`K_bounce` 3 to 8 costs +8 % forward; reverse compile 34 to 49 s. The fixes are worth about a factor
2 to 3 on the wall stage while resolving every bounce the fold path could not (4.1).

### 4.8 Production estimators: Couette and inlet-shock controls (`examples/gradient_bench`)

Cases: `examples.gradient_convergence` Couette (256 particles, 4 cells, 10 steps, eps = 0.005) and
inlet shock (128 / 8 / 4; `inlet2`: 256 / 8 / 8). Four score arms on identical seeds 32 to 415:
terminal-label (TL, [P2] attribution) and full trajectory (traj), each with `b = 0` and with a fixed
baseline = mean of an independent 32-seed pilot; certified majorant; 4 FD steps (0.1 to 0.0125, CRN);
zero diagnostics on every accepted run. Entries: AD estimate (z versus FD; SE(AD)/SE(FD)).

| control (FD h = 0.1) | TL, b=0 | TL, b=pilot | traj, b=0 | traj, b=pilot |
|---|---|---|---|---|
| Couette `u_wall` (0.598 +- 0.007) | 0.597 (-0.2; 0.95) | 0.596 (-0.4; 0.95) | 0.77 +- 0.59 (0.3; 90) | 0.530 +- 0.059 (-1.2; 8.9) |
| Couette `T_left` (0.169 +- 0.012) | 0.164 (-0.3; 1.1) | 0.160 (-0.7; 0.91) | 1.35 +- 1.2 (1.0; 106) | 0.134 +- 0.093 (-0.4; 8.0) |
| inlet `T_res` (1.368e5 +- 1.1e3) | 1.379e5 (+2.6; 1.0) | 1.375e5 (+1.7; 1.0) | 1.32e5 +- 1.1e4 (-0.5; 10) | 1.346e5 +- 2e3 (-1.2; 1.8) |
| inlet2 `T_res` (1.372e5 +- 1.2e3) | 1.366e5 (-0.8; 0.89) | 1.364e5 (-1.3; 0.88) | 6.8e4 +- 3.3e4 (-2.1; 28) | 1.383e5 +- 3.4e3 (+0.3; 2.8) |
| inlet `n_res` (1.371e5 +- 2.6e3) | -2.7e6 +- 3.3e6 (-0.8; 1310) | -8.9e5 +- 4.7e5 (-2.2; 185) | -2.7e6 +- 3.3e6 (1310) | -7.4e5 +- 4.7e5 (-1.9; 184) |

Reading:

* For wall and reservoir-temperature controls **terminal-label matches the CRN FD at the same
  precision** (SE ratio 0.9 to 1.1, |z| <= 0.7) but sits +0.8 % high on `inlet_temp` at z = 2.6: the
  attribution bias of [ATTRIBUTION.md](ATTRIBUTION.md) at the size expected for a 4-step horizon.
* **Trajectory is unbiased where resolvable** (|z| <= 1.2) but costs 90 to 106x the FD noise without a
  baseline and 8 to 9x with one on Couette; on inlet temperature it is competitive (1.8 to 2.8x)
  because the reservoir score dominates.
* **Inlet density is not obtainable by any score route** (SE 185 to 1300x FD; single seeds up to
  7.5e7 against a 1.4e5 signal, because a stochastically rounded count with fractional part `f`
  carries a score `-1/f` or `1/(1-f)`). It needs the smooth intensity derivative of the ensemble
  adjoint (5.4). Figures: `examples/gradient_bench/figures/estimator_triad.png`, `estimator_inlet.png`.

### 4.9 Variance scaling of the trajectory estimator (`scaling` study, Couette `u_wall`, b = pilot)

SE(AD) over M = 5, 10, 20, 40 steps: 0.0089, 0.059, 0.31, 3.07 (SE ratio to FD 1.7, 8.9, 11.8, 59);
over N = 128, 256, 512, 1024 particles: 0.026, 0.059, 0.038, 0.11 (ratio 2.4, 8.9, 4.5, 15.8). Fitted
exponents: SE(AD) `M^+2.77` and `N^+0.56`, against the FD's `M^+1.20` and `N^-0.15`. The once-per-event
likelihood accumulates every collision and wall event, so the noise is set by the number of events,
not by the signal. Gate U11 of `phase0_verification.py` (SE ratio <= 3 and flat in M) fails; this is
the quantitative case for the ensemble adjoint. Figure: `figures/trajectory_variance_scaling.png`.

### 4.10 The same-trajectory mean baseline the main repo used (`shockbl_*`, reflected shock, 256 particles, 16 cells, 8 steps)

The `b = 1` arm lets `report.py` rebuild the old `baseline="mean"` estimator per seed from `g(b) =
g(0) - b S`. For `diam` (192 seeds) the mean baseline shifted the estimate by **+2660 on -14 430
(18 %)** on identical seeds (terminal-label `b=0` -14 430 +- 3500; `b=mean` -11 770 +- 2900; pilot
baseline -11 900 +- 2900; FD -2299 +- 1163 and drifting with h): `Cov(J, S)` is not small when `J` and
the count score share a cell. For `omega` the shift is 13 % of a value consistent with zero. The
terminal-label `diam` gradient is 3 sigma from the FD in every arm, but the FD is barely resolved, so
the VARIANCE.md-era |z| = 6.4 cannot be reattributed; what is established is that the baseline was a
first-order effect, not a variance trick.

### 4.11 Certified versus thermal majorant, dynamic (`majorant` study, Couette, trajectory b = pilot)

The certified bound needs 31 % fewer attempt slots (max 2.76 versus 3.98 per cell per step). For
wall speed (192 seeds) thermal gives 0.4025 +- 0.11 (-1.8 sigma from FD 0.597) and certified 0.6121 +-
0.061 (z = 0.09), score SE reduced 1.8x. For `diameter` neither resolves the near-zero gradient at 192
seeds. **Certified is the right default for gradient runs**; the forward gates ran with the thermal
default and are unaffected.

### 4.12 P3 replication at scale, JAX backend on the GPU (`python3 -m examples.p3_replication.report`)

**Test 3, two-sided inflow, normal-temperature derivatives** (one budget varied per run around N0 =
4000, R = 512 retained, P = 8 partners, K = 32 neighbours, ballistic; N0 = 64 000 dropped at over 1
h/seed; R = 2048 ran 2 seeds at 80 min/seed because the device kNN is global):

* Only two axes move the answer: the **retained count** (G_L 0.139 / 0.143 / 0.148 at R = 128 / 512 /
  2048, toward the paper's 0.1517) and the **reconstruction scheme** (affine 0.1787 versus ballistic
  0.1432 on identical trajectories, about +25 %). Particles, partners and neighbours do nothing,
  and the fixed-count FD (+-0.02 to 0.13) cannot arbitrate.
* The affine offset is the interpolant's Jensen bias, measured in isolation on the collisionless case
  (`scratch/bench/probe_recon_bias.py`): +11 %/+36 % at R = 128, +3 %/+15 % at R = 512.
* P3's method reproduces in the limit the paper runs it at (32 000 retained/cell); at affordable
  budgets its answer is set by the interpolant. Figure `p3_inlet_convergence.png`.

**Tests 4 and 1, walls** (`--localize`, 32 seeds): Couette wall-speed derivative at N = 32 000 is
**3.032 +- 0.010 against the paper's 3.028** (N = 2000: 2.998 +- 0.068; N = 8000: 3.011 +- 0.014; FD
3.05 +- 0.04 deterministic), a replication at the paper's own precision; heat case at N = 8000: left
0.878 +- 0.027, right 0.674 +- 0.020 against deterministic FD 0.794 +- 0.033, 0.728 +- 0.045 (no scalar
in the paper; the left derivative sits 2 sigma from FD).

**Is the paper's 0.1517 a target for this transcription? No.** Forward-only statistics of the
transcribed Test 3 (N0 = 16 000 scaled): injected per side per step and the interior match (997 375
versus 997 356; interior temperature max 4.92 both), but the final population is +1.7 % and the inlet
cells are hotter and denser (peculiar temperature 4.20 to 4.26 versus 3.73; density max 18.92 versus
18.43). Some inlet convention (entrant velocity law, injection timing, birth-step transport) differs
between the paper's text and its code. The transcription's own gradient is known to 1 %: a 1600-seed
fixed-count FD gives **0.1439 +- 0.0014 / 0.1453 +- 0.0015**; route (b) at N0 = 16 000 gives 0.1480
+- 0.0011 / 0.1447 +- 0.0012 (5.2 item 7); route (a) at R = 2048 gives 0.1477 / 0.1454; all 2 to 5 %
below 0.1517. The gap is on the model side, not the adjoint's.

## 5. Extending the ensemble adjoint to this solver: route (a) versus route (b)

### 5.0 State of the code

| piece | where | status |
|---|---|---|
| event capture | `simulate.run(capture=True)` returns a stacked `EventRecord` (post-move `x, v, alive`; `born_face, born_v`; `PairRecord` per attempt slot with pre/post velocities; `v_post`; `WallEvents`, later `RecordEvents`); forward arithmetic and streams unchanged | done |
| backward sweep (host) | `jax_dsmc/ensemble_backward.py`: `backward()` returns `Adjoint(lambda0, births, hits, pairs)`; `configure("sequential" or "linearised" or "virtual", S, K, nodes, basis)`; `LocalFit`, `BASIS`, `CELL_BASIS`; diagnostics `exact_last`, `collision_term`; `from_device()` | done; `tests/test_ensemble_backward.py` |
| virtual-partner pull-back | host `_collision_pullback_virtual`, `_staged_velocities`; device `ensemble_device.backward_virtual` | done, 5.12 |
| device port | `jax_dsmc/ensemble_device.py` (`field_query` returning value, `grad_x`, `grad_v`; nested reverse scans; empty-slot skip; `node_gradients`, `value_query`); `examples/gradient_bench/device_cost.py` | done, 5.8; resident capture, no block re-capture |
| per-cell field fit | `jax_dsmc/cell_fit.py` (default `CELL_BASIS`) | done 2026-10, PSEUDOCODE 9.7.2 |
| case modules | `examples/ensemble_inlet_shock.py`, `ensemble_couette.py`, `ensemble_relaxation.py`, `ensemble_mesh.py` | done |
| P3-model route (b) | `jax_dsmc/ensemble_particle.py`; `examples/p3_replication/route_b_experiments.py`, `route_b_report.py` | done; `tests/test_ensemble_particle.py` |
| artifact provenance | `examples/gradient_bench/provenance.py` | every score pilot reproduces to 1e-16; ensemble artifacts' objectives exactly, gradients to <= 3e-3 (ridge) |
| pathwise terms inside route (b) | `ensemble_particle.backward(source_term="pathwise")` | measured and rejected, 5.9 |
| shape derivatives | `jax_dsmc/shape.py`, `mesh/levelset.py`, `examples/shape_planar_oracle.py` | 5.11, section 7 |
| `alpha` (angular law) | | not in the ensemble adjoint (needs `grad_v lambda` at post-collision states); it is reparameterised, so the pathwise AD route handles it (consistent with zero in every measured configuration) |

The benchmark leaves one family standing for parameters acting through decisions (inlet density and
temperature, `diam`/`omega`/`alpha`, accommodation, cell membership): P3's optimise-then-discretise
ensemble adjoint. What changes for this solver:

| P3 assumption | this solver | change in the adjoint |
|---|---|---|
| Maxwell kernel `q(sigma) = 1` | VSS rate `q(g)`, angular law | with realised-pair quadrature the kernel never appears explicitly (accepted pairs are samples of the collision measure); only kernel-parameter derivatives need `d_theta log q` weights |
| collision then transport | transport, walls, inject, collide | backward order flips; frames/pairs recorded after injection, before collision |
| absorbing outflow only | specular/periodic/diffuse walls | specular/periodic: deterministic composition; diffuse: the wall is a *kernel*, pulled back by expectation over re-emission, no regularisation; wall T/velocity become local scores |
| fixed nominal birth counts | stochastically rounded counts | `dE[J]/dn = (dN/dn) Sum_k E[lambda_{k+1}(born state)]`, the smooth intensity derivative replacing the `1/f` count score |

### 5.1 Two quadratures of the same adjoint

**(a)** P3's: retained subsets per cell, `partners` random partner draws per retained state, weighted
local-affine reconstruction of the scalar field at every retained state every step
(`ensemble_inflow.py`, `ensemble_inflow_jax.py`).

**(b)** particle-carried (`ensemble_particle.py`): carry `mu_i = lambda` at each particle's own state
and use the **realised forward pairs** as the Monte Carlo sample of the collision measure. For a pair
`(i,j) -> (i',j')` the four-term sum is `lambda(v_i') + lambda(v_j') - lambda(v_i) - lambda(v_j)`; the
first two are carried values and `lambda(v_i)` cancels against the transport term, leaving

    mu_i^k = mu_i + [i collided] (mu_j - lambda~(ghost_j)),   ghost_j = partner's pre-collision state, advected

so the *only* interpolated quantity is one ghost per collided particle. Sources use every birth's own
carried value (escaped births 0). The forward consumes exactly the reference's random streams, so (a)
and (b) see bit-identical trajectories.

### 5.2 Like-to-like verification on P3's own model (`scratch/bench/probe_two_step_ab.py`, `arbiter_two_step.py`; never tracked in git)

Ladder (`tests/test_ensemble_particle.py` unless noted):

1. forward bit-identical to `ensemble_inflow.forward`;
2. collisionless: (b) versus Gauss-Laguerre x Legendre quadrature, normal components within 1 %;
3. recorded pairs are the collisions that happened (pairwise energy conserved to 1e-12; an uncollided
   particle's ghost is its next state);
4. **two steps, strong collisions** (N0 = 20 000, rate 0.20/0.36 per step, every particle retained for
   (a); 240 seeds): `exact_last` makes (b) interpolation-free and an exact-quadrature reference:
   G_L 0.1914 +- 0.0008, G_R 0.1902 +- 0.0006, identical to 4 digits with interpolated ghosts (per-seed
   error 0.02 to 0.08 %). CRN FD paired against (b): +0.0008 +- 0.0015 / +0.0005 +- 0.0010 (h = 0.2).
   (a) with full retention is 0.8 % low at 3 sigma on the left (-0.0016 +- 0.0005): the residual
   reconstruction bias, while the whole collision contribution is only 4 to 6 % of the gradient;
5. one step is vacuous for the source gradient (births enter after the collision stage); the test
   records it so nobody mistakes it for evidence;
6. 20 steps, N0 = 2000, 16 seeds (`scratch/bench/compare_ab.py`): (b) 0.1471 +- 0.0050 / 0.1484 +-
   0.0033, (a) at R = 512 0.1427 / 0.1431; the FD (+-0.04) arbitrates nothing. (b)'s per-seed spread is
   about 1.7x (a)'s (one realised pair per collision versus 8 partner draws per retained state) at
   1/40 of the cost (2 s versus 80 s per seed on the host);
7. **N0 = 16 000, 20 steps (the paper's setting)** (`route_b_scale.py`, `fd_scale.py`): (b) over 160
   seeds **0.1469 +- 0.0005 / 0.1466 +- 0.0005**; 1600-seed fixed-count CRN FD **0.1439 +- 0.0014 /
   0.1453 +- 0.0015** (+2.1 %, z = 2.0 / +0.9 %, z = 0.8); (b) at N0 = 64 000 (8 seeds) 0.1444 /
   0.1461. Agreement at the 1 to 2 % level the FD resolves; the paper's 0.1517 is 3 to 5 % above all
   of them (4.12). Figure `route_b_agreement.png`.

Reconstruction bias in isolation: affine fit of a convex field gives a Jensen bias of the intercept
(`scratch/bench/probe_recon_bias.py`, 4.12).

### 5.3 What (b) needs to become the production estimator (all built)

1. Pair/birth capture inside `simulate.run`, bounded by `checkpoint_block_size`: `capture=True`.
2. Backward scan with the flipped operator order and the wall kernels (specular/periodic composition;
   diffuse by expectation over `sampling.standard_diffuse_draws`).
3. Source and kernel scores: temperature/velocity of the drifting half-Maxwellian flux; the intensity
   derivative for `nrho`; `d_theta log q` on accepted pairs for `diam`/`omega`; wall T/velocity/`acc`
   through the score of the accommodation-mixed re-emission density
   (`sampling.log_diffuse_density`, the "kernel scores" of the ensemble adjoint). `alpha` pathwise
   through `scatter` is the one not done.
4. Ghost interpolation on the device from cell-local nodes of the next-step particles (not a global
   tiled search; R = 2048 cost 80 min/seed for that reason): `ensemble_device.field_query`.
5. Oracles at each step: collisionless quadrature, BKW for `diam`/`omega`, Couette closed form for
   walls, Rankine-Hugoniot `d(rho_2/rho_1)/dM_1` as the final acceptance test (6).

Not a VJP: `jax.grad` and Hessians do not apply; validation is against ensemble FD at several steps
because the collision-transport splitting is O(dt)-consistent, not exact.

### 5.4 Route (b) on the production solver: implemented and validated on the inlet shock

Built: `simulate.run(..., capture=True)`; `ensemble_backward.py` (carried-adjoint sweep: `lambda_i = mu_i
+ Sum_pairs [Lambda(partner post) - Lambda(partner pre)]`, carried value where that was the partner's
last collision in the step, cell-local ghost interpolation, births with their values, move pull-back
by survival; with an exact field and a collision-invariant reward it reproduces the pre-collision field
to round-off including slots colliding twice); `examples/ensemble_inlet_shock.py` (`INLET_CONTROLS`,
RH reservoirs, rounded initial counts; the four smooth parameter-gradient pieces: the drifting
half-Maxwellian flux score `log b = log v_n - m|v-u|^2/2kT - log Z`, the source intensity `dN_f/dc *
mean lambda_born` by `jax.grad` of `inflow.target_count`, the initial Maxwellian score and the initial
count intensity, all through the RH map so `temp` moves temperatures and streams on both sides).

Arbiter: the same forward's CRN FD seed by seed. **Affine ghost basis, diagnostics**
(`results/affine_basis/`), inlet `temp`:

| run | ensemble versus FD (h = 0.1) |
|---|---|
| N = 128, M = 1, exact ghosts (no interpolation) | -2.4 % (z -0.9), 192 seeds |
| N = 128, M = 4, affine | **-8.3 % (z -4.3)**, 384 seeds |
| N = 128, M = 4, collision term dropped | -26.8 % (z -10.4) |
| N = 512 / 2048, M = 4 | -2.6 % / +1.7 % |
| `nrho`, N = 128, M = 4 | +0.9 % (z 0.3) |

The scores and intensities are right; the collision term is 26 % of this gradient; the affine ghost
interpolant recovers about 70 % of it at 16 to 60 particles per cell and all of it by about 1000. The
adjoint field of an energy-like reward is quadratic in `v`, and a cell-wide affine fit is biased by
curvature times node spread (the Jensen effect of 4.12). **Adding the `|v|^2` and `x^2` columns**
(`BASIS = "quadratic"`, then `"quadratic_full"`) removes it: temp, N = 128, M = 4 gives 1.348e5 +-
4.6e3 versus FD 1.385e5 +- 1.5e3, z = -0.95. And the result the exercise was for: reservoir **density**,
where every score route was 185 to 1300x noisier than FD, is **0.43x the FD's noise** and unbiased
(z = 0.3), with no count score anywhere.

### 5.5 Route (b), quadratic ghost basis, Delta-t-linearised collision step: inlet cases (archived in `results/linearised_step/`; the current table is 5.7)

Every inlet case agreed with its FD within noise (|z| <= 1.8, the largest on 64 seeds) from 16 to 1000
particles per cell and at 4 and 8 steps (`nrho` N = 128, M = 4: 1.37e5 +- 1.1e3 versus 1.377e5 +- 2.5e3;
`temp` N = 128, M = 4: 1.344e5 +- 3.5e3 versus 1.373e5 +- 1.1e3; SE ratio `nrho` 0.435, `temp` 2.5 to
3.2). The `_nocoll` diagnostic (collision term dropped) reads -27 % (z -10), i.e. the term is 27 % of the
gradient. Host cost about 4 s/seed at N = 128, M = 4. These cases run at <= 0.05 collisions/particle/
step, which is why the linearised step did not show its bias there (5.6). Figure
`ensemble_inlet_shock.png`.

### 5.6 Walls as kernels, kernel parameters, and the sequential step (artifacts in `results/linearised_step/`)

**Diffuse walls.** The tracker records each step's diffuse hits (face, re-emitted velocity, branch,
hit count; `WallEvents` of shape `(K_bounce, N)`, later `RecordEvents`) and the backward collects them
with `lambda` at the post-move state (`Hits`). The wall is a kernel whose realised re-emission is the
unbiased sample, so the value passes through unchanged and `T_wall`, `u_wall`, `acc` enter only through
the score of the accommodation-mixed re-emission density (`examples/ensemble_couette.py`). **No
epsilon enters the estimator**: P3 section 4 needs it because it differentiates *through* the hit;
section 5 and this route differentiate the distribution. Couette (256 particles, 4 cells, 10 steps,
eps = 0.005, 384 seeds): wall speed 0.5915 +- 0.0096 versus FD 0.5999 +- 0.0062 (z -1.0, SE ratio 1.55);
`T_left` 0.1628 versus 0.1804 (z -1.3, 0.72); accommodation 0.294 +- 0.009 versus an FD that is a step
function of a decision parameter (0.26 to 0.36, h-dependent; resolving 3 % would take about 1e4 FD
seeds), ensemble 2 to 8x more precise; eps = 0 (61 seeds) 0.600 +- 0.026 versus 0.584 +- 0.018.

**Kernel parameters** (`examples/ensemble_relaxation.py`: 0-D anisotropy relaxation, `J = <v_x^2>`,
`dJ/dtheta = Sum_pairs d_theta log q(g) * four_term`, `d_theta log q` by `jax.grad` through
`species_params`). With the Delta-t-linearised pull-back (all accepted attempts against the
end-of-step field) the diameter gradient was **-9.9 % at z = -4.2** (dt = 0.2, 0.16 collisions/
particle/step), -4.2 % at dt = 0.1, -5.7 % at dt = 0.05, while the ghost interpolant was exact to
0.04 % on the last step. The cause is the linearisation itself: a third of accepted pairs have a member
colliding again in the same step, and the end-of-step field is not the value function after *their*
attempt. Route (b) admits the **exact discrete adjoint of the sequential NTC step** at the same cost:
walk each cell's accepted attempts in reverse order keeping the current `mu` and velocities, so at
attempt `a` the carried values are the value function right after `a` and the ghost nodes are the
population as it stood then (`_collision_pullback_sequential`, `SEQUENTIAL = True`): diameter at dt =
0.2 **-0.667 +- 0.039 versus FD -0.675 +- 0.017, z = 0.19**; per-seed noise about 1.7x the linearised
form's.

**Audit of other first-order-in-events approximations:**

| place | approximation | regime where it breaks | status |
|---|---|---|---|
| collision pull-back | Delta-t-linearised step | >= 0.1 collisions/particle/step | replaced by the sequential (exact) form, then the virtual form (5.12) |
| wall hits | one record per slot per step | multiple hits per step | fixed: every bounce scored with `lambda` at the end of the move (exact by the tower property) |
| newborns' birth-step wall hits | not recorded | fast inflow, small domains | fixed: `inject` records `born_walls`, merged in `_event_record` |
| ghost interpolant | local polynomial, exact for quadratic fields | bimodal cells (two-stream layers of strong shocks) | diagnose with `exact_last` and N-scaling; mitigations: velocity-space neighbours, more nodes, higher order (the per-cell cubic fit of PSEUDOCODE 9.7.2) |
| accepted pairs as the collision measure | exact iff `majorant_violations == 0` | thermal majorant at high energy | certified majorant; `require_valid` enforces |
| source / IC intensity terms | linear in expected count | | exact under stochastic rounding |

### 5.7 Route (b), current form: every validated control (`python3 -m examples.gradient_bench.report`, ensemble table)

Sequential collision pull-back, `quadratic_full` ghost basis, every wall bounce and newborn hit scored
individually; seeds 32 on, same seeds' CRN FD (the report's `vs FD h=...` rows show the same agreement
at every FD step).

| control | seeds | ensemble mean +- SE | FD(h=0.1) mean +- SE | z | SE ratio |
|---|---:|---:|---:|---:|---:|
| couette_accommodation_N256_M10_eps0.005 | 384 | 0.2917 +- 0.009 | 0.2615 +- 0.017 | 1.7 | 0.523 |
| couette_speed_N256_M10_eps0 | 192 | 0.5865 +- 0.014 | 0.5933 +- 0.0093 | -0.53 | 1.47 |
| couette_speed_N256_M10_eps0.005 | 384 | 0.5916 +- 0.0097 | 0.5999 +- 0.0062 | -0.97 | 1.56 |
| couette_temperature_left_N256_M10_eps0.005 | 384 | 0.1634 +- 0.0088 | 0.1804 +- 0.012 | -1.2 | 0.725 |
| nrho_N128_M4 | 384 | 1.375e5 +- 1.3e3 | 1.377e5 +- 2.5e3 | -0.08 | 0.531 |
| nrho_N256_M8 | 192 | 1.361e5 +- 1.9e3 | 1.34e5 +- 2.3e3 | 0.71 | 0.813 |
| relax_diameter_N2000_M10 | 192 | -0.663 +- 0.026 | -0.6723 +- 0.012 | 0.32 | 2.14 |
| relax_omega_N2000_M10 | 192 | 0.2824 +- 0.01 | 0.2607 +- 0.014 | 1.2 | 0.693 |
| temp_N128_M4 | 384 | 1.345e5 +- 4.1e3 | 1.373e5 +- 1.1e3 | -0.77 | 3.82 |
| temp_N256_M8 | 192 | 1.364e5 +- 3.6e3 | 1.374e5 +- 1.1e3 | -0.32 | 3.26 |

Every control agrees within noise (|z| <= 1.7; the largest is accommodation, whose FD is a step
function). Noise relative to the FD: reservoir density 0.5 to 0.8x, wall temperature 0.7x, omega 0.7x,
wall speed 1.5x, diameter 2.1x, reservoir temperature 3.3 to 3.8x. The eps = 0 Couette run agrees with
eps = 0.005, as independence of the regularisation requires. Figure `ensemble_inlet_shock.png`.

### 5.8 Device port and cost (`jax_dsmc/ensemble_device.py`, `examples/gradient_bench/device_cost.py`)

The backward is two nested reverse `lax.scan`s (captured steps; attempt slots inside, mirroring
`ntc._slot` with every cell handling its slot's accepted pair in parallel), built on one primitive,
`field_query`: the value and velocity gradient of the carried adjoint at arbitrary `(x, v)` from the
cell's particle nodes (the k nearest in the scaled metric; same weights, basis and relative ridge as the
host `LocalFit`). Slots in which no cell accepted an attempt are skipped by `lax.cond` (35x to 3.5x the
forward on CPU). Host = device to 1e-8 on identical records after the fit's ridge was made *relative*
to the normal matrix (an absolute 1e-12 left cells with fewer nodes than basis columns
solver-dependent at the 0.3 % level, in the host too). GPU (RTX 3050, 50 steps, Couette-like, `k = 32`,
kNN fit):

| N | cells | forward ms/step | ensemble backward | pathwise VJP | backward / forward |
|---:|---:|---:|---:|---:|---:|
| 10 000 | 100 | 5.4 | **8.6** | 21.5 | 1.6 |
| 40 000 | 100 | 11.7 | **25.0** | 60.5 | 2.1 |
| 40 000 | 400 | 6.8 | **11.7** | 36.3 | 1.7 |

The cost gate passes: 1.6 to 2.1x the forward and 2.5 to 3x *faster* than the pathwise VJP. N =
160 000 exhausted the 4 GB device (resident capture about 50 N floats per step, 40 N of them the wall
record; fixed later by the compact record, 7.10; block re-capture of the forward not built). Later
cost work (per-cell fit, two-column value sweep, guarded fits): working notes and `scratch/perf/`.

### 5.9 Pathwise terms inside the ensemble adjoint: measured, not adopted

The reparameterised source term `d/dT E[lambda(x_b, v_b)] = E[(grad_x lambda (x_b - L) + grad_v lambda .
v_b)/2T]` was implemented on the P3 model (`ensemble_particle.backward(source_term="pathwise")`) and
compared with the score form on identical trajectories. Collisionless (exact 0.0465/0.0425): score
0.033/0.038, pathwise 0.053/0.050; N0 = 16 000, 12 seeds: score 0.1462 +- 0.0071 per seed, pathwise
0.1528 +- **0.0128**. Biased and 1.8x noisier. The score form reads the newborn's *carried* value
(exact); the pathwise form needs the field's *derivative* at the newborn from a local fit of a sharply
varying field. The trajectory-pathwise route's low variance comes from exact along-trajectory
derivatives, which a field representation does not have. **Scope:** this verdict is about *fitted-field
derivatives*; exact per-hit rewards taken pathwise are fine and are used by the hybrid normal term
(7.8).

### 5.10 The collision pull-back's bias-variance structure (`results/linearised_step/`, `results/ensemble_relax_*`)

The exact sequential and the Delta-t-linearised forms differ by the chained corrections of particles
that collide more than once in a step; those are realised single-collision samples, so the exact form
pays their variance and the linearised form omits them (bias). Relaxation `diameter`:

| collisions / particle / step | linearised - FD | sequential - FD | noise, seq / lin |
|---:|---:|---:|---:|
| 0.16 (dt = 0.2) | -9.9 % (4.1 sigma) | +1.4 % (0.3 sigma) | 2.3 |
| 0.08 (dt = 0.1) | -4.2 % (1.7 sigma) | -5.7 % (1.0 sigma, 96 seeds) | 1.8 |
| 0.04 (dt = 0.05) | -5.7 % (2.0 sigma) | +1.4 % (0.2 sigma, 96 seeds) | 1.8 |

The sequential noise is set by the total number of realised collisions and does not fall with dt;
the linearised bias does. Above about 0.5 collisions/particle/step (the 2048-particle inlet at dt = 0.5
flights) the sequential form is unstable: each update writes a ghost-fit error into a particle's
`lambda`, the next fit reads the rougher field, and `max|four_term|` grew 369 to 2.1e5 over four steps.
Three intermediate schemes were built, measured and dropped: chunked sub-batches (noise doubles
already at 2 chunks), a position-interpolated field (3x the linearised noise), and chained values with
end-of-step ghosts (**-23 % biased**, 3.6 sigma: the ghost must be evaluated on the field as it stands
after the attempt, or the sequencing buys nothing; `GHOST_FIELD = "current"`). Shipped for the
sequential form: `SEQUENTIAL = True` with an operating envelope of about 0.2 collisions/particle/step
(standard DSMC practice); the linearised form stays available (`SEQUENTIAL = False`). The variance is
intrinsic to one-sample quadrature of each realised collision; P3's `N_col` virtual partners average
over it, which is 5.12.

### 5.11 Shape derivative oracle (`examples/shape_planar_oracle.py`, `results/shape_planar_oracle.json`)

Collisionless planar impulse (`geometry.averaged_planar_impulse`): particles at `u * gap` from a
specular wall, one step, `J` = impulse per particle; analytic `dJ/dgap = -0.5917` (20 000 particles,
16 keys; sign by the impulse convention). Pathwise `jax.grad` at frozen hits (eps = 0): **0.0000**
(the hit-set term is invisible to it). Value-weighted eps-hit score: 0.5877 +- 0.008 / 0.5883 +-
0.012 / 0.5872 +- 0.016 at eps = 0.02 / 0.01 / 0.005. Ensemble FD over independent keys: 0.582 to
0.591 +- 0.03. The hit-set term comes out of the existing regularised-time machinery
(`conditional_time`'s interval log-probability differentiated with respect to the wall position)
weighted by the *value*, no new derivation; its variance grows as eps shrinks. On this single-hit
oracle P3's reward-weighted form and the value-weighted form coincide; they separate on collisional
multi-step cases. Figure `shape_planar_oracle.png`.

### 5.12 Virtual-partner pull-back: P3's `N_col` averaging on the production solver (`_virt{S}x{K}[st]` artifacts, `scratch/bench/relax_summary.py`, `relax_fd_arbiter*.json`)

5.10 left the sequential form with a variance that grows with collisions per particle per step and blows
up above about 0.5 (the two `N = 2048, dt = 0.5` inlet cases, +-100 % intervals). The invariance tests
pass there, so it is the quadrature, not the bookkeeping.

**Estimator.** `ensemble_backward.VIRTUAL = (S, K)` (`configure("virtual", S, K, nodes)`; device
`ensemble_device.backward_virtual`) applies the backward collision operator `e^{dt L*}` in `S` substeps.
In each, every alive particle draws `K` virtual partners from its cell and fresh VSS angles, weighted by
the NTC acceptance `q(g)/Sigma_c` times the cell's realised attempts per particle `2 n_att,c / N_c`; all
four values of `Lambda(v_i') + Lambda(v_j') - Lambda(v_i) - Lambda(v_j)` come from one smooth per-cell fit
of the carried values, so their interpolation errors cancel. No realised pair enters except through the
attempt count: P3's MC collision correction (`main.tex:1474-1492`) with exact NTC weights and a
substepped operator.

**Nodes.** With end-of-step velocities as fit nodes (`nodes="post"`, the first version) the field inside
the step is frozen: an O(dt) splitting error more substeps cannot remove (biased +5 to +6 %, 4 to 5
sigma). The NTC step processes slots in index order, so the state at any slot boundary is `v_pre`
overwritten by each particle's last accepted slot before it (`_staged_velocities`, device
`staged_velocities`, tested against a sequential replay); `nodes="staged"` fits substep `s` on the state
at the end of its own slot range.

**Arbiter.** Relaxation `diameter`, dt = 0.2, N = 2000 (0.16 collisions/particle/step): FD-only over
**1800 independent seeds**, -0.6798 +- 0.0034 (h = 0.1; -0.6817 and -0.6795 at 0.05, 0.025; N = 20 000:
-0.686 +- 0.005). Bias against -0.680 and per-seed sd (`pullback_scan.png`):

| pull-back (K = 8) | bias | per-seed sd |
|---|---:|---:|
| sequential (192 seeds) | +2.5 % (0.7 sigma) | 0.36 |
| Delta-t-linearised (192) | -8.6 % (5 sigma) | 0.15 |
| virtual, post nodes (48) | +5.2 to +5.8 % | 0.05 |
| virtual, **staged**, S = 2 / 4 / 8 (48) | +1.8 % / +0.7 % / +0.1 % | 0.05 |
| staged S = 2 at dt/2, dt/4 | +0.6 % / +0.3 % | 0.04 |
| `omega`, sequential / linearised / staged S = 2 | +4.4 % / +11.2 % / +0.9 % | 0.14 / 0.07 / 0.027 |

The staged residual is the substep splitting error, O(dt/S), with **no change in variance** from `S` or
`K` (K = 2 equals K = 8 at 2000 particles per cell). The sequential form is unbiased at every dt but 7 to
9x noisier and does not improve with dt.

**A stability limit the short cases could not show** (`python3 -m examples.gradient_bench.k_stability`,
`results/k_stability.json`, `figures/k_stability.png`; one seed of the 300-step RH case of section 6, 80
cells, 3236 particles, S = 2; rms of the carried `lambda` over alive particles relative to the terminal
rms at steps 200 / 100 / 0):

| K | rms | backward time |
|---:|---|---:|
| 2 | 146 / 4.1e4 / **3.2e6** (gradients come out at 1e7) | 162 s |
| 4 | 0.39 / 0.25 / 0.18 (excursions, max to 60) | 283 s |
| 8 | 0.24 / 0.12 / 0.028 | 512 s |
| 16, 32 | the K = 8 curve | 861, 3649 s |

K = 2 grows geometrically (an e-fold every about 20 steps); K >= 8 gives one decaying curve (the physics:
sensitivity of a 300-step-old state to the final density ratio). The noise each substep injects is
smoothed by the next fit, and whether the net gain is below one depends on K against occupancy and
horizon, so equal variance at K = 2 and 8 in a short case does not license K = 2 in general.
`examples.ensemble_inlet_shock.device_backward` refuses a backward whose rms `lambda` grew more than 100x
(`GROWTH_LIMIT`).

**Shipped default:** `VIRTUAL = (4, 8)`, `VIRTUAL_NODES = "staged"` (under 1 % residual at standard DSMC
steps, twice the cost of S = 2); runners tag artifacts `_virt4x8st`. The sequential form remains the exact
reference (untagged artifacts).

**Where it matters** (staged S = 2, K = 8, device, paired with the same seeds' CRN FD): where the noise is
the source or wall score (128-particle inlets, Couette) the two forms are equally noisy; where it is the
collision term it is **5 to 56x quieter** (per-seed sd ratio sequential/virtual): inlet `temp` N = 2048 at dt
= 0.5 (0.8 collisions/particle/step) 37x, 1.361e5 +- 3.9e3 versus FD 1.386e5 +- 1.5e3 (z -0.7; sequential
sd 8.1e5 versus 2.2e4); inlet `nrho` 56x; relaxation `diameter` 7 to 9x; `omega` 5x. On the N = 2048 cases
the ensemble SE is now below the FD's.

**Cost and robustness.** Host about 15 s/seed per substep on the 2000-particle single-cell case; device
(RTX 3050) 10 s/seed at S = 2, K = 8 including nine FD forwards. The backward costs `S (2K + 3)` field
passes per step (the 300-step RH case at K = 8: about 3 min/seed against 1 min of forwards). Known waste:
the device backward queries every particle *slot*, dead ones included (4x the population on the inlet
cases). The fit's relative slope ridge (1e-8) is not innocent: on the inlet N = 256 case the sequential
gradient moves 3e-3 between `RIDGE_REL` 1e-10 and 1e-8 and 6 % at 1e-6 through about 0.5 % of ghost
queries with condition number about 1e6 (`probe_ridge_sensitivity.py`); a per-column ridge does not
change this. The virtual form averages 8 partners through one fit and is less exposed; its exactness on a
quadratic field is 1e-4 at the shipped ridge and 1e-12 at zero
(`test_virtual_pullback_carries_an_invariant_exactly`).

## 6. Toward shape optimisation: the estimator's terms and the acceptance case

A wall position `theta` enters the expected objective through three routes; the ensemble adjoint gives
each an estimator that needs no new machinery. All terms are weighted by the carried value `lambda`,
not the reward.

| route | where theta acts | term | status |
|---|---|---|---|
| hit set | which particles reach the wall inside the step | eps-regularised hit-time score of `conditional_time` differentiated with respect to wall position, times `lambda(hit) - lambda(no hit)` | oracle passed (5.11); in the level-set body the fixed-hit-set tally term is exact (7.6) and the miss-to-hit part is the silhouette term (7.8) |
| reflected position | `x' = 2 x_wall - x` (specular) / re-emission point (diffuse) | `grad_x lambda . dx'/dtheta` from `field_query`'s gradient at the post-hit state | built; the fitted `grad_x lambda` is the weak link (7.6) |
| cell volume | NTC attempt count `~ 1/V_c` and cell membership | virtual pull-back weight `2 n_att,c / N_c`: `d_theta log n_att,c = -d_theta log V_c` times the substep four-term sum | built 2026-10 (`shape.volume_weights/volume_cotangent`, `Backward.cell_collision`); -2.9 % of `dD/ds` at Kn 0.5 (7.11) |

**Acceptance case** (`examples.ensemble_inlet_shock --objective density-ratio --parameter mach`; 80
cells x 40 particles, 300 steps; the case whose pathwise derivative had the wrong sign in
`examples/shock/gradients.py`: -1.9e-3 against an FD of +1.8e-3): the density ratio by the delta
method (totals frozen at realised values, O(1/N) bias). Result (virtual staged S = 2, K = 8, device,
16 seeds, about 6 min/seed): `J = rho_2/rho_1` at 300 steps 2.301 (RH stationary 2.286); ensemble
adjoint **2.74 +- 0.66** (per-seed sd 2.63); CRN FD 2.54 +- 0.17 (h = 0.1) / 2.17 +- 0.42 (0.05);
paired difference +0.20 +- 1.39 (95 %), z = 0.3 (RH stationary slope 1.96, reference only). The sign is
right and the magnitude agrees within noise, but it is a coarse acceptance: the ensemble per-seed sd is
4x the FD's, driven by outliers (one seed at 11.6, another at 0.06). That is the long-horizon face of
the K stability limit (5.12): at 40 particles per cell K = 8 keeps the carried field from blowing up
but not from wandering. Consequences: budget K by occupancy and horizon, and prefer windowed,
quasi-steady objectives over long transients (variance grows with the horizon whatever the
pull-back). Figure `rankine_hugoniot.png`.

Planned arbiter for the moving-wall collisional case (not done in this form): collisional Couette with
`gap` as the control against ensemble FD over independent keys (a moving wall changes the hit sets, so
CRN pairing is unavailable: about 2000 seeds for a 5 % FD). The plane (7.6) and level-set circle/
squash (7.7 to 7.11) gates replaced it.

## 7. Embedded surfaces (SPARTA-style cut cells): geometry and forward gates (2026-09-18)

Code: `jax_dsmc/mesh/`, the element branch of `wall_adjoint.py`, `GridSpec.volume`; runners and readers
in `examples/mesh/` (its README lists every script and artifact). References are the shipped SPARTA
`circle` and `sphere` examples rerun in **argon** with a locally built serial SPARTA (4 seeds each,
per-cell flow volumes, `compute surf` and `compute grid` window averages under
`examples/mesh/results/sparta/`). Full tables: `python3 examples/mesh/review_section.py` (writes `examples/mesh/results/mesh_gates.md`: 7.1 to
7.4 from the artifacts); this file keeps the conclusions.

### 7.1 Geometry (`examples/mesh/geometry_checks.py`, `results/geometry_checks.json`)

Circle (sub 32): cells outside/inside/cut (264, 88, 48), identical to SPARTA's; flow volume 71.80005
(ours) / 71.80002 (SPARTA dump and exact); max cell error 1.3e-3; watertight (2076 segments); 0
brute-force candidate mismatches. Sphere (sub 16): (7256, 304, 440), identical; volume 59.86164 versus
59.86171; max cell error 4.7e-5; watertight (3518 segments); 7 s build. Volume-fraction convergence
(circle, max cell error): 1.1e-1, 5.3e-2, 1.8e-2, 5.1e-3, 1.7e-3 for sub 4 to 64, about `sub^-1.5`. The
four circle cells SPARTA lists as "overlapping" with full volume are the ones touched at the grid
corners (2, 5) and (8, 5), reproduced by the inclusive clip test.

### 7.2 Forward gates (`examples/mesh/report.py`, `results/forward_gates.json`)

Circle (4 runs, 65 s/run on GPU) and sphere (4 runs, 301 s/run). Every tracking diagnostic is zero over
4 x 1000 steps at 50 to 80 k particles (FG1).

| gate | circle: ours / SPARTA | sphere: ours / SPARTA |
|---|---|---|
| FG2 final Np | 47 594 +- 61 / 47 683 +- 58 (z -1.05) | 80 757 +- 57 / 80 642 +- 105 (z 0.97) |
| FG2 collisions/step (tail) | 136.2 / 140.5 (tail-mean z -3.1, the one soft miss) | 177.0 / 174.6 (z 1.6) |
| FG3 total drag | 3.69e-20 +- 7e-23 / 3.679e-20 (+0.31 %, z 1.36) | 1.626e-20 / 1.628e-20 (-0.11 %, z -0.30) |
| FG3 per-element pressure | rms z 1.31, 93 % within 2 | rms z 1.19, 92 % within 2 |
| FG4 `nrho`, `u`, `temp` per cell | rms z 1.2, 90 to 92 % within 2 (310 cells) | rms z 1.2, 91 % within 2 (7696 cells) |
| FG7 polygon versus analytic body | Np -0.35 %, drag +0.51 % | Np -0.21 %, drag +0.75 % |
| walk versus stencil tracker | Np -0.32 %, drag +0.41 % | Np -0.14 %, drag +0.29 % |

Two SPARTA conventions were needed to compare: `compute grid temp` is the un-centred `(m/3k) <v^2>`,
and `fix ave/grid` is a particle-step (count-weighted) average.

**G3a window score** (1-D closed box, windowed impulse, 12 seeds, W = 1, 10, 100): the value-weighted
eps hit-timing score (eps = 0.02 / 0.01) has no resolvable mean at any W (|z| <= 1.6) and its sd is 4
to 8x the ensemble FD's. Shape runs on steady windows therefore run at **eps = 0** and the hit set
enters through the replayed wall map and the edge-band score.

### 7.3 Free-molecular drag (FG5, `examples/mesh/free_molecular.py`)

With the stream the inlet actually produces (the reservoir Maxwellian restricted to `v_x > 0`, whose
specular circle `C_D` is 17.17, not the full-Maxwellian 12.22), the drag extrapolated to zero blockage
matches the surface integral to |z| <= 1.2 for specular and diffuse, circle and sphere (circle
specular 17.080 +- 0.080 versus 17.175; diffuse 20.429 versus 20.429; sphere 14.684 versus 14.804 and
20.990 versus 21.211). Two lessons: the slow part of the half-range stream takes seconds to fill
(fill-from-empty at 0.4 s was 7 % low on a small body's hit rate, so sweeps start from the steady
distribution), and the 60 % blockage of the reference case raises its drag about 50 % above the
unbounded value.

### 7.4 Record budget (G9, `examples/mesh/record_budget.py`)

16 000 slots x 300 captured steps: 0.72 GiB at `K_bounce = 2` (162 B per slot-step), 0.98 GiB at
`K_bounce = 4` (220 B); `walls.v_out` is the largest leaf and is replayable from the element ids.
Within budget without a new record type (later compacted, 7.10).

### 7.5 What the gates say

Geometry is exact where SPARTA is exact (counts, cut set, candidate lists). Forward: Np, surface
collision rate, total drag (+0.3 % / -0.1 %), per-element pressure and window-averaged fields agree
with SPARTA within seed spread on both cases; the cell walk and the analytic body reproduce the polygon
run within 0.5 %. FG5 closes the drag against free-molecular theory. G3a sets eps = 0 for steady
windows. G9 sets the memory budget.

### 7.6 The mesh-aware adjoint: phase C (2026-09-18; PSEUDOCODE 9.7, `tests/test_mesh_adjoint.py`)

**Built** (additively to the ensemble core): `field_query` returns `(value, grad_x, grad_v)`;
`backward[_virtual](phi_step=, hit_reward=)` carry value-form rewards (a per-step reward joins the
carried value before its step's pull-back, a hit's reward joins on the move pull-back, and a hit's
kernel score is weighted by the reward-to-go **from its own draw onward**: leaving out the hit's own
reward made G2 read 40x too small); `jax_dsmc/shape.py` replays each captured step's wall map
(`advance(trace=True)`) and takes `jax.vjp` with respect to the geometry with the backward's cotangents
(`grad_x lambda`, `grad_v lambda` at survivors' post-move states, `lambda + r` on the log-probability, the
objective's tally weights, the diffuse normal score, the hit-timing band term). The replay reproduces the
record to round-off for both trackers (so the runners hold it to `REPLAY_TOL = 1e-9` and report 1 on any
flipped outcome, 7.10); the VJP matches central differences on the analytic circle and the 50-gon.

**Gates** (`examples/mesh/report.py` "Adjoint gates"; 48 seeds, CRN FD at four steps): G1 Couette
windowed `v_y^2`, `speed`: +0.491 +- 0.025 versus +0.4925 +- 0.016 (z -0.05); G2 windowed shear (per-hit
reward), `speed`: -0.0177 +- 0.0014 versus -0.0165 +- 0.0008 (z -0.77), `temperature_left` z +0.46.

**G3** (`examples/mesh/ensemble_plane.py`: 2-D collisionless Couette whose moving wall is the left face of
an embedded rectangle, control = its translation, 4096 particles, 48 warm-up steps, CRN FD) is where the
shape machinery was exercised. Findings (items 3 to 5 are cited elsewhere by number):

1. **Velocity-dependent per-step rewards jump at hits, and the pathwise VJP of a fixed hit set misses the
   jump.** Windowed `v_y^2`: pathwise alone +0.82 +- 0.23 versus FD +0.05 +- 0.15. The missing term is
   the density of hit times at the step boundary times the jump `phi(v_out) - phi(v_in)`, estimated from
   hits within a band of the step end (`shape.timing_cotangent`): -0.43 +- 0.02, closing the gap. A
   per-hit reward jumps only at the window's two edges (zero mean in a steady window); a position-based
   reward not at all.
2. **The two-sided fit's `grad_x lambda` is biased next to a wall** (windowed shear -0.0122 versus FD
   +0.0174, wrong sign): the post-move population splits into particles about to hit (value about the
   hit's reward) and ones that just left (value about what follows), an O(reward) contrast no polynomial in
   `v` represents, correlated with the distance to the wall. `node_gradients(direction=)` fits over the
   nodes on the hit particle's own side (`v . n > 0`).
3. **Is what remains bias or horizon? Measured with an exact value gradient** (collisionless plane: the
   true `grad V` is a CRN central difference of a 1024-replica average;
   `scratch/mesh/probe_plane_fitbias.py`, `probe_plane_truncated_fd.py`). (a) **With the exact value
   gradient the estimator equals the truncated FD on both objectives** (z = 0.5 each; shear +0.003 versus
   +0.0017 +- 0.0026, `v_y^2` -0.09 +- 0.12 versus -0.149 +- 0.037): the replayed VJP, value-form
   rewards and the timing band term are right. (b) **The gap between truncated and full FD is the
   horizon** (gate G6): 12 captured steps before the window hold almost none of the steady rate's
   sensitivity to the gap (one round trip about 40 steps); with 72 captured steps before a 24-step window
   the truncated FD is +0.0170 +- 0.0023 = the full FD, estimator +0.014 +- 0.014. (c) **The fitted `grad_x
   lambda` is the weak link**: the one-sided fit removes most of the two-sided bias but `v_y^2` still
   reads -0.37 +- 0.07 against -0.15 +- 0.04 (z -2.9) because its x-part is +0.07 where the exact one is
   +0.36.
4. **One-sided fits: where needed, what they cost, the general form** (`probe_fit_classes.py`,
   `figures/gates/fit_classes.png`). Needed for the *receding* population within a cell of a surface whose hits
   carry reward (halves the slope error, cuts the value error 20 to 30 %); costs 0 to 3 % of the sd
   elsewhere (approaching, fixed diffuse wall, interior). The two-sided bias is below the noise per
   particle; the -0.03 of item 3 is the weighted sum, where grazing hits (`1/v_n` weights) dominate. The
   rule is general: split a query's nodes by the sign of `v . n` for the nearest surface within one cell
   width (`mesh.geometry.nearest_normal`, `shape.local_normals`), applied to every shape cotangent and,
   optionally, to the collision pull-back (`backward_virtual(normals=)`; `None` is bitwise the old
   estimator; collisional Couette gates reproduce to three digits with it, so it is safe; off by default
   in the pull-back).
5. **Better estimators of `grad_x lambda`: pooled one-sided fits and the along-flight identity**
   (`node_gradients(pool=)`, `shape.along_flight`, `shape.gradient_variants`, `shape.VARIANTS`; PSEUDOCODE
   9.7; LaTeX note "Estimating grad_x lambda"). Pooling draws the `k = 96` nearest one-sided nodes from the
   `3^D` block around the query's cell. In collisionless free flight `V(x - s v, v)` is the value of the
   same particle delayed by `s`, so `v . grad_x V = -T` with `T` the value of the *timing rewards* (a
   second value-form backward); the cotangent keeps the fitted perpendicular part and replaces the
   component along `v` by `-T` (holds per particle to 1e-5 on a scale of 1.6e-3). Against the exact
   gradient (windowed `v_y^2`, exact x/v-part +0.374 +- 0.111) the bias is +0.68 (two-sided), -0.29
   (one-sided), -0.20 (pooled), -0.08 (one-sided + along), -0.07 (pooled + along). Production runs
   against the truncated FD (z): plane `v_y^2` one-sided -2.9, pooled -3.6, along -0.7, pooled + along
   -1.0; circle radius (CRN FD 1.395 +- 0.02 e-20) pooled 1.377 +- 0.016, pooled + along 1.353 +- 0.024.
   **Verdict.** (i) Along-flight removes the bias where the along-flight component carries the signal;
   pooling removes variance (4 to 7x on the circle) but alone keeps a bias; the combination became the
   default (now `along_pool_plane`, 7.11). (ii) **None of the estimators resolves the x/v-part of a
   translated flat wall**: the exact shear x/v-part is +0.0032 +- 0.0006 and the pooled/along variants
   read about 0 with small error bars (pooling smooths a slope of 3e-6 per unit length to zero; the
   along-flight replacement uses the realised sparse `T`, whose SE is understated); the totals "agree"
   with the truncated FD only because both are 0 +- 0.005, a test without power. For radius-type controls
   the fixed-hit-set tally term is exact by construction and carries about 90 % of the gradient, so any
   variant is FD-consistent at the few-per-cent level; for translation-type controls the x/v-part is the
   whole signal and is unresolved. Not built: replica re-simulation (exact but collisionless-only),
   the eps-regularised timing score (4 to 8x the FD's sd, G3a).

**Decision (2026-09-20): shape controls restricted to normal displacements of curved surfaces.**
Tangential translations of flat walls, and any control whose gradient is a window-end delay sensitivity,
are **not supported** and must not be used as optimisation controls. Tracked in PLAN Phase 5 and
PSEUDOCODE 9.7.

**Representation (2026-09-20): C1 level set, not polygon vertices.** The shape is a C1 level set
(`jax_dsmc/mesh/levelset.py`, uniform cubic B-spline coefficients on a finer grid, single surface id like
the analytic body). The polygon's vertex "edge-band" term is representation-induced (the corner delta of
the Hadamard derivative at a normal jump) and disappears on a C1 surface, where the hit normal varies
smoothly and the pathwise VJP is complete for specular walls; a C0 interpolant would relocate the jumps
to the level-set cell faces, so C1 is load-bearing. What survives is the silhouette term of a diffuse body
(miss to grazing hit), in phi-form on the near-miss set `min phi in (0, eps)`; it vanishes for specular
walls. **Trap:** the bisection of the ray-surface root has zero derivative through its `where`
selections; a final Newton step with the bracket under `stop_gradient` restores `dt/dcoef = -B_j(x_hit)/
(v . grad phi)`. The polygon edge-band score is **not built**.

**Phase D, first shape gradient** (`examples/ensemble_mesh.py --parameter radius`: the SPARTA circle at
`fnum x 4` (about 12 k particles), 700 warm-up steps, 300 captured of which the last 200 are the windowed
drag; specular analytic body; virtual pull-back S = 1, K = 2; 48 seeds, about 2 min each on GPU):
**ensemble 1.44 +- 0.09 e-20 versus CRN FD 1.395 +- 0.02 (h = 0.05) / 1.40 +- 0.01 (h = 0.1), z = +0.5**;
analytic `F/r` 1.23 e-20. Terms: tally (fixed hit set, reflected velocity through the normal) 1.28 +- 0.01,
`grad_v lambda` 0.02 +- 0.02, `grad_x lambda` 0.14 +- 0.09, timing 0.07 +- 0.05 (excluded). For a specular
cylinder the whole of `F/r` is the fixed-hit-set term (`F ~ int cos^2 phi db = 4r/3`, derivative at
fixed rays `4/3 = F/r`, new grazing rays contributing `cos^2 phi = 0`), which the tally term reproduces to
4 %; the 0.15 e-20 the FD has above `F/r` is the flow-field feedback of the 60 % blockage, carried by the
`grad lambda` terms, and it is the noisy part (per-seed sd 45 % of the gradient). `centre_x` (32 seeds):
-0.9 +- 0.6 e-21 versus FD -0.07 +- 0.17 (tally and `grad_v lambda` parts cancel). Figure
`shape_circle_terms.png`.

### 7.7 Phase E: the level-set body (2026-09-20; `jax_dsmc/mesh/levelset.py`, PSEUDOCODE 4.4, `tests/test_mesh_levelset.py`)

`phi(x) = Sum_j c_j B_j(x)` with uniform cubic B-splines on a grid finer than the DSMC grid (`h_phi = h/4`
on the circle), body `phi < 0`, a third `Mesh.kind` with the analytic body's single surface id, so
tracking, tallies, kernel scores and the replayed shape VJP are the phase B to D paths; the VJP lands on
the coefficients and any control is chained through `coef(theta)` (`radius`: `dcoef/dr = -1`; the
`cos 2θ` normal mode: `coef + θ interpolate(-cos 2θ)`). Hits: sampled `+ -> -` bracket at spacing
`h_phi/2` over the reach `n_sample h_phi/2`, bisection, one Newton step with the bracket search a function
of `stop_gradient` inputs (without it the coefficient gradient is identically zero). Host tables: CUT by
the coefficient-sign convex-hull test refined on a centred sub-grid; volumes by sub-sampling.
Construction from an analytic SDF or a polygon (`from_surface`); `reinitialise` by clipped closest-point
projection in a band plus a KD-tree distance outside it (an unclipped projection diverged where `grad phi
~ 0`).

**LS0** (20 CPU tests + geometry checks): interpolation exact at interior knots; surface error and hit
position error fall as `h^4`; hit sets identical to the analytic body's on 40 k random rays except
grazing incidence (`|v . n| < 1e-3 |v|`); `dt/dcoef` contracted with -1 equals the analytic `dt/dr` to
1e-4; tracking reproduces the analytic body to the surface error; the replayed wall-map VJP matches FD;
tunnelling (a grazing dip shorter than one sample spacing) is rate 1 below about 0.5 spacing, 0 above, a
fraction about 5e-5 of the circle's hits, all grazing; reinitialisation keeps the zero set to 1e-3 and
`|grad phi| = 1` to 2 %.

**LS1/LS2** (4 seeds, CRN with the analytic artifact): same simulation to the surface error (objective to
2e-6, identical hit counts); gradient per estimator variant agrees to 1e-4 on three of four seeds. On seed
1 the tally term differs by 4 %, caused by **one** hit with `|v . n|/|v| = 3e-3` (15 % of the window's
sum): the pathwise tally derivative of a hit is `O(1/(v . n))`, so its mean converges but its variance
diverges logarithmically in the grazing cut-off (the heavy tail behind the radius' 5 % per-seed sd).
Level-set gate: ensemble 1.207 +- 0.089 e-20 versus CRN FD 1.398 +- 0.010 (z -1.4, 4 seeds; analytic
48-seed 1.44 +- 0.09); 515 s/seed (analytic 332 s). **`cos 2θ` mode** (`--parameter mode2`, 4 seeds):
ensemble **-1.587 +- 0.030 e-20 versus CRN FD -1.617 +- 0.029 / -1.571 +- 0.059** (z +0.7 / -0.2). Its
per-seed sd is **3.8 %** (radius: 45 %): a zero-mean normal mode leaves the fitted `grad lambda` parts
small and the exact tally term dominant, so the estimator is far better conditioned on shape modes than
on the radius, which is what the optimiser uses.

**First optimisation** (`examples/mesh/optimize.py`): minimum windowed drag of the level-set circle at
fixed area; 8 iterates x 2 CRN seeds (55 min on GPU), step = largest normal displacement 0.05, H1
smoothing sigma = 2 knots, exact area by a uniform shift of phi after reinitialisation. Drag 3.663 to
3.392 e-20 (-7.4 %) with area constant to 1e-11 (observed decrease 2.70e-21 against a summed first-order
prediction of 4.21e-21: a fixed-size step, not a line search); a nose forms and the top and bottom
flatten, as expected for a specular body in free-molecular flow. The later optimiser is in 7.11.

### 7.8 The two remaining terms (2026-09-21): grazing regularisation and the silhouette band

**Grazing regularisation** (`LevelSet.grazing`, `AnalyticBody.grazing`; PSEUDOCODE 4.4;
`figures/gates/grazing_regularisation.png`). The hit-time derivative's `1/(v . n)` is evaluated with `|v . n| >=
eps |v||n|` in the tangent only (`levelset.regularised_quotient`, primal exact). Radius tally term, 8 CRN
seeds:

| eps | 0 | 0.01 | 0.03 | 0.1 | 0.3 |
|---|---:|---:|---:|---:|---:|
| seed sd of the tally term | 8.4 % | 5.9 % | 4.2 % | 2.9 % | 2.1 % |
| paired bias versus eps = 0 | | +1.0 % | +0.5 % | -1.2 % | -7.4 % |

Variance falls 3x at a bias below the 2 % resolution up to eps = 0.1; **eps = 0.05** is the default of the
optimiser and cluster study.

**Silhouette band term** (`shape.silhouette_cotangent`, `ShapeTerms.silhouette`, `ensemble_mesh
--silhouette-band`; `docs/math/ensemble_adjoint.tex`). Built on the closest approach of every survivor's
undeflected line (`levelset.closest_approach`); cotangent `-(V_hit - V_miss)/(2 eps_b)` on `phi(x*)` for
`|phi_min| < eps_b`; the virtual outcome (straight flight for a hit particle, re-emission under the wall
model for a miss) is valued by the step's `lambda` fit at the virtual post-move state
(`ensemble_device.value_query`, nodes of the *leaving* population) plus the virtual hit's reward. Two
traps (`scratch/mesh/probe_silhouette.py`): (1) **count each passing once per trajectory** (a miss in the
step whose move contains the closest-approach point, strictly inside; a hit in the step it hits): counting
every particle-step inside a band wider than its move overestimated the density 5x; (2) a fragile hit's
virtual miss must be valued from the leaving population, not the approaching one (which carries the value
of the hit it is about to make). Constructed rays: specular about 0, diffuse = new hits per unit
displacement x absorbed momentum to 0.04 %.

**GP gate** (diffuse capsule plate normal to the stream, half-width 1.0, thickness 0.4; control =
half-width; `ensemble_mesh --case plate --body levelset --acc 1 --parameter width --silhouette-band
0.05`; 4 seeds). First pass (`h_phi = 0.125`, `grazing = 0`): 1.104 +- 0.025 versus CRN FD 1.369 +- 0.068,
80 % of FD (41 % without the silhouette). The score-only constructed-ray probe
(`probe_score_resolution.py`) explains it: at `h_phi = 0.125` the score is 92 +- 4 % of the reference (the
0.4-radius cap is three knots wide); at `h_phi = 0.0625` without the grazing cut-off 240 +- 62 % (single
draws 0.03 to 18.7x), because the normal score `(lambda + r) d_n log q` multiplies the hit point's
sliding `dx_hit/dtheta ~ 1/(v . n)` by the resolved curvature and carries a `1/(v_out . n)` Rayleigh tail;
with `grazing = 0.05` it is 100.9 +- 5.7 %. So the diffuse score term needs the cut-off once the surface is
resolved, and the capsule needs `h_phi <= 0.0625` (16 samples, band 0.03): ensemble **1.274 +- 0.159**
versus FD 1.336 +- 0.060 / 1.169 +- 0.096 (z -0.37 / +0.57), decomposition matching the reference
(silhouette `2 Phi_tip` = 0.59; score + x `2 (Phi_flat - Phi_tip)` = 0.75). The SE (12 %) is all the score
term. The cut-off also floors the emission-side `v_t/v_n` tail (`sampling.log_diffuse_density(...,
floor)`, `shape.normal_score`), which removed one outlier only (sd 44 to 35 %): the rest is intrinsic
score-function variance.

**Hybrid form of the normal term (shipped).** The hit's own reward is exact in `v_out`, so its normal
dependence is taken **pathwise** through the reparameterised sampler (`v_out = σ(r n + z1 t1(n) + z2
t2(n)) + u` at fixed draws, weights bounded by `m|v||dn/dw|`), while the value-to-go after the hit
(`lambda + r_later`) stays on the score, so no fitted field derivative enters (5.9's objection does not
apply). Implementation: `wall_adjoint._element_events` forms the tally from the traced sample and the
post-hit state from the frozen one; `shape._gradient_step` weights the normal score by `to_go[1:]`.
**Trap:** the SPARTA tangent frame `t1 ~ v_in - (v_in . n) n` has a `1/|v_t,in|` derivative at normal
incidence (log-divergent variance for no change in the law), so `sampling.diffuse_velocity_general`
differentiates the frame **parallel-transported** from the primal one
(`tests/test_mesh_adjoint.py::test_diffuse_reparameterisation_in_the_normal_is_bounded_and_unbiased`).
Normal-term seed sd 3.1 (score, hit-time cut-off only) to 2.3 (score + emission floor) to **0.77** (hybrid)
e-21; total sd 3.2 / 2.6 / 1.4. **GP result: ensemble 1.191 +- 0.068 e-20 versus CRN FD 1.169 +- 0.096 (h
= 0.05, z +0.19) / 1.336 +- 0.060 (h = 0.1, z -1.6)**; terms: tally 0.620, silhouette 0.585 +- 0.039,
timing 0.065, x -0.015, score 0.000. Five FD steps (0.2 to 0.0125) show no drift, only the `1/h` growth of
a CRN difference across hit-set jumps; |z| <= 1.6 at every step. Figure `plate_estimators.png`.

### 7.9 The specular capsule's `grad lambda` parts: population mixing, and the stratified fit (2026-09-22)

> **Status 2026-10-21.** The diagnosis stands; the fix does not generalise. `shape.hit_ahead` /
> `value_class="hit_ahead"` is **diagnostic-only** since 2026-10-02: a fit within each class drops the
> `grad P (lambda_1 - lambda_0)` term of `grad lambda` (P = probability of a later hit) and conditions on
> the future (coarse circle `mode2`, 4 paired seeds: no class minus FD = +0.007 +- 0.071 e-20, hit-ahead
> class -0.68; also +4 to 20 % in free-molecular squash). `shape.gradient` and the runners default to
> `value_class="none"` and `shape.gradient` warns on `hit_ahead`. Use the class only where `P(later hit)` is
> constant (collisionless convex bodies). The specular-capsule deficit with the mixed pool has no accepted
> general fix; the per-cell fit with guards and normals (7.11) is the current treatment.

**The measurement.** The specular capsule at the GP resolution (4 CRN seeds) gave ensemble 0.964 +- 0.037
versus CRN FD 1.351 / 1.355 e-20 (z -11 paired, -27 %): resolution is not the cause. The decomposition
is the diagnosis: the **tally term alone matches the FD seed by seed** (tally + timing paired Δ = +0.01
+- 0.35 e-21), the silhouette is about 0 as it must be for a specular wall, and the whole deficit sits in
the post-hit value-gradient parts (`grad_x lambda` -0.63, `grad_v lambda` -3.26 +- 0.32 e-21). For a convex
body in a collisionless flow the own-reward derivatives of the cap hits are exactly `dJ/dw` and a
specularly reflected particle never re-hits, so its value-to-go and gradients are zero. The earlier
specular artifacts say the same (tally + timing versus the full estimator: z -0.9 / +1.0 analytic circle,
-1.1 / -1.1 level-set circle, +0.3 / +0.5 `cos 2θ`); the `grad lambda` parts there only double the paired SE.

**Why the fit is biased at a cap.** The value function next to a wall is discontinuous along the hit/no-hit
boundary (about-to-hit particles carry `lambda ~ 2 m v_x`, particles that just left about 0). The
one-sided rule (7.6 item 4) separates them exactly at a *flat* wall but not at the cap: a particle
leaving the cap's tilted tangent plane can still head for the flat face below it, so the pool mixes in
front-face-bound nodes. The size is set by curvature: `dv_out/dw ~ 1/R`, 7.5x larger on the cap (R = 0.4)
than on the circle (R = 3), which is why the circle showed noise where the capsule shows a bias.

**The stratified fit** (`ensemble_device.node_gradients(klass=)`, `shape.hit_ahead`): a slot's nodes are
the slots of its own class, class = *a body hit at a later captured step* (`rec.walls.face >= 2D`,
reverse cumulative OR). Specular capsule: mixed pool 0.964 +- 0.037 e-20; hit-ahead class **1.356 +-
0.011** (x -0.02, v +0.05 +- 0.22 e-21; paired Δ vs FD +0.05 +- 0.56 e-21, z +0.08). Artifacts from before
2026-09-22 (phase D, LS2, 2026-09-21 cluster jobs) are mixed-pool.

**Diffuse circle (gate GD, `acc = 1`, `T_w = 300 K`, hybrid normal term, silhouette band 0.05, grazing
0.05, stratified fits; 4 CRN seeds).** Terms (e-20): silhouette 0.818 +- 0.046, tally 0.373, x 0.141 +-
0.038, score -0.020; **estimator (report convention, no timing) 1.312 +- 0.052**; timing 0.036 +- 0.21;
analytic `F/r` 1.381; CRN FD 1.442 +- 0.03 (h = 0.1) / 1.128 +- 0.16 (h = 0.05); paired z (no timing)
-2.3 / +1.0, with timing -0.4 / +0.8. The silhouette term is 61 % of a diffuse body's radius derivative
(45 % on the plate), so its band-width bias is the first suspect for the residual few per cent; the
paired diffuse results after the timing-band and bias work are in 7.11.

### 7.10 Tracking on optimised shapes: failures of the cluster runs and SPARTA's remedies (2026-09-23/24)

The first cluster optimisations (`optc`: the circle at `fnum x 1` on the 20 x 20 grid, level set `h_phi =
0.0625`, `n_sample = 16`, S in {1, 2, 4, 8} CRN seeds x {GD, Adam}) failed one after another as shapes
left the circle. Each failure was a real defect, fixed at its root rather than by constraining the shape:

| symptom | cause and fix |
|---|---|
| `build_mesh`: no clean flow corner (GD iterate 2) | `levelset.reinitialise` projected band knots with 8 clipped steps but the band is `3h + reach` = 11h at `n_sample = 16`: stopped-short knots entered the surface cloud, giving rings of spurious zero crossings. Fix: `ceil(band/h) + 4` steps; unconverged knots left out of the cloud |
| `wall_recrossed` (Adam iterate 1) | Adam's elementwise normalisation discards the H1 smoothing (`k = 0` step = `sign(g)`): overhangs. Fix: smooth the Adam step again (roughness 294 to 26) |
| `wall_recrossed` (later) | concave windward notches: 5 to 7 specular reflections in one step, more than the 4 passes. Fix: stage 1, an unresolved particle is deleted and counted (`surface_stuck`, SPARTA MAXSTUCK); stage 2, the lane tracker below |
| `inside_body` | after a reflection the level-set hit search starts on the surface with `phi = +-` round-off; `phi <= 0` made the first sample interval never a `+ -> -` bracket, so a grazing re-entry on a concave wall closer than one sample spacing was missed. Fix: a leaving start on the surface counts as `phi = +0` |
| `SplitCellError` | a notch cut a pocket (2 % of cell 91) off a cell's flow. Fix: SPARTA split cells with sub-cells (PSEUDOCODE 4.5) |
| `build_mesh`: no clean flow corner (Adam) | a flow pocket entering a cut cell through a face with all four corners in the body; the corner is the reference of the *polygon* inside test only. Fix: required for polygons only |
| `replay error` 1.8e-12 to 7.6e-12 above 1e-12 (shape VJP replay) | round-off of one arithmetic compiled twice, amplified by grazing reflections on concave walls (each moves the hit point by `δ/θ`: one ulp becomes 1e-12 after four reflections at 0.4°; `fnum x 4` runs of the same shape: about 8e-16): a rare physical amplification, not the tracker. Fix: the check is split, any flipped outcome reports 1, the continuous error is held to `REPLAY_TOL = 1e-9` (mesh and plane runners; `ensemble_couette` keeps 1e-12, box walls only) |

**The lane tracker (PSEUDOCODE 9.6.1).** The fixed pass count was the one non-SPARTA piece of the tracker,
and every pass searched all N particles although a particle without an event in pass 0 has none later.
The lane tracker searches everyone once, keeps reflecting the about 0.5 % with an event on a compacted
lane until their time is used up, and runs a pass only while one is needed (lane forward: `lax.while_loop`;
traced replay: `fori_loop`). The examples default to `tracker="lane"`, `K_BOUNCE = 32`; the library default
is still `SimStatic.tracker = "stencil"`. Roughest optimised shape (`circle_coarse_S1_gd` coef_8, RTX 3050
Ti laptop, float64, `examples/mesh/tracker_bench.py`): forward 115 s (stencil, 4 passes), 62 s (lane, 4),
55 s (lane, 32 passes, 0 stuck), identical drag tally; shape gradient equal to the stencil's to 2.3e-15
relative in every coefficient. The inflow's entrants were still tracked by the dense stencil, `K` passes
every step, a hidden per-budget cost; they now use the lane too, and the pass budget costs nothing
measurable.

**Memory per particle slot** (stencil 4 passes / lane 32 passes / lane + compact record + slim replay):
event record (300 captured steps) 62.7 / 43.7 / 36.7 kB; peak captured forward 95.3 / 45.6 / 46.0 kB; peak
full adjoint chain 115.9 / 83.0 / 63.7 kB (-45 %); chain time at 15 256 slots 271 / 225 / 213 s. The record
keeps later passes as the events actually recorded (`wall_adjoint.RecordEvents`, -16 %). A stage probe of
`shape_terms` (`scratch/mesh/probe_memory_stages.py`, 15 k slots) found the replay kept alive through both
backwards (+292 MB) with per-event rewards next to their totals, timing rewards in per-event form, and
every device backward running the host bookkeeping (`from_device`) that `shape_terms` discards. Now:
totals on demand (`HitRewards.totals()`), timing rewards as totals only,
`device_backward(host_adjoint=False)`: replay +145 MB, peak 1243 MB. The remaining about 300 MB spike in
the shape VJP is the gradient step's fixed working set (pooled fits gather nodes per query chunk), not
growing with particle count. At the fine circle (240 k slots) the chain is about 15 GB (stencil about 28
GB), inside a 24 GB rtx slice (to be confirmed on the slice).

### 7.11 Status after 7.10: bias study and per-cell fit (2026-10-02 to 2026-10-19)

Recorded from the working notes and checked against the code defaults named; the studies themselves
live in `scratch/bias/`, `scratch/cellfit/`, `scratch/gapgate/`, `scratch/guardgate/`,
`scratch/thingate/` and the validation deck (`docs/math/shape_validation.tex`, `shape_bias_*.tex`;
PSEUDOCODE 9.7.2), not in this file. Not re-run for this cleanup.

* **Defaults now:** `--variant along_pool_plane` (the pooled one-sided fit with `node_gradients(halfspace=)`
  cutting nodes more than a quarter cell behind the hit's tangent plane: a pooled fit reached across a thin
  body, `one_pool` -19 %, `along_pool` -9 % at s = 0.35); `value_class="none"`; the window-edge timing term
  is included in reported gradients with a 10-step band (`--timing-band 10`, `EM.timing_band_setting`): about
  0 on a steady window (+0.03 +- 0.25 %); a 0.2-step band's apparent +2.8 % was noise and the term was
  -35 % of `dD/ds` on one diffuse seed; `grazing = 0.05` for optimisers.
* **New terms:** the cut-volume term (`shape.volume_weights` / `volume_cotangent`, from
  `Backward.cell_collision`: the NTC attempt count scales as `1/V_c`; gated by volume scaling, all cells z
  -1.2, cut cells z +0.8; -2.9 % of `dD/ds` at Kn 0.5); along-flight with `L* lambda`
  (`gradient_variants(collision=)`: the fit agrees with `T + L*` to 0.8 %, with `T` alone off by 3.2 %);
  the collision-order term (`ShapeTerms.order`, `shape.order_jump`: +0.4 % at Kn 0.5).
* **Free-molecular squash:** specular within about 2 % of FD at s = 0.35 / 0.6 / 1; diffuse s = 1 within
  about 2 % (silhouette about 50 % of the gradient). The residual -2 % is systematic and mostly *horizon*
  (300 to 600 captured steps: +0.8 % to +1.1 %, wall re-hits), not FD curvature or the grazing cut-off
  (the cut-off is the *variance*: the sd ratio scales with it; smoothing features cuts it 3x).
* **Kn 0.5:** the earlier "adjoint 7 to 9 % below precise FD" gap was FD noise (a 64-seed FD gives h = 0.1
  = h = 0.2 to +0.14 +- 0.67 %); the shipped kNN/quad fit was -3.7 +- 0.8 % and cubic-v -0.7 +- 0.8 %,
  which led to the default `CELL_BASIS` (cubic-in-velocity per-cell fit; horizon 1000 versus 300
  captured steps +2.1 %). Residuals not located: free-molecular squash about -1.5 %, thin-body squash
  -4 % in every fit, nose gradients of the free boundary (`optm`) at M10.
* **Per-cell fit (`jax_dsmc/cell_fit.py`):** about 10x faster than kNN k64 per squash seed (about 100 s
  versus about 1000 s). Unguarded `cell:quad` and `cell:quad:rb` were unstable (sparse cells'
  underdetermined quadratics feed back; seeds +97 % and -40 %); `adapt1.5` (order floor by occupancy)
  stabilised it; the cubic-v blow-up on thin M10 bodies was velocity-space extrapolation (post-collision
  queries in the gap between the beam and the wall-reflected gas; leverage p99 about 1e3 versus about 30
  for quadratics), cured by the `guard<h>` option (per-query highest order with leverage <= h; `guard32`
  shipped; `h = 2` ruled out). Complex cells (`R < 0.9`) fall back to the per-query one-sided fit
  (`align0.9`, `both`).
* **Optimiser (`examples/mesh/optimize.py`):** `--area-mode tangent` holds the area exactly (gradient and
  step projected off `grad A`; a uniform shift restores the `O(step^2)` remainder: 1e-12 versus 1 to 2
  cm/iterate for the old projection; the augmented-Lagrangian mode wound up and is void for drag), `--smooth
  8` knots, convergence on a 5-iterate window (`window_converged`), constant normalised step (the largest
  knot moves exactly `step`, so it random-walks near the optimum; step decay proposed, not built),
  `--restart-steps` warm start. Observed/predicted step ratio 0.88 with `smooth8`, falling to 0.4 to 0.7
  late in all runs. The `squash` Kn 0.5 optimum is ill-conditioned (slope ratio within 2 % of 1 over s =
  0.5 to 0.9), so a 1 to 2 % gradient bias moves it substantially.
