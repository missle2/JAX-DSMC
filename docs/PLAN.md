# JAX-DSMC — roadmap (rewritten 2026-10-21)

The goal: a DSMC solver built in JAX from the ground up so that an adjoint for the inhomogeneous
Boltzmann equation with a general collision operator falls out of the forward solve, and so that the
gradient can drive **shape optimisation** of bodies in rarefied flow. This file says why the project
is shaped the way it is, what is done (one line per phase, with a pointer), what is current, and what
is open together with the reasoning for the choices not yet taken. State of play, traps and run
instructions: `docs/HANDOFF.md`. The contract: `docs/PSEUDOCODE.md`.

## 1. Why the project is shaped this way

References: **SPARTA** (`~/sparta/`, cited `file.cpp:lines`) and **dsmcFoam+** (`~/OpenFOAM-v1706/`,
cited `[DF]`) are implementation references - machine-local, not vendored - and **where they disagree is
the most useful part**: each divergence is an explicit decision in spec 18. Adjoint theory: `[P1]`
2009.01363 (0-D, Maxwell molecules), `[P2]` 2207.11579 (0-D, general kernel), `[P3]` 2603.20946
(spatially inhomogeneous), in `papers_reference/`.

1. **The base adjoint is reverse-mode AD** (original Findings 1, 2, 4). `gamma_k = A^T gamma_{k+1}` is the VJP of the collision map,
   the non-colliding branch is the identity Jacobian, parameter sensitivities are VJPs through the
   reparameterised samplers; pair choice, collision times, sigma are "determined externally" = frozen
   randomness = common random numbers + integer decisions. The score terms of `[P2]` are obtained from a
   DiCE surrogate (`magic(L) = exp(L - stop_grad(L))`, one `(N,)` accumulator). Transport and specular
   reflection adjoints are the VJPs of `x += v dt` and the mirror map. `tests/test_adjoint_oracle.py` makes
   the bet falsifiable by hand-coding both papers' recursions.
2. **Every stage is a pure function of (state, params, frozen random draws) with static shapes**, and every
   stochastic branch exposes a differentiable `log h`. That is the one hard requirement on the forward.
3. **The pathwise/score estimator is not the production gradient.** For expectations over DSMC noise the
   score routes were measured biased (growing with trajectory length) or far too noisy; the particle-carried
   ensemble adjoint (`[P3]` section 5, extended to the real solver) is (decided 2026-09-15, `HANDOFF.md` 2.2).
4. **Embedded bodies are SPARTA-style**: a Cartesian grid with cut / split cells keeps cell lookup
   closed-form and confines geometry to per-cell element tests. dsmcFoam+'s polyhedral tet-walking tracker
   needs a data-dependent loop per particle - structurally hostile to static shapes; kept as the fallback
   if unstructured meshes are ever required.
5. **Main stays differentiable.** The fast non-differentiable SPARTA-faithful forward lives on branch
   `sparta-match` only.

## 2. Standing design decisions

| # | decision | note |
|---|---|---|
| 1 | positions `(N, D)`, `D in {1,2,3}`; velocities always `(N, 3)` | Bird / SPARTA convention; keeps VSS and Chapman-Enskog theory valid |
| 2 | uniform orthogonal grid with traced `lo`/`hi`, static `n`; mesh interface kept to `cell_of`, `cell_volume`, `cell_bounds`, `apply_bc` | the embedded-surface layer replaced what is behind those names without touching `collide` / `sample` |
| 3 | production gradient = particle-carried ensemble adjoint + shape VJP; pathwise AD exact where applicable | **supersedes** the original "pathwise plus score terms"; the AD-derived surrogates remain the oracles' subject and serve pathwise parameters |
| 4 | backward pass: AD-derived (pathwise) or the hand-written virtual-partner pull-back (ensemble); Algorithms 2/3 hand-coded as oracles | |
| 5 | gradients to IC, wall models, VSS parameters, `dt`, `fnum`, domain extent, and body shape (level set) | |
| 6 | single species, monatomic; arrays are `(S,)`/`(S,S)`/`vremax (C,G,G)` so multi-species is an addition | |
| 7 | long-term target N ~ 1e9-1e10 on clusters; single device first | measured: the layout did *not* shard for free; x-slab decomposition (`decomp.py`) was needed |
| 8 | f64 throughout | energy conservation to ~1e-14 |
| 9 | stale-max NTC (dsmcFoam+ rule) with an a-priori thermal majorant floor | spec 7.2.3, 17.5 |
| 10 | pluggable pair selection: NTC with replacement (production), disjoint permutation (oracles; the papers pair without replacement) | |
| 11 | Gaussian Maxwellian sampling (`sqrt(kT/m) N(0,1) + vstream`); Rayleigh form only for diffuse-wall normal and inflow | |
| 12 | `fnum` is a per-cell field `(C,)`; the weight field is in, the RWF clone/delete machinery is out | clone/delete changes particle count and breaks static shapes; guard: non-uniform `fnum` must be asserted absent in multi-cell runs (gap U7, open) |
| 13 | ~~NTC remainder seeded to U[0,1)~~ | superseded: stochastic rounding of the attempt count with `log P(n given f)` removed the remainder (the deterministic carry gave exactly zero `dt`/`fnum`/volume gradients) |
| 14 | `ruff` gates the code; **no nested function definitions** (scan bodies are module-level functions bound with `functools.partial`) | config in `pyproject.toml`; cited from code as "decision 14" |
| 15 | RWF splits in two: the weight field is adopted, clone/delete is not | its hard part (sensitivity under a changing particle count) is the same problem the ensemble adjoint solves |

Other superseded plans: "line search with periodic reinitialisation" for the optimiser (as built: fixed
largest displacement per iterate, reinitialisation every iterate, tangent area mode); tests on the GPU
(they run on CPU with four forced host devices; `--gpu` before a milestone).

## 3. Done

| phase | what | where |
|---|---|---|
| 0 | specification, revision 3 (unapproved) and executable structural verification + gap register | `docs/PSEUDOCODE.md`, `docs/phase0_verification.py` |
| 1 | foundations: `safe`, `state`, `rng`, `grid`, `sampling`, `sort` | `jax_dsmc/`, `tests/` |
| 2 | **M1** 0-D BKW relaxation; gates U1 and U3 closed 2026-09-04 (U3 changed the design: thermal majorant) | `examples/bkw`, spec 17.4-17.5 |
| 3 | **M2/M3** periodic transport 1-3 D: Bird normal shock vs Bird and Alsmeyer, Taylor-Green viscosity (+0.38 %), turbulence cases; `A_max` solved by `run_adaptive`; multi-device x-slab decomposition (`decomp.py`; `parallel.py` the negative result) | `examples/shock`, `examples/turbulence`, `examples/scaling` |
| - | Oracle 3 / gap U5 closed 2026-09-08: `move -> collide` ordering pinned | `tests/test_adjoint_oracle.py` |
| 4 | **M4** diffuse / thermal walls, inflow / outflow, Couette viscosity gate (Kn -> 0 intercept), two-sided-inflow shock (0.7 % of Rankine-Hugoniot); U6 closed | `examples/couette`, `examples/shock/inflow_driven.py` |
| 6 | **M6** pathwise + score adjoint: oracles exact, pathwise gradients vs flip-free FD in 1-3 D; score-route bias and variance measured and the routes retired (2026-09-10..15) | `jax_dsmc/adjoint.py`, `docs/ATTRIBUTION.md`, `examples/shock_adjoint`, `examples/gradient_bench` |
| 7/8 | **M7/M8 boundary adjoints, via the ensemble adjoint** (route b, 2026-09-15): inlet temperature / density, Couette wall parameters, VSS kernel parameters (virtual-partner pull-back, 2026-09-16); wall `T` / velocity also pathwise (1e-11). The `[P3]` epsilon-regularised thermal wall (`eps = 0` bitwise exact) exists as a reference; diffuse walls enter the ensemble adjoint as kernels with no epsilon | `ensemble_*.py`, review section 5 |
| 5A | embedded surfaces, geometry vs SPARTA dumps (2026-09-17) | `jax_dsmc/mesh/`, `examples/mesh/geometry_checks.py` |
| 5B | mesh forward vs SPARTA argon circle / sphere: drag +0.3 %, fill, fields; free-molecular drag | `examples/mesh/forward.py`, `report.py` |
| 5C | mesh-aware adjoint: value-form rewards, `grad_x` fit, replay + VJP w.r.t. geometry (2026-09-18) | `jax_dsmc/shape.py`, spec 9.7 |
| 5D | analytic-circle radius gradient vs CRN FD (z 0.5); shape controls restricted to normal displacements (2026-09-20) | `examples/ensemble_mesh.py` |
| 5E | C1 level-set bodies; LS0-LS3 gates (circle, `cos 2 theta`, sphere); diffuse silhouette, hybrid normal term, grazing cut-off (2026-09-20..22) | `jax_dsmc/mesh/levelset.py` |
| 5F | split cells with sub-cells, lane tracker with `K_BOUNCE = 32`, compact records (2026-09-23/24) | spec 4.5, 9.6.1, review 7.10 |
| 5G | shape-bias study: cut-volume, order, along-flight + `L* lambda`, timing band, `along_pool_plane`; `hit_ahead` retired (2026-10-02..12) | `scratch/bias/`, validation deck |
| 5H | per-cell backward fit (2026-10-11), cubic-in-v (10-12), leverage guard (10-19) | `jax_dsmc/cell_fit.py`, spec 9.7.2 |
| 5I | free-space box (2026-10-15), warm start (10-19) | `forward.build(boundary="free")`, `--restart-steps` |
| perf | forward 67.6 -> 6.3 s (CPU, 1000 steps); optimiser seed 99 -> 38 s; paused 2026-10-02 | `HANDOFF.md` 2.5 |
| opt | cluster optimisation studies: `optc`, `squash`, `optab` (smoothing / area), `optm` (regimes) | `examples/mesh/cluster/` |

## 4. Current (2026-10-21)

**Optimiser comparison.** Question: which step rule survives the noisier gradients expected from
complex 3-D flows. Done (`optrules`, `optrules2`, `optrules1seed`; `scratch/optrules/analysis.py`,
`noise.py`): Adam robust even at one seed per iterate and kept as the reference; `tr` fast then parks at
its floor; L-BFGS pairs are noise; MMA badly scaled and fragments shapes (`HANDOFF.md` 2.6). Adam's limit
is its fixed step size (too small early, overshoot late), so the running `optax1seed` group tests step-size
control through optax (`examples/mesh/optax_rules.py`: cosine schedule, schedule-free, Prodigy, AdEMAMix,
and the optax Adam control, now the default `--optimizer ox_adam`). Decision criteria: noise robustness
first, per-iterate speed second. No GCMMA (forward too expensive).

**Roughness of optimised shapes**: analysis in `scratch/optrules/` (`roughness.py`, `reinit_drift.py`,
`noise_shapes.py`). The level set has ~400 boundary DOF, a 0.5 m DSMC cell resolves ~50; the gradient has
no information below the cell size, so the optimiser fills the null space with noise.

## 5. Next / open

Ordered by value. Reasoning is kept for the choices still to be made.

1. **Choose the production optimiser rule** (from section 4), then run the production set of regimes to
   convergence. Candidate additions: step decay, momentum reset on spikes.
2. **Control the design space.** Options: wider `--smooth`; coarser `--levelset-h` (measured equal to
   smoothing at equal width, so no free lunch); a curvature penalty in the objective (proposed
   2026-10-12, not built; breaks the "pure drag" comparison, so it needs its own A/B); fewer controls
   (e.g. a low-order B-spline basis). The choice should be settled by the roughness analysis.
3. **Close the remaining adjoint residuals**: thinnest bodies (~-4 % in every fit), optm nose (-3.8 %),
   free-molecular squash (~-1 %), diffuse circle (-9 %, GD gate: more seeds + a band sweep). Method: pair
   each term against CRN FD seed by seed.
4. **Translation-type controls** (x/v part of a moved flat wall). Recorded options: replica evaluation at
   the displaced state (collisionless only), the epsilon timing score (4-8x the FD sd), a different
   estimator. Not needed for normal-displacement shape optimisation; blocks e.g. moving flat plates.
5. **3-D shape optimisation.** Sphere gradient is gated (z -1.75 / -0.91); needs the optimiser with a 3-D
   level set, memory sizing (the adjoint's records are the budget, ~64 kB/slot/step 2-D), and A100/H200
   (f64). Expect noisier gradients - this is what the optimiser comparison is for.
6. **Forward speed (paused 2026-10-02).** Bounded options total ~20 % (6.3 -> ~5 s); < 5x SPARTA serial
   needs a tracker-bookkeeping rewrite - not done for the risk to verified behaviour. Cluster: try XLA
   command buffers on native-Linux GPUs.
7. **Hierarchical (multi-level) grids** - deliberately not built (SPARTA `create_grid ... level`,
   `adapt_grid`). Mesh tables are per leaf cell so the extension is confined to:

   | touch point | change for a two-level grid |
   |---|---|
   | `grid.cell_of` | static-depth descent: coarse `floor` -> `child_offset[c0]` (-1 if unrefined) -> child `floor` |
   | `grid.cell_bounds` / `cell_volume` | per-leaf `lo[C], hi[C]` tables |
   | `output.write_vtr` | `.vtu`, or one `.vtr` per level |
   | `viz.field_to_grid` | rasterise leaves onto the finest level |
   | `decomp.slab_of`, `sub_grid` | slabs by coordinate, leaf ranges per slab |
   | `ensemble_backward` / `ensemble_device` fit metric `scale = dx` | per-query cell `h` |
   | stencil sufficient test `abs(v) dt <= R h` | per-cell `h` (already read from `cell_bounds`) |

   Nothing in `sort`, `collide`, `sample`, `inflow` or the trackers needs to change.
8. **Real multi-GPU scaling numbers** for `decomp.py` (only forced-host correctness so far). Re costs `Re^D`
   in cells: 4 x H200 reaches Re ~ 1500 in 2-D, ~66 in 3-D; the 3-D Re = 1600 benchmark is not claimed.
   Sharding to N ~ 1e9-1e10 would replace the global `argsort` with per-shard sort + migrate; per-device
   capacity and the migration buffer are silent-overflow capacities (test by counting).
9. **Reversible O(1)-memory adjoint** - not built. The collision *map* is an involution (re-applying it with
   sigma and alpha exchanged recovers the input; this rests on the map, not on `A`, which is rank 4 so
   `A^-1` does not exist), and advection / periodic / specular are invertible, so forward state could be
   recomputed backwards instead of stored. Diffuse re-emission is the irreversible piece. Revisit when the
   adjoint's memory (not time) caps a run; `jax.checkpoint` covers the near term.
10. **Multi-species, Larsen-Borgnakke, chemistry.** Arrays are already species-shaped. The LB
    differentiability hazard is narrower than first stated: the diatomic case `zeta_rot = 2` has a
    closed-form inverse CDF (`E_rot/E_c = 1 - R^(1/(5/2 - omega))`, [DF]
    `LarsenBorgnakkeVariableHardSphere.C:243-278`); only general polyatomic `zeta > 2` needs a gamma / Beta
    rejection loop (SPARTA `collide_vss.cpp:457-603`). [DF] fixes the mode order (electronic ->
    vibrational -> rotational) and gives the temperature-dependent `Z_v(T)`.
11. **Parked, with reasons**
    - *Fixed attempt count folded into the acceptance probability* (`p = 0.5 npc (npc-1) dt fnum q / (A V)`):
      verified equivalent to NTC (collision counts within 0.4 %, 2.3x fewer slots, no stochastic rounding
      needed) but changes the core acceptance probability and was verified for a single group only; not
      adopted (spec 8.5.2, `scratch/probe_fixed_attempts*.py`).
    - *Sub-cell partner selection* ([DF] default; SPARTA `partners subcell`): a real accuracy gain at coarse
      resolution but makes pair probability position-dependent, which [P2] section 3.1.2 relies on not
      happening; would need its own score term. Not taken.
    - *Custom `jax.custom_vjp` on `scatter`* for the AD route: only if plain AD ever needs to be faster
      (it is ~3-5x forward); Oracle 2 exists to check it.
    - *Differentiable mesh / adjoint-driven AMR*: deferred beyond the level-set shape path.
    - *Reference cases from [DF]* as extra validation: `hypersonicCorner` (Bird 1994 section 16.2,
      Scanlon 2010) and `supersonicFlatPlate` (Bird 1994 section 14.3) give published profiles for
      M3/M4-style checks; `orion107kmNR` (STL capsule) for 3-D bodies; `axisymmetricFlatnosedCylinder` needs
      radial weighting. None built.
    - *Remaining register rows* (`phase0_verification.py`): U4 `A_max` sizing from the inflated `Sigma`,
      U7 non-uniform `fnum` guard, U8 `h_crit` extrapolation, U9 cost of the scratch-row fix, U10 which
      collision reparameterisation has lower gradient variance. U4 and U7 are silent-failure class and
      blocking before the milestones that make them live.

## 6. Verification strategy (kept short)

Each milestone gate is a physics test against an analytic or independently published result (BKW,
Chapman-Enskog, Bird, Alsmeyer, SPARTA argon, closed-form free-molecular drag), never a regression
against our own prior output. Gradient checks use common random numbers and a flip guard; shape-gradient
estimators are compared against the CRN FD of the same horizon, seed by seed, with standard errors.
Performance is gated structurally (jaxpr shape, XLA cost model), not by wall-clock. The commands are in
`README.md` and `docs/HANDOFF.md` section 5.
