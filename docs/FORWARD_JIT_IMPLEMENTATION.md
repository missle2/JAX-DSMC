# Forward/JIT follow-up implementation

> **Status (cleaned up 2026-10-21).** Record of the forward-solver and source corrections made on
> `fix/paper-adjoints` after [ADJOINT_CORRECTIONS.md](ADJOINT_CORRECTIONS.md) (baseline commit
> `1de4928`, verification snapshot 2026-09-14). The branch is merged into `main`; the interfaces
> below exist as described unless marked otherwise. Both documents were reviewed in
> [ADJOINT_BRANCH_REVIEW.md](ADJOINT_BRANCH_REVIEW.md), which is later and wins on disagreement.
>
> Superseded since this was written:
>
> * the "remaining acceptance work" and "solver gates" lists (end of this file) are closed or
>   replaced: route (b), the ensemble adjoint, is the production gradient and handles gates 1 and 2
>   by construction (2026-09-15); surface tallies, windowed surface objectives and an embedded-surface
>   geometry now exist (`simulate.run(emit_surface=)`, `jax_dsmc/mesh/`, `jax_dsmc/shape.py`);
> * the 198-job validation matrix was never launched and is not the plan;
> * score-estimator gradient experiments are retained only as the baseline the ensemble adjoint is
>   compared with (`examples/gradient_bench`);
> * the forward speed work after this record (backend-dispatched tracker, sort carry) is in
>   `docs/HANDOFF.md` section 2.5 and `scratch/perf/`. `main` must stay differentiable; the faster
>   non-differentiable `sparta_jax` forward lives only on branch `sparta-match`.

NTC with uniform random partners within each cell is the collision scheme; there is no Poisson
collision mode and no collision subcell/neighbour scheme.

## Implemented interfaces

* `SimStatic.count_scheme`: `"stochastic"` (default) or `"remainder"` (SPARTA's carry,
  `CollideState.remainder` persisted across calls/chunks). Remainder mode rejects trajectory-score
  requests: its expectation derivative is not implemented, and a carry has no probability to score.
* `SimStatic.majorant_mode`: `"thermal"` (compatibility default) or `"certified"` (`ntc.certified_majorant`:
  bounds relative speed by twice the cell peculiar energy, includes the collision-rate speed clamp,
  equal-mass monatomic VSS with `0.5 <= omega <= 1`; fixed during each collision stage). The
  dependence on the pre-stage state is `stop_gradient`-ed, so it makes the frozen quantity a
  certified bound, not a proof of a complete finite-step expected gradient (random-rounded NTC laws are
  not exactly majorant-invariant). Measured: at production occupancies the certified bound proposes
  *fewer* attempts than the thermal rule and is a true bound; use it for gradient runs
  (branch review 4.3, 4.11).
* `ntc.rounded_weak_derivative(mean, continuation, side=...)` evaluates the forced count continuation
  difference at an integer mean (left/right derivatives can differ). Isolated oracle only, not on the
  production path; the `count_support_boundary` diagnostic (checked by `validation.require_valid`)
  flags the situation.
* `simulate.checked_run` validates concrete physical inputs and all returned diagnostics; pure `run`
  stays JIT-compatible, so traced callers must validate at their host boundary. `run_adaptive`
  rejects unresolved majorant/boundary/source diagnostics after capacity repair.
* `sample.zero_accumulator(grid, mass)` pools centred sufficient statistics; `merge_accumulators`
  merges chunks stably; `finalize` returns pooled velocity/temperature and time-averaged density,
  pressure and heat flux (fixed cell measures and uniform weights required). This replaced averaging
  separately normalised temperatures.
* End-of-step moments now correspond to the returned post-collision particles; pre-collision moments
  (only the cell temperature, for the majorant floor) stay internal.
* `SimStatic.checkpoint_block_size`: block checkpoint scans, tested for value and gradient
  equivalence. Optimal scheduling and "constant-horizon memory" are not claimed.

## Sources, tracking and geometry

* Physical inlet area includes `grid.transverse` (`inflow.face_area`). Candidate allocation compacts
  actual births, supports per-face capacities above particle storage, and `InjectionDiagnostics`
  separates requested, inserted, physically escaped, sampler-failed, candidate-overflow,
  storage-overflow and tracking-exhausted births (the identity is checked).
* `inflow.flux_quantile` inverts the untruncated inward drifting-Maxwellian flux distribution; its
  custom JVP uses the implicit CDF derivative instead of differentiating rejection decisions
  (the old derivative was 31 % low). `flux_integral` keeps the small inward tail under strong outward
  drift and reduces to Bird eq. 4.22. `target_count` uses `flux_integral`; the legacy `mol_inflow`
  keeps SPARTA's -3 cutoff for historical comparison. Tests cover speed ratios -10 to 10 against an
  independent SciPy flux integral (256 Gauss-Legendre nodes; 96 showed endpoint error). More
  extreme ratios are not covered.
* Production newborns use chronological tracking with their remaining travel time:
  `inject(track=True)` is what `simulate.run` calls; `track=False` is the legacy direct placement
  for standalone callers. Trajectory mode preserves the incoming scalar score across slot reuse and
  scores count events including no-birth outcomes.
* Cartesian earliest-hit tracking handles specular, diffuse, periodic and absorbing faces in
  D = 1, 2, 3; exhausted budgets are explicit; surface ids break simultaneous-hit ties in axis
  order. `advance(return_tallies=True)` returns extensive momentum, total energy and wall-relative
  heat transfers; `surface.rates` supplies physical normalisation. `simulate.run(emit_surface=True)`
  now accumulates the tallies per step (the original gap named here is closed).
  General corner/grazing derivatives are handled only by the later grazing cut-off and
  silhouette term (branch review 7.8).
* `jax_dsmc/geometry.py` (43 lines) holds shared Cartesian measures, surface sampling, `first_hit`,
  and `averaged_planar_impulse`, the collisionless planar shape reference that integrates the initial
  wall distance analytically (a pathwise AD at frozen hits gives 0 for it). General geometry is in
  `jax_dsmc/mesh/` (cut cells, level set), not here.
* Attempt-slot random streams (`fold_in` per slot) and inlet candidate draws are capacity-independent.
  The same change altered every forward stream, so BKW/shock/Couette/variance artifacts from before
  the branch are not bit-reproducible (branch review B6). Wall re-emission and P3 retention streams are
  not yet independent of capacity/partition.

## JIT P3 inlet

`ensemble_inflow_jax.forward(cfg, temperatures, key, injection_counts, capacity)` and
`backward(trace, cfg, key)` are compiled array-only entry points; config, capacities, fixed birth
counts and the reward callable are static; data are padded arrays and masks; the trace carries its
configuration and terminal reward. The mathematics is P3's (disjoint Maxwell collisions,
collision-before-transport, all nominal births in the normalisation, both partner terms, fixed
source counts, affine or ballistic reconstruction), separate from production NTC; exact tiled top-k
selection without SciPy or host callbacks, which bounds scratch storage but not work (queries x
retained population). Host-versus-device collisionless agreement is tested on CPU. An open item: on
JAX 0.6.2 with an RTX 3050 a 16-particle capacity-failure case compiled for more than 8 minutes
(see [ADJOINT_CORRECTIONS.md](ADJOINT_CORRECTIONS.md), GPU compile stall). Smoke command:

```bash
source /home/aakash/jax_dsmc/activate_jax_env.sh
python -m examples.p3_replication --backend jax --particles 256 --cells 4 \
  --steps 2 --retained 8 --source-samples 8 --partners 2 --neighbors 8 --seeds 4 \
  --output examples/p3_replication/results/jit_inlet_smoke.json
```

## Gradient experiments (`examples.gradient_convergence`)

Runs production NTC Couette, reflected shock and two-sided inlet-shock transients with pathwise
sampling/collision derivatives plus the full trajectory stochastic correction; independent seed rows,
four paired FD steps (0.1, 0.05, 0.025, 0.0125), normalised equivalence intervals. `--baseline` is a
fixed unscaled objective value estimated independently of the seeds (never reuse one across mesh,
horizon or parameter); `--seed-start` gives disjoint pilot/confirmation ranges (pilot 0-31,
confirmation from 32); `--seeds` fixes the run length; results are appended atomically and resumed
only when the configuration, FD steps, JAX version and the SHA-256 fingerprint of the solver sources
(`implementation_sha256`) match. Equivalence requires the confidence interval to fit inside a
predeclared band, not merely contain zero.

The 32-seed pilots were inconclusive and the two interrupted 2048-seed confirmation runs
(`ntc_couette_speed_confirmatory.json`, `ntc_shock_temperature_confirmatory.json`) predate the
fingerprint and cannot be resumed. These runs were then superseded by the production-estimator
benchmark (`examples/gradient_bench`, branch review 4.8-4.11), which uses the same case definitions
with common-random-number FD and showed the score arms are either biased or too noisy.

## Verification snapshot (2026-09-14, historical)

The focused analytic suite (`test_forward_corrections`, `test_geometry_corrections`,
`test_checkpoint_blocks`, `test_ensemble_inflow_jax`, `test_ntc_reference`,
`test_gradient_validation`) passed 34 tests; the integrated five-case flux-moment regression exposed
Gauss-Legendre endpoint error at 96 nodes (256 passes). The branch review later recorded the full
CPU suite and lint (`ruff`) state; re-run the suite for any current claim.

## Original acceptance work and gates: outcome

| Item | Outcome |
|---|---|
| Controlled timestep/mesh/population/regularisation and P3 reconstruction studies | Targeted studies in branch review 4.9-4.12; paper-scale P3 not reproduced (and not a target). |
| Parameter-induced hard membership changes (gate 1) | Handled by construction by the ensemble adjoint; cut-volume term for shapes. |
| Complete weak count continuation at integer means (gate 2) | Avoided by route (b)'s intensity derivative; `rounded_weak_derivative` off the critical path. |
| Concrete static-connectivity geometry interface (gate 3) | Built as `jax_dsmc/mesh/` (SPARTA-style cut and split cells, analytic and level-set bodies), gated against SPARTA (branch review 7.1-7.5). |
| Windowed surface force/heat objectives, differentiable moving-face oracle (gate 4) | `emit_surface`, windowed value-form rewards, `examples/shape_planar_oracle.py` (branch review 5.11, 7.6). |
| P3 JIT compile time/runtime/memory versus host; small-GPU compile stall (gate 5) | Stall not re-measured; ensemble adjoint device port costs 1.6-2.1x the forward (branch review 5.8). |
| Powered finite-horizon FD validation matrix (gate 6) | Done for inlet, wall, kernel controls and shape radius/mode controls (branch review 5.7, 7.6-7.9). A transient gradient is not a stationary one. |
| Automatic source-capacity replay, distributed source handling | Not built (serial host replay driver only). |
| External SPARTA comparison | Done: cut-cell forward gated against SPARTA argon `circle`/`sphere` (branch review 7.2). |
