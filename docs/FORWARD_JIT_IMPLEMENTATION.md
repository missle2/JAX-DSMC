# Forward/JIT follow-up implementation

This work follows `ADJOINT_CORRECTIONS.md` on `fix/paper-adjoints`. The baseline for
this follow-up is `1de4928`. It retains NTC and uniform random collision partners;
there is no Poisson collision mode and no collision subcell/neighbor scheme.

## Implemented interfaces

- `SimStatic.count_scheme`: `stochastic` (unchanged default) or `remainder`.
  `CollideState.remainder` persists across calls/chunks. Remainder mode rejects
  trajectory-score requests; its expectation derivative is not implemented.
- `SimStatic.majorant_mode`: `thermal` (compatibility default) or `certified`.
  The latter bounds relative speed using twice the cell peculiar energy, includes
  the collision-rate speed clamp, and supports equal-mass monatomic VSS with
  `0.5 <= omega <= 1`. It is fixed during each collision stage. Its dependence on
  the pre-stage state is stopped; **this is not proof of a complete finite-step
  expected gradient**. Random-rounded NTC laws are not exactly majorant invariant.
- `ntc.rounded_weak_derivative(mean, continuation, side=...)` evaluates the forced
  count continuation difference. It isolates the count-law contribution; it does
  not automatically supply every continuation's direct/pathwise derivative.
  At integer means, left/right derivatives can differ. The ordinary interior
  score in the main solver must not be presented as an integer-support solution.
- `simulate.checked_run` validates concrete physical inputs and all returned
  diagnostics. Pure `run` remains JIT-compatible; traced callers must validate
  setup and returned diagnostics at their host boundary. `run_adaptive` now
  rejects unresolved majorant/boundary/source diagnostics after capacity repair.
- `sample.zero_accumulator(grid, mass)` now requires particle mass. It pools
  centered sufficient statistics; `merge_accumulators` merges chunks stably.
  `finalize` returns pooled velocity/temperature and time-averaged density,
  pressure and heat flux. Counts identify empty cells. Fixed cell measures and
  uniform simulator weights are required. This replaces the old average of
  separately normalized sample temperatures.
- End-of-step moments now correspond to returned post-collision particles.
  Pre-collision moments remain internal to collision preparation.

## Sources, tracking, and geometry

Physical inlet area includes `grid.transverse`. Candidate allocation compacts actual
births, supports per-face capacities larger than particle storage, and reports
requested integer births minus inserted/physically escaped births. Sampler failures
are included instead of silently reducing requested flux.

`flux_quantile` inverts the untruncated inward drifting-Maxwellian flux distribution;
its custom JVP uses the implicit CDF derivative rather than differentiating
rejection decisions. `flux_integral` retains the small inward tail for strong
outward drift. The legacy `mol_inflow` helper retains the SPARTA-style -3 cutoff
for historical comparisons; production `target_count` uses `flux_integral`.
Quantile/residual/derivative tests cover speed ratios -10 through 10. Still more
extreme ratios require separate numerical analysis; do not assume arbitrary-tail
accuracy from those tests.

Production newborns use chronological tracking with their remaining travel time.
`inject(track=False)` remains the legacy direct-placement interface for existing
standalone callers; production `simulate.run` explicitly uses `track=True`.
Trajectory mode preserves the incoming scalar score across slot reuse and scores
count events including no-birth outcomes. It does not solve moving-support/cell
membership derivatives automatically.

Cartesian earliest-hit tracking handles specular, diffuse, periodic and absorbing
faces in all supported dimensions; exhausted event budgets are explicit. Surface
IDs break simultaneous-hit ties in axis order. General corner/grazing derivatives
are not validated. `advance(return_tallies=True)` additionally returns extensive
momentum, total energy and wall-relative heat transfers; `surface.rates` supplies
physical normalization. Tallies are a tracking API, not yet a built-in time-window
surface-objective accumulator in `simulate.run`.

`geometry.py` contains shared Cartesian measures, surface sampling and first-hit
queries, plus a collisionless planar shape reference that integrates initial wall
distance analytically. General-grid protocol/partition/output adapters are **not
complete**; this is a foundation, not a claim that only four grid functions remain.

Attempt-slot random streams and inlet candidate draws are capacity independent.
Wall re-emission and P3 retention streams still require a broader logical particle
identity design before asserting complete invariance under arbitrary capacity,
partition or mesh changes.

## JIT P3 inlet

`ensemble_inflow_jax.forward(cfg, temperatures, key, injection_counts, capacity)`
and `backward(trace, cfg, key)` are compiled array-only entry points. Config,
capacities, fixed birth counts and reward callable are static. Runtime data are
padded arrays and masks. A registered dataclass is only JAX array packaging, not a
spatial tree. The trace carries its configuration and terminal reward.

The retained-ensemble mathematics remains separate from production NTC: disjoint
Maxwell collisions, collision-before-transport, all nominal births in normalization,
full cell populations in collision coefficients, both partner terms in the backward
operator, and fixed source counts. Both affine and ballistic residual reconstruction
are available. CPU host-trace collisionless backward agreement is independently tested.

Reconstruction performs exact tiled top-k selection in position/velocity space and
weighted local affine fits, without SciPy or host callbacks. It does **not** select
collision partners. Tiling bounds scratch storage but work remains proportional to
queries times retained population. Large-budget reconstruction performance and
controlled reconstruction convergence remain acceptance work.

Run the same example through either backend:

```bash
source /home/aakash/jax_dsmc/activate_jax_env.sh
python -m examples.p3_replication --backend jax --particles 256 --cells 4 \
  --steps 2 --retained 8 --source-samples 8 --partners 2 --neighbors 8 --seeds 4 \
  --output examples/p3_replication/results/jit_inlet_smoke.json
```

## Gradient experiments

`examples.gradient_convergence` runs production NTC Couette, reflected shock, and
physical two-sided inlet-shock transient experiments. It retains pathwise sampling
and collision derivatives and adds the existing full trajectory stochastic
correction. The inlet-shock case holds simulator weight fixed, differentiates
Rankine-Hugoniot reservoir controls, and includes rounded initial-population and
source-count scores. These estimators remain subject to membership, majorant and
count-support qualifications above.

Results contain independent seed rows, four paired FD steps, normalized equivalence
intervals and explicit limitations. `--baseline` must be fixed independently of
these seeds; `--seed-start` supports disjoint pilot/confirmation sets. `--seeds`
sets a predetermined run length. Existing artifacts resume only matching setups.
They do not automatically establish steady, mesh, population or regularization
convergence. Conditional/frozen-history FD agreement is insufficient.

Local pilots used 32 seeds and were inconclusive. Two 2048-seed confirmation
runs were interrupted after the user requested that weak local signals be left
for larger compute. Their partial JSON files are historical pilots, not accepted
confirmation runs; they predate the implementation fingerprint and cannot safely
be resumed with this code. Equivalence requires the confidence interval to fit
within a predeclared band, rather than merely contain zero. Inconclusive outcomes
remain inconclusive. Report physical derivative scales and all finite-difference
steps.

## Remaining acceptance work

- Full controlled timestep/mesh/population/regularization and P3 reconstruction
  studies, independent-stream checks, surface objectives and stationary horizons.
- A general correction for parameter-induced hard membership changes; complete
  count-boundary replay in the production adjoint, not just the isolated oracle.
- Complete general-grid adapters and a general moving-surface estimator.
- Source event diagnostics now distinguish requested/inserted/escaped/sampler
  failures/candidate overflow/storage overflow/tracking exhaustion in serial
  runs. Distributed source handling and automatic source-capacity replay remain.
  Collision-attempt replay is the implemented host driver.
- Block checkpoint scans are implemented and tested for value and gradient
  equivalence on a short trajectory. Optimal scheduling, event replay, and
  quantitative GPU memory/scaling studies remain. No claim of constant-horizon
  memory is made.
- External SPARTA comparisons are left to the user; this follow-up validates
  against analytical references and internal independent computations only.

## Verification snapshot (2026-09-14)

The final focused analytical suite (`test_forward_corrections`,
`test_geometry_corrections`, `test_checkpoint_blocks`,
`test_ensemble_inflow_jax`, `test_ntc_reference`, and
`test_gradient_validation`) passed **34 tests in 25.71 s**. The integrated
five-case flux-moment regression first exposed
Gauss–Legendre endpoint quadrature error at 96 nodes; increasing to 256 nodes
passed all five cases and the full focused suite. The analytical reference is an independent
SciPy integral over the inward flux distribution, for speed ratios -10, -3, 0,
3 and 10. It checks the mean squared inlet-normal velocity and its parameter
derivative, not only a single quantile. The focused decomp/state/inlet suite
passed 60 tests after fixing a `CollideState` placeholder issue; another core
focused set passed 96 tests before later changes. The earlier full suite was
518 passed and 6 failed; all six failures were caused by that placeholder issue.
The full suite has **not** yet been rerun after all follow-up changes.
Ruff passes on the touched inlet, NTC, RNG, wall-tracker and analytical-inlet
test files; project-wide lint has not been re-established.

The JAX P3 tiny CPU forward/backward tests pass. On JAX 0.6.2 with the local
RTX 3050, one 16-particle GPU capacity-failure case compiled for more than
eight minutes; a 120 s probe with constant folding disabled also timed out.
This is an unresolved GPU compile-cost issue. Small CPU agreement is not a
large-budget performance result. No paper-scale P3 convergence is established.

Pilot artifacts under `examples/p3_replication/results/` include 32-seed
Couette-speed, reflected-shock-temperature and inlet-shock-temperature runs.
The two `*_confirmatory.json` files are incomplete historical runs and must
not be counted as confirmation. A new runner embeds a SHA-256 fingerprint of
the solver sources and refuses to append rows after code changes.

## Analytical validation handoff

The generated `examples/p3_replication/validation_jobs.json` contains 198
**unlaunched** configurations. It covers each supported Couette, reflected
shock and inlet-shock control across baseline, timestep, mesh, particle count,
horizon and (for Couette) regularization axes, plus separate P3 inlet
reconstruction/sample-budget axes. These commands are a design matrix, not a
claim that every configuration fits available GPU memory or has a sufficient
collision/source capacity. Pilot each configuration, inspect diagnostics,
increase capacities if necessary, then use its own independent pilot objective
mean as the confirmation baseline. Do not reuse a baseline from another mesh,
horizon or parameter. The confirmation seed range starts at 32 and is disjoint
from the pilot range. The JSON row count and `complete` flag must match the
predeclared seed count before applying its equivalence status.

For one job, activate the GPU environment and run the `pilot_argv` array from
the matrix. Substitute that pilot's objective mean for
`PILOT_OBJECTIVE_MEAN` in `confirmation_argv`. The runner writes each seed
atomically and resumes only when configuration, FD steps, JAX version and
implementation SHA-256 match. `--baseline` is a fixed *unscaled objective*
value. After confirmation, require a finite-difference step plateau (the four
steps are 0.1, 0.05, 0.025, 0.0125), then paired agreement across timestep,
mesh, population, regularization and horizon refinements. A confidence interval
that merely contains zero does not establish equivalence. All trajectories
must have zero capacity/majorant/boundary/source failures. Integer rounding
support boundaries require one-sided replay and currently invalidate an
ordinary production-gradient claim.

For wall/shape derivatives, first use the collisionless planar impulse oracle
in `geometry.py` and the multidimensional earliest-hit/surface-tally tests.
Then introduce a physical surface observable accumulated over a time window
in `simulate.run`; compare its gradient with an independently integrated
collisionless moving-wall reference over several wall positions and velocities.
Only after that should a collision-bearing Couette wall or shock reflection
gradient be treated as shape-sensitive evidence. The present moving-wall
tracker and surface tallies do not yet constitute a general moving-mesh adjoint.

## Remaining solver gates before general-grid shape optimization

1. Isolate and correct hard cell-membership sensitivity when a control moves
   particles across a cell face; current pathwise VJP freezes the discrete cell
   lookup. Establish a one- or two-step finite-volume/kinetic analytic oracle
   before broad Monte Carlo testing.
2. Implement the complete weak one-sided count continuation at integer expected
   NTC and source counts in the production gradient. `rounded_weak_derivative`
   is only an isolated oracle; the host diagnostic now rejects this boundary.
3. Replace Cartesian-face assumptions in source placement, hit finding, wall
   normals and cell lookup with a concrete static-connectivity geometry
   interface. Preserve fixed-shape arrays, deterministic tie rules and
   reproducible logical event keys. Grid utilities and particle tracking will
   require the main changes; the collision kernel should continue to receive
   a flat cell occupancy table and physical volumes.
4. Add windowed surface force/heat objectives and a differentiable moving-face
   oracle, then test corners/grazing trajectories and event-budget exhaustion.
5. Measure P3 JIT compile time, runtime and peak memory against the host
   implementation at increasing particle, step, retained and neighbor budgets.
   Resolve the observed small-GPU compile stall before claiming acceleration.
6. Complete a statistically powered finite-horizon analytical/FD validation
   matrix and demonstrate timestep, mesh, particle-count and horizon behavior.
   A transient gradient is not a stationary Couette or shock sensitivity.

These gates mean a general grid and shape optimization still require substantive
geometry/event and gradient work. NTC remains the intended collision scheme,
partners remain uniform random members of each cell, and the estimator design
retains pathwise derivatives wherever a reparameterization is available.
