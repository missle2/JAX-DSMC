# Adjoint corrections and P3 reference implementations

P3 source: [arXiv 2603.20946v2](../papers_reference/arXiv-2603.20946v2/main.tex), especially `MC_collision_adjoint_inflow`, `sampled_left_score_gradient_inflow`, and numerical Tests 1, 3 and 4.

This document supersedes adjoint-validity claims in the historical PSEUDOCODE,
PLAN, HANDOFF and ATTRIBUTION documents where they conflict. The review is
[REVIEW_2026-09-14.md](REVIEW_2026-09-14.md); this file describes implemented changes and their limits.

## Isolation and review

Worktree: `/home/aakash/jax_dsmc/JAX-DSMC-adjoint`, branch `fix/paper-adjoints`.
The original development directory was left unchanged. Commit **bafec3d** is a
snapshot of its tracked and untracked development files, including the reference
papers. Review this work against that snapshot, not against the older original
HEAD (which would mix these corrections with pre-existing development changes):

```sh
git diff bafec3d..HEAD
```

Initial failing regressions were committed before the implementations:

- `11f099d`: independent stochastic expectation and wall tests (five failures).
- `0084997`: P3 ensemble source/collision/reconstruction identities (missing module).
- `5152424`: production trajectory likelihood bookkeeping (unsupported mode).

Additional tests extend these with collisionless quadrature, escaped newborns,
chronological mixed walls, anisotropic temperatures and conservation.

## Changed estimators

### P2 overlap and complete trajectory likelihood

The existing P2 hand-written recursion tests remain. They establish agreement in
that oracle's setup, not an independent expected-gradient identity for arbitrary
future partner changes. No discrepancy in that exact paper-recursion overlap is
claimed here.

`adjoint.surrogate` retains terminal-label score attribution for those comparisons.
The new `trajectory_surrogate(J,L)` uses the complete scalar reward and each event
log probability **once**. Exact enumeration over all 16 acceptance histories of a
four-particle, two-matching VSS experiment checks its expected derivative against
an independently differentiated expectation and a central difference. Scattering
and matchings are conditioned on in this small experiment; it is not a full
ensemble reproduction of P2's tables.

The production NTC collision driver and `simulate.run` now accept
`score_mode="trajectory"`. Pair acceptance/rejection scores contribute once per
attempt, and rounded-count scores once per cell. Boundary scores contribute once
per original particle, including particles that leave. For compatibility with the
particle-state layout the same scalar prefix is stored in every slot; it must
start at zero or be continued from the same scoring mode. The prefix is scalar inside
the attempt scan and broadcast only on return, avoiding an N-vector update for
every attempt. The public state still carries an N-vector for compatibility.

`terminal_objective` and `windowed_objective` default to trajectory scoring.
`simulate.run`, `dice_reward` and the low-level collision driver retain the
`"paper"` default so existing low-level calls do not silently change accumulator
meaning. The P2 example explicitly selects `"paper"`. For a manual trajectory
run, use `trajectory_surrogate(objective(...), state.logh_acc[0])`, not
`surrogate` with that repeated scalar. Windowed rewards use the causal prefix.

A same-realization `baseline="mean"` is rejected. Stopping its derivative does
not make it statistically independent of the scored event. A fixed constant or
independently estimated baseline is supported. Historical claims of unbiased
84x variance reduction from the same-trajectory mean are withdrawn. A full
trajectory estimator can have substantially higher variance than terminal-label
attribution; it is not a universal performance improvement.

### P3 walls

The 1D diffuse/mixed face path now dispatches to `wall_adjoint.advance`:

1. Process the first physical hit, including specular, periodic and absorbing
   faces, in chronological order. Resolve up to four events and report exhausted
   capacity instead of treating a clamp as a resolved trajectory.
2. Draw Gaussian travel time, determine the realized event history, and score
   the probability of its admissible time interval.
3. Differentiate travel time **conditional on that interval**. For bounds a,b,
   write F_a=Phi((a-dt)/epsilon), F_b similarly and
   `tau=dt+epsilon*Phi_inverse(F_a+u*(F_b-F_a))`, holding the recovered conditional
   uniform u fixed. The forward sample is preserved exactly. The interval score
   is `log(F_b-F_a)`. Conditional pathwise derivative plus interval score gives
   the weak derivative of this regularized transition. Scoring a branch while
   retaining the unconditional time derivative alone misses a term for position
   observables.
4. Hard accommodation retains its Bernoulli score. Relaxed accommodation has
   zero hard-branch score, avoiding double counting a pathwise blend.

The conditional-time derivative is an explicit correction/extension of the paper
transcription, not a claim that its displayed wall recurrence already contained
this implementation. Independent one-particle Gaussian quadrature checks both
constant and position rewards. Quantiles are clipped at 1e-15 in extreme tails;
this is a numerical tail approximation. Use epsilon much smaller than dt;
negative sampled time is clamped to zero as before.

`WallParams.temp` now accepts either scalar temperature per face or three
component temperatures per face. The diffuse sampler uses the normal component
for Rayleigh sampling and the tangential components for Gaussian sampling.
Physical API temperatures are kelvin; the paper driver uses mass=k_B so its
numbers are nondimensional velocity variances.

`paper_walls` supplies a JAX reference driver with disjoint Maxwell pairs,
local-density collision frequency, collision-before-transport splitting, and
P3 heat/Couette objectives. It offers both terminal-label and trajectory scoring.
Hard cell memberships and rounded collision schedules remain frozen, as in P3's
wall formulation. Default regularization applies to all particles, avoiding an
extra hard proximity interface. `Config(localize=True)` / example `--localize`
reproduce P3's three-sigma proximity selection; that selection is frozen under
AD and does not carry a general weak-derivative guarantee at its interface.

The corrected chronological/conditional treatment is **1D only**. The older
2D/3D face implementation is still available but has not acquired the same
mathematical validation. Cell score spreading is a compatibility heuristic,
not a substitute for complete trajectory likelihood.

### P3 inflow

`ensemble_inflow` replaces trajectory AD as the supported P3 inlet-gradient
method. It implements the local Maxwell model in the paper's inflow section:

- Full forward population, exact cell counts, unbiased rounded disjoint pairs,
  collision then absorbing transport then source injection.
- Fixed nominal injection counts in both adjoint and finite differences.
  `nominal_counts` uses the dx-prefactor of numerical Test 3; it is not a generic
  physical-area inlet formula.
- Anisotropic zero-drift half-Maxwellian flux: Rayleigh normal, Gaussian tangent.
  Its temperature score is `v_l^2/(2*T_l^2)-c_l/T_l`, with c=(1,1/2,1/2).
- Fixed normalization `N0+M*(NL+NR)`, including escaped newborns. Newborn source
  samples include immediate exits, whose adjoint value is zero.
- Uniform retained states per cell, exact full-population density in the
  backward coefficient, retained source expansion weights N_source/m_source.
- One scalar backward field. Pull through absorbing transport and apply all
  four collision terms: target gain + partner gain - target loss - partner loss.
  Omitting the partner terms omits the distribution/rate response.
- Weighted local affine reconstruction in (x,v1,v2,v3), batched nearest-neighbor
  queries, configurable neighbor/partner/reservoir budgets. Spatial coordinates
  are scaled by dx, velocities by initial standard deviation. Fits use a 1e-12
  slope ridge. These numerical reconstruction choices are explicit because the
  paper does not fully prescribe them.
- Optional `reconstruction="ballistic"` subtracts the analytically known
  collisionless terminal field, fits the remaining field, and adds the analytic
  part back at queries. This is an explicit numerical extension to reduce
  interpolation error, not a change to the four-term collision adjoint. The
  default `"affine"` follows the direct field reconstruction described in P3.

This is a NumPy/SciPy host reference, not a JIT/GPU implementation or a custom VJP
of the JAX NTC solver. It is a distributional (optimize-then-discretize) adjoint;
finite-population and reconstruction errors are expected. The collisionless limit
uses exact characteristics, independently checked by Gauss-Laguerre/Legendre
quadrature of the flux objective. Reservoir sizes do not reduce the physical
collision rate. Invalid disjoint-pair time steps raise an error.

The old `inflow.inject` remains a forward API for general drifting reservoirs.
It is not rebranded as the P3 gradient implementation. Trajectory-scored
`simulate.run(..., res=...)` rejects this unsupported combination and directs
callers to the ensemble solver. P3 currently supports temperature derivatives
for the fixed-count, zero-drift Maxwell source; arbitrary drift, source-count
sensitivities, species mixtures and shapes are not implemented by this reference.

## Reproduction and evidence

Run from the isolated worktree. NumPy/SciPy and the existing JAX environment are
required. Activate `/home/aakash/jax_dsmc/activate_jax_env.sh` for the correct
JAX environment. Pytest still defaults to CPU: use `--gpu` explicitly for an
accelerator check. The sandbox required external GPU access in this session.
Artifacts record their backend; no GPU throughput claim is made.

```sh
python3 -m pytest -q --no-jax-cache
python3 -m pytest --gpu --no-jax-cache -q tests/test_stochastic_reference.py tests/test_paper_walls.py tests/test_trajectory_scores.py tests/test_walls.py
OPENBLAS_NUM_THREADS=1 python3 -m examples.p3_replication \
  --particles 1000 --retained 64 --source-samples 499 --partners 4 \
  --seeds 4 --fd-step .1 --output examples/p3_replication/results/scaled_inflow.json
python3 -m examples.p3_replication.walls --particles 2000 --seeds 8 \
  --output examples/p3_replication/results/scaled_couette.json
python3 -m examples.p3_replication.walls --particles 2000 --seeds 8 --localize \
  --output examples/p3_replication/results/localized_couette.json
```

JSON artifacts include settings, means, standard errors, fixed counts and
normalization. The inlet runner shares the initial particle sample across seeds,
and holds counts fixed through FD perturbations. Collision streams are reset
per time step; changed occupancies can still decorrelate the collision draws.
Wall comparisons report both regularized and original deterministic FD, as the
paper's benchmark uses deterministic FD.

At N=2000, eight Couette seeds with global time regularization gave adjoint
3.0893 +/- 0.1108 and deterministic FD 3.0097 +/- 0.2374 (standard errors),
compatible with P3's reported 3.0283 / 3.0011 at much greater resolution. The
small inlet run gave normal-temperature adjoints 0.1552 +/- 0.0125 and
0.1471 +/- 0.0071, near P3's 0.15172 and 0.15157. Its noisy finite differences
are insufficient for a stringent validation. These are scaled checks, **not**
a reproduction of the paper's large-population confidence intervals.

The localized Couette run gives 3.0345 +/- 0.0780 against deterministic FD
3.0097 +/- 0.2374. The GPU heat-conduction run gives left/right derivatives
0.6980 +/- 0.1720 and 0.8295 +/- 0.2007, versus deterministic FD
0.7218 +/- 0.1720 and 0.6038 +/- 0.2351. These short runs establish compatibility
at their available precision, not the much smaller error bars of the paper.

Inflow reconstruction dependence is material:

| Initial N | Retained/cell | Partners | Reconstruction | Left gradient | Right gradient |
| ---: | ---: | ---: | --- | ---: | ---: |
| 1000 | 64 | 4 | affine | 0.1552 | 0.1471 |
| 1000 | 512 | 8 | affine | 0.1814 | 0.1679 |
| 1000 | 128 | 8 | ballistic residual | 0.1404 | 0.1346 |
| 4000 | 512 | 8 | ballistic residual | 0.1460 | 0.1405 |

These change multiple budgets and are not a controlled one-variable convergence
study. They show why the apparent match of the smallest affine run to the paper
must not be overinterpreted. The last row has standard errors 0.00425 / 0.00458;
its fixed-count FD estimates are 0.1787 +/- 0.0334 and 0.1837 +/- 0.0486.
Both discrepancies are below one *paired* standard error. Larger populations,
reservoirs and seed ensembles remain necessary to establish paper-level inlet
precision. No tolerance was tuned to force these noisy examples to pass.

```python
import numpy as np
from jax_dsmc import ensemble_inflow as p3
cfg = p3.Config(reconstruction="ballistic")
trace = p3.forward(cfg, np.full((2, 3), 4.0), seed=0)
gradient = p3.backward(trace)  # shape (2, 3): all six temperature components
```

A custom terminal reward supplied to `forward` is retained for `backward`.
The stored initial sample is fixed across collision/inflow seeds, as in P3.
Terminal-label oracles, exact-history expectation tests, and distributional
quadrature test different statements; none replaces the others.

## Remaining barriers and costs

The new inlet method bounds stored backward data by retained states rather than
all forward particles: O(M*C*R + M*m_source), with full forward population still
resident. Local-affine queries are batched; no all-query/all-population distance
matrix is built. One scalar solve yields all six temperature components. The
host implementation and stored time frames are further acceleration targets.
The chronological JAX wall loop uses a scan-compatible bounded loop rather than
unrolling every bounce; reverse steps are rematerialized. Its own JIT entry point
keeps eager calls from generating a fresh loop compilation as particle values
change, checked by a compilation-cache reuse regression. Inside an existing JAX
trace, it uses the same implementation directly without another JIT boundary.
Fresh suite checks use
`--no-jax-cache` to avoid relying on previously cached executables.

These changes do not establish general shape derivatives. Hard cell/interface
membership, moving geometry support, wall/source measure derivatives and
multi-dimensional simultaneous hits still need explicit treatment. NTC rounded
count likelihood is valid only inside a fixed noninteger support interval; at
an integer expected count its support changes and the usual score is not a
valid derivative. Neither the global score nor the P3 oracle fixes that.
P3's distributional collision linearization is the appropriate direction for
mesh/source sensitivity, but extending its scalar reconstruction to general
meshes and boundary geometry needs new convergence and weak-derivative tests.

The existing Couette verification also exposed truncated collision attempts and
diagnostics that covered only the last warm-up block. The test uses adequate
attempt capacity (default increased to 256), explicitly checks zero truncation,
and the example accumulates diagnostics over warm-up and sampling and refuses
to report unresolved runs. Tests sample every 20 rather than 200 steps, yielding
30 rather than 3 snapshots with the same 600-step sampling horizon and unchanged
stress tolerances. Warm-up is 600 steps, aligned with block boundaries. Its
short stress comparison is
assessed separately from a steady-state viscosity validation.


### GPU compilation limitation

On this machine (RTX 3050 Laptop GPU, JAX 0.6.2), the tiny 64-particle/two-cell
six-temperature reverse-mode check exceeded a 120-second compilation budget
(and an earlier attempt was still compiling after eight minutes). The same
six-temperature check at 256 particles/four cells compiled and executed in
10.9 seconds with finite gradients and zero unresolved events. Both sizes remain
in CPU tests; only the tiny case is explicitly skipped for this GPU family and
JAX version. Other GPU/JAX combinations still exercise it. This is a measured
compiler-performance limitation, not evidence of numerical agreement for the
skipped accelerator case. The representative 2000-particle GPU heat example also
runs with the final code, and its artifact records `backend="gpu"`.

## Final validation record

The current tree collects **507 CPU tests**. The full CPU sweep passed 505 tests
and exposed one issue in the new cache-reuse fixture: its initial array used
JAX weak typing, while the returned array had an explicit dtype, legitimately
requiring another compilation. The fixture now uses explicit float64 throughout.
The final wall/score/inlet regression group passed **22 tests**; the expanded
P3 wall group passed **3 tests**; and the final Couette group passed **8 tests**.
Together these cover the corrected fixture, the additional particle-size case,
and the code changed after the full sweep. Validation is reported as a full sweep followed by targeted reruns.

Ruff checks on the changed Python files and `git diff --check` pass. The
reference result files are small JSON artifacts with their exact numerical
budgets; they retain settings and uncertainty estimates for review.

The broader GPU wall sweep reached the long equilibrium tests without numerical
failures but exceeded a five-minute command budget. Its partial progress is not
reported as a completed GPU suite; the complete CPU sweep covers those tests.
The core accelerator regressions are run separately from that long sweep.

The final core GPU run completed with **13 passed, 1 skipped in 67.85 s**:

```sh
source /home/aakash/jax_dsmc/activate_jax_env.sh
python -m pytest --gpu --no-jax-cache -q \
  tests/test_stochastic_reference.py tests/test_paper_walls.py \
  tests/test_trajectory_scores.py
```

The skip is the explicitly documented JAX 0.6.2 / RTX 3050 tiny-grid case.
The original development files were compared byte-for-byte with the baseline
snapshot after implementation; no differences were found.
