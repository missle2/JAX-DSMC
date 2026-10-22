# Adjoint corrections and P3 reference implementations

P3 source: [arXiv 2603.20946v2](../papers_reference/arXiv-2603.20946v2/main.tex), especially
`MC_collision_adjoint_inflow`, `sampled_left_score_gradient_inflow`, and numerical Tests 1, 3 and 4.

> **Status (cleaned up 2026-10-21).** Record of the 16 commits on `fix/paper-adjoints`
> (`bafec3d..d5f6856`, 2026-09-14), the first response to
> [REVIEW_2026-09-14.md](REVIEW_2026-09-14.md). The branch is merged into `main`. The code described
> here exists and is tested, but the conclusions about which estimator to *use* were overtaken the
> next day:
>
> * **Production gradient (decided 2026-09-15):** the particle-carried ensemble adjoint, route (b)
>   (`simulate.run(capture=True)`, `jax_dsmc/ensemble_backward.py`, `ensemble_device.py`;
>   [ADJOINT_BRANCH_REVIEW.md](ADJOINT_BRANCH_REVIEW.md) section 5). Score-function routes were measured
>   dead (biased, or noise growing as `M^2.8`; [ATTRIBUTION.md](ATTRIBUTION.md)).
> * What this file still defines: the wall estimator (`wall_adjoint`, conditional-time tangent), the
>   trajectory-likelihood *reference* estimator, and the P3 transcriptions (`ensemble_inflow*.py`,
>   `paper_walls.py`) used as verification oracles for route (b).
> * Where this file and the branch review disagree, the review is later. Older PSEUDOCODE / PLAN /
>   HANDOFF / ATTRIBUTION estimator-validity claims are superseded by it.

## Estimators on the branch

### P2 overlap and complete trajectory likelihood

* The P2 hand-written recursion tests remain: they establish agreement in that oracle's setup, not an
  independent expected-gradient identity for arbitrary partner changes. `adjoint.surrogate` keeps
  terminal-label attribution for those comparisons.
* `adjoint.trajectory_surrogate(J, L)` scores each event log probability **once** with the full
  scalar reward. Exact enumeration over all 16 acceptance histories of a four-particle,
  two-matching VSS experiment checks its expected derivative against an independently differentiated
  expectation and a central difference
  (`tests/test_stochastic_reference.py::test_global_likelihood_matches_exact_enumeration_after_changing_partners`).
* `score_mode="trajectory"` is accepted by the NTC collision driver and `simulate.run`: pair
  acceptance/rejection scores count once per attempt, rounded-count scores once per cell, boundary
  scores once per original particle (including ones that leave). The scalar prefix is stored in every
  slot for layout compatibility, so it must start at zero or continue the same mode. For a manual
  run use `trajectory_surrogate(objective(...), state.logh_acc[0])`, not `surrogate` with that
  repeated scalar. Windowed rewards use the causal prefix.
* Defaults: `terminal_objective` / `windowed_objective` use `"trajectory"`; `simulate.run`,
  `dice_reward` and the low-level collision driver stay `"paper"`; the P2 example pins `"paper"`
  explicitly (`examples/shock_adjoint.evaluate(..., score_mode="paper")`) after a silent default
  change was caught (branch review 1.2, B9).
* `baseline="mean"` (same-realization) is **rejected** by `adjoint._resolve_baseline`: stopping its
  derivative does not make it independent of the scored event. A constant or independently estimated
  baseline is supported. The historical "unbiased 84x variance reduction from the same-trajectory
  mean" is withdrawn. A full-trajectory estimator can have much higher variance than terminal-label
  attribution; it is a reference, not a universal improvement.

### P3 walls (`jax_dsmc/wall_adjoint.py`)

`boundary.apply_faces` dispatches to `wall_adjoint.advance` whenever any face is DIFFUSE or an
embedded surface is present (any `D` in 1, 2, 3; the exact fold remains for boxes with only
periodic/specular/outflow faces). It

1. processes the first physical hit chronologically (specular, periodic, diffuse, absorbing),
   reporting exhausted capacity instead of treating a clamp as resolved (the later lane tracker
   deletes and counts stuck particles; `Diagnostics.surface_stuck`);
2. draws a Gaussian travel time, determines the realised event history and scores the probability
   of its admissible time interval;
3. differentiates the travel time **conditional on that interval**: with bounds `a, b`,
   `F_a = Phi((a-dt)/eps)`, `F_b` likewise, `tau = dt + eps * Phi^-1(F_a + u (F_b - F_a))` holding the
   recovered conditional uniform `u` fixed; the interval score is `log(F_b - F_a)`. The forward
   sample is unchanged. Scoring a branch while differentiating the raw `tau` misses a term for
   position observables: with it, the one-particle position derivative reads 38 % high
   (+0.530 versus exact +0.383; branch review 4.2). Velocity observables are unaffected;
4. keeps the Bernoulli score for hard accommodation and gives relaxed accommodation (`acc_delta > 0`)
   none, avoiding double counting a pathwise blend.

The conditional-time derivative is an extension of the paper's displayed recurrence, not a
transcription. Quantiles are clipped at 1e-15 in extreme tails; use `eps << dt`; negative sampled
time is clamped to zero. Independent one-particle Gaussian quadrature checks constant and position
rewards.

`WallParams.temp` takes a scalar or three component temperatures per face (Rayleigh normal, Gaussian
tangential). Physical API temperatures are kelvin; the paper driver uses `mass = k_B`.
`paper_walls.py` is a JAX reference driver (disjoint Maxwell pairs, local-density collision
frequency, collision-before-transport, P3 heat/Couette objectives, terminal-label or trajectory
scoring). Hard cell memberships and rounded schedules stay frozen as in P3. `Config(localize=True)` /
`--localize` reproduces P3's three-sigma proximity selection, which is frozen under AD and has no
weak-derivative guarantee at its interface; the default regularises every particle.

### P3 inflow (host reference and JIT version)

> **Superseded 2026-09-15** (branch review section 5): the production inlet gradient is route (b)
> on the *actual* solver, validated against FD for reservoir temperature and density. The modules
> below stay as the reference transcription of P3's own Maxwell model and as the verification oracle
> for route (b) (`jax_dsmc/ensemble_particle.py`). The paper's Test-3 value 0.1517 is *not* a target
> for them (branch review 4.12).

`ensemble_inflow` (NumPy/SciPy host) and `ensemble_inflow_jax` (JIT, array-only) implement the local
Maxwell model of P3 section 5:

* full forward population with exact cell counts, unbiased rounded disjoint pairs, collision then
  absorbing transport then source injection; **fixed nominal injection counts** in both adjoint and
  FD (`nominal_counts` follows Test 3's dx prefactor, not a physical-area formula);
* anisotropic zero-drift half-Maxwellian flux (Rayleigh normal, Gaussian tangent), temperature score
  `v_l^2/(2 T_l^2) - c_l/T_l`, `c = (1, 1/2, 1/2)`; fixed normalisation `N0 + M (N_L + N_R)` including
  escaped newborns, whose source samples are kept with value zero;
* uniform retained states per cell, exact full-population density in the backward coefficient,
  source expansion weights `N_source/m_source`; one scalar backward field gives all six temperature
  components; all four collision terms (target gain + partner gain - target loss - partner loss;
  omitting the partner terms omits the rate response);
* weighted local-affine reconstruction in `(x, v1, v2, v3)` (batched kNN; coordinates scaled by `dx`
  and the initial std; 1e-12 slope ridge); `reconstruction="ballistic"` subtracts the analytic
  collisionless terminal field, fits the rest and adds it back (an explicit numerical extension);
* JIT version: `forward(cfg, temperatures, key, injection_counts, capacity)` and `backward(trace,
  cfg, key)` with static config/capacities/reward and padded arrays and masks; exact tiled top-k
  selection (work still scales with queries x retained population; collision partners are not
  selected by it). `partner_indices` factored out; self-pairs excluded (branch review B5).

```python
import numpy as np
from jax_dsmc import ensemble_inflow as p3
cfg = p3.Config(reconstruction="ballistic")
trace = p3.forward(cfg, np.full((2, 3), 4.0), seed=0)
gradient = p3.backward(trace)  # shape (2, 3): all six temperature components
```

The old `inflow.inject` remains the forward API for general drifting reservoirs; trajectory-scored
`simulate.run(..., res=...)` rejects an unsupported combination. Not implemented here: drift,
source-count sensitivities, mixtures, shapes.

## Validation limits (what to believe)

* These are *scaled* checks. At N = 2000 the Couette wall derivative was 3.089 +- 0.111 (adjoint)
  versus deterministic FD 3.010 +- 0.237, and the small inlet run gave 0.155 / 0.147 against P3's
  0.1517 / 0.1516 with FD errors far too large to discriminate. At scale (N = 32 000) the Couette
  derivative is 3.032 +- 0.010 against the paper's 3.028. The inlet value depends on the
  reconstruction: affine sits about 20 to 25 % above ballistic on identical trajectories (a Jensen bias of
  the interpolant, +11 %/+36 % at 128 retained states) and only the retained count moves ballistic
  toward the paper's value. Details and the explanation that the transcribed *forward* differs from
  the paper's at the inlet: branch review 4.12.
* Wall comparisons report regularised and original deterministic FD (the paper's benchmark uses the
  latter). Inlet FD holds counts fixed and shares the initial sample.
* A conditional or frozen-history FD agreement is insufficient; the arbiters used since are paired
  ensemble FD with common random numbers at several steps (`examples/gradient_bench`).
* Several pilot JSON files under `examples/p3_replication/results/` (the two `*_confirmatory.json`)
  are incomplete historical runs and must not be counted as confirmation; the
  198-job `validation_jobs.json` design matrix was never launched (superseded by the targeted
  studies of the branch review).
* The couette tests use capacity 256 attempts, check zero truncation, and the example accumulates
  diagnostics across warm-up and sampling and refuses unresolved runs.
* **GPU compile stall.** On JAX 0.6.2 / RTX 3050 the tiny JAX P3 inlet reverse pass compiled for
  more than 2 to 8 minutes (a 64-particle, two-cell six-temperature check exceeded 120 s; a
  16-particle source case exceeded 180 s; the 256-particle, four-cell case ran in 10.9 s). The
  16-particle capacity-failure case is skipped on that GPU/JAX combination
  (`tests/test_ensemble_inflow_jax.py::test_capacity_failure_is_explicit`); the 64-particle overflow
  is still tested. A measured compiler limitation, not evidence of numerical agreement; not
  re-measured on later JAX versions.

## Running the tests and artifacts

Activate `/home/aakash/jax_dsmc/activate_jax_env.sh`. Pytest defaults to CPU; `--gpu` requests an
accelerator; `--no-jax-cache` avoids stale executables. Run JAX jobs through
`/home/aakash/jax_dsmc/run_locked.sh` on the shared laptop (GPU: one job; CPU: two). The relevant
files are `tests/test_stochastic_reference.py`, `test_paper_walls.py`, `test_trajectory_scores.py`,
`test_walls.py`, `test_ensemble_inflow*.py`. Commands for the P3 reference runs are in
[`examples/p3_replication/README.md`](../examples/p3_replication/README.md).

## Remaining barriers (status 2026-10-21)

| Barrier named here | Status |
|---|---|
| Hard cell/interface membership sensitivity | Handled by construction in the ensemble adjoint (branch review 5, 6); a cut-volume term exists for shapes. |
| NTC rounded-count score at integer means | Not fixed in score estimators; avoided by route (b)'s intensity derivative. |
| Moving geometry, wall/source measure derivatives | Built as the mesh/level-set shape adjoint (branch review 6, 7); restricted to normal displacements of curved surfaces. |
| Multi-dimensional simultaneous hits and corners | The chronological tracker is validated in D = 1, 2, 3 (branch review 4.1); the lane tracker resolves unbounded bounces (7.10). Corner/grazing derivatives are handled by the grazing cut-off and silhouette term (7.8), not exact. |
| P3 reconstruction convergence at scale | Done at affordable budgets (4.12); paper-scale not reproduced and not a goal. |
| Distributed adjoint | Not built. |
