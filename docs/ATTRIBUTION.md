# Reward attribution in the DiCE surrogate: measured bias, and what it costs

> **Status (cleaned up 2026-10-21).** Measurements made in September 2026 while building the M7
> boundary adjoint (probes in `scratch/m7/`; nothing here describes a behaviour change, the effect
> was always present). The measurements and the explanation still hold and are cited by tests and
> examples (`tests/test_walls.py`, `examples/shock/gradients.py`, `examples/couette/gradients.py`,
> `docs/phase0_verification.py`). What changed is the decision they led to:
>
> * **Superseded 2026-09-15:** the production gradient for any parameter that acts through discrete
>   decisions (collision parameters, accommodation, inlet density and temperature, cell membership)
>   is the particle-carried ensemble adjoint (route b, `jax_dsmc/ensemble_backward.py`), which has no
>   attribution step. Terminal-label scores are biased (this file); full-trajectory scores are
>   unbiased but their error grows as roughly `M^2.8` with the number of steps and `inlet_nrho`
>   is 185 to 1300 times noisier than FD. See [ADJOINT_BRANCH_REVIEW.md](ADJOINT_BRANCH_REVIEW.md)
>   sections 4.8, 4.9, 5.
> * The unbiased global estimator that this file said was missing now exists:
>   `adjoint.trajectory_surrogate` / `score_mode="trajectory"` score every event once with the full
>   scalar reward ([ADJOINT_CORRECTIONS.md](ADJOINT_CORRECTIONS.md)). It is a small-system
>   reference, not a production path.
> * Wall temperature, wall velocity and inlet temperature/stream remain pathwise and are unaffected
>   by anything below, but only as *conditional* derivatives: see REVIEW_2026-09-14 section 2B.

## The claim

`adjoint.surrogate` implements the DiCE estimator

```
grad J~  =  (1/N) Sum_i [ grad phi_i  +  (phi_i - b) grad L_i ]
```

where `L_i` is the score accumulated *by particle i*. The score-function identity gives the exact
score contribution

```
dE[J]/dtheta |score  =  E[ J . Sum_k grad l_k ]  =  (1/N) Sum_k Sum_j E[ phi_j grad l_k ]
```

The surrogate keeps only the terms where `j` is in the attribution set of decision `k`. Since
`E[grad l_k] = 0`, **every dropped term is exactly `Cov(phi_j, grad l_k)`**: free if particle `j`
never felt decision `k`, a **bias** if they interacted, directly or through a chain of collisions.
The per-particle surrogate is unbiased **iff decisions do not propagate between particles.** The
omitted covariances have no universal sign, so the surrogate is not a lower bound on gradient
magnitude in general; it was low in every case measured below.

## Controlled experiment

Cheapest Couette (`D=1`, 10 cells, 50 ppc, N = 500), `J = <|v|^2>`, derivative with respect to the
accommodation coefficient `acc`, a pure Bernoulli threshold (all score, no pathwise part):

| configuration | AD (per-particle) | ensemble FD | verdict |
|---|---|---|---|
| collisions **off** (`d_ref` x 1e-3) | 2.190e4 +- 2.8e2 | 2.183e4 +- 2.9e2 | exact, \|z\| = 0.18 |
| collisions **on** | 5.978e3 +- 2.6e2 | 1.686e4 +- 4.1e2 | low by 2.8x, \|z\| = 22.3 |

With no propagation the estimator is exact; with it, not. That isolates attribution as the cause
(the log-probability is shared by both runs). Controls: pathwise `dJ/dT_wall` and `dJ/dv_wall` match
fixed-key FD to 3.3e-12 and 2.5e-11 (no score: `sample_flux_normal` is reparameterised in `T`);
a fixed-key FD in `acc` is exactly 0.0 at every step, so a pathwise derivative provably cannot see
this parameter and an ensemble FD is the only reference.

## What sets the size: trajectory length

| `M` (steps) | 2 | 5 | 10 | 20 | 40 |
|---|---:|---:|---:|---:|---:|
| FD / AD | 1.13 | 1.32 | 1.57 | 1.95 | **2.82** |

The per-particle estimator is close to exact on short trajectories and degrades steadily as they
lengthen, which is the direction the project is heading.

**Rejected hypothesis: attribution radius.** The collision score already goes to both partners and
the NTC count score to every particle in the cell. Widening the wall score to the whole cell
(`boundary.spread_score_over_cell`, kept as the `score_spread="cell"` option) moved 5.978e3 to
7.080e3 against a truth of 1.686e4: about 10 % of the gap for 7x the variance. Only the global form
recovers it (1.958e4 +- 5.3e3, FD/AD = 0.86): the causal cone is spread thinly over many weakly
correlated particles, so no compact attribution set works.

## Does it affect the shipped collision score?

Yes, much less, consistent with the `M` dependence. On `examples/shock_adjoint`'s ladder (`M = 16`),
`d_ref` (whose pathwise sd is exactly 0): N = 4000 gives FD/AD about 1.1, N = 16 000 gives 1.06
(1.357e13 against 1.43-1.45e13). That is about 6 % at the configuration shipped then. For comparison
`examples/shock_adjoint/VARIANCE.md` section B records FD/AD = 1.137 for `d_ref` at N = 256 000 and
attributes it to O(Delta m^2) truncation; attribution bias is a competing explanation for about half
of it. The `alpha` results are unaffected because that gradient is consistent with zero.

## The exact fix and why it is not cheap

Adding the dropped terms gives identically `per-particle + correction = Sum_k grad l_k . Sum_all j
phi_j`, the global form. There is no cheaper local attribution: for a terminal objective every
decision is an ancestor of essentially every particle's reward. The cost is variance: 20x worse SE
for `acc`, about 100 % relative for `d_ref`.

**Trap.** `Sum_i logh_acc_i` is not `Sum_k l_k`: the accumulator has multiplicities that are
correct for the per-particle estimator and wrong for the global one (wall accommodation 1; pair
collision 2, both partners; NTC attempt count about `npc`, every particle in the cell). A global
estimator built on `Sum_i logh_acc_i` is valid only where every multiplicity is 1 (true for `acc`,
false for every collision parameter); an earlier "global" `d_ref` number measured that way was
retracted. `trajectory_surrogate` avoids this by scoring each event exactly once.

## Relax the Bernoulli instead of scoring it (`acc_delta`)

The problem exists only because `acc` is reached through a score (hard threshold in the
accommodation branch). `acc_delta` (`boundary._diffuse_velocity`) replaces it with a sigmoid blend,
the same move P3 section 4 makes for the crossing event:

```
w      = sigmoid((acc - u) / delta)          # delta = 0 -> the hard threshold, exactly (bitwise)
v_out  = w * v_diffuse + (1 - w) * v_specular
```

Measured on the same case against the unrelaxed ensemble FD 1.686e4 +- 4.1e2:

| `delta` | 0.30 | 0.20 | 0.10 | 0.05 | 0.02 |
|---|---:|---:|---:|---:|---:|
| pathwise `dJ/d(acc)` | 2.668e4 | 2.958e4 | 2.388e4 | 1.876e4 | 1.914e4 |
| ratio to truth | 1.58 | 1.75 | 1.42 | 1.11 | 1.14 |
| SE | 1.5e2 | 2.1e2 | 3.1e2 | 4.6e2 | 7.0e2 |

* `acc` becomes pathwise (fixed-key FD reproduces AD to about 1e-10 at every `delta`) and carries no
  attribution bias. SE is 7-35x below the global form (5.3e3).
* It costs an O(`delta`) bias, a bias-variance knob like `eps`. The forward `<|v|^2>` shift (single
  key, so weak; only "small at 0.02, not small at 0.1 and above" is safe to say) needs the ensemble
  treatment before being quoted. Off by default.
* A relaxed branch must carry no Bernoulli score: the first implementation deposited it anyway
  (REVIEW_2026-09-14 section 3); fixed.
* The same idea for inflow counts (fractional entrant weights) was never built. The ensemble
  adjoint superseded it: `dE[J]/dn` uses the smooth intensity derivative, 0.43x the FD's noise.

## Practical guidance (as of 2026-10-21)

| parameter class | route | status |
|---|---|---|
| wall T, wall velocity, inflow temperature/stream | pathwise | exact conditional derivative (1e-11 vs fixed-key FD); not an expected-gradient proof (REVIEW_2026-09-14 2B) |
| `acc` | ensemble adjoint (production); or relaxed pathwise `acc_delta > 0` | score with `acc_delta = 0` is biased low, 2.8x at `M = 40` |
| collision parameters `diam`, `omega` | ensemble adjoint (validated, branch review 5.7) | terminal-label score about 6 % low at `M = 16` and growing |
| `alpha` (angular law) | pathwise (reparameterised); not in the ensemble adjoint | consistent with zero in every configuration measured |
| inlet density | ensemble adjoint only | no score route works |

**Rule:** a score-carried terminal-label gradient from a long trajectory is biased, and the deficit
grows with the step count. Either use the ensemble adjoint, relax the decision onto the pathwise
route, or check against an ensemble FD at the intended trajectory length.

## The other wall: chaos limits the horizon whatever the attribution

Even for an exact per-key pathwise estimator, the derivative of a single trajectory grows like
`exp(lambda t)` while the quantity it estimates stays O(1). Cheapest Couette, `d u_s/d u_wall` in the
first slab (analytic -0.571):

| steps | 10 | 50 | 100 | 200 | 400 | 800 |
|---|---:|---:|---:|---:|---:|---:|
| mean | -0.314 | -0.520 | -0.536 | -0.529 | -1.056 | -2.486 |
| per-seed spread | 3.7e-02 | 7.4e-02 | 4.3e-01 | 8.1e-01 | 5.4e+00 | 3.4e+02 |

A factor 9254 in spread over 80x the trajectory. The estimator never becomes wrong; it becomes
uninformative, a failure that looks like success if only the mean is reported.

**The shock density ratio fails both ways.** At a 300-step horizon the AD gradient has the **wrong
sign** against an ensemble FD of the same functional:

| `M_1` | ensemble FD | AD | Rankine-Hugoniot |
|---|---:|---:|---:|
| 1.60 | +1.572e-03 +- 2.2e-04 | -2.538e-03 | +4.034e-03 |
| 2.00 | +1.763e-03 +- 1.8e-04 | -1.937e-03 | +3.182e-03 |
| 3.00 | +2.567e-04 +- 1.6e-04 | -4.298e-04 | +1.624e-03 |

Per-seed spread 8.7e-03 against a 3.2e-03 signal: precise and wrong, an attribution failure. The
ratio reaches `u_1` only through discrete decisions of opposite sign (the initial-condition `alive`
Bernoulli raises it, the injection count lowers it during the transient), and under-counting two
opposing terms by different factors flips the sum. Chaos sits behind it: reaching Rankine-Hugoniot
compression needs about 1000 steps (forward error -38.7 % at 300, -2.6 % at 1000), by when the
pathwise spread is 4.0e6 (1.0e12 at 1500 steps). Fixing attribution leaves the horizon broken;
shortening the horizon leaves attribution broken. The temperature ratio in the same run validates at
300 steps because it reaches `u_1` pathwise through the prescribed inflow temperature. The question
to ask is "does the quantity reach the parameter pathwise, and does its equilibration horizon sit
inside the chaos horizon". Both must hold.

*Status:* the ensemble adjoint passes this acceptance case with the correct sign
(`d(rho_2/rho_1)/dM_1`: 2.74 +- 0.66 against CRN FD 2.54 +- 0.17, z = 0.3, ADJOINT_BRANCH_REVIEW
section 6), but its per-seed sd is 4x the FD's at 300 steps: the horizon problem is reduced, not removed.

## Particle count fixes the solve, not the gradient

Sampling variance averages over particles like the forward observable; the chaotic part is set by
the Lyapunov instability and does not fall with `N`. Couette sweep, 512 seeds, relative per-seed
spread exponents:

| | forward `u_s` | `du_s/du_wall` | forward `T` | `dT/dT_wall` |
|---|---|---|---|---|
| 25 steps | `N^-0.498` | `N^-0.513` | | `N^-0.521` |
| 200 steps | `N^-0.483` | `N^+0.002` | `N^-0.506` | `N^-0.099` |

At 200 steps the gradients are flat (115 % relative spread at N = 180 and at N = 2880). Particles buy
gradient accuracy only up to the chaos horizon; past it only ensemble size helps, at `M^-1/2`. A
precise short-horizon column is not converged: `du_s/du_wall` reads -0.46 at 25 steps against -0.571
because 25 steps is far short of the 290-step momentum-diffusion time. Low variance is not
correctness. (The flat-N result is evidence for this protocol, not a theorem.)
