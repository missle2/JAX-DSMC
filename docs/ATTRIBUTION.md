# Reward attribution in the DiCE surrogate — measured bias, and what it costs

**Read this before trusting any score-carried gradient from a long run.**

Everything here was measured in September 2026 while building the M7 boundary adjoint. The
probes are in `scratch/m7/`. Nothing in this document describes a change in behaviour: the
effect it measures was always present and simply had not been looked for.

## The claim

`adjoint.surrogate` implements the DiCE estimator

```
grad J~  =  (1/N) Sum_i [ grad phi_i  +  (phi_i - b) grad L_i ]
```

where `L_i` is the score accumulated *by particle i*. The score-function identity says the
exact score contribution is

```
dE[J]/dtheta |score  =  E[ J . Sum_k grad l_k ]  =  (1/N) Sum_k Sum_j E[ phi_j grad l_k ]
```

The surrogate keeps only the terms where `j` is in the attribution set of decision `k`. Since
`E[grad l_k] = 0`, **every dropped term is exactly `Cov(phi_j, grad l_k)`**. So:

* if particle `j` never felt decision `k`, the covariance is zero and dropping it is free;
* if `j` and the decision-maker have interacted -- directly, or through a chain of
  collisions -- the covariance is non-zero and dropping it is a **bias**.

The per-particle surrogate is unbiased **iff decisions do not propagate between particles.**

## The controlled experiment

Cheapest Couette (`D=1`, 10 cells, 50 ppc, N = 500), `J = <|v|^2>`, differentiated with
respect to the accommodation coefficient `acc` -- a pure Bernoulli threshold, so its whole
sensitivity is score and none of it is pathwise.

| configuration | AD (per-particle) | ensemble FD | verdict |
|---|---|---|---|
| collisions **off** (`d_ref` x 1e-3) | 2.190e4 +- 2.8e2 | 2.183e4 +- 2.9e2 | **\|z\| = 0.18 — exact** |
| collisions **on** | 5.978e3 +- 2.6e2 | 1.686e4 +- 4.1e2 | \|z\| = 22.3, low by 2.8x |

With no propagation the estimator is exact; with propagation it is not. That isolates
**attribution** as the cause, and rules out the log-probability itself, which is shared by
both runs.

Two further controls:

* **The pathwise gradients are unaffected.** `dJ/dT_wall` and `dJ/dv_wall` agree with a
  fixed-key finite difference to **3.3e-12** and **2.5e-11**. They carry no score
  (`sample_flux_normal` is reparameterised in `T`, and `vwall` is added afterwards), so
  nothing in this document touches them.
* **A fixed-key FD in `acc` is exactly 0.0**, at every step size. That is the control that
  says a pathwise derivative *provably cannot* see this parameter, so the score is the only
  route and there is nothing to compare it against except an ensemble.

## What sets the size of the bias: trajectory length

The bias grows with the number of steps the decision has had to propagate, and with nothing
else that was tested:

| `M` | per-particle AD | ensemble FD | FD/AD |
|---|---|---|---|
| 2 | 1.046e3 | 1.178e3 | **1.13** |
| 5 | 2.349e3 | 3.100e3 | 1.32 |
| 10 | 3.977e3 | 6.227e3 | 1.57 |
| 20 | 5.562e3 | 1.084e4 | 1.95 |
| 40 | 5.978e3 | 1.686e4 | **2.82** |

**This is the practically important result.** It says the per-particle estimator is close to
exact on short trajectories and degrades steadily as they lengthen -- which is the direction
this project is heading.

### A hypothesis that was tested and rejected

`collide` already attributes its scores widely: the pair score goes to *both* partners
(`ntc.py:255`) and the NTC attempt-count score to *every particle in the cell*
(`ntc.py:368`). The natural guess was that attribution **radius** drives the bias, and that
matching that convention for the wall score would fix it.

It does not. Widening the wall score from one particle to the whole cell
(`boundary.spread_score_over_cell`) moved the estimate from 5.978e3 to 7.080e3 against a
ground truth of 1.686e4 -- **~10 % of the gap closed for 7x the variance**:

| attribution | `dJ/d(acc)` | +- SE | FD/AD |
|---|---|---|---|
| single particle | 5.978e3 | 2.6e2 | 2.82 |
| whole cell | 7.080e3 | 1.9e3 | 2.38 |
| global | 1.958e4 | 5.3e3 | **0.86** |

So the causal cone is **not** concentrated near the bouncer. It is spread thinly over many
particles, each weakly correlated, and only the sum is large. That is why no compact
attribution set recovers it, and why the correction has to be global.

## Does this affect the shipped collision score?

Yes, but far less, and the `M`-dependence above explains why. Audited on
`examples/shock_adjoint`'s own ladder (`M = 16`), differentiating `d_ref` -- the purest test
available, since the split-variance study measures its `sd(pathwise)` as *exactly* 0:

| configuration | per-particle AD | ensemble FD | ratio |
|---|---|---|---|
| N = 4 000, M = 16 | 1.333e13 | 1.44-1.72e13 (trends with `h`) | ~1.1 |
| N = 16 000, M = 16 | 1.357e13 | 1.43-1.45e13 (flat in `h`) | **1.06** |

**~6 % at the shipped configuration.** For comparison, `VARIANCE.md` section B records
`FD/AD = 1.137` for `d_ref` at `N = 256 000` and attributes it entirely to `O(Delta m^2)`
truncation. Attribution bias is now a **competing explanation for roughly half of it**, and
that section should be re-read with both in mind rather than rewritten on this evidence
alone -- the audit above is at `N <= 16 000`.

`alpha`'s published results are unaffected in practice for an uninteresting reason: its
gradient is consistent with **zero** at every configuration measured, and a few per cent of
a zero is a zero.

## The exact fix, and why it is not cheap

Adding back the dropped terms gives, identically,

```
per-particle  +  correction  =  Sum_k grad l_k . Sum_all j phi_j   ==  the global form
```

So there is no cheaper local attribution to extend to: for a terminal objective, every
decision is an ancestor of essentially every particle's reward. The cost is variance --
measured **20x** worse SE for `acc`, and ~100 % relative for `d_ref`.

### A trap in implementing it

`Sum_i logh_acc_i` is **not** `Sum_k l_k`. The accumulator carries multiplicities that are
*correct* for the per-particle estimator and wrong for the global one:

| decision | added to | multiplicity in `Sum_i L_i` |
|---|---|---|
| wall accommodation | the bouncing particle | 1 |
| pair collision (`ntc.py:255`) | both partners | 2 |
| NTC attempt count (`ntc.py:368`) | every particle in the cell | ~`npc` |

A global estimator built on `Sum_i logh_acc_i` is therefore only valid where every
multiplicity is 1 -- true for `acc`, false for every collision parameter. The `acc`
validation above is sound for this reason; an earlier `d_ref` "global" number measured the
same way was **retracted**, and its +-100 % error bar had no power to notice.

Doing it properly needs a second accumulator in which each decision is owned by exactly one
particle. That is additive -- `logh_acc` and `surrogate` keep their present meaning
untouched -- but it does mean writing a parallel array in `ntc.py`.

## The better answer for `acc`: relax the Bernoulli instead of scoring it

The attribution problem exists only because `acc` is reached through a **score**. It is not
reached that way by necessity -- it is reached that way because the accommodation branch is
sampled with a hard threshold. Relaxing that threshold moves `acc` onto the pathwise route
and the problem disappears rather than being paid for.

This is the same move [P3] section 4 already makes for the *crossing* event, where
`tau ~ N(dt, eps^2)` replaces a hard indicator with a smooth probability. `acc_delta`
applies it to the accommodation draw (`boundary._diffuse_velocity`):

```
w      = sigmoid((acc - u) / delta)          # delta = 0 -> the hard threshold, exactly
v_out  = w * v_diffuse + (1 - w) * v_specular
```

Measured on the same cheapest-Couette case, against an ensemble FD of the **unrelaxed**
physics (`1.686e4 +- 4.1e2`):

| `delta` | pathwise `dJ/d(acc)` | +- SE | ratio to truth | fixed-key FD error | forward `<\|v\|^2>` shift |
|---|---|---|---|---|---|
| 0.30 | 2.668e4 | 1.5e2 | 1.58 | 1.5e-10 | -16.8 % |
| 0.20 | 2.958e4 | 2.1e2 | 1.75 | 3.4e-10 | -14.5 % |
| 0.10 | 2.388e4 | 3.1e2 | 1.42 | 5.3e-10 | -7.8 % |
| 0.05 | 1.876e4 | 4.6e2 | **1.11** | 5.5e-09 | -7.9 % |
| 0.02 | 1.914e4 | 7.0e2 | **1.14** | 1.4e-10 | **-0.35 %** |

Three things this buys, and one it costs:

* **`acc` becomes pathwise.** The fixed-key finite difference reproduces AD to ~1e-10 at
  every `delta`. That is the *strong* per-key test, which a score-carried parameter can
  never pass -- its fixed-key FD is identically 0.
* **No attribution bias at all**, because there is no score term left to attribute.
* **7-35x lower variance than the global form** (SE 1.5e2-7.0e2 against 5.3e3).
* **It costs an `O(delta)` bias**, and `delta` is a genuine bias-variance knob exactly like
  `eps`: as `delta -> 0` the sigmoid derivative concentrates into a spike at `u = acc`, so
  the bias falls and the variance climbs (SE rises 1.5e2 -> 7.0e2 across the sweep above).
  `delta = 0` reproduces the hard threshold **bitwise**, so this ships switched off.

**Caveat on the last column.** The forward `<|v|^2>` shift was measured on a *single key*,
so it is a weak number -- the non-monotonicity between `delta = 0.10` and `0.05` is almost
certainly its own noise rather than structure. It needs the same ensemble treatment as
everything else before any of it is quoted; what is safe to say from it is only that the
shift is small at `delta = 0.02` and not small at `delta >= 0.1`.

The same trick should extend to the **inflow rate**: `stochastic_round` is the hard
threshold there, and giving entrants a fractional *weight* instead of rounding the count
would make `nrho` pathwise for the same reason. That is not built yet.

## Practical guidance

| parameter class | route | status |
|---|---|---|
| wall temperature, wall velocity, inflow temp/vstream | pathwise | **exact** -- verified to 1e-11 against fixed-key FD, unaffected by any of this |
| `acc` | relaxed pathwise (`acc_delta > 0`) | recommended; `O(delta)` bias, 7-35x cheaper than the unbiased alternative |
| `acc` | score (`acc_delta = 0`) | biased low, growing with `M` -- 2.8x at `M = 40` |
| collision parameters (`d_ref`, `omega`, `alpha`) | score | ~6 % low at `M = 16`; grows with `M`; no relaxation built |

**The single rule to take away:** a score-carried gradient from a long trajectory is a lower
bound on its own magnitude, and the deficit grows with the step count. Before quoting one,
either relax the decision onto the pathwise route or check it against an ensemble FD at the
trajectory length you actually intend to use.

## The other wall: chaos limits the horizon, whatever the attribution

Attribution is not the only thing that bounds a DSMC gradient. Even where the estimator is
pathwise -- exact per key, no score, no attribution question at all -- the derivative of a
*single trajectory* grows like `exp(lambda t)` while the quantity it estimates stays O(1).

Measured on the cheapest Couette case, `d u_s/d u_wall` in the first slab (analytic
-0.571):

| steps | mean | +- SE | per-seed spread |
|---|---|---|---|
| 10 | -0.3144 | 3.3e-03 | 3.7e-02 |
| 50 | -0.5197 | 6.5e-03 | 7.4e-02 |
| 100 | -0.5363 | 3.8e-02 | 4.3e-01 |
| 200 | -0.5288 | 7.2e-02 | 8.1e-01 |
| 400 | -1.0560 | 4.7e-01 | 5.4e+00 |
| 800 | -2.4859 | 3.0e+01 | 3.4e+02 |

**A factor of 9 254 in spread over 80x the trajectory.** The estimator never becomes
*wrong* -- every horizon is consistent with the analytic value -- it becomes uninformative,
which is a failure mode that looks like success if only the mean is reported.

### Two obstacles, not one: the shock density ratio fails both ways

The shock density ratio is where this bites hardest, and it was initially mis-diagnosed
here as a pure chaos problem. It is not. **At a 300-step horizon the AD gradient has the
wrong sign**, measured against an ensemble finite difference of the same functional:

| `M_1` | ensemble FD (truth) | AD | Rankine-Hugoniot |
|---|---|---|---|
| 1.60 | **+1.572e-03** +- 2.2e-04 | -2.538e-03 | +4.034e-03 |
| 2.00 | **+1.763e-03** +- 1.8e-04 | -1.937e-03 | +3.182e-03 |
| 3.00 | **+2.567e-04** +- 1.6e-04 | -4.298e-04 | +1.624e-03 |

At that horizon the per-seed spread is 8.7e-03 against a 3.2e-03 signal -- a factor of
three. **The gradient is precise and wrong**, which is the attribution failure, not the
chaos one. The density ratio reaches `u_1` only through discrete decisions whose
contributions have *opposite* signs -- the initial-condition `alive` Bernoulli raises the
ratio, the injection count lowers it during the transient -- and under-counting two
opposing terms by different factors flips the sum.

Chaos is the *second* obstacle, and it sits behind the first:

| steps | forward error in `rho_2/rho_1` | per-seed spread of its derivative |
|---|---|---|
| 300 | -38.7 % | 8.7e-03 |
| 1000 | **-2.6 %** | **4.0e+06** |
| 1500 | -- | 1.0e+12 |

Reaching Rankine-Hugoniot compression needs mass to redistribute across the domain (~1000
steps); by then the pathwise gradient is nine orders of magnitude above its own signal.

**The two bracket the problem from opposite sides.** Fixing the attribution leaves the
horizon broken; shortening the horizon leaves the attribution broken. The temperature ratio
in the same run validates at 300 steps because it reaches `u_1` **pathwise**, through the
prescribed inflow temperature and local equilibration -- so it meets neither obstacle.

The distinction worth carrying forward is not "which parameter" but **"does the quantity
reach the parameter pathwise, and does its own equilibration horizon sit inside the chaos
horizon"**. Both must hold.

## Particle count fixes the solve, not the gradient

A third scaling, separate from attribution and from trajectory length: **how does gradient
variance fall with particle count?**

A gradient's variance has two parts that scale differently. The **sampling** part averages
over particles exactly as the forward observable does. The **chaotic** part is set by the
trajectory's Lyapunov instability and does not fall with `N` at all -- more particles add
degrees of freedom, not averaging, to a derivative that diverges coherently.

Same Couette sweep, same observables, two horizons (512 seeds, relative per-seed spread):

| | forward `u_s` | `du_s/du_wall` | forward `T` | `dT/dT_wall` |
|---|---|---|---|---|
| **25 steps** | `N^-0.498` | **`N^-0.513`** | -- | **`N^-0.521`** |
| **200 steps** | `N^-0.483` | **`N^+0.002`** | `N^-0.506` | `N^-0.099` |

At 25 steps everything obeys `N^-1/2`, gradients included. At 200 steps the forward
quantities still do and **the gradients are flat** -- 115 % relative spread at `N = 180` and
115 % at `N = 2880`, a 16x increase in particles buying exactly nothing.

**The practical rule:** particles buy gradient accuracy only up to the horizon where chaos
takes over. Past that, only ensemble size helps, at `M^-1/2`, and that is the axis to spend
on.

One trap in reading the short-horizon column: it is precise and **not converged**.
`du_s/du_wall` reads -0.46 at 25 steps against an analytic -0.571, because 25 steps is well
short of this case's 290-step momentum-diffusion time. Low variance is not correctness --
the same confusion that made an earlier cell-size study look "flat".
