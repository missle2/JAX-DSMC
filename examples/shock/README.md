# M2 — periodic transport and shocks

Two cases share one driver. Both are periodic in every direction, which is all the boundary
machinery M2 has; inflow and walls arrive with M4/M5.

```bash
python3 -m examples.shock --dims 1 2 3 --plot        # sine steepening, all dimensions
```

## 1. Sine steepening — the cross-dimensional gate

`u_x(x, 0) = U0 sin(2πx/L)`, uniform in the other directions. Fast fluid overtakes slow, the
profile becomes a sawtooth, and a shock forms at `t_s = L/(2πU0)`.

**The acceptance gate is that D = 1, 2 and 3 agree.** The initial condition varies only in `x`, so
this is one-dimensional physics whatever `D` is; run with the same `dx` and the same particles per
`x`-slab, the profiles must match to within Monte Carlo error. That one test covers a lot:

- `GridSpec.transverse`, without which `cell_volume` is a *length* at `D=1` and an *area* at
  `D=2`, corrupting the collision rate and `nrho` together (and not cancelling — rescaling `fnum`
  to fix the rate leaves `nrho` wrong by the square);
- advection and periodic wrapping along each axis;
- that `move`'s `v[:, :D]` slice really does leave the physics alone.

Measured at D = 1, Mach 2, Kn = 0.0133: **front thickness 6.0 mean free paths**, against 23.9 for
an un-steepened sine — a genuine shock, and a thickness of a few mfp is the kinetic answer a
continuum solver cannot produce, since Navier–Stokes has no intrinsic thickness scale.

## 2. Riemann problem — against the exact Euler solution

A density and pressure jump at `x = L/2`, compared with the exact Riemann solution
(`exact.py`, Toro ch. 4 — validated against Toro's published star states to 5 decimals).
The domain stays periodic: there are simply two interfaces, and the comparison is made before
waves from one reach the other.

Measured at Kn = 0.0013, 1024 cells, N = 205k, 4 seeds (`figures/sod_vs_exact.png`): the
rarefaction fan, the left uniform state and the `u* = 190 m/s` plateau all track the exact
solution closely. Wave positions:

| wave | DSMC | exact | offset |
|---|---|---|---|
| contact | 0.5454 L | 0.5617 L | −12.2 mfp |
| shock | 0.6284 L | 0.6494 L | −15.7 mfp |

**Both waves lag, and that is expected.** An initially sharp discontinuity is not a shock; it takes
several collision times to resolve into one, and only then propagates at the Euler speed. 58 mean
free times elapsed here, so a startup delay of ~10 mft accounts for the observed 12–16 mfp. The
testable prediction is that the lag stays roughly constant **in mfp** as the run is extended, so
its share of the travel distance shrinks — that has not yet been measured.

**What the rms errors do and do not mean.** DSMC's contact discontinuity diffuses and its shock has
a finite thickness; Euler has neither. Any such structure inside the comparison mask contributes
Boltzmann-vs-Euler difference, not solver error. That is why the report leads with wave *positions*
and masks a few mfp either side of each wave.

## 3. Bird's normal-shock structure — the kinetic gate

```bash
python3 -m examples.shock.normal        # or scratch/bird/run_case.py for the raw profiles
```

The Riemann problem in §2 is *inviscid*: it pins wave positions and plateau values but says
nothing about the one quantity that is purely kinetic — the **thickness** of the shock. Bird
§12.11 measures exactly that, and it is anchored to experiment, so it catches errors in the
collision rate and the transport coefficients that an Euler comparison cannot see.

Bird uses a piston/stream boundary (`DSMC1S`). We have neither inflow nor a moving wall until
M4/M5, so we run the equivalent **reflected shock**, which needs only the specular wall M2
already has: uniform gas approaching a stationary wall at `-u_p`, with
`u_p = M a_1 (1 - 1/r)`. Same steady structure, different frame.

### Results at Mach 2, argon (ω = 0.81, α = 1.40 from SPARTA's `ar.vss`)

| quantity | D=1 | D=2 | D=3 | reference |
|---|---|---|---|---|
| ρ₂/ρ₁ | 0.59 % | 0.18 % | 0.37 % | Rankine–Hugoniot 2.2857 |
| T₂/T₁ | 0.13 % | 0.23 % | 0.56 % | Rankine–Hugoniot 2.0781 |
| shock speed | 0.07 % | 0.22 % | 0.30 % | 269.4 m/s (RH) |
| **L/λ₁ (Bird 4.52)** | **5.98** | **6.12** | **6.18** | **Bird 6.2** |
| **L/λ₁ (Alsmeyer HS)** | **4.53** | **4.64** | **4.68** | **Alsmeyer 4.8** |
| T leads ρ | 2.09 λ₁ | 2.10 λ₁ | 2.07 λ₁ | Bird: T leads ρ always |

The thickness agrees with **both** references simultaneously, which is the real test: Bird and
Alsmeyer are only mutually consistent if the mean-free-path conversion between them is right.

Three independent thickness estimators agree at D=3 — max-slope at 5-cell smoothing 6.18,
max-slope extrapolated to zero smoothing 6.10, and a `tanh` fit with no smoothing at all 6.14.

### The mean free path is the whole game, and it is easy to get wrong

**This normalisation produced a 24 % false discrepancy before it was caught.** Three different
"mean free paths" appear in this problem and they differ by up to 60 %:

| definition | argon at n = 1e21 | what it is |
|---|---|---|
| `mfp_hard_sphere` | 1.332e-3 m | `1/(√2 π d_ref² n)`, the VHS *reference* diameter |
| `mfp_vss` (Bird 4.52) | **1.294e-3 m** | what Bird normalises by |
| `mfp_alsmeyer` | 1.708e-3 m | hard spheres matched to argon's **viscosity** |

Bird defines eqn (4.52) through the viscosity,
`λ = 2μ(5−2ω)(7−2ω) / [15 ρ √(2πRT)]`. Substituting the VSS viscosity (Bird 4.64) collapses the
`ω` dependence entirely and leaves

```
λ_4.52 = [(α+1)(α+2) / (6α)] · 1/(√2 π d_ref² n)
```

a function of **α only**. The original implementation instead used
`α(5−2ω)(7−2ω)/[4(α+1)(α+2)] × λ_HS`, which is 1.2455× too small for argon.

That wrong factor is not a random error — it is `μ_HS(d_ref)/μ_VSS = (d_HS/d_ref)²`, a
**cross-section** ratio, and it equals 0.8244 at ω=0.75, α=1.5325, a number that appears in
Bird's text near his mean-free-path discussion. Matching one quoted number is not verification.
What settled it was deriving λ from the *measured* viscosity, which reproduces Bird's stated
"76 % of the hard sphere mean free path" (0.7577) and gives exactly 1.000000 in the hard-sphere
limit ω=½, α=1.

### What was ruled out along the way

The 24 % gap was chased through five wrong explanations before the mfp was found; each is now a
guard in the code:

- **under-settling** — refuted by measurement. One long run sliced into travel-distance blocks
  shows L flat from 20 to 177 λ₁ (`scratch/bird/settle_curve.py`). Bird settles ~20 shock widths;
  we are converged far earlier.
- **the alignment estimator** — real, and worth 20 %. Aligning each snapshot on its own
  half-density crossing pins the ensemble to the half level at ξ=0 and puts a cusp exactly where
  the maximum gradient is read. Fixed by an equal-area estimator plus a per-seed linear fit of
  `x₀(t)`, which also yields the shock speed as a free check (0.3 %).
- **the far boundary** — real. Gas streams away from `x = L`, so an expansion runs back at
  `u_p + a₁`, **faster** than the shock, and always catches it. `setup` computes when and
  `run` refuses to start a run that would sample after it.
- **smoothing width** — real, worth ~10 %. A boxcar adds width in quadrature (measured
  coefficient 0.96–1.10 against a predicted 1), so it is extrapolated away, and cross-checked
  against a `tanh` fit that needs no smoothing.
- **the collision model** — ruled out independently: the 2-D Taylor–Green case measures the
  viscosity to **+0.38 %** of Chapman–Enskog, and `test_vss.py` now asserts the mean collision
  rate equals the hard-sphere rate at `T_ref` for every ω.

---

## 4. Two-sided-inflow shock — the open-boundary gate

The three cases above drive their shocks with a **piston off a specular wall**: the domain
is closed, so the boundary conditions under test are reflection and periodicity. This
fourth case holds the *same* Mach-2 shock stationary with **inflow at both ends** — the
left face injects the upstream state, the right face the Rankine–Hugoniot post-shock
state — which is [P3]'s test3 geometry and the verification case for the inflow boundary
condition.

```bash
python3 -m examples.shock.inflow_report              # run, log and plot
python3 -m examples.shock.inflow_report --plot-only  # re-plot from the stored log
```

It checks two different things at once, and the split matters:

* the **upstream** rows test the *boundary condition* — does inflow deliver the state it
  was asked for;
* the **downstream** rows test the *interior physics* — does the solver reproduce
  Rankine–Hugoniot from that inlet.

`L = 60 λ₁`, 200 cells, 40 000 particles, 20 000 warm-up steps
(`results/inflow_driven.log`):

| moment | measured | Rankine–Hugoniot | error |
|---|---|---|---|
| `nrho` upstream | 9.9793e+20 | 1.0000e+21 | −0.21 % |
| `T` upstream | 2.7075e+02 | 2.7315e+02 | −0.88 % |
| `ux` upstream | 6.1602e+02 | 6.1580e+02 | +0.04 % |
| `nrho` downstream | 2.2738e+21 | 2.2857e+21 | −0.52 % |
| `T` downstream | 5.6583e+02 | 5.6764e+02 | −0.32 % |
| `ux` downstream | 2.7015e+02 | 2.6941e+02 | +0.27 % |

**Worst moment 0.88 %**, with the mass flux `n·uₓ` conserved plateau-to-plateau to
**0.09 %**, `inflow_shortfall = 0` and `attempts_clipped = 0`.

This is a **single run**, not an ensemble, and the per-moment errors carry run-to-run
scatter of a few tenths of a per cent — an independent repeat of exactly this
configuration moved the worst moment between 0.68 % and 0.88 % and the flux conservation
between 0.09 % and 0.17 %. The claim the case supports is therefore "every moment inside
~1 %", which is robust at that scatter; do not read the individual digits as resolved.
(`examples/couette` learned this the hard way — see `docs/HANDOFF.md`.)

![two-sided-inflow shock](figures/inflow_driven.png)

### Initialise with the shock already in the middle

Filling the box with the upstream state instead makes the shock form at the outlet and
never travel back. Measured, it parked at `x/L = 0.900` with only **6.0 mfp** of
downstream run — less than a Mach-2 shock is thick — so the "downstream plateau" was
inside the shock structure and read **9 % off** Rankine–Hugoniot.

What identified the fault as the *layout* rather than the solver was that the mass flux
stayed conserved to **0.07 %** throughout. A solver that had got the physics wrong would
not conserve `n·uₓ`; one that had simply been asked the wrong question would. Placing the
shock at mid-domain gives 16.5 mfp of run and the table above.

### `A_max` must be sized for the *downstream* side

The post-shock gas is 2.29× denser and the NTC attempt count goes as `npc²`, so the
downstream cells need roughly five times the upstream attempt budget. A first attempt at
the reduced test configuration ran with `a_max = 64` and clipped **5328** attempts — gap
U4's silent-under-collision failure mode exactly, caught only because
`Diagnostics.attempts_clipped` is asserted rather than logged. The shipped case uses
`a_max = 128` at `npc = 200`, where the measured peak is well inside it.

---

## 5. Gradients against Rankine-Hugoniot

```bash
python3 -m examples.shock.gradients
```

`d(T_2/T_1)/du_1` **agrees with the closed form across `M = 1.6 ... 4.0`** at a 300-step
horizon -- the temperature response reaches the inlet velocity *pathwise*, through the
prescribed inflow temperature and local equilibration.

`d(rho_2/rho_1)/du_1` **does not, and comes out with the wrong sign.** It is not a noise
problem: at that horizon the per-seed spread is 3x the signal, so the gradient is precise
and wrong. The density ratio reaches `u_1` only through discrete decisions with opposing
contributions, and the per-particle attribution under-counts them by different factors.
A second obstacle sits behind it -- reaching the Rankine-Hugoniot compression takes ~1000
steps, by which point chaotic amplification puts the derivative nine orders of magnitude
above its own signal. Full analysis in [`docs/ATTRIBUTION.md`](../../docs/ATTRIBUTION.md).

![shock gradients](figures/gradients.png)

**A forward bug this study found.** `inflow.inject` silently dropped entrants when dead
slots ran out: `shortfall` counted only `target > capacity`, though its contract says both.
A saturated particle array then looks exactly like correct physics -- at Mach 4 the density
ratio flattened at 1.83 against a Rankine-Hugoniot 3.37 (**-46 %**) with every diagnostic
reading zero. Fixed; the counter now reports it.
