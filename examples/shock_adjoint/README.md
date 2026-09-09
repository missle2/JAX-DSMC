# `shock_adjoint` — do the gradients agree with finite differences?

Reuses the reflected-shock case that reproduced Bird's §12.11 thickness in
[`examples/shock`](../shock/) and asks the adjoint question of it: **is
`jax.grad` of the forward solve the gradient it claims to be?**

```bash
python3 -m examples.shock_adjoint --mode pathwise          # fast, ~2 min
python3 -m examples.shock_adjoint --mode ensemble --keys 4096
python3 -m examples.shock_adjoint                          # both + figures
```

The claim under test is the project's central architectural bet (`PSEUDOCODE.md`
§11): that [P1] Algorithm 2 and [P2] Algorithm 3 *are* reverse-mode AD of the
forward solve plus one DiCE magic box, so no hand-written backward pass is needed.
`tests/test_adjoint_oracle.py` checks that against the papers' own recursions on a
0-D forward pass. **This example checks it against finite differences on the real
thing** — NTC driver, cell sort, specular wall, time loop, `D = 1, 2, 3`.

## The verification needs two modes, and using the wrong one gives a confident wrong answer

The parameters split in two, by *how* they reach `J`. Measured, not assumed
(`scratch/adjoint/probe_split.py`):

| parameter | route to `J` | pathwise `dJ/dθ` at fixed key | checkable by |
|---|---|---|---|
| `temp` | reparameterised Maxwellian | nonzero | **mode A** |
| `ux` (stream/piston speed) | reparameterised Maxwellian | nonzero | **mode A** |
| `alpha` | VSS deflection law `cos X = 2U^(1/α) − 1` | nonzero | **mode A** |
| `diam` | `q = σ_T v_r` → accept/reject **only** | **exactly 0.0** | mode B only |
| `omega` | `q` → accept/reject **only** | **exactly 0.0** | mode B only |
| `tref` | `q` → accept/reject **only** | **exactly 0.0** | mode B only |
| `nrho` | `fnum` → attempt count **only** | **exactly 0.0** | mode B only |

At frozen randomness `J` is a **step function** of anything in the bottom half.
Perturb `diam` by less than it takes to flip an accept bit and `J` does not move at
all — so a fixed-key finite difference returns exactly `0.0`, the pathwise gradient
is also exactly `0.0`, and **they agree perfectly on a number that is not the
sensitivity**. The real sensitivity is the score term, it estimates `dE[J]/dθ`, and
only an ensemble over base keys can see it.

**Mode A** — fixed base key. `jax.grad` of the plain objective vs a central
difference whose step is bisected down until no frozen decision flips.

**Mode B** — many base keys. Mean *scored* gradient vs a central difference of
`E[J]` at a step large enough that decisions do flip.

## The flip guard

`Diagnostics.decision_checksum` fingerprints every frozen decision of a step — cell
membership, attempt count, pair indices, accept bits — weighted by an irrational
rotation so two flips cannot cancel. Mode A compares it between the `±` evaluations
and **refuses to report a number** unless they match exactly.

This is not defensive decoration. §13.4.1 measured the failure it prevents: at `h` a
factor of three too large, one particle changing cell rebinds 60 pair indices
(3.9%) and the FD estimate is wrong by a factor of **11** — with no error, no `NaN`
and no diagnostic. The usable window is a cliff, not a slope, and `h_crit` varies up
to 30× between seeds, so the step is re-bisected at every sweep point rather than
chosen once.

## Why this case is small

`h_crit ~ 5.8 / (N·C·M)` against a round-off floor near `1e-9`. The production shock
of `examples/shock` runs `N·C·M ~ 1e11`, which puts `h_crit` at `6e-13` — **below
the floor, so no valid step size exists at all**. The FD gate is available only on
small cases; that is a property of the method, not of this code. Above it, the
oracles of §13.1–13.3 are the only usable check.

The cases here sit at `N·C·M` = `1.3e6` (D=1), `5.1e6` (D=2), `2.0e7` (D=3).

## Results — mode A, `D = 1, 2, 3`

`python3 -m examples.shock_adjoint --mode pathwise` (250 s on one RTX 3050 Ti).
Every sweep is 7 points spanning ±15% of Bird's Mach-2 argon reference, and the
finite-difference step is re-bisected at every point until the decision checksum
matches on both sides.

| D | parameter | `N·C·M` | flip-free `h` range | max rel error | `err/medAD` |
|---|---|---|---|---|---|
| 1 | `T₁` | 1.3e6 | 1.6e-7 – 1.0e-5 | **9.3e-10** | 9.5e-10 |
| 1 | `u_stream` | 1.3e6 | 1.3e-6 – 1.0e-5 | **8.1e-11** | 9.3e-11 |
| 1 | `α` | 1.3e6 | 5.0e-6 – 1.0e-5 | 1.7e-7 | **6.7e-8** |
| 2 | `T₁` | 5.1e6 | 6.3e-7 – 1.0e-5 | **4.4e-10** | 4.4e-10 |
| 2 | `u_stream` | 5.1e6 | 1.6e-7 – 5.0e-6 | **3.0e-10** | 3.1e-10 |
| 2 | `α` | 5.1e6 | 2.5e-6 – 1.0e-5 | 2.0e-6 | **3.2e-7** |
| 3 | `T₁` | 2.0e7 | 3.1e-7 – 1.0e-5 | **3.1e-10** | 3.1e-10 |
| 3 | `u_stream` | 2.0e7 | 3.1e-7 – 2.5e-6 | **3.0e-10** | 3.3e-10 |
| 3 | `α` | 2.0e7 | 6.3e-7 – 1.0e-5 | 4.8e-7 | 4.8e-7 |

All nine **PASS**. `err/medAD` normalises by the sweep's median `|AD|` instead of the
per-point one — see "Why `α`'s column looks worse than `T₁`'s" below for why that matters
and why it moves `α` but not the other two.

Note the `h` ranges above top out at the harness cap `1e-5`, so **this run does not
measure `h_crit`** and is not evidence for §13.4.1's `h_crit ~ 5.8/(N·C·M)`. The earlier
uncapped run gave two usable points (`D = 2, 3`: `1.3e-5` and `6.3e-6` against a
predicted `1.1e-6` and `2.9e-7` — the fit is conservative). Gap **U8** stays open.

## What this does and does not establish

| claim | status |
|---|---|
| the **pathwise** gradient is exact through `move → sort → collide`, all `D` | **verified here**, to round-off |
| the **score** terms reproduce [P2] eq. `back_eta`, all three branches | **verified exactly** by `tests/test_adjoint_oracle.py` (0-D, rtol 1e-10) |
| dropping the reject-branch score is detectable | **asserted** to change the gradient |
| the *combined* estimator converges to `dE[J]/dθ` on the production solve | **open** — needs the ensemble of mode B |
| `diam`, `omega`, `tref`, `nrho` gradients are right | **open** — score-carried, see below |

[P1] is the pure-pathwise case: `Σ = q`, so `p ≡ 1`, nothing is ever rejected and no
score term exists. [P2] adds the `η` terms but still reports sensitivities with
respect to the *initial distribution* parameter `m = (T⁰ₓ, T⁰ᵧ, T⁰_z)` — our `T₁`,
which is the parameter that comes out cleanest here. Neither paper differentiates a
collision-model constant.

## Mode B, and why it is not a verification claim yet

[P2] §6 does not treat the score estimator as cheap. `main.tex:1425` averages
**`M_s = 100`** i.i.d. runs for *both* the AD and FD gradients; `main.tex:1431` gives
the random error as **`O((√(N·M_s)·|Δm|)⁻¹)`** and fixes **`Δm = 0.1`** — a 10%
finite-difference step — as "nearly optimal for `N = 10⁶` Maxwellian particles,
`M_s = 100`". `main.tex:1502` reports the standard deviations growing as `O(√M)`.

So a large step plus a big ensemble is the published protocol. Their budget is
`N·M_s = 10⁸`; a 512-key probe at `N = 2000` is `10⁶`, a hundredfold less, which on
their own scaling law predicts error bars ~10× wider.

**Their `N^-1/2` law is measured here, not assumed.** `scratch/adjoint/probe_variance.py`
sweeps `N = 4·10³ → 2.56·10⁵` at fixed `A_max` with `φ = T_y`, 128 keys per point:

| parameter | fitted `p` in `sd ~ N^p` |
|---|---|
| `temp` | −0.508 |
| `ux` | −0.464 |
| `alpha` | −0.495 |
| `diam` | −0.499 |

`diam`'s ensemble mean is stable to ~10% over that 64× range while its relative error
falls 9.7% → 1.2%. `alpha`'s relative error looks large only because its mean is
consistent with **zero** at every `N` — its *absolute* error bar falls 8× like
everything else. See `docs/PSEUDOCODE.md` §23.3.

### Choosing the gas — argon is the wrong one for α

The case defaults to **argon** (Bird §12.11 Mach 2), which keeps the published shock
comparison intact. But argon is the **worst** available gas for measuring an α
sensitivity, and that is measured rather than argued:

```python
from jax_dsmc import species_data as sd
cs    = sa.case(2, gas=sd.HELIUM)        # 12.6x argon's α conditioning
theta = sa.reference_theta(sd.HELIUM)
```

Anisotropy relaxation is driven by `⟨cos²χ⟩`, whose α-derivative vanishes at exactly
`√2 = 1.41421`; argon's tabulated `α = 1.40` sits 1.0% away, so a temperature-like
objective has structurally almost no α sensitivity there. `examples/p2_replication`
measures `|AD|/SE` across Bird's five monatomic gases: **He 4.7, Ne 3.1, Kr 2.6, Ar 0.3,
Xe 0.0**.

The gas lives **on the `Case`**, not as a separate argument, because `mass` and `dt` are
derived from it — helium is 16× lighter, so ~4× faster thermally, so it needs a ~3×
smaller `dt`. The jitted wrappers are `functools.cache`d on the case, so a gas passed
outside it would let two gases silently share one cache entry and run with the wrong `dt`.

That is 0-D, though, and the verification target is the 2-D reflected shock, which adds
transport, a specular wall and a developing flow. Measured there directly
(`scratch/adjoint/probe_gas_alpha.py`, 256 keys, `N = 16000`, `φ = T_y`):

| gas | α | cond | shock `AD ± SE` | `\|AD\|/SE` | `\|FD\|/SE` | `SE_FD/SE_AD` |
|---|---|---|---|---|---|---|
| He | 1.26 | 12.6× | `860 ± 318` | **2.7** | 0.57 | **8.4×** |
| Ar | 1.40 | 1.0× | `111 ± 139` | 0.8 | 1.45 | 2.2× |

**Helium helps, but the shock eats most of the gain.** The realised improvement is 3.4×
against 12.6× analytic (0-D realised 15.7×), and helium lands at 2.7σ rather than 0-D's
4.7σ. So α is *marginally* usable in the shock at this budget, not comfortably so:
reaching 10σ needs ~14× the ensemble (~3500 keys, or `N ≈ 2.2e5` at 256 keys).

Two things fall out that matter more than the headline:

- **FD cannot do this at all.** Helium's finite difference sits at 0.57σ — the signal is
  below its own noise — so the `|z| = 0.24` AD-vs-FD agreement is a *low-power* check, not
  a verification. Reaching 3σ on FD alone would need ~28× the ensemble.
- **The adjoint is 8.4× less noisy than FD** for helium in the shock, against the 2.0–2.3×
  measured in the 0-D `p2_replication`. That is the first configuration here where [P2]'s
  claimed ~10× advantage actually shows up, and it appears precisely where the pathwise
  term is doing the work.

Argon's `|FD|/SE = 1.45` exceeding its `|AD|/SE = 0.8` is not evidence against AD: argon's
true derivative is ~0, so both are fluctuations about zero and neither is resolvable.

Figures: `figures/gas_resolvability.png`, `figures/gas_shock_alpha.png`, from
`python3 -m examples.shock_adjoint.gas_report`.

This affects **mode B only**. The mode-A pathwise check is a fixed-key comparison with no
ensemble averaging, and it passes at argon to `2e-6` regardless.

### Why `alpha`'s column looks worse than `temp`'s

Three separate effects, and it took all three to account for it:

1. **Truncation above `h ~ 1e-5`.** `alpha` flips fewer frozen decisions than a velocity
   does, so its flip-free window is wide and the bisection stopped at the harness cap
   rather than at `h_crit`. Sweeping `h` shows a clean `O(h²)`. The cap is now `1e-5`.
2. **Near-stationary sweep points.** `alpha`'s sweep crosses points where `|dJ/dα|` falls
   to ~10 against a sweep-typical 100–700, and a per-point relative error divides by
   exactly that. Across the 21 points the error tracks `|AD|` (Spearman ρ = −0.835,
   p < 10⁻⁴) and not `h` (ρ = +0.025, p = 0.92). The report now also prints
   **`err/medAD`**, normalised by the sweep's median gradient; it moves `alpha` by 2.5–6×
   and leaves `temp`/`ux` unchanged to two digits — the control it needed.
3. **Genuinely larger higher derivatives.** What remains is ~100×, and it is structural:
   `u_stream` enters `J` **linearly**, `T₁` through a **square root**, but `α` through
   `cos χ = 2u^(1/α) − 1` — all orders in `1/α`. At a common relative step the parameter
   with the larger `|J'''|/|J'|` pays more truncation.

All nine cases PASS. Worst-case agreement is `2e-6` relative on a chaotic Monte Carlo
objective.

**Full study — tables, figures and provenance: [`VARIANCE.md`](VARIANCE.md)**,
regenerated by `python3 -m examples.shock_adjoint.variance_report`.

### Where the noise actually comes from

`scratch/adjoint/probe_split_variance.py` separates the pathwise and score halves of the
gradient and measures both, plus their correlation (256 keys, `φ = T_y`):

| parameter | `sd(pathwise)` | `sd(score)` | corr | dominated by |
|---|---|---|---|---|
| `temp` | 1.275 | 0.469 | −0.01 | **pathwise** |
| `ux` | 1.095 | 0.263 | +0.08 | **pathwise** |
| `alpha` | 679 | 121 | −0.10 | **pathwise** |
| `diam` | 0.0 | 3.45e12 | — | score (necessarily) |

Every parameter with a pathwise route carries **3–5× more pathwise noise than score
noise**, and the halves are uncorrelated. Two consequences:

- **A baseline cannot help much.** The halves are uncorrelated, so *variances* add and
  the score's share of the **variance** — what a baseline removes — is only 3.1% for
  `α`, 5.5% for `u_stream`, 11.9% for `T₁`. A perfect baseline therefore caps at
  1.016–1.065× here. (That is *not* an explanation of §11.4's measured ~1.2×, which
  exceeds this ceiling and was measured on a windowed `φ` at `D = 1, N = 2000` — a
  different configuration. The directional claim is what holds.)
- **More particles is the only lever** on the dominant term, exactly as the `N^-1/2`
  law says. That is the answer to "why not 10⁶ particles?": for mode-B statistics,
  yes — it is the one thing that works.

### `Δm = 0.1` is a truncation test, not a precision test

Running [P2]'s protocol as closely as this GPU allows (`N = 2.56·10⁵`, `M_s = 100`,
`Δm = 0.1`) gives AD-vs-FD `|z|` of 11 (`temp`), 43.5 (`ux`) and 6.4 (`diam`). That is
the **10% step**, not a disagreement: `Δm = 0.1` sits at a truncation-versus-noise
optimum evaluated at [P2]'s noise level, and our AD standard error here is 0.03–0.25%,
low enough that `O(h²)` truncation dominates. `ux` is worst because a 10% step moves
this case from `Ma = 2.0` to `Ma = 2.2`. **Mode A is the precision test**; mode B's step
is deliberately large enough to flip frozen decisions.

### The spatial window costs signal, not variance

Comparing `φ = T_y` against the Gaussian-windowed `φ` on the same case and keys, the
window *lowers* the absolute standard deviation in every parameter — but it lowers the
mean faster. For `diam` the spread falls 2.4× while the signal falls 15×, a net **6.3×
loss of signal-to-noise**; `temp` and `ux` lose 1.7× and 2.6×. A localised objective is
a legitimate thing to want; it just costs sample budget in proportion to the fraction of
the gas it looks at.

### Longer runs do not degrade the gradient

[P2] reports `sd ~ √M` in the step count. That is measured against a 0-D relaxation, where
the signal saturates; in a *developing* shock the gradient itself grows with `M`, so the raw
`sd` exponent is not comparable. Measuring both (`scratch/adjoint/probe_M_scaling.py`):

| parameter | `q(sd)` | `q(\|mean\|)` | `q(sd/\|mean\|)` |
|---|---|---|---|
| `temp` | 0.291 | 0.017 | +0.273 |
| `ux` | 1.143 | 1.418 | **−0.275** |
| `diam` | 0.720 | 0.721 | **−0.002** |

For the pure-score `diam`, signal and noise grow at the *same* rate to three digits, so its
relative precision is flat in run length. `temp` is the only parameter here in [P2]'s
saturated-signal regime, and its `q(sd) = 0.291` is below their `0.5`. `alpha` is omitted:
its ensemble mean flips sign across the sweep, so the fit is meaningless.

### Reading the figure

`figures/adjoint_pathwise.png` overlays the AD curve with FD markers. **The curves for
`T₁` and `α` are jagged, and that is by design, not instability.** Sweep points are 5%
apart in the parameter — five orders of magnitude more than `h_crit` — so consecutive
points are *different realisations* with different frozen decisions. What is being
tested is that FD lands on AD *at each point*; the smoothness of `u_stream` is the
exception (a uniform velocity shift moves every particle together and re-crosses far
fewer decision boundaries), not the standard.

`figures/adjoint_diagnostics.png` gives the residual sizes on a log axis and the
flip-free `h` against `N·C·M`, with §13.4.1's fitted `h_crit ~ 5.8/(N·C·M)` and the
round-off floor drawn in. The measured points sit *above* the fitted line, so the
fit is conservative where it has now been checked.
