# Decaying turbulence, D ∈ {1, 2, 3}

Status: **reference** (M2 turbulence gates; the 2-D Taylor–Green viscosity gate, +0.38 % vs
Chapman–Enskog, is the project's strongest quantitative check of the VSS collision model;
passed and frozen).  Run from the repository root; `python3 -m examples.turbulence --help` lists
the flags (`--dim`, `--side`, `--npc`, `--mach`, `--efolds`, `--ic`, `--spectrum`, `--ensemble`,
`--reach`, `--movie-dir`, ...).

Same solver and same periodic boundary as the shock cases; only the initial condition and the
post-processing differ. Each dimension is anchored to a reference that is a **number**, not a
slope to be eyeballed — a log-log plot of a noisy spectrum can be made to look like almost any
power law.

| D | case | reference |
|---|---|---|
| 1 | compressive sinusoid | steepens into a shock; jump discontinuities give Fourier coefficients ~1/k, hence **E(k) ~ k⁻²** (Burgers) |
| 2 | Taylor–Green vortex | an **exact** Navier–Stokes solution: `E(t) = E₀ exp(−4νk²t)`, so the fitted rate **measures ν** |
| 3 | Taylor–Green vortex | the standard DNS benchmark, compared at the Reynolds number *actually run* |

## The 2-D case is the strongest quantitative gate in the project

```bash
python3 -m examples.turbulence --dim 2 --side 128 --npc 100 --efolds 2.5
```

It has a closed-form answer, and the quantity it measures — the viscosity — is the transport
coefficient the entire VSS collision model exists to reproduce. Unlike the shock thickness it
needs no smoothing, no shock-frame alignment and no fitted maximum slope. Nothing in the solver
is told ν; it emerges from the collision rate and the scattering law.

**Measured: ν = 3.2036e-01 vs Chapman–Enskog 3.1915e-01 m²/s — +0.38 %, R² = 0.9994**
(128² cells, N = 1.64e6, Kn = 0.049, Ma = 0.3, 2.5 viscous e-folds).

The expected systematic is compressibility at O(Ma²) ≈ 9 %, so landing at 0.4 % says the
compressible correction is small at this Mach number, not that it is absent.

### The noise floor will ruin this if you let it

A cell's bulk velocity is a mean of `npc` thermal velocities, so it carries an error of variance
`v_th²/npc` per component **whatever the flow is doing**. The measured energy is

```
E_meas(t) = E_signal(t) + (3/2) v_th² / npc
```

and that offset is *constant in time* while the signal decays exponentially. Fitting `log E_meas`
without removing it flattens the curve — the fitted rate is biased low, the reported viscosity is
too small, the fit still looks excellent (a decaying exponential plus a constant is smooth), and
no diagnostic fires. `validate.energy_history` subtracts it using the *actual* per-cell counts.

The same floor appears in spectra, and is worse there: shell `k` in D dimensions averages ~k^(D−1)
modes, so binned white noise **rises** as k^(D−1) and can be mistaken for a physical range.
`report_spectrum` returns the floor alongside `E(k)` and a slope is quoted only where the signal
stands 10× clear of it.

**Do not subtract the floor from a spectrum.** It is tempting — it is the same known positive bias
that `energy_history` removes on a scalar — and it was tried here. Above the crossover
`E(k) − noise` is a difference of comparable numbers that fluctuates through zero, so about half
the shells survive any positivity test by chance and the fitted slope is set by that survivorship.
Measured on the 1-D run: subtraction claimed "usable to k = 507" with slope **−0.71** against a
true Burgers −2, and looked entirely healthy. A bias correction that is safe where the signal sits
far above the floor is not safe under a logarithm near the crossover.

**Lower the floor instead.** It is `1.5 v_th²/npc` per cell, so either raise `npc`, or

```bash
python3 -m examples.turbulence --dim 1 --side 1024 --npc 200 --spectrum --ensemble 16
```

`--ensemble M` runs M independent realisations and averages the **velocity field** before
transforming. The initial conditions here are deterministic, so every seed evolves the same mean
flow and differs only in collision noise: averaging M fields is exactly equivalent to `M × npc`,
at the same total cost, and it parallelises across devices with no communication at all. That is
the honest way to buy decades in `k`, and it is what the H200 node is for.

### It was run, and the floor turned out not to be the binding limit

`--ensemble 16` at npc 200 (`npc_eff` = 3200, 3.6 h on a laptop 3050 Ti). The floor fell by
exactly 16×, 8.33e-1 → 5.21e-2. Two things came out of it.

**The single-seed spectrum above `k ≈ 5` was entirely noise**, which averaging makes visible:
`E(8)` fell 2.13 → 0.142 and `E(13)` fell 1.49 → 0.0184, i.e. by 15× and 81×, while `E(2)` and
`E(5)` did not move at all. The −2.45 slope this file previously quoted "over k = 2..9, above 3×
the floor" was measuring the floor.

**But the verdict did not change**, and for two reasons, neither of which was the floor.

*The initial condition combs the spectrum.* `u = u₀ sin(kx)` is antisymmetric under `x → x + L/2`;
the nonlinearity breaks that only slowly, so odd harmonics are populated and even ones are not:

| k | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|
| E(k) | 1.18e+1 | 3.96e-1 | 4.69e+0 | 1.42e-1 | 1.82e+0 |
| E / floor | 226 | 7.6 | 90 | 2.7 | 35 |

`E(7)/E(6) = 11.9` at shells standing 90× and 7.6× clear of the floor — structure, not scatter. **No
shell-averaged slope can be fitted through a factor-12 comb**, and a threshold-selected window just
samples it. This is a defect of the IC, and `--ic broadband` is the fix: random phases over
`1 ≤ |k| ≤ k_max`, solenoidal in 2-D/3-D and compressive in 1-D. Taylor–Green stays the default
because the viscosity gate *needs* an exact NS solution.

*Underneath the comb, viscosity ends it at `k ≈ 10`.* The odd family alone tracks `k⁻²` to within a
factor of 2.5 out to `k = 9`, then collapses — `E(11)` is 6× and `E(13)` is 38× below the power-law
continuation. At `Re = 61` that is `k_d ≈ Re/6`, against an energy peak at `k = 2`: under a decade.

`validate.dissipation_wavenumber` reports `k_d ~ C_D Re^p` (`p` = 1, ½, ¾ for D = 1, 2, 3) and the
run prints it up front. The exponents are standard; `C₁ = 1/6` is the measurement above and
`C₂ = C₃ = 1` are placeholders. **Do not export the 1-D constant** — an earlier revision applied it
in all three dimensions and the reach table then said "NO RANGE" for every H200 configuration,
contradicting the standard DNS requirement `k_d/k₀ ~ Re^(3/4)`.

To establish the 1-D gate, both must be fixed at once. `Re ∝ Ma`, so Mach number is much the
cheaper knob than box size — it costs *nothing*, since a shorter eddy time means fewer steps for
the same `--n-eddy`:

```bash
python3 -m examples.turbulence --dim 1 --side 1024 --npc 200 --mach 1.0 \
    --n-eddy 1.5 --spectrum --ic broadband --k-max 4 --ensemble 8 --a-max 2400
```

`Re ≈ 203`, `k_d ≈ 34`, per-cell SNR 18.3 (against 61, 7.6 and 5.5 at Ma = 0.3). Note the
`--a-max`: at Ma = 1 the relative velocities and hence `vremax` are ~3× larger, so the per-cell
attempt count is ~8× the Ma = 0.3 value. `run_adaptive` grows it for you and reports each growth,
but starting too low costs a recompile per growth.

### That run was done, and the gate is still not established — for a third reason

| | Re = 61, single mode, npc_eff 3200 | Re = 204, broadband, npc_eff 1600 |
|---|---|---|
| `E(0)` / floor | 4.9 | **885** |
| clean range above the band | k = 3..5 | **k = 5..19** |
| slope | not quotable | **−2.87 ± 0.65** |
| vs Burgers −2 | — | **1.3σ — consistent, but not a test** |

Both fixes worked. The floor is now four orders of magnitude below the signal and irrelevant. What
is left cannot be fixed with particles at all: **a 1-D shell holds exactly two modes, `k` and
`−k`, at every `k`** — one complex amplitude — so shell-averaging averages nothing and `E(k)`
carries the full scatter of a single realisation. Theory says `std(log E) = π/√6 = 1.28`; this run
measures **1.00**. A 128² grid has 28 modes in shell 5 and a 32³ grid has 350, so 2-D and 3-D do
not have this problem.

So `--ensemble` is the **wrong ensemble here**: it averages fields across noise seeds of the *same*
flow. Use `--ic-ensemble R`, which varies the IC phases so each realisation is a *different* flow
and averages their spectra — error `∝ 1/√R`, unit-tested against synthetic spectra.

**Always read the error bar.** Sub-windows of this run give −1.69, −3.07 and −2.87, which looks
like a real disagreement and is entirely explained by uncertainties of ±0.65 to ±1.11.

**What a real 1-D gate costs.** `R = 11` reaches ±0.20 — but `k = 19` is only 1.8× below the
viscous cutoff, so it would still be contaminated. A clean Burgers range needs the window a decade
below `k_d`, so `Re ≳ 1000`, so `--side 5000` at Ma 1. Minutes on an H200; ~20 h per realisation
on a laptop 3050 Ti. Not attempted here, and not claimed. Reaching the same Re by box size instead needs `--side 3200`, ~10× the
work. (Ma = 1 heats the gas, which raises λ and lowers the realised Re; the run reports what it
actually got.)

### Two rules for the fit window

`validate.clean_window` enforces both, and each one changed a headline number on this run:

- **contiguous from `k = 1`** — taking every shell that passes `E > 10 × noise` lets the lucky
  ones survive while their neighbours drop out. The scattered selection admitted `k = 7, 9, 10`
  while skipping 6 and 8 and reported **−3.06**; contiguous `k = 1..5` gives **−1.76**.
- **above the spectral peak** — an inertial range lies below the energy-containing scales by
  definition. The peak is at `k = 2`, so the window is `k = 3..5`: three shells, and the correct
  verdict of *no measurable range* instead of a plausible-looking "−1.76, expected −2".

All three failures in this section have the same shape: **a window chosen using the data being
fitted returns a slope whatever the data is.**

## What is reachable, and what is not

Cells must resolve the mean free path, so cell count goes as `(L/λ)^D`, while
`Re ~ 2 Ma √γ (L/λ)`. **Reynolds number costs Re^D in cells.**

```bash
python3 -m examples.turbulence --reach     # sizing table
```

| | D=1 | D=2 | D=3 |
|---|---|---|---|
| 1 × H200 (113 GiB) | Re ~ 5e6 | Re ~ 760 | Re ~ 41 |
| 4 × H200 (450 GiB) | Re ~ 2e7 | **Re ~ 1500** | Re ~ 66 |

So **2-D at Re ~ 10³ is real turbulence and is reachable on your node**, and the exact NS
solution applies there. The 3-D Re = 1600 benchmark is *not*: Gallis et al. ran it on Trinity at
~10¹² particles. The 3-D case here is therefore validated against DNS at the Reynolds number
actually run, and the driver prints that number rather than the benchmark's.

## Production recipes for a 4 × H200 node

Start with a single-device run to confirm `A_max` has settled and no attempts are clipped, then
scale up. `--stride` matters: a full `CellMoments` per step is `steps × C × 18 × 8` bytes, which
is terabytes at these cell counts.

```bash
# 2-D, Re ~ 1500, the viscosity gate at scale
python3 -m examples.turbulence --dim 2 --side 12288 --npc 50 --efolds 3 --stride 64 \
    --frame-every 4 --movie-dir out/d2 --save out/d2

# 3-D, three mid-plane cut images per frame for the animation
python3 -m examples.turbulence --dim 3 --side 512 --npc 50 --n-eddy 8 --stride 32 \
    --frame-every 2 --movie-dir out/d3 --save out/d3

# 1-D spectrum, cheap and with excellent statistics
python3 -m examples.turbulence --dim 1 --side 4194304 --npc 50 --n-eddy 3 --spectrum
```

### Frames and the animation

`--frame-every N` writes a PNG every N *kept snapshots* (so the wall-clock spacing is
`stride × frame_every × dt`). In 3-D the default is the three mid-planes `x`, `y`, `z`, one file
each, named `field_<axis><index>_<step>.png`. Images are written host-side, outside the compiled
step, so they cost only the frames requested. Colour limits are fixed across frames — an
animation whose colour scale moves under it cannot be read.

```bash
ffmpeg -framerate 20 -pattern_type glob -i 'out/d3/field_z*.png' -vf scale=900:-1 d3.gif
```

Arbitrary cut planes are available through `jax_dsmc.viz.Plane(axis, index=..., fraction=...)`
and passed as `--planes` equivalents in the API (`case.write_frames(..., planes=...)`).
