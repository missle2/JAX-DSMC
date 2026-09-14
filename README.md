# JAX-DSMC

> Adjoint implementation update: see [corrections, P3 methods, and validation limits](docs/ADJOINT_CORRECTIONS.md). That document supersedes historical estimator-validity claims below.

Differentiable Direct Simulation Monte Carlo in JAX, built so that a **discrete adjoint** for the
inhomogeneous Boltzmann equation falls out of the forward solve rather than requiring a second
implementation.

The central bet: the adjoint DSMC system of Caflisch–Silantyev–Yang *is* reverse-mode AD of the
forward map with the random decisions frozen. If every stage is a pure function of
`(state, params, frozen random draws)` with static shapes, `jax.grad` reproduces the published
algorithms — including the score terms needed for a general (non-Maxwell) collision kernel.

## Status

**Phase 0 complete and verified. M1 passes its gate; M2 passes both of its *physics* gates.
M6's adjoint is built, its oracles pass, and its pathwise gradients are verified against finite
differences in 1D, 2D and 3D.** 421 tests, 6 skipped (multi-device); no approved code yet.

M2's register gap **U5** (Oracle 3 discriminates the intra-step ordering) is now **closed**: `tests/test_adjoint_oracle.py` hand-codes the 1-D two-step recursion with `B_k` taken analytically from [P1] eq. `AB_def`, and the ordering discriminator is exactly diagonal — our `move -> collide` loop matches `beta_k = B_k beta_{k+1} + dt P* alpha_{k+1}` to 1.3e-16 and misses [P3]'s inside-`B_k` form by 9.8e-2, with both swapping when the loop is reordered.

| Phase | Deliverable | State |
|---|---|---|
| 0 | Pseudocode specification + structural verification | **done**, revision 3, unapproved |
| 1 | Foundations: `safe`, `types`, `rng`, `grid`, `sampling`, `sort` | **done** |
| 2 | **M1** — 0D BKW relaxation, collision operator alone | **passing** — see below |
| 3 | **M2** — periodic transport, shocks, turbulence | **physics gates pass in 1D, 2D and 3D**; gap U5 **closed** |
| — | Multi-device domain decomposition + scaling harness | **built and tested**, unbenchmarked on real multi-GPU |
| 4–5 | M4 Couette, M5 vehicle-like flow | not started |
| 6 | **M6** — pathwise + score adjoint | **oracles pass; pathwise gradients verified in 1D/2D/3D**; gaps U2, U10 open |
| 7–8 | M7 thermal-wall adjoint, M8 inflow ensemble adjoint | not started |

## Setup

```bash
source ~/activate_jax_env.sh        # JAX 0.6.2 + CUDA, x64, numpy/scipy/matplotlib/h5py/pytest
```

The **system Python is not usable** — its jaxlib fails to load cuFFT and silently falls back to
CPU, which makes any performance measurement meaningless.

## Where to start reading

Read in this order. `docs/PSEUDOCODE.md` is the authoritative contract; code that disagrees with
it is wrong unless the spec has been updated first.

1. **`docs/HANDOFF.md`** — state of play, and the traps that will bite you. Read this first.
2. **`docs/PSEUDOCODE.md`** — the specification. §0 (conventions), §1 (RNG contract) and §19
   (unverified claims) before anything else.
3. **`docs/PLAN.md`** — why the project is shaped this way; milestone ladder; deferred decisions.
4. `examples/README.md` — the runnable milestone gates, how to run them, and their results.
   Then `examples/shock/README.md` (the Bird validation and the mean-free-path trap),
   `examples/turbulence/README.md` (the viscosity gate) and `examples/scaling/README.md`.
5. `scratch/README.md` — the numerical probes that settled each contested design decision.

## Running the M1 BKW gate

The milestone-1 acceptance test: relax a 0-D Maxwell-molecule gas from the Bobylev–Krook–Wu
initial condition and compare against the exact solution. The gate quantity is the normalised
fourth-moment excess

```
E(t) = 1 - <|v|^4> / (15 v_th^4) = (1 - K0)^2 exp(-nu t / 3)
```

a pure exponential — dimensionless, and independent of temperature, mass and number density — so
it tests both the amplitude and the decay rate.

```bash
# one case, 16 seeds, with the relaxation plot
python3 -m examples.bkw --n-part 40000 --seeds 16 --plot

# the convergence study: sd of the fitted rate vs N, and the sqrt(N) plot
python3 -m examples.bkw --sweep-n 10000 20000 40000 80000 160000 \
        --seeds 24 --plot-convergence
```

Useful flags: `--dt-nu` (timestep in mean collision times), `--seed0`, `--verbose` (per-seed
rates), `--cells-per-side` (see below), `--n-steps`.

**Use at least 16 seeds.** The fitted decay rate is itself a random variable with a wide spread
(sd ≈ 13% at `N = 1e4`), so a handful of seeds will show a convincing-looking trend that is pure
noise — this happened three separate times during development, at 1, 5 and 8 seeds. The driver
prints a warning below 8. Expected result at 24 seeds:

| N | bias in rate (OLS half) | sd |
|---|---|---|
| 40,000 | +1.40 ± 0.93 % | 0.0152 |
| 640,000 | −0.57 ± 0.19 % | 0.0032 |
| 2,560,000 | **−0.08 ± 0.12 %** | 0.0019 |

with **`sd ~ N^(-0.500)`**, exactly the `N^(-1/2)` that independent Monte Carlo sampling predicts, and
zero majorant violations and zero clipped attempts throughout. Three independent rate estimators
are reported (`ols_full`, `ols_half`, `ratio`) because they fail differently: `E` decays while its
absolute noise stays fixed, so `log` of a small noisy value biases an OLS fit. Agreement between
them is the evidence that a deviation is physical.

**The cell count is derived from `--particles-per-cell` (default 1000), not pinned.** This is the
main speed knob. `A_max` is *per cell*, so pinning the cell count makes `npc` — and with it the
**sequential** attempt-slot scan — grow with N, while the **parallel** cell axis stays fixed.
Holding `npc` constant instead keeps `A_max` flat and puts all the growth on the parallel axis.
Same physics (the gas is homogeneous, and total attempts are ~N either way):

| N | cells | A_max | run | µs/particle |
|---|---|---|---|---|
| 640,000 | 729 | 171 | **0.68 s** | 1.06 |
| 10,240,000 | 10,648 | 188 | **10.6 s** | 1.04 |

against 26.4 s at N=640,000 with cells pinned at 8 — **39× slower**. 10⁷ particles runs in ~10 s
on a 4 GB laptop GPU. Use `--cells-per-side` to override explicitly.

**Below ~300k particles, run on the CPU.** The GPU is launch-latency bound there and the CPU is
faster outright; above ~640k the GPU wins by 3–4×.

```bash
JAX_PLATFORMS=cpu python3 -m examples.bkw --n-part 40000 --seeds 24
```

The fitted rate is identical on both backends to every printed digit — the RNG contract is
device-independent.

**Run from the repository root** (the examples are a top-level `examples/` package, and the
figures are written to `examples/figures/`).

Compiled programs are cached to `examples/.jax_cache/`, so the second run of any problem size
skips compilation — measured 4.0 s → 0.8 s. Disable with `--no-cache`.

## Milestone 2: shocks and transport

Two independent gates, both against external references rather than our own prior output.

### Bird's normal shock (§12.11) — the kinetic gate

```bash
python3 -m examples.shock --dims 1 2 3 --plot     # sine steepening, cross-dimensional
python3 -m examples.shock.normal                  # Bird normal-shock structure
```

The maximum-slope thickness `L = (ρ₂−ρ₁)/max|dρ/dx|` is the one quantity an Euler/Riemann
comparison cannot reach — an inviscid solution has no intrinsic thickness.

| D | ρ₂/ρ₁ | T₂/T₁ | shock speed | **L/λ₁ vs Bird 6.2** | **L/λ_HS vs Alsmeyer 4.8** |
|---|---|---|---|---|---|
| 1 | 0.59 % | 0.13 % | 0.07 % | 5.98 | 4.53 |
| 2 | 0.18 % | 0.23 % | 0.22 % | 6.12 | 4.64 |
| 3 | 0.37 % | 0.56 % | 0.30 % | **6.18 (0.3 %)** | **4.68 (2.4 %)** |

Agreeing with Bird *and* Alsmeyer simultaneously is the real test — they are mutually consistent
only if the mean-free-path conversion is right. **That conversion caused a 24 % false discrepancy
that took a day to find**; three different "mean free paths" appear in this problem and differ by
up to 60 %. See `PSEUDOCODE.md` §21.1 before touching the normalisation.

### Taylor–Green viscosity — the strongest quantitative gate in the project

```bash
python3 -m examples.turbulence --dim 2 --side 128 --npc 100 --efolds 2.5
```

The 2-D Taylor–Green vortex is an **exact** Navier–Stokes solution, decaying as
`E(t) = E₀ exp(−4νk²t)`, so the fitted rate measures the viscosity directly against
Chapman–Enskog with no free parameter, no smoothing and no fitted slope:

**ν = 3.2036e−01 vs 3.1915e−01 m²/s — +0.38 %, R² = 0.9994.**

Run it after any change to `collide/`. It is what ruled the collision model out as the cause of
the shock discrepancy above.

## The adjoint (M6)

The architectural bet of the whole project is that [P1] Algorithm 2 and [P2]
Algorithm 3 **are** reverse-mode AD of the forward solve plus one DiCE magic box, so
that no hand-written backward pass is needed. `jax_dsmc/adjoint.py` is four lines of
construction; `tests/test_adjoint_oracle.py` makes the bet falsifiable by hand-coding
both papers' recursions and asserting `jax.grad` agrees.

```bash
pytest tests/test_adjoint_oracle.py -q        # [P1] Alg 2 and [P2] Alg 3, exactly
pytest tests/test_adjoint.py -q               # gradients through the production solve
python3 -m examples.shock_adjoint             # AD vs finite differences, D = 1, 2, 3
```

| check | result |
|---|---|
| [P1] Algorithm 2 vs `jax.grad`, Maxwell molecules | exact, rtol 1e-11 |
| [P2] Algorithm 3 incl. all three `eta` branches | exact, rtol 1e-10 |
| dropping the reject-branch score | asserted **detectable** |
| pathwise gradient vs flip-free FD, `D = 1, 2, 3` | 3e-11 … 8e-10 for `T₁`, `u_stream` |

**Two traps worth knowing before touching this.** [P1]'s matrix `A` is *not* the
Jacobian of our collision map — SPARTA draws the deflection in the `v_r` frame, so
our pair Jacobian is rank 6 where `A` is rank 4, even at Maxwell molecules
(`PSEUDOCODE.md` §17.1.1). And a finite-difference check is meaningless unless no
frozen decision flips: the guard for that must include the boundary fold index and
`scatter`'s degenerate-frame branch, and must not weight any decision by zero — §13.4.1
records how both holes were found.

## Multi-device

```bash
python3 -m examples.scaling --devices 1 2 4 --dim 3 --side 96

# verify without a multi-GPU node (correctness only, not speed)
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
    python3 -m examples.scaling --devices 1 2 4 --dim 2 --side 32 --steps 4
```

`jax_dsmc/decomp.py` gives each device an x-slab of cells and exchanges only slab-crossers.
`jax_dsmc/parallel.py` is kept as the **measured negative result**: sharding the particle axis does
not parallelise DSMC, because the global `argsort` replicates the sort table and every device then
redundantly runs the whole collision. Spec §20.

Reynolds number costs `Re^D` in cells, so 4 × H200 reaches `Re ~ 1500` in 2-D — where the exact NS
solution applies — but only `Re ~ 66` in 3-D. `python3 -m examples.turbulence --reach` prints the
table.

## Verify

```bash
pytest -q                                            # 421 unit tests, ~4 min
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
    pytest tests/test_decomp.py -q                   # multi-device, needs >= 2 devices
python3 docs/phase0_verification.py                  # 55 structural checks, then open gaps
python3 docs/phase0_verification.py --milestone M1   # exit 1 if an M1 gap is still open
ruff check . && ruff format --check .                # lint + format gates
```

## References

Two production DSMC codes are used as implementation references, and **where they disagree is the
most useful part** — each divergence forced an explicit decision, recorded in `PSEUDOCODE.md` §18.
**Every place this code departs from SPARTA is enumerated in §18.9**, with the reason and the
evidence for each.

- SPARTA — `~/sparta/sparta/src/`, cited as `file.cpp:lines`
- JAX-FLUIDS — `~/jax_fluids/JAXFLUIDS/`, a mature JAX continuum CFD solver. Different equations,
  but it independently confirms the host-side time-loop pattern `simulate.run_adaptive` relies on
  (`simulation_manager.py:325`). See `PSEUDOCODE.md` §18.10.
- dsmcFoam+ / hyStrath — `~/OpenFOAM-v1706/`, cited as `[DF]`

Those two are **machine-local checkouts, not vendored here** — the citations are `grep` starting
points, and line numbers drift with upstream commits. Everything else the project needs is in
this repository: the three adjoint papers are in `papers_reference/`, and the specification is
self-contained.

Adjoint theory: three papers in `papers_reference/`, cited as `[P1]` (2009.01363, Maxwell
molecules), `[P2]` (2207.11579, general kernel), `[P3]` (2603.20946, spatially inhomogeneous).
