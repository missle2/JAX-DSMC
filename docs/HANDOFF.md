# Handoff — state of play

Written for someone picking this up cold. It covers what exists, what is in flight, and — mostly
— **the traps**, because this project has already produced two defects that would have shipped
silently wrong physics and survived a careful reading of the spec.

---

## 1. Where things stand

**Phase 0 (specification) is complete and structurally verified.** `docs/PSEUDOCODE.md` is at
revision 3, ~2150 lines, and is the authoritative contract. It has *not* been formally approved,
so treat open questions in it as genuinely open.

**Milestone 1 implementation: the forward collision path is built and tested.** The waves below
existed for a reason -- `state.py` is a hard dependency of nearly everything, and a first attempt
that launched all four scopes simultaneously left three agents coding against a module that did
not exist.

| Scope | Files | State |
|---|---|---|
| **Gates U1 + U3** | `tests/test_gaps.py`, `scratch/gate_U*.py` | **CLOSED 2026-09-04** — see §1.1 |
| Foundations | `constants.py`, `safe.py`, `state.py` | done |
| RNG contract | `rng.py` | done |
| Mesh & sampling | `grid.py`, `sampling.py`, `sort.py`, `initialize.py` | done |
| Collision kernel | `collide/vss.py`, `collide/ntc.py` | done |
| Transport | `move.py`, `boundary.py` (periodic/specular/outflow) | done |
| Time loop | `simulate.py` | done |
| **M1 gate** | `examples/bkw.py` | **runs and passes** — see §1.2 |
| BKW analytic, output | `bkw.py`, `sample.py`, `viz.py`, `output.py` | agent-written, **not second-passed** |

**344 tests pass** in ~4 minutes.

`ruff check` and `ruff format --check` both pass across `jax_dsmc/` and `tests/`.

**What is NOT done.** The M1 acceptance gate itself — a 0D relaxation at `omega = 1` compared
against the exact Bobylev–Krook–Wu solution — has not been run end to end. `bkw.py` and
`test_bkw.py` exist but were written by an agent that was cut off mid-task, and `simulate.py`
(the time loop, §12) does not exist at all, so nothing yet steps the solver forward more than one
collision step. That is the next piece of work.

### 1.1 The two gates, and what they changed

Both closed by measurement on 2026-09-04. **U3 changed the design**, which is exactly why the
gates were run before the code they gate.

**U1 — is `EPS_REL_VR2 = 1e-8` right?** Yes, keep it. Two agent passes recommended raising it to
`1e-6` on a peak-ratio metric `D = sup/rms`. That was measured and **rejected**: `safe_pow` falls
back to a *constant* below the clamp, so both value and gradient are exactly zero there, and every
configuration the `D` argument named (duplicated parcel, zero-`T` IC, the inactive scratch-row
slots) sits below the clamp and injects nothing. Over 40e6 equilibrium pairs neither value clamps
even once, and the gradient standard error is identical to 4 significant figures. The only measured
differences favour `1e-8`. Full evidence: spec §17.4.

**U3 — can a 2-5x majorant margin be held?** Yes, **but not with SPARTA's seed.** SPARTA's
`vremax = 2*cxs*vscale` plus a stale-max ratchet measured a `1.0e-2` violation rate on a
counter-streaming IC with a minimum margin of `0.62` — `Sigma` sitting *below* the true maximum,
and every violated pair contributes a zero score and biases the gradient. The adopted rule is an
**a-priori thermal majorant**: `q` is monotone in `|vr|`, so its supremum over the reachable
interval is bounded by the cell temperature, which is known at the *start* of the step and
therefore cannot lag a transient. Zero violations across all six IC x transient cases, minimum
margin `2.15`. It costs attempts but **not physics** — collision count is `Sigma`-invariant,
measured flat over a 20x sweep. Full evidence: spec §17.5; the rule itself is §7.2.3, implemented
as `thermal_majorant` / `vremax_init` in `collide/ntc.py`.

**Consequence for anyone touching `ntc.py`:** do not "simplify" `Sigma` back to
`cstate.vremax[:, 0, 0]`. The thermal floor is load-bearing, and removing it fails no test
loudly — it just raises the violation count that `min_majorant_margin` reports.

---

### 1.2 Milestone 1 passes, and how it was verified

`python -m examples.bkw` relaxes a 0D Maxwell-molecule gas from the BKW initial condition
and compares the normalised fourth-moment excess against the exact solution. The gate quantity is
`E(t) = 1 - <|v|^4>/(15 vth^4) = (1-K0)^2 exp(-nu t/3)` — a pure exponential, dimensionless, and
independent of `temp`, `mass` and `nrho`, so it tests both the amplitude and the rate.

**Sample size is the trap here, and it caught me three times.** The gate quantity is a fitted
decay rate — itself a random variable with a wide spread — so any small number of seeds produces
a confident-looking wrong answer:

| seeds | what it appeared to show | what it actually was |
|---|---|---|
| 1 | 4.13% / 7.50% / 2.17% error at N = 20k/45k/80k — "not converging" | noise; sd is ~13% at these N |
| 5 | bias grows to **+11%** at N = 5k, and a monotone dt trend | noise |
| 8 | **+5.78%** bias at N = 40k (z ≈ 3.4), dt trend −4.1% → +2.6% | noise |
| 16 | +3.39 ± 1.28%, and the dt trend **gone** (−1.21, −2.16, +0.25) | consistent with zero |

At 16 seeds, with three independent rate estimators applied to the same trajectories (OLS over the
full trajectory; OLS over the first half only, where the relative noise is small; and a two-point
log-ratio with no fit at all):

| N (dt·ν = 0.15) | OLS full | OLS half | log-ratio |
|---|---|---|---|
| 10,000 | −2.31 ± 3.24 | −0.36 ± 2.10 | +1.28 ± 2.36 |
| 20,000 | −2.16 ± 2.11 | −4.19 ± 1.23 | −3.73 ± 1.77 |
| 40,000 | +3.39 ± 1.28 | +1.49 ± 1.16 | +1.60 ± 1.31 |

(bias in % of the exact rate 1/3; ± is the standard error of the mean over 16 seeds.)

**Verdict: no systematic bias is detectable.** Every deviation is within ~±4% and mostly within
two standard errors of zero, the sign is inconsistent between N values, and the three estimators
agree — so the residual is Monte Carlo scatter, not a defect. Zero majorant violations and zero
clipped attempts throughout. Resolving a genuine few-percent effect would need ~100 seeds; if
someone wants to tighten this, that is the cost.

Reproduce with `scratch/m1_convergence.py` (sweep) and `scratch/m1_rate_estimator.py` (estimator
cross-check). **Do not quote a rate from a handful of seeds** — this file's own history is the
argument.

### 1.2.1 Two traps when running it

**The `types.py` shadowing trap is fixed** — the module is now `jax_dsmc/state.py`. It used to be
`types.py`, which shadowed the *standard library's* `types` module whenever the working directory
was `jax_dsmc/` (Python puts the cwd first on `sys.path`), and since `functools`, `dataclasses` and
`enum` all import stdlib `types`, the interpreter failed to start with a message pointing nowhere
near the cause. **Rule: no module here may take the name of a stdlib module.**

**Scale the cell count with N, do not pin it.** `A_max` is *per cell*, so a fixed cell count makes
the sequential scan grow with N while the parallel axis stays fixed. `--particles-per-cell`
(default 1000) derives the cell count so `npc` stays constant. See §1.3.

### 1.4 M2: periodic transport, and the `A_max` question settled

**`A_max` cannot be guessed, and no longer has to be.** It is the trip count of the attempt-slot
`scan`, so it is baked into the compiled program — but the right value is a property of the flow.
Every a-priori rule tried failed: the cold uniform count is 5.4x too small at the shock time; the
majorant inflation U4 describes accounts for only 1.10x of that (compression contributes ~4.5x);
and bounding compression by Rankine–Hugoniot at 4x is simply **wrong** for a converging wave, which
measured a density ratio of **11**.

`simulate.run_adaptive` resolves it by measuring. Restarting is nearly free — the state is just
arrays, so continuing after a recompile costs only the compile — so the run proceeds in chunks,
reads `attempts_clipped` on the host, and on overflow grows `A_max` and re-runs **that chunk only**.
From a deliberately absurd `A_max = 64` it converged in two resizes and finished in **41.9 s against
48.6 s for a single over-provisioned run**: measuring beats guessing high. `run(..., step0=)` keeps
the RNG stream identical, so `x` and `v` are bitwise equal to an unchunked run (`logh_acc` agrees to
6e-14 — float reassociation, not a decision difference).

Neither reference code helps here: SPARTA has no `A_max` at all (a CPU `for` loop over `nattempt`),
and its `reset_vremax()` is only called from `init()`. Its MCF option (`n ~ Poisson(f)`) is a
different *count distribution*, not a different capacity rule, and its tail makes `A_max` harder.
Spec §8.5.1.

**A reformulation that would remove the bound was verified and deliberately parked.** Fixing `A` and
folding the count into the acceptance probability (`p = 0.5 npc(npc-1) dt fnum q / (A V)`) is exactly
NTC with a derived majorant; it matches collision counts to 0.4 %, needs 2.3x fewer slots, and
carries the `dt`/`fnum`/`V` gradient through `log p` alone — which would delete §8.1's stochastic
rounding entirely. It is **not adopted**: it changes the core acceptance probability, and the
verification is single-group, so multi-species, Larsen–Borgnakke and chemistry would each need the
equivalence re-derived. Spec §8.5.2; probes in `scratch/probe_fixed_attempts*.py`.

**A real spec defect M2 exposed:** `cell_volume = prod(dx)` is a *length* at `D=1` and an *area* at
`D=2`. `GridSpec.transverse` now carries the suppressed extents. It does not cancel — rescaling
`fnum` to fix the collision rate leaves `nrho` wrong by the square. Spec §4.

**Output is fields, not particles.** A field PNG is ~50 KB and scales with cell count; one
already-subsampled particle frame at `N=1e6` is 1.7 MB and scales with `N`. `viz.FieldMovie` writes
three PNGs per frame in 3-D (`x=0`, `y=0`, `z=0` faces). Spec §10.1.1.

### 1.3 Performance: the parallel axis is cells, not attempt slots

Full numbers in spec §8.6. Four things worth knowing before touching `collide`:

**Do not draw random numbers inside the attempt-slot scan.** §1 rule 3 requires one batched
`(A_max, C)` draw per stage, consumed as the scan's `xs`. An early version drew per slot; hoisting
it cut compile 17.0s → 2.4s, halved per-slot cost, and took the **whole test suite from 26 minutes
to 4**. `tests/test_perf.py` asserts structurally that no RNG primitive appears in the scan body.

**Scale cells with N.** `A_max` is per cell, so pinning the cell count grows the *sequential* axis
and starves the *parallel* one. Measured at N=640,000: 26.4 s pinned versus **0.68 s** with `npc`
held at ~1000, and 10⁷ particles goes from unreachable to 10.6 s on a 4 GB laptop GPU. Cost per
particle falls from 36 µs at N=10⁴ to ~1 µs above N≈6×10⁵ and then stays flat.

**Below ~3×10⁵ particles, use `JAX_PLATFORMS=cpu`** — the GPU is purely launch-latency bound there
and the CPU is up to 4.5× faster. Above ~6×10⁵ the GPU wins by 3–4×. Rates agree to every digit on
both backends.

**Two audited non-issues**, recorded so nobody re-investigates them: the scratch-row `concatenate`
is fused by XLA and costs nothing; and the scan carry *is* updated in place for any cell count
≥ 8 — the O(N)-per-slot copy exists only at `C == 1`, which no production configuration uses.
`fori_loop`, `.at[].add` and a compact `(C,K,3)` workspace were all measured and none beats plain
`scan`.

**Benchmarking discipline:** never run these benchmarks alongside another GPU job. Doing so
produced a phantom 36× "cliff" that does not exist.

## 2. The two defects that nearly shipped

Both were in the spec, both read as correct, and **both were found only by running a number.**
They are the reason this project's working method is "measure it" rather than "reason about it".

**The collision-count gradients were exactly zero** (§8.1). `n = floor(f)` froze the path from
`dt`, `fnum` and `cell_volume` into the collision rate, and the spec asserted that "the score
terms carry it". They don't — `log h` depends on velocities and VSS parameters and on none of
those three. Measured `jax.grad = [0.0, 0.0, 0.0]` against a healthy `‖∂J/∂v‖ = 9.5`. Fixed with
stochastic rounding plus a `log P(n|f)` score term.

**Empty cells corrupted real collisions** (§8.4.1). 176 of 300 runs violated momentum
conservation. `jax.random.randint(key, shape, 0, 0)` returns 0 rather than erroring, so empty
cells indexed into a *neighbouring* cell's particles, and the second scatter wrote stale data
over a legitimate collision. Fixed with a clamp and a scratch row at index `N`.

**What they have in common, and why it matters more than the fixes:** neither raised an error,
produced a `NaN`, or tripped any diagnostic. They are the *silent-failure* class. The spec's §19
register marks three outstanding claims (U4, U6, U7) as having the same shape, and they are
flagged blocking for that reason.

There is also a third, subtler failure worth knowing: **the finite-difference cross-check could
not have caught the first defect.** At fixed randomness `J` is a step function, so FD read
`0.000000e+00` for every perturbation that didn't flip `n` — FD and AD agreed, on a broken
gradient. §13.4 now mandates two explicit FD modes.

---

## 3. Invariants you must not break

These are what make `jax.grad` reproduce the published adjoint. Violating any one produces wrong
gradients rather than an error.

1. **The RNG contract (§1).** Keys derive only from `(base_key, step, STAGE_ID)` via `fold_in`,
   never from traced state, and are recomputed rather than carried. This is what gives common
   random numbers for FD checks and bit-identical replay under `jax.checkpoint`.
2. **Frozen decisions (§0.5).** Cell membership, sort permutation, attempt count, pair choice,
   accept/reject, face crossing. These are integers and carry no gradient *by design* — that is
   what the papers specify. Their sensitivity travels via score terms, not through the decision.
3. **Static shapes.** No `while_loop` with a data-dependent trip count anywhere. Every loop is a
   `scan`/`fori_loop` with a static bound: `n_steps`, `A_max`, `K_bounce`. Data-dependent counts
   use static capacity plus an `alive` mask.
4. **No nested function definitions (§0.6).** JAX pushes you toward closures for `scan` bodies;
   use module-level functions bound with `functools.partial` instead. Enforced by review, and it
   is why `_collide_slot` is written the way it is.
5. **The mesh interface is narrow (§4).** Only `cell_of`, `cell_volume`, `cell_bounds`,
   `apply_bc`. M5 replaces everything behind those four names with cut/split cells; if cell-index
   arithmetic leaks into `collide` or `sample`, that rewrite breaks them.
6. **f64 throughout.** Energy-conservation tests assert ~1e-14; f32 cannot deliver it.

---

## 4. The unverified-claims register

`PSEUDOCODE.md` §19 lists nine assertions that are **reasoned but not measured**, each pinned to
the milestone that makes it live. It is executable:

```bash
python3 docs/phase0_verification.py --milestone M3   # exit 1 if any M1-M3 gap is open
```

The gate is cumulative. A row's gate is part of its milestone's definition of done — checked
*before* that deliverable is written. A row may only be deleted by moving its evidence into §17.

`U4`, `U6`, `U7` are tagged `[SILENT-FAILURE RISK]` and are blocking, not advisory.

---

## 5. How decisions were made

Worth knowing, because it explains why the spec is unusually long and why §17/§18/§19 exist.

- **Where the two reference codes disagree, each divergence forced an explicit decision.** §18 is
  the ledger: which we followed, and why. The most consequential is the NTC majorant (§18.1),
  where dsmcFoam+ is right and SPARTA is not — but adopting it carries a cost (§7.2.1) that was
  only discovered by measuring afterwards.
- **Claims were measured, not argued.** §17 is the table of what was actually verified;
  `scratch/` holds the probes, indexed by which decision each settled.
- **Several confident conclusions were reversed by measurement**, including some of the author's
  own: gradient variance *falls* with the majorant rather than rising; a never-decaying `vremax`
  *improves* the gradient; `eps=1e-30` is harmless where it was flagged and dangerous elsewhere;
  FD-vs-AD disagreement is an artifact of step size, not a law. If you find yourself reasoning
  confidently about a numerical property here, write the probe.

---

## 6. Process traps already hit

- **Batch edits that abort before writing lose everything applied earlier in the batch.** This
  silently dropped four edits once, and left a §12.1 that three cross-references pointed at but
  which did not exist. Write per edit, and sweep afterwards.
- **Benchmarks at small problem sizes are noise.** The same configuration gave ratios of 1.02,
  1.40, 1.69 and once 0.14. Any performance claim needs a repeat count and a spread.
- **A gap's gate runs BEFORE the deliverable it gates, not alongside it.** This was violated
  once already: U1 was bundled into the agent implementing `safe.py`, and U3 into the agent
  implementing `ntc.py` — the very code whose design U3's answer determines. Gates that could
  change a design decision must be separate, standalone work that completes first. The §19 rule
  says this; it is easy to skip when it feels like the same subject matter.
- **Fanning out parallel agents needs the dependency order respected.** Four simultaneous scopes
  meant three of them coded against a `state.py` that did not exist yet. Sequence by dependency,
  and keep the parallel width modest — a burst of four long-running agents also hit a session
  rate limit and lost most of its work.
- **Verify that a described fix is actually wired.** Three fixes were correctly explained in prose
  while the pseudocode still contained the defect — `logP` was computed and dropped on the floor.
  A named check per fix caught it; spot-checking did not.

---

## 7. What to do next

1. Land M1: get the four scopes integrated, `pytest -q` green, and the BKW gate passing against
   the analytic solution.
2. Close **U1** and **U3** before calling M1 done (`--milestone M1` enforces this).
3. Build **Oracle 3** (§13.3) before M2. The 0D oracles provably cannot detect the intra-step
   ordering, so without it the ordering question stays untested until M4.
4. Re-read §19 at the start of every milestone.


---

## 1.5 M2 physics gates pass: Bird normal shock, all dimensions (2026-09-05)

**Not** a full M2 close: gap **U5** (Oracle 3 discriminates the intra-step ordering) remains open
and needs M6's adjoint oracles. The two gates below are the physics half.

**Result.** `L/lambda_1 = 5.98 / 6.12 / 6.18` at D = 1/2/3 against Bird §12.11's **6.2**, and
`4.53 / 4.64 / 4.68` against Alsmeyer's experimental **4.8** — both simultaneously. Jump
conditions 0.1-0.7 %, shock speed 0.3 %, temperature leads density by 2.07-2.10 lambda_1 in every
dimension. Resolution-converged over a 3.2x range in cell size (spec §21.3). Figure:
`examples/shock/figures/bird_normal_shock.png`.

**The one thing to know before touching this.** There are three different "mean free paths" in
this problem and they differ by up to 60 %. Bird's eqn (4.52) is
`[(alpha+1)(alpha+2)/(6 alpha)] * lambda_HS(d_ref)` — a function of **alpha only**, no omega.
Getting it wrong cost a day and a 24 % false discrepancy; the wrong factor
(`alpha(5-2w)(7-2w)/[4(alpha+1)(alpha+2)]`) is a *cross-section* ratio that happens to reproduce a
number quoted in Bird's text. Spec §21.1. Reproducing one quoted number is not verification.

**New guards, all of which caught something real:**

- `report()` refuses to present a shock profile when the fitted shock speed misses
  Rankine-Hugoniot by >5 % or the alignment residual exceeds one mean free path. At `npc = 30` the
  tracking collapses and prints a plausible `L/lambda = 20.7` otherwise.
- `run()` refuses to sample after the far-boundary expansion (which travels at `u_p + a_1`,
  *faster* than the shock) reaches the wave.
- `test_vss.py` asserts `<sigma_T v_r>` equals the hard-sphere rate at `T_ref` for six values of
  `omega`. M1 exercises only `omega == 1`; every transport result uses 0.81. Spec §21.4.
- `test_decomp.py` asserts exact particle conservation in D=1,2,3.

## 1.6 Turbulence cases (M2, second batch)

`examples/turbulence/` — one case per dimension, each with a numeric reference:
1-D compressive sinusoid (`E(k) ~ k^-2`), 2-D Taylor-Green (**exact** NS decay), 3-D Taylor-Green
(DNS at matched Re).

**The 2-D case is now the strongest quantitative gate in the project:** it measures the viscosity
against Chapman-Enskog with no free parameter and no fitted slope, and gets **+0.38 %**
(R^2 = 0.9994). It is the check that ruled the collision model out as the cause of the shock
discrepancy above, and it should be run after any change to `collide/`.

**Trap 1 (scalar):** the measured energy carries a *constant* noise floor `1.5 v_th^2 / npc` while
the signal decays exponentially. Fitting without subtracting it biases the rate low, reports a
viscosity too small, and the fit still looks excellent. `validate.energy_history` subtracts it
from the actual per-cell counts. That subtraction is safe here because the signal stays far above
the floor over the fitted window.

**Trap 2 (spectral), and it is the opposite conclusion:** do **not** subtract the floor from
`E(k)`. Near the crossover `E - noise` fluctuates through zero, half the shells survive any
positivity test by chance, and the fitted slope is set by that survivorship — measured: "usable to
k = 507", slope **-0.71** against a true Burgers **-2**, with no diagnostic firing. Lower the
floor instead: raise `npc`, or use `--ensemble M`, which averages M velocity fields (deterministic
ICs, so only the noise differs) and is exactly equivalent to `M x npc` at the same cost. The floor
also **rises** as `k^(D-1)` after shell binning, so it can pass for a cascade.

**Trap 3 — the ensemble was run and the verdict did not change.** 16 seeds, floor down by exactly
16x, and the 1-D gate is still not established, because the floor was never the binding limit. Two
other things were:

* *the initial condition combs `E(k)`.* A single Fourier mode imprints its own symmetry: measured
  `E(7)/E(6) = 11.9` at shells 90x and 7.6x clear of the floor. No shell slope survives a factor-12
  comb. Fix: `--ic broadband` (random phases over `1 <= |k| <= k_max`, solenoidal in 2-D/3-D).
  Taylor-Green stays the default — the viscosity gate needs an exact NS solution.
* *`Re = 61` puts dissipation at `k ~ 10`*, against a peak at `k = 2`. **Check `Re` before the
  floor.** `validate.dissipation_wavenumber` is printed up front. `Re ∝ Ma`, so `--mach 1.0` is the
  cheap knob (`Re ~ 200`) — cheaper than `--side 3200`, which is ~10x the work for the same Re.

**Trap 5 — both fixes were applied, and 1-D still fails, irreducibly.** At `Re = 204` with a
broadband IC the signal stands 885x above the floor over `k = 5..19` and the slope is
**-2.87 +- 0.65**: 1.3 sigma from Burgers -2, so consistent and not a test. The limit is that **a
1-D shell holds exactly two modes at every `k`**, so `E(k)` carries ~100 % scatter whatever the
particle count (measured `std(log E) = 1.00`, theory 1.28). `--ensemble` cannot touch it — that
averages fields of the *same* flow. Use `--ic-ensemble R` (different IC phases, spectra averaged,
error `~1/sqrt(R)`). **Always quote `fit_slope`'s error bar**: sub-windows of that run give -1.69,
-3.07 and -2.87, which is scatter, not disagreement. Spec §22.7.

**Trap 4 (window selection):** `validate.clean_window`. A fit window chosen using the data being
fitted returns a slope whatever the data is. Two rules, each of which moved a headline number on
the ensemble run: the window must be **contiguous** from `k = 1` (scattered "passes the threshold"
selection gave **-3.06**, contiguous gave **-1.76**), and it must lie **above the spectral peak**
(which reduces this run to three shells — the correct answer of no measurable range). Spec §22.5.

## 1.7 Multi-device

`jax_dsmc/decomp.py` — 1-D x-slab domain decomposition, `lax.ppermute` neighbour exchange.
`jax_dsmc/parallel.py` is kept as the **negative result**: array-sharding the particle axis does
not parallelise DSMC, because the global `argsort` replicates `table.order` and every device then
redundantly runs the whole collision (measured: collide 6.8 / 13.0 / 25.2 ms at 1/2/4 devices on a
fixed problem). Spec §20.

Two capacities can overflow silently — per-device particle capacity and the migration buffer — and
both bugs found during development lost or duplicated particles while reporting **zero** overflow.
One of them was invisible in 1-D and appeared only in 2-D. Test by counting, not by comparing
profiles.

Run the scaling harness with `python3 -m examples.scaling --devices 1 2 4`; verify it without a
multi-GPU node using `XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu`.

**Sizing:** Reynolds number costs `Re^D` in cells. 4 x H200 reaches `Re ~ 1500` in 2-D (where the
exact NS solution applies) but only `Re ~ 66` in 3-D. The 3-D `Re = 1600` benchmark is
supercomputer-scale and is not claimed.

## 1.8 M6: the adjoint, and two things that need watching (2026-09-05)

`jax_dsmc/adjoint.py` is 4 real lines of construction (`magic`, `objective`, `surrogate`,
`dice_reward`); everything else in it is entry points and objective functionals. The bet of
§11 holds up: **no hand-written backward pass exists in the production path.**

### What is verified

| check | where | result |
|---|---|---|
| [P1] Algorithm 2, hand-coded, vs `jax.grad` | `tests/test_adjoint_oracle.py` | exact (rtol 1e-11) |
| [P2] Algorithm 3 incl. all three `eta` branches | same | exact (rtol 1e-10), both `phi = v_l^2` and `v_l^4` |
| dropping the reject-branch score is detected | same | asserted to *disagree* |
| pathwise gradient vs flip-free FD, `D = 1, 2, 3` | `tests/test_adjoint.py`, `examples/shock_adjoint` | see the example README |
| score-only parameters are 0 pathwise, nonzero scored | `tests/test_adjoint.py` | the §8.1 regression guard |

### Trap 1: Oracle 1 cannot be run against `vss.scatter`

[P1] eq. `AB_def` assumes `delta sigma = 0`. SPARTA draws the deflection **in the `v_r` frame**,
so our pair Jacobian is **rank 6** where `A` is rank 4, and they differ by 53–81% *at Maxwell
molecules*. Oracle 1 therefore runs [P1]'s own lab-frame map. Neither is wrong — they are
different reparameterisations of the same distribution — but §17.1's "if `A` ever goes full rank
the physics changed" was misleading and is corrected in §17.1.1. Registered as **U10**: nobody
has measured which reparameterisation gives the lower-variance gradient.

### Trap 2: a flip guard is only as good as its weakest weight

Finite-difference checking at frozen randomness is meaningless unless *no discrete decision
flips*, and the guard for that (`Diagnostics.decision_checksum`) had **two holes**, both found by
a failing test rather than by reading:

1. **§13.4.1's list of decisions is incomplete.** It omits the specular **boundary fold index**
   (whose parity flips `v`'s sign, and which is invisible in `cell_of` because both sides of the
   wall fold into cell 0) and `scatter`'s **degenerate-frame branch**.
2. **The checksum weighted index `i` by `mod(i*phi, 1)`, which is exactly 0 at `i = 0`** — so
   every decision in **cell 0** vanished. In a reflected shock that is the cell against the wall.
   The guard passed an FD estimate that was wrong by a factor of three.

Both fixed. The diagnosis route that worked, for next time: scan `J(theta)` on a fine grid to
prove it is a *jump* and not curvature, bisect on `n_steps` to find the step, then diff every
intermediate array of that step.

### Trap 3: `A_max` calibration does not happen automatically inside a gradient

`run_adaptive` grows `A_max` with **host-side Python control flow**, so it cannot appear inside a
differentiated call — `adjoint.terminal_objective` uses plain `run`. The documented pattern is
calibrate first, then differentiate at the settled capacity, and `examples/shock_adjoint.calibrate`
now does it. A hand-picked `A_max = 64` against a peak of 73 ran with `attempts_clipped = 9`
every step: gradients correct for a silently under-collided gas.

### Two concerns raised on review — both now settled by measurement

- ~~**Reverse-pass cost.**~~ **SETTLED: ~4-5x, not 17x** (§23.4). The 17x was a benchmarking
  artifact — measured under `vmap` over 128 keys, where the reverse pass goes memory-bound (a
  512-key batch asks XLA for 4.0 GiB). Un-batched it is **3.1-5.0x** and flat in `N` from 160 to
  64 000. Both structural hypotheses were **refuted by measurement**: the ratio *plateaus* in
  `A_max` rather than growing linearly (so it is not the scan retaining its velocity carry), and
  all four remat policies land in 3.94-4.48 (so it is not `nothing_saveable`). `collide` is
  **90 %** of the forward, so the reverse pass is essentially the VJP of the attempt-slot scan —
  the "AD also transposes the sort and the moments" story is real but worth only ~10 %.

  The comparison to [P1] was not like-for-like either: their `0.58` *excludes* the forward run
  that must happen first, so their gradient costs ~1.6 forwards against our ~5.

  **If it ever needs to be faster**, the lever is a `jax.custom_vjp` on `scatter` exploiting
  `|v_r'| == |v_r|` and the mass-weighted CM decomposition — which is exactly what makes [P1]'s
  stored `6x6` matrix cheap — and Oracle 2 already exists to check it. Not done; at 5x a gradient
  is affordable.
- **Memory is what caps an adjoint run, not time.** Measured from two independent XLA
  allocation failures agreeing to 0.3 %: **~4300 bytes per particle** for one gradient at
  `A_max = 256`, `M = 16` (§23.4.1). That is **~30x the forward solve's per-particle cost**, so
  the 4x H200 node that §20.4 sizes at `~1e9` particles forward-only holds `~1.3e8` for a
  gradient — and one H200 holds `3.3e7`. Plan adjoint-driven optimisation against that number,
  not against the forward capacity. Scales with `A_max`; roughly flat in `M` because the time
  loop is rematerialised.

- **Score-term variance obeys `N^-1/2`, measured.** `probe_variance.py` §A, `D = 2` widened
  transversely so `A_max` stays 256 while `N` and `C` grow together, 128 keys per point,
  `phi = T_y` ([P2]'s objective). Over `N = 4e3 -> 2.56e5` the fitted exponent of `sd ~ N^p` is
  **-0.508 (temp), -0.464 (ux), -0.495 (alpha), -0.499 (diam)** against [P2]'s predicted `-0.5`
  — and that holds for a pathwise parameter, a mixed one, and a pure-score one alike. `diam`'s
  ensemble *mean* is stable to ~10% across the 64x range while its error bar shrinks 8x
  (9.7% -> 1.2% relative). The estimator is behaving as published; the variance is a budget.

  **One earlier reading was wrong and the sweep is what caught it.** `alpha`'s huge *relative*
  error is a near-zero denominator, not a struggling estimator: its mean is consistent with
  **zero** at every `N` (`|z|` = 0.9, 1.3, 0.4, 0.4) while its absolute error bar falls 8x on
  the same law. For this objective `E[dJ/d(alpha)]` is simply weak — `|mean| <~ 30` against a
  per-run `sd = 244` at `N = 2.56e5`. Not to be confused with §23.2's *pathwise* `alpha` check,
  a different objective at frozen randomness, exact to `4e-6`.

  **[P2] §6 does report this, and budgets for it** — a first draft of this section wrongly said
  otherwise. `main.tex:1425` averages **`M_s = 100`** i.i.d. runs for *both* the AD and the FD
  gradient; `main.tex:1431` states the random error as **`O((sqrt(N M_s) |Delta m|)^-1)`** and
  fixes **`Delta m = 0.1`** — a 10% finite-difference step — as "nearly optimal for `N = 1e6`
  Maxwellian particles, `M_s = 100`"; `main.tex:1502` reports the standard deviations growing as
  `O(sqrt(M))` in the step count. So a large FD step plus a big ensemble *is* the published
  protocol, not a workaround.

  Their sampling budget is `N * M_s = 1e8`. The first probe here was `N * M_s = 2000 * 512 =
  1e6` — **a hundredfold less**, which on the `1/sqrt(N M_s)` law alone predicts error bars 10x
  wider. Two further asymmetries: [P2] differentiates only `m = (T_x^0, T_y^0, T_z^0)`, i.e. our
  `temp`, which is the parameter that *does* come out at 0.2% — it never differentiates a
  collision-model constant; and our `phi` is spatially windowed, so most particles carry score
  noise and no signal where the papers' 0-D `phi = v_l^2` is carried by every particle.

  **It is not a correctness problem**: Oracle 2 matches [P2] Algorithm 3 exactly, particle by
  particle, on a shared forward pass. What is open is sample efficiency. `scratch/adjoint/
  probe_variance.py` tests the `1/sqrt(N)` and `sqrt(M)` scalings directly. `1/sqrt(N)` is
  measured and confirmed (above).

- **The noise is mostly *pathwise*, and that settles the baseline question.**
  `probe_split_variance.py` measures `sd(pathwise)`, `sd(score)` and their correlation
  separately: for every parameter with a pathwise route the pathwise half carries **3-5x** more
  noise than the score half, with correlation within `+-0.2` of zero. `alpha` is `1014` against
  `197` — previously an inference, now a measurement. Because the halves are uncorrelated their
  *variances* add, so the score's share of the **variance** — what a baseline removes — is only
  **3.1% (`alpha`), 5.5% (`ux`), 11.9% (`temp`)**, capping a perfect baseline at 1.016-1.065x.

  **Do not read that as explaining §11.4's ~1.2x** — a first draft did, and it is wrong: 1.2x
  *exceeds* the 1.016x ceiling, so they are not the same quantity. §11.4 was measured on a
  windowed `phi` at `D = 1, N = 2000`, a different configuration. What holds is directional:
  wherever a pathwise route exists the score is a minority of the variance, so a baseline is
  limited; the 84x applies only where the score is the *whole* variance (the count score alone,
  and score-only parameters like `diam`). The lever for the dominant term is **more particles**.

- **At [P2]'s own protocol the AD/FD gap is truncation, not noise.** At `N = 2.56e5`,
  `M_s = 100`, `Delta m = 0.1`, the `|z|` between AD and FD is 11 (`temp`), 43.5 (`ux`), 6.4
  (`diam`). That is the **10% step**, not a defect: `Delta m = 0.1` is "nearly optimal" in [P2]
  at *their* noise level, and our AD standard error here is 0.03-0.25%, so truncation dominates.
  `ux` is worst because a 10% step moves the case from `Ma = 2.0` to `Ma = 2.2`. Mode A is the
  precision test; mode B's step is chosen to flip decisions. [P2]'s "AD ~10x less noisy than FD"
  partially reproduces: 5.4x / 2.9x / 1.5x for `temp` / `ux` / `diam`.

- **[P2]'s `sd ~ sqrt(M)` does not hold here — and the raw exponent is the wrong comparison.**
  Their `M`-sweep is a 0-D relaxation, so the signal saturates and `sd` growth is pure score
  accumulation. Ours is a *developing* shock where `dJ/dm` itself grows with `M`, so
  `probe_M_scaling.py` measures the mean too. `diam` is the clean case: signal and noise grow at
  the **same** rate to three digits (`M^0.720` vs `M^0.721`), so relative precision is **flat**
  in run length. `temp` is the only parameter in [P2]'s regime (saturated signal, `q = 0.017`),
  and there `q(sd) = 0.291` — noise growing **more slowly** than the published `0.5`. For `ux`
  the relative spread *falls* as `M^-0.275`. `alpha`'s row is unreadable (its mean flips sign).

### `dJ/d(alpha)` reads zero on the argon shock because argon sits on a stationary point

**Resolved, analytically and by measurement** — `examples/p2_replication`. Anisotropy
relaxation is driven by the second Legendre moment of the VSS deflection, and for
`cos chi = 2 u^(1/alpha) - 1`,

    `<cos^2 chi> = 4a/(a+2) - 4a/(a+1) + 1`,  `d/da = 8/(a+2)^2 - 4/(a+1)^2 = 0` at `a = sqrt(2)`.

**Argon's tabulated VSS `alpha = 1.40` is 1.0 % from `sqrt(2) = 1.41421`.** So a
temperature-like objective has structurally near-zero `alpha` sensitivity there, and
`E[dT_y/d(alpha)] ~ 0` is the *right answer*, not a defect. Measured in 0-D homogeneous
relaxation: `dT_y/d(alpha)` changes sign between `alpha = 1.4` and `1.5`, AD tracks FD at
every point (`|z| <= 0.96`), and regressing on the analytic curve with one free constant
gives **r = 0.973-0.992**.

**Practical rule: never verify an `alpha` gradient at `alpha ~ 1.4`.** Across Bird's five
monatomic gases (`jax_dsmc/species_data.py`) the resolvability `|AD|/SE` at
`N = 32000, M_s = 200` is **He 4.7, Ne 3.1, Kr 2.6, Ar 0.3, Xe 0.0** — argon is the
**worst** available choice and helium the best, at 12.6x argon's conditioning.
`species_data.best_for_alpha_study()` returns helium; a test pins the ordering.

**In the 2-D shock the gain is real but heavily discounted.** All of the above is 0-D; the
verification target adds transport, a specular wall and a developing flow. Measured directly
(`scratch/adjoint/probe_gas_alpha.py`, 256 keys, `N = 16000`, `phi = T_y`):
`|AD|/SE` goes **0.8 (Ar) -> 2.7 (He)** — a 3.4x realised gain against 12.6x analytic, where
0-D realised 15.7x. Helium is *marginally* usable, not comfortably; 10 sigma needs ~14x the
ensemble (~3500 keys, or `N ~ 2.2e5`).

Two consequences that matter more than the headline:

- **The FD cannot do this at all.** Helium's finite difference sits at `|FD|/SE = 0.57`, so the
  `|z| = 0.24` AD-vs-FD agreement is a **low-power** check, not a verification. Never quote it
  as one. FD needs ~28x the ensemble to reach 3 sigma.
- **`SE_FD/SE_AD = 8.4x` for helium** — the first configuration here where [P2]'s claimed ~10x
  AD noise advantage actually shows up (the 0-D replication measured 1.7-2.0x on average). It appears where
  the pathwise term dominates, consistent with the split-variance finding above.

Argon's `|FD|/SE = 1.45` beating its `|AD|/SE = 0.8` is not evidence against AD — argon's true
derivative is ~0, so both are fluctuations about zero. Report:
`python3 -m examples.shock_adjoint.gas_report`; see `PSEUDOCODE.md` §23.3.8.

### Why `alpha` reads worse than `temp` in mode A — three effects, not one

Worth writing down because the first two explanations were each *partly* right, and taking
either as complete would have stopped the investigation early.

1. **Truncation above `h ~ 1e-5`.** `alpha` flips fewer frozen decisions than a velocity does, so
   its flip-free window is wide and bisection stopped at the harness `h0` cap, not at `h_crit`.
   Clean `O(h^2)` on an `h` sweep. Cap tightened to `1e-5`.
2. **Near-stationary sweep points.** Predicted a uniform 100x gain from (1); got **26x at
   `D = 1` and ~1.6x at `D = 2, 3`**. A fix that works in one dimension of three is evidence the
   diagnosis is incomplete. Across all 21 points the error tracks `|AD|` (Spearman
   `rho = -0.835`, `p < 1e-4`) and **not** `h` (`rho = +0.025`, `p = 0.92`): `alpha`'s sweep
   crosses points where `|dJ/dalpha|` falls to ~10 against a sweep-typical 100-700.
   `verify.scaled_error` now reports `err/medAD` beside `max rel err`; it moves `alpha` by
   2.5-6.1x and leaves `temp`/`vstream` unchanged to two digits — the control it needed.
3. **Genuinely larger higher derivatives**, worth the remaining ~100x. Truncation is
   `(h^2/6)|J'''|`, so at a common *relative* step the parameter with larger `|J'''|/|J'|` pays
   more — and `vstream` enters `J` **linearly**, `temp` through a **square root**, `alpha`
   through `cos chi = 2 u^(1/alpha) - 1`, i.e. **all orders in `1/alpha`**. This also explains
   the null `h` correlation: after bisection `h` spans only 4x while the errors span 1000x, so
   `J'''` variation swamps the `h` lever.

All nine cases PASS throughout; worst-case agreement is `2e-6` relative (`3.2e-7` on `err/medAD`).

- **The spatial window costs signal, not variance** — the earlier suspicion was right in outcome,
  wrong in mechanism. The window *lowers* absolute sd in every case; it lowers the mean faster.
  For `diam` the spread falls 2.4x while the signal falls 15x, a net **6.3x** SNR loss. Note also
  that the §11.4 baseline gives only ~1.2x here, not the 84x it gave on the count score alone.

### The test suite runs on the CPU, and that is a measured decision

`pytest tests/` was 599 s. It is now ~170 s, running **more** tests than before, with
nothing deleted that carried unique coverage. Four findings, in order of size:

**1. Per-test durations are not cost attribution.** `test_no_nan_in_any_gradient` bills
**0.0 s** inside a full run and **39 s** when it is the only test in the process. The
duration column charges a shared XLA compile to whichever test reaches it first, so
deleting an "expensive" test usually moves the bill rather than removing it. Any future
attempt to trim the suite on the strength of `--durations` output should start here.

**2. The CPU is faster than the GPU for this suite, and covers more.** 599 s (GPU) vs
329 s (CPU) for identical tests. Every test here is a correctness check on a tiny case,
where GPU compilation dominates execution; nothing asserts device throughput
(`test_perf.py` inspects jaxprs and recompilation counts, both platform-independent).
Separately, `test_decomp.py` needs `>= 2` devices and **skipped all six of its tests on
the single GPU** — the CPU default runs them under
`--xla_force_host_platform_device_count=4`. `pytest --gpu` still runs everything on the
accelerator; that check belongs before a milestone, not in the edit loop.

**3. A Python float argument to a JAX function recompiles per value.** `bkw.distribution_K`
called with a weakly-typed Python `K` bakes it in as an XLA literal: a 13-point sweep cost
**1.53 s** that way and **0.01 s** with `jnp.asarray(K)` — same numbers, 150x the wall
time. Worth remembering anywhere a scalar sweeps in a loop.

**4. `np.polynomial.legendre.leggauss` is not cheap and was recomputed every call.**
0.53 s at `n = 400`, **2.57 s** at `n = 800`. It was called once per quadrature evaluation
in `test_bkw.py`, which alone accounted for that file's entire runtime: **46 s -> 5.7 s**
from one `functools.cache`.

Also: `test_ntc.py::test_stochastic_rounding_is_unbiased` now vmaps over its 4000 seeds
instead of dispatching 4000 eager calls (4.2 s -> 1.7 s, identical keys), and
`test_vss.py::test_momentum_and_energy_conserved` lost its four-value `omega` axis
because `scatter` never reads `omega` — that axis ran four bit-identical copies of every
case (36 -> 9, the only test cases removed in this pass).

Config lives in `tests/conftest.py`: CPU default, four host devices, and a persistent XLA
compilation cache in `.pytest_cache/jax` (303 s cold, 167 s warm; `--no-jax-cache` to
disable). **Caution on totals**: this box shows ~1.6x run-to-run variance on the full
suite (161 s, 166 s and 264 s for identical code), so per-file aggregates are the
trustworthy comparison, not a single wall-clock number.

### Performance regressions: what is gated and what is not

**Was a gap.** `test_perf.py` asserts only the *shape* of the compiled program (no
`while_loop`, one scan, no RNG inside it, no recompilation) and says so deliberately --
"a timing threshold in CI is a flaky test on shared hardware", which is right: this box
shows ~1.6x run-to-run variance. But that left nothing catching the *symptom*. A change
that kept the jaxpr shape and still did five times the work would have passed everything.
`PLAN.md` also advertised `pytest tests/ -m perf --benchmark`, which **collected zero
tests** -- there was no `perf` marker and never had been.

**Now gated, without wall-clock.** `tests/test_perf_budget.py` budgets XLA's own
`cost_analysis()` and `memory_analysis()`: flops, bytes accessed and peak temp allocation
of the compiled program. These are properties of the HLO, **bit-identical across runs**,
so a 5x slower kernel moves them and noise does not. Verified to fail with an actionable
message on a synthetic 40 % regression, and to pass on both backends (9 tests, 10 s CPU /
29 s GPU). Baselines are per backend -- the forward solve is 9.6e5 flops on CPU and 2.3e6
on GPU, because XLA counts a fused GPU kernel differently -- and refresh with
`python3 -m tests.test_perf_budget --print-baseline`. Two of the tests are ratio-based and
so backend-independent; they pin section 23.4's reverse-pass claims directly (measured:
flops 1.96x forward on CPU, 1.81x on GPU; temp memory 6.6x / 7.5x).

**Compile time is checked by hand, not gated** -- a deliberate call. It cannot be measured
reproducibly in a test: JAX caches compiled programs in process, so only the first
`compile()` is real (5.28 s, then 0.00 s three times running), and `conftest.py`'s
persistent cache makes even that a deserialisation on a warm run. It also does not need a
gate, because a compile-time blow-up surfaces immediately in any real run, especially one
that sweeps `A_max`. Report it with `python3 -m tests.test_perf_budget --compile-time`
after any change to the attempt-slot scan; for scale, drawing RNG per slot once cost 7x
in compile time (spec section 8.6).

### Oracle 3 is built; gap U5 is closed (2026-09-08)

`docs/phase0_verification.py --milestone M2` goes from `BLOCKED: U5` to
`gating on M2: 0 gap(s)`.

The 0-D oracles could never have caught this: with a single cell `alpha == 0`, both candidate
adjoint forms collapse to `beta_k = B_k beta_{k+1}`, and the ordering is invisible. Oracle 3 is
1-D, 2 cells, 2 steps, periodic, with `B_k` taken analytically from [P1] eq. `AB_def` rather
than from `jax.jacobian` (which would be circular). Our `move -> collide` loop matches
`beta_k = B_k beta_{k+1} + dt P* alpha_{k+1}` to **1.33e-16** and misses [P3]'s inside-`B_k`
form by **9.85e-02**; reordering the forward loop swaps both, which is what makes it a
discriminator rather than a coincidence. `alpha_k = alpha_{k+1}` holds to 0.0e+00 in both.

Two traps worth remembering, both found by building it:

- **A velocity-only objective makes this oracle vacuous.** `alpha == 0`, the transport term
  vanishes, and it degenerates into Oracle 1 *while still passing*. The objective carries an
  `x_M` term and a nonzero-`alpha` guard is asserted first.
- **The oracle alone does not pin production.** It validates the recursion against its own 1-D
  forward pass, so a reordering of `simulate._step` would leave it passing while it validated a
  loop we no longer run. `test_production_step_moves_before_it_collides` closes that
  behaviourally: under `move -> collide` the new positions come from **pre**-collision
  velocities, measured 0.0e+00 against 1.31e-02 for the post-collision form.

It also corrected §13.3: two steps is **not** the minimum that discriminates (one step gives
2.66e-2), though two remains the right choice for margin and for testing `B_k` composition.

### The Couette viscosity gate, and how it was got wrong four times (2026-09-09)

**Read this before quoting any ratio from `examples/couette`.**

The M4 gate compares the measured shear viscosity against Chapman-Enskog (Bird 4.64). The
final form of that gate is the **`Kn -> 0` intercept of a Knudsen sweep**, with every point
an ensemble over independent seeds. It reached that form only after four successive
explanations of an apparent viscosity deficit turned out to be wrong:

| # | explanation offered | what actually killed it |
|---|---|---|
| 1 | "DSMC under-resolution at coarse cells" | the coarse point was `A_max` clipping (gap U4), not resolution |
| 2 | "cell size" | **partly right, and initially dismissed wrongly** -- see below |
| 3 | "finite-Kn physics" (first attempt) | the sweep's widest case had run 1.7 % of a diffusion time |
| 4 | "under-convergence" | ruled out properly: warm-up 1.0 -> 3.0 diffusion times moves the gate `+0.0041 +- 0.0065`, `\|z\| = 0.63` |

**The single root cause: a quantity with no error bar was treated as a measurement, four
times over.** `mu_meas/mu_CE` carries several per cent of Monte Carlo scatter at any
affordable particle count. The one-line check that would have caught this at the very
outset -- before any of the four explanations were constructed -- is *run the same
configuration twice with different seeds*.

**The resolution is explanation 3, done properly.** The whole offset is finite-Knudsen
physics. A converged sweep (4 seeds per point, warm-up scaled per case) gives

| Kn | ratio |
|---|---|
| 1/10 | 0.9225 +- 0.0028 |
| 1/15 | 0.9411 +- 0.0069 |
| 1/22 | 0.9561 +- 0.0048 |
| 1/30 | 0.9779 +- 0.0032 |

fitting to `ratio = 0.9971 +- 0.0095 - (0.777 +- 0.143) Kn`, i.e. **-0.29 % from
Chapman-Enskog at Kn -> 0, |t| = 0.31 on 2 dof: the gate passes.** The slope's sign is
what first-order slip theory predicts.

**Explanation 2 was dismissed on a study that could not see the effect.** The cell-size
sweep was read as flat and used to rule resolution out. Re-run with per-case warm-up and
ensembles it is *not* flat: `0.9352 +- 0.0266` / `0.9728 +- 0.0024` / `0.9829 +- 0.0031` at
`dx/lambda = 0.60 / 0.30 / 0.15`, with the two finest points separated by `|z| = 2.58`.
`O(dx^2)` fits and extrapolates to `0.9863` at `dx -> 0`, i.e. **the gate configuration
carries a 1.35 % resolution bias.** "Flat" is exactly what an under-powered study of a
noisy quantity looks like, which is why this one survived the longest.

The two corrections compose because the Knudsen sweep holds `dx/lambda` fixed at 0.30 at
every point, making cell-size error a constant offset that moves that sweep's intercept and
not its slope. **The gate is therefore the double extrapolation:**
`Kn -> 0` gives `0.9971 +- 0.0095`, `dx -> 0` adds `+0.0135 +- 0.0039`, total
**`1.0105 +- 0.0103`, i.e. +1.05 % from Chapman-Enskog, |t| = 1.03: consistent.**

**And explanation 4 was doubly wrong -- including in its retraction.** Warm-up was first
dismissed on a single-draw study, then briefly *re*-blamed on the strength of a 2-seed
point reading `0.9872 +- 0.0004` where 4 seeds at the same configuration give
`0.9779 +- 0.0032`. The 2-seed SE was a fluke of two draws landing close together. Measured
against the ensembles, moving warm-up from 1.0 to 3.0 diffusion times shifts the gate by
`+0.0041 +- 0.0065`, `|z| = 0.63` -- **not significant**, which means the original
measurement that the shear rate settles by ~0.3 of a diffusion time was right all along.
Reading a 2-seed error bar as precision is the same error one level up, committed while
documenting the error.

Three structural fixes now make each failure unrepeatable, and all three live in the code
rather than in this document:

1. **No single-seed mode.** `examples/couette.__main__.run_case` raises on `seeds < 2`.
   Every ratio in every log and table carries a standard error and a seed count.
2. **Warm-up in diffusion times, never steps.** `diffusion_steps(case)` returns `L^2/nu` in
   units of `dt`; `--warm-diff` defaults to 3 of them, per case. This matters specifically
   because a Knudsen sweep *must* change `L`, so a constant step budget under-converges
   the wide end as `L^2` and manufactures a smooth trend. The justification is the *first*
   sweep, not a measured shift at the gate: it ran gaps of 15-120 lambda at a flat 4 000
   steps, where the widest case needs 232 105 -- short by a factor of 58. At the gate
   configuration itself the ratio is already converged at 1.0 diffusion times (see above),
   so 3.0 is margin, not necessity.
3. **The gate is the intercept.** Chapman-Enskog is a continuum limit, so comparing any
   single finite-Kn run to it and calling the difference an error is a category mistake.
   The measured slope is about `-0.9` per unit Kn -- the apparent viscosity falls as
   Knudsen layers eat into the sheared bulk, which is the sign first-order slip theory
   predicts.

A fourth point worth carrying forward: **an SE from 2 seeds is not a usable error bar.**
The 2-seed version of the sweep put points 5.6 and 9.2 "sigma" off a visually excellent
line; the 4-seed version puts the same points within 0.6-2.1 sigma of it, with a smaller
rms residual (0.0051 against 0.0082). Nothing about the physics changed -- only whether the
error bars meant anything. `kn_extrapolation` therefore takes its uncertainty from the
fit's residual scatter rather than from the per-point SEs, and `figure_knudsen`'s second
panel plots those pulls so the reliability of the quoted SEs is visible rather than
assumed. **Use 4 seeds minimum for anything quoted; 2 only to establish that a quantity
scatters at all.**

Numbers are not restated here or in the README on purpose. They live in
`examples/couette/RESULTS.md`, regenerated by `python3 -m examples.couette.report` parsing
`results/*.log` -- the stale-hardcoded-number failure is exactly how the retracted
"0.24 % from Chapman-Enskog" claim outlived the run that produced it.

### Reward attribution in the DiCE surrogate -- measured bias (2026-09-10)

See **`docs/ATTRIBUTION.md`**. Short version, because it changes how a score-carried
gradient should be read:

* `adjoint.surrogate` pairs each score with its own particle's reward. Every dropped term is
  a covariance, so the estimator is unbiased **iff decisions do not propagate between
  particles**. Measured: collisions off -> exact (`|z| = 0.18`); collisions on -> low by
  **2.8x** (`|z| = 22`).
* **The bias grows with trajectory length**, not with attribution radius: `FD/AD` = 1.13,
  1.32, 1.57, 1.95, 2.82 at `M` = 2, 5, 10, 20, 40. Widening attribution to the whole cell
  closed only ~10 % of the gap for 7x the variance, so that hypothesis is rejected.
* **The shipped collision score is ~6 % low at `M = 16`** -- a competing explanation for
  about half of the `FD/AD = 1.137` that `VARIANCE.md` section B attributes to `O(dm^2)`
  truncation. `alpha`'s results are unaffected: its gradient is consistent with zero.
* **Pathwise gradients are untouched** by all of this, exactly: `dJ/dT_wall` and
  `dJ/dv_wall` verify to 3.3e-12 and 2.5e-11 against a fixed-key FD.
* The exact fix is the global-reward form (20x worse SE). The better fix for `acc` is to
  **relax the Bernoulli** (`acc_delta`) so it becomes pathwise -- [P3]'s `eps` trick applied
  to the accommodation draw. `acc_delta = 0` and `score_spread="particle"` reproduce today's
  behaviour bitwise.
* Trap: `sum_i logh_acc_i` is **not** `sum_k l_k`. Multiplicities are 1 (wall), 2 (pair) and
  ~`npc` (NTC count), so a global estimator built on it is valid only for `acc`.

### M7/M8 boundary adjoints -- what is verified and what is not (2026-09-11)

**Forward solves are untouched.** The accommodation relaxation (`acc_delta`) and the new
boundary scores change no forward result: the Couette gate reproduces pre-M7 output to every
digit, and the 30 000-step inflow shock reproduces all six moments, the shock position, the
flux ratio and every diagnostic with **0 mismatches**. Pinned by `tests/test_walls.py`.
One caveat is pinned there too: the velocity blend equals the old `where`-select in *value*
everywhere but differs in *bits* at signed zeros (`-0.0 + 0.0` is `+0.0`), so "bitwise
identical" was claimed and is not literally true.

| parameter | route | status |
|---|---|---|
| wall temperature, wall velocity | pathwise | **verified** -- fixed-key FD to 3.3e-12 / 2.5e-11, and pointwise against the Couette closed form (worst 0.068 / 0.066) |
| inflow rate `nrho` | score | **verified** -- ensemble FD/AD = 1.04, `\|z\| = 0.81` |
| shock `d(T_2/T_1)/du_1` | pathwise | **verified** against Rankine-Hugoniot, `M = 1.6-4.0` |
| accommodation `acc` | score | biased 2.8x at `M = 40`; use `acc_delta > 0` to make it pathwise |
| shock `d(rho_2/rho_1)/du_1` | score | **WRONG SIGN** -- see `docs/ATTRIBUTION.md` |

**Two limits bound every gradient here, and they are different.** Attribution bias (score
route only, grows with trajectory length) and chaotic amplification (all pathwise gradients,
grows exponentially). The shock density ratio fails both: at 300 steps it is precise and
sign-wrong; at 1000 steps, where the forward finally reaches Rankine-Hugoniot, the spread is
9 orders of magnitude above the signal.

**A forward bug found on the way.** `inflow.inject` silently dropped entrants when dead
slots ran out -- `shortfall` counted only `target > capacity` despite its contract naming
both. A saturated array read as correct physics (-46 % density ratio at Mach 4, all
diagnostics zero). Fixed.
