# JAX-DSMC — Pseudocode Specification (Phase 0, revision 3, condensed 2026-10-21)

This is the implementation contract of the package: the shape, dtype, traced/static and RNG
contracts, the algorithms, the constants, and the evidence behind each non-obvious choice.
`jax_dsmc/__init__.py` and about 75 code and test docstrings cite it by section number
(`Spec: section 8.4`), so **section numbers are stable: sections are shortened or retired in
place, never renumbered.** Violating a contract in sections 0-3 produces wrong gradients
silently rather than an error, which is why they are the load-bearing part.

> **Gradient status banner.** The production gradient path for decision-driven parameters (wall
> and inflow state, collision constants, shape) is the **particle-carried ensemble adjoint**
> ("route b", decided 2026-09-15; `ensemble_particle.py`, `ensemble_backward.py`,
> `ensemble_device.py`, with the per-cell backward fit of section 9.7.2). The score-function /
> DiCE surrogate of section 11 and the variance measurements of section 23.3 describe the
> *reference* path that reproduces [P1]/[P2]: exact for pathwise parameters, but measured too
> noisy or biased to be the production route for the rest.
> [ADJOINT_CORRECTIONS.md](ADJOINT_CORRECTIONS.md) (trajectory scoring, P3 reference
> implementations, validity limits) supersedes any estimator-validity claim in this file that
> conflicts with it; [ADJOINT_BRANCH_REVIEW.md](ADJOINT_BRANCH_REVIEW.md) is the later review.

## Summary

**What it covers.** A differentiable DSMC solver (VSS collisions with NTC, static shapes,
`f64`, `lax.scan` everywhere) in which `jax.grad` reproduces the adjoint DSMC of the reference
papers for pathwise parameters, plus the machinery around it: embedded level-set bodies with cut
and split cells, SPARTA-style lane tracking, inflow/outflow/wall boundaries, multi-device slab
decomposition, and the mesh-aware ensemble adjoint used for shape optimisation.

**Reading order.** Sections 0-3 (conventions, RNG, safe primitives, containers) are required
reading before touching any kernel. Sections 4-10 are the forward solver, 11-13 the surrogate
and its oracle tests, 17-19 the verified facts, divergences from SPARTA and the open-claims
register, 20-23 scaling, validation and measured adjoint behaviour.

### Module map

| Package module | Spec section | Role |
|---|---|---|
| `constants.py`, `safe.py` | 2 | clamp constants; singularity-safe primitives |
| `rng.py` | 1 | stage ids, `key_for`, slot-addressed `indexed_*` draws |
| `state.py` | 3, 0.4 | pytree containers, `SimParams` (traced) / `SimStatic` (static), `check_fnum_uniform` |
| `grid.py`, `mesh/` (`levelset`, `cut`, `geometry`, `walk`, `surface`, `analytic`) | 4 | cell lookup, volumes, level-set bodies, cut/split cells |
| `sampling.py`, `initialize.py`, `inflow.py` | 5, 9.3 | reparameterised draws, initial state, flux-weighted inflow |
| `sort.py`, `compaction.py` | 6 | occupancy table (full and partition sort) |
| `collide/vss.py`, `collide/ntc.py` | 7, 8 | VSS kernel, NTC driver, majorant |
| `move.py`, `boundary.py`, `wall_adjoint.py` | 9 | advection, BCs, wall/element tracker (stencil, walk, lane) |
| `sample.py`, `output.py`, `viz.py` | 10 | moments, VTK/HDF5 output, figures |
| `adjoint.py` | 11 | `magic`, `surrogate`, `trajectory_surrogate`, windowed objectives |
| `simulate.py` | 12, 8.5.1 | `run`, `run_adaptive`, event records |
| `ensemble_particle.py`, `ensemble_backward.py`, `ensemble_device.py`, `cell_fit.py`, `shape.py` | 9.7 | production ensemble adjoint, per-cell fit, shape VJP |
| `decomp.py`, `parallel.py` | 20 | multi-device slab decomposition |
| `species_data.py`, `bkw.py`, `gradient_validation.py`, `validation.py` | 21, 23 | gas data, BKW gate, gradient verification helpers |

### Status legend

| Mark | Meaning |
|---|---|
| **Built** | implemented and covered by a named test or script |
| **Measured** | a numerical claim with the probe or test that produced it; numbers are historical unless the section says the test still pins them |
| **Retired (date)** | design no longer used; the heading is kept so citations resolve, with what replaced it |
| **Open** | asserted but not demonstrated; registered in section 19 |

Probe scripts cited as `scratch/...` are not part of the package and may have moved; the test
or example named beside them is the durable evidence.

### What changed since revision 3

- Production gradient is the ensemble adjoint (2026-09-15), not the score surrogate (banner,
  sections 9.4.1, 9.7, 11, 23).
- Embedded bodies: C1 level-set shapes with grazing regularisation and a silhouette term (4.4),
  split cells (4.5), analytic and polygon surfaces with the stencil, walk and **lane** trackers
  (9.6, 9.6.1). 2026-10-21: the lane tracker's post-loop unresolved-particle query samples the
  *remaining move*, not the search reach (`wall_adjoint._lane_distance`).
- Backward value fits: per-query kNN fit replaced by one polynomial per collision cell (9.7.2);
  default `ensemble_backward.CELL_BASIS = "cell:cubicv:adapt1.5:guard32:align0.9:both"`
  (2026-10-19 leverage guard). Two-column value sweep (9.7.1).
- Forward: batched slot-addressed RNG draws (1, 8.4), scratch row as out-of-bounds drop (8.4.1),
  partition sort on XLA:CPU (6.2), `run_adaptive` for `A_max` (8.5.1), lane tracker, lean event
  records, inflow by implicit inverse CDF instead of rejection (5.4), `count_scheme`,
  `majorant_mode` and `score_mode` options.
- Verification: oracle tests 1-3 (13), `test_gaps.py` for U1/U3 (19), cost-analysis budgets
  `test_perf_budget.py` (14, 23.4), decision-checksum FD harness (13.4.1).
- Retired or corrected: nested-closure pseudocode in 8.4/12 (now flat, matching the code),
  the planned `tests/test_grid.py` and `tests/test_sampling.py` (do not exist; coverage is listed in 4.2, 5.6),
  Bird rejection inflow sampler as the production path (5.4), the single-seed score-variance
  conclusions of 23.3 as a production argument.

References:
- SPARTA source, `/home/aakash/sparta/sparta/src/` — cited as `file.cpp:lines`.
- dsmcFoam+ / hyStrath, `/home/aakash/OpenFOAM-v1706/dsmcFoam+_internals.html` — cited as
  **[DF]**. The codes disagree in several places; section 18 records which we follow and why.
- `papers_reference/arXiv-2009.01363v3` — Caflisch, Silantyev, Yang. *Adjoint DSMC for nonlinear
  Boltzmann equation constrained optimization.* **[P1]**.
- `papers_reference/arXiv-2207.11579v2` — Yang, Silantyev, Caflisch. *Adjoint DSMC ... general
  collision model.* **[P2]**.
- `papers_reference/arXiv-2603.20946v2` — Caflisch, Chen, Yang. *Adjoint DSMC ... spatially
  inhomogeneous Boltzmann equation with general boundary conditions.* **[P3]**.

Environment: `source ~/activate_jax_env.sh` (see README; JAX 0.6.2 with `jax_enable_x64`). The
system Python is not usable (its jaxlib silently falls back to CPU).

---

## 0. Notation and global conventions

### 0.1 Symbols

| Symbol | Meaning | Typical |
|---|---|---|
| `N` | particle array capacity (static) | 1e4 – 1e10 |
| `D` | spatial dimension, `1 \| 2 \| 3` (static) | 3 |
| `S` | number of species (static) | 1 for now |
| `G` | number of collision groups (static) | 1 for now, `G <= S` |
| `n[d]` | number of cells along axis `d` (static) | |
| `C` | total cells, `prod(n)` (static; `prod(n) + n_sub` with split cells, §4.5) | |
| `A` | `A_max`, max collision attempts per cell per step (static) | 8 – 64 |
| `K` | `K_bounce`, max wall bounces per step (static) | 4 |
| `M` | number of timesteps (static) | |
| `fnum` | real molecules per simulator particle, **per cell** `f64[C]` (traced) | |
| `Sigma` | NTC majorant, `vremax` (state) | |

### 0.2 The velocity/space dimension split

Positions are `(N, D)`. **Velocities are always `(N, 3)`**, regardless of `D`. Advection uses
only the first `D` components, `v[:, :D]`. This is the Bird/SPARTA convention and it is not
negotiable: the VSS scattering kernel, the `2*U**(1/alpha) - 1` deflection law, and every
Chapman-Enskog transport coefficient we validate against all assume a 3-component velocity
space. A "1D simulation" means 1 spatial dimension and 3 velocity components.

Consequence for the adjoint: the position adjoint `alpha` is `(N, D)` and the velocity adjoint
`beta` is `(N, 3)`, and the transport coupling in [P3] eq. `adjoint_gamma_R`,
`beta_k = B_k (beta_{k+1} + dt * P^* alpha_{k+1})`, uses the projection `P: R^3 -> R^D` that
takes the first `D` components. `P^*` is its adjoint, zero-padding `R^D -> R^3`.

Note that recursion is [P3]'s, and it assumes [P3]'s `collide -> move` intra-step ordering.
**We run `move -> collide`** (matching SPARTA and [DF]), for which the correct form has the
transport term *outside* the collision operator:

```
beta_k  = B_k beta_{k+1} + dt * P^* alpha_{k+1}          # OURS  (move -> collide)
beta_k  = B_k (beta_{k+1} + dt * P^* alpha_{k+1})        # [P3]  (collide -> move)
alpha_k = alpha_{k+1}                                     # both
```

Reverse-mode AD produces the correct form for whatever loop is written, so the implementation
needs no special handling — but do not check a hand-derived adjoint against [P3]'s formula
without first matching the ordering. The two differ by `O(dt * ||B - I||)`, which does not vanish
at realistic timesteps. See §12.1 for the numerical demonstration.

### 0.3 Precision

`f64` throughout; `jax.config.update("jax_enable_x64", True)` at import. The VSS energy
conservation test asserts agreement to ~1e-14 relative, which f32 cannot deliver, and the
adjoint oracle tests compare against hand-coded recursions to machine precision.

### 0.4 Traced vs. static — the discipline

- **Static** = affects array shapes or Python control flow. Must be a compile-time constant,
  lives in a frozen dataclass field marked `static`, and changing it forces recompilation.
- **Traced** = a differentiable leaf. Lives in a pytree-registered dataclass, may carry a
  tangent/cotangent, and must never be branched on with a Python `if`.

A value may not be both. The recurring mistake to avoid: putting `n` (cells per axis) in a
traced field, or putting `dt` in a static one.


### 0.5 Frozen decisions

Every discrete random choice in DSMC is a **frozen decision**: the forward solve makes it, the
adjoint treats it as a constant, and no gradient flows through the choice itself. In JAX this is
automatic for integer- and boolean-valued results (integers carry no tangent), but we mark it
explicitly with `stop_gradient` where the value is float-derived, both as documentation and as a
guard against a future refactor accidentally opening a gradient path.

The frozen decisions are, exhaustively:

| Decision | Where | Type |
|---|---|---|
| cell membership of a particle | `grid.cell_of`, `grid.collision_cell` | int |
| the sort permutation | `sort.build_table`, `sort.update_table` | int |
| number of collision attempts per cell | `ntc.attempt_count` | int |
| which pair is drawn for an attempt | `ntc.pair_ntc` / `ntc.pair_from_draws` / `ntc.pair_perm` | int |
| accept/reject of a candidate collision | `ntc._slot` (`u_acc < p`) | bool |
| accommodation (specular vs diffuse) at a wall | `boundary._diffuse_velocity` | bool |
| which face / element a particle crosses | `boundary.fold_index`, `wall_adjoint` first-hit | int |

**Freezing these is not an approximation we are choosing** — it is what [P1] specifies
("the choice of collision times, collision partners, and collision angles is not included in the
variational principle... determined externally"). What it *costs* is accounted for separately by
the score terms of [P2] (section 11) or by the ensemble adjoint (section 9.7), which recover the
sensitivity of the *probability* of each decision without differentiating the decision itself.

### 0.6 Code-quality conventions

Enforced by `ruff` (config in `pyproject.toml`); `ruff check .` and `ruff format --check .` must
pass.

**No nested function definitions.** JAX pushes you toward them — `lax.scan` bodies want to close
over loop-invariants — but a closure cannot be unit-tested, cannot be read in isolation, and hides
how much state it captures. Bind loop-invariants with `functools.partial` against a module-level
function (`ntc._slot` is the model: every dependency is an argument).

Other conventions:
- One physical concept per function; `collide` orchestrates, `_slot` does one attempt,
  `collision_rate`, `acceptance_prob`, `scatter`, `log_h` are independently testable.
- No mutable default arguments, no bare `except`, no `== True`.
- Every public function documents shapes, traced vs static arguments, and its SPARTA / [DF] /
  paper citation.
- Physics symbols keep their paper names (`N`, `D`, `A_max`, `Sigma`); ruff's naming rules are
  disabled for exactly this reason.
- No module may take the name of a stdlib module (see the `types.py` note in section 3).
- No runtime collisionless shortcuts: tests and production run the real collisional forward and
  backward paths (a collisionless run is a property of the setup, not something to detect).
- `main` stays fully differentiable; the non-differentiable SPARTA-faithful forward lives on the
  `sparta-match` branch only.

---

## 1. The RNG contract

**This is the single most important contract in the system.** Get it wrong and gradients are
silently incorrect, finite-difference checks become pure noise, and `jax.checkpoint` produces a
different trajectory on the backward pass than the forward pass took.

```
STAGE_INIT=0  STAGE_MOVE=1  STAGE_WALL=2  STAGE_NTC_ROUND=3
STAGE_PAIR=4  STAGE_ACCEPT=5  STAGE_SCATTER=6  STAGE_INFLOW=7        # jax_dsmc/rng.py

def key_for(base_key, step, stage_id):
    return fold_in(fold_in(base_key, step), stage_id)    # two folds: step and stage independent
```

Rules, all of which are testable (`tests/test_rng.py`):

1. **Keys depend only on `base_key`, the loop index, and a static stage id.** Never on a traced
   state value. If a key depended on state, the gradient would try to flow through the RNG.
2. **Keys are never carried in the state.** They are recomputed from `(base_key, step, stage)`
   every time, which is what makes rematerialisation under `jax.checkpoint` reproduce the *same*
   draws on the backward pass.
3. **Each stage draws all its randomness in one batched call, addressed by slot.**
   `indexed_uniform(key, n, trailing)` / `indexed_randint` / `indexed_normal` give draw `i` a
   dependence on `(key, i)` only, so growing `n` (e.g. `A_max`) appends draws and leaves every
   earlier one unchanged. This is a *performance* requirement as well: drawing inside the attempt
   scan cost 7x in compile time and halved throughput (section 8.6). `key_for_slot` is the one
   sanctioned way to fold a slot index into a stage key; the lane tracker's per-slot streams
   (`wall_adjoint._slot_uniforms`, `fold_in(pass key, slot)`) follow the same rule so a particle
   meets identical randomness in any row of any tracker.
4. **Common random numbers.** Two runs with different parameters but the same `base_key` consume
   an identical stream, which is what makes `(J(theta+h) - J(theta-h)) / 2h` estimate the
   pathwise derivative at all. A decomposed multi-device run folds in the device index and is
   therefore equal to a single-device run only statistically (section 20.3).

---

## 2. `jax_dsmc/safe.py` — singularity-safe primitives

Built first, because it is the highest-risk module. DSMC is full of expressions whose *value* is
fine at a removable singularity but whose *derivative* is `NaN`. Since `jnp.where` evaluates both
branches, and `NaN * 0 = NaN` in the backward pass, the obvious guard does not work.

```
# WRONG - produces NaN gradient at d == 0 even though the primal is correct
result = where(d > eps, num / d, fallback)

# RIGHT - make the argument safe BEFORE the operation that is singular
d_safe = where(d > eps, d, 1.0)
result = where(d > eps, num / d_safe, fallback)
```

### 2.1 The singularity inventory

| Expression | Singular at | Appears in | Source |
|---|---|---|---|
| `sqrt(vr2)` | `vr2 = 0` | relative speed | `collide_vss.cpp:214` |
| `vr2 ** (1 - omega)` | **value** at `vr2=0` only if `omega > 1`; **gradient** at `vr2=0` for *every* physical `omega` | NTC cross-section | `collide_vss.cpp:222` |
| `d = sqrt(vy^2 + vz^2)`, then `/d` | `v_r` along x | VSS scatter frame | `collide_vss.cpp:429-437` |
| `sqrt(1 - cosX^2)` | `abs(cosX) = 1` | deflection | `collide_vss.cpp:417,427` |
| `sqrt(-log(U))` | `U = 1` | flux-weighted draws **only** | [DF] `dsmcDiffuseWallPatch.C:106-111` |
| `log(h)` | `h -> 0` | score term | [P2] eq. `back_eta` |
| `1 / counts` | empty cell | moments | `sample.py` |

SPARTA guards the third case with a hard `if (d > 1.0e-6)` branch. We reproduce the same
*numerics* but with the safe-argument pattern so the gradient is finite.

Note the fourth row is narrower than it first appears. Following [DF] (see 5.1) the isotropic
Maxwellian is drawn with `normal()` rather than SPARTA's `sqrt(-log U)` polar trick, so this
singularity is confined to genuinely flux-weighted draws — the diffuse-wall normal component and
the inflow sampler. [DF] additionally writes that draw as `sqrt(-2 ln(1 - R))`, which moves the
singular end from `R = 1` to `R = 0`; either way `safe_sqrt` covers it.

### 2.2 API

```
EPS = 1e-30          # for denominators; harmless where the composite is bounded (2.3)
EPS_D = 1e-6         # RELATIVE frame-degeneracy threshold: always used as
                     # `d > EPS_D * vrm`, never as an absolute cut. See 7.4.
EPS_REL_VR2 = 1e-8   # RELATIVE clamp for safe_pow on vr2. The absolute cut is
                     # eps = EPS_REL_VR2 * (2 * K_BOLTZ * tref / mr), i.e. a fixed
                     # fraction of the reference thermal energy scale. See 2.3.
EPS_U = 1e-12        # clamp for safe_pow on a dimensionless uniform in [0,1)

def safe_div(num, den, eps=EPS, fallback=0.0):
    ok = abs(den) > eps
    den_safe = where(ok, den, 1.0)
    return where(ok, num / den_safe, fallback)

def safe_sqrt(x, eps=EPS):
    # derivative of sqrt is 1/(2 sqrt(x)) -> inf at 0. Clamp the argument.
    ok = x > eps
    x_safe = where(ok, x, 1.0)
    return where(ok, sqrt(x_safe), 0.0)

def safe_norm(v, axis=-1, eps=EPS):
    # NOT sqrt(sum(v^2)) - that has a NaN gradient at v = 0
    return safe_sqrt(sum(v * v, axis=axis), eps)

def safe_pow(x, p, eps):
    # x^p for x >= 0, p possibly negative or fractional, p traced.
    # eps is REQUIRED and RELATIVE -- see 2.2.1. For the collision rate the caller
    # passes eps = EPS_REL_VR2 * (2 * K_BOLTZ * tref / mr), i.e. a fixed fraction of
    # the reference thermal energy scale, NOT an absolute 1e-30.
    # NOTE the fallback is p-dependent: x^0 == 1 for every x, including x == 0.
    # Returning a blanket 0.0 is wrong exactly at omega == 1 (Maxwell molecules),
    # where q = prefactor * vr2^0 must equal prefactor even for a coincident pair
    # -- which is precisely Oracle 1's configuration (13.1).
    ok = x > eps
    x_safe = where(ok, x, 1.0)
    fallback = where(p == 0.0, 1.0, 0.0)
    return where(ok, exp(p * log(x_safe)), fallback)

def safe_normalize(v, axis=-1, eps=EPS):
    n = safe_norm(v, axis, eps)
    return safe_div(v, expand_dims(n, axis), eps, fallback=0.0)

def safe_log(p, eps=EPS):
    # for score terms: log h where h in (0, 1] but can underflow
    return log(maximum(p, eps))

def safe_recip_count(c):
    # 1/c for integer counts, 0 for empty cells
    return where(c > 0, 1.0 / maximum(c, 1), 0.0)
```

### 2.3 Clamps must bound the derivative, not merely make it finite

Revision 2 set `EPS = 1e-30` everywhere and tested only that gradients are *finite*. That is far
too weak: just above the clamp, `d/dx sqrt(x) = 1/(2*sqrt(1e-30)) = 5.0e+14` — finite, and
useless. But "raise eps everywhere" is the wrong correction, because the three functions behave
completely differently. Measured in `scratch/probe_num_c1_safe.py`, `probe_num_c1_harm.py`,
`probe_num_c1_realistic.py`:

**`safe_sqrt` / `safe_norm` — already bounded; `eps` is irrelevant here.** Where the spec
actually reaches `safe_sqrt` it is through `safe_norm(v) = safe_sqrt(sum(v*v))`, and the chain
rule cancels `1/(2 sqrt(x))` against `2v` exactly:

```
|v|:        1e-14   1e-12   1e-10   1e-08   1e-04   1e+00   1e+02
||d/dv||:   1.0000  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000   (exactly)
```

`vrm`, `d`, and every other `safe_norm` are structurally bounded by 1 **regardless of `eps`**.
Keep `EPS = 1e-30` there.

**`safe_pow` — genuinely unbounded, and the one that matters.**
`d/d(vr2) [vr2^(1-omega)]`:

| omega | \|vr\| = 1e-4 | 1e-8 | 1e-12 | 1e-14 |
|---|---|---|---|---|
| 0.75 | 5.0e+01 | 5.0e+03 | 5.0e+05 | (clamped) |
| 1.20 | 1.6e+05 | 6.3e+10 | **2.5e+16** | **1.6e+19** |

**`safe_log` — bounded by the majorant gap, not by `eps`.** The reject branch
`d log(1 - q/Sigma)/dq = -1/(Sigma - q)` measures `-1.0e+03` at `q/Sigma = 0.999`. Controlled by
keeping `Sigma` clear of `max q` (see 7.2), not by a clamp.

**Is the unbounded case reachable?** Not in an equilibrium gas: over 1e8 sampled Maxwellian
relative velocities at 300 K the minimum `|vr|` was **1.49 m/s**, and
`P(|vr| < 1e-2 m/s) = 6.8e-15` per pair. It *is* reachable from **cold or duplicated parcels** —
two particles differing only by f64 round-off give `|vr| = 3.4e-11 m/s` and
`|dq/dv| = 1.8e+14`. Cloned parcels under weighting, a zero-temperature IC, and a cold
supersonic freestream all produce exactly that.

**The harm is dominance, not precision loss.** Summing one `1.6e19` cotangent with 4095 normal
ones reproduced the exact long-double sum to 1e-16 relative — no catastrophic cancellation. What
happens instead is that one particle's gradient is **3.2e14x** the rms of the other 4095, so the
descent direction becomes a single degenerate pair.

**Fix.** Give `safe_pow` a **relative** clamp, `vr2 > EPS_REL_VR2 * (2 k T_ref / m_r)`. The
resulting bound must be **asserted**, not assumed — see 2.4.1 for the form it takes, and note
that "bounded" is not the same as "harmless": the residual *dominance* of an at-the-clamp pair
scales as `EPS_REL_VR2^(0.5-omega)`, which is `2.7e2` at `omega=0.75` and `4.0e5` at
`omega=1.2`. (Revision 3 quoted `~3.4e4` here; that figure reproduces as the rms dominance at
`omega ~ 1.1`, not at 1.2.) The value of `EPS_REL_VR2` is settled in 19/U1.

### 2.4 Tests (`tests/test_safe.py`)

For each function, and for each singular point in the inventory above:
- the primal value matches the analytic limit (or the documented fallback);
- **`|grad(f)|` is bounded by a stated per-function constant** — not merely finite, which was
  revision 2's error:

  | function | derivative bound |
  |---|---|
  | `safe_sqrt` via `safe_norm` | `1.0` (structural, independent of `eps`) |
  | `safe_sqrt` bare | `1/(2 sqrt(eps))` |
  | `safe_pow` | `2 |1-omega| eps^(0.5-omega)` — **this bounds `dq/d|vr|`, equivalently `dq/dv`, NOT `dq/d(vr2)`**; see 2.4.1 |
  | `safe_log` reject branch | `1/(Sigma - max q)` |

- the `safe_norm` composite bound of `1.0` holds across 14 decades of `|v|`;
- `grad(grad(f))(x_singular)` is finite (the surrogate is a second-order construction, and
  `check_grads` uses higher-order comparisons);
- away from the singularity, `grad(f)` agrees with the naive unguarded implementation to
  machine precision — the guard changes nothing where it should not.

#### 2.4.1 The `safe_pow` bound is stated in `d/d|vr|`, not `d/d(vr2)`

Revision 3 wrote this bound against `d q / d(vr2)`, which is **wrong and would fail against
correct code.** Measured (N2 at 273 K, `omega = 1.2`, `eps = EPS_REL_VR2 * 2kT/m_r`):

| quantity at the clamp | value |
|---|---|
| `\|dq/d(vr2)\|` | `4.383e+02` |
| `\|dq/d\|vr\|\|` | `3.529e+01` |
| stated bound `2\|1-omega\| eps^(0.5-omega)` | `3.591e+01` |

Against `d/d(vr2)` the stated bound is violated by `1/(2 sqrt(eps))` — a factor of **12.4** here.
Against `d/d|vr|` it is exactly tight.

The `d/d|vr|` form is also the one that matters. What reaches a particle is `dq/dv_i`, and by
the chain rule `dq/dv_i = (dq/d(vr2)) * 2 vr_i`, which at `|vr| = sqrt(eps)` is bounded by
precisely `2|1-omega| eps^(0.5-omega)`. So the formula was right and only the label was wrong —
but a test transcribed literally from the old wording would have failed, or worse, prompted
someone to "fix" correct code to satisfy an incorrect bound.

---

## 3. `jax_dsmc/state.py` — state and parameter containers

> **Renamed from `types.py` (2026-09-04), and the reason is worth keeping.** A module named
> `types` inside the package **shadows the standard library's `types` module** whenever the
> working directory is `jax_dsmc/` — Python puts the cwd first on `sys.path`. `functools`,
> `dataclasses` and `enum` all import stdlib `types`, so the interpreter fails to start at all,
> with `cannot import name 'MappingProxyType' from partially initialized module 'types'` — an
> error that points nowhere near the cause. No module in this package may take the name of a
> stdlib module.

All containers are registered pytrees so `jax.grad` can traverse them. Static fields are
declared with `static_field()` and are compared by value for cache lookup.

### 3.1 Shape and dtype contract

```
ParticleState:  x f64[N,D]   v f64[N,3]   species i32[N]   alive bool[N]   logh_acc f64[N]
                # logh_acc = score accumulator L_i (section 11); velocities always 3 components
CollideState:   vremax f64[C,G,G]     # NTC majorant Sigma; read-only within a step (7.2, 8.4)
                remainder f64[C,G,G]  # only used by count_scheme="remainder" (8.1); zeros otherwise
SortCarry:      order i32[N], key i32[N]   # previous step's sorted order (partition sort, 6.2)
CellTable:      order i32[N]  starts i32[C]  counts i32[C]  cell i32[N]     # sort.build_table (6)
GridSpec:       lo f64[D] TRACED, hi f64[D] TRACED, n tuple STATIC,
                transverse f64 (product of suppressed extents, section 4),
                volume f64[C] (cut-cell volumes; empty = box volumes),
                n_sub STATIC, split_of / split_table / sub_parent (split cells, 4.5)
SpeciesParams:  mass f64[S]; diam, omega, tref, alpha, mr, prefactor f64[S,S]   # prefactor derived
FlowParams:     nrho f64, temp f64, vstream f64[3]
SimParams:      dt f64, fnum f64[C] (per cell), epsilon f64      # TRACED
SimStatic:      N, D, S, G, A_max, K_bounce, n_steps, remat_policy,       # STATIC, frozen, hashable
                checkpoint_block_size, count_scheme ("stochastic"|"remainder"),
                majorant_mode ("thermal"|"certified"), tracker ("stencil"|"lane"|"walk"),
                K_walk, lane_size, sort ("full"|"partition"), record ("full"|"lean")
Diagnostics:    per-step counters, never branched on inside jit:
                attempts_clipped, max_attempts, majorant_violations, min_majorant_margin,
                decision_checksum, wall_recrossed, inflow_* (injected, shortfall, requested,
                escaped, sampler_failed, candidate/storage overflow, tracking_exhausted),
                count_support_boundary, configuration_invalid, surface_hits, stencil_overrun,
                walk_exhausted, inside_body, lane_overflow, record_overflow, surface_stuck
```

`SimConfig` of the original spec is split into `SimParams` (pytree, traced) and `SimStatic`
(passed as a static argument); `SimStatic.__post_init__` rejects values that would give a wrong
shape instead of an error (`G <= S`, positive trip counts, known option strings). Counters that
indicate a *silent* failure (`attempts_clipped`, `majorant_violations`, `lane_overflow`,
`record_overflow`, `inflow_*overflow`, `surface_stuck`, `inside_body`, `configuration_invalid`)
must be zero in every physics and gradient test.

### 3.2 Why each static field is static

| Field | Reason it cannot be traced |
|---|---|
| `N` | shape of every particle array |
| `D` | shape of `x`, and the `v[:, :D]` slice |
| `S`, `G` | shape of parameter tables and `vremax` |
| `n` | `C = prod(n)`, shape of all per-cell arrays |
| `A_max` | trip count of the attempt-slot `scan` |
| `K_bounce`, `K_walk`, `lane_size` | trip counts / buffer sizes of the wall-bounce passes |
| `n_steps` | trip count of the time `scan` |
| `n_sub` | number of split-cell sub-cells (extent of every per-cell array) |
| `tracker`, `sort`, `record`, `count_scheme`, `majorant_mode` | select code paths at trace time |

Everything else is traced, including `dt`, `fnum`, `lo`, `hi`, and every entry of `SpeciesParams`
and `FlowParams`: this is what makes "gradients flow to everything" literally true.

### 3.3 `fnum` as a per-cell field, and the RWF question

[DF] routes every use of the particle weight through `nParticles(cell) = F_N * RWF(cell)`
(`dsmcAxisymmetricI.H:55-58`); SPARTA carries `cinfo.weight`. Radial weighting separates into
two independent halves with very different retrofit costs.

**Half 1 — the weight field. Adopted, traced.** `fnum : f64[C]`, read by the NTC attempt count
(8.1), moment normalisation (10), and boundary-flux measurement. Retrofitting it later would be a
simultaneous diff across `ntc.py`, `sample.py` and `boundary.py`; doing it now costs a `(C,)`
broadcast where a scalar stood. It is a traced leaf, so `dJ/d(fnum_c)` exists.

**Half 2 — the clone/delete machinery. Not adopted.** [DF] clones a parcel moving to a cell of
different RWF with probability `RWF_old/RWF_new - 1` (inward) or keeps it with probability
`RWF_old/RWF_new` (outward) (`dsmcAxisymmetric.C:50-219`). Birth and death are not the obstacle
(the static-capacity + `alive` mask already handles them). The reasons are that (1) RWF only
matters for axisymmetric/spherical meshes and every case is Cartesian, and (2) cloning splits a
particle's future in two, which breaks the pathwise adjoint's particle correspondence — the same
count-changing sensitivity the ensemble adjoint handles, so the hard part rides on infrastructure
that exists for inflow.

> **Loaded gun, now guarded.** Shipping the field without the machinery means a particle crossing
> from a cell with `fnum_a` to one with `fnum_b` silently represents the wrong number of real
> molecules, and mass is not conserved. The invariant "`fnum` must be spatially uniform in any run
> where particles cross cells (legal non-uniform: `C == 1` or reweighting)" is enforced by
> `state.check_fnum_uniform` (host-side, called by `initialize.suggested_a_max`;
> `tests/test_state.py::test_check_fnum_uniform_*`) and by the in-jit
> `Diagnostics.configuration_invalid` flag set by `collide` for non-finite, non-positive or
> non-uniform `fnum`. The *failure itself* (a mass drift under non-uniform `fnum`) has not been
> demonstrated; see U7.

---

## 4. `jax_dsmc/grid.py` — the mesh interface

**Deliberately narrow.** Milestone 5 replaces the implementation behind these four names with
cut/split cells and general particle tracking. Nothing outside this module may do cell-index
arithmetic; if `collide.py` ever computes `floor((x - lo) / dx)` itself, the M5 rewrite breaks it.

The four functions that constitute the interface:

```
cell_of(x, grid)      -> i32[N]        # which cell each particle is in
cell_volume(grid)     -> f64[C]        # volume of each cell
cell_bounds(grid)     -> (f64[C, D], f64[C, D])
cell_centers(grid)    -> f64[C, D]
```

Uniform orthogonal implementation (M1-M4):

```
def spacing(grid):
    return (grid.hi - grid.lo) / array(grid.n)        # f64[D], TRACED

def cell_of(x, grid):
    dx  = spacing(grid)
    idx = floor((x - grid.lo) / dx)                   # f64[N, D]
    idx = clip(idx, 0, array(grid.n) - 1).astype(i32) # i32[N, D]
    # row-major ravel; strides are static because n is static
    return ravel_multi_index(idx, grid.n)             # i32[N]

def cell_volume(grid):
    dx = spacing(grid)
    # `transverse` is the product of the extents of the 3 - D SUPPRESSED axes.
    return full((C,), prod(dx) * grid.transverse)     # f64[C], TRACED
```

> **`prod(dx)` alone is wrong for `D < 3`, and M2 is where it bites.** A
> `D`-dimensional run is not a `D`-dimensional gas: it is a three-dimensional gas
> that happens to be uniform in the `3 - D` directions being suppressed -- which is
> exactly why velocities stay 3-component whatever `D` is (§0.2). So cell *volume*
> must remain a true volume. Without `transverse` it is a **length** at `D = 1` and
> an **area** at `D = 2`, and since the NTC attempt count carries `fnum / V` (§8.1)
> and `nrho` carries `count * fnum / V` (§10), a wrong `V` corrupts the collision
> rate and the density together. It does **not** cancel: rescaling `fnum` to fix the
> rate leaves `nrho` wrong by the square. Measured after the fix: `nrho` reads
> `1.0000e23` against `1e23` set, in `D = 1`, `2` and `3` alike.
>
> SPARTA arrives at the same place from the other side -- its domain is always
> three-dimensional and a "1-D" run is one cell in `y` and `z` with real extents.

**Embedded surfaces (2026-09-17, `jax_dsmc/mesh/`).** The four names above are unchanged;
`cell_volume` returns `grid.volume` when that per-cell array is set (`mesh.cut.with_volume`),
and `cell_of` is used as-is — the cut-cell machinery is a *separate* per-cell table set
(`Mesh`: `cell_type`, `volume`, `candidates[C, S_max]`, `csurfs[C, S_cell]`,
`outside_corner`, element `normals`/`areas`) built once on the host (`mesh/cut.py`, SPARTA
`cut2d.cpp` / `cut3d.cpp` / `grid_surf.cpp` semantics: inclusive clip test, corner parity
along an irrational ray, stratified sub-grid flow fractions, split cells refused -- since
2026-09-23 split into sub-cells, §4.5) and a
surface-aware `mesh.geometry.first_hit(x, v, grid, mesh, exclude)` that returns the box result
unless an element (`2D + e`) or the analytic body (`2D`) is hit strictly earlier.  Every table
is indexed by the flat cell id and padded with `-1`, so a hierarchical grid later replaces only
`cell_of` / `cell_bounds` (see PLAN Phase 5 further note).  `neighbour_cell(cell, axis, side,
grid)` is the one addition to this module: the cell-walk tracker's neighbour resolution.

### 4.1 Gradient behaviour, stated honestly

`cell_of` contains a `floor`, so `d(cell_of)/dx = 0` almost everywhere. Cell membership is a
frozen decision (section 0.5). Gradients reach the mesh geometry through two other routes only:

1. **`cell_volume`** feeds the NTC attempt count `nattempt ~ 1/V` (section 8.1) and therefore the
   collision rate. This sensitivity is real and nonzero — **but only because §8.1 uses stochastic
   rounding and scores the count decision.** With the deterministic `floor` of revision 2 it was
   measured to be *exactly* zero: the path dies at the `floor` and no per-pair score term
   mentions `cell_volume`. The gradient here travels through `log P(n | f)`, not through `n`.
2. **moment normalisation** in `sample.py` divides by `cell_volume`.

What pathwise AD does *not* capture is the delta-function contribution from a particle crossing
a cell boundary as the boundary moves. That is the same class of omission as the frozen
accept/reject decision, and it is why adjoint-driven AMR is deferred to M5 rather than claimed
now: doing it properly needs either a boundary term or a smoothed deposition, and that decision
should be made against the M5 mesh, not guessed at against this one.

Returning `cell_volume` as a full `(C,)` array rather than a scalar is deliberate: it costs
nothing now and means the non-uniform mesh of M5 is a drop-in replacement.

### 4.2 Tests (`D in {1, 2, 3}` where parametrised)

There is no `tests/test_grid.py`; the interface is covered where it is used:
`tests/test_initialize.py` (cell lookup and volumes through the initial state),
`tests/test_sample.py` (`test_cell_index_agrees_with_the_table`,
`test_gradient_wrt_mesh_geometry_is_nonzero`), `tests/test_mesh_geometry.py`,
`tests/test_mesh_cut.py` (cut volumes, split cells). The properties to keep pinned:

- round trip: `cell_centers(grid)[c]` maps back to `c` under `cell_of`;
- a particle exactly at `lo` lands in cell 0, exactly at `hi` in cell `C-1` (the `clip`);
- `sum(cell_volume(grid)) == prod(hi - lo) * transverse` to machine precision;
- `grad(sum(cell_volume))` w.r.t. `lo`/`hi` is finite and matches the analytic box derivative.

### 4.4 Level-set bodies (`jax_dsmc/mesh/levelset.py`, 2026-09-20)

A third body kind next to the polygon and the analytic circle/sphere: `LevelSet(coef
f64[m_1..m_D] traced; lo; h static; n_sample, n_bisect static)`, the shape representation of the
optimiser. phi is a uniform **cubic B-spline** (C2; the normal `grad phi/|grad phi|` is C1), body
`phi < 0`, normal into the flow, single surface id `2D`, model row `2D` exactly as the analytic
body, so tracking, tallies, kernel scores and the replayed shape VJP (`shape.geometry` returns
`coef`) reuse the analytic path.

- **Evaluation.** `u = (x - lo)/h`, `i = floor(u)`; one `4^D` block gather and `D+1` separable
  contractions give `phi` and `grad phi`. The valid domain must contain the simulation box
  (`check_covers`).
- **Hit.** Along `x + s v_hat`, `s in [0, reach]`, `reach = n_sample * h/2`: the first `+ -> -`
  sign change between consecutive samples brackets the entry root; a start with `phi <= 0`,
  `|phi| <= tol`, `v . grad phi < 0` hits at `t = 0`; a start deeper inside returns `inf` (the
  post-step inside check counts it). No `exclude` is needed: after a reflection phi increases
  along `v_out`. Only rays whose cells' coefficient minima (`block_min_table`, exact by the
  B-spline convex hull) allow `phi <= 0` are searched, compacted. The bracket is refined by a
  safeguarded Newton iteration on `stop_gradient` inputs, then **one Newton step from the
  bracket** restores the implicit-function derivative `dt/dcoef = -B_j(x_hit)/(v . grad phi)`
  without reverse mode tracing the loops. Measured: surface and hit errors fall as `h^4`;
  tunnelling only for chords shorter than one sample spacing (~5e-5 of the circle's hits,
  grazing, carrying no momentum); counted by the post-step inside check, not solved.
- **Host tables** (`cut.py`): a cell is CUT iff the coefficients supporting it change sign
  (conservative, never misses); `stencil` is the same test on the dilated cell; volume fractions
  by sub-sampling; `exact_flow_volume` is not available.
- **Construction.** `from_analytic`, `from_surface` (unsigned distance, sign from host ray
  parity; linear in the knot values), `interpolate` (separable `(1,4,1)/6` system, differentiable),
  `reinitialise(ls, n_iter, band)` (closest-point projection from knots within the band, refit).
- **Silhouette term** (`shape.silhouette_cotangent`, `gradient(silhouette_band=...)`): for each
  survivor the closest approach of its undeflected flight, `phi_min = min_s phi(x + s v_hat)`
  (`levelset.closest_approach`). Particles with `|phi_min| < eps_b` change outcome when the
  surface moves by more than `phi_min`; the term is the VJP of `phi(x*)` with cotangent
  `-(V_hit - V_miss)/(2 eps_b)` (band on both sides estimates the density of `phi_min` at zero).
  Specular walls: `V_hit -> V_miss` at grazing, the term is O(eps_b) -> 0; diffuse walls: reproduces
  (new hits per unit radius) x (absorbed momentum) to 0.04 % at `eps_b = 0.02`
  (`tests/test_mesh_adjoint.py::test_silhouette_term`).
- **Grazing regularisation** (`LevelSet.grazing`, static `eps`; `levelset.regularised_quotient`):
  the hit-time derivative's `1/(v . n)` is evaluated with `|v . n| >= eps |v| |n|` in the tangent
  only (primal exact). Its mean converges but its variance diverges logarithmically at grazing
  incidence; measured seed sd 8.4 % -> 2.9 % at eps 0.1 with bias within noise. Default for the
  optimiser and cluster study: `eps = 0.05`. The same `eps` floors the emission side
  (`sampling.log_diffuse_density(..., floor)`, the `v_t/v_n` part of the diffuse normal score).
- **Hybrid normal term** (diffuse element hits): the hit's own reward is exact in `v_out`, so
  `wall_adjoint._element_events` forms the surface tally from the **traced** sample (reparameterised,
  bounded weights) while the post-hit state and events carry the frozen sample; the normal score is
  weighted by the value-to-go after the hit. The frame is parallel-transported from the primal one
  (`sampling.diffuse_velocity_general`). Measured on the GP plate: normal-term seed sd 3.1 (score)
  -> 0.77e-21 (hybrid) at no bias.

---

### 4.5 Split cells (SPARTA sub-cells, 2026-09-23)

A cut cell whose flow is disconnected (a notch of an optimised shape cutting a pocket off,
a plate thinner than a cell) is a **split cell**, as SPARTA's `nsplit > 1`
(`cut2d.cpp:163-305`, `grid_surf.cpp:157-204`, `update.cpp:1753-1798`): its largest flow
region keeps the cell's index, every other region is a sub-cell `prod(n) + j` with its own
volume, particles and collisions.

- `GridSpec` carries static `n_sub` and integer tables `split_of i32[prod(n)]` (row or −1),
  `split_table i32[S, sub^D]` (the collision cell of every stratum of a `sub^D` sub-grid
  of each split cell), `sub_parent i32[n_sub]`; `num_cells = prod(n) + n_sub`.  Without
  split cells `n_sub = 0` and every shape and random stream is unchanged.  Sub-cell slots
  come in blocks of 4 (unused ones have zero volume) so an optimiser recompiles rarely.
- `grid.collision_cell(x, grid)` = `cell_of`, or inside a split cell the region of the
  particle's stratum.  The per-cell **physics** uses it (the collision table, moments, the
  adjoint's node tables and queries, the host `_cells`); the **geometry** keeps the
  Cartesian `cell_of` (candidate lists, the stencil test, the cell walk, the inside test).
- The regions come from the split detection itself (`cut.split_regions`): the stratified
  samples, joined when the segment between them crosses no element, re-attached through
  uncrossed segments, components below `min_samples` unresolved; body and sliver strata
  take the region of the nearest resolved flow sample.  Region volume = the cell's cut
  volume × the region's share of the flow samples.  Departure from SPARTA: a raster
  lookup, not the segment test to `xsplit` -- consistent with our sub-sampled cut volumes;
  a particle within ~h/sub of a surface may be counted in the neighbouring region's
  collision cell (tracking is unaffected).
- Pooled fits: a sub-cell pools its parent's block with itself at the centre.  Outputs
  (`.vtr`, `viz.field_to_grid`, the optimiser's frames) show the Cartesian cells, i.e. each
  split cell's largest region.
- `outside_corner` (the polygon inside test's reference corner) is required only for
  polygonal surfaces: a level set / analytic body keeps `−1` in a cut cell whose corners are
  all in the body (a flow pocket entering through a face), since its inside test is exact.

## 5. `jax_dsmc/sampling.py` — reparameterised distribution draws

Every sampler here is written as a **deterministic differentiable function of a
parameter-independent random draw**. This is the reparameterisation trick, and it is what makes
`d/dT` and `d/dvstream` pathwise-exact. [P3] states the requirement explicitly: initial particles
are generated as `(x_0i, v_0i) = Psi(xi_i; m)` where `xi_i`'s distribution does not depend on `m`.

### 5.1 Shifted Maxwellian — Gaussian form

Two references, two constructions of the same distribution. SPARTA
(`create_particles.cpp:761-780`) uses a Rayleigh magnitude with a uniform phase, giving two
Gaussian-like Cartesian components per pair of uniforms. [DF] (`dsmcCloud.C:1043-1051`) draws
three independent standard normals directly. **We follow [DF].**

```
def thermal_speed(temp, mass):
    return sqrt(K_BOLTZ * temp / mass)                # per-component sigma

def sample_maxwellian(key, n, temp, mass, vstream):
    # returns f64[n, 3]
    return vstream + thermal_speed(temp, mass) * normal(key, (n, 3))
```

The two forms are statistically identical — both yield three independent `N(0, kT/m)`
components. Check on SPARTA's: with `vscale = sqrt(2kT/m)` and `vn = vscale*sqrt(-log U)`,
`E[(vn cos t)^2] = vscale^2 * E[-log U] * E[cos^2 t] = vscale^2 / 2 = kT/m`. Same variance.

The Gaussian form is preferred because it is trig-free, needs one draw instead of four, and
**removes `sqrt(-log U)` from the initialization path entirely** — one fewer singularity to
guard, and one fewer place a `NaN` gradient can originate.

Gradient paths: `d/dtemp` flows through `thermal_speed` into all three components;
`d/dvstream` is the identity. Both exact and pathwise. No `safe_*` guard is needed here at all.

**Where the Rayleigh form survives.** Flux-weighted draws are genuinely a different
distribution, not a different construction of the same one, so they keep their closed form:

```
# diffuse wall, normal component -- [DF] dsmcDiffuseWallPatch.C:106-111
v_perp = thermal_speed_wall * safe_sqrt(-2.0 * log(1.0 - u))
# tangential components stay Gaussian
```

Note [DF] writes `log(1 - R)` rather than `log(R)`; both are correct, and the choice only moves
which end of the unit interval is singular.

### 5.2 Uniform position fill

```
def sample_positions(key, n, grid):
    u = uniform(key, (n, D))
    return grid.lo + u * (grid.hi - grid.lo)          # TRACED in lo, hi
```

(SPARTA's axisymmetric `sqrt(lo_y^2 + U*(hi_y^2 - lo_y^2))` variant,
`create_particles.cpp:698`, is out of scope; noted for completeness.)

### 5.3 Stochastic rounding

`create_particles.cpp:672-677`. Preserves fractional counts in expectation across the domain
rather than truncating them away cell by cell.

```
def stochastic_round(x, u):
    return floor(x) + (u < (x - floor(x)))            # FROZEN decision (int result)
```

### 5.4 Flux-weighted inflow (`jax_dsmc/inflow.py`)

`fix_emit.cpp:166-174`, Bird eq. 4.22. **Built; the production sampler changed.**

- *Retired as the production path:* Bird's bounded rejection scheme (`sample_normal_beta`,
  `N_TRIALS = 16` as a masked `lax.scan`, never a `while_loop`; truncated range `[max(-3,-s), 3]`
  instead of SPARTA's nested rejection loop). It is kept and tested
  (`tests/test_inflow.py::test_normal_beta_matches_the_analytic_inflow_flux`).
- *Production:* the normal flux speed is drawn through the implicit **inverse CDF**
  (`inflow.flux_quantile`, closed-form start plus `QUANTILE_ITERATIONS` safeguarded Halley steps,
  custom JVP giving the implicit derivative w.r.t. drift and `u`), tangents Gaussian, projected onto
  an arbitrary inward normal (`sample_surface_velocities`, `sample_face_velocities`). This makes the
  entrant velocity pathwise differentiable in the reservoir state and removes the acceptance
  failure mode; non-converged draws are flagged (`accepted` -> `Diagnostics.inflow_sampler_failed`).
  Evidence: `tests/test_forward_corrections.py::test_flux_quantile_*` (quadrature, bisection
  reference, tangent equals implicit derivative).

```
mol_inflow(s) = (exp(-s^2) + sqrt(pi) s (1 + erf(s))) / (2 sqrt(pi)),   s = v_stream_normal / vscale
```

### 5.5 `jax_dsmc/initialize.py` — assembling the initial state

`initial_state` allocates the static capacity `N` and sets `alive = arange(N) < n_real`, with
`np_target = nrho * sum(cell_volume / fnum)` stochastically rounded
(`create_particles.cpp:588`); positions via `sample_positions`, velocities via
`sample_maxwellian`. `target_particle_count`, `initial_collide_state` (seeds `vremax` with the
thermal majorant, 7.2.3) and `suggested_a_max` (advisory, sized from the *inflated* majorant,
8.2) are the other entry points. The static-capacity-plus-mask pattern is used everywhere a count
is data-dependent; it is the most pervasive structural difference from SPARTA, which grows arrays
freely.

### 5.6 Tests

There is no `tests/test_sampling.py`; sampling is covered by `tests/test_initialize.py`,
`tests/test_ntc.py::test_stochastic_rounding_is_unbiased`, `tests/test_inflow.py`,
`tests/test_walls.py` and `tests/test_forward_corrections.py`. Properties to keep pinned:

- moments of `sample_maxwellian` within `4 sqrt(var/n)` of `vstream`, `kT/m`; KS test of speeds;
- `grad(T -> mean(sum(v**2)))` equals the analytic `3k/m` to 1e-10 (pure reparameterisation);
- `stochastic_round` is unbiased;
- with a fixed key, changing `temp` moves every velocity smoothly (the CRN property FD relies on).

---

## 6. `jax_dsmc/sort.py` — cell occupancy table

SPARTA builds per-cell linked lists (`collide.cpp:508-512`). That is a CPU-appropriate structure;
on GPU the right primitive is a sort.


```
def build_table(state, grid):
    c      = collision_cell(state.x, grid)               # i32[N]  (split-cell aware, 4.5)
    c      = where(state.alive, c, C)                    # dead -> sentinel cell C
    order, keys = stable_cell_order(c, C + 1)            # packed (cell, slot) lax.sort   FROZEN
    bounds = searchsorted(keys, arange(C + 1))           # starts, then first dead slot
    return CellTable(order, starts=bounds[:C], counts=bounds[1:] - bounds[:C], cell=c)
```

`cell` is returned because the count score of 8.1 is a per-cell term applied to every particle in
the cell, and `moments` reuses it. Particle `j` of cell `k` is `order[starts[k] + j]`, valid for
`j < counts[k]`.

### 6.1 Why this is differentiable

`order` is an integer permutation — a frozen decision. But the *gather* `v[order]` is a linear
operation whose VJP is a scatter, so cotangents flow back to the correct particles automatically.
Nothing special is needed. The permutation being frozen is correct and matches the papers.

### 6.2 Scaling note, and the partition sort (2026-10-01)

The full sort is one `lax.sort` of packed `(cell, slot)` keys (`sort.stable_cell_order`: unique
keys, so any sort is the stable order), `O(N log N)` -- 2.2 ms per step on one CPU core at
`N = 6e4`, the largest single stage of the mesh forward after the tracker work.

`SimStatic.sort = "partition"` (`sort.update_table`; the mesh forward `examples/mesh/forward.py`
sets it) is SPARTA-JAX's partition sort.  The previous step's sorted order is carried
(`CollideState.sort: SortCarry(order, key)`, set by `simulate.run` before its scan and cleared
after it, so a run's input and output collide states have the same structure):

```
c      = cell_of(x), C for dead                  # this step's cells
knew   = c[order_prev]                           # new cell at each previous sorted position
stay   = knew == key_prev                        # kept its cell
cs     = exclusive prefix count of stay
movers = the ~stay positions, compacted (buffers N/16, N/8, N/6, N/4; more -> full sort)
start  = exclusive scan(stayers per cell + movers per cell)
stayer -> start[knew] + its rank among the cell's stayers (order kept)
movers sorted by (cell, slot) -> after their cell's stayers
```

The same cells, starts, counts and members as `build_table`, exactly; within a cell the stayers
come first in their previous order, then the newcomers by slot.  NTC indexes a cell's members
uniformly at random, so its pairs are drawn from the same distribution but not draw for draw
(`tests/test_sort.py`: membership exact over chained steps and every buffer; a collisional
relaxation over 12 seeds within 3 standard errors of the full sort).  XLA:CPU only
(`sort.partition_active`): on GPU the full sort is one fast kernel and the partition's branch
and searches cost launches.  Measured 2.2 -> ~1.0 ms per step on one CPU core.  It stays at
~6x SPARTA's linked-list build (`Particle::sort`) for the reason SPARTA-JAX's does: four passes
over all `N` (a gather, a prefix count, a scatter, an elementwise pass) at XLA:CPU's per-element
cost against SPARTA's single loop (`sparta_jax/README.md` on the `sparta-match` branch).


### 6.3 Tests (`tests/test_sort.py`, `tests/test_ntc.py`, `tests/test_sample.py`)

`tests/test_sort.py` covers the partition sort only (membership exact over chained steps and every
buffer tier; carry type; collisional relaxation within 3 standard errors of the full sort). The
full-sort properties are exercised through the collision and moment tests: every alive particle
appears once in `order[:sum(counts)]`, `counts` matches a bincount of the collision cell, dead
slots sort past `sum(counts)`, and `table.cell` agrees with the cell lookup
(`test_sample.py::test_cell_index_agrees_with_the_table`).

---

## 7. `jax_dsmc/collide/vss.py` — Variable Soft Sphere kernel

### 7.1 Cross-section prefactor

Precomputed once from the species table, `collide_vss.cpp:122-125`:

```
def make_prefactor(species):
    cxs = pi * species.diam**2
    return cxs * (2*K_BOLTZ*species.tref / species.mr)**(species.omega - 0.5) \
               / gamma(2.5 - species.omega)                    # f64[S, S], TRACED
```

Traced in `diam`, `tref`, `omega` — so `dJ/d(VSS params)` has a path here, and another through
the deflection law in 7.3. This is the model-calibration gradient.

### 7.2 The collision rate `q` and the acceptance probability `h`

`collide_vss.cpp:200-232`. The quantity SPARTA calls `vre` is the physical
`q = sigma_T(v_r) * v_r`:

```
def collision_rate(vi, vj, isp, jsp, species):
    vr   = vi - vj                                    # f64[3]
    vr2  = sum(vr * vr)
    # RELATIVE clamp (2.3): scaled to the pair's reference thermal energy, not an
    # absolute 1e-30, which would leave d(q)/d(vr2) unbounded at ~2.5e16.
    eps  = EPS_REL_VR2 * (2.0 * K_BOLTZ * species.tref[isp, jsp] / species.mr[isp, jsp])
    # sigma_T * v_r = prefactor * (v_r^2)^(1 - omega)
    return species.prefactor[isp, jsp] * safe_pow(vr2, 1.0 - species.omega[isp, jsp], eps)
```

The NTC acceptance test compares `q` against the per-cell majorant `Sigma = vremax`, **held
fixed for the whole step**:

```
def acceptance_prob(q, Sigma):
    return clip(q / stop_gradient(Sigma), 0.0, 1.0)
```

> **Design decision — the stale-max rule.** The two reference codes disagree here, and the
> disagreement matters.
>
> SPARTA ratchets the majorant *live* inside the candidate loop and tests against the updated
> value: `vremax = MAX(vre, vremax); if (vre/vremax < random) return 0;`
> (`collide_vss.cpp:225-231`). [DF] does not. It captures `Sigma` before the loop, tests every
> candidate against that captured value, and publishes the new maximum only for the *next* step
> — and documents the reasoning inline (`noTimeCounter.C:231-246`): `N_pairs` was itself
> computed from the old `Sigma`, so raising `Sigma` mid-loop yields both too few candidates and
> a lower acceptance rate. Under-collision twice over.
>
> **We follow [DF].** Beyond being more statistically consistent, it makes `Sigma` genuinely
> loop-invariant within a step — which is exactly the **fixed** majorant that [P2]'s
> virtual-collision framework assumes and that its unbiasedness argument (§5.6) rests on. Under
> SPARTA's rule the `log h = log(q/Sigma)` score term would be differentiating against a
> quantity that moved during the very loop it is scoring.
>
> `stop_gradient` still appears, but now it says something clean: `Sigma` enters the step as
> carried state from the *previous* step, so a cross-step gradient path exists even though the
> intra-step one is gone. We cut it, because `Sigma` is an algorithmic device for exact
> rejection sampling, not physics.
>
> Both parts are testable: the Algorithm-3 oracle in section 13 disagrees if `stop_gradient` is
> dropped, and a dedicated test asserts `Sigma` is bitwise unchanged across the attempt loop.

#### 7.2.1 Majorant violations bias the gradient — and the stale rule causes them

`clip(q/Sigma, 0, 1)` saturates when `q > Sigma`. At saturation `p = 1`, so `log h = log(1) = 0`
**with zero derivative**, and because `u` is drawn on `[0,1)` the reject branch is unreachable —
the violated attempt contributes *literally nothing*, with no error and no diagnostic. Measured
(`scratch/probe_num_c2a_zero.py`): `|d log h/dv| = 0.786` at `q = 0.949 Sigma`, and **exactly
`0.0`** at every `q >= Sigma`.

This is worse than a dropped term. It **biases** the gradient:

| `Sigma / max q` | violating pairs | mean `dJ/dtheta` | z vs. truth (`-2.94e-1`) |
|---|---|---|---|
| 0.70 | 15.4% | `-7.44e-02` | **+12.9** (75% low) |
| 0.90 | 1.06% | `-2.54e-01` | +2.4 (14% low) |
| >= 1.0 | 0% | `-2.85e-01 … -2.99e-01` | \|z\| <= 0.64 |

> **This is the real cost of the stale-max rule (18.1), and it needs stating plainly.** SPARTA's
> live ratchet raises `Sigma` *before* the test, so `q > Sigma` is impossible **by construction**
> — measured `0/12000` violations in every regime. The stale rule permits them: `145/12000`
> (1.2%) isothermal, `191/12000` (1.6%) under 5%-per-step heating, from a deliberately coarse
> seed. We traded a guaranteed-valid majorant for a fixed one.
>
> The trade is still right — a `Sigma` that moves inside the loop it is scoring is a category
> error in the estimator, not a small correction — **but it is only right if `Sigma` is kept a
> genuine majorant.** That requires the diagnostic below and a safety factor, not good luck.

**Required diagnostic.** `diag.majorant_violations = sum(q > Sigma)`, alongside
`attempts_clipped`, asserted zero in every physics and gradient test. One `sum` in the scan.

#### 7.2.2 `Sigma`'s value is a variance knob — and the dangerous end is a *tight* majorant

Revision 2 said `Sigma` affects "only the efficiency". That is true of the forward distribution
and **false for the adjoint** — but not in the direction one might guess. The reject score is
`d log(Sigma - q)/dv = -(dq/dv)/(Sigma - q)`, which is singular as `Sigma -> max q from above`,
*not* as `Sigma -> infinity`. Measured with the attempt count scaled physically as `n ~ Sigma`
(`scratch/probe_num_c2d_variance2.py`, 8192 seeds):

| `Sigma / max q` | `n_att` | var of `dJ/dtheta` | relative |
|---|---|---|---|
| 1.000 | 16 | **1.498** | 4.60x |
| 1.010 | 16 | 1.212 | 3.72x |
| 1.100 | 18 | 0.942 | 2.90x |
| 2.000 | 32 | 0.470 | 1.44x |
| 10.00 | 160 | **0.325** | 1.00x |
| 100.0 | 1600 | 0.302 | 0.93x |

Bias is flat throughout (`|z| <= 0.64`); variance **falls 4.97x** from `Sigma = max q` to
`100 max q` and plateaus by about `10x`. The `sqrt(n) * (1/Sigma) ~ Sigma^(-1/2)` scaling means
drawing more attempts against a looser majorant *shrinks* the reject sum's fluctuation.

**Consequence for the default.** Both the violation bias (7.2.1) and the variance (here) push the
same way: keep `Sigma` comfortably above `max q`. Target a safety factor of **2-5x**, which
costs proportional compute, eliminates violations, and sits on the variance plateau. A majorant
that hugs `max q` is the configuration to avoid — it is simultaneously the most likely to be
violated and the noisiest.

#### 7.2.3 How the safety factor is actually obtained (U3, closed)

7.2.1 requires `Sigma` to be "actively kept above `max q` by a 2-5x safety factor". U3 measured
whether that is achievable when `max q` is not known ahead of time. **It is — but not from
SPARTA's seed, and not from a ratchet.** Full evidence in §17.5; the rule that follows is binding
on §8.

A pure ratchet cannot deliver a margin, because it only ever raises `Sigma` *after* a violation
has already been counted and biased the gradient. Measured with SPARTA's own seeding
(`2 * cxs * vscale`) plus the stale-max ratchet: `1.1e-3` violation rate in equilibrium,
`1.0e-2` under counter-streaming beams, minimum margin **0.62** — i.e. `Sigma` sat *below* the
true maximum.

The fix is to make the majorant **a priori**. `q(vr) = prefactor * (vr2)^(1-omega)` is monotone in
`|vr|`, so over the reachable interval its supremum sits at an endpoint, and the endpoint is bounded
by the cell temperature — which is known at the **start** of the step (it is sampled anyway), so
unlike an observed maximum it does not lag a transient:

```
K_EV  = 5.0     # extreme-value multiplier: P(|vr| > K_EV * v_ref) * npc^2/2 ~ 1.6e-4   (ntc.py)
F_UP  = 2.0     # safety factor; the 2-5x band of 7.2.1

def thermal_majorant(temp, omega, tref, mr, prefactor, k_ev=K_EV, f_up=F_UP):
    """A-priori upper bound on q from the cell temperature.  f64[C].  Used under stop_gradient."""
    vref2 = 2.0 * K_BOLTZ * temp / mr
    hi    = prefactor * safe_pow(k_ev**2 * vref2, 1.0 - omega, EPS)       # omega < 1: top end
    lo    = prefactor * safe_pow(EPS_REL_VR2 * 2*K_BOLTZ*tref/mr, 1.0 - omega, EPS)  # omega > 1: clamp
    return f_up * maximum(hi, lo)

vremax_init(temp0, omega, tref, mr, prefactor, n_cells, g_groups)    # NOT SPARTA's 2*cxs*vscale (fails U3)

# per step, inside collide, before the attempt scan (section 8.4):
Sigma = stop_gradient(maximum(cstate.vremax[:, 0, 0], thermal_majorant(cell_temp, ...)))
```

`Sigma` stays loop-invariant within the step, so [P2]'s fixed-majorant assumption is untouched;
the `stop_gradient` is unchanged from 7.2. The carried `vremax` still ratchets on the observed
`qmax` for the next step (§8.4) — the thermal term is a *floor* under it, not a replacement.

Measured with this rule: **zero** violations across all six IC x transient combinations
(equilibrium / cold / counter-streaming beams, each with and without a 3%-per-step heating
transient), minimum margin **2.15**, median **2.28**.

**The margin costs compute, not physics.** `n_attempts ~ Sigma` while `P_accept = q/Sigma`, so the
expected collision count is `Sigma`-invariant — measured flat (`1676, 1655, 1649, 1694`) over
`Sigma/Sigma_0 = 1, 2, 5, 10`. What it costs is scan slots, which is why **`A_max` must be sized
against `F_UP * thermal_majorant`, not against the cold-start `Sigma`** (U4). And since gradient
variance *falls* with `Sigma` (7.2.2), the factor buys margin and variance reduction together.

### 7.3 Mapping NTC onto the papers' virtual-collision framework

[P2] §2.3 defines three mutually exclusive outcomes per selected pair, with probabilities
`h_1, h_2, h_3`. Our NTC realises exactly this:

| [P2] outcome | Our realisation | `h` | Contributes to `logh_acc`? |
|---|---|---|---|
| `j=1` no virtual collision | slot `a >= nattempt`, or `np < 2` | `P(n \| f)` | **yes — at cell level**, see below |
| `j=2` virtual, and real | attempt with `U < q/Sigma` | `q/Sigma` | **yes**, `log(q/Sigma)` |
| `j=3` virtual, not real | attempt with `U >= q/Sigma` | `1 - q/Sigma` | **yes**, `log(1 - q/Sigma)` |

The `j=3` branch is easy to miss: **a rejected candidate still contributes a score term**,
`d log(Sigma - q)/dv` in [P2] eq. `back_eta`. Omitting it gives a subtly biased gradient that
still passes a naive smoke test. The Algorithm-3 oracle catches it.

The `j=1` row is the one that *was* missed. A pair goes unselected because `a >= nattempt`, and
`nattempt` is itself a random, parameter-dependent draw — so the decision *how many pairs to
select* carries its own score, `log P(n | f)` from §8.1. It is not a per-pair term (there is no
pair); it is a **per-cell** term, and it is where the entire `dt` / `fnum` / `cell_volume`
sensitivity lives. Revision 2 marked this row "no score contribution" and thereby lost three of
the headline gradient paths outright — measured as exactly `0.0`, not merely inaccurate. §8.1
documents the measurement and the fix.

```
def log_h(q, Sigma, accepted):
    p = acceptance_prob(q, Sigma)
    return where(accepted, safe_log(p), safe_log(1.0 - p))
```

`log_h` is differentiable in `v_i`, `v_j` (through `q`) **and** in `diam`, `omega`, `tref`
(through `prefactor`). Both paths are wanted; both are picked up by the surrogate in section 11.

### 7.4 Scattering

`collide_vss.cpp:405-452`. Two deviations from SPARTA, both deliberate:

1. **SPARTA's `if (fabs(alpha_r - 1.0) < 0.001)` fast path is dropped.** `alpha` is a traced
   parameter, so a Python branch on it is illegal, and the general frame-aligned branch is
   already correct at `alpha = 1` (it reduces to isotropic scattering in the `v_r` frame, which
   is statistically identical to isotropic in the lab frame). SPARTA's special case is a
   performance shortcut, not different physics.
2. **SPARTA's `if (d > 1.0e-6)` frame-degeneracy branch becomes the safe-argument pattern** of
   section 2, preserving the numerics while keeping the gradient finite.

```
def scatter(vi, vj, isp, jsp, sp, u_chi, u_eps):   # draws arrive batched (vss.scatter_key splits a key)
    mi, mj = species.mass[isp], species.mass[jsp]
    mr     = species.mr[isp, jsp]
    a_inv  = 1.0 / species.alpha[isp, jsp]

    eps  = 2*pi * u_eps
    # u_chi is a dimensionless uniform on [0,1), so a plain small clamp is right
    # here -- the relative-to-thermal-scale form of collision_rate does not apply.
    cosX = 2.0 * safe_pow(u_chi, a_inv, EPS_U) - 1.0   # VSS angular law
    sinX = safe_sqrt(1.0 - cosX*cosX)

    vr   = vi - vj
    vr2  = sum(vr*vr)
    vrm  = safe_sqrt(vr2)

    # No internal energy yet, so etrans is unchanged and |v_r'| == |v_r|.
    # scale is identically 1.0 here; kept as the hook for Larsen-Borgnakke,
    # where scale = sqrt(2*etrans_post / (mr * vr2)).
    scale = safe_sqrt(safe_div(2.0 * (0.5*mr*vr2), mr * vr2, fallback=1.0))

    # frame aligned to v_r; collide_vss.cpp:429-437
    d      = safe_norm(vr[1:3])
    # RELATIVE threshold. SPARTA's `if (d > 1.0e-6)` is dimensional and only works
    # because SPARTA is SI with thermal speeds ~1e2 m/s. See the note below.
    d_safe = where(d > EPS_D * vrm, d, 1.0)
    ok     = d > EPS_D * vrm

    ua_g = cosX*vr[0] + sinX*d*sin(eps)
    vb_g = cosX*vr[1] + sinX*(vrm*vr[2]*cos(eps) - vr[0]*vr[1]*sin(eps)) / d_safe
    wc_g = cosX*vr[2] - sinX*(vrm*vr[1]*cos(eps) + vr[0]*vr[2]*sin(eps)) / d_safe

    # degenerate fallback: v_r along x
    ua_d = cosX*vr[0]
    vb_d = sinX*vr[0]*cos(eps)
    wc_d = sinX*vr[0]*sin(eps)

    vr_new = scale * stack([where(ok, ua_g, ua_d),
                            where(ok, vb_g, vb_d),
                            where(ok, wc_g, wc_d)])

    # mass-weighted CM decomposition; collide_vss.cpp:446-452
    vcm = (mi*vi + mj*vj) / (mi + mj)
    return vcm + (mj/(mi+mj))*vr_new, vcm - (mi/(mi+mj))*vr_new
```

**Conservation is structural, not approximate.** Momentum: the two offsets are mass-weighted and
opposite about a shared `vcm`, so `mi*vi' + mj*vj' == mi*vi + mj*vj` identically. Energy:
`|v_r'| == |v_r|` by construction (`scale == 1`), and relative kinetic energy in the CM frame
depends only on `|v_r|`. Both hold to floating-point round-off regardless of `alpha` or `omega`.

> **Why the frame threshold is relative.** SPARTA's `if (d > 1.0e-6)` is **dimensional**, and it
> only works because SPARTA runs in SI with thermal speeds around 1e2 m/s. Under any
> nondimensionalisation the branch fires on different physical configurations. Measured on the
> same physical state `vr = (V, theta*V, 0)` in three unit systems
> (`scratch/probe_num_c1_harm.py`):
>
> | `d/|vr|` | SI (`V=340`) | nondim (`V=1`) | CGS (`V=3.4e4`) |
> |---|---|---|---|
> | 1e-6 | general | **degenerate** | general |
> | 1e-9 | **degenerate** | degenerate | general |
>
> The consequence is not subtle. `max |vr_new/V - ref|` across those three unit systems:
>
> | `d/|vr|` | `d > 1e-6` (absolute) | `d > 1e-6 * vrm` (relative) |
> |---|---|---|
> | 1e-2 | 1.11e-16 | 1.11e-16 |
> | 1e-7 | **1.28e+00** | 5.55e-17 |
> | 1e-9 | **1.28e+00** | 5.55e-17 |
>
> An **order-unity** difference in post-collision velocity purely from a change of units. The
> relative form is exactly scale-invariant, and it also caps the scatter Jacobian at `1/EPS_D`
> in every unit system, where the absolute form lets `||J||` reach `9.5e7` at `V = 340`. `vrm`
> is already computed one line above, so this costs nothing.

### 7.5 Note for the adjoint

`sigma`, the post-collision relative-velocity direction, is built **in the frame of `v_r`**, so
`sigma` genuinely depends on the pre-collision velocities. [P1] §4.3 flags exactly this as the
reason its Maxwell-molecule result does not extend directly ("additional terms in
`d w_k'/d w_k`"), and defers it. Reverse-mode AD differentiates the code above as written and
picks these terms up with no special handling — this is `D` in [P2] eq. `back_D`, which the
paper is careful to note "may not be the same" as the Maxwell-molecule `A`.


### 7.6 Tests (`tests/test_vss.py`, `tests/test_gaps.py`, `tests/test_adjoint_oracle.py`)

- momentum and energy conserved over random pairs for `alpha` and mass-ratio sweeps
  (`test_momentum_and_energy_conserved`, `test_relative_speed_is_preserved`);
- `cos chi` follows `2U^(1/alpha) - 1` (KS) and `alpha = 1` is isotropic;
- degenerate inputs give finite values and gradients; `scatter` is unit-scale invariant
  (`test_scatter_is_scale_invariant`, the relative `EPS_D`);
- `h_accept + h_reject = 1`; `acceptance_prob` stops the gradient through `Sigma`; `log_h` is
  differentiable in `v` and in `(diam, omega, tref)` (finite-difference checked);
- the prefactor matches SPARTA's formula and the mean rate at `T_ref` equals the hard-sphere rate
  for six values of `omega` (21.4);
- `tests/test_gaps.py` pins the U1 clamp bound and the U3 majorant margin (19).

---

## 8. `jax_dsmc/collide/ntc.py` — No Time Counter driver

### 8.1 Attempt count — stochastic rounding, and why it is not optional

`collide_vss.cpp:137-161`. SPARTA offers three schemes: `mcflag` (`n ~ Poisson(f)`, has a score
term), `remainflag` (`n = floor(f + remain)`, carry the fraction, **no** score term; SPARTA's
default, `collide.cpp:87`), and `collide_modify remain no` (`n = floor(f + U)`, stochastic
rounding, has a score term). We take stochastic rounding:

```
f    = 0.5 * npc * (npc - 1) * Sigma * dt * fnum / volume          # f64[C] TRACED
up   = u < (f - floor(f));   n = floor(f) + up                     # FROZEN (int)
logP = where(up, log(frac), log(1 - frac))                         # per-cell score term
```

(`ntc.attempt_count(counts, vremax, volume, dt, fnum, key)`; zero-volume body cells get no
attempts; `logP` is frozen at `frac == 0`, where there is no two-sided likelihood derivative,
counted by `Diagnostics.count_support_boundary`.)

> **Why not the deterministic carry (revision-2 defect, measured).** With `n = floor(f)` and a
> `remain` carry, `jax.grad(J)` w.r.t. `(dt, fnum, volume)` was `[0.0, 0.0, 0.0]` exactly while the
> velocity path read `9.5`. `f` is traced but `n` is an integer, so the path dies at the `floor`,
> and no per-pair score mentions `dt`, `fnum` or `volume`: they live *only* in `f`. Under a carry
> `P(n | f, remain)` is degenerate, so there is no probability to differentiate. Stochastic
> rounding makes the count a random, parameter-dependent decision whose score `logP` carries the
> sensitivity.

**Unbiasedness check.** Rao-Blackwellising the count decision at a fixed configuration gives a
zero-variance truth (`J(n=3) = 11.99258`, `J(n=4) = 12.59169`, so `dE[J]/d(dt) = 8.087997e4`); the
stochastic-rounding estimator converges to it (`8.373e4 +- 1.19e4` at 1e5 samples, z 0.24).
`tests/test_ntc.py::test_defect1_regression_count_score_carries_dt_fnum_volume` and
`test_log_p_is_the_log_probability_of_the_realised_count` keep this pinned.

> **The count score needs a baseline when used alone.** It carries `1/frac` and `1/(1-frac)`
> factors and is heavy-tailed: `b = 0` gives sd `3.76e6`, a mean-`phi` baseline `4.50e4` (84x).
> The 84x does not transfer to windowed spatial objectives (11.4), where the score is a minority
> of the variance.

Options kept for comparison: `SimStatic.count_scheme = "remainder"` reproduces SPARTA's carry
(`CollideState.remainder`, no score, no expectation gradient; `score_mode="trajectory"` refuses it),
and `rounded_weak_derivative` evaluates the coupled count-law derivative for oracle checks.
Stochastic rounding is unbiased from step one, so [DF]'s `U[0,1)` seeding of the carry
(18.5) addresses a transient we do not have.

> **`[:, 0, 0]` is a `G == 1` placeholder.** `collide` raises unless `S == G == 1`. The
> multi-species form is a *different formula*, not a broadcast: SPARTA uses `0.5 n_i (n_i - 1)`
> within a group but `n_i n_j` across groups (`collide_vss.cpp:165-196`).

### 8.2 Static shape strategy

`n` varies per cell and is data-dependent, so it cannot size an array. We pad to the static bound
`A_max` and mask:

```
active(c, a) = (a < n[c]) and (counts[c] >= 2)
```

`A_max` is a correctness-relevant knob: if `max(n) > A_max` the surplus attempts are silently
dropped and the gas under-collides. This must never be silent, so `collide` returns a diagnostic:

```
diag.attempts_clipped     = sum(maximum(n - A_max, 0))
diag.max_attempts         = max(n)
diag.majorant_violations  = sum(active & (q > Sigma))   # see 7.2.1 -- biases the gradient
diag.min_majorant_margin  = min(where(active, Sigma / q, inf))  # target >= 2, see 7.2.2
```

surfaced by `summary()` and asserted to be zero in every physics test.

> **`A_max` must be sized from the *inflated* `Sigma`, not the cold-start one.** `vremax` never
> decays (§16, open item 2), so a thermal transient permanently inflates it. Measured on a `T: 1 -> 4 -> 1`
> pulse (`scratch/probe_num_c2e_vremax.py`): after the pulse `Sigma / max q = 2.29`, held
> forever, with mean acceptance falling `0.340 -> 0.151`. Since `n ~ Sigma`, that same 2.29x
> lands directly on the attempt count — and an `A_max` chosen at cold start is then exceeded, so
> the surplus attempts are **silently dropped**. That is under-collision produced by a
> *diagnostic-free* interaction between two separately-reasonable choices. Size `A_max` from the
> worst-case inflated `Sigma`, and keep `attempts_clipped` asserted zero.

### 8.3 Pair selection — a pluggable strategy

Both reference codes ship more than one pairing scheme ([DF] `noTimeCounter` vs.
`noRepeatCollisions`; SPARTA `partners nearcp`/`subcell`), and the papers use a third. We expose
two behind one interface, because the oracle tests need a different one from production.

```
pair_ntc (key, counts, slot) -> (i, j)     # with replacement    [default]
pair_perm(key, counts)       -> (i, j)     # disjoint pairing    [oracles]
```

**`pair_ntc` — with replacement.** SPARTA / [DF] `noTimeCounter`, the production default and the
correct scheme for spatially inhomogeneous flow, where the attempt count must be set by the
physical rate rather than by the particle count. SPARTA resamples until `i != j`
(`collide.cpp:530-533`) — a `while_loop` with a data-dependent trip count, neither
reverse-differentiable nor statically shaped. Closed-form equivalent:

```
def pair_ntc(key, npc):
    # i uniform on {0..npc-1}; j uniform on {0..npc-1} \ {i}
    # npc is CLAMPED: jax.random.randint(k, shape, 0, 0) silently returns 0 rather
    # than erroring, and (i+1+r) % 0 returns 0 too, so an empty cell would produce
    # i == j == 0 and index into a neighbouring cell. See 8.4.1.
    npc_safe = maximum(npc, 2)
    i = randint(k1, 0, npc_safe)
    r = randint(k2, 0, npc_safe - 1)
    j = (i + 1 + r) % npc_safe
    return i, j                                       # FROZEN (ints)
```

`r` ranges over `npc-1` values and `(i + 1 + r) mod npc` covers every index except `i` exactly
once, so `j` is uniform on the complement of `{i}`. Verified by chi-square in section 17.

**`pair_perm` — disjoint pairing.** [DF] `noRepeatCollisions`, and the scheme both papers
actually assume: [P2] eq. `factorizationEa` writes `E_a^k = Prod_{i=1}^{N/2} E_ai^k`, a product
over `N/2` disjoint pairs, and [P1] Algorithm 1 says "select `N_c/2` collision pairs uniformly
among all possible pairs **without replacement**". Simpler than `pair_ntc`:

```
def pair_perm(key, npc):
    perm = permutation(key, npc)                      # FROZEN (ints)
    return perm[0::2], perm[1::2]                     # npc//2 disjoint pairs
```

The oracle tests of section 13 run under `pair_perm`, so they are faithful implementations of
Algorithms 2 and 3 rather than approximations of them applied to a different pairing scheme.
`pair_perm` is not the production default because it fixes the collision count at `npc/2` per
cell per step, decoupling it from `sigma_T c_r * dt` — correct only for a carefully tuned `dt`.

> **Not adopted: sub-cell partner selection.** [DF]'s *default* biases selection toward spatially
> nearby pairs via a fixed 8-octant subdivision about the cell centroid, and
> `noTimeCounterAdaptiveSubcelling` re-sizes a sub-cell grid to hit a target parcels-per-subcell
> count; SPARTA offers `partners subcell`/`nearcp`. This genuinely improves accuracy at coarse
> resolution by reducing the mean collision separation.
>
> It is nonetheless an adjoint hazard. [P2] §3.1.2 passes the velocity derivative through `E_p`
> — the expectation over pair selection — precisely because "`E_p^k` also commutes with velocity
> derivatives", which holds only while pair selection is independent of the particle state.
> Octant sub-celling makes the selection probability a function of **position**, so `E_p` no
> longer commutes with position derivatives and the inhomogeneous adjoint acquires a further
> score term of its own. Recorded as a known accuracy-versus-adjoint-complexity trade-off, not
> taken. If it is ever wanted, the cost is a third `log h` contribution, not a redesign.


### 8.4 The attempt-slot scan — the central parallelisation decision

The collision loop is sequential within a cell (a particle may be selected more than once and
each collision sees the previous one's velocities), so naive vectorisation over attempts would
change the physics. The structure that preserves the ordering while being GPU-parallel is a
**`scan` over the attempt slot, vectorised across cells** (`ntc.collide`, body `ntc._slot`):

```
Sigma  = stop_gradient(max(cstate.vremax[:,0,0], thermal_majorant(cell_temp, ...)))   # read ONCE
n, logP = attempt_count(counts, Sigma, volume, dt, fnum, key_for(.., STAGE_NTC_ROUND))
# every stage draws its randomness for ALL slots in one call, shape (A_max, C), slot-addressed:
u_i, u_r = indexed_randint(.. STAGE_PAIR ..)   u_acc = indexed_uniform(.. STAGE_ACCEPT ..)
u_chi, u_eps = indexed_uniform(.. STAGE_SCATTER ..)

def _slot(carry, xs, n, table, sigma, ...):       # module level, bound with functools.partial
    v, logh, qmax, nviol, margin, chk = carry
    a, u_i, u_r, u_acc, u_chi, u_eps = xs
    active = (a < n) & (counts >= 2)                                   # bool[C]
    i, j   = pair_from_draws(u_i, u_r, counts)                         # npc clamped >= 2
    pi, pj = where(active, order[starts + i/j], N)                     # inactive -> scratch slot N
    vi, vj = v.at[pi].get(mode="fill", fill_value=0), ...
    q      = collision_rate(vi, vj, ...);  p = acceptance_prob(q, Sigma)   # FIXED Sigma
    accept = active & (u_acc < p)                                      # FROZEN
    v      = v.at[pi].set(where(accept, vi_new, vi), mode="drop") ...  # same for pj
    lh     = where(active, log_h(q, Sigma, accept), 0);  logh = logh.at[pi].add(lh).at[pj].add(lh)
    qmax   = where(active, max(qmax, q), qmax)                         # masked running max
    nviol += sum(active & (q > Sigma));  margin = min(margin, min(Sigma/q over tested pairs))
    chk   += decision fingerprint (13.4.1)

(v, logh, qmax, nviol, margin, chk) = scan(_slot, init=(v, logh, Sigma, 0, inf, 0), xs)
logh += where(alive, logP[table.cell], 0)           # the COUNT score: per-cell, every particle
vremax_next = max(cstate.vremax, qmax)              # published for the NEXT step only
```

Why it is correct and fast: sequential within a cell; parallel across cells; `Sigma` is
loop-invariant (closed over, not carried; `qmax` never feeds back into this step's acceptance);
writes are disjoint by construction; the trip count is static so the scan is reverse-differentiable
and rematerialisable (keys are recomputed from `(base_key, step, slot)`). The `active` mask on
`qmax` matters: an inactive slot still evaluates `q`, but from a pair that was never drawn, and
letting it into the maximum would inflate the majorant and silently lower acceptance thereafter.

Further options on `collide`: `score_mode="trajectory"` stores one scalar score prefix in every
slot (each event counted once; see ADJOINT_CORRECTIONS.md), `capture=True` additionally returns
the per-slot `PairRecord` (pair, accept bit, pre- and post-collision velocities) that the ensemble
adjoint replays, and `majorant_mode="certified"` replaces the thermal rule by a closed energy
bound (`certified_majorant`, needs `0.5 <= omega <= 1`).

#### 8.4.1 Why the scratch slot is mandatory

Revision 2 called `pair_ntc` for every cell and asserted "no write conflicts". **False, and it
silently corrupted results**: with `A_max = 3`, 176 of 300 runs disagreed with a serial reference
and violated momentum conservation. The chain, all silent:

1. `jax.random.randint(key, shape, 0, 0)` returns zeros without error in JAX 0.6.2 (2000/2000
   seeds), as does `(i + 1 + r) % 0`. An empty cell therefore gets `i = j = 0`.
2. For an empty cell `starts[c] == starts[c+1]`, so `order[starts[c]]` is the *next cell's* first
   particle; with trailing empty cells `order[N]` is clamped by JAX to `order[N-1]`.
3. The clobber is deterministic, not a duplicate-index race: `vi` and `vj` are both gathered from
   the same pre-scatter `v`, so when `pi == pj` (every inactive cell) the second `.set` writes a
   stale value over whatever the first left, undoing a legitimate collision of the true owner.

The revision-2 loop was also not self-consistently differentiable (VJP accumulated into duplicate
indices while the forward kept one: `grad` vs FD relerr `6.3e-3`; after the fix `1.1e-10`).

**Fix.** `npc_safe = maximum(counts, 2)` in the pair draw, and inactive cells address the scratch
slot `N`. *Implementation (current):* the scratch slot is no longer a physical `(N+1)`-row array;
reads at index `N` use `mode="fill"` (zeros) and writes use `mode="drop"`, giving the same values
bitwise without copying the array through the scan (`ntc._gather_rows`). Corruption went `176/300
-> 0/300`, results match the serial reference, and the gradient is FD-exact. The original
`(N+1)`-row form cost 3-10 % wall time (the smallest measured configuration was launch-bound noise:
ratios 1.02/1.40/1.69 from identical runs; only the two larger rows were meaningful).

### 8.5 Tests (`tests/test_ntc.py`)

- `pair_ntc` uniform on the complement of `{i}` and `j != i` (`test_pair_ntc_is_uniform_on_the_complement`);
  clamped for degenerate counts (`test_pair_ntc_clamps_degenerate_cell_counts`);
  `pair_perm` is a perfect matching (`test_pair_perm_is_a_perfect_matching`);
- stochastic rounding unbiased and `logP` is the log-probability of the realised count;
  the count is frozen (`test_attempt_count_itself_is_frozen`);
- **defect-1 regression**: the count score carries `dt`, `fnum`, `volume`
  (`test_defect1_regression_count_score_carries_dt_fnum_volume`);
- **defect-2 regression**: empty cells do not clobber real collisions
  (`test_defect2_regression_empty_cells_do_not_clobber`), momentum and energy conserved,
  dead particles never collide;
- majorant: thermal rule bounds every equilibrium pair, `vremax_init` is the thermal rule, no
  violations (`test_collide_reports_no_majorant_violations`), Maxwell margin is exactly `F_UP`;
- jittable without recompiling, vmappable over an ensemble, no data-dependent `while_loop`;
  scores finite and reach colliding particles;
- structure of the compiled program: `tests/test_perf.py` (no RNG inside the slot scan, batched
  draws shaped `(A_max, cells)`, jaxpr identical across particle counts).

### 8.5.1 How `A_max` is actually chosen: measure it, do not guess it

`A_max` is the static trip count of the attempt scan, but the right value is a property of the
flow, and no a-priori rule survived contact with a real case: sizing from the cold uniform attempt
count was 5.4x too small at an M2 shock time; adding the majorant ratchet (1.10x) was still short;
bounding compression by Rankine-Hugoniot (16x) was *wrong* because a symmetric converging wave is
not a steady normal shock (measured density ratio 11). The decomposition at the shock time is
ratchet 1.10x, temperature 1.10x, **compression (`n ~ npc^2`) ~4.5x**, so the inflated-majorant
note of 8.2 is minor once transport is in the loop.

**Resolution: `simulate.run_adaptive`.** Restarting costs nothing (state is just arrays). The run
proceeds in chunks; after each, `attempts_clipped` is read on the host and, if capacity was
exceeded, `A_max` grows (`growth`, `max_resizes`) and **that chunk alone** re-runs from the state
it began with. `run(..., step0=)` offsets the RNG step index so chunking changes nothing (x, v
bitwise identical to an unchunked run; `logh_acc` to 6e-14 relative, float reassociation).
Measured: starting from `A_max = 64` it converged in two resizes in 41.9 s against 48.6 s for a
single over-provisioned run. Not for use inside a differentiated run (the resize is host-side
control flow): calibrate with `run_adaptive`, then differentiate through `run` at the settled
`A_max`. Slot-addressed draws (section 1) keep earlier attempts unchanged when `A_max` grows.
(No test in `tests/` exercises `run_adaptive` directly; the bit-equality claim is from the
original measurement. See U4.)

SPARTA has no `A_max` (a CPU loop costs nothing) and its `reset_vremax()` runs only at `init`;
SPARTA's `mcflag` Poisson count has a tail, making `A_max` harder, not easier.

### 8.5.2 Rejected alternative: fixed attempt count with rescaled acceptance

Recorded because it was verified and will be proposed again. Make exactly `A` attempts per cell
and accept each with `p = (n/A)(q/Sigma) = 0.5 npc(npc-1) dt fnum q / (A V)` (`Sigma` cancels).
Verified in `scratch/probe_fixed_attempts*.py`: collision count unchanged within 0.4 %, variance
not worse, 294 slots instead of 673 needed, and it carries the `dt`/`fnum`/`V` gradient with **no
count score** (z 0.82). **Not adopted**: it changes the fundamental NTC acceptance probability;
multi-species NTC needs a per-group-pair `n` and `Sigma` (`collide_vss.cpp:163-196`) so the
equivalence would have to be re-derived per group pair, for Larsen-Borgnakke and for chemistry.
The gain over `run_adaptive` is a 2.3x slot saving. Revisit only if slot count becomes binding and
only after the multi-group derivation.

### 8.6 Measured cost of the attempt-slot scan

Measured in 2026-09 on a laptop GPU (RTX 3050 Ti) and one CPU; absolute numbers are historical, the
shapes of the costs are the durable content. Any single timing implying a qualitative change must
be reproduced on an idle device (a phantom 36x "cliff" at `A_max = 8192` came from a concurrent
GPU job), and performance claims carry repeat counts and spreads.

- **The parallel axis is cells, not attempt slots.** At `N = 128k`, `A_max = 512`: 118 / 161 / 111
  / 188 us per slot for `C` = 1 / 64 / 512 / 4096 (4096x the work for 1.6x the time). At fixed
  `npc = 32`, `N` from 2k to 1M: ns per particle 19,637 -> 47. Below ~1e5 particles the kernel is
  launch-latency bound.
- **A single cell (the 0-D BKW case) is the worst case**: `A_max ~ N`, hence thousands of
  sequential tiny iterations; still `O(N)`, not `O(N^2)`. Splitting a homogeneous gas across
  cells preserves the physics (`npc` and the volume fall together) while `A_max`, which is per
  cell, falls: 1 / 8 / 64 cells -> 16.2 / 2.6 / 1.3 s per seed, 12.7x faster; `examples/bkw/`
  defaults to 8 cells.
- **Pinning the cell count while `N` grows is a scaling defect**: `A_max ~ npc` grows and the
  sequential axis lengthens while the parallel one stays fixed. Holding `npc` fixed instead:
  `N = 6.4e5` in 0.68 s against 26.4 s pinned (39x); `N = 1e7` in 10.6 s on a 4 GB GPU, ~1 us per
  particle for `N >= 6e5`.
- **CPU vs GPU crossover is ~3e5-5e5 particles** (below it the GPU is launch-bound and the CPU
  wins by up to 4.5x; above it the GPU wins 3-4x). The fitted rate was identical on both backends
  to every printed digit, an independent check that the RNG contract is device-independent.
- **Section 1 rule 3 is a performance requirement**: hoisting draws out of the scan took compile
  time 17.0 -> 2.4 s and per-slot cost 102 -> 53 us.
- **Audit by ablation, not reading** found: `moments` called every step but only `temp` used
  (2.9-3.5x; now `sample.cell_temperature`, a single segment sum about the population mean);
  `moments` recomputing the cell lookup although `table.cell` has it (1.6-3.0x). *Not* real: the
  scratch-row concatenation (XLA fuses it). Scan carry copy: real only at `C == 1` (52x growth of
  per-slot cost for `N` 64e3 -> 8e6), flat for `C >= 8`; `fori_loop`, `.at[].add` and compact
  workspaces were no better, so no change was made.

---

## 9. `jax_dsmc/move.py` and `jax_dsmc/boundary.py` — advection and BCs

On a uniform grid with no embedded surfaces, SPARTA's general cell-walking loop
(`update.cpp:937-1029`) collapses to its own `OPT` fast path (`update.cpp:764-864`): advance the
full step, apply the face BC, recompute the cell by arithmetic. There is no walk, no per-crossing
face test, and no neighbour lookup.

```
def move(state, grid, dt, bc):                         # move.py: box faces only
    x_new = state.x + state.v[:, :D] * dt             # only the first D components
    x, v, alive = boundary.apply(x_new, state.v, state.alive, grid, bc)
    return replace(state, x=x, v=v, alive=alive)
# walls, inflow and embedded bodies go through boundary.apply_faces / wall_adjoint (9.4, 9.6)
```

### 9.1 Periodic — `domain.cpp:301-330`

```
L = grid.hi - grid.lo
x = grid.lo + mod(x_new - grid.lo, L)
# v unchanged
```
Adjoint: identity in both `alpha` and `beta`, matching [P3] §3.1. `mod` is differentiable a.e.
with derivative 1, which is correct — the wrap is a translation.


### 9.2 Specular — closed-form multi-bounce fold

`domain.cpp:337-357` mirrors once and assumes at most one crossing. [P3] §3.2 makes the same
assumption ("we neglect the possibility that a particle... may cross the boundary more than
once"). We do better at no cost: a triangle-wave fold handles **any** number of bounces in one
branch-free expression.

```
y     = (x_new - grid.lo) / L                         # f64[N, D]
m     = floor(y)                                      # fold index, FROZEN
frac  = y - m
even  = (mod(m, 2) == 0)
pos   = where(even, frac, 1.0 - frac)
x     = grid.lo + pos * L
sgn   = where(even, 1.0, -1.0)                        # f64[N, D]
v     = v.at[:, :D].multiply(sgn)
```

Verification by hand: `x_new = lo + 1.2L` gives `m=1` (odd), `frac=0.2`, `pos=0.8`, so
`x = lo + 0.8L` — the correct reflection of a `0.2L` overshoot past `hi` — and `v` flips.
`x_new = lo - 0.3L` gives `m=-1` (odd), `frac=0.7`, `pos=0.3`, so `x = lo + 0.3L`, again correct.

Adjoint: `d x/d x_new = sgn` and `d v/d v = sgn`, i.e. `C = diag([-1, 1, 1])` on the crossed
axis applied to **both** the position and velocity adjoints — exactly [P3] eq. `C_ki`. Reverse-
mode AD of the code above produces this with no special handling.


### 9.3 Outflow — `domain.cpp:294-295`

```
inside = all((x_new >= grid.lo) & (x_new <= grid.hi), axis=-1)
alive  = state.alive & inside
```
Dead particles keep their array slots (static shapes) and are sorted to the sentinel cell.


> **Slot reuse resets per-particle accumulators (gap U6, closed).** Through M4 particles are only
> destroyed, so `alive` is monotone non-increasing. Inflow refills dead slots, and a refilled slot
> **must** have `logh_acc` zeroed or the new particle inherits the score history of the previous
> occupant. `inflow.revive_slots` (mask) and `inflow.revive_at` (slot list, used by `inject`) are
> the *only* sanctioned ways to set `alive` False -> True and both zero every field in
> `ACCUMULATOR_FIELDS` (checked by reflection over `ParticleState`). The monotone-alive test is
> replaced by a mass balance (`injected - deleted == change in alive count`,
> `inflow.mass_balance_residual`), asserted at every step of a 400-step inflow run.
> **The obvious test proves nothing:** `magic(L) = exp(L - stop_gradient(L))` has gradient `dL/dtheta`,
> so a stale *constant* differentiates to zero; the gate makes `logh_acc` a genuine function of
> `theta` and predicts the contamination, `sum(inherited * phi[dead]) / N`, to 1e-10
> (`tests/test_inflow.py::test_revived_slot_gradient_equals_a_fresh_particles`).

### 9.4 Diffuse / thermal — closed form, not a sub-move scan

[P3] eq. `thermal_v_L`/`thermal_v_R` gives the whole bounce as one expression,
`v_{k+1} = g`, `x_{k+1} = L + (x' - L)/(P v') * (P g)`: the overshoot time `(x' - L)/(P v')` is
re-advected with the **new** normal velocity. The mesh stays static and every face rule is a
closed-form function of the end-of-step position. The flux bias lives in the half-Maxwellian draw
(`sampling.sample_flux_normal`; `surf_collide_diffuse.cpp:206-320`:
`v_perp = v_rm(T_wall) * safe_sqrt(-log U)`), reparameterised in `T_wall`, with `vwall` added after
the draw, so `dJ/dT_wall` and `dJ/dv_wall` are pathwise.

*Retired design:* a bounded `scan` of `K_bounce` masked sub-moves carrying `dtremain` (SPARTA's
approach). Both were measured: they agree with the closed forms for `PERIODIC`/`SPECULAR` to 2e-19
and reach the same wall equilibrium; the closed form ran a 60-step 1e5-particle case in 1.65 s
against >7 min.

Multiple crossings: neither [P3] nor SPARTA's specular handles them. Measured at `L = 1e-3`, 8
cells, argon 300 K, the fraction crossing two faces in a step is 0 up to `dt/(dx/v_th) = 1`
(production 0.15), 1e-6 at 2, 4.2e-3 at 4, 8.5e-2 at 8. A single pass is therefore right and the
multi-crossing case is **detected and reported** (`wall_recrossed`) rather than iterated. In
`D >= 2` a particle can leave through two *different* faces; `boundary.n_bounce_passes` is a static
function of the face layout (two thermal faces on different **axes** need two passes; one axis,
e.g. Couette, needs one). Each pass leaves the particle at its wall with `t_used` advanced and a
final advance by `dt - t_used` completes the step. Thermal on every face: `D = 3` equilibrium
309.9 K -> 300.90 K against a 300 K wall; Couette (periodic `x`, thermal `y`) and shock (outflow
`x`, specular `y`) have 0 recrossings. **Order inside `apply_faces` is load-bearing: wall bounce
first, exact periodic/specular/outflow folds second.**

### 9.4.1 M7 boundary scores, and what each parameter costs

> **2026-09-15:** the score-function routes here are superseded for decision-driven parameters by
> the ensemble adjoint ([ADJOINT_BRANCH_REVIEW.md](ADJOINT_BRANCH_REVIEW.md) section 5). The table
> stays as the record of what the pathwise and score routes can and cannot do.

Built (all default-off, so earlier forward results are unchanged): the crossing-event score
(`boundary.crossing_score`, knob `params.epsilon`; 0 gives exactly 0 since `log 1 = 0`), the
accommodation Bernoulli score and its relaxed form (`acc_delta`), cell-wide attribution
(`spread_score_over_cell`, `score_spread="cell"`), and the inflow injection-count score. The
per-step boundary score is added to the *same* `logh_acc` as the collision score, since
`surrogate` weights the reward by the gradient of the whole prefix.

| parameter | route | status |
|---|---|---|
| `T_wall`, `v_wall` | pathwise | verified: fixed-key FD to 3e-12 / 3e-11, and against the Couette closed form |
| inflow `nrho` | score | verified: ensemble FD/AD = 1.04 |
| `acc` | score, or pathwise with `acc_delta > 0` | score form biased 2.8x at `M = 40` |
| shock `d(rho_2/rho_1)/du_1` | score | wrong sign |

A score-carried gradient is **biased low, growing with trajectory length** (1.13x at `M = 2` to
2.82x at `M = 40`); the cause is reward attribution, not the log-probability (with collisions off
the estimator is exact, |z| = 0.18). `docs/ATTRIBUTION.md` is the reference.

### 9.4.2 Two limits that bound every gradient here

1. **Attribution bias** (score route only), above.
2. **Chaotic amplification** (every pathwise gradient). A single trajectory's derivative grows like
   `exp(lambda t)` while the estimated quantity stays O(1): the per-seed spread of `du_s/du_wall`
   grows 9,254x between 10 and 800 steps. The estimator never becomes wrong, it becomes
   uninformative. Consequently **particle count does not fix a gradient** past the horizon where the
   chaotic part dominates (at 200 steps the gradients are flat in `N`, `N^+0.00 / N^-0.10`, while the
   forward still scales `N^-1/2`); only ensemble size helps, at `M^-1/2`.

Validation: `examples/couette/gradients.py` (continuum Couette with Maxwell slip, `dT/dT_wall = 1`)
and `examples/shock/gradients.py` (Rankine-Hugoniot).

### 9.6 Embedded surfaces: the element branch and the two trackers (2026-09-17)

`wall_adjoint._advance_impl(..., mesh, tracker, max_walk)` extends the chronological tracker
of 9.4 to elements.  Per pass, `first_hit` is the minimum over the box faces and the
particle's candidate elements (`mesh.candidates[cell_of(x)]`, the stencil) with SPARTA's
rules: a hit needs `v . n < 0` (normals point into the flow), the element just left is
excluded until the next box-face event (`exclude i32[N]`, `update.cpp`), ties go to the box
face, then the lowest element id, no push-off epsilon.  The element branch
(`_element_events`) gathers the flat normal (exact for an analytic body), applies the model
row `2D + model` of `WallParams` — specular `v - 2 (v . n) n`, or SPARTA's diffuse law in the
`(n, t1, t2)` frame (`sampling.diffuse_velocity_general`) with the accommodation Bernoulli
drawn once per pass like the box faces — records every element hit in `WallEvents` under
its surface id, and tallies per surface (`SurfaceTallies` has `2D + E` rows).  The stencil's
sufficient condition `|v| (budget - elapsed) <= radius * h_min` is checked every pass and
counted as `stencil_overrun`.  `tracker = "walk"` (`mesh/walk.py`) is SPARTA's cell walk:
`max_walk` iterations of {earliest element of the cell's own list strictly before the exit
face -> reflect; else exit face -> box rule or step to the neighbour and reset `exclude`; else
done}; unfinished particles are discarded and counted (`MAXSTUCK`).  Both end with the
inside-body parity test (`inside_polygon`, SPARTA `outside_surfs` toward the cell's known
outside corner) that kills and counts a mis-tracked particle.  The fourth return value is a
`TrackerCounts` whenever a mesh is present; `simulate.run(mesh=..., emit_surface=True)`
appends the per-step tallies to the output.  Newborn tracking in `inflow.inject` takes the
same path (entrants' birth-step hits are recorded, not tallied).


### 9.6.1 The lane tracker: SPARTA's move loop (2026-09-24)

`tracker = "lane"` (the default of the mesh examples, `K_bounce = 32`) keeps reflecting a
particle until its time step is used up, as SPARTA's `Update::move` does, instead of
stopping after a fixed number of passes over every particle.

```
pass 0 (GPU, _first_pass_dense): the stencil pass over all N slots (first_hit,
          _apply_events)                                                  -- as 9.6
pass 0 (XLA:CPU, _first_pass_routed, 2026-10-01 -- SPARTA's optimised move):
          dense:  one elementwise pass over all N -- near = the path |v| dt may reach
                  the body (mesh_geometry.near_tables: per fine cell, a free path below
                  which no move from there can meet the surface or end inside it;
                  levelset.free_path from block_min_table for a level set); box = the
                  straight end point may leave the box.  Box candidates (compacted) take
                  their first box event in closed form (_box_events); a diffuse or
                  periodic face or a second box event joins near
          near:   the near slots compacted (chunks of N/16) run the stencil pass 0
          the post-move inside check runs on the near slots only
          (the same results as the dense pass 0: tests/test_mesh_tracking.py
          test_lane_dense_pass_and_chunks_equal_the_stencil, both variants)
lane:     slots with a pass-0 event (alive after it: an outflow ends the flight),
          compacted by cumulative sum into B = lane_capacity(N) entries (N/32, >= 256;
          overflow -> deleted + Diagnostics.lane_overflow, an error)
passes 1 .. K-1 on the lane only:
          the same _bounce body on the gathered (x, v, elapsed, exclude, ...);
          re-emission draws per slot, counter-based (fold_in(pass key, slot):
          wall_adjoint._slot_uniforms), so a particle meets the same randomness in
          any row of any tracker and the result equals the stencil tracker's with
          the same K;
          a pass runs only while some lane particle had an event in the previous one
          (an untraced move: while_loop; the traced replay: fori_loop + lax.cond, which
          reverse mode can differentiate)
scatter back; the post-loop query over all N (exhausted test, conditional-time bound);
still unresolved -> deleted + Diagnostics.surface_stuck (SPARTA MAXSTUCK)
```

Events are `SplitEvents(head: WallEvents(1, N), tail: WallEvents(K-1, B), slot i32[B])`;
a dense record is the special case `tail = passes 1.., slot = arange(N)` (`as_split`), so
every consumer reads one layout: per-event functions run on head and tail separately,
`particle_sum` folds a per-event array onto the slots, the shape adjoint's reward-to-go
(`shape._to_go`) takes the suffix sums across head -> tail, and a lane replay returns its
rewards as `shape.HitRewards(values, slot)` (the device backward reads the per-step totals
`HitRewards.totals()`, the host bookkeeping `ensemble_backward.from_device` expands the events per step).  Entrants
use the same tracker; `inflow.inject` expands their split events over its entrants, and
`simulate._event_record` splits the record again (the entrants' later passes in an extra
tail segment).  The **stored** record compacts once more: `wall_adjoint.RecordEvents` keeps
pass 0 dense and the later passes as a list of the events actually recorded (`slot`,
`pass_`, `face`, `v_out`, `diffuse`; capacity a quarter of the lane,
`simulate._record_capacity`; a full list is counted, `Diagnostics.record_overflow`, an
error) -- nothing downstream needs the empty rows, since the shape adjoint's replay
recomputes its events; `shape.hit_ahead` and the host `ensemble_backward._step_walls` read
the list.  The replay keeps only what its consumers read: `HitRewards(values, slot)` with
the per-slot totals formed on demand (`totals()`), and `shape_terms` keeps the timing
rewards as totals only and skips the host bookkeeping (`device_backward(host_adjoint=
False)`).  Measured on the roughest optimised shape (review 7.10,
`examples/mesh/tracker_bench.py`): the same results as the stencil to round-off, 2.1x
faster at 32 passes than the stencil at 4, 27 % less adjoint memory per slot.


**2026-10-21 fix: the post-loop unresolved-particle query.** After the lane passes, a particle is
still *unresolved* if its remaining move could meet the surface. `wall_adjoint._lane_distance` now
queries the surface along each lane particle's **remaining move** (`budget - elapsed`), not the
search reach. A level set's hit search samples its length at `n_sample` points, so over the reach
(0.5 m at `h = 0.0625`) the spacing is 3 cm and a re-entry into a sliver of the body closer than that
went unseen: the particle was not counted unresolved and its final move ended inside (optrules2 M 10
iterate 68: 32 specular bounces along a concave stretch, the 33rd hit 1 cm away, `inside_body`
fired). Over the remaining move the end point is itself a sample, so a move that ends inside is
always bracketed.

### 9.7 The mesh-aware adjoint: value-form rewards and the replayed wall map (2026-09-18)

Additions around the ensemble core (`ensemble_device.py` takes two optional inputs; the collision
pull-back is untouched):

* **Value-form rewards.** `backward[_virtual](..., phi_step f64[M,N], hit_reward f64[M,K,N])`. The
  carried value at step `k`'s post-collision state includes the reward collected there, so
  `phi_step[k]` joins `mu` before the step's pull-back (`_add_step_reward`); the rewards of the
  step's hits join on the move pull-back (`_move_pullback`: `mu_prev = lam + sum_j r_j` on
  survivors; a newborn's go to its source score). A hit's kernel-score weight is the reward-to-go
  **from that draw onward** (`lam + r_j + r_{j+1} + ...`): the hit's own reward is a function of
  the drawn velocity and must be in the weight (leaving it out made the G2 shear gradient read
  -0.0004 against an FD of -0.0165).
* **`grad_x` from the fit.** `field_query` returns `(value, grad_x, grad_v)`; `node_gradients`
  evaluates them at every alive slot's own state.
* **The replayed wall map (`jax_dsmc/shape.py`).** Under the RNG contract the move of step `k` is
  a pure function of the record's previous post-collision state; `shape.wall_map` re-runs it with
  `trace=True` and `shape.gradient` is one `lax.scan` applying `jax.vjp` w.r.t. the geometry
  (`AnalyticBody`, `Surface.points`, or `LevelSet.coef`) with the cotangents the backward provides
  (`grad_x lambda`, `grad_v lambda` at each survivor's post-move state, `lam + r` on the move's
  log-probability, the objective's weights on tallies, and `(lam + r_after) grad_n log q(v_out|n)`
  on each diffuse element hit). The diffuse re-emission velocity is `stop_gradient` in the post-hit
  state (a kernel's parameters and normal enter the value only through its score) but the surface
  tally uses the traced sample (the hybrid normal term, 4.4). `ShapeTerms` reports pathwise and
  score parts separately plus `replay_error` (relative round-off, < 1e-12; a flipped discrete
  outcome reports 1; runners hold the continuous part to `REPLAY_TOL = 1e-9`). `shape.replay`
  evaluates per-hit rewards for the backward. Not in the VJP by design: the cut-volume dependence
  of the attempt count (gate G5) and element-assignment jumps of a flat-normal polygon.
* **The hit-timing band term** (`shape.timing_cotangent`, `gradient(timing_jump=, timing_band=)`).
  The pathwise VJP differentiates a *fixed* hit set; a reward that jumps at a hit (a
  velocity-dependent per-step reward, or a per-hit reward at the window edges) also depends on
  *which step* a hit falls in. Estimated from hits within a band `Delta*dt` of the step end
  (`-jump/(Delta dt)` on each band hit's traced `time`, then through the same VJP). Measured on G3
  (plane, windowed `v_y^2`): pathwise alone `+0.82 +- 0.23` vs FD `+0.05 +- 0.15`; with the band term
  `-0.43 +- 0.02`, closing the gap. Rule: a value-form objective is either continuous across hits or
  carries this term.
* **Fit-bias findings (G3, review section 7.6).** With the plane collisionless the exact value
  gradient is a replica average, so the fitted `grad lambda` was testable: the two-sided fit is
  biased (-0.030 +- 0.011 on +0.0030), the one-sided fit much less so. The estimator with the exact
  gradient equals the *truncated* FD, not the full FD: the difference is the capture-start state's
  own theta-dependence (gate G6's horizon term), for a translated wall the accumulated delay of every
  earlier bounce. Per-hit sensitivities of a translation are time delays
  (`dx_post/dtheta = (v_in - v_out)/v_n`), so translation-type controls are poorly resolved while
  radius-type controls (exact tally term) are FD-consistent to a few per cent.
* **One-sided fits** (`mesh.geometry.nearest_normal`, `shape.local_normals`,
  `backward_virtual(normals=)`): a query's nodes are restricted to the sign of `v . n` of the nearest
  surface within one cell width. Halves the slope error and cuts value error 20-30 % on the receding
  population; `normals=None` is bitwise the old two-sided estimator.
* **Pooled fits and the along-flight identity** (`node_gradients(pool=)`,
  `ensemble_device.pooled_tables`, `grid.cell_neighbourhood`; `shape.along_flight`,
  `gradient(lam_timing=)`, `shape.gradient_variants`). In collisionless free flight `V(x - s v, v)`
  is the value of the same particle delayed by `s`, so `v . grad_x V = -T` with `T` the value of the
  *timing rewards*, carried by a second value-form backward; the cotangent keeps the fitted
  perpendicular part and replaces the component along `v` by `-T` (collisions add `-Q*V + d_t V`,
  neglected at <= 0.3 % collisions per particle-step). Along-flight removes the bias (plane `v_y^2`:
  z -2.9 -> -0.7), pooling removes variance (4-7x on the circle radius) but keeps a bias. **Not
  resolved:** the x/v-part of a translated flat wall.
* **Stratified fits (2026-09-22; `shape.hit_ahead`, `value_class`).** The one-sided rule separates
  about-to-hit and just-left populations at a flat wall only; at a cap the fitted `d lambda/d v_x`
  of a reflected particle reads the mixture. The record holds every slot's future, so class =
  "a body hit at a later captured step" is exact. Specular capsule `0.964 +- 0.037 -> 1.356 +- 0.011`
  vs CRN FD 1.35e-20. **Superseded 2026-10-02**: the hit-ahead class is biased by construction (it
  drops `grad P . Delta lambda`; mode 2 +40 %); the default is `value_class="none"` (see HANDOFF and docs/math).
* **Decisions (2026-09-20).** Shape controls are normal displacements of curved surfaces
  (radius-type); tangential translation of flat walls is not supported as a control. Shapes are
  optimised as a C1 level set (4.4) rather than polygon vertices, because a flat-normal polygon's
  vertex jump term is representation-induced and vanishes on a C1 surface, where the pathwise VJP of
  the replayed wall map is complete for specular walls; only the diffuse silhouette term survives.
  On the analytic circle the tally term alone reproduces `F/r` (1.28 vs 1.23e-20) and the total
  1.44 +- 0.09e-20 agrees with CRN FD 1.40 +- 0.01e-20 (48 seeds, z 0.5).

#### 9.7.1 Several value fields in one sweep: the two-column backward (2026-10-02)

A shape seed needs two value fields over the same captured run: `lambda` of the objective's hit
rewards and `lambda_timing` of the timing rewards. They are **one sweep carrying `lambda in
f64[N, 2]`**.

*Why it is exact.* The value backward is linear in its rewards and nothing it decides depends on
the values: cell tables, virtual partners and their draws, VSS scatterings, staged velocities, and
in every fit the `k` nearest nodes, weights, design and normal matrix `M = D^T W D + ridge` are
functions of the record and query states only. Values enter each fit as the right-hand side
`D^T W y` alone, so column `j` of a sweep with reward columns `(r_1, r_2)` is the backward of `r_j`.

*Shared work.* Per fit (`field_query` through `cell_query` at five places per substep: `f_node`,
the partner pair `f_i, f_j` per partner, `f_prev, f_here` of the boundary move) the neighbour
search, gathers, design, `M` and its factorisation are formed once; each column costs a gather, a
weighted sum and two triangular substitutions. Partner draws and VSS kernels are made once.

*Interfaces.* `field_query(..., node_val f64[Q,P,R])` returns `value f64[Q,R]`,
`grad_x f64[Q,D,R]`, `grad_v f64[Q,3,R]` (2-D `node_val` keeps single-column shapes);
`backward_virtual(phi_final [N,R], phi_step [M,N,R], hit_reward [M,K,N,R])` returns
`Backward(lambda0, lam [M,N,R], kernel [M,R])`. The sequential `backward` stays single-column.
`examples/ensemble_inlet_shock.device_backward` takes stacked rewards and requires
`host_adjoint=False` with columns; `examples/ensemble_mesh._backwards` stacks hit totals and timing
into one call.

*The solve.* The normal matrices are symmetric positive definite, so batched LU is replaced by
`ensemble_device._spd_solve` (Cholesky written out over the static column count, elementwise over
queries so XLA fuses it; a pivot rounded to `<= 0` is floored at the smallest normal float).

*Measured* (RTX 3050 Ti, f64): a seed's two value backwards (Kn 0.5, virtual 1x2) 90.5 s -> 43.3 s;
optimiser seed backward / whole seed 65 / 99 s -> 29 / 64 s. *Agreement:* column `j` equals the
single-column sweep to 1e-13 relative
(`tests/test_ensemble_device.py::test_virtual_backward_value_columns_are_separate_backwards`;
`_spd_solve` matches LU to `eps*cond(M)`); against the old two-call code the optimiser's gradient
terms agree to 3.7e-9 relative, the objective bitwise.

*Collisionless runs.* Where a step makes no attempts the fits do not change `lambda` at all
(identical to 16 digits), so they are pure cost there. Skipping them is deliberately **not** done:
the collisional path must be the one every run exercises.

#### 9.7.2 The per-cell field fit: one polynomial per collision cell (2026-10-11)

**What it replaces.** Every field value the virtual pull-back needs (`f_node`, the two
post-collision ghosts of each of `K` virtual partners, `f_prev`/`f_here` of the boundary move:
`(3 + 2K) N` queries per substep, `19 N` at the shipped `4 x 8`) was a **per-query** weighted fit
(`field_query`: `k` nearest nodes in the scaled `(x, v)` metric, weights `1/(1+(d/d_k)^2)`, a
14-column `quadratic_full` design, a Cholesky solve per query). Its cost is linear in `k` (~300 s
fixed + ~10.7 s per unit of `k` for a Kn 0.5 squash seed) and its answer depended on `k` (32 -> 64:
+1.6 +- 0.7 % of the shape gradient, flat from 64).

The per-cell fit forms **one** least-squares polynomial per collision cell per substep and answers
every query in the cell by evaluating it. Module `jax_dsmc/cell_fit.py`; selected by
`ensemble_backward.configure(..., basis=...)` (`VIRTUAL_BASIS`), the CLI flag `--backward-basis`,
default `ensemble_backward.CELL_BASIS`:

    CELL_BASIS = "cell:cubicv:adapt1.5:guard32:align0.9:both"
    # 2026-10-19: leverage guard added; unguarded cubicv from 2026-10-12; cell:quad:... from 2026-10-11

```
# fit (cell_fit.fit), once per cell per substep; node tables (C, P, .), values with R columns
z      = concat(node_x, node_v) / scale        # cell widths, thermal speeds
centre = mean(z over valid nodes)              # per cell
A      = features(z - centre, poly)            # monomials of order <= 2 | 3
keep_j = order(j) <= max order g with cols(g) * adapt <= count       # the ORDER GUARD
M      = A_kept^T A_kept + ridge * tr(M)/#kept * I(non-intercept) + I(dropped)
M      = I if the cell (or side) has no node                         # value 0, finite
beta   = solve(M, A_kept^T y)
# evaluate (cell_fit.evaluate), every query: O(p)
value(q) = features(z_q - centre[c_q]) . beta[c_q];   grad(q) = d features/dz . beta / scale
# guard<h>: per query, the highest order whose leverage <= h (nested degree-ordered prefixes via one
# Cholesky whitening per cell; leverage = cumsum (W f)^2)
```

*Bases* (`"cell:<poly>[:opt]..."`): `mean`, `lin`, `quad` (`lin` + `v (x) v` + `x_d^2`, 14 in 2-D),
`quadfull` (21), `cubicv` (`quad` + the 10 cubic monomials of `v`, 24), `cubic` (56). Options:
`adapt<a>` (order guard), `guard<h>` (per-query leverage cap), `loo`, `pool<r>`, `ridge<e>` (default
1e-8 relative), `rb`, `own`, `align<r>`, `nofallback`, `both`.

**Rao-Blackwellised partners (`rb`).** With a polynomial field the expectation of a ghost over the VSS
scattering angle is computable via six symmetric sigma points (`_scatter_sigma`; exact for
quadratics at any `alpha`, cubics only at `alpha = 1`). Measured: backward angle noise sd 1.37 ->
0.91 % but it does not cure sparse-cell instability and costs +13 s/seed. **Not adopted.**

**Next to surfaces (the one-sided form).** The carried value jumps across `v . n = 0`, which no
polynomial in `v` represents. `fit_split` fits each cell three times (nodes with `v . n_c > 0`,
`<= 0`, all), with `n_c` the unit mean of the cell's node normals and `R_c = |sum n_i| / #n_i`:
`both` is used where no surface is near, where a side is empty (its matrix is the identity, finite;
without this a ghost on an empty side read a singular fit, NaN on the optc shapes) or, option `both`,
in complex cells (`R_c < align`); otherwise `ahead`/`behind` by the sign of `v_q . n_c`. Complex cells
originally fell back to the per-query one-sided fit (`_Hybrid`, buffer `fallback_capacity`, sized on
the host); that fallback is **unstable on complicated geometry** (lambda rms x150 on S1-adam
iterate 100), hence `both`. With every cell complex the hybrid *is* the per-query one-sided backward
(`tests/test_cell_fit.py`).

**Why one polynomial per cell is the right object.** The collision operator only sees the part of
`lambda` that is not a collision invariant (a field in the span of `(1, v, |v|^2)` gives
`f_i + f_j - f_node_i - f_node_j = 0` for every pair). What the pull-back needs is the cell's
`v`-dependence beyond the invariants, a property of the cell's population, not of a neighbourhood of
each query. A cell's ~40 nodes carry values that are mostly noise (the cell mean predicts held-out
nodes at the noise floor, CV mse 3.0e-12 vs variance 3.1e-12, while every unguarded quadratic
overfits to 3x), and an underdetermined quadratic in a sparse cell answers the partners' ghosts with
extrapolated noise that the backward recursion compounds. The order guard bounds columns by nodes.
Value cross-validation cannot rank bases for the backward (the cell mean wins it and has no
collision term); the gradients against FD do.

**The cubicv blow-up and the leverage guard (2026-10-18/19).** Unguarded `cubicv` blew up on thin
bodies at high Mach (nose gradient spikes 70-680x): the cause is **velocity-space extrapolation**
(post-collision queries in the gap between the beam and the wall-reflected gas; cubic leverage p99
~1e3 vs ~30 for quad). `guard32` kills it (max lambda <= 2e-3), blunt-squash G == C (-1.4 % vs FD),
unit gates rule out `h = 2`. Residuals: optm iterate 58 nose -0.02 (z -4..-6), thinnest squash -4 %
in every fit. The guard was rewritten as one Cholesky whitening per cell (local eval 83 -> 17 ms).

**Measured** (FD = per-seed CRN central difference; paired = same forward). Kn 0.5 squash,
`s = 1`, fnum x1, 16 seeds, relative to the same seed's FD:

| backward fit | adjoint vs FD | seed time |
|---|---|---|
| `cell:quad:adapt1.5:align0.9:both` | -3.7 +- 0.8 % | 46 s |
| per-query k = 32 | -3.4 +- 0.8 % | 1100 s |
| `cell:quad` (unguarded) | -3.2 +- 0.8 % (stable at fnum x1) | 47 s |
| **`cell:cubicv:adapt1.5`** | **-0.7 +- 0.8 %** | 50 s |

Selection: `cell:quad:adapt1.5` was the cheapest *stable, unbiased-within-resolution* basis (matches
k = 64 to 0.05 +- 1.5 %, a tenth of a seed's time); unguarded `quad` and `rb` rejected as heavy
tailed; `pool1` read low. On 2026-10-12 the gap gates moved the default to `cubicv`: horizon
(capturing 1000 vs 300 steps) adds +2.1 +- 0.3 % at both run lengths (adjoint memory beyond 300
steps), `cubicv - quad` adds +3.0 +- 0.2 %, and the two corrections overshoot together
(+1.9 +- 0.65 %): at the production horizon `cubicv` is the closest setting (pooled -1.2 +- 0.6 % vs
-4.3 % for the quadratic). Stored collisional gates (0-D relaxation, Couette, inlet `nrho`/`temp`,
kernel `diameter`/`omega`) agree with the k = 32 runs to within 0.2-2 %.

Things that are **not** adjoint errors: the circle's -14 % at `T = 1000` (the FD of the unsteady
60 %-blockage box sees how the shape changes the slow fill; at `T = 2000`, -1 +- 4 %); the 2026-10-03
"7-9 % gap" (an FD noise artefact: 64-seed FD h = 0.1 vs 0.2 agree to +0.14 +- 0.67 %). The grazing
cut-off is not the bias but IS the variance (sd ratio scales with the cut-off; smoothing sub-cell
features, radius 3.5-14 cm vs 6.25 cm knots, cuts it ~3x). Rough optc iterates have no FD
derivative at `h ~ 0.05`, so "biases" measured on them are not meaningful, only their scatter.
Unlocated: FM squash ~-1.5 %, S8 iterate 30 `cos 2 theta` +0.14 +- 0.03 of shift.

*Tests*: `tests/test_cell_fit.py` (polynomial reproduction for every basis, least-squares solution
and exact slopes, order-guard column caps, finite empty sides deferring to `both`, sigma points vs
Monte Carlo, the all-complex hybrid equal to the per-query one-sided backward).

### 9.5 Tests (`tests/test_move.py`), parametrised over `D in {1, 2, 3}`

- periodic: particle count and total energy exactly conserved over 1e4 steps; a particle pushed
  `k` box-lengths returns to the same position for every integer `k`;
- specular: energy exactly conserved; no particle ever outside `[lo, hi]`, tested with velocities
  large enough to force 1, 2, 5, and 50 bounces per step;
- specular fold verified against a brute-force iterative reflect-until-inside reference, to
  machine precision, over randomised inputs;
- specular VJP equals `diag([-1,1,1])` on the crossed axis (checks the [P3] `C` matrix directly);
- outflow: `alive` count is monotone non-increasing, and no dead particle influences any moment.

---

## 10. `jax_dsmc/sample.py` — macroscopic moments

Per-cell moments via `segment_sum` with a **static** `num_segments = C`. Two-pass for the
central moments, to avoid the catastrophic cancellation of `<v^2> - <v>^2` in near-equilibrium
cells where the bulk speed is small compared to the thermal speed.

```
def moments(state, table, grid, cfg, species):
    c   = cell_of(state.x, grid)
    w   = state.alive.astype(f64)
    cnt = segment_sum(w, c, num_segments=C)                     # f64[C]
    inv = safe_recip_count(cnt)
    V   = cell_volume(grid)

    nrho = cnt * cfg.fnum / V                                   # number density
    u    = segment_sum(w[:, None] * state.v, c, C) * inv[:, None]   # f64[C, 3]

    cpec = state.v - u[c]                                       # peculiar velocity, pass 2
    e    = segment_sum(w * sum(cpec*cpec, -1), c, C) * inv
    temp = species.mass[0] * e / (3.0 * K_BOLTZ)                # f64[C]

    P = species.mass[0] * nrho * segment_sum(
            w[:,None,None] * cpec[:,:,None]*cpec[:,None,:], c, C) * inv[:,None,None]
    q = 0.5 * species.mass[0] * nrho * segment_sum(
            w[:,None] * cpec * sum(cpec*cpec,-1)[:,None], c, C) * inv[:,None]

    return CellMoments(nrho=nrho, u=u, temp=temp, P=P, q=q, count=cnt)
```

All of this is differentiable in `v` and, through `cell_volume`, in `lo`/`hi`. This module is
where objective functionals `r(x, v)` are built, so its differentiability is the entry point for
the whole adjoint.

Time averaging: a running accumulator over the sampling window, exactly as SPARTA's
`fix ave/grid`. Accumulate sums, not averages, and divide once at the end.

### 10.1 Tests (`tests/test_sample.py`)

- a Maxwellian at known `(nrho, T, vstream)` recovers those values within MC error;
- `trace(P)/3 == nrho * k * T`: note this is an **identity by construction**, since `temp` is
  defined from the same `<|c|^2>` that forms `trace(P)`. Keep it as a typo/refactor regression
  guard, but it is not a physics check and must not be counted as one. The real check is
  `P` against the analytic Chapman-Enskog stress at a known shear rate (M4);
- `q == 0` within MC error at equilibrium;
- empty cells produce `0`, not `NaN`, in every field, and finite gradients;
- moments are invariant to particle ordering (permutation test).

---

### 10.1.1 Field images are the default output; particles are opt-in

**Do not write particle state per frame.** It is the obvious thing to reach for in a particle
method and it is the wrong default: the cost scales with `N`, which is the one number this project
intends to push to `1e9`-`1e10`. Measured per frame:

| output | bytes/frame | scales with |
|---|---|---|
| field PNG, `48 x 32 x 24` cells | **50 KB** | cell count only |
| particle `.vtp`, `N = 1e6`, already subsampled 50x | 1.7 MB | `N` |
| particle state, `N = 1e6`, unsubsampled | ~85 MB | `N` |

So the routine output path renders the **fields** — the same choice SPARTA makes with `dump image`
— and `jax_dsmc.viz.FieldMovie` is that path. Particles remain available through
`output.VTKSeries.add_particles`, which is a separate call and subsampled by a frame-stable rule
(§10.2); nothing writes them unless asked.

`viz.draw_field` dispatches on dimension: `D = 1` a profile against `x`, `D = 2` a heatmap,
`D = 3` an axis-aligned cut. In 3-D the default is **three files per frame** — the `x = 0`,
`y = 0` and `z = 0` faces (`viz.PLANES_ORIGIN`) — because one slice of a three-dimensional flow
says almost nothing about the other two directions. `viz.PLANES_MID` gives the mid-planes instead,
and `viz.Plane(axis, index=...)` names any other axis-aligned cut.

A **slice** rather than a projection, deliberately: averaging along a direction is exactly what
hides a shock front, which is the structure these pictures exist to show.

> **General (non-axis-aligned) planes are not implemented.** They need interpolation onto the cut,
> which is a different operation with its own accuracy question — nearest-cell aliases badly on a
> front, trilinear smears it. The `Plane` type exists so that adding one later is a new constructor
> in `viz` rather than a change at every call site: everything downstream takes a `Plane` and asks
> it for a 2-D array plus its two surviving axes.

**Colour limits are fixed across a series.** Per-frame autoscaling makes an animation unreadable —
the colours stop meaning a value and start meaning "relative to this frame", so a decaying shock
looks like it is not decaying. `FieldMovie.set_limits` takes them from the physics you expect; left
unset they are captured from the first frame and then held.


### 10.2 `jax_dsmc/output.py` and `jax_dsmc/viz.py` — writing results out

*2026-09-17 amendment:* cut cells keep the `.vtr` rectilinear output (cells are still boxes)
with a `volume_fraction` field; the surface itself goes out as `.vtp` polydata
(`output.write_vtp_surface(points, elements, cell_data)`, lines in 2-D / triangles in 3-D)
carrying the per-element loads; `viz.draw_surface` / `field_snapshot(surface=...)` overlay
it on the field pictures (3-D: the triangles' intersection with the drawing plane).  Because
cells stay boxes, a per-cell field drawn on the `.vtr` looks blocky at the body; the
`.vtu` written by `output.write_vtu_flow(mask=cut.flow_mask(...))` and `draw_field(mask=)`
draw the same per-cell values on the flow sub-cells only (body carved out at `h/sub`), so
the pictures show where the surface is without inventing sub-cell field values.

Separate modules from `sample.py`; listed here because they consume its moments. **All output is
host-side, called outside the jitted step**, so nothing perturbs the compiled loop.

Four paths:

**1. matplotlib figures** for milestone demos — BKW distribution vs. analytic, moment relaxation,
Couette profile vs. Chapman-Enskog, gradient-vs-finite-difference validation. These run in CI and
double as milestone acceptance evidence. Headless: non-interactive backend, save to file, never
`show()`.

**2. ParaView-ready VTK.** The naive version of this — one `.vtr` per step, cell fields only — is
what a first draft reaches for, and it is wrong in two ways that only show up when you actually
open the result. ParaView sees N disconnected datasets rather than one animatable series, and
there is no way to look at the particles at all, only at their cell averages. For DSMC the
particles are frequently the thing you want to see.

| File | Contents | Purpose |
|---|---|---|
| `step_%05d.vtr` | per-cell moments: `nrho`, `u`, `temp`, `P`, `q` | rectilinear grid — correct and compact for a uniform mesh, spacing is *implied* rather than stored |
| `part_%05d.vtp` | particle points + `v`, `species`, `alive` | polydata — the particles themselves |
| `fields.pvd`, `particles.pvd` | `(timestep -> filename)` collection | **this is what makes ParaView animate.** Without it you get a pile of unrelated files |
| `*.pvtr` / `*.pvtp` | one piece per rank plus a master file | partitioned output, needed once the run is sharded |

> **Particle output must be subsampled, and the subsample must be stable across frames.** At
> `N ~ 1e9` writing every particle is not an option. The subsample therefore has to be a
> *deterministic function of particle identity* — a stride, or a hash of the particle index —
> and **not** a fresh random draw per frame. A per-frame random subsample shows a different
> population every step, which turns an animation into visual noise and makes it impossible to
> follow a trajectory. Reuse the RNG contract (§1) shape: derive the selection from a fixed key
> and the particle index, never from state.

> **Unstructured cell output.** Cut and split cells are not a rectilinear grid. `.vtr` is kept for
> uniform meshes (spacing implied, compact); `output.write_vtu_flow` writes the flow sub-cells only
> (body carved out at `h/sub`) and `output.write_vtp_surface` the surface; the `.pvd` collection
> layer and the `.vtp` particle path are unaffected.

**3. HDF5/npz state checkpoints** — `x`, `v`, `alive`, `logh_acc`, cell moments, the RNG base key
and the step index — for restart, offline analysis, regression baselines, and eventually the
frozen-decision record if the reversible adjoint (§18 deferred) is ever built. Restart
correctness requires the base key and step, because the RNG contract (§1) reconstructs every draw
from `(base_key, step, STAGE_ID)`; saving particle state alone is not resumable.

**4. Interactive diagnostics** — `summary(state, diag)`: per-cell moment table, `ncollide` /
`nattempt`, and the two counters that must never be silently nonzero,
`diag.attempts_clipped` (§8.2) and `diag.majorant_violations` (§7.2.1).


## 11. `jax_dsmc/adjoint.py` — the surrogate objective

> **Status.** This is the *reference* score/DiCE path that reproduces [P1]/[P2] ("paper"
> scoring: each collision deposits its log probability on both participants' terminal labels). It
> is exact for pathwise parameters and is the right oracle, but it is not the production gradient:
> terminal-label scoring is biased for trajectory objectives, and the trajectory score's standard
> error grows steeply with horizon (measured 2026-09-15, ADJOINT_BRANCH_REVIEW.md). The production
> route is the ensemble adjoint (section 9.7). `adjoint.trajectory_surrogate` and
> `simulate.run(score_mode="trajectory")` (each event counted once; the default of
> `terminal_objective` and `windowed_objective`) are documented in
> [ADJOINT_CORRECTIONS.md](ADJOINT_CORRECTIONS.md).

### 11.1 The construction

> **Sign convention.** [P1] seeds the backward pass with `gamma_F = -d r/d v` (a Lagrange
> multiplier); [P2] with `gamma_i^M = (rho/N) phi'(v_i^M)` (a plain influence function). **We follow
> [P2]: `gamma` is the derivative of the objective, no sign flip**, which is also what `jax.grad`
> produces. Oracle 1 (13.1) transcribes [P1]'s recursion and must negate at the seed.

[P2] eq. `gamma_rules` for a general kernel:

```
gamma_i^M = (rho/N) phi'(v_i^M)
(gamma_i^k, gamma_i1^k) = D_i^k (gamma_i^{k+1}, gamma_i1^{k+1}) + (eta_i^k, eta_i1^k)
eta_i^k   = (rho/N) d(log h_i^k)/d(v_i^k) * (phi_i^M + phi_i1^M)
```

`D` is the transposed collision Jacobian (what reverse mode computes). The `eta` term is recovered
by the **DiCE magic-box** applied to the per-particle accumulator `logh_acc` that `collide` fills:

```
magic(L)    = exp(L - stop_gradient(L))               # value 1, gradient dL        (adjoint.magic)
surrogate   = rho * sum(alive_weights * (magic(logh_acc) * (phi - b) + b))   # adjoint.surrogate
```

(`rho` is the distribution's total normalisation, carried so `J = (rho/N) sum phi_i` stays
dimensionally correct as `N` varies; `b` is an independent constant baseline. A same-trajectory
reward mean as baseline is biased and rejected.)

### 11.2 Why this is exactly the paper's adjoint

The value is unchanged (`magic == 1`), so `surrogate == J`. The gradient is
`(rho/N) sum_i [grad phi_i + phi_i grad L_i]`: the first term is pathwise (reverse mode transports it
back through the collision maps, giving the `D` recursion); in the second, `L_i` and `L_i1`
accumulated the **same** `log h` for a pair, so the coefficient of `d log h/dv` is `phi_i + phi_i1`,
precisely `eta`. `magic` rather than `sum(phi) + sum(stop_gradient(phi) * L)` keeps the primal value
equal to `J` and higher derivatives correct.

### 11.3 What flows where

`log h` depends on the pre-collision velocities **and** on `diam`, `omega`, `tref` through
`prefactor`, so the score supplies the velocity sensitivity the papers derive and the direct
parameter sensitivity of the collision rate (the model-calibration gradient) with no extra machinery.

### 11.4 Variance

> **The count score (8.1) requires a baseline when used alone**: `b = 0` sd `3.76e6`, mean-`phi`
> baseline `4.50e4` (84x).
>
> **MEASURED (2026-09-05): the 84x does NOT carry over to a windowed spatial objective.** On the M2
> shock at `D = 1` with Gaussian-windowed kinetic energy, 512 base keys, the mean baseline moves the
> per-seed sd by only ~1.2x (`diam` `1.60e12 -> 1.34e12`, `omega` `3.10e2 -> 2.61e2`, `alpha`
> `3.55e1 -> 3.27e1`, `nrho` `1.52e-16 -> 1.50e-16`). `phi` varies by orders of magnitude within a
> run, so subtracting its global mean removes almost none of the variance. Keep the baseline (free,
> never hurts) but do not rely on it; section 23.3 gives the sample counts actually required.

For per-pair scores the papers use reward `(phi_i + phi_i1)` with no baseline; that is the default.

### 11.5 Windowed objectives — the causal form

For `J = sum_k r(x_k, v_k)` (time-averaged drag, wall heat flux, a sampling-window moment), one magic
box on the *final* `logh_acc` scores each reward against collisions after it was emitted: unbiased
but noisier. The causal form `sum_k magic(L_{<=k}) r_k` is one scalar in the scan carry, since
`logh_acc` already is the running prefix:

```
J_acc += (rho / N) * sum(magic(state.logh_acc) * r(state.x, state.v))      # adjoint.dice_reward
```

Verified bit-identical to the explicit form (gradient 215.215669837607 both ways); the primal equals
the plain windowed objective. **Causality buys less than "architectural" suggests**
(`scratch/probe_scan_dice.py`): `Var(naive)/Var(causal)` saturates at ~3x, flat over `M = 8..4096`
(fitted `M^-0.005`), 4x with the 11.4 baseline. A bounded ~1.7-2x in standard error, not a
horizon-dependent blow-up. Pinned by
`tests/test_adjoint.py::test_windowed_objective_is_causal_and_value_neutral`.

### 11.6 Entry point

`simulate.run(..., reward=callable, score_mode=...)` carries the accumulated objective in the scan;
`adjoint.terminal_objective` and `adjoint.windowed_objective(state, cstate, grid, sp, params,
static, base_key, phi, ...)` return `(J, (state, cstate, diag))`, and `jax.value_and_grad` of that
is the whole gradient. No hand-written backward pass exists on this path (the ensemble adjoint of
9.7 is a separate, hand-built backward, validated against it and against FD).

---

## 12. `jax_dsmc/simulate.py` — the time loop

```
_step(carry, k, grid, sp, params, static, base_key, bc, ...):      # module level, bound with partial
    move     (advection + boundary / embedded-surface tracker; STAGE_MOVE / STAGE_WALL)
    inflow.inject (if an inflow reservoir is configured; STAGE_INFLOW)
    sort     (sort.build_table, or sort.update_table with the carried SortCarry)
    collide  (ntc.collide; STAGE_NTC_ROUND / PAIR / ACCEPT / SCATTER)
    reward   (optional causal DiCE accumulation, 11.5)
    emit     (moments or cell temperature; event record when capture=True)
```

`run(state, cstate, grid, sp, params, static, base_key, bc, collide_only, emit_moments, step0,
reward, faces, wall, res, capacity, score_spread, acc_delta, score_mode, capture, mesh,
emit_surface)` scans `static.n_steps` steps (in blocks of `checkpoint_block_size` when set) under
`jax.checkpoint` with `static.remat_policy`, returns `((state, cstate), (out, diagnostics))`, and
with `capture=True` also the stacked `EventRecord` the ensemble adjoint consumes. `collide_only`
short-circuits advection for the single-cell 0-D case (a static Python bool, so it selects a code
path at trace time). `run_adaptive` wraps it for `A_max` (8.5.1); `checked_run` asserts the silent-
failure counters.

### 12.1 Intra-step ordering, and the adjoint form it implies

**We run `move -> sort -> collide`.** SPARTA does `move -> migrate -> sort -> collide`
(`update.cpp:337-364`); [DF]'s pipeline is `move -> buildCellOccupancy -> collisions`. Both
production codes move first, and we have no MPI migration step, so this is the natural choice.

**[P3] specifies the opposite order**, and that matters for the adjoint algebra.
`main.tex:721`: *"Within each global time interval `[t_k,t_{k+1}]`, collisions are first
performed using the particle states synchronized at `t_k`. The particles are then advanced
through the advection and boundary update."* That is `collide -> move`, and it is what produces
[P3] eq. `adjoint_gamma_R` with the transport term **inside** `B_k`.

The two orderings give genuinely different adjoint recursions. Verified by taking real
`jax.vjp`s of both (`scratch/probe_grad_claim2.py`), with a deliberately non-symmetric linear
`C` and again with a nonlinear VSS-like map:

| forward ordering | `beta_k = B beta_{k+1} + dt P* alpha_{k+1}` | `beta_k = B(beta_{k+1} + dt P* alpha_{k+1})` |
|---|---|---|
| `move -> collide` (**ours**) | **exact match, 0.000e+00** | mismatch, 2.45e-02 |
| `collide -> move` ([P3]) | mismatch, 2.45e-02 | **exact match, 0.000e+00** |

`alpha_k = alpha_{k+1}` in both. The forms differ by 0.70% relative at `dt = 0.017`; the
discrepancy is `O(dt * ||B - I||)` and so does **not** vanish at realistic timesteps.

> **Consequence.** Revision 2 asserted that reverse-mode AD of `x += v[:, :D]*dt` reproduces
> [P3]'s formula "with no special handling". That is false for the loop we actually run — ours
> has the transport term **outside** — and revision 2's §12 was internally self-contradictory,
> its code block running `move -> sort -> collide` while its prose claimed [P3]'s opposite order.
>
> This is a documentation defect, not a production one: `jax.grad` differentiates whatever loop
> is written, so the implemented gradient is correct either way. The danger is to anyone
> hand-checking the adjoint or writing a non-0D oracle against the paper's formula.
>
> **It is also invisible to the entire planned test suite.** §13.1 and §13.2 are 0D single-cell,
> so `alpha == 0`, both forms collapse to `beta_k = B_k beta_{k+1}`, and neither oracle can tell
> the orderings apart. It would first surface at M4. That is why §13.3 adds a 1D two-step oracle
> whose only job is to pin the ordering down.

- `jax.checkpoint` with a configurable policy wraps the step body; a no-op in the forward pass.
- **No data-dependent `while_loop` in the differentiated path.** Every loop is a `scan` or
  `fori_loop` with a static trip count (time loop, attempt slots, wall passes, inflow quantile
  iterations). A jaxpr audit covers the collision step (`tests/test_ntc.py::test_no_data_dependent_
  while_loop_in_the_step`, `tests/test_perf.py::test_collide_contains_no_while_loop`). One
  deliberate exception (2026-09-24, 9.6.1): the *untraced* forward move of `tracker="lane"` ends its
  lane passes with a `while_loop`; the traced replay the shape adjoint differentiates uses
  `fori_loop` + `lax.cond`.
---


## 13. Verification: the oracle tests

`tests/test_adjoint_oracle.py` is the load-bearing test of the whole architecture. It hand-codes
the papers' backward recursions and asserts `jax.grad` agrees.

**Both oracles run under `pair_perm`** (8.3), not `pair_ntc`. The papers pair without
replacement — [P2] eq. `factorizationEa` is a product over `N/2` disjoint pairs — so an oracle
built on with-replacement NTC pairing would be validating our recursion against a *different*
algorithm from the published one. With `pair_perm` these are faithful implementations of
Algorithms 2 and 3.

### 13.1 Oracle 1 — [P1] Algorithm 2, Maxwell molecules

Setup: 0D single cell, `pair_perm`, `omega = 1.0` (so `sigma_T * v_r` is constant — genuinely Maxwell
molecules), `alpha = 1.0`, and `Sigma` set exactly to that constant so every attempt is accepted,
`p == 1`, `log h == 0`, and no score term arises. The adjoint is then pure pathwise.

```
gamma = -dphi/dv (v_M)
for k = M-1 .. 0:
    for each collision pair (i, i1) recorded in the forward pass:
        alpha_k = unit(v_i^k - v_i1^k)          # pre-collision direction
        sigma_k = unit(v_i^{k+1} - v_i1^{k+1})  # post-collision direction
        A = 0.5 * [[I + sigma alpha^T, I - sigma alpha^T],
                   [I - sigma alpha^T, I + sigma alpha^T]]
        (gamma_i, gamma_i1) = A^T (gamma_i, gamma_i1)
assert allclose(gamma, jax.grad(...), rtol=1e-12)
```

This oracle is **fully independent**: `A` is written from [P1] eq. `AB_def` analytically, not
obtained from AD. It validates the seeding, the recursion, the non-colliding identity branch, and
the frozen-randomness contract.

> **REVISION (2026-09-05): the forward pass must be [P1]'s collision map, not `vss.scatter`.**
> Eq. `AB_def` is derived under `delta sigma = 0` — the post-collision direction drawn
> independently of the pre-collision velocities. SPARTA draws it *in the frame aligned to `v_r`*,
> so at frozen randomness it is not independent, and the pair Jacobian is **rank 6** where `A` is
> rank 4. Measured divergence `‖J − A‖/‖A‖ = 0.53…0.81` at `omega = 1, alpha = 1`; see §17.1.1.
>
> Oracle 1 therefore runs `_labframe_scatter` — `cos(theta) = 2 u_chi − 1`, azimuth `2 pi u_eps`,
> `v' = v_cm ± |v_r| sigma / 2` — which reproduces eq. `AB_def` to **3.5e-18**. That is not a
> weakening: it is what makes the test a test of [P1]'s *algorithm*. The production sampler's
> gradient is checked instead by Oracle 2 (whose `D` comes from `jax.jacobian`, as §13.2 already
> allows) and by the finite-difference gate of §13.4.
>
> The structural assertion `matrix_rank(A) == 4`, `svd(A) == [1,1,1,1,0,0]` is kept, **on the
> analytic matrix `_p1_matrix`**. Asserting it of `jax.jacobian(scatter)` would fail today, and
> the original wording ("if a refactor ever makes `A` full rank, the collision map has been
> altered") wrongly implied the two were the same object.

### 13.2 Oracle 2 — [P2] Algorithm 3, general VSS kernel

Setup: 0D, `omega != 1`, `alpha != 1`, `Sigma` a genuine strict majorant so both accepted and
rejected candidates occur. Hand-codes eq. `gamma_rules` including `eta` from eq. `back_eta`,
**with all three branches**, in particular the virtual-but-not-real branch contributing
`d log(Sigma - q)/dv`.

Honest scoping of what this proves: `D` for VSS has no simple closed form (which is exactly why
[P2] notes `A` and `C` "may not be the same"), so the oracle obtains `D` from `jax.jacobian` of
the single-collision `scatter` function. The test therefore independently validates the
**recursion structure** — seeding, the `eta` terms, the both-branches requirement, the
`(phi_i + phi_i1)` reward attribution, and the accumulator bookkeeping — but not `scatter`'s own
Jacobian. That is covered separately by a finite-difference test on `scatter` in isolation
(section 7.6). Stated here so the coverage is not overclaimed.

This test is what catches: dropping the reject-branch score term; stop-gradienting the wrong
thing; accumulating `log h` onto only one partner; letting `Sigma` vary within the step
(SPARTA's live ratchet, 7.2 — the score term would then be differentiating against a moving
target); and any future refactor that breaks the RNG contract.

### 13.3 Oracle 3 — 1D two-step, to pin the intra-step ordering

Oracles 1 and 2 are 0D, so `alpha == 0` and they cannot distinguish `move -> collide` from
`collide -> move` (§12.1). A minimal third oracle closes that hole: **1 spatial dimension, 2
cells, 2 timesteps, periodic BCs**, hand-coding

```
beta_k  = B_k beta_{k+1} + dt * P^* alpha_{k+1}
alpha_k = alpha_{k+1}
```

and asserting it matches `jax.grad`. The test must **fail** if the loop is reordered or if
[P3]'s inside-`B_k` form is substituted — assert both, so it is a real discriminator rather
than a coincidence.

**Built** — `tests/test_adjoint_oracle.py::test_oracle3_*`, five tests. `B_k` is [P1]'s
analytic `A^T`, never `jax.jacobian`, which would be circular. The discriminator matrix is
exactly diagonal, and it reproduces §12.1's toy-map result on the real solve:

| forward ordering | `beta_k = B b + dt P* a` (OUTSIDE) | `beta_k = B(b + dt P* a)` (INSIDE) |
|---|---|---|
| `move -> collide` (**ours**) | **1.33e-16** | 9.85e-02 |
| `collide -> move` ([P3]) | 9.58e-02 | **1.81e-16** |

`alpha_k = alpha_{k+1}` holds to 0.0e+00 in both orderings, as predicted. The objective must
depend on **both** `x_M` and `v_M`: with a velocity-only `phi`, `alpha == 0`, the transport
term vanishes and the oracle silently degenerates into Oracle 1 — so a nonzero-`alpha` guard
is asserted before the comparison.

> **Correction, found by the test.** This section previously claimed two steps is *the minimum
> that exercises the coupling*. Measured, that is wrong: the two forms differ by
> `dt (I - B) P* alpha`, which is already nonzero after **one** step (2.66e-2, against 9.85e-2
> at two). One step would discriminate. Two steps is still the right configuration — it is the
> first that pushes a *transported* cotangent back through a second `B_k`, so it tests the
> composition rather than a single application, and the margin is ~3.7x wider — but the stated
> reason was not the real one. Pinned by
> `test_oracle3_one_step_would_already_discriminate`.

**Scope.** The oracle validates the recursion against its own faithful 1-D forward pass, so on
its own it would not notice `simulate._step` being reordered — after which it would be pinning
the adjoint of a loop we no longer run. `test_production_step_moves_before_it_collides` closes
that behaviourally rather than by reading the source: under `move -> collide` the new positions
are built from **pre**-collision velocities, so `x_1 == wrap(x_0 + dt v_0[:, :D])` exactly
(measured 0.0e+00, against 1.31e-02 for the post-collision form).

### 13.4 Finite-difference cross-check

For every traced parameter, central differences with **common random numbers** (the same
`base_key` for both perturbed evaluations). Without CRN the FD estimate is dominated by
`O(1/sqrt(N))` Monte Carlo noise and the test is meaningless. Swept at `omega = 1` (where the
per-pair score terms vanish and agreement should be tight) and `omega != 1` (where they do not).

> **CRN alone is not sufficient, and this nearly hid a real defect.** At fixed randomness `J` is
> a *step function* of any parameter entering a frozen decision. Measured on the count decision
> (`scratch/probe_grad_claim1.py`):
>
> | relative perturbation | `n(+)` | `n(-)` | central FD |
> |---|---|---|---|
> | 1e-8 … 3e-2 | 3 | 3 | **0.000000e+00** |
> | 1e-1 | 4 | 3 | 1.198e+05 |
>
> So for every perturbation small enough not to flip `n`, FD reports exactly zero — **and so did
> `jax.grad`**. The revision-2 defect in §8.1 would have passed this test cleanly. There is no
> step size that fixes it: too small and no decision flips, large enough to flip one and the
> estimate is a difference quotient across a discontinuity.
>
> **Therefore the FD harness must run in two modes**, and the distinction must be explicit in
> the test names:
> 1. **Fixed-key mode** — validates the *pathwise* gradient only. Instrument it to count decision
>    flips (cell membership, `n`, pair indices, accept bits) between the ± evaluations and
>    **assert zero flips**; a nonzero count means the comparison is meaningless, not that the
>    gradient is wrong.
> 2. **Ensemble mode** — average over many `base_key` values, so the score-term contributions
>    become visible. This is the only mode that can validate `dJ/d(dt)`, `dJ/d(fnum)` and
>    `dJ/d(cell_volume)`, and it needs enough samples to resolve a heavy-tailed estimator (~1e5
>    at the reference configuration, or ~1e3 with the baseline of §11.4).
>
> Where a ground truth is cheaply available, prefer it to FD outright: Rao-Blackwellising a
> single frozen decision (enumerate its outcomes, weight by probability) gives a zero-variance
> reference, which is how the §8.1 fix was validated.

#### 13.4.1 There *is* a valid FD window — but it closes at production scale

It is tempting to conclude that FD can never agree with AD once cell binning exists, because a
perturbation moves a particle between cells and the pairing diverges. **That is too strong.**
Measured (`scratch/probe_num_c3_fd2.py`, 1-D, `C=8`, `N=256`, `M=6`, perturbing `temp`):

| `h` (relative) | cell flips | pair flips | FD rel. err vs AD |
|---|---|---|---|
| 1e-3 | 403 | 638 | **1.11e+01** |
| **3.16e-04** | **0** | **0** | **3.31e-13** |
| 1e-5 | 0 | 0 | 3.94e-12 |
| 1e-8 | 0 | 0 | 6.15e-09 |
| 1e-12 | 0 | 0 | 5.08e-05 |

With zero decision flips, FD and AD agree **to round-off**, over a **5.5-decade** usable window.
So the disagreement is an artifact of choosing `h` too large, not a law of nature.

Two properties make it dangerous anyway:

**It is a cliff, not a slope.** `3.3e-13 -> 11.1` across a factor of ~3 in `h`. And a single cell
flip is hugely amplified: one particle changing cell rebinds **60 pair indices (3.9%)**, because
`counts` changes so `randint(k, 0, npc)` returns different integers *and* `starts` shifts every
higher-index cell. At `h = 1e-2`, 84% of pairings diverge.

**The window closes as the problem grows.** Bisecting the largest flip-free `h`
(`scratch/probe_num_c3_window.py`) gives, empirically, `h_crit ~ 5.8 / (N * C * M)`:

| `N` (at `C=8, M=6`) | 64 | 256 | 1024 | 4096 | 65536 |
|---|---|---|---|---|---|
| `h_crit` | 1.9e-3 | 4.0e-4 | 3.4e-5 | 9.7e-6 | **1.1e-6** |

Against a round-off floor near `1e-9`, that is 6.3 decades of headroom at `N = 64` and 3.0 at
`N = 65536`. Extrapolated to an M3-scale run (`N = 1e6`, `C = 1e4`, `M = 1e3`):
`h_crit ~ 6e-13` — **below the floor. There is no valid `h` at all.** `h_crit` also varies up to
**30x between seeds**, so a single globally-chosen `h` is unsafe even where a window exists.

**Therefore:** instrument the harness to record every frozen decision on both evaluations and
**assert zero flips** — converting a silent wrong answer into a test failure.

> **CORRECTION (2026-09-05): the six-item list above is incomplete, and the missing items are
> not exotic.** Implemented as `Diagnostics.decision_checksum`, the guard must also cover:
>
> | decision | where | why it is not in the original list |
> |---|---|---|
> | **boundary fold index** `m = floor((x_new − lo)/L)` | `boundary.apply_specular` | Its *parity* flips the sign of `v`. A particle just inside the wall and one just outside **fold into the same cell**, so `cell_of(x)` cannot see it. |
> | **`scatter`'s degenerate-frame branch** `d > EPS_D·|v_r|` | `vss.scatter` | Crossing it swaps to the fallback formula — an `O(|v_r|)` jump in the post-collision velocity. |
>
> Both are now folded in, via `boundary.fold_index` and `vss.frame_is_well_defined`. Measured on
> the 1-D reflected shock: with the boundary index missing, the guard reported **no flips** while
> FD returned `346` against a true `127`.
>
> **And a warning about the checksum itself.** Weighting decision `i` by `mod(i·phi, 1)` puts a
> weight of **exactly 0.0 on index 0**, so every decision taken in **cell 0** vanished from the
> fingerprint. In a reflected shock cell 0 is the cell against the wall — precisely where accept
> bits flip. The guard passed a finite-difference estimate that was wrong by a factor of three,
> and three separate hypotheses were chased before the weight was the answer. Use
> `mod((i+1)·phi, 1) + 0.5`, so every weight lies in `[0.5, 1.5)`. **A checksum with a zero
> weight is not a checksum.**
>
> Diagnosis procedure that actually worked, for next time: scan `J(theta)` on a fine grid to
> confirm a *jump* rather than curvature; bisect on `n_steps` to find the step it appears in;
> then diff every intermediate array of that step. The scan is what ruled out truncation error,
> which two of the three wrong hypotheses would have predicted. Then either bisect `h` down per seed until flips vanish (skipping if
that drives `h` below ~`1e-9`), or state that the FD gate applies only to configurations with
`N*C*M <= 1e7`. Above that, the oracles of 13.1-13.3 and Rao-Blackwellised references are the
only usable checks.

#### 13.4.2 Oracle FD safety

`pair_perm` uses `permutation(key, N)`, which depends only on the key and the static `N`, so
**zero permutation flips were observed at every `h` from 1e-1 to 1e-14**. But that does not make
both oracles equally safe (`scratch/probe_num_c3_0d2.py`, `N=64`, `M=8`):

| config | `h=1e-1` | `h=1e-2` | `h=1e-4` | `h=1e-8` |
|---|---|---|---|---|
| Oracle 1 (`omega=1`, `p == 1`) | **2.0e-16** | 1.3e-14 | 5.8e-13 | 3.0e-09 |
| Oracle 2 (`Sigma = 2 max q`) | 1.29e-01 | 8.2e-15 | 9.1e-13 | 1.1e-08 |
| Oracle 2' (`Sigma = 1.05 max q`) | 3.24e-01 | 1.9e-14 | 1.0e-12 | 7.5e-09 |

**Oracle 1 is unconditionally safe** — `p == 1` means there is no accept randomness at all, and
FD matches AD to `2.0e-16` even at `h = 0.1`. **Oracle 2 is not**: the accept bit is a live flip
channel even in 0-D. Over 400 seeds (256 accept bits each): `1.9e-3` flips/bit at `h=1e-2`,
`9.8e-6` at `h=1e-4`, and **zero at `h <= 1e-5`**. Oracle 2's FD check must therefore pin
`h <= 1e-5`, and note that a tighter `Sigma` makes it worse — another reason for the 2-5x safety
factor of 7.2.2.

Beyond agreeing with our own hand-coded recursions, the gradients should **reproduce the
behaviour the papers report** — [P2] §6 and [P3] §6 give gradient-accuracy and convergence
results for their test problems. Matching those qualitatively (error scaling with `N`, the
relative size of pathwise vs. score contributions) is a stronger check than self-consistency
alone, and is the one that would catch an error common to both our forward solve and our oracle.


---

## 14. Performance and parallelizability suite

Correctness across dimensions is not sufficient; these are first-class tests.

| Property | Test | Asserts |
|---|---|---|
| no RNG inside the slot scan, draws batched `(A_max, cells)` | `tests/test_perf.py::test_no_rng_inside_the_attempt_slot_scan`, `test_batched_draws_are_shaped_a_max_by_cells` | section 1 rule 3 |
| no data-dependent `while_loop` | `test_perf.py::test_collide_contains_no_while_loop`; `test_ntc.py::test_no_data_dependent_while_loop_in_the_step` | static trip counts |
| jaxpr independent of particle count | `test_perf.py::test_jaxpr_is_identical_across_particle_counts_at_fixed_capacity` | static shapes |
| zero recompilation | `test_perf.py::test_no_recompilation_across_traced_values`; `test_ntc.py::test_collide_is_jittable_and_does_not_recompile` | no Python branching on traced values |
| `vmap` composability | `test_perf.py::test_step_is_vmappable`; `test_ntc.py::test_collide_is_vmappable_over_an_ensemble` | purity, ensemble reuse |
| cost budgets | `tests/test_perf_budget.py` | see below |

**Why not wall-clock.** A timing threshold is flaky on a shared box (~1.6x run-to-run variance), so
`test_perf_budget.py` budgets XLA's own `cost_analysis()` and `memory_analysis()`, which describe the
compiled HLO and are bit-identical across runs (`test_cost_analysis_is_deterministic`). Baselines are
keyed per backend and refreshed with `python3 -m tests.test_perf_budget --print-baseline`;
`TOL = 0.35`. Backend-independent ratio tests pin the reverse-pass claims of section 23.4. Compile
time is deliberately not asserted (JAX caches in process, so only the first `compile()` is real);
check by hand with `--compile-time` after any change to the attempt scan.

---

## 15. Cross-reference: block -> SPARTA -> dsmcFoam+ -> paper

| Block | SPARTA | dsmcFoam+ [DF] | Paper |
|---|---|---|---|
| `sample_maxwellian` | `create_particles.cpp:761-780` | **`dsmcCloud.C:1043-1051`** (followed) | [P3] `Psi(xi; m)` |
| `stochastic_round` | `create_particles.cpp:672-677` | `dsmcMeshFill.C:68-238` (per-tet) | — |
| `create_particles` | `create_particles.cpp:588,696-700` | `dsmcMeshFill.C`, `tetrahedronI.H:244-276` | [P3] eq. `BoltzmannIC_space` |
| `cell_of` | `update.cpp:764-864` (OPT path) | `particleTemplates.C:225-735` (tet walk) | — (frozen) |
| `build_table` | `collide.cpp:508-512` (linked list) | `buildCellOccupancy()` | — (frozen) |
| `attempt_count` | `collide_vss.cpp:137-161` | `noTimeCounter.C:142-153` | [P2] §2.3 `N_c` |
| attempt-count rounding | **`collide_vss.cpp:147-149`** default `+U` path (followed) | `dsmcCloud.C:105-116` (seeding, now moot) | [P2] `j=1` branch, §7.3 |
| `pair_ntc` | `collide.cpp:530-533` | `noTimeCounter` | [P1] "determined externally" |
| `pair_perm` | — | **`noRepeatCollisions`** (followed, oracles) | [P2] eq. `factorizationEa` |
| sub-cell pairing | `partners subcell`/`nearcp` | 8-octant default | *not adopted*, 8.3 |
| `collision_rate` `q` | `collide_vss.cpp:214-224` | `VariableHardSphere.C:80-121` | [P2] `q(v-v1, sigma)` |
| `acceptance_prob` | `collide_vss.cpp:225-231` (live) | **`noTimeCounter.C:231-246`** (stale, followed) | [P2] `h_2 = q/Sigma` |
| `log_h` reject branch | `collide_vss.cpp:231` | — | [P2] eq. `back_eta`, `j=3` |
| `make_prefactor` | `collide_vss.cpp:122-125` | `VariableHardSphere.C:80-121` | — |
| `scatter` | `collide_vss.cpp:405-452` | `VariableSoftSphere.C:189-241` | [P1] eq. `w_eqn_both`; [P2] `back_D` |
| `fnum` per cell | `cinfo.weight` | **`dsmcAxisymmetricI.H:55-58`** (followed) | — |
| `move` periodic | `domain.cpp:301-330` | `dsmcCyclicBoundary` | [P3] §3.1 |
| `move` specular | `domain.cpp:337-357` | `dsmcSpecularWallPatch.C:47-62` | [P3] §3.2, eq. `C_ki` |
| `move` outflow | `domain.cpp:294-295` | `dsmcDeletionPatch` | — |
| diffuse wall (M4) | `surf_collide_diffuse.cpp:206-320` | `dsmcDiffuseWallPatch.C:106-111` | [P3] §4 |
| CLL wall | `surf_collide_cll.cpp:234-320` | `dsmcCLLWallPatch.C:151-206` | — |
| inflow flux (M5) | `fix_emit.cpp:166-174` | `dsmcFreeStreamInflowPatch.C:91-142` | [P3] §5 |
| LB rotational (deferred) | `collide_vss.cpp:457-603` (rejection) | **`LarsenBorgnakke*.C:243-278`** (closed form, zeta=2) | — |
| `moments` | `compute_grid.cpp` | `dsmcVolFields` | [P3] eq. `OTD_obj_space` |
| `surrogate` | — | — | [P2] eq. `gamma_rules` |
| transport adjoint | — | — | [P3] eq. `adjoint_gamma_R` |

Bold entries are where we follow [DF] over SPARTA; section 18 explains each.

---

## 16. Open items to settle during implementation (status 2026-10-21)

1. **`A_max` default — settled by measurement, not a constant.** `initialize.suggested_a_max`
   gives an advisory start from the inflated majorant; `simulate.run_adaptive` settles the working
   value (8.5.1).
2. **`vremax` reset cadence — settled: reset at run start only.** SPARTA supports
   `collide_modify vremax Nevery startflag` (`collide.cpp:2353-2361`); [DF] never decays it. The
   worry that a never-decaying `Sigma` degrades the gradient is backwards (7.2.2: the most inflated
   `Sigma` had the lowest variance, bias flat); steady-state inflation is mild (1.40x over 1e4 -> 1e8
   attempts). The real costs are proportional compute (2.29x after a `T: 1->4->1` pulse) and the
   silent `A_max` clipping hazard (8.2): size generously, keep both diagnostics asserted.
3. **Sort at scale — settled:** the full packed-key sort is used on GPU and the carried partition sort
   on XLA:CPU (6.2).
4. **Environment:** see the README (`source ~/activate_jax_env.sh`).

### 16.1 Reference cases adopted as validation targets

[DF] ships six cases under `run/hyStrath/dsmcFoam+/` mapping onto the milestone ladder:
`heatBath-5species` (0-D relaxation, M1; nearest realised analogue: the BKW gate), `hypersonicCorner` and
`supersonicFlatPlate` (literature profiles, Bird 1994 sections 16.2 / 14.3), `couette_N2-O2`
(M4; nearest analogue `examples/couette`), `orion107kmNR` (capsule from STL, M5; the nearest realised
analogue is the SPARTA circle/level-set deck in `examples/mesh`), and `axisymmetricFlatnosedCylinder`
(deferred; needs RWF, 3.3). Only the BKW, Couette, Bird shock (21) and Taylor-Green (22) targets have
been run.

---

## 17. Structural facts verified numerically during Phase 0

These were checked with a standalone NumPy script, `docs/phase0_verification.py` (runnable:
`python3 docs/phase0_verification.py`, exits non-zero on failure), before any JAX code was
written, because the whole architecture rests on them. Each becomes a unit test in the suite.

| Claim | Result |
|---|---|
| Triangle-wave specular fold == brute-force reflect-until-inside | agrees to 1e-9 over 2e4 randomised inputs spanning ±50 box lengths, in both position and velocity sign |
| Folded position always inside `[lo, hi]` | holds for every input tested |
| `j = (i + 1 + r) mod np`, `r ~ U{0..np-2}` gives `j != i` | holds exactly, `np in {2,3,5,17,64}` |
| ...and `j` uniform on the complement of `{i}` | chi-square `p in [0.03, 0.73]` across all `np`, 6e5 draws each |
| `A(sigma, alpha)` == Jacobian of the collision map at fixed `sigma` | agrees with central differences to 1e-6 |
| `B == A^T` ([P1] eq. `AB_def`) | exact |
| Collision map conserves momentum and energy | exact |
| The collision **map** is an involution | `coll(coll(z, sigma), alpha) == z` to machine precision |
| DiCE `magic(L)` has value 1 and gradient `dL` | exact; surrogate value == `J` |
| Score reward attribution == `phi_i + phi_i1` | exact |
| `pair_perm` never self-pairs or reuses a particle within a step | holds, `np in {2,4,16,65}` |
| `pair_perm` is a perfect matching (even `np`) | every particle used in every step, exactly |
| `pair_perm` drops exactly one particle per step (odd `np`) | holds, and uniformly at random |
| `pair_perm` partner uniform over the other `np-1` | chi-square `p in [0.27, 0.87]` |
| Stale-max leaves `Sigma` bitwise invariant across the attempt loop | holds by construction |
| SPARTA's live ratchet accepts fewer pairs, **but only when `Sigma` is not yet a true majorant** | 0.00% deficit once `Sigma >= max q`; 9% at `Sigma = 0.6 max q`; 21% at `0.2 max q`. See `scratch/probe_ntc_majorant.py` |
| `qmax` mask: inactive slots cannot inflate the next majorant | holds |
| **rev-2 defect:** `grad` w.r.t. `dt`/`fnum`/`volume` through `floor(f)` | **exactly `[0,0,0]`**, while `‖dJ/dv‖ = 9.5`. Fixed by stochastic rounding + count score, §8.1 |
| Count-score estimator is unbiased vs. Rao-Blackwellised truth `8.088e4` | `8.373e4 ± 1.19e4` at 1e5 samples |
| Count score needs a baseline | sd `3.76e6` -> `4.50e4`, **84x** |
| FD at fixed randomness cannot see the count sensitivity | FD `= 0.000e+00` for every eps that does not flip `n`; `1.2e5` when it does |
| `move->collide` gives transport **outside** `B_k`; `collide->move` inside | exact match / `2.45e-02` mismatch each way; differ by `O(dt·‖B−I‖)` |
| **rev-2 defect:** empty cells clobber real collisions in the attempt scan | **176/300 runs** broke momentum conservation. Fixed by the `N+1` scratch row, §8.4.1 |
| `randint(key, shape, 0, 0)` returns 0 silently in JAX 0.6.2 | 2000/2000 seeds; no error, no NaN |
| Out-of-bounds `order[N]` is clamped by JAX to `order[N-1]` | holds — trailing empty cells reach an unrelated particle |
| rev-2 loop was not self-consistently differentiable | `grad` vs FD relerr `6.3e-3`; after fix `1.1e-10` |
| Scratch-row fix cost | 1.02x–1.11x wall time, 24 bytes |
| Causal vs naive DiCE variance ratio | saturates at **~3x**, flat over `M = 8..4096` (`M^-0.005`) — bounded, not horizon-scaling |
| `safe_norm` cotangent is bounded by 1.0 regardless of `eps` | exact, across 14 decades of `\|v\|` — so `eps` is the wrong knob for `safe_sqrt` |
| `safe_pow` cotangent is genuinely unbounded | `2.5e16` at `omega=1.2, \|vr\|=1e-12`; harm is **dominance** (`3.2e14x` the rms), not precision loss |
| Degenerate `\|vr\|` unreachable in equilibrium, reachable from cold/cloned parcels | `min \|vr\| = 1.49 m/s` over 1e8 Maxwellian draws; `3.4e-11 m/s` from round-off-duplicated parcels |
| `EPS_D` as an absolute cut is unit-dependent | **order-unity** (`1.28`) output divergence between SI/nondim/CGS; relative form exact to `5.6e-17` |
| Majorant violation *biases* the gradient, not just drops a term | **75% low** at 15% violation, 14% low at 1% |
| Stale rule causes violations; live ratchet cannot | `145-191/12000` vs **`0/12000`** |
| Gradient variance **falls** with `Sigma`, plateauing ~10x max q | `4.97x` decrease from `1x` to `100x`; singularity is at `Sigma -> max q+`, not infinity |
| Monotone `vremax` inflates `2.29x` after a thermal pulse | costs compute and silently overruns a cold-sized `A_max`; does **not** hurt the gradient |
| A valid FD window exists but closes with problem size | `h_crit ~ 5.8/(N*C*M)`; agreement `3.3e-13` inside it; **no valid `h`** at `N*C*M ~ 1e13` |
| One cell flip rebinds 60 pair indices | 3.9% of pairings; 84% at `h=1e-2` |
| Oracle 1 FD-safe at any `h`; Oracle 2 needs `h <= 1e-5` | `2.0e-16` at `h=0.1` vs. accept-bit flips at `1.9e-3`/bit |
| U1 **closed**: the `safe_pow` clamp bound holds and is exactly tight | ratio bound/achieved = **1.0000** across `omega in [0.5,1.5]`, attained at `vr2 = eps`; `omega=1.2` clamped max `22.1` vs `2.52e16` unclamped |
| U1 **closed**: `EPS_REL_VR2` stays at `1e-8` | at `omega <= 1` nothing distinguishes `1e-8` from `1e-6`; the two measured differences both favour `1e-8` (17.4) |
| U3 **closed**: a 2-5x majorant margin is achievable | thermal a-priori rule holds min margin **2.15** with **zero** violations; SPARTA's own seed fails at `1.0e-2` (17.5) |
| The bound is on `dq/d\|vr\|`, not `dq/d(vr2)` | as-labelled it is violated by `1/(2 sqrt(eps))` = **12.4x**; see 2.4.1 |
| `safe_norm` is gradient-safe where the naive form is not | `grad(naive_norm)(0)` = `NaN`, `grad(safe_norm)(0)` = `0` |

### 17.4 U1 closed: `EPS_REL_VR2` stays at `1e-8`

The gate had two halves. The **bound** half was settled in §2.4.1: `2|1-omega| eps^(0.5-omega)`
is the exact, attained supremum of `||dq/dv||`, and the original §19 row mislabelled its LHS.

The **value** half was contested — two independent agent runs recommended raising `1e-8` to
`1e-6`, on a peak-ratio metric `D = sup||grad q|| / rms||grad q||` that improves as
`eps^(0.5-omega)` (at `omega=0.81`, `273 -> 65`). That recommendation is **rejected**. `D` is a
*conditional* number: it says how bad one at-the-clamp pair is, never how often one occurs. Three
measurements decide it:

| Measurement | `eps=1e-8` | `eps=1e-6` | Reading |
|---|---|---|---|
| Exactly-degenerate pair `vi == vj`, `\|\|dq/dv\|\|` | **0.0** | **0.0** | the premise of `D` is false |
| 40e6 equilibrium pairs, count reaching the clamp | 0 | 0 | the clamp is never active either way |
| sd of the mean gradient, ratio `1e-6/1e-8`, `omega=0.75/0.81` | — | `0.9998` / `1.0005` | indistinguishable |
| sd ratio at `omega=1.2` | — | `1.166` | `1e-8` is **17% quieter** |
| Value jump at the clamp, `eps^(1-omega)`, `omega in [0.6,0.95]` | — | `2.4-6.3x` larger | `1e-8` is smoother |

The decisive line is the first. `safe_pow` is a double-`where` onto a **constant fallback**, not a
`maximum(x, eps)` clamp, so *below* the clamp both the value and the gradient are exactly zero. A
duplicated parcel, a zero-`T` IC, a cold freestream — every configuration the `D` argument named —
sits below the clamp and injects **nothing**. The supremum is attained only in a thin shell at
`|vr| = sqrt(eps) * v_ref`, a set of probability `~eps^1.5`: the same rare set whose variance
contribution the second and third rows show is unmeasurable.

Scaling note, recorded because it is *correct but irrelevant*: the clamp region's contribution to
gradient variance goes as `eps^(2.5-2*omega)`, which for `omega < 1.25` *decreases* with `eps` —
predicting `1e-8` should be `10x` quieter at `omega=0.75`. It is not, because that contribution is
`~1e-8` of the bulk variance. Both the peak argument and this one are below the noise floor.

**`omega = 1.0` has `dq/dv == 0` identically** (`q = prefactor * vr2^0`), so U1 cannot bind at the
M1 BKW gate at all. It first becomes live at M3/M4 argon (`omega = 0.81`), where the table above
says the choice does not matter. `1e-8` is kept because it is the spec-frozen value and no measured
evidence supports moving it.

### 17.5 U3 closed: the margin is achievable, but SPARTA's seed does not achieve it

Gate: can a seeding rule hold `Sigma >= max q` with a 2-5x margin when `max q` is **not** known
ahead of time? Argon `omega=0.81`, `N=600`, `A_max=3000`, 60 steps, three ICs x with/without a
3%/step heating transient. `q` is monotone in `|vr|`, so the exact `max q` over all `N(N-1)/2`
pairs is computable each step and the margin is measured, not inferred.

| Seeding rule | Violation rate | Min margin | Attempt cost | |
|---|---|---|---|---|
| SPARTA seed `2*cxs*vscale` + stale-max ratchet | `1.1e-3` equil, **`1.0e-2` beams** | **0.62** | 1.4-3.0x | **FAILS** |
| SPARTA seed x5 + ratchet | 0 | 1.38 | flat 5.0x | passes, but pays 5x even in a cold gas |
| **A-priori thermal, `K=5`, `F_up=2`** | **0** all 6 cases | **2.15** | 1.4-6.7x | **adopted** |

The failing rule is the one currently in the spec. A pure ratchet *lags*: it can only raise `Sigma`
after a violation has already been counted, and a violated pair contributes zero score and biases
the gradient (75% low at 15% violation, 14% at 1%). The fix is to make the majorant **a priori**
rather than observed — `q` is monotone in `|vr|`, so on the reachable interval its supremum sits at
an endpoint, and the cell temperature bounding that endpoint is known at the *start* of the step:

```
thermal_majorant(T, omega) = F_up * prefactor * max( (K_EV^2 * 2 k T / m_r)^(1-omega),
                                                     eps_abs^(1-omega) )      # K_EV=5, F_up=2
Sigma_step = stop_gradient(max(cstate.vremax, thermal_majorant(T_cell, omega)))
```

This is a **design input to §8, not a tuning knob**: `vremax_init` must be the thermal majorant,
not SPARTA's `2*cxs*vscale`. It also explains why the beam IC — where the ratchet is worst — is
the case the thermal rule handles most easily: the beams inflate the cell temperature, so the
a-priori bound rises with them.

**The margin costs compute, not physics.** `n_attempts ~ Sigma` and `P_accept = q/Sigma`, so the
collision count is `Sigma`-invariant — measured flat over a 20x sweep (`1676, 1655, 1649, 1694`
collisions at `Sigma/Sigma_0 = 1, 2, 5, 10`). At `Sigma/Sigma_0 = 0.5` the acceptance rate saturates
at `0.957` and collisions fall to `1436`: under-collision is *visible in the collision count*, which
is what `min_majorant_margin` diagnoses. Since gradient variance was separately measured to **fall**
with `Sigma` (4.97x from 1x to 100x), the margin buys accuracy and variance together and pays only
in scan slots — which is also why `A_max` must be sized against the inflated `Sigma` (U4).

### 17.1 Correction to [P1]: `A` is singular, so `A^-1` does not exist

[P1] §2.2 writes "`B = A^T = A^{-1}`". The first equality is correct and is the one the adjoint
uses. The second is not literally true of the 6x6 matrix: **`A` has singular values
`[1, 1, 1, 1, 0, 0]`** — it is rank 4.

The reason is structural and worth understanding, because it explains what the adjoint can and
cannot recover. At fixed `sigma`, the collision map

```
v'  = 0.5*((v + v1) + |v - v1| sigma)
v1' = 0.5*((v + v1) - |v - v1| sigma)
```

depends on `(v, v1)` only through `v + v1` (3 dof) and `|v - v1|` (1 dof) — 4 degrees of freedom
out of 6. The two lost directions are rotations of the relative-velocity vector that preserve its
magnitude; with `sigma` held fixed they leave the output completely unchanged. So `A` annihilates
a 2D subspace and cannot be inverted.

What [P1] means by the involution is the **change of variables** `(v, v1, sigma) -> (v', v1', alpha)`,
which is indeed an involution with unit Jacobian — the map, not the matrix. Applying the collision
a second time with `sigma` and `alpha` exchanged recovers the input exactly (verified above). That
map-level property is what a future reversible-memory adjoint would rest on; the matrix identity is
not needed and is not available.

Practical consequences:
- The backward recursion `gamma^k = A^T gamma^{k+1}` is a VJP, not an inversion. It is *supposed*
  to lose the 2 annihilated directions — that is correct adjoint behaviour, not a bug.
- **Sanity check for the Oracle-1 test:** assert `rank(A) == 4` and
  `svd(A) == [1,1,1,1,0,0]`. If a future refactor ever produces a full-rank `A`, the collision
  map has been altered.
- Do not attempt to "invert the collision" by solving with `A`. Invert it by re-applying the
  map with `alpha` in place of `sigma`.

#### 17.1.1 CORRECTION (2026-09-05): `A` is not the Jacobian of *our* collision map

The bullet above says "if a future refactor ever produces a full-rank `A`, the collision map has
been altered". **Our collision map was never the one `A` describes**, and the rank is already 6.
Measured (`scratch/adjoint/probe_p1_frame.py`, and asserted in
`tests/test_adjoint_oracle.py::test_production_scatter_is_not_p1_A_and_that_is_expected`), at
`omega = 1`, `alpha = 1` — precisely [P1]'s regime:

| pair Jacobian at frozen `(u_chi, u_eps)` | `‖J − A‖ / ‖A‖` | rank |
|---|---|---|
| [P1]'s map, `sigma` drawn in the **lab frame** | **3.5e-18 … 9.2e-17** | 4 |
| our `vss.scatter`, `sigma` drawn in the **`v_r` frame** | **0.53 … 0.81** | **6** |

The cause is §7.5's note, taken one step further than §7.5 takes it. SPARTA builds the
post-collision direction *in the frame aligned to `v_r`* (`collide_vss.cpp:429-437`): the
deflection is by polar angle `chi` **from the `v_r` axis** with azimuth `eps` about it. At frozen
`(u_chi, u_eps)` the map `v_r -> v_r'` is therefore a rotation whose axis is `v_r` itself, so
`v'` depends on the *direction* of `v_r` and not only on `v_cm` and `|v_r|`. That is the two extra
degrees of freedom, and it is exactly the `delta sigma = 0` assumption
(`revision_v1_clean_version.tex:468`) that [P1] states and we do not satisfy.

**This is not a defect and not a physics difference.** Both maps push forward the same
distribution — `cos chi` uniform on `[-1, 1]` about any axis is isotropic — so both are valid
Maxwell-molecule DSMC, both pass the M1 BKW gate, and both give *unbiased* pathwise gradients.
They are different **reparameterisations**, in the same sense that a Gaussian has many.

Three consequences, all of which cost something:

1. **§13.1 as originally written cannot be implemented.** An Oracle 1 that drove `vss.scatter`
   and compared against eq. `AB_def` fails by 44%–435% on the first try; that is what happened.
   The oracle must run [P1]'s own map. §13.1 is revised accordingly.
2. **The rank-4 assertion is about the analytic matrix, not our code.** A test asserting
   `rank(jax.jacobian(scatter)) == 4` would fail today. The structural check is kept, on
   `_p1_matrix`, where it is meaningful.
3. **Open question, not yet measured: which reparameterisation has lower gradient variance?**
   Both are unbiased; nothing says they are equally noisy, and §23.3 shows variance is the
   binding constraint on this whole approach. Registered as **U10**.

---


## 18. Where the two reference codes disagree, and which we follow

SPARTA and dsmcFoam+ are both mature, validated production DSMC codes, and they make different
choices in several places. Each divergence is a decision we had to make explicitly. The pattern
that emerges is not "one code is better" — it is that **dsmcFoam+ tends to be closer to the
probabilistic framework the adjoint papers are written in**, while SPARTA tends to be closer to
what a Cartesian-grid GPU implementation wants structurally.

### 18.1 The NTC majorant — we follow [DF]

| | SPARTA | dsmcFoam+ |
|---|---|---|
| Within the candidate loop | `vremax = max(vremax, vre)` then test against the **new** value (`collide_vss.cpp:225-231`) | test every candidate against the value captured **before** the loop (`noTimeCounter.C:231-246`) |
| New maximum published | immediately | for the next step only |

[DF]'s inline rationale: `N_pairs` was computed from the pre-loop `Sigma`, so raising `Sigma`
mid-loop leaves you with too few candidates *and* a reduced acceptance rate — under-collision
twice over. The rule is documented and held consistently across every NTC variant in that
codebase.

**How much does it actually matter?** Measured in `scratch/probe_ntc_majorant.py`, sweeping
majorant tightness with a realistic VHS `q` distribution:

| `Sigma_0 / max(q)` | ratchet fires | acceptance deficit, live vs. stale |
|---|---|---|
| `>= 1.0` | 0% | **0.00%** — the rules are identical |
| 0.8 | 12% | 0.22% |
| 0.6 | 100% | 9.3% |
| 0.2 | 100% | 21.0% |

So the accuracy argument is **narrower than it first appears**: whenever `Sigma` is a genuine
majorant the two rules agree exactly, as they must, and both codes converge to the same physics
once `vremax` has settled. The divergence is a warm-up and transient effect — it bites while
`vremax` is still climbing from its coarse seed, and again whenever the flow heats up, since the
majorant only ever adjusts upward.

**And it has a real cost, measured after the fact.** SPARTA's live ratchet makes `q > Sigma`
impossible by construction (`0/12000`); the stale rule permits violations (`145-191/12000`,
1.2-1.6%), and a violated attempt contributes a **zero** score with no diagnostic, *biasing* the
gradient — 14% low at a 1% violation rate, 75% low at 15% (7.2.1). We traded a
guaranteed-valid majorant for a fixed one, and the trade only pays if `Sigma` is actively kept
above `max q` by a 2-5x safety factor plus the `majorant_violations` assertion.

**Followed anyway, and mainly for a different reason.** The decisive argument is not accuracy but
that the stale rule makes `Sigma` genuinely constant within a step — exactly the **fixed
majorant** [P2]'s virtual-collision framework assumes and its unbiasedness argument (§5.6) rests
on. Under SPARTA's rule the score term `log h = log(q/Sigma)` would be differentiating against a
quantity that moved during the very loop it is scoring, which is not a small correction to the
estimator but a category error in it. The consistency gain during warm-up is a bonus, and so is
the structural one: `Sigma` leaves the scan carry (8.4).

### 18.2 Pair selection — we ship both

The papers pair *without* replacement ([P2] eq. `factorizationEa` is a product over `N/2`
disjoint pairs). SPARTA and [DF]'s default `noTimeCounter` pair *with* replacement; [DF] also
ships `noRepeatCollisions`, which is the papers' scheme.

**Both, behind one interface** (8.3). `pair_ntc` is the production default because the attempt
count must follow the physical rate. `pair_perm` is what the oracle tests use, so they validate
against the published algorithm rather than a variant of it.

Not adopted from either code: sub-cell / nearest-neighbour partner selection. It improves
accuracy but makes the selection probability position-dependent, breaking the `E_p` commutation
that [P2] §3.1.2 relies on. See the note in 8.3.

### 18.3 Maxwellian construction — we follow [DF]

SPARTA's Rayleigh-magnitude polar sampler and [DF]'s three independent normals produce the same
distribution (5.1). [DF]'s is trig-free, uses one draw instead of four, and removes
`sqrt(-log U)` from the initialization path — one fewer singularity in the gradient inventory.
The Rayleigh form is retained only where flux weighting genuinely requires it.

### 18.4 Particle weighting — we follow [DF]'s shape, not its mechanism

[DF] routes every use of the particle weight through `nParticles(cell) = F_N * RWF(cell)`;
SPARTA carries a per-cell `cinfo.weight`. Both are per-cell; our original spec had a scalar.
**Followed:** `fnum` is `f64[C]` (3.1).

**Not followed:** [DF]'s radial-weighting clone/delete step, which duplicates a parcel moving
inward and probabilistically destroys one moving outward (`dsmcAxisymmetric.C:50-219`). That
changes the particle count at runtime, which breaks the static-shape contract of 0.4 and makes
the adjoint's particle correspondence ill-defined. The per-cell field is the useful half; it can
carry a weighting scheme later without the clone/delete machinery.

### 18.5 Collision-count rounding — SPARTA's `remain no` path, not its default `remainflag` carry

Revision 2 transcribed SPARTA's `remainflag` carry scheme and adopted [DF]'s `U[0,1)` seeding of
`collisionSelectionRemainder` (`dsmcCloud.C:105-116`) to remove the resulting startup transient.

Revision 3 **drops the carry entirely.** SPARTA has three schemes
(`collide_vss.cpp:137-161`); its default is the `remainflag` carry (`collide.cpp:87`,
`remainflag = 1`; corrected 2026-10-21 -- earlier revisions called stochastic rounding the
default), and `collide_modify remain no` gives `n = floor(f + U)`, plain stochastic rounding.
That is what we take, because the deterministic
carry makes `P(n | f, remain)` degenerate, so there is no probability to differentiate and the
`dt` / `fnum` / `cell_volume` sensitivities are lost outright (§8.1, measured as exactly zero).

Stochastic rounding is unbiased from step one, so [DF]'s seeding fix addresses a transient we no
longer have. The citation stays as the reason we looked at the remainder handling at all.

### 18.6 Mesh and tracking — we follow SPARTA

This is the one where SPARTA's design wins decisively for our purposes.

| | SPARTA | dsmcFoam+ |
|---|---|---|
| Mesh | Cartesian hierarchical, cut/split cells | arbitrary polyhedra |
| Locating a particle | closed-form arithmetic (+ tree descent if refined) | incremental tet walk from the previous position |
| Per-particle state | cell index | `celli, facei, tetFacei, tetPti` |
| Loop shape | none on the fast path | data-dependent `while` over tet crossings |

[DF]'s tracker is genuinely more general — it is correct on warped, non-planar, arbitrary
polyhedral cells, which is real engineering value. But in JAX a data-dependent walk becomes a
bounded `scan` at a worst-case trip count with heavy masking, paying for the worst case on every
particle every step, and it adds four persistent integers per particle to the state that the
adjoint must carry. SPARTA's approach keeps the fast path branch-free.

**Followed: SPARTA-style, for M5.** [DF]'s tet tracker is recorded as the fallback if genuinely
unstructured meshes ever become necessary.

Two [DF] details worth carrying into M5 regardless: per-tet stochastic fill
(`dsmcMeshFill.C:68-238`) makes particle counts follow exact sub-volume rather than
cell-averaged volume — precisely what cut cells need; and freshly injected inflow parcels get a
randomised `stepFraction()` on their first move, a deliberate anti-pileup measure preventing an
artificial density spike on the inflow face.

### 18.7 Larsen-Borgnakke — [DF] narrows a hazard we had overstated

Earlier drafts flagged LB as a differentiability hazard on the strength of SPARTA's
Beta-distribution rejection sampler (`collide_vss.cpp:457-603`). [DF]
(`LarsenBorgnakkeVariableHardSphere.C:243-278`) shows the diatomic case `zeta_rot = 2` — the
common one — has a closed-form inverse CDF:

```
E_rot / E_c = 1 - R^(1/(5/2 - omega))
```

No rejection sampler, and differentiable as written. Only general polyatomic `zeta > 2` needs the
gamma-distribution rejection loop. [DF] also fixes the mode ordering (electronic -> vibrational ->
rotational, each gated by its own collision-number probability) and supplies the
temperature-dependent `Z_v(T)` formulation. The hazard is real but confined to polyatomics.

### 18.8 RNG reproducibility — neither, we are stricter

[DF] seeds from `clock::getTime() + const * Pstream::myProcNo()`, so runs are not reproducible
unless `seedNumber` is pinned — and even pinned, each MPI rank takes a distinct offset by design.

Our contract (section 1) is strictly stronger and has to be: the finite-difference gradient
checks in 13.3 are meaningless without bit-identical common random numbers across perturbed
runs, and `jax.checkpoint` rematerialisation requires the backward pass to replay exactly the
stream the forward pass consumed.

---

### 18.9 Running register: every divergence from SPARTA

SPARTA is the primary reference implementation, so **every place our algorithm differs from it is
a decision that has to be defensible.** This is the running list. It is deliberately a single flat
table rather than prose scattered through the document, so that a reviewer can audit the whole
surface at once, and so that anything added later has an obvious place to go.

`Driver` says *why* we diverged, and is the column that matters:

| driver | meaning |
|---|---|
| **JAX** | SPARTA's construction cannot be traced, jitted, or statically shaped |
| **ADJ** | it would break the adjoint (frozen randomness, a missing score, or a moved majorant) |
| **GRAD** | the primal is fine but the *gradient* is `NaN`, unbounded, or exactly zero |
| **MEAS** | we measured SPARTA's form to be wrong or hazardous **for our use**, with numbers |
| **[DF]** | dsmcFoam+ does it differently and we judged its version better |
| **GPU** | SPARTA's data structure is a CPU choice; a GPU wants another |

Anything marked **MEAS** has a scratch probe behind it. Nothing here is a matter of taste.

| # | Topic | SPARTA | Ours | Driver | Why, and where it came from |
|---|---|---|---|---|---|
| S1 | NTC majorant | ratchets `vremax` **inside** the candidate loop and tests against the updated value, `collide_vss.cpp:225-231` | `Sigma` read once, fixed for the whole loop; new max published for the next step | **ADJ**, [DF] | [P2]'s score `log(q/Sigma)` must not differentiate against a quantity that moved during the loop scoring it — a category error, not a small correction. §18.1 |
| S2 | Majorant **seed** | `vremax = 2*cxs*vscale`, `collide_vss.cpp:107-133` | a-priori thermal majorant `F_UP * prefactor * (K_EV^2 * 2kT/m_r)^(1-omega)`, `K_EV=5`, `F_UP=2` | **MEAS** | **Diverges from both reference codes.** SPARTA's seed under the stale rule measured a `1.0e-2` violation rate and margin **0.62**; each violation contributes a zero score and biases the gradient. Gate U3, §7.2.3/§17.5 |
| S3 | Attempt-count rounding | three paths; the `remainflag` path carries a deterministic remainder, `collide_vss.cpp:137-161` | SPARTA's `collide_modify remain no` path (its default is the carry), stochastic rounding `n = floor(f + U)`, plus the score `log P(n\|f)` | **GRAD** | Under a deterministic carry `P(n\|f,remain)` is degenerate, so `dt`, `fnum` and `cell_volume` have **no gradient path at all** — measured as exactly `[0,0,0]`. §8.1, §18.5 |
| S4 | Partner resampling | resamples until `i != j`, `collide.cpp:530-533` | closed form `j = (i + 1 + r) mod npc` | **JAX** | A `while_loop` with a data-dependent trip count is neither reverse-differentiable nor statically shaped. Uniformity on the complement verified by chi-square. §8.3 |
| S5 | Empty/degenerate cells | not reachable — linked lists never present an empty cell to the pair chooser | `npc` clamped to `>= 2`, plus a scratch row at index `N` | **MEAS**, JAX | `jax.random.randint(k, shape, 0, 0)` returns 0 **silently** (2000/2000 seeds), so empty cells indexed into a neighbour's particles: 176/300 runs violated momentum, with no error and no `NaN`. §8.4.1 |
| S6 | `vr2` clamp | `vr2 ** (1-omega)` unguarded, `collide_vss.cpp:222` | `safe_pow` with a **relative** clamp `EPS_REL_VR2 * (2kT_ref/m_r)` | **GRAD** | Unclamped, `dq/d\|vr\|` is unbounded (~`2.5e16`). Relative, not absolute: the bound becomes exactly `2\|1-omega\| eps^(0.5-omega)`. Gate U1, §2.3/§17.4 |
| S7 | Scattering-frame guard | `if (d > 1.0e-6)`, `collide_vss.cpp:429` | `d > EPS_D * vrm` — **relative** — via the double-`where` pattern | **GRAD**, MEAS | SPARTA's test is *dimensional* and only works because it runs SI with thermal speeds ~1e2 m/s. Measured an **order-unity** change in post-collision velocity from a change of units alone. §7.4 |
| S8 | `alpha == 1` fast path | `if (fabs(alpha_r - 1.0) < 0.001)` branch, `collide_vss.cpp:405+` | dropped; the general branch is used always | **JAX** | `alpha` is a traced parameter, so a Python branch on it is illegal. The general form is already correct at `alpha = 1`; SPARTA's case is a speed shortcut, not different physics. §7.4 |
| S9 | Cell occupancy | per-cell linked lists, `collide.cpp:508-512` | sort: packed-key `lax.sort` + bucket bounds; on XLA:CPU optionally the carried partition sort (`SimStatic.sort`) | **GPU** | Linked lists are a CPU structure. A sort is the right GPU primitive and its permutation is a frozen integer decision, so gathers stay differentiable. The partition sort keeps SPARTA's per-cell membership with a different within-cell order. §6 |
| S10 | Maxwellian sampling | polar `vn/vr` construction, `create_particles.cpp:761-780` | three independent normals | [DF], GRAD | Statistically identical; removes `sqrt(-log U)` and its singular gradient at `U=1` from the init path entirely. §5.1, §18.3 |
| S11 | Specular reflection | mirrors once, assumes at most one crossing, `domain.cpp:337-357` | closed-form triangle-wave fold, any number of bounces, branch-free | **JAX** | Strictly more general at no cost, and branch-free is what a static-shape kernel wants. [P3] §3.2 makes the same single-crossing assumption. §9.2 |
| S12 | Particle weighting | scalar `fnum` with an optional `cinfo.weight` | `fnum` is a per-cell field `f64[C]` from the first commit | [DF] | Retrofitting it later would mean a simultaneous diff across `ntc.py`, `sample.py` and `boundary.py`. Comes with a mass-conservation guard (gap U7). §3.3, §18.4 |
| S13 | RNG | `clock::getTime() + const*myProcNo()`, per-rank offsets | `fold_in(fold_in(base_key, step), stage_id)`, recomputed never carried | **ADJ** | Finite-difference gradient checks are meaningless without bit-identical common random numbers across perturbed runs. §1, §18.8 |
| S14 | RNG draw site | drawn per candidate inside the loop | **all** draws batched to `(A_max, C)` before the scan | **MEAS** | Per-slot draws are ~9 tiny kernels per iteration: compile `17.0s -> 2.4s` and per-slot `102 -> 53 us` when hoisted; the test suite went 26 min -> 4 min. §1 rule 3, §8.6 |
| S15 | Sub-cell partners | offers `partners subcell` / `nearcp` | **not adopted** | **ADJ** | It makes the pair-selection probability position-dependent, so [P2] §3.1.2's `E_p` no longer commutes with position derivatives and the inhomogeneous adjoint acquires another score term. A known accuracy trade, recorded not taken. §8.3 |
| S16 | Intra-step ordering | `move -> migrate -> sort -> collide`, `update.cpp:337-364` | `move -> sort -> collide` (**follows SPARTA**; [P3] specifies the opposite) | — | Listed because it is a divergence *from the papers*, not from SPARTA. It puts the transport term outside `B_k`; verified by real `jax.vjp`. Invisible to every 0D oracle (gap U5). §12.1 |
| S17 | Cell lookup | `ravel_multi_index`-equivalent index arithmetic | explicit static row-major strides | **JAX** | `jnp.ravel_multi_index` defaults to `mode="raise"`, which needs concrete indices and cannot be traced. `mode="clip"` would trace but silently absorb an out-of-range index — the S5 failure mode. §4 |

| S18 | Multi-device | `comm->migrate_particles()` over a decomposed grid, MPI | 1-D x-slab decomposition, `lax.ppermute` nearest-neighbour exchange, fixed-size migration buffer | **JAX** | Same decomposition SPARTA uses; the difference is that static shapes make the buffer a *capacity* that can overflow, so `overflow_send`/`overflow_recv` are counted every step and are errors. Array-sharding the particle axis instead was measured and does **not** parallelise: the global `argsort` replicates. §20 |
| S19 | Per-device RNG | per-rank clock offsets | `fold_in(base_key, axis_index)` | **ADJ** | Slabs must draw independent streams or cell *j* of every slab gets identical uniforms. Consequence: a decomposed run is statistically but not bitwise equal to a single-device one, so a finite-difference check must hold the device count fixed. §20.3 |

#### 18.9.1 What is *not* a divergence

Worth stating, because the list above can read as though we rewrote everything. The physics core
is SPARTA's, unchanged: the VSS cross-section prefactor (`collide_vss.cpp:122-125`), the
`sigma_T v_r = prefactor (vr2)^(1-omega)` rate law, the VSS deflection law `cosX = 2U^(1/alpha)-1`,
the mass-weighted centre-of-mass decomposition, the NTC attempt-count *formula*
`0.5 npc(npc-1) Sigma dt fnum / V`, and the `move -> sort -> collide` ordering. Where we diverge it
is almost always the *numerics around* the physics — guards, data structures, and the frozen
random decisions the adjoint needs — not the physics itself.

### 18.10 A third reference code: JAX-FLUIDS

`~/jax_fluids/JAXFLUIDS/` is a mature JAX continuum CFD solver (finite-volume compressible
Navier-Stokes, level-set interfaces). It solves a different equation, so none of its physics
transfers; what transfers is structure. Load-bearing observation: it drives its time loop from
host-side Python (a plain `while` over a jitted step body), independent confirmation that per-step
host control is affordable, which `simulate.run_adaptive` (8.5.1) relies on. Its halo exchange and
initialisation modules were never examined (not verified here; the path was not re-checked on
2026-10-21).

---

## 19. Unverified claims register

Everything in section 17 was measured. Rows below are assertions that are reasoned but untested or
measured too weakly. Rule: a row may only be deleted by moving its evidence into section 17.
Status below was re-checked against `tests/` on 2026-10-21.

| # | Claim asserted | Evidence today | Gate that would settle it | Status |
|---|---|---|---|---|
| U1 | `EPS_REL_VR2` bounds the collision-rate cotangent, `1e-8` is right | **CLOSED 2026-09-04.** Bound exactly tight (ratio 1.0000) on `d/d|vr|`; raising to `1e-6` measured and rejected (17.4) | regression: `tests/test_gaps.py::test_u1_*` | closed |
| U2 | Score estimator is unbiased at a **non-zero** gradient | Half-closed. At a known zero it passes (23.3.7: `\|z\| <= 1.53`); the `N^-1/2` law holds (23.3). Open: non-zero value, because mode-B FD at `Delta m = 0.1` is `O(h^2)` truncation-biased (`\|z\| = 6.4` for `diam`) | sweep `Delta m` at large `M_s`, Richardson-extrapolate FD to `h -> 0`, require `\|z\| < 2` at SE < 10 % | **open**; deprioritised since the production route is the ensemble adjoint, whose gates are FD arbiters in `examples/gradient_bench/results/` |
| U3 | A 2-5x safety factor on `Sigma` eliminates violations | **CLOSED 2026-09-04.** SPARTA seed + ratchet fails (`1.0e-2`, margin 0.62); the a-priori thermal majorant passes (zero violations, margin 2.15; 17.5) | regression: `tests/test_gaps.py::test_u3_*`, `tests/test_ntc.py::test_vremax_init_is_the_thermal_rule_not_sparta_s_seed` | closed |
| U4 | Sizing `A_max` from the inflated `Sigma` prevents silent clipping | The 2.29x inflation is measured; the mitigation changed: `run_adaptive` measures instead of predicting (8.5.1), and `attempts_clipped` is asserted in `test_ntc.py`, `test_mesh_forward.py` and noted in `test_inflow.py` as the catching diagnostic. No test drives a `T: 1->4->1` pulse, and none exercises `run_adaptive` | pulse test: clipping > 0 cold-sized, == 0 inflated; a `run_adaptive` bit-equality test | **open** |
| U5 | Oracle 3 discriminates intra-step ordering | **CLOSED.** `tests/test_adjoint_oracle.py::test_oracle3_*` and `test_production_step_moves_before_it_collides` (13.3) | n/a | closed |
| U6 | `logh_acc` is zeroed when inflow refills a dead slot | **CLOSED.** `inflow.revive_slots` / `revive_at`, reflection over `ACCUMULATOR_FIELDS`; contamination predicted to 1e-10 (`tests/test_inflow.py`, 9.3) | n/a | closed |
| U7 | Non-uniform `fnum` without reweighting breaks mass conservation | The guard is **built** (`state.check_fnum_uniform`, tested; `Diagnostics.configuration_invalid`). The *hazard itself* (mass drift) has never been exhibited | run non-uniform `fnum` with crossing particles; show the drift and that the guard fires | guard built, hazard unexhibited |
| U8 | `h_crit ~ 5.8/(N*C*M)` closes the FD window at production scale | Fitted over `N*C*M <= 3e6`; the production-scale conclusion is a 6-decade extrapolation; the by-product evidence of 23.2 is weak | measure `h_crit` at `N*C*M ~ 1e8` with the cap lifted | **open** |
| U9 | Scratch-row fix costs ~3-10 % | Measured at 3 sizes; the smallest is noise. The implementation changed (drop-mode scatter, 8.4.1) so the number is historical | re-benchmark with repeats and spreads | open, low value |
| U10 | `v_r`-frame and lab-frame collision reparameterisations are equally good pathwise estimators | Only unbiasedness argued (17.1.1: different maps, rank 6 vs 4); variance never compared | `Var(dJ/dtheta)` over seeds under both samplers at matched cost, 0-D and 1-D | **open** |

### 19.1 Guards against silent failure

U4, U6 and U7 share a property that makes them worse than ordinary untested claims: the failure
produces no error, no `NaN`, no diagnostic. A clipped attempt count under-collides quietly; a stale
`logh_acc` contributes a plausible wrong gradient; non-uniform `fnum` loses mass without any
conservation check noticing. The two most serious defects of this project (8.1 and 8.4.1) had exactly
that shape and survived a careful read. **Reasoning is not evidence for this class**: U4 and U7 stay
blocking for any milestone that depends on them, and the run-time counters of section 3.1 must be
asserted zero in every test.

---

## 20. Multi-device execution

### 20.1 Why the obvious approach fails, measured

Sharding the particle arrays and letting XLA insert the collectives is the first thing to try. It
is correct — the program is unchanged, so the physics is unchanged — and it does **not
parallelise DSMC**. Inspecting the shardings through one step (`jax_dsmc/parallel.py`):

```
x            PartitionSpec('dev')     <- sharded
table.cell   PartitionSpec('dev')     <- sharded (cell_of is element-wise)
table.order  PartitionSpec()          <- REPLICATED
table.counts PartitionSpec()          <- REPLICATED
moments.temp PartitionSpec()          <- REPLICATED
```

The global `argsort` in §6 collapses the sharding, and everything downstream that indexes through
`order` is replicated: every device redundantly performs the whole collision. On four
forced-host devices the collide stage took 6.8 / 13.0 / 25.2 ms at 1 / 2 / 4 devices on a
*fixed* problem — growing in proportion to the device count, the signature of replicated work
sharing one processor, not of a distributed computation.

The parallel axis of the collision operator is the **cell** axis (§8.6), so each device's
particles must be co-located with its cells. That is spatial domain decomposition.

### 20.2 The decomposition

Each device owns a contiguous slab of cells along `x`. Every stage becomes local: `move` and the
boundary fold are element-wise; `sort` becomes `P` independent sorts of `N/P` elements; `collide`
is per-cell and a cell has one owner; `sample` reduces into that device's own cells. The only
communication is the particles that cross a slab boundary, exchanged with the two neighbours by
`lax.ppermute`. The timestep is CFL-limited, so a particle moves a fraction of a cell per step and
the exchange is nearest-neighbour and small.

**Cells shard for free.** With row-major ordering and `x` on axis 0, a slab
`ix in [r*per, (r+1)*per)` maps to global cell indices `[r*C/P, (r+1)*C/P)` — a contiguous block,
in the same order as the local subgrid's own `cell_of`. So `PartitionSpec('dev')` on any `f64[C]`
field (`fnum`, `vremax`, every moment) hands each device exactly its own cells with no permutation
anywhere. This is a consequence of §4's index arithmetic, and it is why the 1-D cut is along `x`.

Boundary conditions are applied against the **global** domain before migration: a periodic wrap
spans the whole box, not a slab. The other order wraps particles at slab faces, which is a
different and wrong problem.

### 20.3 Two capacities, two silent failures

Static shapes make both the per-device particle array and the migration buffer fixed sizes, and
overflowing either loses particles rather than raising:

| capacity | overflows when | diagnostic |
|---|---|---|
| `n_dev` | a slab holds more than its share (compression) | `overflow_recv` |
| `m_max` | more particles need to leave than the buffer holds | `overflow_send` |

Both are counted every step and both are **errors**: a lost particle is lost mass and nothing
downstream notices. `far_jumps` counts particles that tried to move more than one slab, which
means the timestep is too large for the decomposition.

**Two bugs found while building this lost or duplicated particles while reporting zero overflow**,
which is why `tests/test_decomp.py` asserts exact particle conservation in D=1,2,3 rather than
comparing profiles:

1. **Padding indices clamped to slot 0.** Invalid entries in a statically-shaped scatter were
   mapped to index 0, so a padded write fought a real write to slot 0 and one won arbitrarily.
   Cost: six duplicated particles per step. Fix: an out-of-bounds sentinel with `mode="drop"`.
2. **Arrivals left uncompacted.** `incoming` is two `ppermute` buffers concatenated, so its live
   entries sit in two blocks, while the free-slot list is one run at its front. The second block
   addressed slots past the end of that run and was dropped. Cost: 328 particles in the 2-D test,
   **zero** in 1-D, where `2*m_max` happened to be smaller than the free-slot count — a bug that
   hides at one configuration and appears at another. Fix: compact by `argsort(~alive)` first.

Randomness is per-device: `fold_in(base_key, axis_index)`. Without it every slab's cell *j* draws
identical uniforms. The consequence is recorded as S19 — the frozen-randomness contract of §1
holds exactly *at a fixed decomposition*, which is what gradient checking needs, but not across a
change in device count.

### 20.4 What the node buys

Cells must resolve the mean free path, so cell count goes as `(L/lambda)^D` while
`Re ~ 2 Ma sqrt(gamma) (L/lambda)`: **Reynolds number costs `Re^D` in cells.** On 4 x H200
(450 GiB usable, `npc = 50`): `Re ~ 1500` in 2-D but only `Re ~ 66` in 3-D. The 2-D Taylor-Green
case has an exact Navier-Stokes solution and is genuinely turbulent at that Reynolds number; the
3-D `Re = 1600` benchmark is a supercomputer-scale problem and is not claimed.

---

## 21. Validation against Bird's normal shock, and the mean-free-path trap

M2's kinetic gate is Bird §12.11: the maximum-slope shock thickness
`L = (rho2 - rho1) / max|d rho/dx|`. It is the sharpest test available at this milestone because
it is the one quantity a Riemann/Euler comparison cannot reach — an inviscid solution has no
intrinsic thickness.

Result at Mach 2 in argon, all three dimensions: **L/lambda_1 = 5.98 / 6.12 / 6.18 against Bird's
6.2**, and simultaneously **4.53 / 4.64 / 4.68 against Alsmeyer's experimental 4.8**. Agreeing
with both at once is the real test, because the two references are only mutually consistent if the
conversion between their mean free paths is right.

### 21.1 Three mean free paths, differing by 60 %

**This normalisation produced a 24 % false discrepancy that survived a full day of investigation.**

| definition | argon, `n = 1e21` | what it is |
|---|---|---|
| `mfp_hard_sphere` | 1.332e-3 m | `1/(sqrt(2) pi d_ref^2 n)` with the VHS **reference** diameter |
| `mfp_vss`, Bird eqn (4.52) | **1.294e-3 m** | what Bird normalises shock thicknesses by |
| `mfp_alsmeyer` | 1.708e-3 m | hard spheres matched to argon's **viscosity** |

Bird defines (4.52) through the viscosity,
`lambda = 2 mu (5-2w)(7-2w) / [15 rho sqrt(2 pi R T)]`. Substituting the VSS viscosity (Bird 4.64)
cancels `(5-2w)(7-2w)` and collapses the `omega` dependence **entirely**, leaving

```
lambda_4.52 = [(alpha+1)(alpha+2) / (6 alpha)] * 1/(sqrt(2) pi d_ref^2 n)
```

a function of `alpha` only. The original implementation used
`alpha(5-2w)(7-2w)/[4(alpha+1)(alpha+2)] * lambda_HS` instead, which is **1.2455x too small** for
argon and made every shock read 24 % too thick.

That wrong factor is not arbitrary: it is exactly `mu_HS(d_ref)/mu_VSS = (d_HS/d_ref)^2`, a
**cross-section** ratio, and it equals 0.8244 at `w=0.75, alpha=1.5325` — a number that appears in
Bird's text near his mean-free-path discussion. Reproducing one quoted number was taken as
verification and was not. What settled it was deriving `lambda` from the *measured* viscosity,
which (a) returns exactly 1.000000 in the hard-sphere limit `w=1/2, alpha=1`, (b) matches the
closed form above to six figures at every `(w, alpha)` tried, and (c) reproduces Bird's stated
"76 % of the hard sphere mean free path" as 0.7577.

**Lesson, and it generalises past this case:** agreement with a single number quoted in a
reference is weak evidence when two plausible formulas both produce it. A limit the formula must
satisfy exactly (hard spheres) plus an independent derivation route (via viscosity) is strong
evidence.

### 21.2 What else was ruled out, and how

The gap was chased through five other explanations first. Each is now a guard:

| candidate | verdict | evidence |
|---|---|---|
| under-settled shock | **refuted** | one long run sliced by travel distance shows `L` flat from 20 to 177 lambda_1 |
| alignment estimator | **real, ~20 %** | aligning each snapshot on its own half-density crossing pins the ensemble to the half level at `xi=0`, putting a cusp exactly where the maximum gradient is read |
| far boundary | **real** | gas streams away from `x=L`, so an expansion returns at `u_p + a_1` — *faster* than the shock, always catching it. `setup` computes when; `run` refuses to sample after it |
| smoothing width | **real, ~10 %** | a boxcar adds width in quadrature (measured coefficient 0.96-1.10 against a predicted 1); extrapolated to zero and cross-checked against a `tanh` fit needing no smoothing |
| collision model | **refuted** | 2-D Taylor-Green measures the viscosity to **+0.38 %** of Chapman-Enskog |

The alignment fix — an equal-area position estimator plus a per-seed linear fit of `x0(t)` —
yields the shock speed as a free by-product, and Rankine-Hugoniot predicts it independently:
measured 268.6 against 269.4 m/s. That check is now a **failure guard**: at `npc = 30` the
tracking collapses (fitted speed `-34` m/s, alignment residual 46 lambda_1) and reports a
plausible-looking `L/lambda = 20.7` with no error anywhere, so `report` now refuses to present the
numbers when the fitted speed misses by more than 5 % or the residual exceeds one mean free path.

### 21.3 Resolution convergence

| cells per lambda_1 | npc | dt / mct | L max-slope | L (h->0) | L tanh |
|---|---|---|---|---|---|
| 2.49 | 120 | 0.075 | 6.21 | 6.15 | 6.41 |
| 4.98 | 120 | 0.037 | 5.98 | 5.94 | 6.36 |
| 8.00 | 120 | 0.019 | 6.04 | 5.93 | 6.07 |
| 4.00 | 480 | 0.037 | 6.15 | 6.06 | 6.22 |
| 4.00 | **30** | 0.037 | *20.72* | *18.90* | *18.09* | 

Converged in cell size and timestep across a 3.2x range, and in `npc` above ~120. The `npc = 30`
row is the tracking failure above, not a physics result.

### 21.4 A gap this closed: the collision rate at `omega != 1`

The M1 BKW gate validates the collision rate only at `omega == 1`, where `collision_rate`
collapses to `prefactor` and the `(v_r^2)^(1-omega)` factor — and hence the whole
`Gamma(5/2-omega)` normalisation — is never exercised. Every shock and transport result uses
`omega = 0.81`. `tests/test_vss.py` now asserts, for six values of `omega`, that

```
<sigma_T v_r>|_{T = T_ref} == pi d^2 <v_r>     (the hard-sphere rate)
```

which is exact: the `Gamma(5/2-omega)` in the prefactor cancels the one in the Maxwellian moment,
and at `T = T_ref` the powers combine to leave no `omega` dependence at all. A wrong Gamma
argument or a sign error in the exponent would have left M1 passing while silently scaling every
transport coefficient.


---

## 22. Statistical floors, and when a bias correction is safe

Every per-cell quantity carries a Monte-Carlo floor set by the particles in the cell: a **positive
bias** of size `1.5 v_th^2 / npc` on the fluctuation kinetic energy, whatever the flow does. Two uses
in the turbulence gates (`examples/turbulence`, `tests/test_turbulence_gates.py`) reach opposite
conclusions.

### 22.1 Subtracting it is safe on a scalar

The 2-D Taylor-Green viscosity gate fits `E(t) = E0 exp(-4 nu k^2 t)`. The floor is constant while
the signal decays, so leaving it in flattens the curve and biases the viscosity low *while the fit
still looks excellent*. `energy_history` subtracts it using actual per-cell counts. Safe because the
fit window keeps the signal well above the floor; result +0.38 % against Chapman-Enskog.

### 22.2 Subtracting it is NOT safe under a logarithm near the crossover

Applied to `E(k)` the same reasoning fails silently: above the crossover `E(k) - noise` fluctuates
through zero, about half the shells pass a positivity test by chance, and the slope is decided by
*which shells survived*. 1-D case: floor subtracted over k = 2..507 gave **-0.71** against the true
-2 (Burgers); no diagnostic fired. **Rule: a bias correction is safe where the signal is large
compared with the bias, and dangerous where a nonlinear function (here `log`) is applied near the
point at which the corrected quantity passes through zero.**

### 22.3 The right fix is to lower the floor

Raise `npc`, or **ensemble-average the velocity field over independent seeds before transforming**
(`examples.turbulence.case.run_ensemble`, which sets `info["n_ensemble"]` so `report_spectrum` lowers
the quoted floor). With deterministic ICs, averaging `M` fields equals `M * npc` at identical cost and
needs no communication between realisations.

### 22.4 It was measured, and the floor was not the binding limit

With `--ensemble 16` the floor fell exactly 16x (8.33e-1 -> 5.21e-2); low-k shells did not move
while high-k shells fell ~16x (the signature of noise), so the earlier slope -2.45 "above 3x the
floor" measured nothing. Two further findings. **The spectrum has a factor-12 odd/even comb, and it is
physical**: `u = u0 sin(kx)` is antisymmetric under `x -> x + L/2`, so odd harmonics populate and even
ones barely do (`E(7)/E(6) = 11.9`, `E(9)/E(8) = 12.8` at shells 90x and 35x above the floor); a
shell-averaged slope cannot be fitted through it. A spectral test needs a broadband IC (random phases)
or a single right-going simple wave. **Underneath the comb there is a viscous cutoff at `k ~ 10`**
(`E(11)` 6x, `E(13)` 38x below the `k^-2` continuation at `Re = 61.1`, `k_d ~ Re/6`).
`validate.dissipation_wavenumber` reports `k_d ~ C_D Re^p`, `p` = 1, 1/2, 3/4 for `D` = 1, 2, 3; only
`C_1 = 1/6` is measured, `C_2 = C_3 = 1` are placeholders, and **the 1-D constant must not be
exported** (an earlier `1/8` in all dimensions made the reach table report "NO RANGE" for every H200
configuration, contradicting `k_d/k_0 ~ Re^(3/4)`).

### 22.5 Two rules for choosing a fit window

`validate.clean_window` enforces both; each changed a headline number. (1) **Contiguous from `k = 1`**:
a threshold-selected window is decided by the data being fitted (it admitted the comb's maxima and
rejected its minima, reporting -3.06 against the contiguous window's -1.76). (2) **Above the spectral
peak**: an inertial range lies below the energy-containing scales. Common shape of every error in
22.2/22.4/22.5: **a window selected using the data being fitted will report a slope whatever the data
is**; the window must come from contiguity, the spectral peak, and `Re`. Open for 2-D/3-D: a
broadband IC is needed before a spectrum is quoted; Taylor-Green stays the default for the viscosity
gate.

### 22.6 A control must vary one thing

The first viscosity cross-check compared corrected-over-full-decay with uncorrected-over-clean-window,
varying two things, and reported a 7.9-point "correction failure" that was mostly the extra late-time
range. `report_2d` now fits three times (uncorrected clean window, corrected same window, corrected
full decay) so the first pair isolates the correction and the second the window. **When a diagnostic
compares two numbers to decide whether something is trustworthy, check that exactly one thing differs
between them.**

### 22.7 The 1-D spectrum is limited by how many modes a shell holds, and that is two

Broadband IC (`--ic broadband`) plus `Ma = 1.0` gave `Re = 204`, signal 885x the floor, contiguous
range k = 5..19, slope **-2.87 +- 0.65** (1.3 sigma from Burgers: consistent, not a test). The floor is
irrelevant; the scatter is irreducible by particle count because **in 1-D a shell holds exactly two
modes** (`k`, `-k`), so `E(k)` inherits a single realisation's scatter (`std(log E) = pi/sqrt(6) = 1.28`
expected, 1.00 measured). `--ensemble` is the wrong ensemble for a 1-D spectrum; `run_ic_ensemble` /
`--ic-ensemble R` varies `ic_phase_seed` so averaging spectra shrinks the scatter as `1/sqrt(R)`
(`test_slope_stderr_matches_the_scatter_it_claims_to_describe`,
`test_averaging_independent_realisations_shrinks_the_slope_error`). Always report the error bar
(`fit_slope`); sub-window slopes -1.69, -3.07, -2.87 are fully explained by +-0.65..1.11. A clean
Burgers range needs `Re >~ 1000` (`side ~ 5000`): minutes on an H200, ~20 h per realisation on a
laptop. **Not attempted, not claimed.**

---

## 23. The adjoint, measured

### 23.1 Parameters split into two groups, and the split decides the test

Measured on the M2 shock (`scratch/adjoint/probe_split.py`), `D = 1, 2, 3`:

| parameter | route to `J` | pathwise `dJ/dtheta` at frozen randomness |
|---|---|---|
| `temp` | reparameterised Maxwellian | nonzero |
| `vstream` | reparameterised Maxwellian | nonzero |
| `alpha` | VSS deflection law `cos X = 2 U^(1/alpha) - 1` | nonzero |
| `diam`, `omega`, `tref` | `q = sigma_T v_r` -> accept/reject **only** | **exactly 0.0** |
| `nrho` | `fnum` -> attempt count **only** | **exactly 0.0** |

The bottom half reaches `J` *only* through a frozen decision. At fixed randomness `J` is a step
function of them, so a fixed-key finite difference returns exactly `0.0` — and so does the
pathwise gradient. **They agree perfectly on a number that is not the sensitivity.** This is
§13.4's two-mode requirement made concrete: the top half is checkable by a fixed-key difference,
the bottom half only by an ensemble.

Note which group [P1] and [P2] test. [P1] is Maxwell molecules with `Sigma = q`, so `p == 1`,
nothing is rejected and **no score term exists**. [P2] adds the `eta` terms but still reports
`nabla_m J` for `m = (T_x^0, T_y^0, T_z^0)` — the initial temperature, i.e. our `temp`. **Neither
paper differentiates a collision-model constant.**

### 23.2 Pathwise gradients are exact through the production solve, all three dimensions

`python3 -m examples.shock_adjoint --mode pathwise` (and `tests/test_adjoint.py::
test_pathwise_gradient_matches_flip_free_finite_difference`): 7-point sweeps over +-15 % of Bird's
Mach-2 argon reference, `h` re-bisected at every point until the decision checksum matches.
Maximum relative error over the sweep, all nine PASS at `h0 = 1e-5`:

| D | `N*C*M` | `temp` | `vstream` | `alpha` | `alpha`, `err/medAD` |
|---|---|---|---|---|---|
| 1 | 1.3e6 | 9.3e-10 | 8.1e-11 | 1.7e-7 | 6.7e-8 |
| 2 | 5.1e6 | 4.4e-10 | 3.0e-10 | 2.0e-6 | 3.2e-7 |
| 3 | 2.0e7 | 3.1e-10 | 3.0e-10 | 4.8e-7 | 4.8e-7 |

**Lessons that survive the numbers.** (a) The first `alpha` figures were blamed on a cancellation
floor; the `|FD - AD|/FD-noise` column showed that wrong (alpha sat at 40-1853x its floor). Sweeping
`h` separated the mechanisms: a clean `O(h^2)` truncation above `h ~ 1e-5` (the bisection had stopped
at the harness cap, not at `h_crit`), so the cap is now `1e-5`. "Largest flip-free `h`" is the wrong
rule when the window is wide: the step must sit between truncation above and cancellation below, and
only a sweep tells you which. (b) The residual `alpha` error correlates with gradient magnitude
(Spearman -0.835), not `h`: `alpha`'s sweep crosses near-stationary points of `J`, where a per-point
relative error divides by a small number. `verify.print_pathwise` therefore also reports
`err/medAD` (normalised by the sweep's median `|AD|`, `verify.scaled_error`); PASS on any of three
criteria. The remaining ~100x gap to `temp`/`vstream` is higher-derivative content: a central
difference's truncation is `(h^2/6)|J'''|` and `alpha` enters through `u^(1/alpha)` (all orders in
`1/alpha`) while `vstream` is linear. **A per-point relative error is the wrong summary for a sweep
that crosses a stationary point, and a fix that works in one dimension of three is evidence the
diagnosis is incomplete.**

**Caveat on U8.** Under `h0 = 1e-5` the top of every `h*` range *is the cap*, so this run no longer
measures the flip-free window and cannot support `h_crit ~ 5.8/(N*C*M)`; the supporting evidence is
the superseded `h0 = 1e-4` run at `D = 2, 3` only (largest flip-free steps `1.3e-5`, `6.3e-6` against
the fit's `1.1e-6`, `2.9e-7`: the fit is conservative). U8 stays open.

### 23.3 The score estimator's variance, in the papers' own terms

> Consolidated tables and figures: `examples/shock_adjoint/VARIANCE.md`, regenerated (no GPU) by
> `python3 -m examples.shock_adjoint.variance_report`. These measurements describe the **reference
> score estimator** (section 11), not the production ensemble adjoint.

**[P2] section 6 reports and budgets for this variance** (`main.tex:1425`: `M_s = 100` i.i.d. runs for
both estimators; `:1431`: random error `O((sqrt(N M_s)|Delta m|)^-1)`, `Delta m = 0.1` (a 10 % finite
difference) "nearly optimal" for `N = 1e6`, `M_s = 100`; `:1502`: both sds grow as `O(sqrt(M))`, and AD
about 10x less noisy than FD). A large step plus a large ensemble is the published protocol, not a
workaround; "mode B" is that protocol.

#### 23.3.1 The `N^-1/2` law holds, measured over a 64x range in `N`

`D = 2`, `npc` (hence `A_max`) fixed at 256, `M = 16`, 128 keys, `phi = T_y` ([P2]'s own objective;
23.3.7 says why the windowed one cannot be used for a variance study). Fitted `sd ~ N^p`:
`temp` -0.508, `ux` -0.464, `alpha` -0.495, `diam` -0.499, for a pathwise, a mixed, and a pure-score
parameter alike. **The variance is a budget question, not a defect** (which is what U2 asks).

#### 23.3.2 The percentage error bars were misleading, and the reason matters

Large relative standard errors (`alpha` 390 %, `nrho` 136 %) were read as the estimator struggling.
The `N` sweep exposes it: `diam`'s mean is stable to ~10 % across a 64x change in `N` while its SE
shrinks 8x (relative error 9.7 -> 1.2 %), whereas `alpha`'s mean is consistent with zero at every `N`
(`|z|` 0.9, 1.3, 0.4, 0.4), so its relative error is a number divided by ~0. Its *absolute* error bar
falls 8x on the same law. `E[dJ/d(alpha)]` is simply small against a single run's spread
(`|mean| <~ 30` vs `sd = 244`); do not confuse it with 23.2's frozen-randomness check. The old
512-key probe had `N*M_s = 1e6` against [P2]'s `1e8`, a hundredfold smaller budget. Not a correctness
question: Oracle 2 (13.2) matches Algorithm 3 particle by particle.

#### 23.3.3 The noise is mostly **pathwise**, which is why the baseline barely helps

Splitting full = pathwise + score (256 keys): correlations within +-0.2 of zero, so variances add;
for every parameter with a pathwise route that route carries 3-5x the score's noise (`alpha` sd 1014
pathwise vs 197 score). The score's share of *variance* (what a baseline removes) is only 11.9 %
(`temp`), 5.5 % (`ux`), 3.1 % (`alpha`): ceilings of 1.065x, 1.029x, 1.016x for a perfect baseline;
only score-only parameters (`diam`, `omega`, `tref`, `nrho`) are in the regime a baseline or larger
`M_s` improves. (This does not *explain* 11.4's measured ~1.2x, which exceeds the 1.016x ceiling:
that was a windowed `phi` at `D = 1, N = 2000`. Quoting one as the explanation of the other would be a
false tidy-up.) For `temp`, `ux`, `alpha` the only lever on the dominant noise is more particles.

#### 23.3.4 At [P2]'s own protocol, our AD/FD gap is truncation-limited, not noise-limited

`N = 256 000`, `M_s = 100`, `Delta m = 0.1` (`N*M_s = 2.6e7`): `temp` AD 212.47 vs FD 216.88
(|z| 11.0, 2.1 %), `ux` AD -36.2 vs FD -48.3 (|z| 43.5, 33 %; a 10 % step moves Mach 2.0 -> 2.2 and
the shock is strongly nonlinear), `diam` 14 %, `alpha` |z| 0.88. These `z` values are `Delta m = 0.1`
being a 10 % step with `O(h^2)` truncation, visible only because our AD SE (0.03-0.25 %) is below
[P2]'s noise level. **Mode A (23.2) is the test that resolves AD against FD**; mode B's step is
chosen to make decisions flip and is not a precision comparison. [P2]'s "AD ~10x less noisy"
partially reproduces: SE ratio 5.4x (`temp`), 2.9x (`ux`), 1.5x (`diam`); for `alpha` FD is 1.5x
tighter.

#### 23.3.5 The spatial window costs signal, not variance

Windowed vs `T_y` objective (`N = 64 000`, 128 keys): the window reduces the absolute sd in every case
but reduces the mean faster; `diam`'s spread falls 2.4x while its signal falls 15x (net 6.3x loss of
SNR); `temp` and `ux` lose 1.7x and 2.6x. A localised objective costs sample budget in proportion to
the fraction of gas it looks at; a design note, not a defect.

#### 23.3.6 `sd ~ sqrt(M)` does not hold here, and the raw exponent is the wrong thing to compare

[P2]'s `M`-sweep is a 0-D relaxation, where the gradient saturates; ours is a developing shock where
`dJ/dm` itself grows. Fitting both `sd` and `|mean|` over `M = 4..32` (`X ~ M^q`): `diam` signal and
noise grow together (`q` 0.720 vs 0.721, `q(sd/|mean|) = -0.002`: relative precision flat in run
length); `temp` is the only parameter in [P2]'s regime (saturated signal `q = 0.017`) and its raw sd
exponent is 0.291, *below* [P2]'s 0.5; for `ux` the relative spread falls as `M^-0.275`. `alpha`'s row
must not be read (its mean flips sign). Four points over an 8x range: indicative, not tight.

#### 23.3.7 A zero-check with an analytically known answer

`sum |v|^2` is a collision invariant (VSS conserves pair energy; a specular wall flips a sign), so the
total kinetic energy cannot depend on `alpha`, `diam`, `omega`, `tref`, pathwise or in expectation.
Measured (`D = 2`, `N = 16 000`, 32 keys): pathwise max `1.0e-12` / `0.0` / `0.0`; scored means
consistent with exactly zero (`|z|` 0.25, 1.53, 0.99). A genuine unbiasedness check against a ground
truth, though only at a zero. It cannot measure variance (the signal is identically zero). Kept
permanently by `tests/test_adjoint.py::
test_collision_invariant_has_exactly_zero_collision_parameter_gradient`.

#### 23.3.8 `alpha` is ill-conditioned *for argon specifically*, and the shock only partly recovers

VSS gives `<cos^2 chi> = 4a/(a+2) - 4a/(a+1) + 1`, whose derivative `8/(a+2)^2 - 4/(a+1)^2` vanishes
at exactly `a = sqrt(2)` (with `<cos^2 chi> = 1/3` at both `a = 1` and `a = 2`). Bird's argon sits at
`alpha = 1.40`, 1.005 % from that stationary point, so a temperature-like objective has almost no
`alpha` sensitivity there. `jax_dsmc/species_data.py` carries five monatomic gases and
`alpha_conditioning` (ratio to argon). 0-D confirms across all five (`examples/p2_replication`, one
constant fits at `r = 0.9732`). In the 2-D reflected shock (256 keys, `N = 16 000`, `phi = T_y`): He
(`alpha = 1.26`, 12.6x analytic conditioning) `AD = 860 +- 318` (|AD|/SE 2.7), `|FD|/SE` 0.57,
`SE_FD/SE_AD = 8.4x`; Ar |AD|/SE 0.8. So: the gain is real but heavily discounted (3.4x realised vs
12.6x; 10 sigma would need ~14x the ensemble); **the finite difference cannot do this at all**, so a
helium AD-vs-FD agreement is a low-power check and must not be quoted as verification; and the 8.4x
is the first sighting of [P2]'s ~10x AD advantage, where the pathwise term dominates. `alpha` stays a
mode-A parameter (passes at 2e-6 regardless of gas); any mode-B `alpha` claim must use helium and state
its ensemble size. Figures: `examples/shock_adjoint/figures/gas_{resolvability,shock_alpha}.png`.

### 23.4 Reverse-pass cost: ~4x the forward, not 17x

> Pinned by `tests/test_perf_budget.py`: reverse/forward FLOPs within 1.2-4.0x
> (`test_reverse_pass_costs_about_twice_the_forward_in_flops`) and temp memory under 15x
> (`test_reverse_pass_memory_multiple_is_bounded`). Wall-time ratios below are the 2026-09
> laptop-GPU measurements.

[P1] Table 8 reports adjoint/forward of `0.22/0.38 = 0.58` at `N = 1e6`, rising to `30/60 = 0.50`
at `N = 1e8` — 0-D, CPU. A first measurement here gave **17x**, which prompted the check.

**The 17x was a benchmarking artifact.** It was measured under `vmap` over 128 base keys, where
the reverse pass becomes memory-bound: a 512-key batch asks XLA for **4.0 GiB** and
rematerialisation cannot get it below 2.85 GiB, which is an out-of-memory on a 4 GB card.
Un-batched (`scratch/adjoint/probe_cost.py`, one key, `D = 1`, `A_max` calibrated):

| `N` | `C` | `A_max` | forward | forward+reverse | ratio |
|---|---|---|---|---|---|
| 160 | 8 | 116 | 32.6 ms | 106.5 ms | **3.26** |
| 2 000 | 40 | 64 | 59.0 ms | 264.8 ms | **4.49** |
| 8 000 | 80 | 64 | 60.9 ms | 235.7 ms | **3.87** |
| 32 000 | 160 | 64 | 89.3 ms | 369.7 ms | **4.14** |
| 64 000 | 160 | 153 | 293.4 ms | 1284.4 ms | **4.38** |

So reverse-mode sits in the textbook 2-4x band and is **flat in `N`**. Note separately that the
forward time barely moves from `N = 160` to `N = 8000` (32.6 -> 60.9 ms): at these sizes the step
is **launch-bound**, `A_max * M` sequential scan iterations of tiny kernels, and the GPU is idle.
That is a forward-solve property, not an adjoint one, and it is the same effect §8.6 measured.

**The comparison to [P1] is not like-for-like, in two ways.** Their `0.58` *excludes the forward
solve*, which must run first to record the collisions — so their gradient costs `1 + 0.58 = 1.58`
forwards against our `~4`. And their backward replays recorded collisions through a stored `6x6`
matrix: no sort, no moments, no rejected candidates, no trigonometry. General AD must transpose
everything the forward did, including work that contributes nothing to the gradient. A factor of
~2.5 against a hand-derived, problem-specific backward is the price of not hand-deriving it.

**Two mechanisms were hypothesised and both are refuted.**

*The attempt-slot `scan` retaining its `(N, 3)` velocity carry at every slot* — if that dominated,
the ratio would grow linearly in `A_max` at fixed `N`. It does not; it **plateaus**:

| `A_max` (at `N = 8000`) | 64 | 128 | 256 | 512 |
|---|---|---|---|---|
| forward | 38.9 ms | 61.5 ms | 105.6 ms | 210.2 ms |
| ratio | 3.09 | 3.89 | 4.79 | **5.00** |

The rise from 3.1 to 5.0 is the *forward* escaping the launch-bound regime (8x more slots costs
only 5.4x more time), not the reverse blowing up. Once both are compute-bound the ratio settles
near **5**.

*The `nothing_saveable` remat policy* — no lever at all:

| policy | `nothing_saveable` | `everything_saveable` | `checkpoint_dots` | `..._no_batch_dims` |
|---|---|---|---|---|
| ratio | 4.48 | 4.23 | 4.28 | 3.94 |

**Where the cost actually is.** Per step at `N = 8000`, `D = 1`:

| stage | move | sort | moments | **collide** |
|---|---|---|---|---|
| forward | 0.19 ms | 1.66 ms | 1.56 ms | **30.40 ms** |

`collide` is **90%** of the forward, so the reverse pass is essentially the VJP of the
attempt-slot scan. The "AD must also transpose the sort and the moments" story is real but worth
only ~10%; the honest statement is that differentiating the collision scan costs ~4x running it.

**The lever, if this ever matters.** The bulk of that is `scatter`'s VJP — a nonlinear map of six
inputs with trigonometry, differentiated slot by slot. [P1]'s adjoint is cheap precisely because
it replaces that with a stored `6x6` matrix. A `jax.custom_vjp` on `scatter` exploiting the
structure (`|v_r'| == |v_r|`, mass-weighted CM decomposition) is the contained change that would
close most of the remaining gap, and it would be checked by the Oracle-2 test that already
exists. **Not done, and not yet needed** — at `~5x` a gradient is affordable.

#### 23.4.1 Memory, not time, is what caps an adjoint run

The reverse pass's *peak memory* is the binding constraint at scale, and it is much tighter than
the forward's. Measured from two independent XLA allocation failures at `D = 2`, `A_max = 256`,
`M = 16`:

| requested | batch | `N` | bytes per (particle x key) |
|---|---|---|---|
| 8.582 GiB | 125 keys | 16 000 | **4291** |
| 4.270 GiB | 62 keys | 16 000 | **4304** |

Two points agreeing to 0.3 % — call it **~4300 bytes per particle** for one gradient, at this
`A_max`. It should scale roughly linearly in `A_max` (the attempt-slot scan's residuals) and be
roughly flat in `M` (the time loop is rematerialised).

Consequences:

| device | usable memory | max `N` for **one** gradient |
|---|---|---|
| RTX 3050 Ti (development) | 3.2 GB | **7.4e5** |
| one H200 | 141 GB | **3.3e7** |
| 4x H200 node | 564 GB | **1.3e8** |

So the adjoint caps a run at `~1e8` particles on the node §20.4 sizes for `~1e9` forward-only.
**A gradient costs roughly 30x the memory per particle that a forward solve does**, and that —
not the ~5x in time — is what will decide how large an adjoint-driven optimisation can be.

It is also why `N = 1e6` could not be reached in the §23.3 variance study on the development
card: 4.3 GB for a single key. The study runs at the largest `N` that fits and uses the fitted
`N^-1/2` law for the rest, which is stated where it is used rather than hidden.
