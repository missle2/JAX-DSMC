# JAX-DSMC — Pseudocode Specification (Phase 0)

Status: **specification only, revision 3 — not yet approved.** No JAX code is written until this
document is reviewed.

- **Revision 2** incorporated the dsmcFoam+ review; section 18 was new.
- **Revision 3** incorporates an external review plus three verification passes that measured
  every load-bearing claim. It fixes **two defects that would have produced silently wrong
  results**: §8.1 (the `dt`/`fnum`/`cell_volume` gradients were exactly zero) and §8.4.1 (empty
  cells clobbered real collisions, breaking momentum conservation in 59% of runs). It also
  reverses two conclusions that were stated confidently and measured false — see §7.2.2 and
  §13.4.1. Changed: §0.2, §0.6 (new), §2.3-2.4, §3.3, §4.1, §7.2-7.4, §8.1-8.5, §11.4-11.6,
  §12.1, §13.3-13.4, §16, §17, §18.

Environment: `source ~/activate_jax_env.sh` (`~/envs/jax-env`) — JAX 0.6.2 on `CudaDevice(id=0)`
with `jax_enable_x64`, NumPy 2.2.6, SciPy 1.15.3, h5py 3.16.0, matplotlib 3.10.9, pytest 9.1.1.
The system Python is not usable: its jaxlib fails to load cuFFT and silently falls back to CPU.

This is the complete system in Python-like pseudocode. It is deliberately JAX-agnostic in the
algorithm bodies, but it is *not* implementation-agnostic in the contracts: the shape, dtype,
traced/static, and RNG contracts below are exactly what makes `jax.grad` reproduce the adjoint
DSMC algorithms of the reference papers. Those contracts are the load-bearing part of this
document. Violating any one of them silently produces wrong gradients rather than an error.

Reference material:
- SPARTA source, `/home/aakash/sparta/sparta/src/` — cited as `file.cpp:lines`.
- dsmcFoam+ / hyStrath, `/home/aakash/OpenFOAM-v1706/dsmcFoam+_internals.html` — cited as
  **[DF]**. The two production codes disagree in several places; section 18 collects the
  divergences and records which we follow and why.
- `papers_reference/arXiv-2009.01363v3` — Caflisch, Silantyev, Yang. *Adjoint DSMC for nonlinear
  Boltzmann equation constrained optimization.* Cited as **[P1]**.
- `papers_reference/arXiv-2207.11579v2` — Yang, Silantyev, Caflisch. *Adjoint DSMC for nonlinear
  spatially-homogeneous Boltzmann equation with a general collision model.* Cited as **[P2]**.
- `papers_reference/arXiv-2603.20946v2` — Caflisch, Chen, Yang. *Adjoint DSMC method for spatially
  inhomogeneous Boltzmann equation with general boundary conditions.* Cited as **[P3]**.

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
| `C` | total cells, `prod(n)` (static) | |
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
| cell membership of a particle | `grid.cell_of` | int |
| the sort permutation | `sort.build_table` | int |
| number of collision attempts per cell | `ntc.attempt_count` | int |
| which pair is drawn for an attempt | `ntc.pair_ntc` / `ntc.pair_perm` | int |
| accept/reject of a candidate collision | `vss.test_collision` | bool |
| specular-vs-diffuse at a wall (M4+) | `boundary.thermal` | bool |
| which face a particle crosses | `boundary.apply` | int |

**Freezing these is not an approximation we are choosing** — it is what [P1] specifies
("the choice of collision times, collision partners, and collision angles is not included in the
variational principle... determined externally"). What it *costs* is accounted for separately by
the score terms of [P2] (§7 below), which recover the sensitivity of the *probability* of each
decision without differentiating the decision itself.

### 0.6 Code-quality conventions

Enforced by `ruff` (config in `pyproject.toml`, verified to stay quiet on JAX idioms while
catching real defects). `ruff check .` and `ruff format --check .` must both pass.

**No nested function definitions.** JAX pushes you toward them — `lax.scan` bodies want to close
over loop-invariants — but a closure defined inside another function cannot be unit-tested,
cannot be read in isolation, and hides how much state it captures. Bind loop-invariants with
`functools.partial` against a module-level function instead:

```python
# NO -- nested def, captures an unclear amount of enclosing scope
def collide(state, table, cstate, ...):
    def slot(carry, a):
        ...                                   # what does this close over?
    return lax.scan(slot, init, arange(A_max))

# YES -- flat, explicit, independently testable
def _collide_slot(sigma, table, species, keys, carry, a):
    ...                                       # every dependency is an argument

def collide(state, table, cstate, ...):
    body = partial(_collide_slot, sigma, table, species, keys)
    return lax.scan(body, init, arange(A_max))
```

The pseudocode in sections 8.4 and 12 is written in the nested style for readability *as
prose*; the implementation must use the flat form. Private helpers take a leading underscore.

Other conventions:
- One physical concept per function. `collide` orchestrates; `_collide_slot` does one attempt;
  `collision_rate`, `acceptance_prob`, `scatter`, `log_h` are each independently testable.
- No mutable default arguments, no bare `except`, no `== True` (ruff catches all three).
- Every public function gets a docstring stating shapes, which arguments are traced vs. static,
  and its SPARTA / [DF] / paper citation.
- Physics symbols keep their paper names (`N`, `D`, `A_max`, `Sigma`, `I`); ruff's naming rules
  are disabled for exactly this reason, and it is not a licence for one-letter locals generally.

---

## 1. The RNG contract

**This is the single most important contract in the system.** Get it wrong and gradients are
silently incorrect, finite-difference checks become pure noise, and `jax.checkpoint` produces a
different trajectory on the backward pass than the forward pass took.

```
STAGE_INIT      = 0      # particle initialisation
STAGE_MOVE      = 1      # advection sub-steps / randomised tau (M7)
STAGE_WALL      = 2      # diffuse re-emission draws (M4+)
STAGE_NTC_ROUND = 3      # stochastic rounding of the attempt count
STAGE_PAIR      = 4      # collision partner selection
STAGE_ACCEPT    = 5      # NTC accept/reject uniform
STAGE_SCATTER   = 6      # VSS deflection angle and azimuth
STAGE_INFLOW    = 7      # inflow flux sampling (M5+)

def key_for(base_key, step, stage_id):
    # base_key is a fixed PRNGKey for the whole run.
    # step is the lax.scan loop index (an integer, not a traced float).
    # stage_id is a module-level Python int constant.
    return fold_in(fold_in(base_key, step), stage_id)
```

Rules, all of which are testable:

1. **Keys depend only on `base_key`, the loop index, and a static stage id.** Never on a traced
   state value, never on a particle's position or velocity. If a key depended on state, the
   gradient would try to flow through the RNG, which is meaningless.
2. **Keys are never carried in the state.** They are recomputed from `(base_key, step, stage)`
   every time. This is what makes rematerialisation under `jax.checkpoint` reproduce the *same*
   random draws on the backward pass. Carrying and splitting a key would also work in principle
   but makes the invariant much harder to verify.
3. **Each stage draws all its randomness in one batched call**, e.g.
   `uniform(key_for(bk, k, STAGE_ACCEPT), shape=(C, A))`, not per-element splits. Cheaper, and
   the shape is manifestly static.
4. **Common random numbers.** Two runs with different parameters but the same `base_key` consume
   an identical random stream. This is what makes the finite-difference gradient check
   meaningful: `(J(theta+h) - J(theta-h)) / 2h` estimates the pathwise derivative only if both
   evaluations use the same randomness. Without this the FD estimate has O(1/sqrt(N)) noise that
   swamps the signal entirely.

Within a stage that needs several independent streams (e.g. the attempt-slot scan needs a fresh
uniform per slot `a`), fold in the slot index as well:
`key_for(base_key, step, STAGE_ACCEPT)` then `fold_in(that, a)`. The slot index `a` is a `scan`
loop index, so this preserves rule 1.

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
> error that points nowhere near the cause. It cost a real debugging session. No module in this
> package may take the name of a stdlib module.

All containers are registered pytrees so `jax.grad` can traverse them. Static fields are
declared with `static=True` metadata and are compared by value for cache lookup.

### 3.1 Shape and dtype contract

```
@pytree_dataclass
class ParticleState:
    x        : f64[N, D]     # positions
    v        : f64[N, 3]     # velocities, always 3 components
    species  : i32[N]        # species index, all zeros while S == 1
    alive    : bool[N]       # occupancy mask; dead slots are reusable
    logh_acc : f64[N]        # score accumulator L_i, see section 11

@pytree_dataclass
class CollideState:
    vremax : f64[C, G, G]    # NTC majorant Sigma, per cell per group pair
                             # read-only within a step; see 7.2 and 8.4
                             # no `remain` field: 8.1 uses stochastic rounding,
                             # which needs no carry and yields a score term

@pytree_dataclass
class CellTable:             # built once per step by sort.build_table (section 6)
    order  : i32[N]          # particles sorted by cell
    starts : i32[C]          # exclusive prefix sum of counts
    counts : i32[C]
    cell   : i32[N]          # per-particle cell index; needed by the count score (8.1)

@pytree_dataclass
class Diagnostics:           # per-step, host-side; never branched on inside jit
    attempts_clipped    : i32     # n > A_max: silent under-collision (8.2)
    max_attempts        : i32
    majorant_violations : i32     # q > Sigma: zero score, BIASED gradient (7.2.1)
    min_majorant_margin : f64     # min(Sigma/q); target >= 2 (7.2.2)

@pytree_dataclass
class GridSpec:
    lo : f64[D]              # TRACED  - domain lower corner
    hi : f64[D]              # TRACED  - domain upper corner
    n  : tuple[int, D]       # STATIC  - cells per axis; C = prod(n)

@pytree_dataclass
class SpeciesParams:
    mass      : f64[S]
    diam      : f64[S, S]    # VSS reference diameter
    omega     : f64[S, S]    # viscosity exponent
    tref      : f64[S, S]    # reference temperature
    alpha     : f64[S, S]    # VSS angular parameter; alpha == 1 -> VHS
    mr        : f64[S, S]    # reduced mass, derived
    prefactor : f64[S, S]    # derived, see 6.1

@pytree_dataclass
class FlowParams:
    nrho    : f64            # number density
    temp    : f64            # translational temperature
    vstream : f64[3]         # bulk velocity

@dataclass(frozen=True)
class SimConfig:
    # TRACED
    dt      : f64
    fnum    : f64[C]         # per-cell, not scalar -- see below
    epsilon : f64            # randomised-timestep width; 0.0 until M7
    # STATIC
    N       : int
    D       : int
    S       : int
    G       : int
    A_max   : int
    K_bounce: int
    n_steps : int
    remat_policy : str
```

Note `SimConfig` mixes traced and static; in the implementation it is split into
`SimParams` (pytree, traced) and `SimStatic` (frozen, hashable, passed as `static_argnums`).
Written together here only for readability.

### 3.2 Why each static field is static

| Field | Reason it cannot be traced |
|---|---|
| `N` | shape of every particle array |
| `D` | shape of `x`, and the `v[:, :D]` slice |
| `S`, `G` | shape of parameter tables and `vremax` |
| `n` | `C = prod(n)`, shape of all per-cell arrays |
| `A_max` | trip count of the attempt-slot `scan` |
| `K_bounce` | trip count of the wall-bounce `scan` |
| `n_steps` | trip count of the time `scan` |

### 3.3 `fnum` as a per-cell field, and the RWF question

[DF] routes every use of the particle weight through `nParticles(cell) = F_N * RWF(cell)`
(`dsmcAxisymmetricI.H:55-58`); SPARTA carries `cinfo.weight`. Radial weighting (RWF) separates
into two independent halves, and they have very different retrofit costs.

**Half 1 — the weight field. Adopted, traced.** `fnum : f64[C]`. Read by the NTC attempt count
(8.1), moment normalisation (10), and boundary-flux measurement. This is the expensive half to
retrofit: it touches every one of those call sites, so doing it later would mean a diff across
`ntc.py`, `sample.py`, and `boundary.py` simultaneously. Doing it now costs a `(C,)` broadcast
where a scalar stood. It is a traced leaf, so `dJ/d(fnum_c)` exists.

**Half 2 — the clone/delete machinery. Not adopted.** When a parcel moves to a cell with a
different RWF, [DF] clones it with probability `RWF_old/RWF_new - 1` moving inward, or keeps it
with probability `RWF_old/RWF_new` moving outward (`dsmcAxisymmetric.C:50-219`).

The case for deferring it is stronger than "it breaks static shapes", which was the earlier and
somewhat lazy justification. Birth and death of particles is something the static-capacity +
`alive` mask design already handles — it is how outflow works, and how M5 inflow will work. The
real reasons are:

1. **It is not on the critical path.** RWF exists to fix uneven sampling in axisymmetric and
   spherical meshes, where cell volume scales with `r` or `r^2`. Every milestone through M5 is
   Cartesian, and [DF] itself makes `dsmcCartesian::evolve()` an empty no-op with `RWF == 1`
   everywhere. We would be building and testing machinery that never executes.
2. **Its adjoint is M8's problem, not a new one.** Cloning splits a particle's future in two and
   deletion discards a cotangent, so the pathwise adjoint's particle correspondence breaks. This
   is exactly the count-changing sensitivity [P3] §5 addresses with the *ensemble* adjoint. So
   the hard part of half 2 rides on infrastructure already committed to for inflow — retrofitting
   it is cheap in a way half 1 would not have been.

> **Loaded-gun warning.** Shipping the field without the machinery creates a failure mode that
> did not exist before. If `fnum` is spatially non-uniform and a particle crosses from a cell
> with `fnum_a` to one with `fnum_b`, it now represents the wrong number of real molecules, and
> mass is silently not conserved. Nothing in the code notices.
>
> **Invariant, asserted at setup:** in any run where particles cross cells, `fnum` must be
> spatially uniform. Non-uniform `fnum` is legal only in a 0-D single-cell run, or once
> reweighting (clone/delete, or a continuous weight-transfer scheme) exists. The check is one
> line — `assert allclose(fnum, fnum[0]) or C == 1 or has_reweighting` — and it must be present
> from the first commit, because the bug it prevents is invisible in every diagnostic we plan to
> compute.

Everything else is traced, including `dt`, `fnum`, `lo`, `hi`, and every entry of
`SpeciesParams` and `FlowParams`. This is what makes the plan's "gradients flow to everything"
requirement literally true.

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

### 4.2 Tests (`tests/test_grid.py`), parametrised over `D in {1, 2, 3}`

- round trip: `cell_centers(grid)[c]` maps back to `c` under `cell_of`, for all `c`;
- a particle exactly at `lo` lands in cell 0; exactly at `hi` lands in cell `C-1` (the `clip`
  handles the upper-edge case, matching SPARTA's behaviour of never losing a boundary particle);
- every corner of every cell maps to a cell index in `[0, C)`;
- `sum(cell_volume(grid)) == prod(hi - lo)` to machine precision;
- `grad(lambda lo: sum(cell_volume(GridSpec(lo, hi, n))))` is finite and matches the analytic
  derivative of the box volume.

---

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

### 5.4 Flux-weighted inflow (deferred to M5)

`fix_emit.cpp:166-174`, Bird eq. 4.22. Recorded here so the interface is stable:

**Built** — `jax_dsmc/inflow.py`. Bird's bounded rejection scheme as a `lax.scan` of a static
`N_TRIALS = 16` with an acceptance mask, never a `while_loop`. One improvement over SPARTA:
its inner `do beta_un = 6U-3 while (beta_un + scosine < 0)` (`fix_emit_face.cpp:576-577`) is
itself a rejection loop; sampling the truncated range `[max(-3,-s), 3]` directly is the
identical distribution with the nested loop removed. Validated against numerical quadrature of
`(beta + s) exp(-beta^2)` rather than a stored baseline — mean and sd to 1-2 % across
`s in [-1, 5]`, acceptance `>= 99.6 %` within the 16 trials.

```
def mol_inflow(s):                 # s = vstream_normal / vscale
    return (exp(-s*s) + sqrt(pi)*s*(1 + erf(s))) / (2*sqrt(pi))

def sample_inflow_velocity(key, temp, mass, vstream, normal):
    # rejection sampling with acceptance prob ~ (beta + s) exp(-beta^2)
    # fix_emit_face.cpp:576-583
    # NOTE: this is a rejection LOOP. It must be implemented as a bounded
    # scan with a static trip count and a mask, not a while_loop, or the
    # M8 inflow work cannot differentiate it. See section 12.
    ...
```

### 5.5 `jax_dsmc/initialize.py` — assembling the initial state

```
def create_particles(key, cfg, grid, flow, species):
    # np_target = nrho * flow_volume / fnum          create_particles.cpp:588
    vol       = sum(cell_volume(grid))
    np_target = flow.nrho * sum(cell_volume(grid) / cfg.fnum)   # per-cell weight
    n_real    = stochastic_round(np_target, uniform(key_for(key, 0, STAGE_INIT)))
    # n_real is data-dependent -> cannot size an array with it.
    # Instead: allocate the static capacity N and set alive = (arange(N) < n_real).
    x = sample_positions(k_x, cfg.N, grid)
    v = sample_maxwellian(k_v, cfg.N, flow.temp, species.mass[0], flow.vstream)
    return ParticleState(x=x, v=v, species=zeros(N, i32),
                         alive=(arange(N) < n_real),
                         logh_acc=zeros(N))
```

The static-capacity-plus-mask pattern is used everywhere a count is data-dependent. It is the
single most pervasive structural difference from SPARTA, which grows arrays freely.

### 5.6 Tests (`tests/test_sampling.py`)

- moments: `mean(v) -> vstream` and `var(v) -> k T / m` per component, within
  `4 * sqrt(var / n)` of the analytic value, seeded so the test is deterministic;
- KS test of `|v - vstream|` against the Maxwell speed distribution;
- `grad(lambda T: mean(sum(v**2)))` matches the analytic `3k/m` to 1e-10 (this is a pure
  reparameterisation check and should be near-exact, not statistical);
- `stochastic_round` is unbiased: `mean(stochastic_round(x, U)) -> x` over many draws;
- with a fixed key, changing `temp` moves every velocity smoothly (no reshuffling) — this is the
  common-random-numbers property the FD gradient checks depend on.

---

## 6. `jax_dsmc/sort.py` — cell occupancy table

SPARTA builds per-cell linked lists (`collide.cpp:508-512`). That is a CPU-appropriate structure;
on GPU the right primitive is a sort.

```
def build_table(state, grid):
    c = cell_of(state.x, grid)                        # i32[N]
    c = where(state.alive, c, C)                      # dead -> sentinel cell C
    order  = argsort(c)                               # i32[N]   FROZEN (int)
    counts = bincount(c, length=C + 1)                # i32[C+1]
    starts = cumsum(counts) - counts                  # i32[C+1] exclusive scan
    return CellTable(order=order, starts=starts[:C], counts=counts[:C], cell=c)
    # `cell` (per-particle cell index) is computed anyway and is needed by the
    # count score of 8.1, which is a PER-CELL term applied to every particle in
    # that cell. Returning it avoids recomputing cell_of in collide.
```

Particle `j` of cell `k` is `order[starts[k] + j]`, valid for `j < counts[k]`.

### 6.1 Why this is differentiable

`order` is an integer permutation — a frozen decision. But the *gather* `v[order]` is a linear
operation whose VJP is a scatter, so cotangents flow back to the correct particles automatically.
Nothing special is needed. The permutation being frozen is correct and matches the papers.

### 6.2 Scaling note

`argsort` is `O(N log N)`. At `N ~ 1e9` this becomes the bottleneck and should be replaced by a
counting sort (the cell index is already a bounded integer, so a `bincount` + prefix-sum +
scatter does it in `O(N)`). Deferred; the interface does not change.

### 6.3 Tests (`tests/test_sort.py`)

- every alive particle appears exactly once in `order[:sum(counts)]`;
- `counts` matches a direct `bincount` of `cell_of`;
- dead particles all sort past `sum(counts)`;
- for every cell `k` and every `j < counts[k]`, `cell_of(x[order[starts[k]+j]]) == k`;
- gather/scatter round trip: `zeros().at[order].set(v[order]) == v` for alive particles.

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
K_EV  = 5.0     # extreme-value multiplier: P(|vr| > K_EV * v_ref) * npc^2/2 ~ 1.6e-4
F_UP  = 2.0     # safety factor; the 2-5x band of 7.2.1

def thermal_majorant(temp, omega, sp, cfg):
    """A-priori upper bound on q from the cell temperature.  f64[C].  STOP-GRADIENT."""
    vref2 = 2.0 * K_BOLTZ * temp / sp.mr            # f64[C]  reduced-mass reference speed^2
    hi    = sp.prefactor * safe_pow(K_EV * K_EV * vref2, 1.0 - omega, EPS)   # omega < 1: top end
    lo    = sp.prefactor * safe_pow(eps_abs(sp), 1.0 - omega, EPS)           # omega > 1: the clamp
    return F_UP * maximum(hi, lo)

def vremax_init(temp0, omega, sp, cfg):
    """Seed for `CollideState.vremax`.  NOT SPARTA's `2 * cxs * vscale` -- that seed
    fails the U3 gate (17.5)."""
    return broadcast(thermal_majorant(temp0, omega, sp, cfg), (C, G, G))

# per step, before the attempt scan (section 8.4):
Sigma = stop_gradient(maximum(cstate.vremax[:, 0, 0],
                              thermal_majorant(cell_temp, omega, sp, cfg)))
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
def scatter(vi, vj, isp, jsp, species, key):
    mi, mj = species.mass[isp], species.mass[jsp]
    mr     = species.mr[isp, jsp]
    a_inv  = 1.0 / species.alpha[isp, jsp]

    u_chi, u_eps = split_uniforms(key, 2)
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

### 7.6 Tests (`tests/test_vss.py`)

- momentum and energy conserved to 1e-14 relative over 1e5 random pairs, swept over
  `alpha in [1, 2]`, `omega in [0.5, 1.0]`, and mass ratios up to 100;
- `cosX` empirical distribution matches `2U^(1/alpha) - 1` by KS test;
- `alpha == 1` gives isotropic scattering: `mean(cosX) -> 0`, `var(cosX) -> 1/3`;
- degenerate inputs — `vi == vj`, `v_r` exactly along x, `v_r` exactly along y — give finite
  values *and* finite gradients;
- `sum(h_accept) + sum(h_reject) == 1` for every candidate;
- `log_h` gradient w.r.t. `v` and w.r.t. `(diam, omega, tref)` matches finite differences.

---

## 8. `jax_dsmc/collide/ntc.py` — No Time Counter driver

### 8.1 Attempt count — stochastic rounding, and why it is not optional

`collide_vss.cpp:137-161`. SPARTA offers **three** schemes, not two:

| SPARTA path | Formula | Score term exists? |
|---|---|---|
| `mcflag` | `n ~ Poisson(f)` | yes |
| `remainflag` | `n = floor(f + remain)`, carry the fraction | **no** |
| default (neither flag) | `n = floor(f + U)` — stochastic rounding | **yes** |

We take **SPARTA's default**, stochastic rounding:

```
def attempt_count(counts, vremax, volume, cfg, key):
    npc = counts.astype(f64)                                  # f64[C]
    f   = 0.5 * npc * (npc - 1.0) * vremax[:, 0, 0] \
          * cfg.dt * cfg.fnum / volume                        # f64[C]  TRACED
    u    = uniform(key, (C,))                                 # STAGE_NTC_ROUND
    frac = f - floor(f)
    up   = u < frac                                           # FROZEN (bool)
    n    = (floor(f) + up).astype(i32)                        # i32[C]  FROZEN
    logP = where(up, safe_log(frac), safe_log(1.0 - frac))    # f64[C]  score term
    return n, logP
```

> **Why this replaces the deterministic carry.** The earlier draft used `n = floor(f)` with a
> `remain` carry-over and asserted that "the gradient of the collision rate w.r.t. these
> parameters is carried by the score terms." **That was wrong**, and it was the most serious
> defect in revision 2. Measured on a one-cell toy (`scratch/probe_grad_claim1.py`):
> `jax.grad(J)` w.r.t. `(dt, fnum, volume)` is `[0.0, 0.0, 0.0]` — exactly zero, while the
> velocity path is healthy at `‖∂J/∂v‖ = 9.5`.
>
> The reason is structural. `f` is traced, but `n = floor(f)` is an integer, so the path dies at
> the `floor`. And no score term rescues it: the only scores in §7.3 are `log(q/Σ)` and
> `log(1-q/Σ)`, and `q` depends on `v` and `prefactor(diam, ω, tref)` — on **none** of `dt`,
> `fnum`, or `lo`/`hi`. Those three parameters live *only* in `f`. §7.3's own table gives it
> away: the `j=1` "no virtual collision" row is marked "no score contribution", and `nattempt`
> is exactly where `dt`, `fnum` and the cell volume live.
>
> Under the deterministic carry there is no probability to score — `P(n | f, remain)` is
> degenerate — so no `log P` exists to differentiate. Stochastic rounding makes the count a
> genuinely random, parameter-dependent decision, which is exactly the shape the rest of the
> adjoint already handles.
>
> Note the spec already had both pieces and failed to connect them: §5.3 defines
> `stochastic_round`, and `STAGE_NTC_ROUND` was declared in §1 — but was used only to *seed*
> `remain`, never to round.

**Verification that the fix is unbiased.** Rao-Blackwellising the count decision at a fixed
configuration gives a zero-variance ground truth: `J(n=3) = 11.99258`, `J(n=4) = 12.59169`, so
`dE[J]/d(dt) = (J_up - J_down) * df/d(dt) = 8.087997e4` exactly. The stochastic-rounding
estimator converges to it:

| samples | `dJ/d(dt)` estimate | z vs. truth |
|---|---|---|
| 1e3 | 1.33e5 ± 1.2e5 | 0.44 |
| 1e4 | 7.22e4 ± 3.8e4 | −0.23 |
| 1e5 | 8.373e4 ± 1.19e4 | 0.24 |

> **A baseline is mandatory for this score term, not optional.** The count score carries `1/frac`
> and `1/(1-frac)` factors, so it is heavy-tailed. At `M = 1e5` on a single configuration:
> `b = 0` gives sd `3.76e6`; `b = 12` (a mean-`phi` baseline) gives sd `4.50e4` — an **84x
> variance reduction** — and tightens the estimate to `8.0914e4 ± 1.4e2` against a truth of
> `8.088e4`. §11.4 offers baselines as opt-in; for the count score that is not good enough.

`f` is traced through `dt`, `fnum`, `vremax`, and `cell_volume` (hence `lo`/`hi`). `n` and `up`
are frozen decisions, as the papers require; the sensitivity now travels via `logP`, which is a
genuine score term rather than an asserted one.

Dropping the carry also retires the `remain` field from `CollideState` and supersedes [DF]'s
`U[0,1)` remain seeding (§18.5): stochastic rounding is unbiased from step one, so there is no
startup transient left to fix. Keep the [DF] citation as the reason we went looking.

> **`[:, 0, 0]` is a `G == 1` placeholder.** The shape contract declares `vremax` as `(C, G, G)`,
> but the attempt count above indexes group pair `(0, 0)` only. The multi-species form is a
> *different formula*, not a broadcast: SPARTA computes `npairs = 0.5*n_i*(n_i-1)` within a group
> but `npairs = n_i*n_j` across groups (`collide_vss.cpp:165-196`), and sums an attempt count per
> group pair. Flagged here rather than left to be discovered when `G > 1` silently under-collides
> cross-group pairs.

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

The collision loop is sequential within a cell: a particle may be selected more than once in a
step, and each collision sees the velocities left by the previous one. Naively vectorising all
attempts at once would change the physics.

The structure that preserves that ordering while being GPU-parallel is a **`scan` over the
attempt slot, vectorised across cells**. Note the `N+1`-th scratch row and the clamped `npc` —
both are load-bearing, for reasons given immediately below.

```
def collide(state, table, cstate, grid, species, cfg, step, base_key):
    volume  = cell_volume(grid)
    k_cnt   = key_for(base_key, step, STAGE_NTC_ROUND)
    n, logP = attempt_count(table.counts, cstate.vremax, volume, cfg, k_cnt)

    # Stale-max rule (7.2): Sigma is read ONCE and held fixed for the whole
    # loop. It is not part of the scan carry.
    # The carried vremax is only a FLOOR-ed ratchet: on its own it lags a
    # transient and fails the U3 gate (7.2.3 / 17.5, violation rate 1.0e-2 and
    # margin 0.62). The a-priori thermal majorant is what supplies the 2-5x
    # margin, and cell_temp is known here because sample.py computed it.
    Sigma   = stop_gradient(maximum(cstate.vremax[:, 0, 0],
                                    thermal_majorant(cell_temp, omega, sp, cfg)))  # f64[C]

    k_pair  = key_for(base_key, step, STAGE_PAIR)
    k_acc   = key_for(base_key, step, STAGE_ACCEPT)
    k_sca   = key_for(base_key, step, STAGE_SCATTER)

    # v and logh carry an extra SCRATCH ROW at index N. Every inactive cell
    # writes there instead of into a neighbour's particles.  See 8.4.1.
    v0    = concat([state.v,        zeros((1, 3))])            # f64[N+1, 3]
    logh0 = concat([state.logh_acc, zeros((1,))])              # f64[N+1]

    def slot(carry, a):
        v, logh, qmax, nviol = carry
        active = (a < n) & (table.counts >= 2)                 # bool[C]

        i, j = pair_ntc(fold_in(k_pair, a), table.counts)      # clamps npc internally
        pi   = where(active, table.order[table.starts + i], N) # inactive -> scratch row
        pj   = where(active, table.order[table.starts + j], N)
        vi, vj = v[pi], v[pj]

        q      = collision_rate(vi, vj, ...)                   # f64[C]
        u      = uniform(fold_in(k_acc, a), (C,))
        p      = acceptance_prob(q, Sigma)                     # FIXED Sigma
        accept = active & (u < p)                              # FROZEN (bool)

        vi_n, vj_n = scatter(vi, vj, ..., fold_in(k_sca, a))
        v = v.at[pi].set(where(accept, vi_n, vi))
        v = v.at[pj].set(where(accept, vj_n, vj))

        # score accumulator: BOTH partners get the SAME log h, and BOTH the
        # accept and reject branches contribute. See section 11.
        lh   = where(active, log_h(q, Sigma, accept), 0.0)
        logh = logh.at[pi].add(lh).at[pj].add(lh)

        # next step's majorant: a running max over slots, masked so that an
        # untested slot's meaningless q cannot inflate it.
        qmax = where(active, maximum(qmax, q), qmax)

        # 7.2.1: a violated majorant contributes a ZERO score and biases the
        # gradient, so it must be counted rather than silently absorbed.
        nviol = nviol + sum(active & (q > Sigma))

        return (v, logh, qmax, nviol), None

    (v, logh, qmax, nviol), _ = scan(slot, (v0, logh0, Sigma, 0), arange(cfg.A_max))

    # --- the COUNT score (8.1) -------------------------------------------
    # logP is a PER-CELL term: the decision "how many pairs to attempt" applies
    # to every particle in that cell, so each of them carries it. This is the
    # j=1 branch of 7.3, and it is where the entire dt / fnum / cell_volume
    # sensitivity enters. Omitting this line reproduces the revision-2 defect
    # exactly -- those three gradients read 0.0.
    logh = logh.at[:N].add(where(state.alive, logP[table.cell], 0.0))

    vremax_next = maximum(cstate.vremax, qmax[:, None, None])
    diag = Diagnostics(majorant_violations=nviol, ...)
    return replace(state, v=v[:N], logh_acc=logh[:N]), CollideState(vremax_next), diag
```

#### 8.4.1 Why the scratch row is mandatory

Revision 2 called `pair_ntc` unconditionally for every cell and asserted "no write conflicts:
`pi` and `pj` for different cells index disjoint particles". **That was false, and it silently
corrupted results.** Measured (`scratch/probe_scan_clobber.py`): with `A_max = 3`, **176 of 300
runs** disagreed with a serial reference and violated momentum conservation.

The failure chain, all of it silent:

1. `jax.random.randint(key, shape, 0, 0)` does **not** error in JAX 0.6.2 — it returns
   all-zeros, deterministically (2000/2000 seeds). So does `randint(..., 0, -1)`, and
   `(i + 1 + r) % 0` returns 0 rather than raising. An empty cell therefore gets `i = j = 0`.
2. For an empty cell `c`, `starts[c] == starts[c+1]`, so `pi = order[starts[c]]` is the **next
   cell's first particle**. With *trailing* empty cells `starts[c] == N`, and the out-of-bounds
   gather `order[N]` is **clamped by JAX** to `order[N-1]` — a particle in an unrelated cell.
3. The clobber itself is **deterministic, and is not a duplicate-index race.** (Duplicate-index
   `.at[].set` *is* unspecified in JAX — with 4096 updates aimed at index 0 the winner was update
   1375, not 4095 — but that is not the mechanism here.) The mechanism is the **two-scatter
   structure**: `vi` and `vj` are both gathered from the *same* pre-scatter `v`, so the second
   `.set` writes a `vj` that is stale with respect to the first. When `pi == pj == p`, as it is
   for every inactive cell, S2 unconditionally writes `v_old[p]` back over whatever S1 left
   there — undoing a legitimate collision written by the cell that actually owns `p`.

Traced on a 6-cell, `N=10` table (`counts = [2,0,4,0,1,3]`):

```
pi = [0 2 2 6 6 7]   pj = [1 2 3 6 6 8]   active = [T F T F F T]
after S1:  v[2] = 4.0      # cell 2's collision landed correctly
after S2:  v[2] = 3.0      # empty cell 1 overwrote it
momentum 55.0 -> 54.0,  energy 385 -> 378
```

Cells with `counts == 1` are harmless — they write their own particle's own stale value and
nobody else writes there. `counts == 0` cells are the corruption source.

**The revision-2 loop was also not self-consistently differentiable**, independent of the
corruption: `jax.grad` against central finite differences on a smooth scatter gave
**relerr 6.3e-3**, because the scatter's VJP accumulates cotangents into all duplicate indices
while the forward pass keeps only one. After the fix: **1.1e-10**.

**The fix**, and its cost. `npc_safe = maximum(counts, 2)` inside `pair_ntc`; `v` and `logh`
allocated at `N+1`; `pi`/`pj` routed to row `N` when inactive. `logh` needs the extra row too, or
`.at[N].add()` clamps back into `N-1`. Corruption goes `176/300 -> 0/300`, results match the
serial reference exactly, and the gradient becomes FD-exact. Measured on GPU in f64:

| C | N | A_max | rev-2 | fixed | ratio (3 runs) |
|---|---|---|---|---|---|
| 4096 | 262144 | 8 | ~0.22 ms | ~0.30 ms | **1.02 – 1.69, not reproducible** |
| 16384 | 1048576 | 8 | ~0.75 ms | ~0.80 ms | 1.03 – 1.10 |
| 65536 | 4194304 | 16 | ~6.2 ms | ~6.5 ms | 1.03 – 1.04 |

**3–10% wall time and 24 bytes of memory** to remove a silent correctness bug.

> The smallest configuration is **not a usable measurement** — repeated runs of the identical
> config gave ratios of 1.02, 1.40 and 1.69 (and once 0.14, i.e. the "fixed" version apparently
> 7x faster). At `C = 4096` the kernel is dominated by launch overhead, so the benchmark is
> measuring noise. Only the two larger rows are meaningful, and they agree: **~3-10%**. This is
> recorded because an earlier draft of this document quoted "1.02x" from a single run of that
> row as though it were a result. Any performance claim here needs a repeat-count and a spread,
> not a single number.

Why the loop is correct and fast:

- **Sequential within a cell**: slot `a+1` sees the velocities written by slot `a`. A particle
  can collide twice in a step, matching both reference codes.
- **Parallel across cells**: at each `a`, all `C` cells are processed as one vectorised batch.
- **`Sigma` is loop-invariant** (7.2). It is closed over, not carried, so the acceptance test is
  the same comparison for every slot. `qmax`
  accumulates the *next* step's majorant as a masked running maximum; it never feeds back into
  this step's acceptance.
- **Writes are disjoint, by construction rather than by assertion**: active cells write only to
  particles in their own slice of `order` (`i, j < counts[c]`, guaranteed by the clamp), and
  every inactive cell writes only to the scratch row `N`, which is discarded. Note the scratch
  row genuinely does receive many duplicate writes — that is fine precisely because nothing
  reads it.
- **Reverse-differentiable**: `scan` with a static trip count; `at[].set` and `at[].add` both
  have well-defined VJPs; the trip count does not depend on data.
- **Rematerialisable**: keys are recomputed from `(base_key, step, a)`, so a checkpointed
  backward pass replays the identical random stream.

The `active` mask on the `qmax` update matters. An inactive slot still evaluates `q` (both
branches of a `where` are computed), but that `q` comes from a pair that was never really drawn
— letting it into the maximum would inflate the majorant, silently lowering the acceptance rate
for every subsequent step.

Substituting `pair_perm` for `pair_ntc` is the only change needed to run the oracle
configuration; the rest of the loop is identical.

### 8.5 Tests (`tests/test_ntc.py`)

- `pair_ntc` uniformity: over many draws, `j` is uniform on the complement of `{i}`, chi-square
  test, and `j != i` always;
- `pair_perm` is a valid perfect matching: every particle appears exactly once across all pairs,
  no self-pairs, and the induced pairing is uniform over matchings by chi-square;
- **`Sigma` is invariant across the attempt loop** — capture it before and after and assert
  bitwise equality. This is the stale-max rule (7.2) as an executable check;
- the `qmax` mask works: with `A_max` far larger than any `n`, `vremax_next` is unchanged from a
  run with `A_max = max(n)`, i.e. inactive slots contribute nothing;
- **momentum and energy are conserved on a table containing empty cells** — the regression
  test for 8.4.1. Build `counts = [2,0,4,0,1,3]` explicitly, including trailing empty cells, and
  assert exact conservation and agreement with a serial (Python-loop) reference. Without the
  scratch row this fails on ~59% of seeds;
- `jax.grad` of the collide loop matches central FD on a smooth scatter to ~1e-10; the
  unfixed loop gives ~6e-3, so this catches the duplicate-index VJP inconsistency too;
- `attempt_count` matches the formula on hand-computed values;
- stochastic rounding is unbiased: `mean(n)` over many draws equals `f`, from step one (no
  startup transient, which the old deterministic carry did have);
- **the count score restores the lost gradients**: `jax.grad` w.r.t. `dt`, `fnum` and
  `cell_volume` is nonzero and matches the Rao-Blackwellised truth
  `(J_up - J_down) * df/d(param)` within MC error. This is the regression test for the
  revision-2 defect -- without the `logP` term it reads exactly `0.0`;
- the count-score baseline reduces variance by >10x at the reference configuration;
- with `A_max` deliberately too small, `diag.attempts_clipped > 0` — the diagnostic fires;
- equilibrium collision rate matches the analytic VHS rate
  `nu = 4 n d^2 sqrt(pi k T / m)` (at `omega = 0.5`) within MC error;
- the scan is order-dependent in the right way: running with `A_max` slots gives a different
  (correct) result from `A_max` independent parallel attempts, confirming sequencing is real.

---

### 8.5.1 How `A_max` is actually chosen: measure it, do not guess it

`A_max` is the static trip count of the attempt-slot `scan`, so it is baked into the compiled
program and cannot change while that program runs. But **the right value is a property of the
flow, not of the setup**, and no a-priori rule survives contact with a real case:

| attempt to bound it | outcome |
|---|---|
| size from the cold uniform attempt count | 5.4x too small at the shock time (M2, Mach 1) |
| add the majorant inflation of U4 | still short -- the ratchet was only **1.10x** of the 5.4x |
| bound compression by Rankine-Hugoniot, `(g+1)/(g-1) = 4`, so `n ~ npc^2` gives 16x | **wrong**: measured density ratio **11**, because a symmetric *converging* wave is not a steady normal shock and RH does not bound it |

The decomposition is the useful part, and it corrects §8.2's U4 note. U4 attributes `A_max` growth
to the inflated `Sigma`; with transport in the loop that term is almost irrelevant:

| contribution at the shock time | factor |
|---|---|
| `vremax` ratchet (what U4 describes) | 1.10x |
| cell temperature 1.66x, worth on `Sigma` | 1.10x |
| **compression** (`n ~ npc^2` at fixed cell volume) | **~4.5x** |

**The resolution is `simulate.run_adaptive`.** What makes it cheap is that *restarting costs
nothing*: the simulation state is just arrays, so continuing after a recompile pays only for the
compile. The run proceeds in chunks; after each chunk `attempts_clipped` is read on the host, and
if capacity was exceeded `A_max` grows and **that chunk alone** re-runs from the state it began
with. Measured: starting from a deliberately absurd `A_max = 64` it converged in two resizes and
finished in **41.9 s against 48.6 s for a single over-provisioned run** -- measuring is cheaper
than guessing high, because guessing high pays for unused slots on every step.

`run(..., step0=)` offsets the RNG step index so chunking changes nothing: `x` and `v` come back
**bitwise identical** to an unchunked run, and `logh_acc` agrees to `6e-14` relative (841 ULPs),
the residual being float reassociation in the accumulator chain, not a difference in any decision.

Not for use *inside* a differentiated run -- the resize branch is host-side Python control flow.
Calibrate with `run_adaptive`, then differentiate through `run` at the settled `A_max`. That is the
natural split anyway: an optimiser's outer loop resizes between iterations.

> **SPARTA does not solve this, and cannot be copied here.** It has no `A_max` at all: a CPU `for`
> loop over `nattempt` costs nothing, so the question never arises. Its `reset_vremax()`
> (`collide.cpp:2353`) is called only from `init()` (lines 368, 396), so it is not a running-time
> remedy either. Static shapes are what force the question on us.
>
> **MCF does not help either.** SPARTA's `mcflag` (`collide_vss.cpp:147,188`) draws
> `n ~ Poisson(f)` -- the same `f` and the same `q/Sigma`, only a different *count distribution*.
> It is more physically faithful (stochastic rounding under-disperses the count), but Poisson has a
> tail, so the peak `n` **exceeds** `ceil(f)` and `A_max` gets harder, not easier.

### 8.5.2 Rejected alternative: fixed attempt count with rescaled acceptance

Recorded because it was verified and is genuinely attractive, so it will be proposed again.

Make exactly `A` attempts per cell and accept each with

```
p = (n/A) * (q/Sigma) = 0.5 npc(npc-1) dt fnum q / (A V)          # Sigma cancels
```

Expected collisions `A <p> = n <q>/Sigma`, identical to NTC. It is simply NTC with the majorant
chosen so the attempt count comes out exactly `A`: `Sigma' = A V / (0.5 npc(npc-1) dt fnum)`.

Verified in `scratch/probe_fixed_attempts.py` and `scratch/probe_fixed_attempts_grad.py`:

| claim | measured |
|---|---|
| collision count unchanged | within **0.4%** across `A` from `0.5n` to `8n` |
| variance not worse | sd 11.7 at `A = n` vs 11.5 for NTC; +17% at `16n` |
| needs fewer slots | `A = 294` required vs `n = 673`, since `Sigma` carries the `F_UP = 2`, `K_EV = 5` factors |
| carries the `dt`/`fnum`/`V` gradient with **no count score** | all three match Rao-Blackwellised truth at `\|z\| = 0.82` |

That last row is the striking one: it would **delete** §8.1's stochastic rounding and its
`log P(n\|f)` term -- the machinery that fixed defect 1 -- because the count stops being a random
integer and the parameters enter `p` smoothly instead.

**Not adopted.** It changes the fundamental NTC acceptance probability, and the verification above
is single-species, single-group. Multi-species NTC computes a separate `n` and `Sigma` per *group
pair* (`collide_vss.cpp:163-196`, with `npairs = n_i n_j` across groups), so `A` would have to
become per-group-pair and the equivalence re-derived for each -- and again for Larsen-Borgnakke and
for chemistry, where the acceptance test carries more than one factor. The gain over `run_adaptive`
is a 2.3x slot saving; the cost is re-justifying the core of the collision algorithm in every
future context. Revisit only if slot count becomes the binding constraint, and only with the
multi-group derivation done first.

### 8.6 Measured cost of the attempt-slot scan

Measured on the implemented kernel (`scratch/perf_collide_scaling.py`,
`scratch/perf_production_scaling.py`), because the shape of this cost is not obvious from the
code and one plausible-sounding worry turned out to be wrong.

**The parallel axis is cells, not attempt slots.** The scan is *sequential* in `a` -- it has to
be, since slot `a+1` must see the velocities slot `a` wrote -- and *vectorised* across `C`. So
throughput is governed by `C`, and the per-slot cost is amortised over the cells that share it:

| `C` (at `N = 128k`, `A_max = 512`) | 1 | 64 | 512 | 4096 |
|---|---|---|---|---|
| us per slot | 117.9 | 161.3 | 110.8 | 188.4 |

4096x the useful work for 1.6x the time. **In the production regime -- particles per cell held
fixed, cell count growing -- the step is flat to 1e6 particles:**

| `N` (at `npc = 32`) | 2,048 | 16,384 | 131,072 | 1,048,576 |
|---|---|---|---|---|
| step time (s) | 0.0402 | 0.0333 | 0.0186 | 0.0496 |
| ns per particle | 19,637 | 2,032 | 142 | **47** |

512x the particles for 1.23x the wall time: below ~1e5 particles the kernel is launch-latency
bound and the GPU is mostly idle, and only past ~1e6 does it begin to do real work.

**The 0-D milestone-1 case is the worst case for this design and is not representative.** With a
single cell the "vectorised across cells" axis is exactly 1 wide, and the attempt count
`f = 0.5 npc(npc-1) Sigma dt fnum / V` has `npc = N`, so `A_max ~ N`: thousands of sequential
iterations each doing arithmetic on one cell. That is why GPU utilisation looks *spiky* during the
BKW gate -- it is one tiny kernel at a time with sync latency between, not a recompilation loop.
It is nonetheless **O(N), not O(N^2)** (measured 1.00 / 1.96 / 3.20 / 6.30x for `N` doubling from
4k to 32k, against the 1 / 4 / 16 / 64x that `O(N^2)` predicts): the per-slot cost is dominated by
a fixed ~40 us floor at these sizes, and the full-array term only becomes visible past `N ~ 1e5`
(per-slot 59 -> 117 -> 336 us at `N` = 32k -> 128k -> 512k).

**Section 1 rule 3 is a performance requirement, not a style preference.** The first
implementation drew inside the scan body -- `fold_in` plus `randint`/`uniform` per slot, about nine
tiny kernels per iteration. Hoisting every draw into one batched `(A_max, C)` array per stage,
consumed as the scan's `xs`:

| | inside the loop | batched (section 1 rule 3) |
|---|---|---|
| compile time | 17.0 s | **2.4 s** |
| us per slot (`N = 8k`) | 102.4 | **53.1** |

**Audit for other defects of the same class.** Having found one piece of work in the wrong place,
the rest of the step was audited the same way -- by ablation, not by reading. Four suspects, three
real:

| suspect | verdict | measured |
|---|---|---|
| `moments` called every step but only `temp` used | **real** — the pressure tensor materialises an `(N,3,3)` intermediate for nothing | 2.9-3.5x; fixed by `sample.cell_temperature` |
| `moments` recomputes `cell_of` although `table.cell` holds it | **real** — `build_table` returns it for exactly this reason | a further 1.6-3.0x |
| scratch-row `concatenate` on every `collide` call | **not real** — XLA fuses it | no measurable overhead at `N` up to 4e6 |
| scan carry copied per iteration instead of updated in place | **real, but only at `C == 1`** | see below |

**The carry copy is a `C == 1` pathology, not a general defect.** A `lax.scan` whose carry is an
`(N+1,3)` array written by `v.at[idx].set(...)` was measured against an identical loop that only
*reads* the carry, sweeping both axes:

| `C` | growth in us/slot, `N` = 64e3 -> 8e6 (writing body) |
|---|---|
| **1** | **52.4x** (44.8 -> 2344) |
| 8 | 1.3x |
| 64 | 1.1x |
| 512 | 2.1x |
| 4096 | 1.5x |

The read-only control is flat at every `C`. So for `C >= 8` XLA updates the buffer in place and the
per-slot cost is independent of `N`, as the design intends; only a *single-index* scatter degenerates
into an O(N) copy. `fori_loop`, `.at[].add`, and a compact `(C,K,3)` workspace were all measured as
alternatives and none beat plain `scan` once `C >= 8`, so **no code change is warranted** -- the
production regime was never affected.

**Consequence for the M1 gate.** A single cell is exactly the degenerate shape, *and* it is where
`A_max ~ N`, so the two compound. Since the BKW gas is homogeneous, splitting it across cells leaves
the physics unchanged -- `npc` falls by the cell count and the cell volume falls with it, so the
per-cell rate `0.5 npc(npc-1) Sigma dt fnum / V_cell` is preserved and the total attempt count changes
only by `(N-C)/(N-1)` -- while `A_max`, which is *per cell*, falls with it:

| cells | `npc` | `A_max` | s/seed | fitted-rate bias |
|---|---|---|---|---|
| 1 | 40,000 | 7,800 | 16.16 | +2.51 +- 2.05 % |
| 8 | 5,000 | 975 | 2.56 | +2.63 +- 2.10 % |
| 64 | 625 | 122 | **1.27** | -2.62 +- 1.16 % |

**12.7x faster with the physics unchanged**, which is what makes the seed counts the gate actually
needs affordable. `examples/bkw.py` therefore defaults to 8 cells (`--cells-per-side 2`).

**The cell count is the dominant speed knob, and pinning it is a scaling defect.** `A_max` is
*per cell*, so if the cell count is held fixed while `N` grows, `npc = N/C` grows, `A_max` grows
with it, and the **sequential** attempt-slot scan lengthens while the **parallel** cell axis stays
fixed -- precisely backwards. Holding `npc` fixed instead keeps `A_max` constant and puts every bit
of the growth on the parallel axis. Same physics, same total attempts (`~N` either way):

| `N` | cells | `npc` | `A_max` | run (s) | us/particle | fitted rate |
|---|---|---|---|---|---|---|
| 160,000 | 125 | 1,280 | 250 | 1.44 | 8.97 | 0.3340 |
| 640,000 | 729 | 877 | 171 | **0.68** | 1.06 | 0.3310 |
| 2,560,000 | 2,744 | 932 | 182 | 2.38 | 0.93 | 0.3325 |
| 10,240,000 | 10,648 | 961 | 188 | 10.65 | 1.04 | **0.3332** |

versus pinned cells, where `A_max ~ N`:

| `N` | cells | `A_max` | run (s) |
|---|---|---|---|
| 160,000 | 8 | 3,900 | 7.45 |
| 640,000 | 8 | 15,600 | **26.38** |

**39x at `N = 640,000`**, and 1e7 particles goes from unreachable to 10.6 s -- on a 4 GB laptop
GPU. Cost per particle falls from 36 us at `N = 1e4` to ~1 us for `N >= 6e5` and then stays flat:
that is the launch-bound to work-bound transition, and it is where the design's parallelism finally
gets used. Note the *accuracy* also improves, because more of the attempt budget is real work:
`0.3332` against an exact `1/3` at `N = 1e7`.

**CPU versus GPU: the crossover is around 3e5-5e5 particles.**

| `N` | GPU (s) | CPU (s) | faster |
|---|---|---|---|
| 10,000 | 0.362 | **0.080** | CPU 4.5x |
| 160,000 | 1.435 | **0.964** | CPU 1.5x |
| 640,000 | **0.680** | 2.112 | GPU 3.1x |
| 10,240,000 | **10.65** | 43.14 | GPU 4.1x |

Below the crossover the GPU is entirely launch-latency bound and the CPU simply wins; above it the
GPU pulls ahead by 3-4x. Run small cases with `JAX_PLATFORMS=cpu`. The fitted rate is **identical
on both backends to every printed digit**, which is an independent check that the RNG contract
(section 1) really is device-independent.

> **Benchmarking discipline, learned the hard way.** An earlier run of this suite reported a
> dramatic 36x "cliff" in per-slot cost at `A_max = 8192`. There is no cliff: 59.0 / 41.0 / 47.8 /
> 40.5 / 40.2 us per slot across `A_max` = 1024...16384, flat. The phantom came from measuring
> while a second GPU job was running. **Do not run these benchmarks concurrently with anything
> else**, and treat any single timing that implies a qualitative change as unproven until it is
> reproduced on an idle device. This is the same failure mode as the "1.02-1.11x" cost claim in
> U9, which is why performance numbers in this document carry repeat counts and spreads.


## 9. `jax_dsmc/move.py` and `jax_dsmc/boundary.py` — advection and BCs

On a uniform grid with no embedded surfaces, SPARTA's general cell-walking loop
(`update.cpp:937-1029`) collapses to its own `OPT` fast path (`update.cpp:764-864`): advance the
full step, apply the face BC, recompute the cell by arithmetic. There is no walk, no per-crossing
face test, and no neighbour lookup.

```
def move(state, grid, cfg, step, base_key):
    x_new = state.x + state.v[:, :D] * cfg.dt         # only the first D components
    x, v  = boundary.apply(x_new, state.v, grid, cfg, step, base_key)
    return replace(state, x=x, v=v)
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

> **Slot reuse resets per-particle accumulators.** Through M4 particles are only destroyed, so
> `alive` is monotone non-increasing and a dead slot is never touched again. From M5, inflow
> refills dead slots — and a refilled slot **must** have `logh_acc` zeroed, or the newly injected
> particle inherits the score history of whatever previously occupied that index and contributes
> a spurious `eta` term for collisions it never took part in. The rule is: any code path that
> sets `alive[i] = True` for a previously dead `i` must also set `logh_acc[i] = 0.0`, in the same
> update. Same for any future per-particle accumulator.
>
> This also retires the M1-M4 test assertion in 9.5 that the alive count is monotone
> non-increasing; from M5 it is not, and the replacement invariant is a mass balance
> (`injected - deleted == change in alive count`).
>
> **Built** — `jax_dsmc/inflow.py`. `revive_slots` is the *only* sanctioned way to set
> `alive` from `False` to `True` and zeroes every field in `ACCUMULATOR_FIELDS` in the same
> update; `ACCUMULATOR_FIELDS` is checked by reflection over `ParticleState`, so a future
> accumulator cannot be silently missed. The mass balance is asserted at **every** step of a
> 400-step inflow run and holds exactly.
>
> **The obvious U6 test proves nothing, and this was found the hard way.** Asserting "a stale
> `logh_acc` changes the gradient" *fails*: `magic(L) = exp(L - stop_gradient(L))` has gradient
> `dL/dtheta`, so a stale **constant** differentiates to zero. The real hazard is that the
> previous occupant's score was itself built from `theta`-dependent collisions, so the inherited
> term carries a nonzero `dL/dtheta` into the new particle's reward. The gate now makes
> `logh_acc` a genuine function of `theta` and predicts the contamination quantitatively —
> exactly `sum(inherited * phi[dead]) / N`, matched to `1e-10`.

### 9.4 Diffuse / thermal — M4 forward, M7 adjoint (see 9.4.1)

**Closed form, no sub-move scan.** [P3] eq. `thermal_v_L`/`thermal_v_R` gives the entire bounce
as one expression:

```
v_{k+1} = g,        x_{k+1} = L + (x' - L)/(P v') * (P g)
```

`(x' - L)/(P v')` is the *overshoot time* — how long the particle spent past the wall — and it is
re-advected with the **new** normal velocity. So the mesh stays static, every face rule stays a
closed-form function of the end-of-step position, and the fast path is preserved. The flux bias
lives in the half-Maxwellian draw (`sample_flux_normal`), not in geometric tracking.

> **Revision.** This section previously specified "a bounded `scan` of `K_bounce` masked
> sub-moves carrying `dtremain`". That was SPARTA's approach imported wholesale, and SPARTA needs
> hit times only because it has general cut-cell geometry — on a static axis-aligned mesh it buys
> nothing. Both were implemented and measured: they agree with the closed forms for
> `PERIODIC`/`SPECULAR` to `2e-19`, reach the same wall equilibrium, and the closed form runs the
> same 60-step, 1e5-particle case in **1.65 s against >7 min**.
>
> [P3] notes the closed form "does not include the possibility of multiple boundary crossings";
> neither does SPARTA's specular (`domain.cpp:337-357`). **Measured**, at `L = 1e-3`, 8 cells,
> argon at 300 K, the fraction of particles crossing two faces in a step is:
>
> | `dt / (dx/v_th)` | 0.15 (production) | 1.0 | 2.0 | 4.0 | 8.0 |
> |---|---|---|---|---|---|
> | recrossed fraction | **0** | **0** | 1e-6 | 4.2e-3 | 8.5e-2 |
>
> Zero up to ~7x the production CFL step. So a single pass is right, and the multi-crossing case
> is **detected and reported** (`recrossed`, the wall analogue of `attempts_clipped`) rather than
> iterated over — a silent one would be a momentum sink. Anything still outside is clamped, which
> is a safety net rather than physics: without it `cell_of` would clip it into an edge cell
> silently.
>
> **Corners are a separate effect, and they do not vanish with `dt`.** In `D >= 2` a particle can
> leave through two *different* faces in one step; that fraction scales as `(v dt / L)^(D-1) N`,
> not as the same-face rate above.
>
> **Both wall interactions are resolved.** `n_bounce_passes` is a *static* function of the face
> layout: a particle can only reach two thermal faces in a step if they lie on different **axes**,
> since after bouncing off one face of an axis it travels away from the opposite face. So thermal
> on one axis (Couette) runs one pass and pays nothing; thermal on two or more runs two. Each pass
> leaves the particle *at* its wall with `t_used` advanced, and a single final advance by
> `dt - t_used` completes the step — which is what makes the passes composable. Measured on a box
> thermal on every face: `D = 3` equilibrium went from **309.9 K to 300.90 K** against a 300 K
> wall, and the residual `recrossed` fell 1.8e-3 -> 3.3e-4 (`D = 2`). What is left is the tail
> needing a *third* crossing, worth ~0.3%.
>
> The target geometries are exact regardless, because periodic, specular and outflow keep their
> exact any-number-of-crossings folds and are applied *after* the wall bounce:
>
> | configuration | recrossed / particle-step |
> |---|---|
> | thermal on every face (worst case, 2 passes) | 3.3e-4 |
> | **Couette** — periodic `x`, thermal `y` | **0** |
> | **shock** — outflow `x`, specular `y` | **0** |
>
> So the ordering inside `apply_faces` is load-bearing: wall bounce first, exact folds second.

`surf_collide_diffuse.cpp:206-320`:

```
v_perp = v_rm(T_wall) * safe_sqrt(-log(U))            # flux-weighted Rayleigh
v_tan  = maxwellian tangential components
```

`v_perp` is reparameterised in `T_wall`, so `dJ/dT_wall` is pathwise. `vwall` is added after the
draw, so `dJ/dv_wall` is too.

### 9.4.1 M7: what was built, and what each parameter costs

**Built** (all default-off, so every M1-M5 forward result is unchanged -- verified: the Couette
gate reproduces pre-M7 output to every digit, and a 30 000-step inflow shock reproduces all six
moments and every diagnostic with zero mismatches):

| piece | where | knob |
|---|---|---|
| crossing-event score, [P3] app:`log` | `boundary.crossing_score` | `params.epsilon` (0 → score is exactly 0, since `log 1 = 0`) |
| accommodation Bernoulli score | `boundary._diffuse_velocity` | always on; zero-mean, so harmless |
| **relaxed** accommodation | same | `acc_delta` (0 → the hard threshold) |
| cell-wide score attribution | `boundary.spread_score_over_cell` | `score_spread="cell"` |
| inflow injection-count score | `inflow.inject` | always on |

The per-step boundary score is returned by `apply_faces` as a fifth value and added to
`logh_acc` by `simulate._step` -- the *same* accumulator the collision score uses, because
`surrogate` weights the reward by the gradient of the whole prefix and a separate accumulator
would drop the cross terms.

**Parameters divide by route, and the route decides everything:**

| parameter | route | status |
|---|---|---|
| `T_wall`, `v_wall` | pathwise | **verified** -- fixed-key FD to 3.3e-12 / 2.5e-11, and pointwise against the Couette closed form (§9.4.2) |
| inflow `nrho` | score | **verified** -- ensemble FD/AD = 1.04 |
| `acc` | score, or pathwise with `acc_delta > 0` | score form biased 2.8x at `M = 40` |
| shock `d(rho_2/rho_1)/du_1` | score | **wrong sign** |

A score-carried gradient is **biased low, and the bias grows with trajectory length** -- 1.13x
at `M = 2` rising to 2.82x at `M = 40`. The cause is reward attribution, not the
log-probability: with collisions switched off the same estimator is exact (`|z| = 0.18`).
**`docs/ATTRIBUTION.md` is the reference for this and should be read before quoting any
score-carried gradient.**

### 9.4.2 Two limits that bound every gradient here

Both were measured, and they are independent:

1. **Attribution bias** (score route only). See above.
2. **Chaotic amplification** (every pathwise gradient). The derivative of a *single*
   trajectory grows like `exp(lambda t)` while the quantity it estimates stays `O(1)`: the
   per-seed spread of `du_s/du_wall` grows **9 254x** between 10 and 800 steps. The estimator
   never becomes wrong, it becomes uninformative.

A third scaling follows from (2) and is the practical one: **particle count does not fix a
gradient.** At a 25-step horizon the forward observable and both gradients all scale
`N^-1/2`; at 200 steps the forward still does and the gradients are **flat** (`N^+0.00`,
`N^-0.10`). The chaotic part of the variance does not average over particles, so past the
horizon where it dominates, only ensemble size helps -- at `M^-1/2`.

Validation lives in `examples/couette/gradients.py` (against continuum Couette with Maxwell
slip, and against `dT/dT_wall = 1` exactly) and `examples/shock/gradients.py` (against
Rankine-Hugoniot).

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

> **M5 migrates cell fields to `.vtu`, and nothing else changes.** Cut and split cells are no
> longer a rectilinear grid, so `.vtr` cannot represent them and unstructured output is required.
> The `.pvd` collection layer and the `.vtp` particle path are unaffected. `.vtr` is kept until
> then rather than writing `.vtu` from the start, because on a uniform mesh `.vtr` implies the
> geometry while `.vtu` must store every point and cell explicitly — a real cost at large `C`.

**3. HDF5/npz state checkpoints** — `x`, `v`, `alive`, `logh_acc`, cell moments, the RNG base key
and the step index — for restart, offline analysis, regression baselines, and eventually the
frozen-decision record if the reversible adjoint (§18 deferred) is ever built. Restart
correctness requires the base key and step, because the RNG contract (§1) reconstructs every draw
from `(base_key, step, STAGE_ID)`; saving particle state alone is not resumable.

**4. Interactive diagnostics** — `summary(state, diag)`: per-cell moment table, `ncollide` /
`nattempt`, and the two counters that must never be silently nonzero,
`diag.attempts_clipped` (§8.2) and `diag.majorant_violations` (§7.2.1).

## 11. `jax_dsmc/adjoint.py` — the surrogate objective

This is the piece that makes the whole architecture work, and it is four lines.

### 11.1 The construction

> **Sign convention.** The two papers seed the backward pass with opposite signs. [P1] eq.
> `gammaFinal` sets `gamma_F = -d r/d v` because its `gamma` is a Lagrange multiplier on a
> constraint; [P2] eq. `gamma_final_disc` sets `gamma_i^M = (rho/N) phi'(v_i^M)` because its
> `gamma` is a plain influence function `d J^k / d v_i^k`. **We follow [P2]: `gamma` is the
> derivative of the objective, no sign flip.** That is also what `jax.grad` produces, so the
> production path needs no adjustment. Oracle 1 (13.1) transcribes [P1]'s recursion and must
> therefore negate at the seed, or compare against `-jax.grad`; the test states which.

[P2] eq. `gamma_rules` gives, for a general collision kernel:

```
gamma_i^M   = (rho/N) * phi'(v_i^M)
(gamma_i^k, gamma_i1^k) = D_i^k (gamma_i^{k+1}, gamma_i1^{k+1}) + (eta_i^k, eta_i1^k)
eta_i^k     = (rho/N) * d(log h_i^k)/d(v_i^k) * (phi_i^M + phi_i1^M)
```

`D` is the transposed collision Jacobian, which is what reverse-mode already computes. The `eta`
term is recovered by the **DiCE magic-box operator** applied to the per-particle accumulator
`logh_acc` that `collide` already fills:

```
def magic(L):
    # value is identically 1; gradient is dL. Sometimes written (x)_|.
    return exp(L - stop_gradient(L))

# rho = int f(v,T) dv, the total mass/number normalisation of the distribution
# the N particles represent ([P2] eq. obj1). With a single species and a fixed
# particle count it is a constant; it is carried explicitly because the objective
# J = (rho/N) sum_i phi_i must stay dimensionally correct as N varies.
def surrogate(state, phi, rho, N):
    phi_i = phi(state.x, state.v)                     # f64[N]
    return (rho / N) * sum(magic(state.logh_acc) * phi_i)
```

### 11.2 Why this is exactly the paper's adjoint

The value is unchanged, `magic == 1`, so `surrogate == J` numerically and can be reported
directly as the objective. The gradient is

```
grad = (rho/N) * sum_i [ grad(phi_i)  +  phi_i * grad(L_i) ]
```

- The first term is the pathwise contribution: reverse-mode transports it back through the
  collision maps, giving the `D` recursion.
- The second term: `L_i` accumulated `log h_i^k` for every attempt particle `i` took part in, and
  `L_i1` accumulated the **same** `log h_i^k`. Collecting the coefficient of
  `d(log h_i^k)/d(v_i^k)` therefore gives `phi_i + phi_i1` — precisely `eta_i^k`.
- Reverse-mode then continues transporting that `eta` contribution further back through earlier
  steps, which is what the recursion `gamma^k = D gamma^{k+1} + eta^k` does.

Using `magic` rather than the simpler `sum(phi) + sum(stop_gradient(phi) * L)` buys two things:
the primal value stays equal to `J`, and higher-order derivatives remain correct (the naive form
is right only to first order).

### 11.3 What flows where

`log h` is a function of the pre-collision velocities **and** of `diam`, `omega`, `tref` through
`prefactor`. So the score term supplies not only the velocity sensitivity the papers derive, but
also the direct parameter sensitivity of the collision rate — the model-calibration gradient —
with no extra machinery.

### 11.4 Variance

> **The count score (§8.1) requires a baseline; it is not optional.** Unlike the per-pair
> scores, `log P(n | f)` carries `1/frac` and `1/(1-frac)` factors and is heavy-tailed. Measured
> at the reference configuration: `b = 0` gives sd `3.76e6`, a mean-`phi` baseline gives sd
> `4.50e4` — **84x** — and tightens the estimate to `8.0914e4 ± 1.4e2` against a ground truth of
> `8.088e4`. Treat the baseline as part of that estimator's definition.

> **MEASURED (2026-09-05): the 84x does NOT carry over to a windowed spatial objective.**
> On the M2 shock at `D = 1` with `phi` = Gaussian-windowed kinetic energy, 512 base keys
> (`scratch/adjoint/probe_baseline.py`), the mean baseline moves the per-seed standard deviation
> by only **~1.2x** — `diam` `1.60e12 -> 1.34e12`, `omega` `3.10e2 -> 2.61e2`, `alpha`
> `3.55e1 -> 3.27e1`, `nrho` `1.52e-16 -> 1.50e-16`. The 84x was measured on the count score *in
> isolation*, where the reward is a scalar with a large mean relative to its spread. Here `phi`
> varies by orders of magnitude *within* a single run (a Gaussian window times `|v|^2`), so
> subtracting its global mean removes almost none of the reward's variance.
>
> Keep the baseline — it is free and never hurts — but **do not rely on it to make the score
> estimator affordable**. §23.3 gives the sample counts that are actually required, and a
> better baseline (per-cell, or leave-one-out) is the obvious next thing to try.

For the *per-pair* scores the papers' estimator uses reward `(phi_i^M + phi_i1^M)` with no
baseline; that is the default so results are comparable to theirs. An opt-in constant baseline `b` is provided:
`phi_i - b` in the reward leaves the estimator unbiased (because `E[score] = 0`) and can cut
variance substantially. A benchmark reports gradient variance against finite differences at
several `b`, so the choice is made from data rather than asserted.

### 11.5 Windowed objectives — the causal form

Everything above scores a **terminal** objective `J = phi(x_M, v_M)`. The objectives that
actually matter — time-averaged drag, integrated wall heat flux, a moment averaged over a
sampling window — are `J = sum_k r(x_k, v_k)`.

Extending naively (one magic box on the *final* `logh_acc`, times the summed rewards) scores each
reward against collisions that happened **after** it was emitted. That is still unbiased, since
`E[score] = 0` for decisions independent of an earlier reward, but it adds pure noise. The
correct estimator is causal DiCE, `sum_k magic(L_{<=k}) * r_k`.

**This is one scalar in the scan carry, not a redesign.** `logh_acc` *is already* the running
prefix `L_{<=k}` — it is accumulated in place as the scan proceeds — so the causal form is
obtained by accumulating the reward inside the loop instead of reading `phi` off the final state:

```
# inside step_fn, after collide:
J_acc = J_acc + (rho / N) * sum(magic(state.logh_acc) * r(state.x, state.v))
```

Verified bit-identical to the explicit `sum_k magic(L_{<=k}) r_k` form (gradient
`215.215669837607` both ways), and the primal value equals the plain windowed objective
`sum_k r_k` exactly, so the magic box stays value-neutral. The terminal objective is the special
case where `r` is zero except at `k = M`.

> **How much does causality actually buy?** Less than "architectural" suggests, and this is worth
> stating because it sets expectations. Measured (`scratch/probe_scan_dice.py`) on a stationary
> sampling window with a closed-form analytic gradient, both estimators unbiased (all `|z| < 1.2`):
>
> | M | Var(naive)/Var(causal) | SE ratio |
> |---|---|---|
> | 8 | 2.31 | 1.52 |
> | 128 | 3.03 | 1.74 |
> | 1024 | 3.02 | 1.74 |
> | 4096 | 2.99 | 1.73 |
>
> The ratio **saturates at ~3x and is flat over three decades of horizon** (fitted `M^-0.005`);
> with the §11.4 baseline it saturates at 4x, and on a non-stationary random-walk toy at 1.85x.
> So the penalty is a **bounded ~1.7-2x in gradient standard error, ~3-4x in sample count** — a
> fixed constant, *not* a horizon-dependent blow-up. We adopt the causal form because it is
> nearly free and touches the carry signature (so it is cheaper to settle now than retrofit),
> not because the naive form degrades as the window grows. It does not.

### 11.6 Entry point

```
@partial(jit, static_argnums=STATIC)
def run(params, cfg_static, base_key):
    state = initialize.create_particles(base_key, params, ...)
    carry = (state, CollideState(vremax_init(...)), 0.0)   # trailing scalar: J_acc
    body  = partial(_step_fn, grid, species, cfg, base_key)   # flat, per 0.6
    carry, hist = scan(checkpoint(body, policy=cfg.remat_policy),
                       carry, arange(cfg.n_steps))
    _, _, J = carry
    return J, hist

value_and_grad(run)(params, ...)
```

No hand-written backward pass exists in the production path.

---

## 12. `jax_dsmc/simulate.py` — the time loop

```
def _step_fn(grid, species, cfg, base_key, carry, k):     # module level, per 0.6
    state, cstate, J_acc = carry
    state         = move(state, grid, cfg, k, base_key)          # STAGE_MOVE/WALL
    table         = sort.build_table(state, grid)
    state, cstate, diag = collide(state, table, cstate, grid, species, cfg, k, base_key)
    # causal DiCE (11.5): reward is scored against the score prefix L_{<=k} only
    J_acc = J_acc + (cfg.rho / cfg.N) * sum(magic(state.logh_acc) * r(state.x, state.v))
    return (state, cstate, J_acc), sample.moments(state, table, grid, cfg, species)
```

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

- `jit` with `donate_argnums` on the state so buffers are reused in place. At `N ~ 1e9` this is
  the difference between fitting in HBM and not.
- `jax.checkpoint` with a configurable policy wraps `step_fn`. It is a no-op in the forward pass,
  so it costs nothing until a gradient is taken.
- **No `while_loop` anywhere.** Every loop is a `scan` or `fori_loop` with a static trip count:
  the time loop (`n_steps`), the attempt-slot loop (`A_max`), the wall-bounce loop (`K_bounce`),
  and the inflow rejection sampler when it arrives (M5). This is a hard invariant, checked by a
  jaxpr audit in the test suite.

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

Correctness across dimensions is not sufficient; the plan requires these to be first-class tests.

| Test | Asserts |
|---|---|
| zero recompilation | repeated same-shape calls do not grow the jit cache — catches Python branching on traced values |
| jaxpr audit | no `while_loop` with a data-dependent trip count; jaxpr identical across particle counts at fixed capacity |
| `vmap` composability | `vmap(step_fn)` over an ensemble runs — strong evidence of purity and static shapes, and directly reusable for M8's ensembles |
| throughput scaling | doubling `N` doubles wall time within a band; doubling `C` at fixed `N` is roughly flat |
| compile-time budget | `jit(run)` compiles under a threshold — the `A_max` scan plus checkpointing is the likeliest blow-up |
| peak memory vs `(N, M)` | reported for the reverse pass; makes the checkpointing-vs-reversibility decision data-driven |
| device placement | no host round-trips inside the step |

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

## 16. Open items to settle during implementation

1. **`A_max` default.** Needs a measurement: run the M3 3D box at target density and report the
   distribution of `nattempt` per cell. Set `A_max` at a high percentile plus margin, and keep
   the clipping diagnostic loud.
2. **`vremax` reset cadence — settled: reset at run start only.** SPARTA supports
   `collide_modify vremax Nevery startflag` (`collide.cpp:2353-2361`); [DF] never decays it.
   Revision 2 worried that a monotone, never-decaying `Sigma` would inflate after a transient and
   *degrade the gradient*. **Measured, that worry is backwards** (7.2.2): the most inflated
   `Sigma` tested had the *lowest* gradient variance, and bias was flat. Steady-state inflation
   from merely seeing more samples is mild — 1.40x over 1e4 -> 1e8 attempts.
   The real costs of inflation are (a) proportional compute, `2.29x` after a `T: 1->4->1` pulse,
   and (b) the silent `A_max` clipping hazard in 8.2. Neither is a reason to add periodic resets;
   both are reasons to size `A_max` generously and keep the two diagnostics loud.
3. **Counting sort at scale.** `argsort` is fine to M4; profile at M5.
4. **Environment.** Use `source ~/activate_jax_env.sh` (`~/envs/jax-env`). Verified: JAX 0.6.2 on
   `CudaDevice(id=0)` with `jax_enable_x64`, NumPy 2.2.6, SciPy 1.15.3, h5py 3.16.0, matplotlib,
   pytest. The system Python is *not* usable — its jaxlib fails to load cuFFT and falls back to
   CPU, which would make the performance suite meaningless.

### 16.1 Reference cases adopted as validation targets

[DF] ships six self-contained cases under `run/hyStrath/dsmcFoam+/` that map onto our milestone
ladder, several with published references:

| [DF] case | Physics | Milestone |
|---|---|---|
| `heatBath-5species` | 0-D relaxation, single cell, all-specular walls | **M1** |
| `hypersonicCorner` | supersonic corner, Ar — Bird (1994) §16.2, Scanlon et al. (2010) | M3 |
| `supersonicFlatPlate` | leading-edge flat plate — Bird (1994) §14.3 | M3/M4 |
| `couette_N2-O2` | planar Couette between heated plates, one moving | **M4** |
| `orion107kmNR` | Orion capsule at 107 km from STL, non-reacting air | **M5** |
| `axisymmetricFlatnosedCylinder` | axisymmetric, radial weighting | deferred (needs RWF) |

`hypersonicCorner` and `supersonicFlatPlate` are the valuable additions: literature-validated
problems with published profiles, giving M3/M4 an external check beyond the analytic BKW and
Chapman-Enskog gates.

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

### 18.5 Collision-count rounding — we follow SPARTA's *default*, not its `remainflag` path

Revision 2 transcribed SPARTA's `remainflag` carry scheme and adopted [DF]'s `U[0,1)` seeding of
`collisionSelectionRemainder` (`dsmcCloud.C:105-116`) to remove the resulting startup transient.

Revision 3 **drops the carry entirely.** SPARTA has three schemes
(`collide_vss.cpp:137-161`) and its *default* — neither `mcflag` nor `remainflag` — is
`n = floor(f + U)`, plain stochastic rounding. That is what we take, because the deterministic
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
| S3 | Attempt-count rounding | three paths; the `remainflag` path carries a deterministic remainder, `collide_vss.cpp:137-161` | SPARTA's **default** path, stochastic rounding `n = floor(f + U)`, plus the score `log P(n\|f)` | **GRAD** | Under a deterministic carry `P(n\|f,remain)` is degenerate, so `dt`, `fnum` and `cell_volume` have **no gradient path at all** — measured as exactly `[0,0,0]`. §8.1, §18.5 |
| S4 | Partner resampling | resamples until `i != j`, `collide.cpp:530-533` | closed form `j = (i + 1 + r) mod npc` | **JAX** | A `while_loop` with a data-dependent trip count is neither reverse-differentiable nor statically shaped. Uniformity on the complement verified by chi-square. §8.3 |
| S5 | Empty/degenerate cells | not reachable — linked lists never present an empty cell to the pair chooser | `npc` clamped to `>= 2`, plus a scratch row at index `N` | **MEAS**, JAX | `jax.random.randint(k, shape, 0, 0)` returns 0 **silently** (2000/2000 seeds), so empty cells indexed into a neighbour's particles: 176/300 runs violated momentum, with no error and no `NaN`. §8.4.1 |
| S6 | `vr2` clamp | `vr2 ** (1-omega)` unguarded, `collide_vss.cpp:222` | `safe_pow` with a **relative** clamp `EPS_REL_VR2 * (2kT_ref/m_r)` | **GRAD** | Unclamped, `dq/d\|vr\|` is unbounded (~`2.5e16`). Relative, not absolute: the bound becomes exactly `2\|1-omega\| eps^(0.5-omega)`. Gate U1, §2.3/§17.4 |
| S7 | Scattering-frame guard | `if (d > 1.0e-6)`, `collide_vss.cpp:429` | `d > EPS_D * vrm` — **relative** — via the double-`where` pattern | **GRAD**, MEAS | SPARTA's test is *dimensional* and only works because it runs SI with thermal speeds ~1e2 m/s. Measured an **order-unity** change in post-collision velocity from a change of units alone. §7.4 |
| S8 | `alpha == 1` fast path | `if (fabs(alpha_r - 1.0) < 0.001)` branch, `collide_vss.cpp:405+` | dropped; the general branch is used always | **JAX** | `alpha` is a traced parameter, so a Python branch on it is illegal. The general form is already correct at `alpha = 1`; SPARTA's case is a speed shortcut, not different physics. §7.4 |
| S9 | Cell occupancy | per-cell linked lists, `collide.cpp:508-512` | sort: `argsort` + `bincount` + exclusive scan | **GPU** | Linked lists are a CPU structure. A sort is the right GPU primitive and its permutation is a frozen integer decision, so gathers stay differentiable. §6 |
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

`~/jax_fluids/JAXFLUIDS/` — a mature JAX continuum CFD solver (finite-volume compressible
Navier-Stokes with level-set interfaces). It solves a *different equation*, so none of its physics
transfers; what transfers is **how a large differentiable JAX solver is structured**, and one
observation is directly load-bearing for us:

**It drives its time loop from host-side Python.** `simulation_manager.py:325` is a plain `while`
over steps, with the step body jitted (`@partial(jax.jit, static_argnums=...)`, line 1234) — not a
`lax.scan` over the whole run. That is independent confirmation that per-step host control is
affordable in production JAX CFD, which is exactly what `simulate.run_adaptive` (§8.5.1) relies on
to resize `A_max` between chunks. We had reasoned our way to that pattern; it is reassuring to find
it is the norm rather than a workaround.

Not yet examined, and worth a pass before M5 and before any multi-device work:
`halos/halo_manager.py` and `domain/domain_information.py` (domain decomposition and halo
exchange — the shape of the problem we will have when sharding particles), and
`initialization/` (how a large parameter/state tree is assembled without losing the traced/static
split). Flagged rather than done: a careful read is its own task.

## 19. Unverified claims register

Everything in §17 was measured. **Everything below was not.** These are assertions that are
reasoned but untested, or measured too weakly to rely on. Each is pinned to the milestone that
first makes it live, and each has a stated gate that must pass **before** that milestone's
deliverable is written — not after.

The rule: if a row's trigger milestone is starting, its gate is part of that milestone's
definition of done. A row may only be deleted by moving its evidence into §17.

| # | Claim asserted | Evidence today | Gate that would settle it | Live at |
|---|---|---|---|---|
| ~~U1~~ **CLOSED 2026-09-04** | `EPS_REL_VR2` bounds the collision-rate cotangent, and `1e-8` is the right value | **Both halves verified; value KEPT at `1e-8`.** Bound exactly tight (ratio 1.0000) on `d/d|vr|`. The proposal to raise it to `1e-6` was measured and **rejected** — see §17.4 | Gate run: `scratch/gate_U1_clamp_bound.py`, `scratch/gate_U1_variance_decider.py`; regression in `tests/test_gaps.py` | done |
| U2 | Score estimator is unbiased at a **non-zero** gradient | **Half-closed.** At a *known zero* it passes: §23.3.7's collision-invariant check gives `\|z\| <= 1.53` for `alpha`/`diam`/`omega` against an exact 0. The SE target is also met — `diam` reaches **1.2%** SE at `N = 2.56e5, M_s = 100`, with its mean stable to ~10% over a 64x range in `N`, and the `N^-1/2` law is confirmed (§23.3). What remains open is unbiasedness at a *non-zero* value, and the obstacle is that **the obvious reference is not a reference**: mode-B FD at [P2]'s `Delta m = 0.1` is `O(h^2)` truncation-biased, measured at `\|z\| = 6.4` for `diam`. An estimator cannot be shown unbiased against a biased yardstick | Sweep `Delta m` at fixed large `M_s` and **Richardson-extrapolate the FD to `h -> 0`** to remove the `O(h^2)` bias, *then* require `\|z\| < 2` against the scored AD at an SE below 10% of the mean; report the sample count needed | **M6** |
| ~~U3~~ **CLOSED 2026-09-04** | A 2-5x safety factor on `Sigma` eliminates majorant violations | **Achievable — but NOT with SPARTA's seed.** SPARTA seed + stale-max ratchet FAILS (violation rate `1.1e-3` equilibrium, `1.0e-2` beams; min margin `0.62`). The a-priori thermal majorant passes: **zero** violations in all 6 IC x transient cases, min margin `2.15`. This is a **design change to `vremax_init`** — see §17.5 | Gate run: `scratch/gate_U3_margin_numpy.py`; regression in `tests/test_gaps.py` | done |
| U4 | Sizing `A_max` from the inflated `Sigma` prevents silent clipping | The 2.29x inflation is measured; **the mitigation is not.** No test shows a cold-start `A_max` actually clips, nor that the rule prevents it | Drive a `T: 1->4->1` transient; assert `attempts_clipped > 0` with cold-start sizing and `== 0` with the inflated rule | **M3** |
| U5 | Oracle 3 discriminates the intra-step ordering | **CLOSED.** Built as `tests/test_adjoint_oracle.py::test_oracle3_*` (§13.3). The 2x2 discriminator is exactly diagonal: our `move -> collide` loop matches the OUTSIDE form to **1.33e-16** and misses [P3]'s INSIDE form by **9.85e-02**; reordering the loop swaps both. `B_k` is [P1]'s analytic `A^T`, not `jax.jacobian`. A companion test pins the *production* step's ordering behaviourally, so the oracle cannot end up validating a loop we no longer run | — (closed) | **M2** |
| U6 | `logh_acc` is zeroed when inflow refills a dead slot | **CLOSED.** `inflow.revive_slots` is the only sanctioned path from `alive=False` to `True` and zeroes every `ACCUMULATOR_FIELDS` entry in the same update; the field list is checked by **reflection** over `ParticleState`, so a later accumulator cannot be silently missed. A revived slot's gradient equals a fresh particle's exactly, and skipping the reset shifts it by exactly `sum(inherited * phi[dead])/N`, matched to `1e-10`. **The obvious test proves nothing**: `magic(L) = exp(L - sg(L))` has gradient `dL/dtheta`, so a stale *constant* differentiates to zero — the accumulator has to be made `theta`-dependent for the check to have any power | — (closed) | **M5** |
| U7 | Non-uniform `fnum` without reweighting breaks mass conservation | The *hazard* is reasoned (§3.3); neither the failure nor the guard is demonstrated | Run non-uniform `fnum` with particles crossing cells; exhibit the mass drift; assert the setup guard fires | **M5** |
| U8 | `h_crit ~ 5.8/(N*C*M)` closes the FD window at production scale | Fitted over `N*C*M <= 3e6`; the M3-scale conclusion is a **6-decade extrapolation** | Measure `h_crit` at one intermediate point (`N*C*M ~ 1e8`) and check the fit holds before relying on the extrapolation | **M3** |
| U10 | The `v_r`-frame and lab-frame collision reparameterisations are equally good pathwise estimators | **Only unbiasedness is argued, never variance.** §17.1.1 measures that they are *different* maps (rank 6 vs 4); nothing measures which produces a noisier gradient, and §23.3 shows variance is the binding constraint | Estimate `Var(dJ/dtheta)` over seeds under both samplers at matched cost, 0-D and 1-D; if the lab-frame form wins materially, it is a cheap forward-solve change | **M6** |
| U9 | Scratch-row fix costs ~3-10% | Measured at 3 sizes; **the smallest is noise** (ratios 1.02/1.40/1.69, once 0.14). Only 2 points are real | Re-benchmark with a repeat count and report a spread, at sizes where the kernel is not launch-bound | **M3** |

### 19.1 Two of these are guards against silent failure

U4, U6 and U7 share a property that makes them worse than ordinary untested claims: **the
failure they prevent produces no error, no `NaN`, and no diagnostic.** A clipped attempt count
under-collides quietly; a stale `logh_acc` contributes a plausible-looking wrong gradient;
non-uniform `fnum` loses mass without any conservation check noticing.

This document's two most serious defects — §8.1 and §8.4.1 — were both exactly that shape, and
both survived a careful read. **Reasoning is not evidence for this class.** U4, U6 and U7 should
be treated as blocking for their milestones rather than as nice-to-have.


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

Every per-cell quantity DSMC reports carries a Monte-Carlo floor set by the particles in that
cell. It is not scatter that averages away over time: it is a **positive bias** of size
`1.5 v_th^2 / npc` on the fluctuation kinetic energy, present whatever the flow is doing. Two
places in M2 depend on handling it, and they reach **opposite** conclusions, which is the point
of this section.

### 22.1 Subtracting it is safe on a scalar

The 2-D Taylor-Green viscosity gate fits `E(t) = E0 exp(-4 nu k^2 t)`. The floor is constant in
time while the signal decays exponentially, so leaving it in flattens the curve: the fitted rate
is biased low, the reported viscosity is too small, and *the fit still looks excellent*, because a
decaying exponential plus a constant is smooth. `energy_history` subtracts it, computed from the
actual per-cell counts rather than nominal `npc`. This is safe because the fit window is chosen
where the signal remains well above the floor, and the result is +0.38 % against Chapman-Enskog.

### 22.2 Subtracting it is NOT safe under a logarithm near the crossover

The same reasoning applied to `E(k)` fails, and fails silently. Above the crossover
`E(k) - noise` is a difference of two comparable numbers that fluctuates through zero, so roughly
half the shells pass any positivity test by chance, and the fitted slope is determined by *which
shells happened to survive* rather than by the flow. Measured on the 1-D case:

| treatment | quoted range | fitted slope | truth |
|---|---|---|---|
| floor subtracted | k = 2..507 | **-0.71** | -2 (Burgers) |
| 10x above floor, no subtraction | k = 2..5 | (too narrow to quote) | -2 |

The first row looked entirely healthy and no diagnostic fired. **The general rule: a bias
correction is safe where the signal is large compared with the bias, and dangerous where a
nonlinear function (here `log`) is applied near the point at which the corrected quantity passes
through zero.**

### 22.3 The right fix is to lower the floor

`1.5 v_th^2 / npc` falls only as `1/npc`, so:

* raise `npc`; or
* **ensemble-average the velocity field over independent seeds before transforming**. The
  turbulence initial conditions are deterministic, so every realisation evolves the same mean flow
  and differs only in collision noise. Averaging `M` fields is exactly equivalent to `M * npc` at
  identical total cost, and — unlike raising `npc` — it needs no communication at all between the
  realisations, so it is the natural thing to spread across a multi-GPU node.
  `examples.turbulence.case.run_ensemble` does this and sets `info["n_ensemble"]`, which is what
  lets `report_spectrum` lower the quoted floor accordingly.

### 22.4 It was measured, and the floor was not the binding limit

The 1-D case was rerun with `--ensemble 16` (npc 200, so `npc_eff` = 3200, 3.6 h wall). The floor
fell by exactly 16x, 8.33e-1 -> 5.21e-2, as predicted. Three things came out of it, and the second
two were *not* what the sizing calculation in 22.3 anticipated.

**It confirms 22.2 directly.** Under averaging the low-`k` shells did not move while the high-`k`
shells fell by ~16x — the signature of pure noise:

| shell | E, 1 seed | E, 16-seed mean | ratio |
|---|---|---|---|
| k = 2 | 1.86e+2 | 1.82e+2 | 1.0 |
| k = 5 | 1.19e+1 | 1.18e+1 | 1.0 |
| k = 8 | 2.13e+0 | 1.42e-1 | 15 |
| k = 13 | 1.49e+0 | 1.84e-2 | 81 |

Everything the single-seed run reported above `k ~ 5` was sampling noise, so the -2.45 slope quoted
from it "over k = 2..9, above 3x the floor" measured nothing at all.

**The spectrum has a factor-12 odd/even comb, and it is physical.** The single-mode initial
condition `u = u0 sin(k x)` is antisymmetric under `x -> x + L/2`; the nonlinearity breaks that
symmetry but only weakly over 1.5 eddy times, so odd harmonics are strongly populated and even
harmonics are not:

| k | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|
| E(k) | 1.18e+1 | 3.96e-1 | 4.69e+0 | 1.42e-1 | 1.82e+0 |
| E/floor | 226 | 7.6 | 90 | 2.7 | 35 |

`E(7)/E(6) = 11.9` and `E(9)/E(8) = 12.8`, at shells standing 90x and 35x clear of the floor — so
this is structure, not scatter. **A shell-averaged slope cannot be fitted through a factor-12 comb**,
and which shells a threshold happens to admit then decides the answer. This, not the floor and not
`Re`, is what breaks the contiguous window at `k = 6`. It is a defect of the *initial condition*:
a single Fourier mode imprints its own symmetry on `E(k)`. A spectral test needs a broadband IC
(random phases over a band of low `k`), or in 1-D a single right-going simple wave, which steepens
into one sawtooth with no counter-propagating partner to interfere with.

**Underneath the comb there is a viscous cutoff at `k ~ 10`.** Tracking the odd family alone, which
is clean to `k = 9`, `E(k)` follows `k^-2` within a factor of 2.5 out to `k = 9` and then collapses:
`E(11)` is 6x below the power-law continuation and `E(13)` is 38x below, while both remain within
reach of the floor. At `Re = 61.1` that gives `k_d ~ Re/6`.

`validate.dissipation_wavenumber` reports `k_d ~ C_D Re^p`, `p` = 1, 1/2, 3/4 for D = 1, 2, 3.
The exponents are standard; `C_1 = 1/6` is the measurement above and `C_2 = C_3 = 1` are
placeholders. **The 1-D constant must not be exported**: an earlier revision applied `1/8` in all
three dimensions and the reach table then reported "NO RANGE" for every H200 configuration, which
contradicts the standard DNS resolution requirement `k_d/k_0 ~ Re^(3/4)`. With the placeholders,
4 x H200 gives `k_d ~ 39` in 2-D and `k_d ~ 23` in 3-D — short but real inertial ranges.

### 22.5 Two rules for choosing a fit window, both learned the hard way

`validate.clean_window` enforces both. Neither is fussiness; each changed a headline number.

1. **Contiguous from `k = 1`.** Selecting every shell that passes `E > 10 * noise` produces a
   ragged window whose membership is decided by the data being fitted. Two ways that goes wrong,
   and the 1-D run supplies both: near the crossover the shells that pass are the ones whose noise
   happened to add (22.2's failure mode, wearing a threshold instead of a subtraction), and where
   the spectrum has real structure the threshold samples that structure — here it admitted the
   comb's maxima `k = 7, 9` and rejected its minima `k = 6, 8`, reporting **-3.06** against the
   contiguous window's **-1.76**. Neither number means anything; only the second is honest about
   how little range there is.
2. **Above the spectral peak.** An inertial or Burgers range lies *below* the energy-containing
   scales by definition, so a window starting at the box mode is not one. Here the peak is at
   `k = 2` and the first comb minimum is at `k = 6`, leaving three shells — which is the correct
   verdict of *no measurable range*, rather than the plausible-looking "-1.76, expected -2".

The general shape of every error in 22.2, 22.4 and 22.5 is the same: **a window selected using
the data being fitted will report a slope whatever the data is.** The window has to come from
somewhere else — contiguity, the spectral peak, and `Re`.

**Open, for the 2-D and 3-D node runs:** the initial condition needs replacing before a spectrum
is quoted from any dimension. Taylor-Green stays the default for the *viscosity* gate, where being
an exact Navier-Stokes solution is the whole point; the *spectral* gate needs a broadband IC with
random phases so that no single mode's symmetry survives into `E(k)`. Until that exists, 22.4's
comb should be expected in 2-D and 3-D too.

### 22.6 A control must vary one thing

22.1 subtracts the floor and 22.2 does not, so the 2-D gate reports the viscosity both
ways and checks they agree — the correction is then verified rather than assumed. The
first version of that check compared *corrected over the full decay* against
*uncorrected over the clean window*, which varies two things at once, and it duly
reported a 7.9 point disagreement as a correction failure when most of the gap was the
extra late-time range (where an acoustic transient and the approach to the floor both
live). `report_2d` now fits three times: uncorrected on the clean window, corrected on
the **same** window, and corrected on the full decay. The first pair isolates the
correction; the second pair isolates the window.

This is worth stating as a rule because the same mistake has now appeared three times in
this section wearing different clothes — a threshold-selected spectral window (22.5), a
floor-subtracted spectrum (22.2), and a two-variable control (here). **When a diagnostic
compares two numbers to decide whether something is trustworthy, check that exactly one
thing differs between them.**

### 22.7 The 1-D spectrum is limited by how many modes a shell holds, and that is two

Both fixes from 22.4 were built and run: a broadband IC (`--ic broadband`, random phases over
`1 <= |k| <= k_max`, solenoidal in 2-D/3-D) and `Ma = 1.0`, which buys `Re = 204` at *lower* cost
than a larger box, since a shorter eddy time means fewer steps for the same `--n-eddy`. Both
worked, and the gate still cannot be established.

| | Re = 61, single mode, npc_eff 3200 | Re = 204, broadband, npc_eff 1600 |
|---|---|---|
| `E(0)` / floor | 4.9 | **885** |
| `E(k)` / floor, peak shell | 3,500 | **22,000** |
| contiguous clean range above the band | k = 3..5 | **k = 5..19** |
| slope | not quotable | **-2.87 +- 0.65** |
| vs Burgers `-2` | — | **1.3 sigma: consistent, but not a test** |

The floor is now irrelevant — the signal stands four orders of magnitude above it — and the
remaining scatter is irreducible by particle count. **In 1-D a shell holds exactly two modes,
`k` and `-k`, at every `k`**: one complex amplitude, so shell-averaging averages nothing and
`E(k)` inherits the full scatter of a single realisation. For a Gaussian field that is an
exponential distribution with `std(log E) = pi/sqrt(6) = 1.28`; measured on this run, **1.00**.
2-D and 3-D are far better off — a 128^2 grid has 28 modes in shell 5 and 116 in shell 19, and a
32^3 grid has 350 in shell 5.

This makes `--ensemble` the **wrong ensemble** for a 1-D spectrum. It averages *fields* across
noise seeds of the *same* flow, which lowers the particle-noise floor and leaves the realisation
untouched; the Re = 204 run spent 2.3 h driving a floor already 1600x below the signal.
`run_ic_ensemble` / `--ic-ensemble R` varies `ic_phase_seed` instead, so each realisation is a
different flow, and averaging their *spectra* reduces the scatter as `1/sqrt(R)` --
calibrated against synthetic exponential-noise spectra in
`test_slope_stderr_matches_the_scatter_it_claims_to_describe` and
`test_averaging_independent_realisations_shrinks_the_slope_error`.

**Report the error bar, always.** `fit_slope` returns one, and the bare slopes over sub-windows of
this run are `-1.69`, `-3.07`, `-2.87` — a spread that looks like a real disagreement and is
entirely explained by `+-0.65` to `+-1.11`.

**What the 1-D gate would actually need.** `R = 11` realisations reach `+-0.20`, but the window
top `k = 19` is only 1.8x below the viscous cutoff `k_d = 34`, so it would still be viscously
contaminated. A clean Burgers range needs the window an order of magnitude below `k_d`, hence
`Re >~ 1000`, hence `side ~ 5000` at `Ma = 1` — minutes on an H200, ~20 h per realisation on a
laptop 3050 Ti. **Not attempted here, and not claimed.**

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

`python3 -m examples.shock_adjoint --mode pathwise`, 7-point sweeps spanning +-15% of Bird's
Mach-2 argon reference, `h` re-bisected at every point until the decision checksum matches:

| D | `N*C*M` | `h*` range | `temp` | `vstream` | `alpha` | `alpha`, `err/medAD` |
|---|---|---|---|---|---|---|
| 1 | 1.3e6 | 1.6e-7 - 1.0e-5 | 9.3e-10 | 8.1e-11 | 1.7e-7 | **6.7e-8** |
| 2 | 5.1e6 | 1.6e-7 - 1.0e-5 | 4.4e-10 | 3.0e-10 | 2.0e-6 | **3.2e-7** |
| 3 | 2.0e7 | 3.1e-7 - 1.0e-5 | 3.1e-10 | 3.0e-10 | 4.8e-7 | 4.8e-7 |

(max relative error over the sweep; all nine **PASS**, at `h0 = 1e-5`. The superseded `h0 = 1e-4`
run read `8.1e-10 / 3.3e-11 / 4.4e-6`, `1.7e-10 / 3.0e-10 / 3.1e-6`, `4.1e-10 / 5.0e-10 /
7.7e-7`. `temp` and `vstream` are unchanged under `err/medAD` to two digits — `9.5e-10 / 9.3e-11`,
`4.4e-10 / 3.1e-10`, `3.1e-10 / 3.3e-10` — which is the control the new column needed: it moves
only where a stationary point exists.)

> **CORRECTION, in two parts.** The first `alpha` figures were explained as the cancellation
> floor of a small derivative on a large objective. **That was wrong**, and the
> `|FD - AD| / FD-noise` column is what exposed it: `alpha` sat at **40-1853x** its floor while
> `temp` and `vstream` sat at 0.7-6.1.
>
> **Part 1 - truncation, above `h ~ 1e-5`.** Perturbing `alpha` flips fewer frozen decisions than
> perturbing a velocity does, so its flip-free window is wide and the bisection stopped at the
> harness's `h0` cap rather than at `h_crit`. Sweeping `h` separates the mechanisms cleanly
> (`scratch/adjoint/probe_alpha.py`, `D = 1`):
>
> | `h` | `\|FD - AD\|` | in units of the FD noise floor |
> |---|---|---|
> | 1.0e-04 | 1.28e-05 | 193.7 |
> | 3.3e-05 | 1.15e-06 | 5.8 |
> | 1.1e-05 | 5.97e-08 | **0.1** |
>
> A clean `O(h^2)`, reaching the floor at `1.1e-05`. The cap is now `1e-5`.
>
> **Part 2 - what survived the fix was NOT truncation, and predicting it would be was also
> wrong.** The stated expectation was a uniform `(1e-5/1e-4)^2 = 100x` improvement into the low
> `1e-8`s. Measured: `D = 1` improved **26x** (`4.4e-6 -> 1.7e-7`), but `D = 2` and `D = 3` moved
> only ~1.6x (`3.1e-6 -> 2.0e-6`, `7.7e-7 -> 4.8e-7`). Across all 21 `alpha` sweep points the
> residual error correlates with the **gradient magnitude**, not with the step:
>
> | correlate | Spearman `rho` | `p` |
> |---|---|---|
> | rel. error vs `\|AD\|` | **-0.835** | `< 1e-4` |
> | rel. error vs `h` | +0.025 | 0.92 |
>
> Splitting at `|AD| = 80`: median relative error `1.7e-7` for the nine points below it against
> `1.2e-9` for the twelve above — a **136x** gap. The mechanism is the reporting metric, not the
> gradient. `alpha`'s sweep crosses **near-stationary points of `J`**, where `|dJ/dalpha|` drops
> to ~10 against a sweep-typical 100-700, and a per-point relative error divides by exactly that
> small number. `temp` and `vstream` never sample such a point — their `|AD|` stays inside
> `[165, 234]` across all 21 of their own points, median relative error `3.8e-11` — which is the
> whole reason they look two decades cleaner. It is not a property of the parameter's gradient
> being worse computed.
>
> `verify.print_pathwise` therefore now reports **`err/medAD`** alongside `max rel err`:
> `|FD - AD|` normalised by the sweep's *median* `|AD|`, which is scale-robust to a stationary
> point (`verify.scaled_error`). PASS is granted on any of the three criteria.
>
> **Part 3 - and a residual that is neither.** The scale-robust column buys `alpha` a factor of
> 2.5 (`D = 1`) to 6.1 (`D = 2`), and nothing at `D = 3`, where the worst point happens to sit at
> the sweep median. So the stationary-point effect is real but partial: `alpha` still sits ~100x
> above `temp` and `vstream` at `6.7e-8 / 3.2e-7 / 4.8e-7` against their `~3e-10`. That residual
> is most likely **higher-derivative content**, and the reason is structural rather than
> numerical. A central difference's truncation is `(h^2/6)|J'''|`, so at a *common relative* `h`
> the parameter with the larger `|J'''|/|J'|` pays more — and the three enter `J` very
> differently:
>
> | parameter | how it enters | nonlinearity |
> |---|---|---|
> | `vstream` | a uniform shift of every `v` | **linear** |
> | `temp` | `sqrt(k T / m)` in the sampler | a square root — mild |
> | `alpha` | `cos chi = 2 u^(1/alpha) - 1`, §7 VSS scattering | `u^(1/alpha)` — **all orders in `1/alpha`** |
>
> This also explains why the error does *not* correlate with `h` within the sweep: after
> bisection `h` spans only 4x (so 16x in `h^2`) while the errors span 1000x, so point-to-point
> variation in `J'''` swamps the `h` lever and no ordering in `h` is visible. It is consistent
> with every measurement above rather than a separate one, and it is **not** a defect: a `2e-6`
> worst-case relative agreement on a chaotic Monte Carlo objective is a pass by any reading.
>
> **Two lessons, and they are different.** (a) "Largest flip-free `h`" is the wrong rule when the
> window is wide: the step must sit between truncation above and cancellation below, and only a
> sweep in `h` tells you which you are looking at. (b) **A per-point relative error is the wrong
> summary for a sweep that crosses a stationary point** - and the tell was that tightening `h`
> fixed one dimension out of three. A fix that works in one place and not two others is evidence
> the diagnosis is incomplete, not evidence it needs re-tuning.

**What this run does *not* show, now that the cap is tightened.** Under `h0 = 1e-5` the top of
every `h*` range above **is the cap**, not `h_crit`, so this run no longer measures the flip-free
window and cannot be read as evidence for §13.4.1's `h_crit ~ 5.8/(N*C*M)`. That evidence comes
from the superseded `h0 = 1e-4` run, and only from two of its three points: at `D = 2` and
`D = 3` the largest flip-free steps were `1.3e-5` and `6.3e-6` against the fit's `1.1e-6` and
`2.9e-7` — above the line, i.e. the fit is conservative — while `D = 1`'s `1.0e-4` was itself
that run's cap and carries no information. Two points above the `3e6` it was fitted over is
weak support; **gap U8 stays open**, and closing it needs a dedicated `h_crit` measurement with
the cap lifted rather than a by-product of the verification sweep.

### 23.3 The score estimator's variance, in the papers' own terms

> **Consolidated tables and figures for all of this: `examples/shock_adjoint/VARIANCE.md`,
> regenerated (no GPU) by `python3 -m examples.shock_adjoint.variance_report`, which
> parses the probe logs rather than restating them.**

**[P2] section 6 reports this variance and budgets for it**; an earlier draft of this section
wrongly said the papers were silent on it. `main.tex:1425` averages **`M_s = 100`** i.i.d. runs
for *both* `nabla_m J^AD` and `nabla_m J^FD`; `main.tex:1431` states the random error as
**`O((sqrt(N M_s) |Delta m|)^-1)`** and fixes **`Delta m = 0.1`** — a **10 percent** finite
difference — as "nearly optimal for `N = 10^6` Maxwellian particles, `M_s = 100`";
`main.tex:1502` reports both standard deviations growing as `O(sqrt(M))` in the step count, and
notes `nabla_m J^AD` is about **10x less noisy than `nabla_m J^FD`**.

So a large step plus a large ensemble is the published protocol, not a workaround, and "mode B"
is simply that protocol.

#### 23.3.1 The `N^-1/2` law holds, measured over a 64x range in `N`

`scratch/adjoint/probe_variance.py` §A. `D = 2`, widened *transversely* so that `npc` — and
therefore `A_max` — is held fixed at 256 while `N` and `C` grow together; `M = 16` steps; 128
base keys per point; `phi = T_y`, which is [P2]'s own objective (§23.3.7 says why the windowed
objective cannot be used for a variance study). `sd` below is the standard deviation of the
gradient over the 128 i.i.d. runs — the per-realisation spread, which is the quantity [P2]'s
`O((sqrt(N M_s)|Delta m|)^-1)` law is about:

| `N` | `C` | `sd(temp)` | `sd(ux)` | `sd(alpha)` | `sd(diam)` |
|---|---|---|---|---|---|
| 4 000 | 80 | 5.451 | 3.756 | 1.961e+03 | 1.351e+13 |
| 16 000 | 320 | 2.931 | 2.964 | 1.005e+03 | 6.599e+12 |
| 64 000 | 1 280 | 1.346 | 1.274 | 5.458e+02 | 3.348e+12 |
| 256 000 | 5 120 | 0.6745 | 0.5842 | 2.441e+02 | 1.686e+12 |
| **fitted `p` in `sd ~ N^p`** | | **-0.508** | **-0.464** | **-0.495** | **-0.499** |

All four exponents land on `-1/2` across a 64x range in `N`, and they do so for a **pathwise**
parameter (`temp`, `ux`), a **mixed** one (`alpha`), and a **pure-score** one (`diam`, whose
pathwise gradient is identically zero, §23.1). The estimator's variance is behaving exactly as
[P2] describes. **The variance is a budget question, not a defect** — which is what U2 asks.

#### 23.3.2 The percentage error bars were misleading, and the reason matters

An earlier draft of this section reported a table of *relative* standard errors measured at
`D = 1`, `N = 2000`, 512 keys with the windowed objective — `temp` 0.2 %, `omega` 22 %, `diam`
32 %, `nrho` 136 %, `alpha` **390 %** — and read the large entries as the estimator struggling.
That reading is wrong for `alpha`, and the `N` sweep above is what exposes it. Ensemble means
with their standard errors, same runs:

| `N` | mean `dJ/d(alpha)` ± SE | mean `dJ/d(diam)` ± SE |
|---|---|---|
| 4 000 | `+164 ± 173` | `1.233e13 ± 1.19e12` |
| 16 000 | `+119 ± 89` | `1.405e13 ± 0.58e12` |
| 64 000 | `-21 ± 48` | `1.322e13 ± 0.30e12` |
| 256 000 | `-8 ± 22` | `1.292e13 ± 0.15e12` |

`diam`'s mean is **stable to ~10 % across a 64x change in `N`** while its error bar shrinks 8x —
the estimator is converging on a well-defined non-zero number, and its relative error falls
9.7 % -> 4.2 % -> 2.2 % -> 1.2 % on the `N^-1/2` law. `alpha`'s mean, by contrast, is consistent
with **zero** at every `N` (`|z|` = 0.9, 1.3, 0.4, 0.4) and the four estimates are mutually
consistent. Its relative error "rises" to 269 % only because the denominator is a number near
zero. **`alpha`'s absolute error bar falls 8x, on the same `N^-1/2` law as everything else.**

So the honest statement is: for this objective and configuration, `E[dJ/d(alpha)]` is small
compared with a single run's spread — `|mean| <~ 30` against `sd = 244` at `N = 2.56e5`, a
per-realisation signal-to-noise of ~0.1 — and 128 runs still do not resolve it from zero. That
is a statement about *this* sensitivity being weak, not about the estimator being unreliable.
Do not confuse it with §23.2's pathwise `alpha` check, which is a different objective at
**frozen** randomness and is exact to `4e-6` relative.

`N * M_s` in the old 512-key probe was `1e6` against [P2]'s `1e8` — **a hundredfold smaller
budget**, which on their own law predicts error bars 10x wider before anything else is
considered. The one parameter [P2] actually differentiates, the initial temperature, is the one
that came out at 0.2 %.

**This is not a correctness question.** Oracle 2 (§13.2) matches [P2] Algorithm 3 exactly, particle
by particle, including all three `eta` branches, and fails when the reject branch is dropped. What
is open is sample efficiency — registered as **U2**.

#### 23.3.3 The noise is mostly **pathwise**, which is why the baseline barely helps

`scratch/adjoint/probe_split_variance.py`, 256 keys, `phi = T_y`. The `sd(score)` column in the
tables above is `sd(full - pathwise)`, which bounds the score *contribution* but cannot attribute
the rest, because the two halves may be correlated. Measuring all three settles it:

| `N` | parameter | `sd(pathwise)` | `sd(score)` | correlation | dominated by |
|---|---|---|---|---|---|
| 16 000 | `temp` | 2.696 | 0.904 | -0.07 | **pathwise** |
| 16 000 | `ux` | 2.519 | 0.535 | +0.19 | **pathwise** |
| 16 000 | `alpha` | 1014 | 197 | -0.10 | **pathwise** |
| 16 000 | `diam` | 0.0 | 6.96e+12 | — | score (necessarily) |
| 64 000 | `temp` | 1.275 | 0.469 | -0.01 | **pathwise** |
| 64 000 | `ux` | 1.095 | 0.263 | +0.08 | **pathwise** |
| 64 000 | `alpha` | 679 | 121 | -0.10 | **pathwise** |
| 64 000 | `diam` | 0.0 | 3.45e+12 | — | score (necessarily) |

The correlations are all within `+-0.2` of zero, so the halves are effectively independent and
`sd(full)^2 = sd(pathwise)^2 + sd(score)^2` (checked: `2.696^2 + 0.904^2 -> 2.84` against a
measured `2.78`). **For every parameter with a pathwise route, that route carries 3-5x more
noise than the score.** `alpha`'s spread is `1014` pathwise against `197` score — previously
*inferred* from `193 << 1005`, now measured directly.

This bears on §11.4's puzzle, though **not** as tidily as a first draft of this section claimed.
Since the halves are uncorrelated their *variances* add, and the score's share of the **variance**
— which is what a baseline removes — is much smaller than its share of the standard deviation:

| parameter | score % of sd | score % of **variance** | ceiling for a perfect baseline |
|---|---|---|---|
| `temp` | 34.7 % | 11.9 % | **1.065x** |
| `ux` | 23.0 % | 5.5 % | **1.029x** |
| `alpha` | 17.8 % | 3.1 % | **1.016x** |
| `diam` | 100 % | 100 % | unbounded — the score is the whole variance |

**The correction.** It is tempting to say this explains §11.4's measured ~1.2x. It does not:
1.2x **exceeds** the 1.016x ceiling computed here, so the two cannot be the same quantity — and
they are not, because §11.4's baseline was measured on a *windowed* `phi` at `D = 1, N = 2000`,
a configuration where the score share must be larger. Quoting one as the explanation of the other
would be a false tidy-up.

What survives is directional and is enough: **wherever a parameter has a pathwise route, the
score is a minority of the gradient variance, so a baseline is correspondingly limited.** The 84x
applies only where the score is the *entire* variance — the count score in isolation, and the
§23.1 score-only parameters such as `diam`.

The practical consequence is the useful one: **for `temp`, `ux` and `alpha` the only lever on the
dominant noise term is more particles**, exactly as the `N^-1/2` law says. No change of
estimator, baseline, or ensemble size touches pathwise Monte Carlo error. Only `diam` and the
other §23.1 score-only parameters are in the regime a baseline or a larger `M_s` improves.

#### 23.3.4 At [P2]'s own protocol, our AD/FD gap is truncation-limited, not noise-limited

`probe_variance.py` §B runs the published protocol as closely as this device allows:
`N = 256 000`, `M_s = 100`, `Delta m = 0.1` (a **10 %** step), giving `N*M_s = 2.6e7` against
[P2]'s `1e8`:

| parameter | AD mean ± SE | FD mean ± SE | `\|z\|` | AD-vs-FD gap |
|---|---|---|---|---|
| `temp` | `2.1247e+02 ± 7.3e-02` | `2.1688e+02 ± 4.0e-01` | 11.0 | 2.1 % |
| `ux` | `-3.6217e+01 ± 9.1e-02` | `-4.8333e+01 ± 2.6e-01` | 43.5 | 33 % |
| `alpha` | `8.67e+01 ± 1.08e+02` | `-2.76e+01 ± 7.2e+01` | 0.88 | — |
| `diam` | `1.2885e+13 ± 1.5e+11` | `1.4656e+13 ± 2.3e+11` | 6.4 | 14 % |

**Those `z` values are not a disagreement between AD and FD — they are `Delta m = 0.1` being a
10 % step.** A central difference carries `O(h^2)` truncation, and `Delta m = 0.1` is "nearly
optimal" in [P2] only because it sits at the minimum of a *truncation-versus-noise* trade-off
evaluated at **their** noise level. Our AD standard error here is 0.03 % (`temp`) to 0.25 %
(`ux`); once the noise floor drops that far, the same step leaves truncation as the dominant
term and it shows up as a large `z`. `ux` is the extreme case for a physical reason: a 10 %
step on the inflow speed moves this case from `Ma = 2.0` to `Ma = 2.2`, and the shock's
structure is strongly nonlinear over that range. **Mode A is the test that resolves AD against
FD** (§23.2, agreement to `1e-9`-`1e-6` at a flip-free step); mode B's step is chosen to make
decisions flip, and cannot be read as a precision comparison.

One published claim does reproduce, partially: [P2] `main.tex:1502` reports AD about **10x** less
noisy than FD. Measured here, AD's standard error is **5.4x** smaller for `temp`, **2.9x** for
`ux` and **1.5x** for `diam` — same direction, smaller factor — while for `alpha` FD is
actually **1.5x tighter**, the one parameter whose mean is consistent with zero.

#### 23.3.5 The spatial window costs signal, not variance

§23.3's earlier draft blamed the windowed objective for wide error bars. `probe_variance.py` §C
tests it directly, same case and keys, `N = 64 000`, 128 keys:

| parameter | `T_y` mean | `T_y` sd | windowed mean | windowed sd | sd ratio | mean ratio | net SNR |
|---|---|---|---|---|---|---|---|
| `temp` | 212.4 | 1.346 | 58.61 | 0.619 | 0.46 | 0.28 | **0.6x** |
| `ux` | -36.05 | 1.274 | -3.578 | 0.326 | 0.26 | 0.099 | **0.39x** |
| `alpha` | -21.1 | 545.8 | -9.889 | 132.1 | 0.24 | 0.47 | 1.9x |
| `diam` | 1.322e13 | 3.348e12 | 8.846e11 | 1.415e12 | 0.42 | 0.067 | **0.16x** |

The window **reduces** the absolute standard deviation in every case — fewer particles carry
weight, so there is less to fluctuate. What it reduces *faster* is the mean. For `diam` the
spread falls 2.4x while the signal falls 15x, a net **6.3x loss of signal-to-noise**; `temp` and
`ux` lose 1.7x and 2.6x. So the earlier suspicion was right in outcome and wrong in mechanism:
the windowed objective is worse to estimate because it **throws away signal**, not because it
adds noise. `alpha` is the exception, and it is the parameter whose mean is consistent with zero
in both, so its ratio is not meaningful.

This is a design note for choosing objectives, not a defect: a spatially localised objective is
a legitimate thing to want, and it simply costs sample budget in proportion to the fraction of
the gas it looks at.

#### 23.3.6 `sd ~ sqrt(M)` does not hold here, and the raw exponent is the wrong thing to compare

[P2] `main.tex:1502` reports both standard deviations growing as `O(sqrt(M))` in the step count.
`probe_variance.py` §D measured the raw `sd` against `M = 4, 8, 16, 32` and got exponents nowhere
near `0.5`. But the raw `sd` cannot settle it: **[P2]'s `M`-sweep is a 0-D relaxation towards
equilibrium**, where the objective saturates and the gradient magnitude stops growing, so their
`sd` growth is pure score accumulation. Ours is a **developing shock** — over `M = 4..32` the
flow is still forming, so `dJ/dm` itself grows with `M` and a steeper `sd` is expected on signal
grounds alone. `scratch/adjoint/probe_M_scaling.py` therefore measures the mean as well
(`D = 2`, `N = 64 000`, 128 keys, `phi = T_y`):

| parameter | `q(sd)` | `q(\|mean\|)` | `q(sd/\|mean\|)` | reading |
|---|---|---|---|---|
| `temp` | 0.291 | **0.017** | +0.273 | signal saturated — the only one in [P2]'s regime |
| `ux` | 1.143 | 1.418 | **-0.275** | signal outgrows noise |
| `alpha` | 1.293 | 1.632 | -0.339 | fit not trustworthy, see below |
| `diam` | **0.720** | **0.721** | **-0.002** | signal and noise grow together, exactly |

(`X ~ M^q`, fitted over `M = 4..32`.)

Three things fall out.

1. **`diam` is the clean case and the result is striking.** Its signal and its noise grow at the
   *same* rate to three digits (`0.720` against `0.721`), so the **relative** precision of the
   score estimator is flat in run length: `q(sd/|mean|) = -0.002`. Integrating longer neither
   helps nor hurts a pure-score gradient here.
2. **`temp` is the only parameter in [P2]'s regime**, with a saturated signal (`q = 0.017` — its
   `T_y` sensitivity relaxes to a constant, as a 0-D relaxation's would). There the raw `sd`
   exponent *is* the comparable number, and it is **0.291**, below [P2]'s `0.5` rather than
   above. So where the comparison is legitimate, our noise grows **more slowly** than published.
3. **For `ux` the relative spread falls** as `M^-0.275`. A longer integration is a better
   gradient, not a worse one.

**`alpha`'s row must not be read.** Its ensemble mean flips sign across the sweep
(`-2.99, +21.1, -21.1, +129.6`), consistent with the near-zero mean established above, so fitting
`log|mean|` on it is meaningless and `q(|mean|) = 1.632` is an artefact of that. Only `temp`,
`ux` and `diam` carry weight here. The lever arm is also short — four points over an 8x range in
`M` — so these exponents are indicative, not tight.

#### 23.3.7 A zero-check with an analytically known answer

There is one objective whose collision-parameter gradient is known exactly, with no finite
difference and no ensemble needed: **`sum |v|^2` is a collision invariant**. VSS conserves the
pair's kinetic energy by construction (`|v_r'| == |v_r|`, `v_cm` untouched) and the specular wall
only flips a sign, so the total kinetic energy of the gas is conserved over the whole run and
**cannot depend on `alpha`, `diam`, `omega` or `tref` at all** — pathwise *or* in expectation.

Measured (`scratch/adjoint/probe_variance.py` §0, `D = 2`, `N = 16 000`, 32 keys):

| parameter | max abs pathwise gradient | scored gradient, mean ± SE | `\|z\|` |
|---|---|---|---|
| `alpha` | **1.0e-12** | `-1.38e+01 ± 5.44e+01` | 0.25 |
| `diam` | **0.0** | `-3.50e+12 ± 2.30e+12` | 1.53 |
| `omega` | **0.0** | `5.54e+02 ± 5.62e+02` | 0.99 |

Both halves pass: the pathwise gradient is machine zero, and the *score* estimator — the noisy
one — is consistent with the true value of exactly zero in every case. That is a genuine
**unbiasedness** check against a ground truth rather than against another estimator, which is
what U2 asks for; it constrains bias at a zero, not the scale of a non-zero gradient.

It is also the reason this objective cannot be used to *measure* variance: the signal is
identically zero. [P2] uses `T_y`, a directional moment that collisions do change, and this is
presumably why. `tests/test_adjoint.py::test_collision_invariant_has_exactly_zero_collision_parameter_gradient`
keeps the check permanently, in all three dimensions.

#### 23.3.8 `alpha` is ill-conditioned *for argon specifically*, and the shock only partly recovers

`alpha`'s ensemble gradient reading zero (§23.3.2) is not an estimator defect. VSS deflection is
`cos chi = 2 u^(1/alpha) - 1`, so the second Legendre moment that drives anisotropy relaxation is

```
<cos^2 chi> = 4a/(a+2) - 4a/(a+1) + 1        d/da <cos^2 chi> = 8/(a+2)^2 - 4/(a+1)^2
```

which vanishes at **exactly `a = sqrt(2)`** (`sqrt(2)(a+1) = a+2`), with `<cos^2 chi> = 1/3` at
both `a = 1` and `a = 2` and a minimum between. Bird Table A3's argon sits at `alpha = 1.40`,
**1.005 %** from that stationary point, so a temperature-like objective has structurally almost
no `alpha` sensitivity there. `jax_dsmc/species_data.py` carries the five monatomic gases and
`alpha_conditioning` as the ratio to argon's.

0-D confirms the mechanism across all five (`examples/p2_replication`, `N = 32 000`, `M_s = 200`):
a single free constant fits every point at `r = 0.9732`, `p = 0.0053`, reduced chi^2 = 0.22.

But 0-D is not the verification target. Measured in the **2-D reflected shock**
(`scratch/adjoint/probe_gas_alpha.py`, 256 keys, `N = 16 000`, `phi = T_y`, `D = 2`):

| gas | `alpha` | analytic cond | shock `AD +- SE` | `\|AD\|/SE` | `\|FD\|/SE` | `SE_FD/SE_AD` |
|---|---|---|---|---|---|---|
| He | 1.26 | 12.6x | `8.60e+02 +- 3.18e+02` | **2.7** | 0.57 | **8.4x** |
| Ar | 1.40 | 1.0x | `1.11e+02 +- 1.39e+02` | 0.8 | 1.45 | 2.2x |

Three readings, in order of how much they change what we do:

1. **The gain is real but heavily discounted.** Helium realises 3.4x against 12.6x analytic,
   where 0-D realised 15.7x. The shock's transport, specular wall and developing flow eat most
   of it. At 2.7 sigma `alpha` is *marginally* usable in the shock, not comfortably so — 10 sigma
   needs ~14x the ensemble (~3500 keys, or `N ~ 2.2e5` at 256 keys).
2. **The finite difference cannot do this at all.** Helium's FD is at 0.57 sigma: its signal is
   below its own noise. So the `\|z\| = 0.24` AD-vs-FD agreement is a **low-power** check and
   must not be quoted as a verification. FD would need ~28x the ensemble to reach 3 sigma.
3. **`SE_FD/SE_AD = 8.4x` is the first sighting of [P2]'s claimed ~10x AD advantage.** The 0-D
   replication measured only 1.7-2.0x on average (§23.3.4). The advantage appears here, where the pathwise
   term dominates — consistent with §23.3.3's finding that pathwise carries 3-5x the score noise
   and is therefore where a method difference can show up at all.

Argon's `\|FD\|/SE = 1.45` exceeding its `\|AD\|/SE = 0.8` is not evidence against AD: argon's
true derivative is ~0, so both numbers are fluctuations about zero.

**Consequence for the milestone gates.** `alpha` stays a mode-A parameter (where it passes at
`2e-6` regardless of gas, §23.2). Any *mode-B* claim about `alpha` must use helium and must state
its ensemble size, and no mode-B `alpha` verification should rest on AD-vs-FD agreement until FD
itself resolves. Figures: `examples/shock_adjoint/figures/gas_{resolvability,shock_alpha}.png`.

### 23.4 Reverse-pass cost: ~4x the forward, not 17x

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
