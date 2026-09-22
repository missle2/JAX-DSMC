# JAX-DSMC — project plan

> Canonical copy. Originally authored as a Claude Code plan file outside the repo; vendored here
> so the repository is self-contained. If you edit it, edit **this** copy.

# JAX-DSMC: differentiable DSMC from scratch

## Context

The repo is empty (README + .gitignore only). The goal is a DSMC solver built in JAX from
the ground up so that a **DTO adjoint** for the inhomogeneous Boltzmann equation with a
general collision operator falls out of the forward solve rather than requiring a separate
implementation.

### References

Two independent production DSMC codes are used as implementation references, and they disagree
in places — those disagreements are the most useful part.

- **SPARTA** (`/home/aakash/sparta/`) — Cartesian hierarchical grid, cut/split cells, linked-list
  cell occupancy. Cited as `file.cpp:lines`.
- **dsmcFoam+ / hyStrath** (`/home/aakash/OpenFOAM-v1706/dsmcFoam+_internals.html`) — arbitrary
  polyhedra, OpenFOAM tet-decomposition tracking, runtime-selected collision/boundary model
  hierarchies. Cited as **[DF]**.

### The adjoint stack (three papers in `papers_reference/`)

| Paper | Regime | Adjoint mechanism | AD-obtainable? |
|---|---|---|---|
| `2009.01363` Caflisch–Silantyev–Yang §4 | 0D, Maxwell molecules | γₖ = B(σ,α)γₖ₊₁, B = Aᵀ | **Yes** — plain `jax.grad` |
| `2207.11579` Yang–Silantyev–Caflisch | 0D, **general kernel** (VHS/VSS) | + `η` score terms from the virtual-collision probabilities | **Yes** — `jax.grad` of a surrogate |
| `2603.20946` Caflisch–Chen–Yang | **spatially inhomogeneous**, general BCs | (β,α) coupled adjoints; thermal needs randomized τ; inflow needs an ensemble adjoint | Partly — see below |

**Finding 1 — the base adjoint *is* reverse-mode AD.** `γ_F = −∂ᵥr(v_F)` is the seed cotangent;
`γₖ = Aᵀγₖ₊₁` is the VJP of the collision map; the non-colliding branch is the identity
Jacobian; `∂_α J = −(1/N)Σγ_{I,i}·∂_α v₀,ᵢ` is the VJP through the reparameterized sampler.
The paper is explicit that pair choice, collision times, and σ are *"determined externally"* —
frozen randomness, which is exactly common-random-numbers plus integer decisions in JAX.

**Finding 2 — the score terms are also AD-obtainable, via a surrogate objective.** For a
general kernel, `2207.11579` eq. `eq:gamma_rules` adds

```
ηᵢᵏ = (ρ/N)·∂log hᵢᵏ/∂vᵢᵏ·(φᵢᴹ + φᵢ₁ᴹ)
```

with `h = q/Σ` on a real collision and `(Σ−q)/Σ` on a virtual-but-not-real one. Carry **one
`(N,)` float accumulator** `L` through the scan (`L[i] += log h; L[i₁] += log h` per attempt)
and define

```
magic(L) = exp(L − stop_grad(L))          # value ≡ 1, gradient ≡ ∇L  (DiCE operator)
J̃ = (ρ/N)·Σᵢ magic(Lᵢ)·φ(vᵢᴹ)
```

`∇J̃ = (ρ/N)Σᵢ[∇φᵢ + φᵢ∇Lᵢ]`. The first term is the pathwise `D` recursion. In the second,
`Lᵢ` and `Lᵢ₁` each accumulated the *same* `log hᵢᵏ`, so the coefficient of `∂log hᵢᵏ/∂vᵢᵏ` is
`φᵢ + φᵢ₁` — exactly `ηᵢᵏ` — which reverse-mode then transports further back through the `D`
matrices. One accumulator and one line in the objective, not a forward rewrite. The magic-box
form is preferred over `Σφ + Σsg(φ)·L` because the primal value stays equal to `J` (so the
objective can be reported directly) and higher-order derivatives stay correct. The same
construction later absorbs the thermal BC's `log pₗ(x,v)` terms.

**Finding 3 — the papers' virtual-collision framework is NTC, and dsmcFoam+ implements it
correctly where SPARTA does not.** The papers' three outcomes (no virtual collision /
virtual→real / virtual-not-real) with majorant `Σ` map directly onto NTC with `Σ = vremax`.

But the two codes disagree on `Σ`. SPARTA ratchets it *live* inside the candidate loop and tests
against the updated value (`collide_vss.cpp:225-231`). dsmcFoam+ tests every candidate against
the value captured **before** the loop, publishing the new max only for the next step, and
documents why inline (`noTimeCounter.C:231-246`): `N_pairs` was computed from that same old
value, so raising `Σ` mid-loop gives both too few candidates *and* a lower acceptance rate —
under-collision twice over.

**We adopt dsmcFoam+'s stale-max rule.** It is more statistically consistent, and it makes `Σ`
loop-invariant within a step — which is exactly [P2]'s **fixed** majorant, the assumption its
entire score-term derivation rests on. Practical consequences: `vremax` leaves the attempt-scan
carry (simpler and more parallel; the new max is a `max` reduction written once at the end), and
`stop_gradient(Σ)` becomes a clean cross-step statement rather than a patch over an intra-step
ratchet.

**Finding 4 — transport and reflection adjoints are free.** `2603.20946` gives
`βₖ = Bₖ(βₖ₊₁ + Δt·P*αₖ₊₁)` and specular `C = diag([−1,1,1])` applied to both β and α. These
are precisely the VJPs of `x += v·Δt` and the mirror map. Periodic is the identity.

**Finding 5 — two things are genuinely not free**, and are deferred to their own phases:
- **Thermal walls** discontinuously resample velocity at the wall. The paper's fix modifies the
  *forward* solve: `τ ~ N(Δt, ε²)` in advection, making hit/no-hit probabilistic so a score on
  `log pₗ` exists. It computes `∇J_ε`, and `J_ε ≠ J` even in expectation. → **Phase M7.**
- **Inflow** perturbations change cell memberships, particle counts, and collision schedules;
  the paper states a pathwise adjoint *"does not, in general, approximate the derivative of the
  expected DSMC objective"*. Their fix is an ensemble adjoint (adjoint of the **linearized**
  collision operator + a local score on the inflow flux) — different machinery. → **Phase M8.**

So the single hard requirement on every line of the forward solve is: **each stage is a pure
function of (state, params, frozen random draws) with static shapes, and every stochastic
branch exposes a differentiable `log h`.**

### Decisions taken

1. Positions `(N,D)`, D ∈ {1,2,3}; velocities always `(N,3)`, advection uses `v[:, :D]`
   (Bird/SPARTA convention — keeps VSS scattering and Chapman–Enskog transport theory valid).
2. **Uniform** orthogonal grid with traced `lo`/`hi`, static `n`. The full differentiable-mesh
   question is deferred to M5, where cut cells and general tracking force a rewrite anyway —
   but the mesh interface is kept narrow now so that rewrite is contained (see `grid.py`).
3. Estimator: pathwise **plus** the score terms — required, not optional, for VSS. Obtained via
   the surrogate above.
4. Backward pass: **AD-derived** in the production path; Algorithms 2 and 3 hand-coded as
   verification oracles.
5. Gradients flow to IC (T, nrho, vstream), wall models, VSS params (diam, omega, alpha, tref),
   `dt`, `fnum`, and domain extent.
6. Single species, monatomic, no internal energy — shaped `(S,)`/`(S,S)` throughout so
   multi-species, LB, and chemistry are additions, not rewrites.
7. Long-term target N ~ 1e9–1e10 on clusters. Single-device first; layout chosen so sharding
   drops in later. No custom reversible VJP until the basics work.
8. Precision: **f64 throughout** (`jax_enable_x64`). Energy-conservation tests need it.
9. **Stale-max NTC** per [DF] (Finding 3), not SPARTA's live ratchet.
10. **Pluggable pair selection.** NTC-with-replacement (SPARTA / [DF] `noTimeCounter`) is the
    production default. Disjoint-permutation pairing ([DF] `noRepeatCollisions`) is a second
    strategy behind the same interface, used by the 0D oracle tests — the papers pair *without*
    replacement (`E_a^k = Π_{i=1}^{N/2} E_ai^k` is a product over N/2 disjoint pairs), so the
    oracles are only faithful implementations of Algorithms 2 and 3 if they use it.
11. **Gaussian Maxwellian** (`v = sqrt(kT/m)·normal(key,(n,3)) + vstream`, [DF]
    `dsmcCloud.C:1043-1051`) rather than SPARTA's `vn/vr` polar trick. Statistically identical,
    and it removes `sqrt(-log U)` from the singularity inventory. The Rayleigh form survives only
    where flux weighting genuinely requires it (diffuse-wall normal component, inflow).
12. **`fnum` is a per-cell field `(C,)`, not a scalar.** [DF] routes everything through
    `nParticles(cell) = F_N · RWF(cell)`; SPARTA has `cinfo.weight`. Free now, and it is the hook
    for radial weighting and variance reduction later. The RWF clone/delete machinery is **not**
    adopted — it changes particle count and breaks static shapes.
13. ~~**NTC remainder seeded to `U[0,1)`**~~ — *superseded.* Stochastic rounding is unbiased from step one, so the remainder (and the transient this fixed) no longer exists. Kept as the reason we went looking. ([DF] `dsmcCloud.C:105-116`), removing a
    systematic startup transient in the collision count. *(Superseded if stochastic rounding is
    adopted — see the Phase 0 revision-2 review.)*
14. **`ruff` gates the codebase.** Config in `pyproject.toml`, tuned so JAX/physics idioms
    (capital symbol names, `I` for identity, many-argument pure functions, scan-body closures)
    pass clean while real defects are caught — verified against both a JAX-idiomatic sample and a
    deliberately bad one (`B006 E712 E722 F401 F821 F841 I001`). `ruff check` and
    `ruff format --check` must both pass. **No nested function definitions**: `lax.scan` bodies
    are module-level functions bound with `functools.partial`, so they are independently
    testable and their captured state is explicit.
15. **RWF splits in two, and only half is adopted.** The *weight field* (`fnum : f64[C]`, traced)
    is in — it is the expensive half to retrofit, touching the NTC rate, moment normalisation and
    boundary flux together. The *clone/delete machinery* is out: it is not on the critical path
    (every milestone through M5 is Cartesian, where [DF] itself makes RWF a no-op), and its hard
    part — sensitivity when the particle count changes — is the same problem M8's ensemble
    adjoint already has to solve, so retrofitting rides on committed infrastructure.
    **Guard required from the first commit:** a non-uniform `fnum` without reweighting silently
    breaks mass conservation when a particle crosses cells, so `fnum` must be asserted spatially
    uniform in any run with more than one cell.

---

## Phase 0 — Pseudocode spec (before any JAX)

**Status: complete, revision 3, unapproved.** `docs/PSEUDOCODE.md` (~2150 lines) and
`docs/phase0_verification.py` (55 checks, all passing, plus an executable register of 8
unverified claims gated per milestone).

Revision 2 folded in the dsmcFoam+ review. Revision 3 folded in an external review plus three
verification passes that measured every load-bearing claim — fixing two defects that would have
produced silently wrong results (§8.1 zero gradients, §8.4.1 momentum violation) and reversing
two conclusions that had been stated confidently and measured false (§7.2.2, §13.4.1).

**Milestone 1 implementation is now in progress.** See `docs/HANDOFF.md` for state and traps.

The revision-2 edit list below is retained as a record of how the spec evolved.

### Phase 0 revision — spec sections to change

| Spec section | Change | Driver |
|---|---|---|
| §2.1 singularity inventory | drop `sqrt(-log U)` from the init path; it survives only for flux-weighted draws | decision 11 |
| §3.1 shape contract | `fnum` scalar → `(C,)` traced field | decision 12 |
| §5.1 Maxwellian | replace the polar sampler with `sqrt(kT/m)·normal(key,(n,3)) + vstream`; keep the Rayleigh form for diffuse-wall normal and inflow only | decision 11 |
| §7.2 `acceptance_prob` | recast the `stop_gradient` rationale around the stale-max rule — `Σ` is now genuinely fixed within a step, so it becomes a cross-step statement rather than a patch | Finding 3 |
| §8.1 `attempt_count` | `remain` seeded to `U[0,1)` at start-up, not zero | decision 13 |
| §8.3 pair selection | split into `pair_ntc` / `pair_perm` behind one interface; add the "sub-celling not adopted" rationale | decision 10 |
| §8.4 attempt-slot scan | `Σ` read once, out of the scan carry; new max becomes a `max` reduction written at the end | Finding 3 |
| §13.1–13.2 oracles | run under `pair_perm`; add the `rank(A) == 4` structural assertion | decision 10, §17.1 |
| §15 cross-reference | add a [DF] column alongside the SPARTA citations | — |
| §16 open items | add the [DF] reference-case ladder as validation targets | — |
| new §18 | SPARTA vs dsmcFoam+ divergences and which we follow, with reasons | — |

`docs/phase0_verification.py` gains two checks: `pair_perm` yields a valid perfect matching
(every particle used exactly once, no self-pairs), and the stale-max rule leaves `Σ` invariant
across the attempt loop.

### Original scope

The whole system in Python-like pseudocode, no JAX details, covering every module with:

- Exact **shape + dtype contract** for every function signature.
- The **RNG contract** (below), written once and referenced everywhere.
- Which arguments are traced (differentiable leaves) vs. static (shape-determining).
- Every stochastic branch, with its `log h` expression written out explicitly.
- A table mapping each block to its SPARTA source citation and its paper equation number.

Nothing else starts until this is reviewed.

### The RNG contract

```
key_for(base_key, step, STAGE_ID) = fold_in(fold_in(base_key, step), STAGE_ID)
```
`STAGE_ID` is a module-level integer constant. Keys depend **only** on the loop index and
static stage ids — never on traced state values. This guarantees (a) common random numbers for
finite-difference gradient checks, (b) bit-identical replay under `jax.checkpoint`
rematerialization, (c) no gradient path through the RNG. Each stage draws its uniforms in one
batched `jax.random.uniform(key, shape)` call rather than per-element splits.

Worth noting the contrast: [DF] seeds from `clock::getTime() + const·Pstream::myProcNo()`, so
runs are not reproducible unless `seedNumber` is pinned — and even pinned, each MPI rank takes a
distinct offset. Our contract is strictly stronger, and it has to be: the finite-difference
gradient checks are meaningless without bit-identical common random numbers across perturbed
runs.

---

## Phase 1 — Foundations

### `jax_dsmc/safe.py` — **highest-risk module, build first**

DSMC is dense with removable singularities that produce `NaN` gradients even where the primal
is fine. Every one of these appears in SPARTA's VSS path:

| Expression | Singular at | Source |
|---|---|---|
| `sqrt(vr2)` | vr2 = 0 | relative speed |
| `vr2 ** (1-omega)` | vr2 = 0 | `collide_vss.cpp:222` |
| `d = sqrt(vrc[1]²+vrc[2]²)`, then `/d` | collinear vr | `collide_vss.cpp:429-437` |
| `sqrt(1 - cosX²)` | \|cosX\| = 1 | `collide_vss.cpp:417,427` |
| `sqrt(-log(U))` | U = 1 | flux-weighted draws only — see decision 11 |
| `log h` | h → 0 | score term, `2207.11579` eq. `eq:back_eta` |

SPARTA guards the `d` case with `if (d > 1.0e-6)`. The naive JAX translation
(`jnp.where(d > eps, expr_using_d, fallback)`) **still produces NaN gradients** because both
branches are evaluated. Correct pattern, used uniformly:

```python
d_safe = jnp.where(d > eps, d, 1.0)      # denominator made safe FIRST
result  = jnp.where(d > eps, f(d_safe), fallback)
```

Provides `safe_norm`, `safe_div`, `safe_sqrt`, `safe_pow`, `safe_normalize`, `safe_log`.
Tests: `grad` and `grad(grad)` finite at every singular point above, for every function.

### `jax_dsmc/state.py`

(Named `state.py`, not `types.py`: a package-level `types` module shadows the stdlib one
whenever the cwd is `jax_dsmc/`, which breaks the interpreter outright.)

Registered-pytree dataclasses; traced leaves and static fields separated explicitly.

```python
ParticleState:  x (N,D) f64 | v (N,3) f64 | species (N,) i32 | alive (N,) bool
                | logh_acc (N,) f64          # score accumulator L
GridSpec:       lo (D,) f64 [traced] | hi (D,) f64 [traced] | n (D,) ints [STATIC]
SpeciesParams:  mass (S,) | diam,omega,tref,alpha,mr,prefactor (S,S)   [all traced]
SimConfig:      dt [traced] | fnum (C,) [traced] | N, D, S, A_max, K_bounce [STATIC]
CollideState:   vremax (C,G,G)                       [carried across steps; NO `remain`]
```

`n` static because array shapes depend on it. `lo`/`hi` traced.

### `jax_dsmc/grid.py` — **keep this interface narrow**

Uniform orthogonal grid, D-generic. `cell_of(x, grid) -> (N,)` via `floor((x-lo)/dx)`, clip,
`ravel_multi_index`. Plus `cell_volume`, `cell_bounds`, `cell_centers`.

The binning is a frozen discrete decision — `floor` kills the gradient, correctly and by
design (matches the papers treating cell membership as external). Gradients reach the geometry
only through cell *volume* (→ NTC rate → collision frequency) and moment normalization.

**Everything downstream (`sort`, `collide`, `sample`) touches the mesh only through
`cell_of`, `cell_volume`, `cell_bounds`, and `boundary.apply`.** M5 replaces the implementation
behind those four names with cut/split cells and general tracking; nothing else should need to
change. Do not let cell-index arithmetic leak into other modules.

### `jax_dsmc/sampling.py`

Reparameterized so gradients flow to `T` and `vstream` pathwise. Following [DF]
(`dsmcCloud.C:1043-1051`) rather than SPARTA's polar trick:

```
v = sqrt(k T / m) * normal(key, (n, 3)) + vstream
```

Statistically identical to SPARTA's `vn/vr` construction (`create_particles.cpp:761-780`) — both
give three independent `N(0, kT/m)` components — but simpler, trig-free, and it removes the
`sqrt(-log U)` singularity from the initialization path entirely. `d/dT` flows through
`sqrt(kT/m)`; `d/dvstream` is the identity.

The Rayleigh/flux-weighted form is kept **only** where the physics requires it: the diffuse-wall
normal component (`sqrt(-2 ln(1-R))`, [DF] `dsmcDiffuseWallPatch.C:106-111` — note `1-R`, which
also avoids the `R=0` end) and the inflow sampler.

Plus `stochastic_round(x, u) = floor(x) + (u < frac(x))` (`create_particles.cpp:672-677`; [DF]
applies the same trick per-tet spatially) and the flux-weighted inflow sampler
(`fix_emit.cpp:166-174`; [DF] `dsmcFreeStreamInflowPatch.C:91-142`), deferred to M5.

---

## Phase 2 — M1: 0D BKW (collision operator alone)

### `jax_dsmc/collide/vss.py`

`test_collision`, `scatter` translated from `collide_vss.cpp:200-232, 405-452` through
`safe.py`. Post-collision velocities via mass-weighted CM decomposition — momentum conserved
by construction, energy conserved because `|v_r'|` is built from the surviving translational
energy.

Returns `log h` alongside the new velocities: `log(q/Σ)` on accept, `log(1 − q/Σ)` on reject,
with `q = σ_T(v_r)·v_r` and `Σ = vremax`. This is the `2207.11579` `η` term's ingredient.

Note SPARTA builds σ **in the frame aligned to v_r**, so σ depends on the pre-collision
velocities under VSS — the paper's *"additional terms in ∂_{w_k}w_k'"*. AD picks this up free.

### `jax_dsmc/collide/ntc.py`

Attempt count per `collide_vss.cpp:137-161`, by **stochastic rounding** — SPARTA's *default*
path, not the `remainflag` carry:

```
f       = 0.5*np*(np-1)*Σ*dt*fnum/volume
n       = floor(f + U)                       # FROZEN
logP    = log P(n | f)                       # the score term — see below
```

The deterministic carry was the revision-2 form and it is **wrong for the adjoint**: `P(n|f,remain)`
is degenerate, so there is no probability to score, and `dt`, `fnum` and `cell_volume` — which
appear nowhere else — get *no gradient path at all*. Measured as exactly `[0.0, 0.0, 0.0]`. `logP`
is that entire sensitivity, and it is a **per-cell** term applied to every particle in the cell.

**Static-shape strategy.** Total attempts are data-dependent, so pad to a static per-cell bound
`A_max` and mask slots `a >= nattempt`. Emit a diagnostic counter for `max(nattempt) > A_max`
(clipped attempts = silent under-collision) so it is observable rather than a silent physics bug.

**The majorant is read once per step** (decision 9). `Σ` is loop-invariant; the new max is a
`max` reduction over slots, written for the next step. `vremax` is *not* in the scan carry.

**`Σ` is floored by an a-priori thermal majorant, not seeded SPARTA's way** — this is gate U3's
answer, and it is a design decision rather than a tuning knob. A pure stale-max ratchet *lags*: it
can only raise `Σ` after a violation has already been counted, and a violated pair contributes a
zero score and biases the gradient. Seeded SPARTA's way (`2*cxs*vscale`) it measured a `1.0e-2`
violation rate on a counter-streaming IC with `Σ/max q = 0.62`. Since `q` is monotone in `|vr|`,
its supremum over the reachable interval is bounded by the cell temperature, which is known at the
*start* of the step:

```
Σ = stop_gradient(max(vremax_carried, F_UP * prefactor * (K_EV² · 2kT/m_r)^(1-ω)))   # K_EV=5, F_UP=2
```

Zero violations across all six IC × transient cases, minimum margin 2.15. The margin costs
attempts but **not physics** — the collision count is `Σ`-invariant (measured flat over a 20×
sweep), and gradient variance actually *falls* with `Σ`. It is why `A_max` must be sized from the
inflated `Σ`.

**Pair selection is a pluggable strategy** (decision 10), one interface, two implementations:

- `pair_ntc` — with replacement, SPARTA / [DF] `noTimeCounter`, the production default. SPARTA
  resamples until `i != j` (`collide.cpp:530-533`), a data-dependent `while_loop` that is neither
  reverse-differentiable nor statically shaped. Closed-form equivalent, verified uniform on the
  complement of `{i}` by chi-square:
  ```
  i = randint(0, np);   j = (i + 1 + randint(0, np-1)) % np
  ```
- `pair_perm` — disjoint pairing, [DF] `noRepeatCollisions` and the papers' scheme. Simpler than
  the above: `perm = permutation(key, np)`, pairs are `perm[0::2], perm[1::2]`. Used by the 0D
  oracle tests.

**Not adopted: sub-cell partner selection.** [DF]'s default biases selection toward spatially
nearby pairs via 8 octants about the cell centroid (SPARTA offers `partners subcell`/`nearcp`).
It reduces mean collision separation, which is a real accuracy gain at coarse resolution — but it
makes the pair-selection probability **position-dependent**. [P2] §3.1.2 relies on `E_p`
commuting with velocity derivatives precisely because pair selection is independent of the
state; sub-celling breaks that for position derivatives and would require its own score term in
the inhomogeneous adjoint. Recorded as a known accuracy/adjoint-complexity trade-off, not taken.

**Parallelization structure — the central design choice.** `lax.scan` over the attempt slot
`a = 0..A_max-1`, vectorized across all cells at each `a`. Sequential within a cell — slot `a+1`
sees the velocities written by slot `a`, so a particle can collide twice in a step, matching both
reference codes — while fully parallel across cells on GPU. It is `scan`, so it is
reverse-differentiable and rematerializable. Scatter is `v.at[pi].set(...)`; indices are
disjoint across cells so there are no write conflicts.

Under the stale-max rule the scan carry is just `(v, logh_acc)`. `Σ` is read before the loop and
held fixed; the next step's majorant is `max` over the slots' `q` values, masked by
`a < nattempt` so only pairs actually tested contribute — the mask matters because an untested
slot's `q` is meaningless.

Also accumulates `logh_acc.at[pi].add(log_h)` and `.at[pj].add(log_h)`.

### M1 acceptance

Single cell, ω = 1.0 (Maxwell molecules), relax a non-equilibrium IC and compare against the
exact **Bobylev–Krook–Wu** solution. Isolates `collide()` with no grid or move involved, and is
the same regime the papers' numerics use, so the M6 adjoint oracle reuses this setup.
Ships with a matplotlib demo (below).

---

## Phase 3 — M2/M3: transport (1D periodic → 3D box)

### `jax_dsmc/sort.py`

`c = cell_of(x)`; dead particles to sentinel cell `C`; `order = argsort(c)`;
`counts = bincount(c, length=C+1)`; `starts = exclusive_cumsum(counts)`. Particle `j` of cell
`k` is `order[starts[k] + j]`. Gathers are differentiable; the permutation is integer and so
frozen. (SPARTA uses linked lists, `collide.cpp:508-512` — a CPU choice; a sort is the right
GPU primitive. At N ~ 1e9 revisit `argsort` → counting sort.)

### `jax_dsmc/move.py` + `boundary.py`

Uniform grid with no embedded surfaces means SPARTA's cell-walking loop
(`update.cpp:937-1029`) collapses to its own `OPT` fast path (`update.cpp:764-864`):
`x += v[:,:D]*dt`, apply the face BC, recompute the cell by arithmetic. No walk needed.

- **Periodic**: `x = lo + mod(x-lo, L)` — `domain.cpp:301-330`. Adjoint: identity.
- **Specular**: closed-form triangle-wave fold handling *any* number of bounces in one
  expression, `v` sign flipped by fold parity. Exact, differentiable a.e. Replaces
  `domain.cpp:337-357`. Adjoint: `C = diag([−1,1,1])` on both β and α — free from AD.
- **Outflow**: `alive = False` — `domain.cpp:294-295`.
- **Diffuse/thermal** (M4, forward only): bounded `lax.scan` of `K_bounce` masked sub-moves
  carrying `dtremain`; `v_perp = v_rm(T_wall)·sqrt(-log U)` — `surf_collide_diffuse.cpp:206-320`.
  **Not differentiated until M7** (see Finding 5).

The transport adjoint coupling `βₖ = Bₖ(βₖ₊₁ + Δt·P*αₖ₊₁)` is the VJP of `x += v·Δt` and
needs no special handling.

### `jax_dsmc/sample.py`

Per-cell moments via `jax.ops.segment_sum` with static `num_segments`: number density, bulk
velocity, translational temperature, pressure tensor, heat flux. Time-averaged over sampling
steps. Differentiable w.r.t. `v` — this is where objective functionals `r(x,v)` are built.

### `jax_dsmc/output.py` + `jax_dsmc/viz.py`

All four output paths:
- **matplotlib** figures for milestone demos — BKW distribution vs. analytic, moment relaxation,
  Couette profile vs. Chapman–Enskog, gradient-vs-finite-difference validation. Runs in CI and
  doubles as milestone acceptance evidence.
- **ParaView-ready VTK** — not a bare per-step dump. Three pieces, because the naive version
  fails in two ways that only appear when you open the result: ParaView sees N disconnected
  datasets instead of an animation, and there is no way to see the particles at all, only their
  cell averages. See `PSEUDOCODE.md` §10.2.
  - `.vtr` rectilinear grid per step for cell moments (correct and compact on a uniform mesh —
    spacing implied, not stored), mirroring SPARTA's `dump grid`;
  - `.vtp` polydata for the **particles themselves**, deterministically subsampled so the same
    population appears in every frame — a per-frame random subsample turns an animation into
    noise;
  - `.pvd` collection files binding the series together — this is what makes ParaView animate —
    with `.pvtr`/`.pvtp` partitioned variants once the run is sharded.
  Essential from M3 onward. At **M5** the cell fields migrate to `.vtu`, since cut and split cells
  are no longer a rectilinear grid; the `.pvd` and `.vtp` layers are unaffected.
- **HDF5/npz** raw state checkpoints — `x`, `v`, `alive`, cell moments, RNG base key, step —
  for restart, offline analysis, regression baselines, and eventually the frozen-decision record
  if a reversible adjoint is built.
- **Interactive diagnostics** — `summary(state)`: per-cell moment table, `ncollide`/`nattempt`,
  `A_max` clipping count, particles-outside-domain count.

All output is host-side, called outside the jitted step, so nothing perturbs the compiled loop.

### `jax_dsmc/simulate.py`

```python
def step(carry, k):
    state, cstate, base_key = carry
    state = move(state, k, base_key, ...)
    table = sort(state, grid)
    state, cstate = collide(state, table, cstate, k, base_key, ...)
    return (state, cstate, base_key), sample(state, table)

run = lambda init: lax.scan(jax.checkpoint(step, policy=cfg.policy), init, jnp.arange(M))
```

`donate_argnums` on the jitted entry point so state buffers are reused in place — this matters
enormously at target scale.

---

## Phase 4 — M4: 3D Couette

Diffuse moving walls, thermal accommodation, the bounded-bounce move path. Forward only — no
gradients through the wall until M7. Acceptance: shear stress / viscosity against the
Chapman–Enskog VSS formula, and velocity slip against theory.

## Phase 5 — M5: vehicle-like flow

Embedded surfaces, cut/split cells, ray–triangle intersection, inflow BCs. This is where the
mesh layer behind `grid.py`'s four-function interface gets rewritten, and where the
differentiable-mesh question (rectilinear traced faces vs. level set) gets its own design pass —
deliberately deferred so it is designed against real requirements rather than guessed at now.

**Direction settled: SPARTA-style.** A Cartesian hierarchical grid with cut/split cells keeps
cell lookup as closed-form arithmetic (plus a tree descent when refined) and confines geometry to
per-cell ray–triangle tests. [DF]'s alternative — arbitrary polyhedra with OpenFOAM's
tet-decomposition tracker — is more general, but each particle carries four persistent indices
(`celli, facei, tetFacei, tetPti`) and the tracker walks tet-to-tet in a data-dependent loop,
which in JAX means a bounded scan at a worst-case trip count with heavy masking. Structurally
hostile to the static-shape discipline. Recorded as the fallback if genuinely unstructured meshes
ever become necessary.

Two [DF] details worth carrying into M5: per-tet stochastic fill (`dsmcMeshFill.C:68-238`) makes
particle counts follow exact sub-volume rather than cell-averaged volume — which is precisely
what cut cells need; and freshly injected inflow parcels get a randomized `stepFraction()` on
their first move (`dsmcFreeStreamInflowPatch.C`), a deliberate anti-pileup measure that prevents
an artificial density spike on the inflow face.


**Status 2026-09-16 — in progress on `fix/paper-adjoints` (`jax_dsmc/mesh/`).** The
embedded-surface layer is being built exactly as settled above: the Cartesian grid stays,
the body is a `Surface` (SPARTA line/triangle files, normals into the flow) or an
`AnalyticBody`, overlapped cells are cut (per-cell flow volume, candidate-element tables),
and two trackers share one event routine — the dilated-stencil earliest-hit loop and
SPARTA's cell-by-cell walk (`update.cpp:879-1547`) as the robust reference. Split cells are
detected and refused (convex bodies only for now). The plan, gates and figure list are in
`docs/ADJOINT_BRANCH_REVIEW.md` §7 once the phases land.

**Status 2026-09-17 — phases A (geometry) and B (forward) built; gates in
`examples/mesh/` (see its README).** Geometry: cell counts, cut-cell set and per-cell flow
volumes match SPARTA's dumps (max error 1.3e-3 of a cell at `sub = 32`, 4.7e-5 at 3-D
`sub = 16`), candidate lists equal a brute-force clip.  Forward: both trackers, element
models, tallies, `.vtp` output; the argon circle duplicate reproduces SPARTA's fill curve,
surface-collision rate (139 vs 140.5 per step), total drag (+0.3 %, |z| = 1.4) and fields
(4 seeds each); the SPARTA `compute grid temp` is un-centred.

**Status 2026-09-18 — phase C (mesh-aware adjoint) built** (PSEUDOCODE §9.7, review
§7.6): value-form rewards in the device backward, `grad_x` from the fit, the traced
replay of the wall map and its VJP w.r.t. the geometry (`jax_dsmc/shape.py`), the diffuse
normal score, the hit-timing band term.  G1 and G2 pass (|z| ≤ 0.8).  G3 (embedded plane
translation) exposed and fixed two estimator defects (the timing jump of velocity-
dependent step rewards; the bias of a two-sided fit next to a wall) and one open problem:
the variance of the fitted-`∇_xλ` term on translation-type controls (unbiased, per-seed sd
several times the CRN FD's).

**Status 2026-09-20 — phase D measured; open issue.** Analytic circle radius
(`examples/ensemble_mesh.py`): estimator 1.44 ± 0.09 e-20 vs CRN FD 1.40 ± 0.01 e-20 (48
seeds, z 0.5), tally term 1.28 ≈ `F/r` 1.23; `centre_x` consistent with zero.  Four `∇_xλ`
estimators (one-sided, pooled, along-flight, pooled + along-flight; review §7.6 item 5)
compared on the plane and the circle: none resolves the x/v-part of a translated flat
wall (exact +0.003 on the shear probe; pooled / along read 0; one-sided consistent at 3×
the noise), and the circle variants spread by 2–7 % of the total through the x-part.
**Decision: shape controls restricted to normal displacements of curved surfaces**
(radius-type; the exact tally term carries the gradient) until the translation-type
x/v-part is resolved — recorded options: replica evaluation at the displaced state
(collisionless only), the ε timing score (4–8× the FD sd), or a different estimator.
First optimisation case to demonstrate feasibility: circle → airfoil-like shape (and
the sphere analogue) by normal displacements only.

**Representation decision (2026-09-20): level set.** Shape optimisation deforms a C1
level set (`jax_dsmc/mesh/levelset.py`, uniform cubic B-spline coefficients on a finer
grid, single surface id like the analytic body), not the 50-point polygon: the polygon's
vertex edge-band term is representation-induced and vanishes on a C1 surface, so the
pathwise VJP is complete for specular walls; only the diffuse silhouette term (physics)
survives, in φ-form.  Density/porous representations rejected (intermediate densities
have no kinetic meaning for an impermeable wall).  Gates: LS0 geometry (CPU), LS1 forward
vs the analytic circle, LS2 `dJ/dcoef · (∂φ/∂r = −1)` vs the analytic radius gradient;
then `examples/mesh/optimize.py` (H1-smoothed gradient, area constraint, CRN line search,
periodic reinitialisation).  The vertex edge-band score is not built.

**Status 2026-09-20 (evening) — phase E built and gated (review §7.7).**
`jax_dsmc/mesh/levelset.py` (`LevelSet` = third `Mesh.kind`; hit search = sampled
bracket + bisection + Newton step with the search under `stop_gradient`; host tables by
the coefficient-sign test; `from_analytic`, `from_surface`, `reinitialise`,
`body_volume`), wired into `geometry.first_hit / surface_normal / inside_polygon /
stencil_sufficient / nearest_normal`, `walk.py`, `wall_adjoint` (the remaining budget
now reaches `first_hit`), `cut.py`, `shape.geometry / with_geometry`.  LS0: 20 CPU tests
(`tests/test_mesh_levelset.py`) + `geometry_checks.py` (`h⁴` convergence, tunnelling only
below one sample spacing).  LS1/LS2 (`ensemble_mesh --body levelset`, 4 seeds, CRN):
objective / hits / FD identical to the analytic run (2e-6 / 0 / 1e-4); gradient per
estimator variant identical on 3 of 4 seeds (1e-4), 4 % apart on one seed through a
near-grazing hit's `1/(v·n)` derivative (the pathwise estimator's heavy tail, measured
by `scratch/mesh/probe_grazing_tally.py`).  `mode2` (`cos 2θ` normal displacement, 4 seeds):
ensemble −1.587 ± 0.030 e-20 vs CRN FD −1.617 ± 0.029 / −1.571 ± 0.059 e-20, z = +0.7 /
−0.2, per-seed sd 3.8 % (the radius: 45 % — a zero-mean mode leaves the exact tally
term dominant).  First optimisation (`examples/mesh/optimize.py`, 8 iterates × 2 CRN
seeds, 55 min): drag −7.4 % at area constant to 1e-11, monotone, `figures/
optimize_circle.png`; resumable.  Grazing heavy tail of the tally term measured
(one hit at `|v·n|/|v| = 3e-3` = 15 % of a window's tally sum).  Not yet: Kn 0.5 (G4/G5).

**Status 2026-09-21 (later) — the two remaining terms (review §7.8).**  Grazing
regularisation of the hit-time derivative (`grazing = eps` on both bodies, tangent only):
the tally term's seed sd 8.4 % → 2.9 % at eps = 0.1 with bias < 2 %; default 0.05.
Silhouette band term for diffuse walls (`shape.silhouette_cotangent`, one passing per
trajectory, virtual outcomes valued from the leaving population) with the GP capsule
plate case (`--case plate --acc 1 --parameter width`): at `h_φ = 0.125` 80 % of the FD (the
three-knot cap under-resolves the score term); at `h_φ = 0.0625` with the cut-off,
1.274 ± 0.159 vs FD 1.336 ± 0.060 e-20, z = −0.4 (4 seeds), silhouette = `2Φ_tip` and
score + x = `2(Φ_flat − Φ_tip)` as the capsule reference requires.  The score term's
44 % seed sd is the emission-side `v_t / v_n` tail of the normal score, now floored by
the same `grazing` (shipped as the **hybrid** normal term: the
hit's own reward pathwise through the reparameterised sample (parallel-transported
frame), the value-to-go after the hit on the score — normal-term seed sd 3.1 → 0.77
e-21 at no bias; GP 1.191 ± 0.068 vs FD 1.169 ± 0.096 (h = 0.05, z = +0.2) e-20).
The sphere gradient, the finer circle (48 k particles, 40 × 40, `h_φ = 0.0625`) and
the seeds × {GD, Adam} optimiser study are set up for CURC Alpine
(`examples/mesh/cluster/`: `check_env.sbatch`, then `submit_curc.sh sphere|mode2|opt` —
one right-sized job per seed for the gradients, one per (S, optimiser) with the S seeds
of an iterate on min(S, 3) GPUs (`optimize.py --workers`) — then `collect.py`), not run
here.  The A100 partition with a CUDA venv: the MI100 `jax` module is 0.4.7, the code
needs ≥ 0.4.36.  Neither the diffuse circle nor the specular capsule gates them (all
cluster bodies are specular and curved everywhere).

**Further note — hierarchical (multi-level) grids, deliberately not built yet.** SPARTA
refines parent cells into child grids (`create_grid ... level`, `adapt_grid`), which is
what makes resolution near a body and later mesh adaptation affordable. It is notable work
here and not needed for the first shape gradients, so it is deferred; but every new table of
`jax_dsmc.mesh` is laid out **per leaf cell** (`cell_type[C]`, `volume[C]`, `candidates[C, S]`,
`csurfs[C, S]`, `outside_corner[C]`, per-cell `h` read through `grid.cell_bounds`), never by
`grid.n` arithmetic, so the extension is confined to:

| touch point | change for a two-level grid |
|---|---|
| `grid.cell_of` | static-depth descent: coarse `floor` → `child_offset[c0]` (−1 if unrefined) → child `floor`; leaf ids are a flat range |
| `grid.cell_bounds` / `cell_volume` | per-leaf `lo[C], hi[C]` tables instead of `unravel_index` |
| `output.write_vtr` | `.vtu`, or one `.vtr` per level |
| `viz.field_to_grid` | rasterise leaves onto the finest level |
| `decomp.slab_of`, `sub_grid` | slabs by coordinate, leaf ranges per slab |
| `ensemble_backward` / `ensemble_device` fit metric `scale = dx` | per-query cell `h` |
| stencil sufficient test `|v| dt ≤ R h` | per-cell `h` (already read from `cell_bounds`) |

Nothing in `sort`, `collide`, `sample`, `inflow` or the trackers needs to change.

---

## Phase 6 — M6: the pathwise + score adjoint

Covers general VSS collisions, transport, periodic and specular BCs, and all IC / VSS /
`dt` / `fnum` / domain-extent parameters. **Excludes thermal (M7) and inflow (M8).**

### `jax_dsmc/adjoint.py`

Thin wrappers: the surrogate objective `J̃` from Finding 2, `value_and_grad` over `run`,
a parameter pytree, and objective functionals. **No hand-written backward pass in production.**

Score-term variance: the papers' estimator uses reward `(φᵢᴹ + φᵢ₁ᴹ)` with no baseline. Ship
that as the default so results match the papers; provide an opt-in constant baseline
`(φᵢᴹ + φᵢ₁ᴹ − b)` (unbiased, since `E[score] = 0`) and a benchmark reporting gradient variance
against finite differences, so the choice is data-driven.

### `tests/test_adjoint_oracle.py` — the load-bearing tests

Hand-code both algorithms literally and assert `jax.grad` matches to machine precision:
Both run with `pair_perm` (decision 10), so they implement the papers' disjoint pairing rather
than an NTC approximation of it.

1. **Algorithm 2** (`2009.01363`), 0D Maxwell molecules: `γ_F = −∂ᵥr`, then `B(σₖ,αₖ)` backward.
   Validates the pure pathwise path. Also assert `rank(A) == 4`, `svd(A) == [1,1,1,1,0,0]` —
   if a refactor ever makes `A` full rank the collision map has been altered.
2. **Algorithm 3** (`2207.11579`), 0D VSS: adds the `η` terms from `eq:back_eta`, including the
   easily-missed *virtual-but-not-real* branch contributing `∂log(Σ−q)/∂v`. Validates the
   surrogate construction.

This makes the whole architectural bet falsifiable, and becomes a permanent regression test:
any refactor that quietly breaks the frozen-randomness contract fails here rather than silently
producing wrong gradients.

## Phase 7 — M7: thermal-wall adjoint

Add `τ ~ N(Δt, ε²)` to advection behind an `epsilon` config parameter (`epsilon = 0` keeps the
exact deterministic scheme, so M1–M5 physics validation is unaffected). Accumulate
`log pₗ(x, v)` into the same `logh_acc`. Ship the paper's ε-stability study as a test sweeping
ε and reporting drift in both objective and gradient.

## Phase 8 — M8: inflow ensemble adjoint

Linearized collision operator `L[f]` and its adjoint `L*[f]`, forward ensembles rather than
single realizations, Monte Carlo approximation of the backward ensemble equation, and the local
score on the inflow flux. Structurally different from Phases 6–7; scoped separately.

---

## Testing strategy

`pytest`, parametrized over `D ∈ {1,2,3}` throughout. Three suites.

### Reference cases

[DF] ships six self-contained cases under `run/hyStrath/dsmcFoam+/` that map almost exactly onto
our milestone ladder, several with published references. Adopted as validation targets:

| [DF] case | Physics | Our milestone |
|---|---|---|
| `heatBath-5species` | 0-D relaxation, single cell, all-specular walls | **M1** |
| `hypersonicCorner` | supersonic corner, Ar — Bird (1994) §16.2, Scanlon et al. (2010) | M3 |
| `supersonicFlatPlate` | leading-edge flat plate — Bird (1994) §14.3 | M3/M4 |
| `couette_N2-O2` | planar Couette between heated plates, one moving | **M4** |
| `orion107kmNR` | Orion capsule at 107 km from STL, non-reacting air | **M5** |

*M5 also migrates VTK cell output from `.vtr` to `.vtu` (cut cells are not rectilinear).*
| `axisymmetricFlatnosedCylinder` | axisymmetric, radial weighting | deferred (needs RWF) |

`hypersonicCorner` and `supersonicFlatPlate` are the valuable additions — literature-validated
problems with published profiles, which give M3/M4 an external check beyond the analytic BKW and
Chapman–Enskog gates.

### Correctness
- `safe.py`: gradients finite at all singular points (table above).
- Grid: `cell_of` round-trip, particles exactly on `lo`/`hi`/corners, all D.
- Sampler: Maxwellian moments (mean = vstream, var = kT/m), KS test, seeded.
- VSS scatter: momentum and energy conserved to f64 tolerance over random pairs; cosχ matches
  `2U^(1/α)−1`; α = 1 recovers isotropic.
- Move: periodic wrap invariance; specular multi-bounce fold verified against a brute-force
  iterative reference; no particle ever outside the domain.
- **BKW analytic solution** (M1 gate).
- Equilibrium preservation: start Maxwellian, stay Maxwellian within MC error.
- Couette viscosity vs. Chapman–Enskog (M4 gate).

### Gradient
- `jax.grad` vs. central finite differences **with common random numbers** (same base key on
  both perturbed runs — without this the FD estimate is pure noise), for every parameter, at
  ω = 1 and ω ≠ 1.
- `jax.grad` == Algorithm 2 and == Algorithm 3 (Phase 6 oracles) and match behavior reported in papers.
- No `NaN`/`Inf` in any gradient, including at degenerate configurations (identical velocities,
  zero relative speed, collinear v_r, `h` → 0).

### Performance / parallelizability — first-class, not an afterthought
- **Zero recompilation**: repeated same-shape calls must not grow the compilation cache.
  Catches Python-level branching on traced values.
- **Static-shape audit**: `jax.make_jaxpr` of a step is identical across different particle
  counts at fixed capacity, and contains no `while_loop` with a data-dependent trip count.
- **`vmap` composability**: `vmap(step)` over an ensemble of independent simulations must work —
  a strong proof of purity and shape-staticness, and directly reusable for M8's ensembles.
- **Throughput scaling**: particles·steps/sec; doubling N doubles wall time within a tolerance
  band; doubling cell count at fixed N should be ~flat.
- **Compile-time budget**: `jit(run)` compiles under a threshold — the attempt-slot `scan` plus
  checkpointing is the thing most likely to blow this up.
- **Peak device memory vs. (N, M)** for the reverse pass, reported as a benchmark. This makes
  the eventual checkpointing-vs-reversibility decision data-driven, given the 1e9–1e10 target.
- Device-placement check: no host round-trips inside the step.

---

## Deferred, with decisions shaped now

- **Sharding to N ~ 1e9–1e10.** Single-device first. State is structure-of-arrays and
  cell-sorted, so `shard_map` over spatial subdomains plus particle migration (SPARTA's
  `comm->migrate_particles()`) is an addition. Global `argsort` becomes per-shard sort + migrate.
- **Reversible O(1)-memory adjoint.** The collision *map* is an involution — re-applying it
  with σ and α exchanged recovers the input, verified numerically — and advection/periodic/
  specular are invertible, so forward state could be recomputed backwards instead of stored.
  (Note: this rests on the map, not on the matrix. [P1] writes `B = Aᵀ = A⁻¹`, but `A` is
  rank-4 singular — at fixed σ the output depends on `(v,v₁)` only through `v+v₁` and `|v−v₁|` —
  so `A⁻¹` does not exist. `B = Aᵀ` is correct and is all the adjoint needs.)
  Diffuse re-emission is the irreversible piece. Not built now;
  `jax.checkpoint` covers the near term and the memory benchmark says when this becomes required.
- **Differentiable mesh / adjoint-driven AMR.** Deferred to M5 by decision — the mesh layer is
  being rewritten there anyway for cut cells. The four-function mesh interface in `grid.py` is
  the contract that keeps that rewrite from touching `collide` or `sample`.
- **Multi-species, Larsen–Borgnakke, chemistry.** Arrays are already `(S,)`/`(S,S)` and `vremax`
  is already `(C,G,G)`. The LB differentiability hazard is **narrower than previously stated**:
  [DF] (`LarsenBorgnakkeVariableHardSphere.C:243-278`) shows the diatomic case `ζ_rot = 2` — the
  common one — has a closed-form inverse CDF, `E_rot/E_c = 1 − R^(1/(5/2−ω))`, needing no
  rejection sampler at all. Only general polyatomic `ζ > 2` needs the gamma/Beta rejection loop
  (SPARTA `collide_vss.cpp:457-603`, [DF] `dsmcCloud.C:1181-1218`). [DF] also fixes the mode
  ordering (electronic → vibrational → rotational, each gated by its own collision-number
  probability) and gives the temperature-dependent `Z_v(T)` formulation.

## Unverified claims — gate each milestone on its own

Nine assertions in the spec are reasoned but **not measured**, registered in `PSEUDOCODE.md` §19
and mirrored as an executable gate. Each is pinned to the milestone that first makes it live, and
its gate is part of that milestone's definition of done — checked *before* the deliverable is
written, not after.

```bash
python3 docs/phase0_verification.py                 # run checks, list open gaps
python3 docs/phase0_verification.py --milestone M3  # exit 1 if any M1-M3 gap is still open
```

| Milestone | Gaps that must close first |
|---|---|
| ~~**M1**~~ | ~~U1 `EPS_REL_VR2` bound~~, ~~U3 Σ safety factor~~ — **both CLOSED 2026-09-04.** U1: bound exactly tight, value stays `1e-8` (the proposed raise to `1e-6` was measured and rejected). U3: achievable, but only via the a-priori thermal majorant above — SPARTA's own seed fails. Evidence in `PSEUDOCODE.md` §17.4/§17.5 |
| **M2** | ~~U5 Oracle 3~~ **closed** — built in `tests/test_adjoint_oracle.py`; the 2x2 ordering discriminator is exactly diagonal |
| **M3** | U4 `A_max` inflated-Σ sizing untested; U8 `h_crit` is a 6-decade extrapolation; U9 cost claim has one noise-dominated point |
| **M5** | U6 `logh_acc` zeroing on slot reuse; U7 non-uniform `fnum` mass-conservation guard |
| **M6** | U2 count-score ensemble unbiasedness has error bars 40× the estimate — low power, not evidence |

**U4, U6 and U7 are blocking, not advisory.** Their failure mode produces no error, no `NaN` and
no diagnostic — the same shape as the two worst defects found so far (§8.1's exactly-zero
gradients and §8.4.1's silent momentum violation), both of which survived careful reading.
Reasoning is not evidence for that class.

## Verification

```bash
pytest tests/                          # all suites, D ∈ {1,2,3} — CPU, ~3 min
pytest tests/ --gpu                    # same tests on the accelerator (pre-milestone)
pytest tests/ -v                       # verbose
pytest tests/test_perf.py tests/test_perf_budget.py   # structure + cost budgets
python3 -m tests.test_perf_budget --compile-time      # compile time: report, not gate
pytest tests/test_adjoint_oracle.py    # Algorithms 2 and 3 equivalence
python -m examples.bkw        # M1: BKW vs. analytic, plotted
python -m jax_dsmc.examples.couette    # M4: viscosity vs. Chapman–Enskog
python -m jax_dsmc.examples.grad_check # M6: adjoint vs. finite differences
```

Each milestone gate is a physics test against an analytic or theoretical result, not a
regression against our own prior output.


---

## Status update — M2 closed, multi-device built (2026-09-05)

**M2's physics gates pass in all three dimensions.** Gap U5 (Oracle 3 / intra-step ordering)
was open here and is now **closed** — see the 2026-09-08 update below. Two independent external
gates:

1. **Bird §12.11 normal-shock structure.** `L/lambda_1 = 5.98 / 6.12 / 6.18` at D = 1/2/3 against
   Bird's 6.2, and simultaneously 4.53 / 4.64 / 4.68 against Alsmeyer's experimental 4.8. Jump
   conditions 0.1-0.7 %, shock speed 0.3 %. Resolution-converged over a 3.2x range in cell size.
   Run as a *reflected* shock off the specular wall, since M2 has no inflow or moving wall.
2. **2-D Taylor-Green viscosity.** An exact Navier-Stokes solution, so the fitted decay rate
   measures `nu` against Chapman-Enskog with no free parameter: **+0.38 %**, R^2 = 0.9994.

Gate 2 was decisive in resolving gate 1: it ruled the collision model out as the source of what
looked like a 24 % thickness discrepancy, which turned out to be the mean-free-path definition
(spec §21.1). **The methodological lesson is worth carrying forward** — agreement with one number
quoted in a reference is weak evidence when two plausible formulas both reproduce it. What settled
it was a limit the formula must satisfy exactly (hard spheres) plus an independent derivation
route (via the measured viscosity).

### Deferred decision now taken: how to shard

Decision 7 said "single-device first; layout chosen so sharding drops in later." It has been
measured, and the layout does **not** drop in:

Sharding the particle axis leaves `table.order`, `table.counts` and every per-cell moment
**replicated**, because the global `argsort` collapses the sharding — so each device redundantly
performs the whole collision (collide: 6.8 / 13.0 / 25.2 ms at 1/2/4 devices on a fixed problem).
The parallel axis of the collision operator is the *cell* axis, so parallelism requires each
device's particles to be co-located with its cells: spatial domain decomposition.

`jax_dsmc/decomp.py` implements a 1-D x-slab cut with `lax.ppermute` neighbour exchange. Cells
shard for free — with row-major ordering an x-slab is exactly a contiguous block of cell indices —
which is a direct payoff from keeping §4's mesh interface narrow. Spec §20.

**This adds a third and fourth capacity to the static-shape inventory** (alongside `N` and
`A_max`): per-device particle capacity and the migration buffer. Both fail the same way — silent
particle loss — and both bugs found during development lost or duplicated particles while
reporting *zero* overflow, one of them invisible in 1-D and appearing only in 2-D. They belong in
the same category as U4/U6/U7: reasoning is not evidence, and the test counts particles rather
than comparing profiles.

### Sizing, which constrains the milestone ladder

Cells must resolve the mean free path while `Re ~ 2 Ma sqrt(gamma) (L/lambda)`, so **Reynolds
number costs `Re^D` in cells**. On 4 x H200: `Re ~ 1500` in 2-D, `Re ~ 66` in 3-D. The 2-D
Taylor-Green case is therefore genuinely turbulent at reachable scale *and* has an exact solution;
3-D Kolmogorov scaling is not reachable on a single node and is not claimed.

### Still open

- ~~U5~~ **closed 2026-09-08.** Oracle 3 is built (`tests/test_adjoint_oracle.py`, five tests
  plus a production-ordering companion). The 0-D oracles provably cannot see the
  `move -> sort -> collide` ordering because `alpha == 0` there; the 1-D two-step oracle can,
  and its 2x2 discriminator is exactly diagonal. It did **not** need to wait for M6.
- U2, U4, U6, U7, U8, U9 (gap register).
- M2's turbulence spectra batch: the 1-D `E(k) ~ k^-2` case runs; 2-D/3-D spectra want the node.
- The scaling harness has been exercised only on forced-host devices, which verify correctness
  and cannot measure speed. Real numbers need the 4 x H200 node.
- `alpha = 1.40` from SPARTA's `ar.vss` vs Bird's `alpha = 1.6625` for the same gas. Both are
  fits; the difference moves the viscosity and the mean free path by ~0.6 % and is inside the
  agreement above, but it should be stated whenever a number is quoted.

## Status update — M6 adjoint built, oracles pass, two concerns open (2026-09-05)

`jax_dsmc/adjoint.py` exists and the Phase-6 bet holds: **no hand-written backward pass in the
production path.** `magic`, `objective`, `surrogate`, `dice_reward` — four constructions, the
rest is entry points and objective functionals.

**Delivered**

- `tests/test_adjoint_oracle.py` — [P1] Algorithm 2 and [P2] Algorithm 3 hand-coded from the
  papers' LaTeX and asserted against `jax.grad`. Exact to rtol 1e-11 / 1e-10. Includes the
  negative control: suppressing the `j=3` reject-branch score **must** change the gradient.
- `tests/test_adjoint.py` — gradients through the production `move → sort → collide` at
  `D = 1, 2, 3`, plus the §8.1 regression guard (score-only parameters read exactly `0.0`
  pathwise and non-zero scored).
- `examples/shock_adjoint/` — AD vs finite differences on Bird's Mach-2 shock, all three
  dimensions, with a decision-flip guard and figures.
- `Diagnostics.decision_checksum` — a frozen-decision fingerprint, without which no
  finite-difference check of this code means anything.
- `collide.species_params` — builds `SpeciesParams` from the five *primitive* VSS constants, so
  `diam`/`omega`/`tref` keep their path through the derived `prefactor`.

**Decisions taken that the plan did not anticipate**

1. **Oracle 1 cannot run against `vss.scatter`.** [P1] eq. `AB_def` assumes `delta sigma = 0`;
   SPARTA samples the deflection in the `v_r` frame, giving a **rank-6** pair Jacobian where `A`
   is rank 4, differing by 53–81% *at Maxwell molecules*. Oracle 1 runs [P1]'s own lab-frame map;
   §17.1.1 records the measurement and §13.1 is revised. New gap **U10**: which reparameterisation
   has the lower gradient variance is unmeasured, and both are unbiased.
2. **The flip guard needs more than §13.4.1's six items** — the specular boundary fold index and
   `scatter`'s degenerate-frame branch as well — and must not weight any decision by zero. Both
   holes were found by a failing test, not by reading.
3. **`A_max` calibration cannot live inside a differentiated call.** `run_adaptive` uses host-side
   control flow; `examples/shock_adjoint.calibrate` settles the capacity first.

**Two open concerns, both being measured rather than argued**

- ~~**Reverse-pass cost.**~~ **SETTLED (§23.4): ~3–5×, and the 17× was a benchmarking artifact**
  — measured under `vmap` over 128 keys, where the reverse pass goes memory-bound. Un-batched it
  is 3.1–5.0× and flat in `N`. Both structural hypotheses were **refuted**: the ratio *plateaus*
  in `A_max` (not the scan residuals) and all four remat policies land in 3.94–4.48. `collide` is
  90 % of the forward, so the reverse pass is essentially the VJP of the attempt-slot scan.
  [P1]'s 0.58 excludes the forward run that must precede it, so their gradient costs ~1.6
  forwards against our ~5; the residual gap is a hand-derived backward versus general AD.

  **What actually binds at scale is memory** (§23.4.1): ~4300 bytes per particle for one
  gradient, ~30× the forward's, capping a 4×H200 node at `~1.3e8` particles for an adjoint run
  against `~1e9` forward-only.
- **Score-term variance.** [P2] §6 budgets `M_s = 100` runs at `N = 1e6` and a **10%** FD step,
  with random error `O((sqrt(N M_s)|Δm|)^-1)`. Our probe ran a hundredfold smaller budget. The
  parameter [P2] actually differentiates — the initial temperature — comes out at 0.2% SE here,
  consistent. **The `N^-1/2` law is now measured**: over `N = 4e3 -> 2.56e5` at fixed `A_max`,
  the fitted exponent of `sd ~ N^p` is -0.508 / -0.464 / -0.495 / -0.499 for `temp` / `ux` /
  `alpha` / `diam`. The large *percentage* error bars quoted earlier for `alpha` are a near-zero
  denominator — its mean is consistent with zero at every `N` while its absolute error bar falls
  8x on the same law. **Correctness is not in question** (Oracle 2 is exact); sample efficiency
  is, and it scales as published.

  The follow-ups all landed. **The noise is mostly pathwise** (3-5x the score half, uncorrelated),
  and in *variance* terms — the units a baseline works in — the score is only 3-12% of the total,
  capping a perfect baseline at 1.02-1.07x. (This does **not** explain §11.4's ~1.2x, which
  exceeds that ceiling and was measured in a different configuration; the directional claim is
  what holds.) More particles is the only lever on the dominant term. **At `Delta m = 0.1` the
  AD/FD gap is truncation**, not disagreement — our AD standard error is 0.03-0.25%, far below
  the noise level at which [P2] calls that step optimal. **The spatial window costs signal**, not
  variance (up to 6.3x SNR for `diam`). And **`sd ~ sqrt(M)` is not the right comparison** in a
  developing shock: for the pure-score `diam`, signal and noise grow at the same rate
  (`M^0.720` vs `M^0.721`), leaving relative precision flat in run length.

---

## Status update — Oracle 3 built, gap U5 closed (2026-09-08)

`python3 docs/phase0_verification.py --milestone M2` now reports **`gating on M2: 0 gap(s)`**;
it previously reported `BLOCKED: U5`.

**What U5 was.** Oracles 1 and 2 are both 0-D, so the position cotangent `alpha` is identically
zero, both candidate adjoint forms collapse to `beta_k = B_k beta_{k+1}`, and neither oracle
*can* distinguish `move -> collide` from `collide -> move`. That is not a coverage oversight but
a provable blind spot — and it mattered, because §12.1 records a real divergence: we follow
SPARTA and move first, putting the transport term **outside** `B_k`, while [P3] `main.tex:721`
collides first and puts it **inside**. The forms differ by `O(dt ||B - I||)`, which does not
vanish at realistic timesteps.

**Oracle 3** (§13.3, `tests/test_adjoint_oracle.py`): 1-D, 2 cells, 2 steps, periodic, `omega = 1`
with `Sigma = q` so every pair collides and the adjoint is purely pathwise. `B_k` is [P1]'s
analytic `A^T` from eq. `AB_def` — never `jax.jacobian`, which would be circular. The
discriminator is exactly diagonal:

| forward ordering | OUTSIDE (ours) | INSIDE ([P3]) |
|---|---|---|
| `move -> collide` | **1.33e-16** | 9.85e-02 |
| `collide -> move` | 9.58e-02 | **1.81e-16** |

Five tests: the positive gate, both negative controls (substituting [P3]'s form must fail;
reordering the loop must swap which form is right), `alpha_k = alpha_{k+1}` to 0.0e+00 in both
orderings, and the spec correction below. A sixth,
`test_production_step_moves_before_it_collides`, pins the **production** step's ordering
behaviourally — without it the oracle could end up faithfully validating the adjoint of a loop
we no longer run.

**Two things the build corrected.**

1. §13.3 claimed two steps is *the minimum that exercises the coupling*. Measured, one step
   already discriminates (2.66e-2), because the forms differ by `dt (I - B) P* alpha`, which is
   nonzero after a single application. Two steps is still right — first to push a *transported*
   cotangent through a second `B_k`, and ~3.7x the margin — but the stated reason was not the
   real one.
2. The objective must depend on **both** `x_M` and `v_M`. With a velocity-only `phi`,
   `alpha == 0` and the oracle silently degenerates into Oracle 1 while still passing. A
   nonzero-`alpha` guard is now asserted before the comparison.

---

## Status update — M4 forward complete: thermal walls, inflow, outflow (2026-09-09)

All wall and open boundary conditions are built **forward-only**, verified against external
physics rather than against our own prior output, and gap **U6 is closed**. Adjoints (M7/M8)
are deliberately not started: the forward physics had to be right first.

### What was built

| piece | where | status |
|---|---|---|
| Per-face BCs (`Faces`, `WallParams`) | `boundary.py` | walls, inflow and periodic can now coexist on one domain |
| Diffuse/thermal wall, SPARTA-style | `boundary.py`, `sampling.py` | flux-weighted half-Maxwellian, accommodation, sliding walls |
| Regularised thermal wall, [P3] §4 | `boundary.py`, `sampling.py` | `tau ~ N(dt, eps^2)`; `eps = 0` **bitwise** exact |
| Inflow / outflow | `inflow.py` | Bird's bounded rejection as a static `scan`; U6 guard |
| Couette gate | `examples/couette/` | Chapman-Enskog recovered as `Kn -> 0`; see below |
| Two-sided-inflow shock | `examples/shock/inflow_driven.py` | all six moments within **0.7 %** of Rankine-Hugoniot |

### The two headline gates

**Couette shear viscosity vs Chapman-Enskog** (Bird 4.64), argon. The gate is the
**`Kn -> 0` intercept** of a Knudsen sweep, not any single run, because Chapman-Enskog is a
continuum limit: at finite Kn the Knudsen layers eat into the sheared bulk and the apparent
viscosity falls, with a measured slope of about `-0.9` per unit Kn over `Kn = 1/10 ... 1/30`.
Current numbers are regenerated into `examples/couette/RESULTS.md` from the run logs and are
deliberately not restated here.

An earlier form of this line read **"0.24 % from Chapman-Enskog"**. That claim is
**retracted**. It came from a single draw of a quantity that carries several per cent of Monte
Carlo scatter, and the same mistake was then made three more times -- attributing the offset
in turn to under-resolution, to cell size, and to under-convergence -- before an ensemble was
finally run. The one-line check that would have caught it at the outset was to run the same
configuration twice with different seeds. `examples/couette` now has no single-seed mode:
`--seeds` has a hard minimum of 2, warm-up is set per case as a multiple of the momentum
diffusion time `L^2/nu` rather than as a step count, and every ratio is reported as mean +- SE
with its seed count.

**Two-sided-inflow normal shock at Mach 2**, `L = 60 lambda`: upstream `nrho`/`T`/`ux` within
0.19/0.68/0.05 %, downstream within 0.66/0.26/0.37 % of Rankine-Hugoniot, mass flux conserved
end to end to 0.17 %.

### Three things the build got wrong first, all caught by measurement

1. **A sub-move scan that was not needed.** The first thermal wall resolved hit times with a
   bounded `K_bounce` scan, imported from SPARTA — which needs it only because it has general
   cut-cell geometry. [P3] eq. `thermal_v_L` is already closed form on a static mesh. Both were
   implemented and compared: identical results, **1.65 s against >7 min**. §9.4 was rewritten.
2. **A corner bug that heated the gas.** Applying each face's rule in turn gave a particle
   leaving through two faces *two* flux-weighted re-emission draws; since each is biased toward
   fast particles, the second pumps in energy — a cold gas settled at 350 K (`D=2`) and 386 K
   (`D=3`) against a 300 K wall. Resolving the first face *in time* and running a statically
   gated second pass took `D=3` to **300.90 K**.
3. **A shock parked inside its own thickness.** Initialising the box at the upstream state made
   the shock form at the outlet, leaving 6 mfp of downstream run for a shock ~5-7 mfp thick; the
   plateau read 9 % off Rankine-Hugoniot. Mass flux was conserved to 0.07 % throughout, which is
   what showed the layout was at fault and not the solver.

### Known limitation, measured

A corner between **two thermal faces on different axes** resolves two crossings, not three, so
an all-thermal box carries a residual `3.3e-4` recrossing rate and ~0.3 % warm bias. It is
**exactly zero** for both target geometries (Couette: periodic/thermal; shock: inflow/specular),
because periodic, specular and outflow keep exact any-number-of-crossings folds applied *after*
the wall bounce. The ordering inside `apply_faces` is load-bearing and asserted.

### Next

M7 (thermal-wall adjoint) then M8 (inflow ensemble adjoint). The groundwork is in place:
`logh_acc` is the accumulator, the branch probabilities `p_l` are implemented and validated
against empirical crossing frequencies, and [P3] eq. `J_final_t1_x` shows the boundary
sensitivity is `(d log p_l/dx) r_i` — structurally the same DiCE score the collision `eta` term
already uses. M7 is wiring, not new theory.

## Status update -- M7/M8 boundary adjoints, partial (2026-09-11)

Boundary gradients are built and the forward solve is provably unchanged (Couette and a
30 000-step inflow shock both reproduce pre-M7 output exactly). What is **verified against
external references**, not against our own prior output:

* `dJ/dT_wall`, `dJ/dv_wall` -- pathwise, fixed-key FD to 1e-11, and pointwise against
  the Couette closed form over the whole profile.
* `dJ/d(nrho)` for inflow -- score, ensemble FD/AD = 1.04.
* `d(T_2/T_1)/du_1` for the shock -- against Rankine-Hugoniot over `M = 1.6-4.0`.

What is **not**: `d(rho_2/rho_1)/du_1` has the wrong sign, and `dJ/d(acc)` is biased 2.8x
at a 40-step horizon. Both are score-carried and both are explained -- with measurements --
in `docs/ATTRIBUTION.md`. Read that before trusting any score-carried gradient; the headline
is that its bias **grows with trajectory length**, so a result that is fine at `M = 16` is
not fine at `M = 400`.

M7/M8 are therefore **partially complete**: the pathwise half is done and validated, the
score half needs the unbiased estimator, and the chaotic-sensitivity problem needs a method
this project does not have yet.
