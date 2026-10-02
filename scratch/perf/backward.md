# Backward / shape-gradient cost: where the time goes and what to cut (2026-09-28)

This was a static read only: nothing that imports jax was run. **[R]** marks something I
checked by reading the code or a result file. **[E]** marks an estimate, with the
arithmetic shown. Paths are relative to the repo root.

## 0. Corrections to the premises

- The mode2 fine rows (`..._grid2_mode2_..._seed*.json`) ran on **A100-PCIE-40GB**, not
  on a MIG slice (`examples/mesh/cluster/logs/mode2_seed0_32925276.out:2`) [R]. The
  optimiser runs (`optc_opt_*`) are the ones on the MIG 1g.24gb slice [R].
- The mode2 runner computes **all four** variants (`ensemble_gradient` passes
  `variants=None`, `examples/ensemble_mesh.py:512` → `shape.gradient_variants`,
  `jax_dsmc/shape.py:946-956`). So its `seconds_shape` ≈ 500 s is **four** full
  gradient scans, and each one compiles on its own. The optimiser asks for
  `(cfg["variant"],)` only (`examples/mesh/optimize.py:150-152`) [R].
- The optimiser uses only `pathwise + score` (`optimize.py:156`). The timing term is
  computed and then dropped (it is also left out of `gradient` at
  `ensemble_mesh.py:531-533`) [R].
- The case really is collisionless. `NRHO = 1.0` (`examples/mesh/forward.py:94`). The NTC
  mean attempts per cell is `f = ½ n(n-1) σg_max dt fnum / V` (`jax_dsmc/collide/ntc.py:167`).
  With n≈250, σg≈2.4e-16 m³/s, dt=1e-4, fnum=1e-3 and V=0.25, that gives f ≈ 1e-19, so
  `floor(f+U)` = 0 [E]. `optimize.py:444-445` already says "Natt = 0: the virtual
  pull-back's partners do no work" [R].
- Timing rows from one optimiser seed (coarse, f1, grid 1, 60 256 slots, ~56 k alive):
  the per-iterate seconds divided by S are 274 / 294 / 282 / 261 / 270 s for S1/S2/S4/S8
  adam and S8 gd (from `results/optimize/circle_coarse_*/history.json`). **Per-seed cost
  does not amortise with S**, so nothing is reused across the seeds of an iterate [R].
- The optimiser stores only the total seconds per seed. `info["seconds_backward"/"seconds_shape"]`
  is thrown away at `optimize.py:150-158` [R]. This is why the stage split below is an estimate.

## 1. Time budget of one `shape_terms` seed

Stages in call order (`examples/ensemble_mesh.py:374-506`):

| # | stage | code | work per step (M = 300 captured steps) |
|---|---|---|---|
| a | forward: 700 warm-up + 300 captured steps | `:389` → `_run` `:217` | full DSMC step |
| b | traced replay of the wall map (rewards, timing rewards, replay check) | `:411-430`, `shape.py:764-805` | wall map over all N slots |
| c | host `max_occupancy` (300 D→H copies + numpy bincount), called **3×** per seed | `ensemble_mesh.py:453`; `ensemble_inlet_shock.py:189` ×2; `ensemble_backward.py:770-780` | host |
| d | **2 × `backward_virtual`** (hit rewards, then timing rewards), same key | `ensemble_mesh.py:347-371`; `ensemble_device.py:611-718` | `cell_nodes` sort; 1 + 2·partners = **5 `cell_query` fits over all N slots** (`:567`, `:497-508`); boundary move skipped (`:593-607`) |
| e | host growth check: `lam` M×N pulled to host, ×2 | `ensemble_inlet_shock.py:236-247` | host |
| f | `shape.gradient` (along_pool: pool=1, k=96) | `shape.py:623-741`, `:899` | wall-map primal + `jax.vjp` (`:631`); `node_gradients` pooled fit over **all N slots** (`:641-650` → `ensemble_device.py:387-397`); **5 pullbacks** d_x, d_v, d_tally, d_score, d_time (`shape.py:667-705`) |

Measured totals:
- **Fine mode2 (A100, 240 k slots, 190 k alive, 32 rows)** [R]: median seconds 931,
  backward 163, shape 503 (four variants). The remaining 265 s covers forward, replay,
  4 FD forwards and the compiles.
- **Coarse f4 levelset (12 k alive)**, other hardware [R]: backward 78–118 s, shape
  214–435 s (`scratch/mesh/mode2_coarse_now_hit_ahead.json`, laptop RTX 3050 Ti).
  Going from 12 k to 190 k alive (16×) grows backward only about 2× and shape about 2.3×.
  **A large part of both stages is N-independent**: per-step latency across the 300-step
  scans plus compile time.
- Seeds 1 and 2 in one process are *not* faster than seed 0 in the shape stage
  (435/433 s against 407 s, same file). That fits the per-seed recompile in item 3 below.

Estimated budget for one coarse optimiser seed (MIG, ~270 s) [E]. I took the fine A100
per-stage times, divided by 4 for N, and assumed the MIG slice runs at ~0.4× an A100 on
this memory- and latency-bound work (1/4 of an RTX PRO 6000's SMs and bandwidth):

| stage | estimate | basis |
|---|---|---|
| a+b forward + replay | 35–50 s | fine ≈ 50 s per 1000-step forward (265 s / ~5.3 forward-equivalents), ×¼ N, ÷0.4 |
| d 2× virtual backward | 50–110 s | fine 163 s ×¼ ÷0.4 ≈ 100 s; the latency-bound share keeps the lower end above 50 s |
| f gradient (one variant) | 90–130 s | fine 503 s / 4 variants, along_pool is the heaviest (9P candidates, k=96) |
| compiles (replay, gradient, maybe backward) | 30–80 s | `docs/ADJOINT_BRANCH_REVIEW.md:245-255`: a wall-path reverse compiles in 34–49 s |
| c+e host round trips | 2–8 s | ~1.3 GB D→H per seed in coarse |

Candidate-width context [R+E]: `npc_max = max(64, 4·max_alive//C, max_occ+8)`
(`ensemble_mesh.py:450-454`). With ~56 k alive over 400 cells, 4·mean ≈ 560. The densest
cell holds ~1.56 × 250 ≈ 390 particles (frame `nrho` max 1.564 in
`results/optimize/circle_coarse_S8_adam/frames_36.npz`). So P ≈ 560 while the median cell
holds ~173 valid nodes. The pooled fit scans 9P ≈ 5000 padded candidates per query and
runs top-k 96 over them.

## 2. Ranked speed-ups

"Exact" below means the same gradient to round-off (reasons in the risk column). N_hit is
the element-hitting slots per step: mean 282, max 309, of 60 256 slots = 0.5 %
(`scratch/mesh/probe_lane_speed3.log`, last line) [R].

| rank | what | where | expected gain (coarse seed ≈ 270 s) | correctness risk | effort |
|---|---|---|---|---|---|
| 1 | **Skip the virtual collision pull-back on steps with no attempts and no collision velocity change** (`lax.cond`, before `cell_nodes`) | `ensemble_device.py:611-651` (`n_att` `:622`, sort `:620`, substep scan `:649`) | Removes ~all of stage d: 50–110 s coarse → ~1–2 s. Fine: 163 s → a few s [E]. Per step the backward drops from `sort + 5 N-query fits` to an O(N) `where` | **Bitwise** up to the sign of zero: the skipped branch computes `lam + Σ 0·four` (`:511`, `:592`) and the boundary move is already cond-skipped (`:593-607`). On collisional runs the branch is taken with the same `step_key`, so the result is unchanged | low (~25 lines) |
| 2 | **Fit grad λ only at the slots that hit an element this step** (compact to a lane of Q ≈ 1024, query only those, scatter back) | `shape.py:641-654`; `ensemble_device.py:347-397`; compaction helper `wall_adjoint.py:196-208` | The pooled fits go from N = 60 256 (240 k fine) queries per step to ≤ Q: **~60× fewer**. Est. 60–100 s → 2–5 s coarse. It also removes the per-chunk gather of 4096 × 9P × 8 f64 (~1.3 GB if XLA materialises it). **Lighter** | Exact when ε = 0 (the optimiser passes `epsilon=0.0`, `ensemble_mesh.py:476`). A non-hitter's `x_post = x + dt v` has no geometry path: `used = where(hit, distance, 0)` (`wall_adjoint.py:485`), `x = where(hit, x_at, x)` (`:439`), `time = where(ε>0, …, dt)` (`:41`, `:776-779`). Its cotangent therefore meets structural zeros, and zeroing it changes nothing. With ε>0 near-misses carry geometry, so keep the full path there. Lane overflow must raise, not truncate | low–med |
| 3 | **Compile once per process.** `reward`/`jump` are fresh `functools.partial` objects per seed and are **static** args of `replay`/`gradient`. Partial hashes by identity, so both retrace and recompile every seed. `npc_max` (static) also changes from seed to seed | partials `ensemble_mesh.py:407-410`; statics `shape.py:764-766`, `:808-823`, `ensemble_device.py:654-666`; `npc_max` `ensemble_mesh.py:450-454` | 30–80 s per seed after the first [E; compile times from `ADJOINT_BRANCH_REVIEW.md:245-255`]. For one-seed-per-process cluster jobs, add the persistent cache (`examples/bkw/__init__.py:37-50` already has `enable_compilation_cache`) | Bitwise (same program). Bucketing `npc_max` only adds padded invalid nodes (d² = inf, `ensemble_device.py:107`); top-k picks the same set | low |
| 4 | **One wall-map pullback per step, not five.** Skip `d_time` in the optimiser (pass `timing_jump=None`, unused there). Either sum the pathwise and score cotangents into one `vjp` call, or `vmap` the pullback over the 4 stacked cotangents to keep the split | `shape.py:667-705`; call `ensemble_mesh.py:483-485` | Pullback work drops 5 → 1. Est. 10–40 s coarse [E, unmeasured: see M4] | `timing_jump=None` is exact for pathwise+score. Summing the cotangents is linear, so only the summation order changes (round-off) | low |
| 5 | **Run the two value backwards as one pass**, `lam` carried as (N, 2). The kNN selection, design, weights and solve depend only on node positions (`ensemble_device.py:104-123`); only `rhs` (`:124`) sees the values, so solve with 2 right-hand sides | `ensemble_mesh.py:353-370` (both passes use `fold_in(key, 12345)`, `:363`) | After #1: ~0 for the circle. On collisional cases: ~45 % of the backward | Same key, so the same partners and the same fits. Round-off only | med |
| 6 | **Study runner only: share one wall-map primal+vjp across the 4 variants.** Score, tally, timing and silhouette cotangents do not depend on the variant; only gx/gv do | `shape.py:946-956` | mode2 shape stage ~500 s → est. 200–280 s (one primal + pullbacks, 4 fits); also 4 compiles → 1 | Round-off | med |
| 7 | **Differentiate the wall map only on the compacted hit lane.** Take the full-N primal from the record, un-differentiated. Lane passes already gather the dense pass's draws per slot | `wall_adjoint.py:474-480`, `:689-693`; `shape.py:630-631` | The `vjp` primal and pullbacks go from N to ~Q slots. Worth the rest of stage f after #2/#4, est. 20–50 s [E] | Must reproduce identical discrete events; the `_mismatch` check (`shape.py:466-488`) guards that. Exact for ε = 0 | high |
| 8 | **Drop the separate replay (stage b).** Store hit `time` (and `v_in`) in the record's event list so rewards and timing rewards come from the record. The replay-error check already runs again inside `gradient` (`ensemble_mesh.py:491-493`) | `ensemble_mesh.py:411-437`; `RecordEvents` `wall_adjoint.py:104-119` | One 300-step wall-map sweep, est. 5–15 s | Rewards then come from the forward-compiled arithmetic: ~1e-12 relative differences (the round-off `REPLAY_TOL` covers) | med |
| 9 | **Host round trips.** Compute `max_occupancy` once on device (per-step `segment_sum` → max) and pass it down; move the growth check onto the device | `ensemble_backward.py:770-780` ×3; `ensemble_inlet_shock.py:236-247` | 2–8 s | Bitwise | low |
| 10 | **Tighter `npc_max`**: `max_occ + 8`, bucketed to 32, instead of 4·mean (≈560 → ≈420) | `ensemble_mesh.py:450-454` | 1.2–1.4× on the remaining fits [E]; minor once #1+#2 are in | Bitwise (same argument as #3) | low |

Not a problem [R]: every stage is a single `jit` with one `lax.scan` over the 300 steps
(`shape.py:804`, `:899-901`; `ensemble_device.py:714-716`). There are no per-step host
syncs or Python loops inside the stages, so fusing steps into bigger scans buys little
beyond `unroll=2`. The timing backward's `lam_timing` enters `along_flight` as a value
(`shape.py:651-652`), not through a second fit, so there is no duplicated fit inside
`gradient`.

Combined [E], coarse seed: #1 (−50…110) + #2 (−60…95) + #3 (−30…80) + #4 (−10…35) +
#9 (−5). That takes ~270 s down to roughly **60–110 s**, i.e. 2.5–4.5× more seeds per
iterate at the same wall time. What remains is forward + replay + the full-N wall-map
vjp, which is #7 and #8.

## 3. Top 3: implementation sketches

**#1 collision skip** (`ensemble_device.py`, `_step_pullback_virtual`):
```python
mu = _add_step_reward(mu, rec_k, phi_k)
key, step_key = jax.random.split(key)            # key stream unchanged
n = rec_k.x.shape[0]
n_att = jnp.sum(rec_k.pairs.pi < n, axis=0)
changed = rec_k.alive & jnp.any(rec_k.v != rec_k.v_post, axis=1)
active = jnp.any(n_att > 0) | jnp.any(changed)
def run(mu):   # today's body :620-649, verbatim, returning (lam, sum(kern))
    ...
lam, kern = lax.cond(active, run, lambda mu: (mu, jnp.float64(0.0)), mu)
mu_prev, lam = _move_pullback(lam, rec_k, r_k)
```
Test: on a collisional case (inlet shock), results must be bitwise equal before and
after. On the circle, `lam` must be equal to the sign of zero. Also add a counter output
(steps skipped) so the regime is visible in `info`.

**#2 hit-only fits** (`shape.py:_gradient_step`, plus a query-subset form of `node_gradients`):
```python
hit_head = split.head.face[0] >= 2 * dim
hit_tail = particle_sum(jnp.zeros((1, n)), (split.tail.face >= 2 * dim).astype(jnp.float64), split.slot) > 0
want = survivors & (hit_head | hit_tail)
q_slot, lost = lane_slots(want, Q)              # Q static, e.g. 1024 coarse / 4096 fine
qi = jnp.maximum(q_slot, 0)
gx_q, gv_q = node_gradients_at(rec_k.x, rec_k.v, rec_k.alive, lam_k, grid, qi,
                               direction=_last_normal(events)[qi], klass=klass_k, **fit)
put = jnp.where(q_slot >= 0, q_slot, n)
gx = jnp.zeros((n, dim)).at[put].set(gx_q, mode="drop")   # likewise gv
# lam_timing / along_flight applied on the lane rows only; carry sum(lost) out -> raise if > 0
```
`node_gradients_at` builds the same tables as `ensemble_device.py:387-393` over all nodes
and calls `cell_query` with `q_c = c[qi]`, `q_x = x[qi]`, `q_v = v[qi]`,
`q_class = klass[qi]`, and `chunk = min(chunk, Q)`. Guard: only when `epsilon == 0` (a
static check, else fall back). Gate test: one seed, full vs lane, `pathwise_x`/`_v`
equal to ~1e-14 relative.

**#3 compile once** (`ensemble_mesh.py`):
```python
@dataclasses.dataclass(frozen=True)          # hash/eq by value
class DragReward:
    mass: float; dim: int; first: int; scale: float
    def __call__(self, events, tallies, rec_k, step): return _drag_reward(events, tallies, rec_k, step, self.mass, self.dim, self.first, self.scale)
# likewise DragJump(mass, dim, first, last, scale); build with float(mass), float(scale_r)
npc_max = -(-max(64, int(ensemble_backward.max_occupancy(rec, grid)) + 8) // 32) * 32   # bucketed
```
Then enable the persistent cache (`jax_compilation_cache_dir`) in `optimize.main` and
the `ensemble_mesh` main. Check with `jax.config.update("jax_log_compiles", True)`: a
second seed in the same process must log **zero** compiles of `replay`, `gradient` and
`backward_virtual`.

## 4. Measurements to take first (one GPU job each, attended, ≤1 JAX process)

M1 (gates #3, and the whole budget). One process, 2 seeds of the current optimiser
config, `jax_log_compiles=True`. Log per seed: forward, replay, each backward, gradient,
and host checks, using `block_until_ready` stage timers (`seconds_backward`/`_shape`
exist but `optimize.py` drops them). Gate: the compile seconds on seed 2 are the prize
for #3; if they are near zero, #3 is dropped.

M2 (gates #1). From one captured record: `sum(rec.pairs.pi < N)` over all steps, and
the count of steps with `any(alive & (v != v_post))`. Expected 0 and 0. If either is
nonzero, #1 is still exact but its gain shrinks in proportion.

M3 (sizes #2). Per step `max(want.sum())` over the 300 steps, coarse and fine. That sets
Q, with the overflow check. Then run the exactness check: full-N vs lane gradients on
one seed.

M4 (ranks #4 and #7 against each other). Time `shape.gradient` three ways on one record
(after #3, so compiles are excluded): (i) as is; (ii) `node_gradients` replaced by zeros,
giving the fit share; (iii) only the `d_x` pullback, giving the pullback share. If the
primal+vjp dominates (ii), #7 moves up; if the pullbacks dominate (iii), #4 is the cheap
first step.

M5 (#10). Print the three terms of `npc_max` (`ensemble_mesh.py:450-454`) for one seed.

The order that follows is M1+M2 in one job, then build #1 and #3 (low effort, exact),
then M3/M4, then #2 and #4.
