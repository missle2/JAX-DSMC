# Forward-solve performance and memory review (static reading only, 2026-09-28)

Scope: the forward `simulate.run` as the optc / mode2 runs call it (`examples/ensemble_mesh.py`
`forward()` -> `_run`), level-set circle, lane tracker, `n_sample = 16`, `h_phi = 0.0625`,
`epsilon = 0`, `K_BOUNCE = 32`. Nothing was executed. **[V]** means verified by reading (file:line
given). **[E]** means an estimate, with the arithmetic shown.

## 0. What one step does over all N slots (the baseline) [V]

* The level-set hit search `levelset.hit_body` (levelset.py:196-253) costs, per particle and call:
  1 `value_and_grad` (phi0, :221), a `fori_loop` of `n_sample = 16` `value`s (:174-176), a
  `fori_loop` of `n_bisect` `value`s (:177-179; `n_bisect` defaults to 40 at types.py:122, and
  `from_analytic`/`from_values` keep that default, levelset.py:384/401), and 1 `value_and_grad` for
  the Newton step (:244). That is **56 `value` + 2 `value_and_grad` evaluations and 56 sequential
  loop iterations per call.** It runs for every slot, dead ones included, whatever the distance to
  the body.
* Full-N calls per step: pass 0 (`_bounce` -> `_next_surface`, wall_adjoint.py:483, called at
  :588) and the post-loop query `_next_surface(x, v, grid, mesh, exclude, None)` (wall_adjoint.py:757).
  The second one uses `remaining=None`, so its search length is the full `reach = 0.5`. Both cost
  the same 58 evaluations. On top of these: lane passes on B = max(256, N/32) = 1883 slots
  (state.py:581-587), `inside` (1 `value`, :786), and in pass 0 `_apply_events` over all N, which
  includes a level-set normal (`value_and_grad`, via `surface_normal`, wall_adjoint.py:314) and
  5N float64 random draws (:381-383).
* The entrants (~150 per step, `capacity` about 434) get their own `advance` call every step
  (inflow.py:409), with the same structure: pass 0, lane, and the post-loop query. That is about
  3 more `hit_body` calls, each with 56 sequential loop iterations, on tiny arrays.
* Per-step sequential loops: `flux_quantile` has 64 bisection iterations (inflow.py:245), the NTC
  scan has `A_max = 4` (ntc.py:469), and there is one `argsort` for the cell table (sort.py:52) and
  another `argsort` for dead slots (inflow.py:368).
* Warm-up memory is O(N), not O(N x steps) [V]: `capture=False` gives no record
  (simulate.py:347-348), `emit_moments="none"` (ensemble_mesh.py:232, simulate.py:394-400), and the
  per-step outputs are only scalar diagnostics plus a (2D+E)-row tally. `jax.checkpoint`
  (simulate.py:555-556) does nothing in a forward that is not differentiated. The captured run's
  memory is the stacked `EventRecord`.

**Where the time goes: two readings of the evidence, both [E].**
(a) Throughput. tracker_bench.json has the dense stencil (4 passes, so 5 full-N searches per step)
at 115 s and the lane tracker (2 full-N searches) at 55 s. Solving 115 = a + 5s and 55 = a + 2s
gives s ≈ 20 s per full-N search per 1000 steps. On that reading the two full-N searches are about
40 of the 55 s.
(b) Latency. There are about 56 x (1 + ~1.5 + 1 + ~3 for entrants) + 64 + ~10 ≈ 440 sequential
loop iterations per step, each costing several kernel launches. The mode2 seed job on an A100
(192 k slots, 4x the particles of the 3050 Ti run and about 100x its float64 rate) spent at most
about 264 s outside backward and shape: 928 s total, minus `seconds_backward` 164 and
`seconds_shape` 500 (…grid2_mode2…_seed0.json:42-43,85). That 264 s covers five forwards (one
captured plus four FD), the replay and the compiles, so about 50 s per forward. That is roughly the
3050 Ti's time at a quarter of the particles, which points to launch/loop overhead rather than
FLOPs.
Which reading dominates decides whether item 1 or item 3 comes first. That is the first measurement
to take (section 3, M1).

## 1. Ranked opportunities

| # | Opportunity | Where | Expected gain (reasoning) | Correctness / bitwise / AD / RNG risk | Effort |
|---|---|---|---|---|---|
| 1 | **Near-body cull of the hit search.** Run the bracket only for slots with `-tol <= phi0 <= L*len`, where `L` is a rigorous bound on `|grad phi|` taken from the coefficient differences. Compact them with the existing `lane_slots`, `lax.cond` to the dense path on overflow, and skip the search entirely when no slot is near (this covers the entrants) | levelset.py:212-253; used via geometry.py:263-264 for every call site | Pass 0: `len = |v| dt ≈ 0.035`, `L ≈ 1.4`, so the band is about 0.05 around a perimeter of about 19. Area about 0.95 of 71.7, i.e. **1-3 % of slots** (more on the windward side). Evaluations per step fall from about 116 N to about 6 N (phi0 plus the dense Newton step plus `inside`), so the **hit search is about 15-20x cheaper**. On reading (a): forward 55 s -> about 20 s. The entrant searches disappear altogether (the inflow face is ≥ 2 away from the body) | **Exact:** a culled slot provably has no `+ -> -` sample pair, so the answer is `inf` exactly as now. Deep and `hit_now` slots are handled as in :223-253. Per-slot arithmetic is unchanged, so results are bitwise identical up to XLA fusion/FMA differences in the last bit. AD: the bracket is already `stop_gradient` (:236-242) and the Newton step stays dense, so the tangents are unchanged. No random draws are involved | ~60 lines + tests |
| 2 | **Restrict the post-loop query to the lane when `epsilon == 0`**, via `lax.cond(epsilon > 0, dense, lane-only)`. With `epsilon = 0`, `upper` is unused (`time = dt` and `logp = 0`, wall_adjoint.py:25-43), so only `exhausted` needs `distance`, and only lane slots can be exhausted | wall_adjoint.py:757-760 (lane branch) | Removes one of the two full-N searches: about -20 s of 55 on reading (a) on its own. With #1 in place the query would otherwise still cover a band of about 0.7 (`reach` times `L`), about 20 % of slots; this cuts it to the lane (0.5 %) | One behavioural corner case goes away. Today a non-lane slot whose coarser final sampling (spacing 0.031) finds a crossing that pass 0's finer sampling missed is deleted as stuck. After the change it would not be. Gate: M3 (expected count 0). AD: a `cond` is differentiable, and with `upper = inf` there are no NaNs (`fu = 1`) | ~25 lines |
| 3 | **Unroll the fixed-trip loops** (`fori_loop(..., unroll=True)` or Python loops): `_bracket`'s 16 + 40 iterations, `flux_quantile`'s 64 (inflow.py:245), optionally the 4-slot NTC scan. Also make `_contract` (levelset.py:107-113) explicit 4-term sums so each evaluation stays elementwise and XLA can fuse the whole chain into one kernel | levelset.py:174-179, inflow.py:245 | On reading (b): about 440 sequential loop iterations per step become about 30, i.e. **~400k fewer loop iterations per forward**. Each is several launches, 10-30 µs by estimate, so **~5-15 s per forward**, more under WSL. Unrolling also removes the carry round-trips through memory (4 f64 arrays per iteration) | The math is unchanged. Fusion can change FMA contraction, so expect last-bit differences, which `REPLAY_TOL = 1e-9` (ensemble_mesh.py:63) absorbs. The replay compiles the same code. AD: the bracket is stop-gradient and `flux_quantile` has a `custom_jvp`, so neither is affected. Cost: compile time and HLO size (8 call sites x 58 evaluations), to be measured (M6) | trivial code, measure compile |
| 4 | **Cut `n_bisect` 40 -> ~20.** The Newton step at :244-249 then polishes a bracket of `ds/2^20 ≈ 2e-9`, leaving an error of order `κ·(2e-9)^2`, which is below round-off | static `LevelSet.n_bisect` (types.py:122; levelset.py:384, 401, 498) | 58 -> 38 evaluations per call (**-35 %** of the search, as long as the search is still dense) | The primal changes at round-off level, so runs are not bitwise identical to old ones. `n_bisect` is a static field shared by forward and replay, so they stay consistent. Gate: M4 | config |
| 5 | **Slim the stored `EventRecord`**, which is the captured-forward peak. At 122 B/slot/step (x 16, v 24, alive 1, born_face 4, born_v 24, v_post 24, head face 4 / v_out 24 / diffuse 1), this matches the measured 36.7 kB per slot per 300 steps (ADJOINT_BRANCH_REVIEW 7.10). Changes: (i) drop `v_post`, since it equals `v` with the `PairRecord`'s accepted `vi_n/vj_n` scattered in slot order (ntc.py:350-352, and the record already stores them, :353-362); (ii) store `born_v` as a list of at most `capacity` entries instead of a dense (N,3); (iii) store `head.v_out` only for slots with a pass-0 event, a lane-width list; (iv) store face ids as int8/int16 | simulate.py:164-179, 82-106; inflow.py:475 | 122 -> ~46 B/slot/step, i.e. **about -60 % of the record and about -55 % of the captured-forward peak**. At FINE (240 k slots) this is the difference between the logged OOM on the 24 GB slice (logs/opt_opt_S1_gd_32924894.out: "ran out of memory trying to allocate 2.77GiB") and fitting | Forward values are untouched. Consumers must rebuild the dropped fields exactly: ensemble_device.py:202, 627-629; ensemble_backward.py:547, 596, 645-646, 681-682, 748; shape.py:463, 487, 732. RNG and AD are not affected | medium (consumers) |
| 6 | **Capacity N closer to the alive count.** There are 60 256 slots for 48 109 alive (tracker_bench.json), so 20 % of every per-slot kernel and of every record row is dead weight | forward.py:63; ensemble_mesh.py:210 | Sizing N at about 1.1 x the peak alive count (~53 k) cuts **~12 % of work and of record memory**. #1 already skips dead slots in the search | Overflow is validated: it shows up as `shortfall` (inflow.py:465, validation.py:28). The non-indexed draws in the wall passes depend on N (wall_adjoint.py:697, 382-383), so results change once and the contract is kept | config |
| 7 | **Dead-slot `argsort` in `inject` -> cumsum compaction.** The stable argsort of `alive` only feeds `order[rank]` for `rank < n_dead` (inflow.py:368, 453), which is exactly `lane_slots(~alive, capacity)` (wall_adjoint.py:196-208) | inflow.py:368 | Replaces one full-N sort per step with a scan and a scatter. Small, probably under 1 ms per step | Bitwise identical (same dead slots in the same order). No effect on AD or RNG | ~10 lines |
| 8 | **Skip the cell sort when there are no attempts.** `table.order` is read only at ntc.py:309-310. In a collisionless flow, `n` (ntc.py:458) is almost always 0 and depends only on `counts` from the bincount. Compute counts and `n` first, then `lax.cond(any(n > 0), argsort, dummy)` | sort.py:50-55, simulate.py:326, ntc.py:458 | One N-sized sort per step (the size is set by M1) | Exact, because an inactive slot addresses the scratch row. Needs `build_table` split into counts and order | small-medium |
| 9 | **Lane-pass draws are O(N).** Every lane pass that has a hit draws 5N float64 values and gathers B of them (wall_adjoint.py:380-385). Indexed draws at the lane's slots (`rng._normal_at` via `fold_in(key, i)`, rng.py:94-118) would be O(B) | wall_adjoint.py:380-385 | A few full-N RNG kernels per step. Small | The streams change once (not bitwise identical to old runs). The contract `key_for(base, step, stage)` still holds, and a lane draw stays equal to the dense pass's draw for that slot | small |
| 10 | **Entrants' dense (K, N) `born_walls` in captured runs.** `no_wall_events(n_part, 32)` is zeroed and scattered every captured step (inflow.py:365, 476-480) and then read back by `_merge_born_split` (simulate.py:92-103). Keep it at `capacity` width | inflow.py:365, 476-480 | 32 x 60 k x 29 B = **56 MB per step** of memset and scatter (224 MB at FINE). A transient, not stored. Minor for time | Exact | small |
| 11 | **Recompiles and flags.** (i) `_run` treats `step0` as static (ensemble_mesh.py:216), so frames mode compiles once per 50-step block: 14 extra compiles per process (:287-289, optimize.py:83). Make `step0` traced; simulate.py:559/562 only do arithmetic on it. (ii) The optc logs show iterates costing an extra **310-380 s** (S1 iterates 80, 83, 92, 96: 580 s vs 270 s; S2 iterate 209; S8 iterate 58). This is a recompile of the chain. The likely cause is `npc_max` (ensemble_mesh.py:450-454), which is data-dependent and static in `backward_virtual`/`shape.gradient` (ensemble_device.py:656, shape.py:810). `grid.n_sub` is also possible. Bucketing `npc_max`, e.g. to multiples of 64, would fix the first. (iii) Command buffers are disabled unconditionally, including on the Linux cluster (ensemble_mesh.py:38, forward.py:29), although the crash was WSL-only | as cited | (i) is a one-time per-process cost. (ii) is about 5-10 % of optc wall time. (iii) is unknown | None, apart from the compile time itself | small |
| 12 | float32 (not recommended now) | levelset.py:47, 222-233 | Float64 runs at 1/64 rate on the 3050 Ti and on the RTX PRO 6000, but after #1 very little float64 search work remains | The on-surface/leaving logic (`SURFACE_TOL = 1e-8 h`) and the ulp amplification on concave walls (7.10) need float64. The record's x/v must stay float64. At most, float32 could serve as a conservative pre-filter | - |
| 13 | NTC scan: not worth it | ntc.py:294-296, 461-469 | The documented floor is about 40 µs per slot, x 4 slots, about 0.16 ms per step, i.e. under 0.2 s per forward | - | - |

## 2. Implementation sketches (top 3)

### 2.1 Near-body cull (#1)
```python
def gradient_bound(ls):                     # levelset.py; traced, cheap (coef diffs)
    # d_a phi = sum (Delta_a c)/h * (quadratic x cubic B-spline weights >= 0, summing to 1)
    per = [jnp.max(jnp.abs(jnp.diff(ls.coef, axis=a))) / ls.h for a in range(ls.dim)]
    return jnp.sqrt(sum(p * p for p in per))

def hit_body(x, v, ls, remaining=None, active=None, near_size=0):
    ...  # speed, vhat, length, ds, phi0/g0, tol, on_surface, hit_now, deep, phi_start (as now)
    L = jax.lax.stop_gradient(gradient_bound(ls))
    near = moving & (phi0 >= -tol) & (phi0 <= L * length * (1 + 1e-12) + tol)
    if active is not None: near &= active                       # dead slots out
    if near_size:                                              # static
        slot, over = lane_slots(jax.lax.stop_gradient(near), near_size)
        take = jnp.maximum(slot, 0)
        def compact(_):
            f, s = _bracket(sg(ls), sg(x)[take], sg(vhat)[take], sg(ds)[take], sg(phi_start)[take])
            put = jnp.where(slot >= 0, slot, n)
            return (jnp.zeros(n, bool).at[put].set(f & (slot >= 0), mode="drop"),
                    jnp.zeros(n).at[put].set(s, mode="drop"))
        def none(_): return jnp.zeros(n, bool), jnp.zeros(n)
        def dense(_): return _bracket(...)                        # today's call
        branch = jnp.where(jnp.any(over), 2, jnp.where(jnp.any(near), 1, 0))
        found, s_b = jax.lax.switch(branch, (none, compact, dense), None)
    ...  # Newton polish and return exactly as levelset.py:244-253 (dense, differentiable)
```
Thread `active` (alive) and a static `near_size` through `geometry.first_hit` (geometry.py:263-264)
and `_next_surface` (wall_adjoint.py:562-567). A good default is `near_size = max(256, N // 8)`,
following the lane sizing rule. The `switch` also removes all search work for the entrants. The
forward and the traced replay go through the same `hit_body`, so their discrete decisions stay
identical. Tests: `hit_body` with and without the cull on random rays near and far from a perturbed
level set must match bitwise, including the overflow and empty branches, and the existing lane and
replay tests must pass.

### 2.2 Lane-only post-loop query when `epsilon == 0` (#2)
In `_advance_impl`'s lane branch (wall_adjoint.py:756-760), `_lane_passes` already returns the
lane's `slot` (:592). Return it, then:
```python
def dense(_):  return _next_surface(x, v, grid, mesh, exclude, None)[0]
def lane(_):
    d, _ = _next_surface(x[take], v[take], grid, mesh, exclude[take], None)
    return jnp.full(n, jnp.inf).at[put].set(d, mode="drop")
distance = jax.lax.cond(epsilon > 0.0, dense, lane, None)
```
`exhausted = alive & (elapsed + distance <= budget)` is unchanged. `upper = inf` for non-lane
slots is harmless when `epsilon = 0`. `epsilon` stays traced, so no API change is needed. Gate M3
confirms that the old dense query never flagged a non-lane slot.

### 2.3 Unrolled, fused search loops (#3)
* levelset.py:174-179: `jax.lax.fori_loop(1, ls.n_sample + 1, ..., unroll=True)` and the same for
  the bisection. The trip counts are static fields (types.py:121-122). Rewrite `_contract` as
  `sum(w[:, k, None] * block[:, k] for k in range(4))` per axis, so no `reduce` op splits the fusion.
* inflow.py:245: `fori_loop(0, 64, bisect, ..., unroll=True)`. The `custom_jvp` (:250-256) is
  unaffected.
* Check the HLO dump (`XLA_FLAGS=--xla_dump_to`) to confirm that `_bracket` becomes one loop fusion
  per call. If compile time grows too much, try `unroll=8` as a compromise, which already cuts the
  loop iterations 8x.

## 3. Measure first (gates, in order)

* **M1: profile 20 steady captured and uncaptured steps** on the MIG slice (`jax.profiler.trace`
  or nsys). Record kernels per step, GPU-busy fraction, and time inside `_bracket` and
  `flux_quantile` versus everything else. This decides between readings (a) and (b), i.e. whether
  #1/#2 or #3 comes first. It is also worth one run with command buffers enabled on the cluster
  (#11 iii).
* **M2: near fraction (numpy only, no JAX).** On a stored optc record, or `coef_k.npy` plus
  recorded x and v, evaluate phi0 and `gradient_bound`, and count alive slots with
  `phi0 <= L |v| dt` (pass 0) and `phi0 <= L reach`. Do this for the circle and for the roughest
  iterate. Gate: #1 is worth building if the pass-0 fraction is ≲ 10 %. I estimate 1-3 %. Also check
  how tight `L` is (≈ 1.4 expected; larger on rough shapes).
* **M3: stuck events outside the lane.** Over a 1000-step forward, count slots flagged `exhausted`
  by wall_adjoint.py:759 that were not in the lane. Gate for #2: 0, or a documented rate.
* **M4: `n_bisect` 20 vs 40.** Compare max |Δt_hit|, the drag tally, `replay_error` and one seed's
  gradient. Gate: differences ≤ 1e-12 relative.
* **M5: recompile cause.** Run 20 optc iterates with `jax_log_compiles` and
  `jax_explain_cache_misses` and identify what changes on the +310-380 s iterates (`npc_max`?
  `n_sub`?).
* **M6: unroll microbenchmark.** Time `hit_body` alone on 60 k synthetic rays: compile time and run
  time for `unroll` = False / 8 / True.
* **M7: memory.** Rerun `scratch/mesh/probe_memory.py` (the per-slot-bytes probe behind
  tracker_bench.json) before and after #5 and #6, reporting per-slot bytes of the record and of the
  captured-forward peak.

The project's caution applies here too (see the memory notes): run M2, M3 and M4 on stored
artifacts before touching the tracker, and do not run them concurrently with other JAX processes.
