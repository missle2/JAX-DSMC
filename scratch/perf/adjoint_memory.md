# Adjoint chain memory: capture + backward + shape VJP (static analysis, 2026-09-28)

Static reading only. Nothing was run. Tags: **[V]** = verified by reading code or result files;
**[E]** = estimated (arithmetic on code-derived shapes, or a hypothesis still to be measured).
Reference case: the optc circle, `fnum_scale 4`, N = 15256 slots, M = 300 captured steps, D = 2,
K_bounce = 32, lane tracker, variant `along_pool` (examples/mesh/optimize.py:75), measured in
examples/mesh/results/tracker_bench.json ("lane_32_compact") and scratch/mesh/probe_memory_stages.log.

## 0. Measurements used [V]
- Records: 198.9 / 336.1 / 611.4 MB at N = 4006 / 7756 / 15256. Slope 36.67 kB/slot, intercept 51.9 MB.
- Chain peak: 525.6 / 766.6 / 1243.0 MB. Slope 63.7 kB/slot (the headline figure is a slope, not
  peak/N; 1243 MB / 15256 = 81.5 kB/slot), intercept about 270 MB.
- Stages at N = 15256 (probe_memory_stages.log): forward live 726.6 MB; after replay live 871.9 (+145);
  backwards live 880.1, then 920.7; shape gradient peak 1243.0 against about 884 live, so the spike is about 360 MB.

## 1. Per-slot budget as the code stands
Shapes come from simulate.py:56-78 (EventRecord), wall_adjoint.py:104-153 (RecordEvents, compact_record),
collide/ntc.py:268-284 (PairRecord), state.py:95-114 (x is `f64[N, D]`, v is `f64[N, 3]`), forward.py:145
(A_max = 4), forward.py:57 (C = 20x20), state.py:581-587 (lane = max(256, N/32)), simulate.py:109-128
(born lane = lane/4, list capacity E = (lane + born)/4).

| array (stacked over M) | dtype, shape | B/slot-step | kB/slot | live | read by shape chain? |
|---|---|---|---|---|---|
| rec.x | f64[M,N,2] | 16 | 4.8 | forward to end | yes: replay carry, fits, mismatch [V] shape.py:463,481-488,644 |
| rec.v | f64[M,N,3] | 24 | 7.2 | ” | yes: mismatch shape.py:486-487, fits :644, staged nodes ensemble_device.py:627 |
| rec.v_post | f64[M,N,3] | 24 | 7.2 | ” | yes: replay carry shape.py:463,737; backward stage ensemble_device.py:454 |
| rec.born_v | f64[M,N,3] | 24 | 7.2 | ” | **no**: only host `from_device` ensemble_backward.py:596,748, skipped (`host_adjoint=False`, ensemble_mesh.py:378-402) |
| walls.head.v_out | f64[M,1,N,3] | 24 | 7.2 | ” | **no**: host only (ensemble_backward.py:624-700) |
| walls.head.face | i32[M,1,N] | 4 | 1.2 | ” | only as `face >= 2D` in `hit_ahead` shape.py:917-921 (1 bit needed) |
| rec.born_face | i32[M,N] | 4 | 1.2 | ” | only as `>= 0` (shape.py:448, ensemble_device.py:188) |
| rec.alive, head.diffuse | bool | 1+1 | 0.6 | ” | alive yes; diffuse no |
| list slot/pass_/face/v_out/diffuse | [M,E], 37 B/entry | ~0.36 | ~0.1 | ” | slot/face only (hit_ahead) |
| **record, per slot** | | **122** | **36.6** | | measured slope 36.67 kB [V] |
| pairs pi,pj,accept,vi,vj,vi_post,vj_post | [M,4,C], 105 B/attempt | fixed | 50.4 MB | ” | virtual backward reads pi,pj,accept,vi_post,vj_post (ensemble_device.py:431-455,622); **vi, vj unread** |
| sigma | f64[M,C] | fixed | ~1 MB | ” | yes (ensemble_device.py:624) |
| **fixed part of the record** | | | **51.4 MB** | | measured intercept 51.9 MB [V] |
| Replay.hit_reward head | f64[M,1,N] | 8 | 2.4 | replay to end | yes |
| Replay.hit_reward tail | f64[M,31,B], B = N/32+N/128 | ~9.7 | 2.9 | replay to end | yes, but 31 rows are almost all zero |
| Replay.timing_reward | same | ~17.7 | 5.3 | replay to shape_terms:468 | replaced by a total |
| timing_total | f64[M,1,N] | 8 | 2.4 | ensemble_mesh.py:467 to :491 | 2nd backward |
| dev.lam, dev_t.lam | f64[M,N] x2 | 16 | 4.8 | backward to end | yes (shape.py:902-904) |
| Backward.four_term | f64[M,4,C] zeros x2 | fixed | 7.7 MB | ” | never (ensemble_device.py:718-719) |
| klass (hit_ahead) | bool[M,N] (+i32 cumsum temps) | 1 (+4 to 8) | 0.3 (+1.2 to 2.4 transient) | gradient jit | yes |

Per-step transients inside the gradient scan (not stacked) [E]:
- The VJP residuals of one step's `wall_map` (shape.py:634). Pass 0 runs over all N: the Newton polish
  `value_and_grad` (levelset.py:241) gathers a `(N,4,4)` f64 block, 128 B/point, plus weights. About 0.5-1.5 kB/slot.
- Node tables `(C, 9*npc_max, ...)` from `pooled_tables` (ensemble_device.py:329-345): about 20 MB at npc_max = 119.
- **The fit chunk working set** (see section 3): `chunk * 9 * npc_max * (60-80 B) + chunk * k * (~350 B)`.

What coexists at the peak (during `gradient` at N = 15256) [V shapes / E transients]: records 611 MB,
hit_reward 82 MB, lam x2 73 MB, klass 5 MB, four_term 8 MB, fit chunk about 300-360 MB. That adds to
about 1.1-1.2 GB against the measured 1243 MB. In slope terms, the M-stacked arrays give
36.7 + 5.35 + 4.8 + 0.3 = **47.2 kB/slot**. The other ~16.5 kB/slot of the 63.7 must come from per-step
O(N) working sets and from the fit spike growing with npc_max, which is density-dependent
(ensemble_mesh.py:481-485) [E].

Captured forward: its peak slope is 46.0 kB/slot against 36.7 for the record, so about 9.4 kB/slot is transient.
My hypothesis [E] is the entrants' dense birth-step events. With capture on, `inflow.inject` builds
`born_walls = no_wall_events(N, K)`, which is (K = 32, N) i32 + (K,N,3) f64 + bool = 928 B/slot per step
(inflow.py:365, 476-479). `_merge_born_split` then gathers from it (simulate.py:82-106). About 10 live
copies would account for 9.4 kB/slot. This is the next peak once the chain is slimmed.

## 2. Ranked reductions
Bytes are per slot at the chain peak unless marked fixed. "Exact" means bitwise-equal replay inputs.

| # | what (file:line) | saves | exactness | effort |
|---|---|---|---|---|
| 1 | **Stream the record through host in blocks of b steps** (forward in blocks via `step0`, as `frames_every` already does, ensemble_mesh.py:284-292; replay/backward/gradient loop over blocks with carried state) | records 36.7 to about 2b/M x 36.7 (b = 20: about 2.4); or 13.0 to 0.9 after #2 | exact: RNG contract keys per step; backward carries (mu, key), ensemble_device.py:617 | medium |
| 2 | **Lean "shape" capture**: drop born_v, head.v_out/diffuse, list v_out/diffuse/pass_; head.face replaced by a `body_hit` bool (or klass emitted by the replay); born_face to int8/bool; **v_post rebuilt from v + pairs**; PairRecord drops vi, vj (simulate.py:131-180, ntc.py:350-358) | born_v 24 + head.v_out 24 + head.diffuse 1 + head.face 4 to 1 (3) + born_face 4 to 1 (3) + v_post 24 = 79 B/slot-step; record 122 to 43 B = **36.6 to 12.9 kB/slot (-23.7)**; pairs 105 to 57 B/attempt (-24 MB fixed at C = 400, -97 MB at grid 2) | bitwise: v_post and vi/vj are copies, rebuilt by replaying ntc.py:311,319-320 writes in slot order | low-medium |
| 3 | **Bound the fit working set**: chunk sized so `chunk*P <= 2^20`; npc_max = max_occ + 8 instead of the `4*alive/C` floor (ensemble_mesh.py:481-485); compact pooled tables to the real 3x3 occupancy (ensemble_device.py:329-345); pre-scale node tables once (ensemble_device.py:101-103) | spike ~300-360 MB to ~50-80 MB now; **several GB at 240k slots** (section 3) | each query independent; only summation order can change (~1e-16); replay untouched | low (chunk) / medium (pool) |
| 4 | **Fuse replay + both backwards + shape VJP in one reverse scan** (ensemble_mesh.py:442-520): per step replay, then r_k/tau_k, then backward step (mu, mu_t), then lam_k, then VJP; hit_ahead becomes a carried bool[N] | hit_reward 5.35 + lam x2 4.8 + klass 0.3 + timing 2.4 (earlier stage) = **~10.5 kB/slot**; one wall-map replay instead of two | same ops; the replay check is known only at the end (then raise) | high |
| 5 | Compact the replay's tail rewards as a list, like RecordEvents (shape.py:443-466, `HitRewards`) | 2.9 kB/slot (5.8 at the replay stage) | bitwise (scatter back per step) | low-medium |
| 6 | Entrants' birth-step events on the candidate lane `(K, capacity)` instead of `(K, N)` (inflow.py:365, 476-479) | ~9 kB/slot of captured-forward transient [E] | bitwise | low-medium |
| 7 | One VJP per step with a summed cotangent when components are not needed. The optimiser uses only pathwise+score (optimize.py:154) but pays 5 transposes (shape.py:670-707) and the timing term it discards | time: VJP transposes 5 to 1 (or 4 to 1); memory neutral | linear: sum order ~1e-16 | low |
| 8 | custom_vjp on `levelset.value_and_grad` w.r.t. coef that saves only x (recomputes the index/weights) (levelset.py:81-131) | per-step residual 128 B/point (2D), 512 B/point (3D) to 16-24 B; ~0.5-1 kB/slot transient in 2D, 2-4 in 3D [E] | exact math; ulp-level | medium |
| 9 | Drop Backward.four_term zeros under the virtual pull-back (ensemble_device.py:719) | 7.7 MB fixed (31 MB at grid 2) | n/a | trivial |
| 10 | lam, lam_timing in f32 | 2.4 kB/slot | **not exact**: values at 1e-7 relative, far below MC noise, but not bitwise | low; optional |
| - | x or v in f32 / int16 | rejected: f32 eps 6e-8 exceeds REPLAY_TOL = 1e-9 (ensemble_mesh.py:63); it would also flip grazing hits (review 7.10 amplification) | - | - |
| - | x kept only on event slots (others as x + v dt) | 16 B/slot-step, but the replay would chain from its own output, so ulp drift could flip cell/hit outcomes. Not recommended | risky | - |

Combined [E], N = 15256: #2+#3+#5 give about 12.9 + 2.4 + 4.8 + 0.3 + ~3 per-step, roughly **24 kB/slot + ~100 MB fixed**
(vs 63.7 + 270 MB). Adding #1 and #4 leaves **a few kB/slot**, and the captured forward (with #6) becomes the peak.

## 3. Origin of the ~300 MB shape-VJP spike and removal
[V] `shape.gradient` defaults `chunk=4096` (shape.py:849,884), and `shape_terms` never passes a chunk
(ensemble_mesh.py:493-520). The optimiser's variant is `along_pool`: pool = 1, k = 96 (shape.py:942-947).
`node_gradients` (ensemble_device.py:347-397) widens the node tables to the 3x3 block, so
P = 9 * npc_max (ensemble_device.py:329-345). `cell_query` then `lax.map`s chunks of 4096 queries
(ensemble_device.py:296-310). Per chunk, `_chunk_query` and `field_query` materialise the following
(ensemble_device.py:253-270, 101-121):
- `node_v[c]` (Q,P,3) f64;
- `pts` (Q,P,5) f64, used twice so it is not fused away;
- `d2` (Q,P) f64;
- `top_k` over P (with k = 96, likely a sort with key+index copies);
- `node_val[c]`, valid/class masks;
- the k-sized design (Q,96,14) f64 = 44 MB, and its weighted copy for the einsum.

[E] That is about 60-80 B per (query, node) plus ~100-140 MB fixed per chunk. At npc_max 64 to 119,
P = 576 to 1071, giving 2.4 to 4.4 M (query, node) pairs and about 250-450 MB. This matches the measured
about 360 MB and the review's own attribution (ADJOINT_BRANCH_REVIEW.md §7.10). The backward fits
(pool 0, k 32, chunk = min(4096, pow2 N), ensemble_inlet_shock.py:199-201) are 9x smaller, which matches
the ~85 MB backward transient.

**The spike is not fixed. It scales with particles per cell.** npc_max >= 4*max_alive/C
(ensemble_mesh.py:481-485). At the fine circle (240k slots, about 190k alive, C = 400) that gives
npc_max ~1900, P ~17 100, and about 70 M pairs per chunk, **~4-6 GB** [E]. The review's "≈ 15 GB at 240k"
(ADJOINT_BRANCH_REVIEW.md §7.10) assumed a fixed 300 MB. Grid 2 (C = 1600, same density) is not
affected in P; only the 4x N matters.

Removal (all exact up to summation order):
- (a) `chunk = clip(2**20 // P, 128, 4096)`, passed through `shape.gradient`, `value_query` and the backward.
- (b) Pre-scale the node table once (`pts = concat(x,v)/scale`, shape (C,P,5)) and gather rows from it.
  Derive `same_side` from the v-columns of `pts`, so node_v is not gathered twice.
- (c) npc_max = max_occupancy + 8 (already computed, ensemble_backward.py:775-781), not 4x the mean.
- (d) Compact pooled tables: width = max over cells of the 3x3 occupancy sum (host-computed like
  max_occupancy, static). Build them from the sorted order with per-neighbour offsets. This gives
  about 3x smaller P, with the same cut in kNN time.

A jax.checkpoint does not help here. The fits sit outside the `jax.vjp` (shape.py:644 vs :634), so nothing
is saved for a reverse pass; the peak is pure forward working set.

## 4. gradient_variants
[V] `gradient_variants` (shape.py:949-959) calls the jitted `gradient` once per variant. Each call:
- recompiles (pool, k static; lam_timing None vs array);
- re-runs `hit_ahead` (shape.py:899);
- per step: re-traces the wall map with `jax.vjp`, re-checks the mismatch, redoes the 3 variant-independent
  transposes (tally, score, timing) plus the silhouette term;
- redoes the fit.

The optimiser runs one variant (optimize.py:150-153). `ensemble_gradient` runs all four (ensemble_mesh.py:486).

Sharing:
- the replay/VJP linearisation, mismatch, klass, g_tally, g_score, g_time and g_sil once;
- **2 fits, not 4**: "one"/"along" share (pool 0, k 32), "one_pool"/"along_pool" share (1, 96); `along_flight`
  (shape.py:494-503) is a cheap post-process of gx;
- gv is identical within each pair;
- transposes: 20 to 3 + 4 (d_x) + 2 (d_v), or one vmapped transpose over the stacked cotangents.

About 3-3.5x less shape time for the 4-variant runs. Memory is unchanged if the variants run inside one step
(only 4 gx/gv of (N,5) each).

## 5. Top 3 changes, sketched
**A. Lean shape capture (#2, +#5).** Add `capture="shape"` to `simulate.run` / `_event_record`
(simulate.py:131-180):
- record x, v, alive, `born = born_face >= 0` (bool) and `body_hit` = any head/list face >= 2D (bool[N]);
- PairRecord keeps pi, pj, accept, vi_post, vj_post (ntc.py:350-358);
- no born_v, v_post or WallEvents.

Add `post_collision_v(rec_k)` in ensemble_device, the exact replay of ntc.py:311,319-320:
```
v = concat(rec_k.v, zeros((1,3)))                     # scratch row N
for a in range(A_max):                                # static, slot order = forward order
    pi, pj, acc = p.pi[a], p.pj[a], p.accept[a]
    v = v.at[pi].set(where(acc[:,None], p.vi_post[a], v[pi]))
    v = v.at[pj].set(where(acc[:,None], p.vj_post[a], v[pj]))
return v[:N]
```
Use it in `_replay_step` / `_gradient_step` carries (shape.py:463,737) and in `staged_velocities`' last
stage (ensemble_device.py:454). The pre-collision partner velocities vi/vj are `v[pi]` / `v[pj]` inside the
same loop, if the sequential host path ever needs them.

`hit_ahead` reads `body_hit` (or the replay emits it, since it recomputes the events at shape.py:447-449).
Keep the full capture for host-adjoint users. Gate: the reconstruction equals the stored v_post **bitwise**
on a full record before switching.

**B. Bounded fits (#3).** Change `cell_query(..., chunk)` callers to pass `chunk = fit_chunk(P, k)`.
In `node_gradients` / `value_query`:
- build `pts_tab = concat(x[ids], v[ids]) / scale` once;
- `_chunk_query` gathers `pts_tab[c]`;
- `same_side = pts_tab[c][..., -3:] @ (direction*scale_v)`.

In `shape_terms`, set npc_max = max_occupancy + 8, and add pooled width = host max of 3x3 sums. Then
replace `pooled_tables` with an offset-based compact gather.

**C. Host-streamed records (#1), optionally as the carrier of D (#4).** Forward: a Python loop
`for k0 in range(warmup, warmup+M, b): _run(..., n_steps=b, step0=k0, capture="shape")`, then
`np.asarray` (or `device_put` to a `pinned_host` memory kind; API unverified on jax 0.6.2, so the numpy
fallback is safe).

Consumers become block loops over jitted scans with the same bodies:
- replay: forward block order, carry (x, v, alive, err);
- backward_virtual: reverse block order, carry (mu, key). The key sequence is unchanged, so the result is bitwise;
- gradient: carry g's; klass from a reverse block pass with a carried `hit_later` bool[N].

Prefetch block j±1 while j runs. Transfer per pass is about 13 kB/slot after A. At 240k slots that is
3.1 GB, about 0.2-0.3 s at PCIe 4 per pass [E], negligible against 213 s per chain at 15k.

Device residency becomes about 2b/M of the record plus per-step working sets. Fusing D into this loop also
removes the stacked lam/hit_reward.

## 6. Measure first (GPU, one process, `XLA_FLAGS=--xla_gpu_enable_command_buffer=`)
1. **Spike attribution**: `probe_memory_stages.py` at fs = 4 with chunk ∈ {4096, 1024, 256} x variant
   ∈ {along, along_pool}, printing npc_max. Alternatively, `gradient.lower(...).compile().memory_analysis()`
   temp bytes, with no execution. Expect the spike ∝ chunk x P.
2. **Record leaf table** (CPU `eval_shape`, record_budget.py adapted to lane + levelset): confirm 122 B/slot-step
   and the 51.4 MB fixed pair part.
3. **Bitwise v_post / vi,vj reconstruction** on one fs = 16 record (gates #2).
4. **Stage times** at fs = 4: replay, each backward, gradient, and gradient with 1 vs 5 VJP calls (sizes #4/#7).
5. **Captured-forward transient**: `_run` memory_analysis with capture at K_bounce 4 vs 32. If the 9.4 kB/slot
   scales with K, the born_walls hypothesis (#6) holds.
6. **npc_max and 3x3 occupancy at fs = 0.25** (host count on a final state): sizes the fine-case spike before
   anyone books a 24 GB slice.
