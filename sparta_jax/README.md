# sparta_jax: SPARTA's algorithm, one to one, in JAX

sparta_jax is a JAX forward simulation (float64, not differentiable) of SPARTA's circle deck, `sparta_jax/deck/in.circle_ar`. The deck:

* argon gas on a 10 x 10 box with 20 x 20 cells;
* a circle of radius 3, read from `data.circle`;
* nrho 1, fnum 1e-3, vstream 100, T 273.15, dt 1e-4;
* inflow `emit/face xlo twopass`;
* surface model `surf_collide diffuse 300 0`, which is always specular;
* `collide vss`;
* boundaries `o r p`;
* 1000 steps from an empty box.

The code follows SPARTA's routines step by step. This has two uses: comparing **C++ against XLA for the same algorithm**, and splitting the main code's slowdown into the part due to XLA and the part due to our own algorithmic additions.

| Path | Contents |
|---|---|
| `sparta_jax/geometry.py` | setup on the host |
| `sparta_jax/sim.py` | the time step |
| `sparta_jax/rng.py` | counter-based uniform random numbers |
| `examples/sparta_jax/bench.py` | throughput benchmark |
| `examples/sparta_jax/profile.py` | per-stage profile (named scopes and an HLO dump, read with `xspace.py`) |
| `examples/sparta_jax/validate.py` | comparison with SPARTA |
| `examples/sparta_jax/check_emit.py` | unit check of emit/face |
| `examples/sparta_jax/plot.py` | timing figure |
| `tests/test_sparta_jax.py` | 8 tests, about 15 s |
| `examples/sparta_jax/sparta_ref/` | SPARTA reference runs, seeds 101-108 |
| `examples/sparta_jax/results/` | all outputs |

```
run_locked.sh gpu python3 examples/sparta_jax/bench.py --tag gpu --set sort=full
run_locked.sh cpu env XLA_FLAGS=--xla_cpu_multi_thread_eigen=false taskset -c 3 \
    python3 examples/sparta_jax/bench.py --tag cpu_1
run_locked.sh gpu python3 examples/sparta_jax/validate.py --seeds 8
run_locked.sh cpu python3 -m pytest -q tests/test_sparta_jax.py
```

## Algorithm mapping

| SPARTA routine | JAX function | Notes |
|---|---|---|
| `create_grid`, `read_surf`, `Surf::compute_line_normal`, `Cut2d` (cell type, csurfs, flow volume) | `geometry.build` | 264 / 88 / 48 cells outside / inside / overlapping and flow volume 71.8, identical to SPARTA's log. Cell volumes match SPARTA's `grid_vol` to 4e-7 |
| `FixEmitFace::create_task`, `FixEmit::mol_inflow` | `geometry.build` (`task_ntarget`) | 20 tasks on the xlo face, ntarget 7.6695 each |
| `Update::run` step order: emit, move, sort, collide, end-of-step fixes | `sim.step`, `sim.run` | All steps run in one `lax.scan` |
| `FixEmitFace::perform_task_twopass` | `sim.emit` | Pass 1 computes `int(ntarget + U)` per task. Pass 2 places y uniformly on the face, samples the normal velocity with SPARTA's accept-reject (Bird eq. 12.5), samples the tangential velocities as `vr sin/cos theta`, and sets `dtremain = dt*U`. The new particle moves for that time in this step (PINSERT). It takes a slot from the dead tail of the sorted slot list, where SPARTA appends at `nlocal` |
| `Update::move<2,1,0>`, case where no surface is reachable | `sim.dense_move` | Applies when the start cell, the end cell and the mirrored end cell all lie inside the cut-free Chebyshev ring of the start cell. Then the move reduces to SPARTA's own `optmove_bc`/`optmove_cell`: mirror in y (REFLECT), delete through x (OUTFLOW), end cell from `floor` |
| `Update::move` per-particle `while(1)` cell walk | `sim._walk_stage` on a compacted lane (`_lane_chunk`, `_later_stage`) | Computes outface and frac as in update.cpp. Tests the part of the path inside the cell against the cell's lines and keeps the earliest `OUTSIDE` hit with `param < minparam`. `exclude` holds the surface just hit and is reset on a cell crossing. Then `dtremain *= 1 - minparam*frac`, with MAXSTUCK = 20. `Domain::collide` REFLECT mirrors xnew and flips v in the same cell; OUTFLOW discards the particle |
| `Geometry::line_line_intersect` | `sim._line_line_intersect` | Same tests, including EPSSQ |
| `SurfCollideDiffuse::collide`/`diffuse` | in `sim._walk_stage`, plus `sim._diffuse` | One uniform draw per hit; if `U > acc`, `reflect3`. The diffuse branch is implemented but unused (acc = 0) |
| `Particle::sort` (per-cell `first`/`next` lists) | `sim.sort_partition` (CPU default), `sim.sort_full` (GPU) | Produces the cell-sorted slot list `plist` and the bucket starts. The partition sort is O(N): particles that stay in their cell keep their order, and only the ones that changed cell are sorted |
| `Collide::collisions_one` + `CollideVSS::{attempt_collision, test_collision, setup_collision, SCATTER_TwoBodyScattering}` | `sim.collide`, `sim._collide_attempt` | NTC with per-cell vremax (initial value `2 cxs vscale`) and the `remain` carry. Cells run in parallel: attempt k is made in every cell with more than k attempts. Argon has no internal energy, so the energy exchange is skipped (in SPARTA, `ave_dof = 0`) |
| `compute surf` + `fix ave/surf`; `compute grid` + `fix ave/grid` | `Config.tally` | Per line: n, fx, fy. Per cell: n, sum of v, sum of v^2. Window: steps 701-1000 |
| stats `np nattempt ncoll nscoll nscheck nexit nbound` | `sim.Stats` | One row per step |

At nrho = 1 the NTC attempt count rounds to 0 in every cell on every step, as in SPARTA's logs. The collision stage still runs on every step, and `test_collisions_conserve_momentum_and_energy` exercises actual collisions on a version of the deck that is 1e20 times denser.

## Departures from SPARTA (and why)

* **Random numbers.** SPARTA draws from one sequential Park-Miller/Knuth stream per process. A data-parallel code cannot reproduce that order, so only the distributions match. `rng.uniform` is SplitMix64 evaluated at (seed, step, stream, counter). It is elementwise and fuses into the kernel that uses it; with threefry, separate RNG kernels had dominated the small per-step draws. The same seed gives the same run on CPU and GPU.
* **Normal velocity of emitted particles.** SPARTA's inner loop draws `beta = 6U - 3` until `beta >= -s`. That is exactly a uniform draw on [-s, 3), so we draw it directly. SPARTA's outer accept-reject loop is kept as 64 trials in parallel per particle, taking the first one accepted. Each trial is accepted with probability 0.39, so the chance that all 64 fail is about 1e-14. Such failures are counted in `Stats.emit_fail`; there were none in any run.
* **Second collision partner j.** SPARTA draws again until j != i. We draw j uniformly from the other np - 1 particles, which is the same distribution.
* **Move.** Particles that provably cannot reach a surface skip the cell walk, like SPARTA's optimised move. SPARTA only takes that path when the deck has no surfaces; here the ring test proves no surface is reachable. All other ("near") particles walk cell by cell exactly as in `Update::move`.
  * The end cell of a straight move comes from `floor`, not from walking. The two agree except for round-off on a cell face.
  * A move that both mirrors in y and exits through x takes the walk, as in optmove, so that `nbound` counts events in SPARTA's order.
* **Particle storage.** Capacity is fixed at 53,248 slots, and a dead particle is marked with cell id DEAD instead of being compressed out. Slots never move. The sorted slot list takes the place of SPARTA's linked lists, and its tail is the free list. Running out of slots is counted in `Stats.overflow`; it never happened.
* **Not implemented, because this deck does not need them:** particle ids, erot/evib, split cells, other surface and boundary models, and emit on cut cells. `geometry.build` raises an error if any of them would be needed.

## Validation against SPARTA

The comparison uses 8 sparta_jax seeds and 8 SPARTA seeds (`spa_serial`, same deck); results are in `results/validate.json`. Agreement is given as z = (difference of the means) / (combined standard error over seeds).

| Observable | sparta_jax | SPARTA | Agreement |
|---|---|---|---|
| Particle count, every 10 steps | final 47,602 ± 58 | final 47,620 ± 51 | max \|z\| 1.16 over 100 points, rms z 0.49 |
| Particle moves | 36.929 M | 36.931 M | -0.007 %, z = -0.08 |
| Surface collisions | 116,999.6 ± 124 | 116,855.6 ± 98 | +0.12 %, z = 0.9 |
| Surface checks | 6.893 M | 6.876 M | +0.25 %, z = 1.7 |
| Boundary exits / reflections | 105,766 / 111,477 | 105,746 / 111,325 | z = 0.3 / 0.9 |
| Drag Fx, steps 701-1000 | 3.697e-20 ± 0.005e-20 N | 3.706e-20 ± 0.011e-20 N | z = -0.67 |
| Lift Fy | 1.2e-23 ± 8.9e-23 | 1.9e-23 ± 1.2e-22 | z = -0.05 |
| Per line, n / fx / fy (50 lines) | | | rms z 1.05 / 0.91 / 0.92 |
| Per cell, nrho / u / v / temp (310 cells) | | | rms z 0.90 / 1.02 / 1.01 / 1.13; \|z\| > 3 in at most 1 % of cells; mean relative difference below 0.1 % |

The cell temperature uses SPARTA's `compute grid temp` definition: m⟨v²⟩/3k, which includes the streaming velocity (see compute_grid.cpp).

**Emit unit check** (`results/check_emit.json`: 614k emitted particles over 4000 steps, compared with drifting-Maxwellian theory):

| Quantity | sparta_jax | Theory |
|---|---|---|
| Insertions per step | 153.45 ± 0.03 | 153.39 |
| Mean normal velocity | 345.97 ± 0.22 | 345.67 |
| Tangential variance / theory | 1.003 | 1 |

The Kolmogorov-Smirnov distances for normal velocity, tangential velocity, y position on the face and dtremain are 0.0009-0.0015. The 5-sigma band is 0.0056.

## Timings

Each timing is 1000 steps from an empty box, about 36.9 M particle moves (counted the way SPARTA counts them), best of 3-5 timed runs after a compile run. All rows were measured in one session on 2026-10-01 (1-minute load 0.5-1.2, nothing else running). Figure: `results/timings.png`; data: `results/timings.json`, `results/sparta_timing_final.txt`, `results/bench_*.json`.

| Code | Device | s / 1000 steps | vs SPARTA serial |
|---|---|---|---|
| SPARTA serial, deck without computes/dumps (`deck/in.circle_ar_bench`) | 1 core | **0.518** | 1 |
| SPARTA serial, full deck (computes + fix ave over the last 300 steps) | 1 core | 0.650 | 1.25 |
| SPARTA MPI x8, deck without computes | 8 cores | 0.209 | 0.40 |
| sparta_jax | CPU, 1 core (`taskset`, eigen threading off) | **1.295** | **2.5** |
| sparta_jax, with tallies (vs SPARTA's full deck) | CPU, 1 core | 1.523 | 2.3 (vs 0.650) |
| sparta_jax | CPU, all 16 cores (XLA default threading) | 1.926 | 3.7 |
| sparta_jax, `sort=full` | GPU (RTX 3050 Ti), command buffers off (WSL-safe flags) | 1.410 | 2.7 |
| sparta_jax, `sort=full` | GPU, command buffers on | **0.805** | 1.6 |
| main code (JAX-DSMC main, polygon, lane tracker, partition sort) | CPU, 1 core | 6.31 | 12.2 |
| main code | GPU, command buffers off | 4.48 | 8.6 |

### Per-stage breakdown on one CPU core

ms per step at steady state (about 46-47.6k particles). sparta_jax: `results/profile_cpu1_final.json`; main code: main's `scratch/perf/profile/forward_cpu_serial.json` (2026-10-01 evening, after the partition sort). SPARTA's own timers average over the 1000-step fill (mean 37k particles), so they are scaled by 1.26 to steady state.

| Stage | SPARTA serial | sparta_jax | ratio | main code |
|---|---|---|---|---|
| Move (straight move, surface collisions, boundaries) | 0.43 | 1.00 (walk 0.70, dense 0.21, lane select 0.09) | 2.3x | 3.35 (surface search + inside check 1.58, tracker 1.47, wall events 0.30) |
| Sort | 0.068 | 0.44 | 6.4x | 0.83 (partition sort) |
| Collide (NTC, 0 attempts) | 0.113 | 0.005 | 0.04x | 0.34 |
| Emit / inflow | 0.030 | 0.051 | 1.7x | 0.54 (Halley sampler 0.37, slots and entrant tracking 0.17) |
| Other | ~0 | 0.08 | | 2.01 (small-op overhead 1.69, cell temperature 0.30) |
| **Total** | **0.64** | **1.57** (1.30 without the profiler) | **2.5x** | **7.1** (8.0 wall under the profiler) |

### Work accounting: is the move gap XLA, or extra work?

The same counters in both codes (sparta_jax `bench_cpu_1.json` `work_second_half`, steps 500-1000; SPARTA's run totals, which average over the fill):

| Work item per step | SPARTA | sparta_jax |
|---|---|---|
| Particles moved | 37.0k (run mean; ~46k at steady state) | 45.8k (steps 500-1000) |
| Particles needing surface work (cut-free ring test) | not counted separately; cells touched 1.09 per move | 4.8k in the walk lane (10.5 %) |
| Line-segment tests (`SurfColl checks`) | 6.9k (run mean; ~8.7k steady) | 8.5k |
| Surface collisions (`SurfColl occurs`) | 117 (run mean; ~147 steady) | 138 |
| Walk iterations (cell crossings + hits) of a lane particle | per particle, as needed | lock step: 4.3 stages on average, 5 at most |

So sparta_jax does the same useful work as SPARTA: the same line tests and collisions, and the straight move for the other ~90 %. What it adds is:

* **Lock-step padding in the walk.** Every walk stage runs on the whole lane (5,120, then 768 after re-compaction). Stage occupancy is 4,808 / 1,414 / 58 / 9, so about 25 % of the lane work is masked.
* **XLA per-op cost.** The walk is 217 XLA ops per step for ~0.7 ms: about 3.2 µs per op on arrays of a few thousand rows. XLA:CPU's thunk runtime dispatches each fusion separately (launch, argument buffers, a parallel-loop decision), at ~1-2 µs even for tiny ops. SPARTA's walk is one C loop with no such per-stage cost.
* **Per element, XLA:CPU is not slower.** The dense straight move does 46k particles in 0.21 ms (4.5 ns each, one fused kernel), against SPARTA's ~9 ns per particle for its whole move.

The residual 2.3x in the move is therefore runtime overhead per XLA op (~0.5 ms of the 1.0) plus lock-step padding (~0.1 ms). Neither is an algorithmic difference. Only fewer, larger fusions could remove it, and there is no XLA:CPU mechanism to force those. The sort's 6.4x is the same story at a smaller scale: about 45 ops (cumsum, a sort of the ~4.5k cell-changers, gathers and scatters) against SPARTA's single linked-list pass.

### The sort: at XLA's floor for this algorithm

The partition sort costs 0.41-0.45 ms per step on one core (`scratch/sparta_jax/sortopt/`: a
sort-only benchmark from a steady-state step, and a per-op profile). The variants tried, each
giving the same buckets:

| Variant | µs per step |
|---|---|
| current partition sort (blocked prefix sum, two scatters) | 405-447 |
| unstable mover sort (`is_stable=False`, unique keys) | 410 |
| movers located by binary search instead of a compaction scatter | 444 |
| one scatter for stayers and movers | 522 |
| `jnp.cumsum` instead of the blocked prefix sum | 544 |
| full sort of every slot | 1594 |

Per op, the time is in four passes over all `N` slots: the new-cell gather (44 µs), the stayer
prefix count (~65 µs), the stayer scatter (~50 µs), and the elementwise passes (~57 µs). Then
come the ~5k-mover sort (~50 µs) and ~40 small ops on cell-sized arrays. SPARTA's
`Particle::sort` does the same work in one C loop over the particles (0.068 ms). A JAX
formulation needs each of those passes as its own XLA op, at ~1-2 ns per element plus a few µs
of dispatch. The 6.4x is therefore XLA's per-pass cost, not extra algorithmic work. The main
code now uses the same partition sort (`jax_dsmc.sort.update_table`, `SimStatic.sort =
"partition"`, XLA:CPU), 2.2 -> ~1.0 ms per step at its 60k slots.

### What makes the JAX version fast, and what limits it

* **One materialised line test per walk stage.** The K line tests run as one (B, K) block, reduced by a single variadic `lax.reduce` that also carries frac, outface and the clipped end point. On XLA:CPU a reduction is the root of its fusion, so later ops read stored results. `optimization_barrier` does not help, because XLA:CPU expands it away before fusion. This cut the walk from 6.3 to 1.2 ms per step.
* **Dense/lane split** (SPARTA's optimised move): ~90 % of the particles take the fused straight move, and the walk runs on a compacted lane.
* **Audit changes (2026-10-01)**, 3.3 -> 1.6 ms per profiled step, results bitwise unchanged (validation reproduces the same particle counts to the last digit):
  * cheaper routing in the dense pass;
  * a blocked two-level prefix sum (XLA:CPU's `cumsum` is a tree of reduce-windows, ~16 ns per element);
  * `materialise` for small tables, which XLA otherwise refuses to compute once;
  * a vectorisable exact `floor_int` (XLA:CPU's float->int convert does not vectorise);
  * fewer walk-stage outputs.
* **Partition sort on CPU; full sort on GPU.**
* **GPU:** limited by kernel launches and loop-predicate syncs under WSL. Command buffers remove ~45 % of the time.
* **Multi-threaded XLA:CPU is slower than one core** at this problem size.

### Speed-ups transferable to a differentiable code

| Trick | Requirement | Gain here | Status in the main code |
|---|---|---|---|
| Dense straight move + compacted near lane (cut-free ring as a distance table) | Exact routing bound; draws keyed per particle so the routing does not change randomness | Move 6.3 -> ~1 ms | Ported for XLA:CPU (`wall_adjoint._first_pass_routed`, `mesh.geometry.near_tables`). On GPU the old dense pass 0 is kept: the lane's gathers/scatters cost more launches than they save. |
| Variadic `lax.reduce` for the earliest hit (time + id + passengers) | Hit time re-formed from the winner's plane for the tangent | Removes recomputed hit tests | Ported (`mesh.geometry._element_hit`) |
| Counter-based per-particle wall draws (`fold_in(key, slot)`) | Forward, replay and adjoint use the same keyed draws | Removes the full-width per-pass draws (~1 ms) | Ported (`wall_adjoint._slot_uniforms`) |
| Blocked prefix sum, `materialise`, `floor_int` | Integer-exact | Large in sparta_jax | The blocked prefix sum is in main (`jax_dsmc.compaction.prefix_count`, a small gain); `floor_int` was bitwise but gave nothing measurable there and was not merged |
| Partition sort (carry the previous order; sort only the slots that changed cell) | A carried order in the collide state; NTC pair draws differ within a cell (same distribution) | Sort 0.9 -> 0.44 ms | Ported for XLA:CPU (`jax_dsmc.sort.update_table`, `SimStatic.sort = "partition"`): 2.2 -> ~1.0 ms per step. GPU keeps the full sort. |
| Cells-parallel NTC attempt loop (`while_loop` trip count = max attempts) | Same recorded pairs format | 1.6 -> 0.005 ms | Partly: the scratch-row copies are gone (1.6 -> 0.6 ms); the slot scan structure stays |

## Conclusions

1. **XLA vs C++, same algorithm, one CPU core: 2.5x** (1.295 s vs 0.518 s), and 2.3x with tallies against SPARTA's full deck. By stage: move 2.3x, sort 6.4x, emit 1.7x, collisions faster in JAX (all cells at once). The work accounting shows the residual is per-op XLA runtime overhead and lock-step padding, not extra algorithmic work. On GPU with command buffers sparta_jax is 1.6x SPARTA serial, and 3.9x SPARTA MPI x8.
2. **The main code is 4.9x slower than sparta_jax on one CPU core** (6.31 s vs 1.295 s; 10.5x on 2026-09-30) **and 3.2x on GPU** (4.48 s vs 1.41 s, same flags). By stage (main minus sparta_jax, ms per step): move/tracker +2.4, other +1.9 (small-op overhead: ~280 tiny reductions and ~40 carry copies per step, and the cell temperature the certified majorant needs), inflow +0.5, sort +0.4 (the same partition sort, over 60k slots), collide +0.3.
3. **Budget on one CPU core: main code 6.31 s ≈ SPARTA 0.52 s × 2.5 (XLA) × 4.9 (the main code's differentiable machinery).** The main code is 12.2x SPARTA serial (51x on 2026-09-28). Getting under 5x needs fewer, larger XLA ops in the tracker and the bookkeeping; the sort is now the same as sparta_jax's.
