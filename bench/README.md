# Scaling study: SPARTA/Kokkos vs sparta_jax

This directory benchmarks two codes on SPARTA's three benchmark decks, `bench/in.free`, `in.collide` and `in.sphere`. All three are 3-D argon with 10 particles per cell, and the decks are run unchanged. The two codes are:

* **SPARTA**, the Kokkos build on GPUs and the MPI or Kokkos-OMP builds on CPUs;
* **sparta_jax**, which is `sparta_jax/sim3d.py` + `decomp3d.py`, a step-by-step JAX port.

The platforms are:

* NERSC Perlmutter, with A100 GPU nodes and EPYC CPU nodes, at 1–32 nodes.
* CURC Alpine, with A100, H200, L40, RTX PRO 6000 Blackwell and MI100 GPUs, **one GPU per job**: the single-GPU comparison across architectures (multi-GPU jobs do not get through Alpine's queue; `matrix.py` enforces it). Alpine has no MI250 or L40S partition (`sinfo`, 2026-11-25).

The metric is million particle-steps per second (Mpsps), as on [bench.html](https://sparta.github.io/bench.html). It is computed as particle moves divided by the timed run's loop time, and reported per device, per node, and for strong and weak scaling.

| File | Role |
|---|---|
| `decks/` | SPARTA's decks; only the seed and the timed run length are parametrised (`-v seed`, `-v steps`) |
| `env/platforms.sh` | the platform table: modules, partitions, Kokkos arch, JAX flavour, `pf_launch`, `pf_resources` |
| `env/build_jax.sh`, `env/build_sparta.sh` | environment and build scripts; each records a manifest |
| `env/runtime.sh`, `env/check_env.py` | job-side environment; check of devices, x64 and multi-process |
| `run_bench.py` | one benchmark point of either code; writes a SPARTA-format log and appends a JSON row |
| `matrix.py` | the study as data: phases → jobs → plan files, node-hour estimates, `--submit` |
| `jobs/bench.sbatch` | runs a plan file; a failed line does not stop the job |
| `prep.py` | login-node work: geometry caches, the compile gate, the device-less AOT probe |
| `pick_best.py` | tuning rows → `configs/<platform>.json`, which later phases read |
| `validate.py` | sparta_jax against SPARTA, per-move event rates over 8 seeds |
| `parse_logs.py`, `plot.py`, `report.py` | results → `results.csv` → figures → tables in the format of bench.html |

## Workflow (per platform)

```bash
# 0. sources on the shared filesystem
git clone <JAX-DSMC repo> -b sparta-match $BENCH_REPO && cd $BENCH_REPO
git clone <SPARTA fork>  $BENCH_ROOT/src/sparta          # the commit is recorded in every build
export PLATFORM=pm-gpu ACCOUNT=m1234                      # NERSC; Alpine defaults to ucb856_asc1

# 1. environments (login node)
bash bench/env/build_jax.sh
bash bench/env/build_sparta.sh                            # + VARIANT=nomallocasync, mpi, kokkos_omp, ...
bash bench/env/show_platform.sh                           # partitions / gres / modules it resolved

# 2. checks (1 node, minutes)
python3 bench/matrix.py check    --platform $PLATFORM --submit
python3 bench/matrix.py validate --platform $PLATFORM --submit   # then: bench/validate.py compare

# 3. tuning (1 GPU on the shared QOS + 1 node), then freeze the winners
python3 bench/matrix.py tune --platform $PLATFORM --submit
python3 bench/pick_best.py --platform $PLATFORM --rows $RESULTS_DIR/rows.jsonl

# 4. single device / single node sweeps
python3 bench/matrix.py single --platform $PLATFORM --submit

# 5. scaling (Perlmutter only): GATE at 1-4 nodes, re-estimate from the measured rows, then 8-32
python3 bench/prep.py geometry --plans bench/plans/$PLATFORM  # sphere caches, login node
python3 bench/matrix.py strong --platform pm-gpu --nodes 1 2 4 --submit
python3 bench/matrix.py weak   --platform pm-gpu --nodes 1 2 4 --submit
python3 bench/matrix.py strong --platform pm-gpu --nodes 8 16 32           # prints the new cost
python3 bench/prep.py compile-gate --rows $RESULTS_DIR/rows.jsonl

# 6. results
python3 bench/parse_logs.py $BENCH_ROOT/results/*
python3 bench/plot.py   $BENCH_ROOT/results/pm-gpu/results.csv --out bench/figures
python3 bench/report.py $BENCH_ROOT/results/pm-gpu/results.csv --logs $BENCH_ROOT/results --out bench/report
```

Before every `--submit`, the planner prints its jobs and estimated charged node-hours. It writes plan files without submitting unless `--submit` is given. Once a platform has rows, its estimates use the measured throughputs instead of the defaults.

## Budget (NERSC, ≤ 100 node-hours)

Estimated charged node-hours (2026-11-25, default throughputs; re-estimate after the 1–4-node gate):

| Block | GPU nodes | CPU nodes |
|---|---|---|
| tuning (1 GPU shared + 1 node) | 1.3 | 3.4 |
| single device + single node | 2.7 | 4.7 |
| strong 8M, 64M × 1–32 nodes | 15.7 | 11.4 |
| weak 1M, 16M per node × 1–32 nodes | 20.4 | 21.5 |
| **total** | **40** | **41** |

The choices that keep the study near 80 node-hours:

* **Charging.** NERSC charges elapsed time, not the job limit. Single-GPU sweeps run on the `shared` QOS, which is charged by GPU fraction.
* **Reps.** 3 reps up to 4 nodes, 2 at 8 nodes, 1 from 16 nodes.
* **Pruning.** A strong-scaling point is dropped if one rep is predicted to take more than 300 s, which removes the slow low-node-count points of the large series. The series then starts at its smallest kept node count, as bench.html's series do.
* **Optional series.** The GPU-sized 512M strong series and 128M-per-node weak series run only with `--big`.
* **JAX on CPU nodes.** Strong scaling uses the 8M series only.
* **Timeouts.** Every plan line is wrapped in `timeout` at 3× its estimate plus 5 minutes.
* **GPU step counts.** GPU free/collide time 1000 steps, as SPARTA's own `*.gpu.steps` decks do. 100 steps at 64M on 4 A100s would take only a few seconds. CPU runs keep the decks' 100 steps.

## Gates (measure before building)

1. **Validation.** `validate.py compare` must give |z| < 3 for every event rate before any timing run. Locally, at 10K particles with 8 seeds each, max |z| was 1.47 (2026-11-25); the 100K comparison runs on the cluster.
2. **Scale-out.** Run 1–4 nodes first, then re-run `matrix.py` for 8–32 nodes. Its estimate now uses the measured throughputs.
3. **JIT before the allocation.**
   * The persistent compilation cache is always on (`JAX_COMPILATION_CACHE_DIR`).
   * `prep.py compile-gate` reports compilation's share of the charged sparta_jax time. Device-less AOT is pursued only if that share is above ~3%. Lowering for CUDA on a CPU host works with jax 0.6.2, and a GPU topology description is available (`prep.py aot`).
   * Laptop GPU compile times at 1M particles are 0.6 s (free), 1.5 s (collide) and 8.7 s (sphere).
4. **Load balance.** Bricks are static and `fix balance` is not reproduced. The decomposed runs report `imbalance`, the maximum over the mean particles per rank. A rebalancer is built only if the sphere's imbalance costs more than ~10%.

## Tuning knobs (`matrix.py tune`)

**sparta_jax, GPU:**

* XLA flags: command buffers off and `graph_min_graph_size=1`, autotune level 0, while-loop unrolling, CUB radix sort off, the latency-hiding scheduler (multi-GPU).
* Allocators: platform allocator, `cuda_malloc_async`.
* Config: `sort=pack` (one int64 key) against `full` (key, value) pairs. On the laptop the packed sort was 2.9× faster at 1M.
* Config: `starts=search|scatter`; `collide_mode=prefix|all` and `collide_unroll` (bitwise-identical results); walk lane size (`lane_scale`); emit trials.
* Process layout: one process per GPU against one per node.
* Not in this sweep: jax versions and CUDA 12/13 wheels (separate envs, `JAX_VERSION`, `JAX_FLAVOR`, `VARIANT`) and NCCL/libfabric settings (`runtime.sh`).

**sparta_jax, CPU:**

* Layout: processes × host devices per process (`--xla_force_host_platform_device_count`), from 1×128 to 128×1, plus 1×1.
* XLA:CPU flags: thunk runtime off, Eigen threading off, the concurrency-optimised scheduler, parallel-codegen split count.

XLA:CPU does not emit OpenMP. Many host devices in one process are this code's equivalent of an MPI rank per core, with in-memory collectives.

**SPARTA/Kokkos, GPU:**

* Runtime: `-pk kokkos comm threaded|serial`, `gpu/aware yes|no`.
* Builds: default, `nomallocasync`, `march`. On MI100: `multikernel` and `-munsafe-fp-atomics`.
* Settings that are measured but kept out of headline runs: ranks per GPU and `global particle/reorder`. The second is a deck change, so it is reported as a labelled variant only.

**SPARTA, CPU:** MPI-only (128 ranks, or 256 with SMT), Kokkos-serial, and Kokkos-OMP layouts from 64×2 to 8×16, with `OMP_PROC_BIND`/`OMP_PLACES`.

## Things that differ between the codes (stated in the report)

* **Random numbers.** The codes use different random streams, so only distributions match. The event rates agree within seed noise (`validate.py`).
* **Load balancing.** sparta_jax uses static bricks with no `fix balance`. Migration is dimension-ordered (x, then y, then z), so a diagonal neighbour is reached in two or three hops. SPARTA sends to it directly.
* **Precision.** Both codes are float64. A float32 ablation (for L40 and RTX PRO 6000, which have weak FP64) is not implemented.
