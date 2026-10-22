# Weak and strong scaling

Status: **reference** (2026-09; the multi-device domain-decomposition path `jax_dsmc/decomp.py`
on the periodic solver; not used by the mesh / optimisation work).  Flags: `--devices` (list),
`--mode weak|strong|both` (default both), `--dim`, `--side` (cells per side on ONE device),
`--npc`, `--steps`, `--headroom`, `--json PATH`.

```bash
python3 -m examples.scaling --devices 1 2 4 --dim 3 --side 96 --npc 50 --steps 20

# verify it runs without a multi-GPU node (correctness only, NOT speed)
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
    python3 -m examples.scaling --devices 1 2 4 --dim 2 --side 32 --steps 4
```

**strong** — fixed total problem, more devices; ideal is wall time ~ 1/P.
**weak** — fixed work *per device*; ideal is constant wall time. This is the regime a
10¹⁰-particle run actually lives in.

## Why the parallelisation is domain decomposition and not array sharding

The obvious thing — shard the particle arrays and let XLA insert the collectives — **does not
parallelise DSMC**, and this was measured rather than assumed. Inspecting the shardings through
one step:

```
x            PartitionSpec('dev')     <- sharded
table.cell   PartitionSpec('dev')     <- sharded (cell_of is element-wise)
table.order  PartitionSpec()          <- REPLICATED
table.counts PartitionSpec()          <- REPLICATED
moments.temp PartitionSpec()          <- REPLICATED
```

The global `argsort` in `sort.build_table` collapses the sharding, and everything downstream that
indexes through `order` is then replicated — every device redundantly performs the whole
collision. On four forced-host devices the collide stage took **6.8 / 13.0 / 25.2 ms** at 1/2/4
devices on a *fixed* problem: growing in proportion to the device count, the signature of
replicated work sharing one processor.

The parallel axis of the collision operator is the **cell** axis, so each device's particles must
be co-located with its cells. `jax_dsmc.decomp` gives each device an x-slab of cells; every stage
is then local, and the only communication is the particles that cross a slab boundary, exchanged
with the two neighbours by `lax.ppermute`. Because the timestep is CFL-limited a particle moves a
fraction of a cell per step, so that exchange is nearest-neighbour and small. With it, cost per
particle-step *falls* with device count (950 → 905 → 687 ns on shared cores) instead of rising.

Cells shard for free: with row-major ordering an x-slab is exactly a contiguous block of cell
indices, in the same order as the local subgrid's own `cell_of`, so `PartitionSpec('dev')` on any
per-cell array needs no permutation anywhere.

## Two capacities, two silent failures

Static shapes mean the per-device particle array and the migration buffer are fixed sizes, and
overflowing either **loses particles rather than raising**:

- `n_dev` — per-device particle capacity; overflow reported as `overflow_recv`
- `m_max` — migration buffer; overflow reported as `overflow_send`

Both are counted every step and both are errors, not warnings: a lost particle is lost mass and
nothing downstream notices. The harness flags `** PARTICLES LOST **` and `** OVERFLOW **` on any
row. Two bugs found while building this module lost or duplicated particles while reporting *zero*
overflow — padding indices clamped to slot 0, and arrivals left uncompacted in two blocks when the
free-slot run is one block. `tests/test_decomp.py` asserts exact particle conservation in D=1,2,3
precisely because neither is visible in a moment profile.

## Randomness

`base_key` is folded with the device index, so slabs draw independent streams. Without it, cell
`j` of every slab receives identical uniforms and collisions correlate across the domain. The
consequence is that a decomposed run is *statistically* equivalent to a single-device run but not
bitwise equal to it. The frozen-randomness contract still holds exactly **at a fixed
decomposition**, which is what gradient checking needs — so a finite-difference check must hold
`P` fixed.

## Benchmarking discipline

Every timing takes a warm-up call (which pays the compile) and `block_until_ready` — without the
block, JAX's async dispatch times the *enqueue*. The harness warns if another process already
holds device memory: a concurrent job on the same GPU once produced a phantom 36× cliff in this
project (spec §8.6).
