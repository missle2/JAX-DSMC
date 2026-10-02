# Forward speed-up work, 2026-09-30 .. 2026-10-01 (archived from the agents' working copies)

Each folder is the scratch work of one change, merged into the main tree on 2026-10-01; the
working copies (`../wt_*`) were deleted afterwards.  Files over 5 MB (digests, traces) were
not kept.  Gates for each change: `scratch/perf/noise_gate/` (tags `inflow`,
`levelset_search`, `track2`, `misc`, `merged*`) and `scratch/perf/regress/*.json`.

| Folder | Change (library files) | Kept |
|---|---|---|
| `inflow_newton/` | Halley inverse CDF for the inflow normal speed (`inflow.flux_quantile`) | prototypes, mpmath reference roots, accuracy and timing scripts, gate logs |
| `levelset_search/` | level-set hit search: block-minimum cull, batched samples, safeguarded Newton (`mesh/levelset.py`) | old-search copy (`levelset_old.py`), comparison and probe scripts, the dropped chunked variant |
| `tracker_compaction/` | compacted inside check, polygon search tiers, compacted element hits and tallies, NTC log fix (`wall_adjoint.py`, `mesh/geometry.py`, `collide/ntc.py`) | A/B and profiling scripts, per-op profiles |
| `routed_tracker/` | SPARTA-structured first pass for XLA:CPU, per-slot counter draws, variadic earliest-hit reduce (`wall_adjoint.py`, `mesh/geometry.py`, `mesh/levelset.py`, `compaction.py`) | stage/near statistics scripts, gate and test logs |
| `misc_rollins/` | scratch-row-free NTC, constant checksum weights, int32 sort keys, straight-line threefry on CPU (`collide/ntc.py`, `rng.py`, `sort.py`, ...) | sort / threefry / op benchmarks, before/after profiles |
| `compaction_with_floor_index.py` | the vectorised exact `floor_index` (tried for `grid.cell_of`; bitwise, no measurable gain; not merged) | the module as tried |
| `partition_sort/` | the carried partition sort and the bookkeeping cuts (`sort.py`, `simulate.py`, `inflow.py`, `sample.py`) | the collisional-box calibration behind `tests/test_sort.py`; the sort-variant benchmarks are on the `sparta-match` branch (`scratch/sparta_jax/sortopt/`) |
