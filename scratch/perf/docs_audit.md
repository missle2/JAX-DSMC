# Docs vs code audit (2026-09-28)

Static reading only (no jax imported, no GPU). Edits are to docs, docstrings and comments
only; no code behaviour, artifact or figure was changed. `ruff check` / `ruff format --check`
pass on every touched `.py`.

**Concurrent edits:** someone else was editing `examples/ensemble_mesh.py` and
`examples/mesh/forward.py` during the audit (new `ellipse` / `spheroid` cases, control
`squash_y`). I did not touch those two files.

**About the premise "lane tracker is the default":** this holds for the examples only.
`examples/mesh/forward.py --tracker` defaults to `lane` and `K_BOUNCE = 32`, and
`examples/ensemble_mesh.default_config` sets `"tracker": "lane"`. The library default is
still `SimStatic.tracker = "stencil"` (`jax_dsmc/state.py:473`). The docs describe it
correctly as "the default of the mesh examples".

Legend: **F** = fixed, **D** = left for a human decision, **L** = left on purpose (historical
record or cosmetic).

## docs/math/ensemble_adjoint.tex

| # | doc location | code | mismatch | status |
|---|---|---|---|---|
| 1 | §3 "The field interpolant", last paragraph | `ensemble_device.py:92` `field_query` | Said `field_query` returns `(Λ, ∇_vΛ)` and that the x-slopes are not returned. It also said the docstring still reads "ridge 1e-12" and that the slopes are only for planned terms. The code returns `(value, grad_x, grad_v)`, the docstring says 1e-8 (relative), and `shape.py` consumes both slopes. | F |
| 2 | §8, replay check (the `shape.replay` sentence) | `shape.py` `_mismatch`; `examples/ensemble_mesh.py` / `examples/mesh/ensemble_plane.py:57` `REPLAY_TOL` | Implied that `shape.replay` itself checks and raises. It only reports `replay_error` (1 on a discrete flip); the runners hold it to `REPLAY_TOL = 1e-9`. | F |
| 3 | §8 optimiser, "Design variables" | `examples/mesh/optimize.py:154` | Said the optimiser's gradient includes the silhouette part. `seed_terms` uses `terms.pathwise + terms.score` only; the silhouette term is 0 here (specular, band 0) and is never added. See also C1. | F |
| 4 | §8 status, "Optimisation study" | `examples/mesh/cluster/study.py:93`; every `circle_coarse_*/history.json` has `iterations = 250` | Said "80 iterates each". The value is 250 (it was raised from 80 in 9e925b7). | F |
| 5 | App. A, "Placement" | `inflow.py:431-436` (inside_body/exhausted → shortfall) | Said "Faces touched by the body are not allowed to emit". No such rule exists. An entrant that starts inside the body is killed by the inside test and counted in `inflow_shortfall`, an invalid step. This is exactly how the optc S2/S8 Adam runs failed. | F |
| 6 | App. A, move, lane tracker bullet list | PSEUDOCODE §9.6.1 | Cited "section 4.5 of PSEUDOCODE.md" for the lane tracker. §4.5 is split cells; the lane tracker is §9.6.1. | F |
| 7 | §2 value-form rewards (`hit_reward`) | `ensemble_device.py:183-191`; `examples/ensemble_inlet_shock.py:188` | Gave only the dense `f64[M,K,N]` shape. A lane replay passes the `HitRewards.totals()[:, None]` per-step totals (`f64[M,1,N]`). | F (parenthetical added) |
| 8 | App. A, state description | `state.py` SimStatic `D in (1,2,3)` | Said "D = 2 or 3". The code supports D = 1, 2 or 3; a body needs 2 or 3. | F |
| 9 | §8 status, "Shape gradients validated": `cos 2θ` mode z = +0.7 / −0.2 | `examples/mesh/cluster/study_results.json` → `mode2` (32 seeds, fine circle, `mode2_gate.png`) | The status gives only the 4-seed workstation gate. The 32-seed fine-circle cluster gate that `collect.py` now summarises has adjoint −2.386e-20 ± 5.1e-23, and adjoint − FD = −7.9e-21 ± 6.6e-23, **z ≈ −121** (a ~33 % discrepancy). No document mentions this result. | **D**: a new result that needs interpretation |
| 10 | §8 status, sphere z = −1.75 / −0.91 | `study_results.json` → `sphere` (8 seeds: 3.389 ± 0.048 vs FD 3.487 ± 0.031, z = −1.71); the sphere artifact is modified and uncommitted | The docs quote the 7-seed CPU run. The merged artifact now has 8 seeds. | **D** |
| 11 | §8 status, "resubmitted runs are in progress" and "failures … each now fixed" | `study_results.json` → `optc_runs` | S1 Adam finished 250/250. The other runs stopped at time limit (S2/S4 gd, S4 adam), host OOM (S8 gd), cancelled (S1 gd), or **`Invalid DSMC trajectory: inflow_shortfall`** (S2 adam at 96 iterates, S8 adam at 37). That last one is a new failure class: the Adam bodies reach the inflow face (`optc_extent` x_min ≈ −0.02). It is not in the "each now fixed" list. | **D** |
| 12 | §8 status, cost "≈ 36–43 min per iterate at S = 8" | `study_results.json` `seconds_per_iterate_median` 34.7 / 36.0 min, p90 40.0 / 40.4 | Slightly high versus the medians. | L (approximate; can be updated with #9–11) |

Everything else in the new parts was checked against the code and matches:

- the NTC count and the thermal majorant `2·C·(25·2kT/m_r)^(1−ω)` (`ntc.py:67-84`);
- the slot algorithm (pair `j=(i+1+r) mod N_c`, violation count, ratchet, `attempts_clipped`);
- the VSS σ_T and `q`, the angular law and the scatter formulas (`vss.py`);
- F(s) and the flux-weighted normal density; SPARTA's β rejection sampler including its acceptance formula; the 64-step bisection inverse CDF (`inflow.py`);
- the level-set hit search, the Newton step and the `+→−` rule (`levelset.py`);
- the lane tracker: B = max(256, N/32), passes 1..31, `surface_stuck`, lane overflow as an error (`wall_adjoint.py`, `state.py:581`);
- cut and split cells (`cut.py`, `grid.collision_cell`); the moments; the drag tally;
- the optimiser steps: band `b + h_φ`, the Gaussian of σ = 2 knots, Adam with bias correction and re-smoothing, `η = s_max/max|d|`, reinitialise before the shift, 40-step bisection on [−4h, 4h] of the smoothed-Heaviside `body_volume` (`optimize.py`, `levelset.py:463-518`);
- the status numbers 2.3e-15, 64 kB (was 116), 2×, and ~300 MB (review §7.10).

No section was renumbered and no numbered equation was reordered.

## Docstrings in jax_dsmc/

| # | doc location | code | mismatch | status |
|---|---|---|---|---|
| 13 | `shape.py` module docstring, last paragraph | `shape.py:529` `silhouette_cotangent` | Said the diffuse silhouette term "is not part of this module". It is: `gradient(silhouette_band=)`. | F |
| 14 | `shape.py` `ShapeTerms.replay_error` | `shape.py` `_mismatch` | Said "`< 1e-12` is roundoff; anything larger is a broken replay". A flip reports 1; round-off reaches ~1e-12; the runners allow 1e-9. | F |
| 15 | `state.py` `Diagnostics.surface_stuck` | `wall_adjoint.py` lane branch of `_advance_impl`; `simulate.py:189` | Said "particles the stencil tracker deleted". The lane tracker deletes and counts them too. | F |
| 16 | `wall_adjoint.py` `TrackerCounts.exhausted`; `_advance_impl` "Both end with the inside-body check" | same | Stencil only / "both". The lane tracker also deletes stuck particles, and all three trackers run the inside check. | F |
| 17 | `simulate.py` `run` docstring: "`static.tracker` selects the stencil or the cell-walk tracker" | `state.py:473-492` | The lane tracker was missing. | F |
| 18 | `simulate.py` `EventRecord.walls`: "`WallEvents` -- the step's diffuse-face hits" | `simulate.py` `_event_record` / `compact_record` | It also records every element hit, and for the lane tracker it is a `RecordEvents`. | F |
| 19 | `simulate.py` module docstring: "No `while_loop` anywhere" | `wall_adjoint.py:615` (lane forward uses `lax.while_loop`) | The lane forward uses a `while_loop` (the traced replay uses `fori_loop`). The jaxpr audit (`tests/test_ntc.py:342`, `tests/test_perf.py:111`) covers the collision step only. | F (documented exception) |
| 20 | `mesh/cut.py` module docstring, "split cells: refused … raises SplitCellError -- refine the grid" | `cut.py:965-1030` `build_mesh` → `split_regions` / `split_tables` | Split cells now get sub-cells. | F |
| 21 | `grid.py` `cell_volume`: "set by `mesh.cut.build_mesh`" | `cut.py:922` `with_volume` | `build_mesh` computes the volumes and `with_volume` sets them on the grid. | F |
| 22 | `ensemble_device.py` `backward` docstring, `hit_reward f64[M,K,N]` | `_move_pullback` sums axis 0; lane callers pass totals | The lane form was undocumented. | F |
| 23 | `ensemble_backward.py` module docstring: "diffuse walls -- not exercised here --" | `ensemble_backward.py` block 2b builds `Hits` per recorded hit | Stale. | F |
| 24 | `ensemble_backward.py` `from_device`: `hit_reward f64[M,K,N]` only | `_step_rewards` accepts `shape.HitRewards` | The lane form was not mentioned. | F |
| 25 | `shape.py` module docstring, the bullet ending "`-0.43 +- 0.02``). -- the kernel score of the re-emission law's…`" | — | A garbled leftover fragment, cosmetic. | L |
| 26 | `shape.replay` docstring: `reward(...) -> f64[K, N]` | `_per_part` calls it on head / tail | With the lane tracker it is called per part (1 and K−1 rows). | L (minor) |

Checked and consistent: `state.lane_capacity`, `SimStatic.lane_size` / `tracker`, `GridSpec.n_sub` / `split_of` / `split_table` / `sub_parent`, `grid.collision_cell`, `wall_adjoint.RecordEvents` / `SplitEvents` / `compact_record`, `shape.HitRewards` / `Replay` / `hit_ahead` / `gradient(value_class="hit_ahead")`, `mesh/levelset.py` (hit search, `reinitialise`, `body_volume`), `cut.outside_corner` (no flow corner needed for a level set), `inflow.py`, `collide/ntc.py`, `collide/vss.py`.

## Other docs

| # | doc location | code | mismatch | status |
|---|---|---|---|---|
| 27 | PSEUDOCODE §4 "Embedded surfaces" paragraph: "split cells refused" | `cut.build_mesh` | Superseded by §4.5. | F (dated pointer) |
| 28 | PSEUDOCODE §4.4 `reinitialise(ls, n_iter)`: "projection from every knot" | `levelset.py:463` | Only knots within the band are projected, with `ceil(band/h)+4` steps. The others take the distance to the projected cloud, and unconverged knots are left out. | F |
| 29 | PSEUDOCODE §9.6.1 `shape.HitRewards(values, slot, total)`; "device backward reads `total`" | `shape.py:82` | The class is `HitRewards(values, slot)` with `totals()`. The same section later says so itself. | F |
| 30 | PSEUDOCODE §9.7 `replay_error … < 1e-12` | `shape._mismatch`, `REPLAY_TOL` | Superseded 2026-09-24. | F (dated note) |
| 31 | PSEUDOCODE §12 "No `while_loop` anywhere … hard invariant" | `wall_adjoint.py:615` | Same as #19. | F (dated exception) |
| 32 | PSEUDOCODE §0.1 `C = prod(n)` | `grid.num_cells` = prod(n) + n_sub | Split cells missing. | F |
| 33 | PSEUDOCODE §3.1 `SimConfig` / `GridSpec` skeleton | `state.py` | Lacks `transverse`, `volume`, `n_sub…`, `tracker`, `K_walk`, `lane_size`. | L (Phase-0 skeleton; the fields are documented in §4.5 / §9.6.1) |
| 34 | HANDOFF header, embedded-surfaces bullet: "split cells are refused", "two trackers" | `cut.build_mesh`; `state.py:473` | Stale. | F (dated notes; lane is the examples' default, the library's stays stencil) |
| 35 | HANDOFF header, mesh-adjoint bullet: `replay_error < 1e-12` | as #30 | Stale. | F (dated note) |
| 36 | PLAN Phase 5, status 2026-09-16: "Split cells are detected and refused" | `cut.build_mesh` | Historical. | F (dated pointer) |
| 37 | PLAN representation decision: optimiser with "CRN line search, periodic reinitialisation" | `optimize.py` `run` | The optimiser as built has no line search, reinitialises every iterate, and holds the area by a uniform φ shift. There was no status for 2026-09-23/24. | F (one dated note after the 2026-09-22 status, pointing to review §7.10 and the tex) |
| 38 | PLAN status 2026-09-21: seeds "on min(S, 4) GPUs" | `study.GPUS_PER_NODE = 3` | Historical (planned for h200). | L |
| 39 | FORWARD_JIT_IMPLEMENTATION: "Tallies are … not yet a built-in … accumulator in `simulate.run`"; "tracker … not a general moving-mesh adjoint" | `simulate.run(emit_surface=)`, `shape.py` | Superseded by later work. | L (historical implementation record) |
| 40 | examples/mesh/README intro: "the stencil and cell-walk trackers" | `wall_adjoint` lane | The lane tracker was missing. | F |
| 41 | examples/mesh/README movie row: GIF only, no `--stride` / `--mp4` | `movie.py:178-196` | Stale. | F |
| 42 | examples/mesh/README cluster row: `sphere\|mode2\|opt`; artifacts list only the fine runs and 2 figures | `study.py` groups, `collect.py:657-713` | `optc` and its figures, and `mode2_gate.png`, were missing. | F |
| 43 | cluster/README `study.py` row: `--group sphere\|mode2\|opt` | `study.py` choices include `optc` | `optc` missing. | F |
| 44 | cluster/README `collect.py` row and paragraph: 2 figures | `collect.py:662-713` | Now also `optimize_convergence_coarse.png`, `optc_drag.png`, `optc_body_extent.png`, `mode2_gate.png`, and `study_results.json`. | F |
| 45 | cluster/README `opt` row: "20 iterates" (and its cost) | `study.py:93` `--iterations 250` | Stale. | F (the cost is marked as being for 20 iterates) |
| 46 | cluster/README `optc` row: "not yet timed" | `study_results.json` `optc_runs` | Now measured: ≈ 4.3–4.9 min per seed evaluation; medians 4.6–4.9 / 9.1–9.8 / 18–19 / 35–36 min per iterate at S = 1/2/4/8. | F |
| 47 | cluster/README sizing: "`submit_curc.sh` runs only `sphere` there" (rtx) | `submit_curc.sh:85` | It runs `sphere` and `optc`. | F |
| 48 | cluster/README: specular capsule listed as still open (−25 %) | review §7.9 | Closed 2026-09-22 (z = +0.08). | F (dated note) |
| 49 | cluster/README movie paragraph: GIF only | `movie.py` | Missing `--mp4` / `--stride`. | F |
| 50 | `examples/mesh/cluster/study.py` docstring, `opt`: "20 iterates" | `study.py:93` | Stale. | F |
| 51 | `examples/mesh/cluster/collect.py` docstring: 2 figures only | `collect.py:657-713` | Missing the new figures, the `contact.json` cache (with its lazy jax import) and `study_results.json`. | F |
| 52 | `examples/mesh/cluster/submit_curc.sh` header: "opt # 8 jobs x {1,2,4} GPUs"; wall time without optc | `submit_curc.sh` a100 `PER_NODE=3`; `opt\|optc` 24 h | a100 gives {1,2,3}; optc is always 24 h. | F |
| 53 | `examples/mesh/optimize.py` docstring: `--plot` → `optimize_circle.png`; movie "into a GIF" | `optimize.py` `main` (`optimize_{tag}.png`); `movie.py --mp4` | Stale. | F |
| 54 | `examples/mesh/optimize.py` comment in `run`: "split cells … raise here" | `cut.build_mesh` | Split cells no longer raise. | F |
| 55 | `examples/ensemble_mesh.py` docstring, controls "radius, centre_x, mode2" | `PARAMETERS` includes `width` (and now `squash_y`) | Incomplete. | **D**: the file was being edited concurrently |

Checked and consistent: docs/math/README.md; examples/gradient_bench/README.md; ADJOINT_BRANCH_REVIEW §7.9–7.10 (lane size, record capacity, `REPLAY_TOL`, the test names, the memory numbers); ADJOINT_CORRECTIONS (the "1e-12 slope ridge" refers to the P3 reference `ensemble_inflow*.py`, which does use 1e-12).

## Code observations (not docs, not changed)

- **C1** `examples/mesh/optimize.py:154` sums `terms.pathwise + terms.score` and drops `terms.silhouette`. For a diffuse body run with `--silhouette-band > 0` the optimiser would silently leave out the silhouette term. The gradient runner (`ensemble_mesh.ensemble_gradient`) does add it. This is harmless for the specular optc study.
- **C2** `examples/ensemble_couette.py:52` still uses `REPLAY_TOL = 1e-12`. This is consistent with its box-only walls (no concave amplification); noted only for completeness.
