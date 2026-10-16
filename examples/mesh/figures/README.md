# Mesh example figures

One folder per topic.  Every script that writes here takes its path from
`examples/mesh/figure_paths.py` (`figure_path(folder, name)`), so new figures land in the
right folder; add a folder there (and a row here) rather than writing to the top level.

| folder | what | written by |
|---|---|---|
| `performance/` | forward and adjoint speed: per-stage breakdowns, histories, SPARTA comparison | `scratch/perf/profile_report.py` (`profile_stages`, `profile_breakdown`, `speedup_history`), `forward_breakdown.py`, `plot_cpu_breakdown.py`, `plot_forward_history.py`, `plot_forward_vs_sparta.py` |
| `circle_forward/`, `sphere_forward/` (`<case>_forward/`) | a forward case: field snapshots, fill, surface loads, SPARTA field comparison, its cut-cell mesh, free-molecular drag | `forward.py`, `report.py`, `geometry_checks.py` (mesh cells), `free_molecular.py` |
| `meshing/` | cut-cell volume error and convergence, level-set checks | `geometry_checks.py` |
| `optc/` | shape optimisation runs and their convergence, gradient seed convergence; `optab_overview` (2026-10-14: the smoothing / level-set A/B, the augmented Lagrangian's wind-up, the projection runs' apparent adjoint error vs seeds); `optab_tan_{drag,shapes,adjoint,timing}` (2026-10-15: the tangent-mode re-run -- drag at fixed area, outlines and roughness, observed vs adjoint step, iterate timings) | `optimize.py --plot`, `cluster/collect.py`, `scratch/optab/analysis.py`, `analysis_tan.py` |
| `squash/` | the aspect-ratio (squash) study: per-run optimisation panels; the `grad_s` gradient gate and the combined optimisation figure; `squashkn2_cubic` (2026-10-14: cubic vs quadratic fit, Kn 0.5 at M 0.32 / M 10); `squashkn2_histories` (2026-10-15: every Kn 0.5 run's J, s and J' per iterate, both fits, both seed counts) | `optimize_squash.py --plot`; `cluster/collect.py`; `scratch/optab/squash_analysis.py`, `squash_histories.py` |
| `movie/` | optimiser movies (`optimize_<tag>.mp4` / `.gif`), optc and squash (`--squash`) runs; their frames (`<tag>/f_*.png`) are git-ignored since 2026-10-14.  The optab and squashkn2 movies: `scratch/optab/make_movies.sh`; the tangent-mode optab movies: `make_movies_tan.sh` | `movie.py` |
| top level | the adjoint gates and estimator studies (`adjoint_gates`, `shape_circle_terms`, `levelset_gradient`, `grazing_regularisation`, `plate_estimators`, `fit_classes`, `estimator_variants`, `window_score`, `mode2_gate`; the 2026-10-12 cluster gates `bias_gate_cluster`, `cellgate_optc`; `gapgate_kn`, `gapgate_fm`), `tracker_lane` | `report.py`, `cluster/collect.py`, `tracker_bench.py`, `scratch/cellfit/cluster_analysis.py`, `scratch/gapgate/analysis.py` |
