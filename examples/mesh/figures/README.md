# Mesh example figures

One folder per topic.  Every script that writes here takes its path from
`examples/mesh/figure_paths.py` (`figure_path(folder, name)`), so new figures land in the
right folder; add a folder there (and a row here) rather than writing to the top level.

| folder | what | written by |
|---|---|---|
| `performance/` | forward and adjoint speed: per-stage breakdowns, histories, SPARTA comparison | `scratch/perf/profile_report.py` (`profile_stages`, `profile_breakdown`, `speedup_history`), `forward_breakdown.py`, `plot_cpu_breakdown.py`, `plot_forward_history.py`, `plot_forward_vs_sparta.py` |
| `circle_forward/`, `sphere_forward/` (`<case>_forward/`) | a forward case: field snapshots, fill, surface loads, SPARTA field comparison, its cut-cell mesh, free-molecular drag | `forward.py`, `report.py`, `geometry_checks.py` (mesh cells), `free_molecular.py` |
| `meshing/` | cut-cell volume error and convergence, level-set checks | `geometry_checks.py` |
| `optc/` | shape optimisation runs and their convergence, gradient seed convergence | `optimize.py --plot`, `cluster/collect.py` |
| `squash/` | the aspect-ratio (squash) study: per-run optimisation panels; the `grad_s` gradient gate and the combined optimisation figure | `optimize_squash.py --plot`; `cluster/collect.py` |
| `movie/` | optimiser movies (`optimize_<tag>.mp4` / `.gif`) and their frames (`<tag>/f_*.png`), optc and squash (`--squash`) runs | `movie.py` |
| top level | the adjoint gates and estimator studies (`adjoint_gates`, `shape_circle_terms`, `levelset_gradient`, `grazing_regularisation`, `plate_estimators`, `fit_classes`, `estimator_variants`, `window_score`, `mode2_gate`; the 2026-10-12 cluster gates `bias_gate_cluster`, `cellgate_optc`), `tracker_lane` | `report.py`, `cluster/collect.py`, `tracker_bench.py`, `scratch/cellfit/cluster_analysis.py` |
