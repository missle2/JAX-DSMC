# Mesh example figures

One folder per topic.  Every script that writes here takes its path from
`examples/mesh/figure_paths.py` (`figure_path(folder, name)`), so new figures land in the
right folder; add a folder there (and a row here) rather than writing to the top level.

| folder | what | written by |
|---|---|---|
| `performance/` | forward and adjoint speed: per-stage breakdowns, histories, SPARTA comparison | `scratch/perf/profile_report.py` (`profile_stages`, `profile_breakdown`, `speedup_history`), `forward_breakdown.py`, `plot_cpu_breakdown.py`, `plot_forward_history.py`, `plot_forward_vs_sparta.py` |
| `circle_forward/`, `sphere_forward/` (`<case>_forward/`) | a forward case: field snapshots, fill, surface loads, SPARTA field comparison, its cut-cell mesh, free-molecular drag | `forward.py`, `report.py`, `geometry_checks.py` (mesh cells), `free_molecular.py` |
| `meshing/` | cut-cell volume error and convergence, level-set checks | `geometry_checks.py` |
| `optc/` | cluster-run summaries at the top: `optimize_circle`, `optimize_convergence_coarse`, `optc_drag`, `optc_body_extent`, `gradient_seed_convergence` | `optimize.py --plot`, `cluster/collect.py` |
| `optc/optab/` | the smoothing / level-set A/B: `optab_overview` (2026-10-14: the augmented Lagrangian's wind-up, the projection runs' apparent adjoint error vs seeds) and `optab_tan_{drag,shapes,adjoint,timing}` (2026-10-15: the tangent-mode re-run) | `scratch/optab/analysis.py`, `analysis_tan.py` |
| `optc/optm/` | the free-space runs (2026-10-16): `optm_{kn0.5,M10,kn0.5_M10}` (drag, outlines, observed vs adjoint step, gradient size, length / thickness), `optm_drag`, `optm_timing`, `optm_kn0.5_M10_oscillation` (2026-10-17: the saw-tooth against the nose's motion) | `scratch/optm/analysis_optm.py`, `step_where.py` |
| `optc/optrules/` | the step-rule comparison: `optrules_{convergence,shapes,mechanisms}`, the one-seed noise runs (`optrules_noise`, `optrules_noise_shapes`) and `optrules_roughness` | `scratch/optrules/` |
| `optc/optax/` | the optax optimiser comparison (2026-10-23): `optax_{convergence,seeds,centroid}`, `optax_shapes_<regime>` | `scratch/optax/figures.py` |
| `optc/gates/` | design gates measured on optimisation runs: `guardgate`, `thingate`, `shapeprof` | `scratch/guardgate/`, `scratch/thingate/`, `scratch/shapeprof/` |
| `squash/` | the aspect-ratio (squash) study: per-run optimisation panels; the `grad_s` gradient gate and the combined optimisation figure; `squashkn2_cubic` (2026-10-14: cubic vs quadratic fit, Kn 0.5 at M 0.32 / M 10); `squashkn2_histories` (2026-10-15: every Kn 0.5 run's J, s and J' per iterate, both fits, both seed counts) | `optimize_squash.py --plot`; `cluster/collect.py`; `scratch/optab/squash_analysis.py`, `squash_histories.py` |
| `movie/<group>/` | optimiser movies `optimize_<tag>.mp4` / `.gif` (`movie.py`, group from `figure_paths.movie_folder`) with their frame directories `<tag>/f_*.png` (git-ignored): `coarse/` (circle_coarse gd / adam), `optab/` (the optab A/B and the M 10 tangent run), `squash/` (ellipse, `--squash`), `runs_kn0.5/` and `runs_kn0.5_M10/` (the per-optimiser one- and four-seed circle_long runs, incl. the optax `ox_*` arms) | `movie.py`; `scratch/optab/make_movies.sh`, `make_movies_tan.sh`, `scratch/optm/make_movies.sh`, `scratch/optax/movies.sh` |
| `movie/optax/`, `movie/optrules/` | the side-by-side comparison movies `optax_{all,present}_<regime>.mp4` and `optrules_<regime>[_noise].mp4` (+ frames) | `scratch/optax/compare_movies.py`, `scratch/optrules/compare_movie.py` |
| `gates/` | the adjoint gates and estimator studies (`adjoint_gates`, `shape_circle_terms`, `levelset_gradient`, `grazing_regularisation`, `plate_estimators`, `fit_classes`, `estimator_variants`, `window_score`, `mode2_gate`; the 2026-10-12 cluster gates `bias_gate_cluster`, `cellgate_optc`; `gapgate_kn`, `gapgate_fm`) | `report.py`, `cluster/collect.py`, `scratch/cellfit/cluster_analysis.py`, `scratch/gapgate/analysis.py` |
| top level | `tracker_lane` | `tracker_bench.py` |
