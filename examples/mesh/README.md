# `examples/mesh` — embedded surfaces: forward duplicates of SPARTA, shape gradients, shape optimisation

Status: **current** (the mesh forward and the particle-carried ensemble adjoint are the
production path; the large runs live on the cluster, `cluster/README.md`).  Sections: the forward
validation scripts, forward performance, the adjoint / shape-gradient scripts, and the
optimiser (`optimize.py`) with its production settings.

The forward scripts validate `jax_dsmc/mesh/` (cut cells, the lane, stencil and cell-walk trackers,
element models) against the shipped SPARTA `circle` (2-D) and `sphere` (3-D) examples, rerun
in **argon** with a locally built serial SPARTA (`~/sparta/sparta/build/src/spa_serial`,
inputs `sparta/in.{circle,sphere}_ar`, outputs under `results/sparta/`). 
Run everything from the repository root with `PYTHONPATH=.`:

| script | what | artifacts |
|---|---|---|
| `geometry_checks.py` | §1a: cell-type counts (264/88/48, 7256/304/440), cut-cell set, per-cell flow volumes vs SPARTA's `compute property/grid vol` dumps, candidate lists vs brute force, watertightness, sub-grid convergence | `results/geometry_checks.json`, `figures/circle_forward/mesh_cells_circle.png`, `mesh_cells_sphere_slice.png`, `mesh_volume_error.png`, `mesh_volume_convergence.png` |
| `forward.py --case circle\|sphere [--body analytic] [--tracker lane\|stencil\|walk] [--acc 1] --seed S` | the SPARTA duplicate: empty start, `emit/face xlo`, 1000 steps, window-averaged loads and fields | `results/forward_<tag>.json`, `figures/<case>_forward/<tag>_t0100/0300/1000.png`, `results/vtk/*.vtr|.vtp` |
| `tracker_bench.py [--import-logs] [--plot] [--table]` | lane vs stencil tracker on the roughest optimised shape: forward time, stuck particles, drag tally, adjoint memory per slot, the shape gradient coefficient by coefficient (from the GPU probe logs in `scratch/mesh/`) | `results/tracker_bench.json`, `figures/tracker_lane.png` (review 7.10) |
| `run_forward_batch.sh` | the runs `report.py` expects (4 seeds each + walk + analytic); run it before `report.py` | |
| `report.py` | FG1–FG4, FG7 and walk-vs-stencil tables; fill curves, surface pressure, field difference figures | `results/forward_gates.json`, `figures/{circle,sphere}_fill.png`, `_surface_pressure.png`, `_fields_sparta_t1000.png`, `_fields_difference.png` |
| `free_molecular.py --case circle\|sphere` | FG5: analytic body at three radii, drag extrapolated to zero blockage vs the Schaaf–Chambré / Bird ch. 7 reference (specular and diffuse) | `results/free_molecular_<case>.json`, `figures/free_molecular_drag_<case>.png` |
| `record_budget.py` | G9: bytes of the captured record for the shape-gradient case by `eval_shape` | `results/record_budget.json` |
| `sparta_io.py` | readers for SPARTA logs and dumps | |

Tests: `tests/test_mesh_geometry.py` (ray–element tests, analytic body, diffuse frame),
`tests/test_mesh_cut.py` (cut tables vs SPARTA), `tests/test_mesh_tracking.py` (billiards vs
exact integrators, tracker equivalence, diffuse law on a curved body, diagnostics, tallies,
box-only invariance), `tests/test_mesh_forward.py` (FG6 diffuse-body equilibrium, FG8 NTC
on cut volumes, `simulate.run(mesh=)`).

**How the cut cells look.** The cells are Cartesian boxes — a cut cell keeps its box; what
is cut is its flow volume (`volume_fraction`) and the surface elements the tracker tests
against.  Particles reflect off the exact surface (the `.vtp`), NTC and the moments use the
cut volume, but a per-cell field is one value per box (SPARTA's `dump grid` is identical).
Outputs therefore come in two forms: `<tag>_fields.vtr` / `fields_*.vtr` hold the honest
per-cell values on the full boxes; `<tag>_flow.vtu` / `flow_*.vtu` hold the **same values**
drawn only on the flow region, the body carved out at sub-cell resolution (`h/8` in 2-D,
`h/4` in 3-D) with the mesh's own point-in-flow test (`cut.flow_mask`).  The PNG snapshots
use the same mask (`figures/circle_forward/mesh_cells_circle_clipped.png` shows the flow fraction that
way).  `--vtk-every N` writes a time series (`fields.pvd`, `flow.pvd`, `surface.pvd` with
loads accumulated over the N steps before each frame) under `results/vtk/<tag>_series/`.

Facts worth knowing before reading numbers:

* SPARTA's `compute grid temp` is the un-centred `(m / 3k) <|v|^2>` (`compute_grid.cpp`),
  so the field comparison uses our `temp_kinetic`.
* SPARTA's `nscoll` in the log is the count of the logged step alone; `forward.py` records
  both the block sum (`nscoll`) and the last step (`nscoll_step`).
* The r = 3 body in the 10 × 10 box blocks 60 % of the stream: the measured drag
  (3.68e-20 N/m) is ~50 % above the unbounded free-molecular value (2.43e-20), which is why
  FG5 extrapolates in blockage.
* GPU runs under WSL need `XLA_FLAGS=--xla_gpu_enable_command_buffer=` (set by
  `forward.py`); see `docs/HANDOFF.md`.

## Forward performance (2026-10-01)

1000 steps of the circle deck from empty on this laptop: SPARTA serial 0.52 s; the main code
6.3 s (polygon) / 11.6 s (level set) on one CPU core and 4.5 / 6.8 s on the GPU; the
SPARTA-faithful JAX port (branch `sparta-match`, not differentiable) 1.30 s / 1.41 s.  History,
the changes and what is left: `docs/HANDOFF.md` section 2.5.

| script (all `scratch/perf/`) | what | artifact |
|---|---|---|
| `bench_forward.py` | 1000-step throughput, polygon and level set (`JAX_PLATFORMS=cpu`, `taskset -c N`, `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` for one core) | `bench_forward_{gpu,cpu_serial}.json` |
| `bench_sparta.sh` | SPARTA serial / MPI timings | `bench_sparta.txt` |
| `profile_cpu.sh BODY TAG` | one-core per-op profile, attributed through the HLO dump | `profile/forward_cpu_serial*.json` |
| `profile_forward.py` (+ `dump_run.py` for the HLO dump) | GPU per-scope profile | `profile/forward_scopes.json` |
| `plot_forward_history.py`, `plot_cpu_breakdown.py`, `forward_breakdown.py`, `plot_forward_vs_sparta.py` | the figures | `figures/performance/forward_speedup_history.png`, `forward_step_breakdown_cpu.png`, `forward_step_breakdown.png`, `forward_vs_sparta.png` |
| `noise_gate.py forward\|gradient` | the seed-noise gate every speed-up passed | `noise_gate/*.json` |
| `regress.py`, `profile_seed.py`, `stage_times.py`, `profile_report.py` | one optimiser seed: regression, GPU stages, untraced stage timers (since 2026-10-15), history | `figures/performance/speedup_history.png`, `profile_stages.png`, `profile_breakdown.png` |

## Adjoint and shape gradients (phase C/D)

| script | what | artifact |
|---|---|---|
| `python3 -m examples.ensemble_couette --device --objective energy-window\|shear-window --window W` | G1 / G2: value-form windowed objectives on the 1-D Couette (per-step and per-hit rewards), wall-parameter gradients vs CRN FD | `gradient_bench/results/ensemble_couette_*_energy-window*.json`, `*_shear-window*.json` |
| `python3 -m examples.mesh.ensemble_plane --objective energy-window\|shear-window` | G3: 2-D Couette whose moving wall is an embedded rectangle's face, control = its translation; `--steps/--window/--warmup`, `--timing-band` | `ensemble_mesh_plane_*.json` (`*_twosided_notiming.json` are superseded) |
| `python3 -m examples.ensemble_mesh --parameter radius\|centre_x` | phase D: the SPARTA circle at `fnum x --fnum-scale`, warm-up then captured window, windowed drag; per-term breakdown (x / v / tally / timing) and `F/r` | `ensemble_mesh_circle_<parameter>_*.json` |
| `python3 -m examples.ensemble_mesh --body levelset --parameter radius\|centre_x\|mode2 [--levelset-h 0.125]` | LS1/LS2 (phase E): the same case with the body a **level set** interpolating the circle's signed distance; the shape VJP lands on the B-spline coefficients and is chained through `coef(θ)` (`radius`: `∂coef/∂r = −1`; `mode2`: a `cos 2θ` normal displacement) | `ensemble_mesh_circle_levelset<h>_<parameter>_*.json` |
| `python3 -m examples.ensemble_mesh --case plate --body levelset --acc 1 --parameter width --silhouette-band 0.03 --levelset-h 0.0625 --levelset-samples 16 --grazing 0.05 --substeps 1 --partners 2` | GP (phase E): a diffuse capsule plate normal to the stream, control = half-width; the silhouette band term (`ShapeTerms.silhouette`) vs CRN FD; `--grazing eps` regularises the hit-time derivative and the emission-side `1/(v_out·n)` of the diffuse normal score, `--grid-scale k` refines the DSMC grid at fixed particles per cell; the cap needs `h_φ ≤ 0.0625` (the `_prefloor` artifact is the same run before the emission floor, for the CRN comparison `report.score_floor_comparison`) | `ensemble_mesh_plate_levelset0.0625_acc1_sil0.03_gz0.05_width_*.json` |
| `python3 -m examples.mesh.optimize --tag T --iterations K --seeds-per-iterate S [--optimizer ox_adam\|adam\|gd\|tr\|lbfgs\|mma\|ox_*]` (`--plot` redraws the figure) | minimum windowed drag of a level-set body at fixed area: H1-smoothed coefficient gradient, band restriction, area constraint (`--area-mode`), reinitialisation, CRN seeds across iterates; resumes from `results/optimize/<tag>/`; stops at convergence (last 5 objectives within 0.125 %) or `--iterations` (default 300).  Every flag is a `default_config()` key of `optimize.py` (plus `ensemble_mesh.default_config()`): `foo_bar` is `--foo-bar` (`--workers W` seeds of an iterate on W GPUs, `--keepout`, `--boundary`, `--restart-steps`, `--smooth`, `--step`, `--band`, `--case`, `--fnum-scale`, `--levelset-h`, `--knudsen`, `--mach`, ...); `--backward-basis` (default `ensemble_backward.CELL_BASIS`, a resumed run keeps its own), `--substeps/--partners` (collisional: 4 / 8).  The first seed of every iterate records the field movie of its forward (`--frames-every`) | `results/optimize/<tag>/history.json`, `coef_<k>.npy`, `frames_<k>.npz`, `figures/optc/optimize_<tag>.png` |
| `python3 -m examples.mesh.movie --tag <tag> [--fps 6 --morph 8 --hold 4 --every 1 --stride 1 --mp4 --squash]` | the optimisation as a GIF (or, `--mp4`, an MP4 through `ffmpeg`; `--stride s` shows every s-th iterate) from those artifacts: the flow filling in around iterate k's body (nrho, \|u\| at fixed colour limits, body carved out), the steady state held, the body's zero contour morphing to iterate k + 1, and so on; drag history alongside | `figures/movie/<group>/<tag>/f_*.png`, `figures/movie/<group>/optimize_<tag>.gif` / `.mp4` |
| `python3 -m examples.mesh.optimize --case circle_long --fnum-scale 2 --warmup 2800 --area-mode tangent --smooth 8 --keepout 1.0 --boundary free --restart-steps 500 …` (the exact command lines per study group are `python3 -m examples.mesh.cluster.study --group <g> --list`) | the production settings (2026-10-15 .. 10-20): the gradient and step projected onto the area's tangent space (exact area; `projection` / `lagrangian` / `penalty` are older modes, the Lagrangian wound up), the step smoothed over 8 knots (`optab` A/B), the body kept 1 m from every face, `--boundary free` (every face free-stream; default `channel`, SPARTA's deck; gates in `scratch/freebox/`), and the **warm start** `--restart-steps N` (a new iterate re-equilibrates the previous iterate's final gas on the new body for N steps instead of the cold warm-up; validated 2026-10-19, 500 recommended).  Optimisers: `ox_adam` (the default since 2026-10-21: the shipped Adam step as an optax chain, identical to `adam`, its numpy twin), `gd` (projected steepest descent with H1 smoothing, not Newton), the step rules `tr` (trust region, reworked 2026-10-20), `lbfgs`, `mma` of `examples/mesh/optimizers.py`, and the optax arms of `examples/mesh/optax_rules.py` (`ox_ademamix`, `ox_adamcos` cosine step schedule, `ox_sfadam` schedule-free, `ox_prodigy` learning-rate free; need `optax`); all but `gd`/`adam` need `--area-mode tangent` (the default).  The 2026-10-20/21 comparisons found Adam best and robust to one seed per iterate (`cluster/README.md`: `optrules`, `optrules1seed`; `optax1seed` running).  Analyses: `scratch/optab/analysis_tan.py`, `scratch/optm/analysis_optm.py`, `scratch/optrules/analysis.py`, `noise.py`, `roughness.py`; optax checks `scratch/optax/` | `figures/optc/optab/optab_tan_*.png`, `optm_*.png`, `optrules_*.png`, `figures/movie/<group>/optimize_<tag>.mp4` |
| `examples/mesh/cluster/submit_curc.sh <group>` / `collect` | the cluster study on CURC Alpine (gradient gates, optimisations, optimiser comparisons): one right-sized SLURM job per seed / per optimisation.  Instructions, group table and results: `cluster/README.md`.  Run data is pulled with `cluster/pull_results.sh` (not in git since 2026-10-21) | `gradient_bench/results/…_dev_seed{i}.json`, `results/optimize/<tag>/`, `figures/optc/*`, `figures/gates/mode2_gate.png` |
| `python3 examples/mesh/report.py` | the "Adjoint gates" table (incl. LS2, GP), `figures/gates/adjoint_gates.png`, `figures/gates/shape_circle_terms.png`, the seed-by-seed level-set vs analytic comparison `figures/gates/levelset_gradient.png`, the grazing sweep `figures/gates/grazing_regularisation.png`, and the GP plate's three estimators of the diffuse normal term seed by seed with the FD vs its step `figures/gates/plate_estimators.png` (`results/plate_fd_steps.json` from `scratch/mesh/probe_plate_fd_steps.py`) | `results/forward_gates.json` (`adjoint_gates`, `levelset_vs_analytic`, `grazing_regularisation`, `score_floor_comparison`, `plate_fd_steps`) |

Both gradient runners evaluate the four estimators of `∇_xλ` on every seed
(`rows[*].variants`: one-sided fit, pooled one-sided fit, one-sided + along-flight
identity, pooled + along-flight; `jax_dsmc.shape.VARIANTS`) so their bias and variance
can be compared on the same seeds; `scratch/mesh/probe_plane_fitbias.py` compares them
against replica-exact value gradients on the plane, `probe_plane_truncated_fd.py` gives
the truncated-horizon FD the plane rows are gated against, `probe_fit_classes.py` the
fit error by particle class.  The report prints all of it and draws
`figures/gates/estimator_variants.png`, `figures/gates/fit_classes.png`.

All of these need the GPU for anything but a smoke test (`examples.ensemble_mesh` at
SPARTA scale is ~2 min/seed there and does not finish a seed on the CPU in half an
hour).  The estimator's parts, what each gate found and what is still open are in
`docs/ADJOINT_BRANCH_REVIEW.md` §7.6 and `docs/PSEUDOCODE.md` §9.7.

**Open issue (2026-09-20):** none of the four `∇_xλ` estimators resolves the x/v-part of
a translated flat wall (exact +0.003 on the shear probe; pooled / along-flight read 0).
Shape controls are therefore restricted to **normal displacements of curved surfaces**
(radius-type, where the exact tally term carries the gradient); tangential translations
of flat walls are not supported as optimisation controls.  First optimisation case:
circle → airfoil-like shape (sphere analogue in 3-D) by normal displacements only.

**Representation (2026-09-20): level set.** Optimisation deforms a C1 level set
(`jax_dsmc/mesh/levelset.py`: cubic B-spline coefficients on a finer grid; `--body
levelset` in `ensemble_mesh.py`), not the 50-gon.  On a C1 surface the hit normal varies
smoothly with the shape, so the pathwise VJP is complete for specular walls and the
polygon's vertex jump term is not needed; the diffuse silhouette term (physics) is the
only band term that remains.  Gates LS0 (`tests/test_mesh_levelset.py`,
`geometry_checks.py`), LS1 (forward vs the analytic circle), LS2 (`dJ/dcoef` contracted
with `∂φ/∂r = −1` vs the analytic radius gradient; z = −1.4 at 4 seeds, per-seed
agreement with the analytic run to 1e-4 except one grazing hit) and the `cos 2θ` mode
(z = +0.7 / −0.2) pass; the first optimisation (`optimize.py`, 8 iterates) lowered the
drag by 7.4 % at fixed area (`figures/optc/optimize_circle.png`).  The level-set body is a
third `Mesh.kind` (`"levelset"`, single surface id like the analytic body): hits by a
sampled bracket + bisection + one Newton step with the bracket under `stop_gradient`
(the bisection alone has no derivative), normals `∇φ/|∇φ|`, no `exclude` (re-hits are
ruled by the sign pattern of φ), host cut tables by the coefficient-sign test; the
search reach `n_sample · h_φ / 2` is the level set's `stencil_sufficient` bound.

The specular capsule (`--case plate --acc 0`) closed the last open estimator question
(review §7.9): its −27 % was the `∇_xλ`/`∇_vλ` parts, which are zero for a convex
collisionless body and were read from a fit pool mixing about-to-hit and reflected particles
at the cap.  The hit-ahead stratified fit (`shape.hit_ahead`, `--value-class hit_ahead`,
artifacts tagged `_vchit`) gave 1.356 ± 0.011 vs CRN FD 1.35 e-20 (z = +0.08) on 2026-09-22,
but **it is biased by construction** (2026-10-02: it drops `∇P·Δλ`, +40 % on the squash mode);
the runner default is `--value-class none` and `hit_ahead` is diagnostic only
(`docs/HANDOFF.md`, shape bias study).  The diffuse circle (`--case circle --acc 1 --parameter
radius --silhouette-band 0.05 --grazing 0.05`, gate GD, 4 seeds, 2026-09-22): 1.312 ± 0.052 vs CRN
FD 1.442 ± 0.03 (h = 0.1), 1.128 ± 0.16 (0.05) and `F/r` 1.381 e-20, 9 % below the precise FD
(z = −2.3, the plate's pattern); the silhouette term is 61 % of it.  Not closed (the cluster
jobs are all specular).

The per-cell backward fit (default `CELL_BASIS = cell:cubicv:adapt1.5:guard32:align0.9:both`,
2026-10-19; PSEUDOCODE §9.7.2) replaced the per-query fit used by the 2026-09 gates quoted
above, whose artifacts are kept as they are.
