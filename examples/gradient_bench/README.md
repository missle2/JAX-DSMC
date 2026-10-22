# gradient_bench — estimator benchmark and the production ensemble adjoint

Status: **reference** for the 1-D / 0-D gradient benchmarks (score-estimator arms, ensemble
adjoint route (b) on Couette, inlet shock and relaxation, pathwise comparison; all gated and
frozen) and **the artifact directory** of the mesh gradient gates: `examples.ensemble_mesh`
(circle, level-set circle / sphere, plate) and the cluster `sphere` / `mode2` seed jobs write
`results/ensemble_mesh_*.json` here (`examples/mesh/README.md`, `examples/mesh/cluster/README.md`).
The production gradient path is route (b), the particle-carried ensemble adjoint
(`jax_dsmc/ensemble_backward.py`); the score-estimator arms are kept as the measured
comparison that ruled the score routes out.  Run from the repository root.

Everything here is reproducible from the CLI; every number in
[`docs/ADJOINT_BRANCH_REVIEW.md`](../../docs/ADJOINT_BRANCH_REVIEW.md) §4.8–4.11 and §5.4–5.7
is printed by `report.py` from the JSON artifacts in `results/`, never typed by hand.

## Score-estimator arms (§4.8–4.11)

`python3 -m examples.gradient_bench --study <name>` runs one study through
`examples.gradient_convergence`: a 32-seed pilot (terminal-label, no baseline) whose
objective mean becomes the fixed baseline of the arms, then the arms on the disjoint
seed range 32…  Studies: `triad_speed`, `triad_temperature_left` (Couette wall controls),
`inlet_temp`, `inlet_nrho`, `inlet2_temp` (inlet shock), `scaling` (trajectory-score
noise vs M and N), `shockbl_diam`, `shockbl_omega` (the old mean baseline reconstructed),
`majorant` (thermal vs certified).  `--seeds-scale 0.75` was used for the recorded runs;
`--only <job>` runs one job, `--study all` (default) runs every study.  Each job resumes from its
own artifact in `results/`.

## Ensemble adjoint, route (b) (§5.4–5.7)

Three case modules run the same forwards with `simulate.run(capture=True)` and pull the
adjoint back on the host (`jax_dsmc/ensemble_backward.py`), then compare with the
same seeds' CRN finite difference at four steps:

| module | case | controls |
|---|---|---|
| `examples.ensemble_inlet_shock` | two-sided reservoir shock (`--particles 128 --cells 8 --steps 4` = the bench case) | `temp`, `nrho` (flux score + intensity; IC score + intensity) |
| `examples.ensemble_couette` | Couette (`256 / 4 / 10`, ε = 0.005) | `speed`, `temperature_left`, `accommodation` (wall-kernel scores, every bounce) |
| `examples.ensemble_relaxation` | 0-D anisotropy relaxation, collisions only | `diameter`, `omega` (`∂_θ log q` on accepted pairs × four-term) |

Artifacts are `results/ensemble_<case>_N<particles>_M<steps>[…].json`; the archived
folders `results/affine_basis/` and `results/linearised_step/` hold the same cases under
the two superseded backward variants (affine ghost basis; Δt-linearised collision
step) and are drawn as hollow "before" markers in the figure.

Diagnostics: `--exact-last` (exact ghost field on the last step), `--no-collision-term`.
Collision pull-backs (`ensemble_backward.configure`): `--pullback virtual --substeps S
--partners K --nodes {staged,post}` (the shipped default, S = 4, K = 8, staged;
`_virt{S}x{K}[st]` artifacts), `sequential` (realized pairs in reverse order, the exact
reference; untagged artifacts) and `linearised` (`_lin`).  The candidate
`_virt2x8st_dev` is drawn as a square marker beside its base case in
`ensemble_inlet_shock.png`; the whole scan (S, K, Δt, nodes) against the FD-only
arbiters in `results/fd_arbiter/` is `figures/pullback_scan.png` (review §5.12).  `python3 -m examples.gradient_bench.k_stability`
records the carried adjoint's rms/max over a 300-step backward per partner count K
(`results/k_stability.json`, `figures/k_stability.png`): K = 2 grows geometrically, K ≥ 8
converge — K is a stability knob on long horizons.  `--device` runs the backward with
`jax_dsmc.ensemble_device` (`_dev` suffix); `examples.ensemble_inlet_shock --objective
density-ratio --parameter mach` is the Rankine–Hugoniot acceptance case (`_rho` suffix).

## Provenance

```sh
JAX_PLATFORMS=cpu python3 -m examples.gradient_bench.provenance
```

re-evaluates seed 0 of every pilot and every ensemble artifact with the code as it
stands and compares objective (forward stream + physics, tolerance 1e-8) and gradient
(estimator, tolerance 1e-4) with the stored row — the check that the figures describe
the current implementation.  Last run 2026-09-16 (`scratch/bench/provenance.log`):
every score-route pilot reproduces to ~1e-16 across the GPU→CPU backend change; the
ensemble artifacts reproduce their objectives exactly and their gradients to ≤ 3e-3
(the host backward's ridge became relative to the normal-matrix diagonal after the
earlier artifacts were produced; `scratch/bench/probe_ridge_sensitivity.py` shows the
inlet gradient moves 3e-3 between `RIDGE_REL` 1e-10 and 1e-8 through ~0.5 % of ghost
queries whose scaled normal matrix has condition number ~1e6 — far below any study's
statistical resolution).

## Report and figures

```sh
python3 -m examples.gradient_bench.report          # tables (markdown) + figures
python3 -m examples.gradient_bench.report --plot-only
```

`figures/estimator_triad.png`, `estimator_inlet.png` (score arms vs FD),
`trajectory_variance_scaling.png` (SE vs M and N), `ensemble_inlet_shock.png`
(every validated route-(b) control vs FD, with the "before" variants, and the noise
comparison against the score routes).

`python3 -m examples.gradient_bench.pathwise` differentiates the same forwards at the same
seeds with plain fixed-key `jax.grad` (every discrete decision frozen; the mode-A estimator
of `examples/shock_adjoint`) into `results/pathwise_<control>.json`; `figures/pathwise_vs_ensemble.png`
and the report's last table join it by seed with the ensemble gradient and CRN FD, plus the
planar shape oracle.  The two adjoints agree with FD where the control has a reparameterised
route (wall speed, wall temperature, reservoir temperature — where the pathwise estimator is
also the least noisy); pathwise is identically zero for σ (VSS diameter), ω, the reservoir
density, the accommodation coefficient and the wall position, which reach `J` only through
accept bits, emitted counts, Bernoulli branches and the hit set.
