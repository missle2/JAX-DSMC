# Cluster study on CURC Alpine — sphere gradient, finer-circle optimisation, seeds × optimiser

Set up 2026-09-21 to run **off the workstation**, one right-sized SLURM job per task on
the University of Colorado Research Computing Alpine cluster.  **Partition: `aa100`**
(12 nodes × 3 NVIDIA A100, 64 cores, 243 GB; `--gres=gpu:a100-40gb:N`, 21 cores per GPU
max; QOS `gpu-normal` 24 h, `gpu-testing` 1 h; account `ucb856_asc1` (`#SBATCH --account` in `job.sbatch` / `check_env.sbatch`); `module
load slurm/alpine` on a login node before `sbatch` — CURC docs, *Alpine hardware* and
*Running jobs*).  JAX comes from a **CUDA venv** (`pip install "jax[cuda12]"`,
self-contained CUDA 12 libraries; **verified on Alpine 2026-09-21: jax 0.6.2,
`CudaDevice`, `check_env` ok**); the `rocm/5.3.0` + `jax` modules of the MI100 partition
give **jax 0.4.7** and the code needs ≥ 0.4.36, so `GPU_KIND=mi100` stays only for a
future ROCm ≥ 6 module or an Apptainer `rocm/jax` image.

**`GPU_KIND=h200` (`ah200`, 4 H200 per node, ~3.5× the A100's fp64) is opt-in, and needs
`check_env` on that partition first.**  On 2026-09-22 all eleven ah200 jobs were
scheduled onto the single node `c3gpu-g5-u5`, whose driver refuses `cuInit`
(`CUDA_ERROR_UNKNOWN`; `slurmstepd` could not read NVML either, and `sinfo` had the
whole `g5` chassis draining).  JAX fell back to the CPU **silently** and every seed ran
to completion at ~4× the wall time (6941 s vs ~1800 s expected).  Three things now stop
that repeating: `env.sh` pins `JAX_PLATFORMS`, so a CPU fallback is an error;
`job.sbatch` runs `check_env --quick` (seconds) before the real command and exits if the
device does not initialise; and a failed preflight appends the node to `bad_nodes.txt`,
which `submit_curc.sh` passes to every later `sbatch` as `--exclude` (`EXCLUDE_NODES`
adds more).  The results of those CPU runs are valid — the backend does not change the
arithmetic — just expensive.

```
# on an Alpine login node, repository at /projects/$USER/JAX-DSMC-adjoint (250 GB, not purged)
module load python                                  # >= 3.10 (`module spider python`)
python3 -m venv /projects/$USER/jax-env && source /projects/$USER/jax-env/bin/activate
pip install --upgrade pip && pip install "jax[cuda12]" numpy scipy matplotlib   # once
module load slurm/alpine
cd /projects/$USER/JAX-DSMC-adjoint
sbatch examples/mesh/cluster/check_env.sbatch      # 20 min, 1 A100: jax version, devices, tiny run
bash examples/mesh/cluster/submit_curc.sh sphere   # 8 jobs  × 1 GPU  (one per seed)      8 h each
bash examples/mesh/cluster/submit_curc.sh mode2    # 32 jobs × 1 GPU  (one per seed)      4 h each
bash examples/mesh/cluster/submit_curc.sh opt      # 8 jobs  × 1/2/3 GPUs (S seeds concurrent) 12–24 h
python3 -m examples.mesh.cluster.collect           # merges the seed files, figures + table
```

| file | role |
|---|---|
| `env.sh` | the job environment: the CUDA venv at `/projects/$USER/jax-env` (`GPU_KIND=a100`, default, or `h200`; `JAX_VENV` overrides the path), `JAX_PLATFORMS` pinned so a CPU fallback is an error or `module load rocm/5.3.0; module load jax` (`GPU_KIND=mi100`, jax 0.4.7 today — unusable), `JAX_ENABLE_X64=1`, no preallocation, the persistent compilation cache `/projects/$USER/jax-cache` (`JAX_COMPILATION_CACHE_DIR`; a toy job 264 s → 104 s warm; clear after a jax upgrade) |
| `bad_nodes.txt` | nodes whose preflight failed (one per line); `submit_curc.sh` excludes them.  Seeded 2026-09-22 with `c3gpu-g5-u5` |
| `check_env.sbatch` / `check_env.py` | asserts JAX ≥ 0.4.36 (the code uses `jax.tree.map`, `jax.random.key`, `register_dataclass(data_fields, meta_fields)`), an accelerator, float64; runs a 40-step level-set circle with its shape terms; `--quick` stops after the device checks and is every job's preflight.  `gpu-normal` on one A100 (on `gpu-testing` only a MIG slice `a100_3g.20gb` is given) |
| `study.py` | the job matrix by group (`--group sphere\|mode2\|opt\|optc\|squash\|squashkn\|biasgate\|cellgate\|gapgate\|optckn\|optc2\|squashkn2 --list\|--names\|--gpus`, `--gpus-per-node` caps an optimisation's workers) |
| `preflight.sh` | local end-to-end check of a group before submission (one toy job per kind, as `job.sbatch` runs it) |
| `job.sbatch` | one job: resources from the `sbatch` line, the command from `CMD`; logs `nvidia-smi -L`, then the preflight, then the command |
| `submit_curc.sh` | one `sbatch` per job line with a GPU's share of the node (H200: 32 cores, 128 GB; A100: 21 cores, 70 GB), the group's wall time; `GPU_KIND=a100\|h200\|rtx\|mi100` (`rtx`: the RTX PRO 6000 partition, gres and per-GPU sizing read from `sinfo`; `submit_curc.sh check` runs `check_env` on the chosen kind first), `PART`, `GRES_TYPE` override the partition / GPU; builds the job list with the venv's python (the login node's `python3` is 3.6; `PYTHON` overrides) |
| `collect.py` | per-seed files → artifacts, `figures/optimize_convergence.png` / `optimize_convergence_coarse.png`, `figures/optc/optc_drag.png`, `figures/optc/optc_body_extent.png`, `figures/optc/gradient_seed_convergence.png`, `figures/mode2_gate.png`, the sphere gate row, `study_results.json` |

Every command **resumes from its own artifact** (seed rows / optimiser iterates already
written are skipped), so a job that hits its time limit is resubmitted with the same
`submit_curc.sh <group>` line (finished jobs exit at once).  Logs in
`examples/mesh/cluster/logs/<group>_<name>_<jobid>.out`.

## The jobs

| group | job | what | GPUs | artifact | cost estimate |
|---|---|---|---|---|---|
| `sphere` | `seed{0..7}` | 3-D level-set sphere (`h_φ = 0.1` on the 0.2 grid, `fnum × 2` ≈ 40 k particles), radius gradient vs CRN FD (h = 0.05, 0.02), one seed per job | 1 | `gradient_bench/results/ensemble_mesh_sphere_levelset0.1_gz0.05_radius_f2_…_dev_seed{i}.json` → merged `…_dev.json` | ~3× the circle's 9 min/seed on the workstation GPU → ≤ 30 min/seed, 8 h limit |
| `mode2` | `seed{0..31}` | finer circle (`fnum × 1` ≈ 48 k particles, 40 × 40 DSMC grid, `h_φ = 0.0625`), `cos 2θ` gradient vs CRN FD, one seed per job | 1 | `…circle_levelset0.0625_gz0.05_grid2_mode2_f1_…_dev_seed{i}.json` → merged | ~15 min/seed on the workstation → < 1 h, 4 h limit |
| `opt` | `opt_S{1,2,4,8}_{gd,adam}` | finer circle drag minimisation at fixed area, to convergence (cap `--iterations 300`; 20 when this table was written), S CRN seeds per iterate evaluated concurrently (`optimize.py --workers min(S, GPUs/node)`), GD (normalised, projected) vs Adam | min(S, 3) | `mesh/results/optimize/circle_fine_S{S}_{opt}/history.json` + `coef_k.npy` | ~8 min per seed-evaluation on the workstation; per iterate ⌈S / GPUs⌉ × 8 min → S = 1, 2, 4: ~3 h; S = 8: ~6 h for 20 iterates (24 h limit; resubmit to continue) |
| `optc` | `opt_S{1,2,4,8}_{gd,adam}` (jobs `optc_opt_…`) | `opt` on the SPARTA 20 × 20 DSMC grid (`--grid-scale 1`, `fnum × 1`, same level set and step): 60 k slots, a quarter of `opt`'s memory, fits a 24 GB rtx MIG slice (`GPU_KIND=rtx`, one slice, seeds run in turn) | 1 | `mesh/results/optimize/circle_coarse_S{S}_{opt}/` → `optimize_convergence_coarse.png`, `optc_drag.png`, `optc_body_extent.png` | measured (`study_results.json`, `optc_runs`): ≈ 4.3–4.9 min per seed evaluation, median 4.6–4.9 / 9.1–9.8 / 18–19 / 35–36 min per iterate at S = 1 / 2 / 4 / 8 |

### 2026-10-11: the per-cell backward, its gates and the longer runs

**The 2026-10-04 `biasgate` jobs all failed on `import jax_dsmc`** (`ModuleNotFoundError`
in every log): they run `python3 scratch/bias/<script>.py`, which puts only the script's
own directory on the import path, where `python3 -m examples…` adds the working
directory.  Fixed for every job: `env.sh` exports the repository root on `PYTHONPATH`, and
`job.sbatch` checks `import jax_dsmc` from `/tmp` before the command (an exit in seconds
instead of after queueing).  **Before submitting a group, run its local preflight**:

```
bash examples/mesh/cluster/preflight.sh <group>     # repo root; CPU, ~minutes per job kind
```

which runs one job of every kind at a toy size exactly as `job.sbatch` would (clean
environment, `PYTHONPATH` only as `env.sh` sets it, the same import check, the job's own
command; `scratch/*/cluster_job.sh` honour `SMOKE=1`, the optimisers get toy flags and a
`/tmp/preflight` tag).  It would have caught the 2026-10-04 failure; on 2026-10-11 it
caught one in its own flags.

On a GPU kind (a new node type, a new allocation) submit the same preflight as one job:

```
GPU_KIND=h200 bash examples/mesh/cluster/submit_curc.sh preflight optckn optc2 squashkn2
QOS=<qos> GPU_KIND=h200 ...                          # if the allocation has its own QOS
```

One 1-hour job on two GPUs: `job.sbatch`'s own checks (driver, `check_env --quick`, import),
then `preflight.sh` of each group with `PREFLIGHT_GPU=1` -- on the GPUs, the optimisers'
toys with a two-worker pool (one seed per GPU) and a 600-step warm-up, so the body is hit
in `circle_long` (30 steps reached it with no particle; the log now warns on `hits 0`).
Verified on the workstation GPU 2026-10-13: optc2 and the optckn Kn 0.5 toy (cubic fit,
Lagrangian area, 156 hits).

The backward's field fit is now one guarded quadratic per collision cell
(`ensemble_backward.CELL_BASIS`, docs/PSEUDOCODE.md §9.7.2): a Kn 0.5 squash seed 985 s →
95 s on the workstation (the backward 897 → 13 s; the per-query fit was ~92 % of a
collisional seed), so collisional optimisations can afford ~10× the iterates.  New groups:

| group | jobs | what | estimate (rtx MIG slice) |
|---|---|---|---|
| `cellgate` | `sq_s1_seed{0..15}`, `sq_s0.6_seed{0..7}` | Kn 0.5 squash at production density (`fnum × 1`, 4 × 8): per-query k = 32, `CELL_BASIS`, `cell:cubicv:adapt1.5`, unguarded `cell:quad`, paired, + CRN FD h = 0.2 | ~1 h / seed (k = 32 most of it), 12 h limit |
| | `optc_{c0,a50,a100,a200,s8a30,g200}_kn_b{0..3}` | six optc iterates at Kn 0.5, 4 seeds a job: two-sided k = 32, two-sided and split `CELL_BASIS` (block 0 also the one-sided k = 32), FDs along the uniform shift and the cos 2θ mode | ~1–1.5 h / seed |
| | `optc_{…}_fm_b{0,1}` | the same shapes free-molecular, 8 seeds a job | ~20 min / seed |
| `gapgate` | `sqh_t{1000,2000}_seed{0..15}` | the Kn 0.5 gap: squash `s = 1`, `fnum × 1`, the same forward captured for its last 300 and 1000 steps, `CELL_BASIS` and `cell:cubicv:adapt1.5:align0.9:both` paired, in the production run (1000 steps) and a filled box (2000); CRN FD h = 0.2 | ~10–15 min / seed |
| | `circh_t{1000,2000}_b{0..3}` | the circle (optc iterate 0) the same way, 4 seeds a job, FD along the shift and cos 2θ | ~15 min / seed |
| | `cub_{a50,a100,a200,s8a30,g200}_b{0,1}` | the cubic fit (split, as it would ship) paired with `CELL_BASIS` on the optc iterates at Kn 0.5, 4 seeds a job | ~5 min / seed |
| | `fmgz_{c0,…,g200}_b{0,1}`, `sqfm_s{0.35,1}_b{0,1}` | free-molecular grazing cut-off 0.05 / 0.01 / 0.2 / 0 paired (derivative-only), every term's projection, FD h = 0.025 / 0.05 / 0.1; optc iterates and the squash; 8 seeds a job | ~5–8 min / seed |
| | `fmsm_{a50,a100,a200,g200}_b{0,1}` | the elongated iterates smoothed (σ = 2 knots, same area): do the heavy-tailed gradients go with the sub-cell features? | ~3 min / seed |
| `optckn` | `opt_S{1,2,4,8}_adam`, `opt_S4_gd` | `optc` at Kn 0.5 with the per-cell fit, to convergence (cap 300), in the **3× longer box** `circle_long` (x ∈ [−10, 20] m, centred on the body: 15 m each side; `fnum × 2`, 120 k slots (90 k ran out at Kn 0.5 on the first iterate, 2026-10-12); warm-up 2800 steps) **and** the bodies kept 1 m from every face (`--keepout 1.0`): the FM Adam runs grew through the outflow face of the 10 m box, S8 died on `inflow_shortfall`; the area in the differentiated objective (`--area-mode tangent` since 2026-10-14: the step projected onto the constant-area set, a Lagrange multiplier per iterate; the `_al` runs' augmented Lagrangian wound up) instead of the projection; tags `circle_long_kn0.5_S{S}_{opt}_tan_sm8` (step smoothed over 8 knots since 2026-10-15, from `optab`) | ~10 min / seed at 90 k slots (now 2× the slots of `optc`: expect ~1.3× that), collisions → S = 4: ~35 iterates / day (resubmit) |
| `optc2` | `opt_S{4,8}_adam` | the free-molecular `optc` Adam runs again in the long box with the margin and `--area-mode tangent`, to convergence (cap 300); tags `circle_long_S{S}_adam_tan_sm8`, step smoothed over 8 knots since 2026-10-15 (the projection runs `circle_long_S{S}_adam` stay as they are) | ~7 min / seed |
| `optab` | `ctl`, `smooth{4,8}`, `lsh{0.125,0.25}` | A/B of the control resolution against the sub-cell wiggles: `optc2`'s S = 4 Adam run with the step smoothed over 2 (control) / 4 / 8 knots (12.5 / 25 / 50 cm), or a coarser level set (`--levelset-h 0.125` / `0.25`, smooth 2 knots: the same 25 / 50 cm widths, representation vs smoothing).  Round 1 (2026-10-13, `--area-mode lagrangian`, tags `circle_long_S4_adam_al_ab_<name>`, 300 iterates each, ~160 s / iterate on rtx): the multiplier wound up and the area swung 4–220 % of A0, so the drag comparison is void; in the first 40 iterates the control grew wiggles and every wider setting stayed convex, coarse level set = smoothing at equal width (`scratch/optab/analysis.py`, `figures/optc/optab_overview.png`).  Round 2 (2026-10-15, `--area-mode tangent`, tags `circle_long_S4_adam_tan_ab_<name>`, first 110–129 iterates pulled mid-run): area exact to 1e-12, drag 0.23–0.26 of the circle's at equal area for all but `lsh0.25` (nose kink at ~95, 0.29); the 50 cm smoothing (`smooth8`) is the smoothest and has the best adjoint check (observed / predicted step 0.88, control 0.70); chosen for the long run.  A 150 → 180–190 s step at 16:47 in three runs at once was the node, not the code (`scratch/optab/analysis_tan.py`, `figures/optc/optab_tan_*.png`) | ~40 s / seed, 1 rtx slice each |
| `squashkn2` | `kn0.5_S{4,8}`, `kn0.5_M10_S{2,4}` | `squashkn` with the per-cell fit, to convergence (`s` of the last 5 iterates within 0.5 %, cap 300; the 2026-10-01 runs managed 10 and 3 iterates at S = 4 in 24 h); tags carry the basis, so they never resume the per-query runs | Kn 0.5: ~200 s / seed; M 10: ~400 s / seed |

Formulation checks behind them (2026-10-11): with the inflow 10 m from the body centre the
empty long box's windowed drag was steady from 1400 warm-up steps (free-molecular 3.606 /
3.691 / 3.667 / 3.696 e-20, Kn 0.5 1.493 / 1.571 / 1.552 / 1.576 e-2 after 700 / 1400 / 2100 /
2800; `scratch/cellfit/warmup_check.py`); the box finally used is centred on the body (15 m
of inflow): free-molecular 3.631 / 3.682 / 3.692 e-20 after 2100 / 2800 / 3500 (2 seeds,
`warmup_check_centred.log`), so the runs warm up 2800 steps; the NTC attempt budget at production density
on the optc shapes (circle, S1-adam 100 and the protruding 200) never clips even at
`a_max 32` (`scratch/cellfit/attempts_check.py`), so 64 has ≥ 2× margin; `keep_inside` +
`project_area` alternated (`optimize.constrain`) hold the margin and the area exactly on a
shape that had crossed a face.  Suggested order (2026-10-12): `gapgate` (done 2026-10-12,
`scratch/gapgate/analysis.py`: the default basis is now `cell:cubicv:adapt1.5:align0.9:both`;
`optimize.py` records it in `history.json` and a resumed run keeps its own; `optimize_squash`
names its artifacts by basis, so `squashkn2` starts new runs), then `squashkn2` and
`optckn`, `optc2` (2026-10-14: these and `optab` now use `--area-mode tangent`, tags `*_tan`: the augmented Lagrangian of the `_al` runs wound up (exact area, the shift restores only the second-order remainder: 0.5 mm vs 1-2 cm per Adam iterate in projection mode); since 2026-10-13 every optimisation runs to convergence, capped at 300 iterates: `study.MAX_ITERATIONS`, `optimize.window_converged`; `collect` counts a converged run as done); of `biasgate` only
`ONLY='^kn_fd|hor'` (the FD arbiter and horizons; `kn_seed*` is superseded by `cellgate`).

Sizing: `submit_curc.sh` gives every job a GPU's share of its node (A100: 21 cores and
70 GB per GPU; H200: 32 cores, 128 GB).  GPU memory: the finer circle (`mode2`, `opt`: 240 k slots × 300 captured steps) peaks above 18.7 GB in the captured forward alone and ran out of memory on the 24 GB rtx MIG slices (2026-09-23), so `submit_curc.sh` runs only `sphere` and `optc` there; `mode2` and `opt` need an A100-40GB or H200 and `--time` 8 h (sphere), 4 h (mode2), 12 h
(opt, S ≤ 2), 24 h (opt, S ≥ 4).  The `opt` jobs with S ≥ 3 take a whole A100 node's GPUs; if the account's concurrent-GPU limit bites, submit `opt` first.  Estimates are
extrapolated from the workstation (RTX 3050 Ti, weak fp64); the first job's log
calibrates them — **measured 2026-09-22 on 32 CPU cores** (the failed-GPU runs):
6900 s per sphere seed, so a GPU seed is well inside the 8 h limit.

Shared settings of the finer jobs (`study.FINE`): `levelset_samples 16` (hit-search reach
`16 h_φ/2 = 0.5` covers the fastest moves on the finer level set), `grazing 0.05` (the
cut-off of the hit-time derivative and of the diffuse score, PSEUDOCODE §4.4), `--substeps
1 --partners 2` (collisionless).

## Seeds in parallel

*Seed jobs* (`sphere`, `mode2`): each seed is its own job (`--seed-start 32+i --seeds 1
--output …_seed{i}.json`, the RNG contract of `ensemble_mesh`: seed `i` = key `32 + i`, so
the merged artifact is identical to a serial run of `--seeds N`).  `collect.py` merges.

*Optimisation*: `optimize.py --workers W` evaluates the seeds of one iterate in `W`
subprocesses pinned round-robin to the job's GPUs (`ROCR_VISIBLE_DEVICES`, set by SLURM
for the `--gres` allocation), each a re-invocation of the script in worker mode
(`--eval-coef coef_k.npy --eval-seeds … --eval-out eval_k_w.npz`); the parent combines the
per-seed gradients into the same mean a serial run gives (checked on the workstation:
objective, gradient norm and step identical, the next iterate's coefficients to 3e-15).  The optimiser's step, projection and
reinitialisation stay in the parent.

## What gates what

The cluster jobs are all **specular** bodies with curvature everywhere (sphere, circle
modes, the smooth C1 optimisation iterates), on the terms validated on the workstation
(phase D circle: pathwise + timing, z = −1.4 / +0.7 for radius / `cos 2θ`).  The two
workstation analyses still open — the **diffuse circle** (the hybrid normal term and the
silhouette on a curved body; GP validated them on the plate) and the **specular capsule**
(the `∇_vλ` part of a flat-faced body, −25 %; closed 2026-09-22 by the hit-ahead
stratified fit, z = +0.08, review §7.9) — do **not** gate them: no cluster job has a
diffuse wall, and none starts from a flat face.  The capsule question would matter only if
an optimisation iterate developed a flat, cap-terminated face; the drag history and
`coef_k.npy` sequence show whether it does.

`collect.py` draws `figures/optimize_convergence.png` (J per iterate and per unit cost `S
× iterates` for every run, the area constraint), `figures/optc/gradient_seed_convergence.png`
(bootstrap SE of the `cos 2θ` gradient vs the number of seeds against 1 % of the FD), for
`optc` `optimize_convergence_coarse.png`, `optc_drag.png` and `optc_body_extent.png`, for
`mode2` `mode2_gate.png`, and prints the sphere gate row (all written to
`study_results.json`); everything from the artifacts.  `python3 -m
examples.mesh.movie --tag circle_fine_S4_gd` strings an optimisation's
`frames_<k>.npz` (the first seed's forward at every iterate: fields every 50 warm-up
steps from the empty box, then the steady state) and `coef_<k>.npy` into
`figures/optimize_<tag>.gif` (`--mp4`: `.mp4` through `ffmpeg`; `--stride s`: every s-th
iterate, for the long runs) — fill, shape step, fill, … with the drag history alongside.

## Disk

The jobs write small files to the repository under `/projects` (250 GB quota): the
gradient jobs one JSON row each (≤ 100 kB), the optimisations per iterate `coef_k.npy`
(~250 kB at `h_φ = 0.0625`) + `frames_k.npz` (16 frames × 5 floats × 1600 cells in
float32 ≈ 0.5 MB) + two small `.log`s — about 15 MB per run, ~150 MB for the whole study;
the venv is ~4 GB.  Nothing here needs `/scratch/alpine` (90-day purge); the 1.1 GB of
VTK series under `examples/mesh/results/vtk` on the workstation comes from the forward
visualisation runs (`forward.py --vtk-every`), which the cluster study does not run.

What the study answers: (1) does the 3-D shape gradient pass the same gate as the circle;
(2) how many seeds a gradient evaluation needs at the finer resolution (the SE vs S curve);
(3) whether the noisy-gradient optimisation converges per *cost* better with few seeds and
many iterates or the reverse, and whether Adam's per-coefficient normalisation helps or
hurts on a smoothed, band-restricted level-set gradient (GD's step is the largest
displacement per iterate; Adam's learning rate is set to the same displacement).

Note on the optimiser names: what `optimize.py` calls `gd` is projected steepest descent
with an H1-smoothed gradient and a step normalised to a maximum normal displacement —
there is no Hessian, it is not a Newton method.

If a ROCm route ever becomes available (`module spider jax` showing ≥ 0.4.36, or an
Apptainer image `rocm/jax`), `GPU_KIND=mi100 bash examples/mesh/cluster/submit_curc.sh
<group>` switches to `ami100` / `gpu:mi100:N` with the module lines of `env.sh`.

## `biasgate` (2026-10-04): the bias study's gates at production settings

The local bias study (`scratch/bias/`, slides `docs/math/shape_validation.tex`)
ran at fnum x2 on the 4 GB workstation GPU; these jobs repeat its open Kn 0.5 gate at
fnum x1 with the shipped 4 x 8 pull-back, and test the horizon against the whole run
captured.  Each job is `bash scratch/bias/cluster_job.sh <name>` (its header documents the
settings; JSON arguments live there because `sbatch --export` splits on commas).  Copy
`scratch/bias/` to the cluster checkout first (it is not tracked).

```
bash examples/mesh/cluster/submit_curc.sh biasgate                       # all 18 jobs, A100
GPU_KIND=h200 ONLY='hor' bash examples/mesh/cluster/submit_curc.sh biasgate          # horizon jobs
GPU_KIND=rtx  ONLY='^kn_(seed|fd)' bash examples/mesh/cluster/submit_curc.sh biasgate # the rest
```

| job | what | cost |
|---|---|---|
| `kn_seed{0..7}` | Kn 0.5 paired gate at s = 1: all grad-lambda variants, the identity with/without L* lambda, volume and collision-order terms, CRN FD (h = 0.2), frozen-volume FD | ~1 h |
| `kn_fd{0..3}` | FD only, 16 seeds each, h = 0.1 and 0.2 (the arbiter) | ~1 h |
| `kn_hor_seed{0..3}` | horizon: 300 captured steps vs the whole run (1000, no truncation) | ~4 h, > 24 GB likely: a100 / h200 |
| `fm_hor_s1.0`, `fm_hor_s0.35` | the same in free-molecular flow, 8 seeds | ~3 h |

Results: `scratch/bias/cluster/<job>.jsonl`; summarise with
`cat scratch/bias/cluster/kn_seed*.jsonl > kn.jsonl; python3 scratch/bias/summarise.py kn.jsonl`
(FD arbiter: the `fd` columns of `kn_fd*.jsonl`), and
`python3 scratch/bias/horizon_summary.py scratch/bias/cluster/fm_hor_s1.0.jsonl`.
