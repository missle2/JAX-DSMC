# Cluster study on CURC Alpine — sphere gradient, finer-circle optimisation, seeds × optimiser

Set up 2026-09-21 to run **off the workstation**, one right-sized SLURM job per task on
the University of Colorado Research Computing Alpine cluster.  **Partition: `aa100`**
(12 nodes × 3 NVIDIA A100, 64 cores, 243 GB; `--gres=gpu:a100-40gb:N`, 21 cores per GPU
max; QOS `gpu-normal` 24 h, `gpu-testing` 1 h; default account `ucb-general`; `module
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
| `env.sh` | the job environment: the CUDA venv at `/projects/$USER/jax-env` (`GPU_KIND=a100`, default, or `h200`; `JAX_VENV` overrides the path), `JAX_PLATFORMS` pinned so a CPU fallback is an error or `module load rocm/5.3.0; module load jax` (`GPU_KIND=mi100`, jax 0.4.7 today — unusable), `JAX_ENABLE_X64=1`, no preallocation |
| `bad_nodes.txt` | nodes whose preflight failed (one per line); `submit_curc.sh` excludes them.  Seeded 2026-09-22 with `c3gpu-g5-u5` |
| `check_env.sbatch` / `check_env.py` | asserts JAX ≥ 0.4.36 (the code uses `jax.tree.map`, `jax.random.key`, `register_dataclass(data_fields, meta_fields)`), an accelerator, float64; runs a 40-step level-set circle with its shape terms; `--quick` stops after the device checks and is every job's preflight.  `gpu-normal` on one A100 (on `gpu-testing` only a MIG slice `a100_3g.20gb` is given) |
| `study.py` | the job matrix by group (`--group sphere\|mode2\|opt\|optc --list\|--names\|--gpus`, `--gpus-per-node` caps an optimisation's workers) |
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
| `opt` | `opt_S{1,2,4,8}_{gd,adam}` | finer circle drag minimisation at fixed area, 250 iterates (`--iterations 250`; 20 when this table was written), S CRN seeds per iterate evaluated concurrently (`optimize.py --workers min(S, GPUs/node)`), GD (normalised, projected) vs Adam | min(S, 3) | `mesh/results/optimize/circle_fine_S{S}_{opt}/history.json` + `coef_k.npy` | ~8 min per seed-evaluation on the workstation; per iterate ⌈S / GPUs⌉ × 8 min → S = 1, 2, 4: ~3 h; S = 8: ~6 h for 20 iterates (24 h limit; resubmit to continue) |
| `optc` | `opt_S{1,2,4,8}_{gd,adam}` (jobs `optc_opt_…`) | `opt` on the SPARTA 20 × 20 DSMC grid (`--grid-scale 1`, `fnum × 1`, same level set and step): 60 k slots, a quarter of `opt`'s memory, fits a 24 GB rtx MIG slice (`GPU_KIND=rtx`, one slice, seeds run in turn) | 1 | `mesh/results/optimize/circle_coarse_S{S}_{opt}/` → `optimize_convergence_coarse.png`, `optc_drag.png`, `optc_body_extent.png` | measured (`study_results.json`, `optc_runs`): ≈ 4.3–4.9 min per seed evaluation, median 4.6–4.9 / 9.1–9.8 / 18–19 / 35–36 min per iterate at S = 1 / 2 / 4 / 8 |

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

The local bias study (`scratch/bias/`, slides `docs/math/shape_sensitivities_slides.tex`)
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
