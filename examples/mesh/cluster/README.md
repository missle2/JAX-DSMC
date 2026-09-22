# Cluster study on CURC Alpine — sphere gradient, finer-circle optimisation, seeds × optimiser

Set up 2026-09-21 to run **off the workstation**, one right-sized SLURM job per task on
the University of Colorado Research Computing Alpine cluster.  **Partition: `aa100`**
(12 nodes × 3 NVIDIA A100, 64 cores, 243 GB; `--gres=gpu:a100-40gb:N` required, 21
cores per GPU max; QOS `gpu-normal` 24 h, `gpu-testing` 1 h; default account
`ucb-general`; `module load slurm/alpine` on a login node before `sbatch` — CURC docs,
*Alpine hardware* and *Running jobs*).  JAX comes from a **CUDA venv**
(`pip install "jax[cuda12]"`, self-contained CUDA libraries): the `rocm/5.3.0` +
`jax` modules of the MI100 partition give **jax 0.4.7**, and the code needs ≥ 0.4.36
(`jax.tree.map`, `jax.random.key`, `register_dataclass(data_fields, meta_fields)`);
`GPU_KIND=mi100` in `env.sh` keeps that route for a future ROCm ≥ 6 jax module or an
Apptainer `rocm/jax` image — it is not usable today.

```
# on an Alpine login node, repository at /projects/$USER/JAX-DSMC-adjoint (250 GB, not purged)
module load python                                  # >= 3.10 (`module spider python`)
python3 -m venv $HOME/venvs/jax-cuda && source $HOME/venvs/jax-cuda/bin/activate
pip install --upgrade pip && pip install "jax[cuda12]" numpy scipy matplotlib   # once
module load slurm/alpine
cd /projects/$USER/JAX-DSMC-adjoint
sbatch examples/mesh/cluster/check_env.sbatch      # 20 min on gpu-testing: jax version, devices, tiny run
bash examples/mesh/cluster/submit_curc.sh sphere   # 8 jobs  × 1 GPU  (one per seed)      8 h each
bash examples/mesh/cluster/submit_curc.sh mode2    # 32 jobs × 1 GPU  (one per seed)      4 h each
bash examples/mesh/cluster/submit_curc.sh opt      # 8 jobs  × 1/2/3 GPUs (S seeds concurrent) 12–24 h
python3 -m examples.mesh.cluster.collect           # merges the seed files, figures + table
```

| file | role |
|---|---|
| `env.sh` | the job environment: the CUDA venv (`GPU_KIND=a100`, default; `JAX_VENV` overrides the path) or `module load rocm/5.3.0; module load jax` (`GPU_KIND=mi100`, jax 0.4.7 today — unusable), `JAX_ENABLE_X64=1`, no preallocation |
| `check_env.sbatch` / `check_env.py` | asserts JAX ≥ 0.4.36 (the code uses `jax.tree.map`, `jax.random.key`, `register_dataclass(data_fields, meta_fields)`), an accelerator, float64; runs a 40-step level-set circle with its shape terms |
| `study.py` | the job matrix by group (`--group sphere\|mode2\|opt --list\|--names\|--gpus`) |
| `job.sbatch` | one job: resources from the `sbatch` line, the command from `CMD` |
| `submit_curc.sh` | one `sbatch` per job line with 21 cores and 70 GB per GPU, the group's wall time; `GRES_TYPE=a100-80gb`, or `PART=ah200 GRES_TYPE=h200`, override the GPU |
| `collect.py` | per-seed files → artifacts, `figures/optimize_convergence.png`, `figures/gradient_seed_convergence.png`, the sphere gate row |

Every command **resumes from its own artifact** (seed rows / optimiser iterates already
written are skipped), so a job that hits its time limit is resubmitted with the same
`submit_curc.sh <group>` line (finished jobs exit at once).  Logs in
`examples/mesh/cluster/logs/<group>_<name>_<jobid>.out`.

## The jobs

| group | job | what | GPUs | artifact | cost estimate |
|---|---|---|---|---|---|
| `sphere` | `seed{0..7}` | 3-D level-set sphere (`h_φ = 0.1` on the 0.2 grid, `fnum × 2` ≈ 40 k particles), radius gradient vs CRN FD (h = 0.05, 0.02), one seed per job | 1 | `gradient_bench/results/ensemble_mesh_sphere_levelset0.1_gz0.05_radius_f2_…_dev_seed{i}.json` → merged `…_dev.json` | ~3× the circle's 9 min/seed on the workstation GPU → ≤ 30 min/seed, 8 h limit |
| `mode2` | `seed{0..31}` | finer circle (`fnum × 1` ≈ 48 k particles, 40 × 40 DSMC grid, `h_φ = 0.0625`), `cos 2θ` gradient vs CRN FD, one seed per job | 1 | `…circle_levelset0.0625_gz0.05_grid2_mode2_f1_…_dev_seed{i}.json` → merged | ~15 min/seed on the workstation → < 1 h, 4 h limit |
| `opt` | `opt_S{1,2,4,8}_{gd,adam}` | finer circle drag minimisation at fixed area, 20 iterates, S CRN seeds per iterate evaluated concurrently (`optimize.py --workers min(S, 3)`), GD (normalised, projected) vs Adam | min(S, 3) | `mesh/results/optimize/circle_fine_S{S}_{opt}/history.json` + `coef_k.npy` | ~8 min per seed-evaluation on the workstation; per iterate ⌈S / GPUs⌉ × 8 min → S = 1, 2: ~3 h; S = 4: ~6 h; S = 8: ~9 h (24 h limit) |

Sizing: `submit_curc.sh` gives every job 21 cores and 70 GB per GPU (a third of an
`aa100` node per GPU) and `--time` 8 h (sphere), 4 h (mode2), 12 h (opt, S ≤ 2), 24 h
(opt, S ≥ 4).  The `opt` jobs with S ≥ 3 take a whole node's GPUs; if the account's
concurrent-GPU limit bites, submit `opt` first.  Estimates are extrapolated from the
workstation (RTX 3050 Ti, weak fp64; the A100's fp64 is ≥ 10× faster, so they are upper
bounds); the first job's log calibrates them.

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
(the `∇_vλ` part of a flat-faced body, −25 %) — do **not** gate them: no cluster job has a
diffuse wall, and none starts from a flat face.  The capsule question would matter only if
an optimisation iterate developed a flat, cap-terminated face; the drag history and
`coef_k.npy` sequence show whether it does.

`collect.py` draws `figures/optimize_convergence.png` (J per iterate and per unit cost `S
× iterates` for every run, the area constraint), `figures/gradient_seed_convergence.png`
(bootstrap SE of the `cos 2θ` gradient vs the number of seeds against 1 % of the FD) and
prints the sphere gate row; everything from the artifacts.

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
