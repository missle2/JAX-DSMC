# Cluster study on CURC Alpine

Status: **current**.  The SLURM tooling for every large run of the mesh study (3-D sphere and
finer-circle gradient gates, shape optimisations, optimiser step-rule comparisons, bias and
adjoint-error gates) on the University of Colorado Research Computing Alpine cluster, one
right-sized job per task.  The job matrix is `study.py` (one *group* = one `submit_curc.sh`
argument); the group table at the end records what each group was for and what it found.

## Quick start

```bash
# once, on a login node (python >= 3.10; JAX_VENV overrides the path)
module load python
python3 -m venv /projects/$USER/jax-env && source /projects/$USER/jax-env/bin/activate
pip install --upgrade pip && pip install "jax[cuda12]" numpy scipy matplotlib
# verified 2026-09-21: jax 0.6.2, CudaDevice, check_env ok

# every session
module load slurm/alpine
cd /projects/$USER/JAX-DSMC                 # the cluster checkout; update it with `git pull` only
bash examples/mesh/cluster/preflight.sh <group>              # local toy run of every job kind (below)
GPU_KIND=h200 bash examples/mesh/cluster/submit_curc.sh check   # check_env on that GPU kind first
GPU_KIND=h200 bash examples/mesh/cluster/submit_curc.sh <group> [<group> ...]
GPU_KIND=h200 ONLY='regex' bash examples/mesh/cluster/submit_curc.sh <group>   # matching jobs only
squeue -u $USER
```

`submit_curc.sh <group>...` with no argument prints the usage line (the accepted groups).

### Submitting

* **`GPU_KIND`** (default `a100`):

  | kind | partition / gres | per job | notes |
  |---|---|---|---|
  | `a100` | `aa100` / `a100-40gb` | 21 cores, 70 GB per GPU | 3 GPUs per node; CUDA venv |
  | `h200` | `ah200` / `h200_2g.35gb` | 10 cores, 40 GB per **MIG slice** (35 GB GPU memory) | since 2026-10-18 the default gres is the slice; a full H200 is `GRES_TYPE=h200` (needed when a job exceeds 35 GB, e.g. `grid_scale 2`) |
  | `rtx` | read from `sinfo` (RTX PRO 6000, Blackwell sm_120, MIG) | one 24 GB slice per job (`gpu-normal` allows one) | only groups that fit 24 GB are submitted (see the `submit_curc.sh` skip list); others print "skip" |
  | `mi100` | `ami100` | 21 cores, 70 GB | **unusable** while CURC's ROCm `jax` module is 0.4.7 (the code needs >= 0.4.36); kept for a future ROCm >= 6 module or an Apptainer `rocm/jax` image |

  Overrides: `PART`, `GRES_TYPE`, `QOS` (e.g. an allocation's priority QOS; default `gpu-normal`,
  24 h, account `ucb856_asc1` in `job.sbatch`), `EXCLUDE_NODES`, `PYTHON` (the login node's
  `python3` is 3.6; the script uses the venv's interpreter), `RTX_MATCH`.
* **Wall time** is fixed per group in `submit_curc.sh`: `sphere` 8 h, `mode2` 4 h,
  `opt`/`optc`, the `squash*`, `optckn`, `optc2`, `optab`, `optm`, `optrules*` 24 h
  (`opt` S <= 2: 12 h), `biasgate`/`cellgate`/`gapgate`/`thingate`/`guardgate`/`shapeprof`/`speedprof` 12 h.
  `all` = `sphere mode2 opt`.  The allocation caps concurrent H200 jobs at 4 (2026-10-20;
  a QOS limit, not in these scripts): split big batches with `ONLY=` across `h200` and `rtx`
  (an rtx slice is ~1.8x slower than an H200 2g.35gb slice), heaviest jobs (M 10) on h200.
* **GPUs per job**: seed jobs 1; an `opt`/`optckn`/`optc2` run S seeds on `min(S, GPUs per node)`
  GPUs (`optimize.py --workers`); everything else 1.  On `rtx` every job gets one slice and the
  seeds run in turn.
* Jobs whose name is already in `squeue` are skipped, so repeating a submission is safe.
* **Resuming**: every command resumes from its own artifact (finished seed rows / optimiser
  iterates are skipped; a converged run exits at once).  A job that hit its time limit is
  resubmitted with the same line.  A resumed optimiser keeps the backward basis it started
  with.  The optimisers write a warm-start gas (`<run>/warm/seed<i>.npz`) every iterate.
* **Node safety**: `env.sh` pins `JAX_PLATFORMS` (a CPU fallback is an error);
  `job.sbatch` runs `check_env --quick` and an `import jax_dsmc, examples.ensemble_mesh` check
  from `/tmp` before the command; a node that fails the device check is appended to
  `bad_nodes.txt`, and `submit_curc.sh` passes that list to `sbatch --exclude`.  (Why: on
  2026-09-22 all eleven `ah200` jobs landed on one node whose driver refused `cuInit`; JAX fell
  back to the CPU silently and ran 4x slower.  The CPU results were valid, only expensive.)
  A persistent XLA compilation cache is shared at `/projects/$USER/jax-cache` (clear it after
  a jax upgrade).

### Preflight (do this before a new group or GPU kind)

```bash
bash examples/mesh/cluster/preflight.sh <group> [name-regex]     # repo root; CPU, a few minutes per job kind
GPU_KIND=h200 bash examples/mesh/cluster/submit_curc.sh preflight <group> [<group>...]   # on 2 GPUs of that kind
```

It runs the `job.sbatch` import check, `CHECK=1` on every `scratch/*/cluster_job.sh` job (name
parses, inputs exist), and one toy run per code path (`SMOKE=1`; optimisers get toy flags and a
`/tmp/preflight` tag).  The GPU variant (`PREFLIGHT_HOURS`, default 2) uses a 600-step warm-up
so the body is hit; the log warns on `hits 0`.  It exists because every `biasgate` job of
2026-10-04 died on `import jax_dsmc` after queueing (`python3 scratch/...` puts only the
script's directory on the path; `env.sh` now exports the repository root in `PYTHONPATH`).
A toy cannot show what depends on density, shape or memory.

### Where outputs land, and getting them

| what | where (cluster checkout) |
|---|---|
| `sphere`, `mode2` seed rows | `examples/gradient_bench/results/…_dev_seed{i}.json` (merged by `collect`) |
| optimisations | `examples/mesh/results/optimize/<tag>/` (`history.json`, `coef_<k>.npy`, `frames_<k>.npz`, `warm/`), `optimize_squash/<tag>/` |
| scratch-script groups (`biasgate`, `cellgate`, `gapgate`, `thingate`, `guardgate`, `shapeprof`, `speedprof`) | `scratch/<probe>/cluster/<job>.json[l]` (+ `maps/`) |
| job logs | `examples/mesh/cluster/logs/<group>_<name>_<jobid>.out` |

**Since 2026-10-21 run data is not in git.**  `examples/mesh/results/optimize*`,
`examples/mesh/cluster/logs`, `bad_nodes.txt` and `scratch/*/cluster/` are gitignored; git
carries code, docs, figures and analyses, and the cluster checkout only ever `git pull`s.
Move data to the laptop with

```bash
bash examples/mesh/cluster/pull_results.sh                     # results, logs, figures, scratch/*/cluster
DRY=1 bash examples/mesh/cluster/pull_results.sh               # list only
WARM=1 bash examples/mesh/cluster/pull_results.sh              # also the warm-start gas (~0.2 GB per seed)
bash examples/mesh/cluster/pull_results.sh examples/mesh/results/optimize   # just these paths
```

(`REMOTE_USER`, `HOST`, `REMOTE_DIR` override the login; password + Duo once per run; pull
only, nothing deleted, a newer local file is kept.)  `adopt_untracked.sh` was the **one-time**
transition on the cluster checkout (2026-10-21) that untracked the data without deleting it;
it is not needed again.

Then `python3 -m examples.mesh.cluster.collect` merges the per-seed files and draws
`figures/optimize_convergence[_coarse].png`, `figures/optc/{optc_drag,optc_body_extent,
gradient_seed_convergence}.png`, `figures/gates/mode2_gate.png`, the squash figures and the sphere gate
row, and writes `study_results.json`.  `python3 -m examples.mesh.movie --tag <tag> [--mp4]
[--stride s]` strings an optimisation's `frames_<k>.npz` / `coef_<k>.npy` into a movie.

### Files

| file | role |
|---|---|
| `study.py` | the job matrix: `python3 -m examples.mesh.cluster.study --group <g> --list\|--names\|--gpus [--gpus-per-node N]` |
| `submit_curc.sh`, `job.sbatch` | one `sbatch` per job line; `job.sbatch` logs the node, runs the preflight, then `$CMD` |
| `env.sh` | the job environment (venv, `JAX_ENABLE_X64=1`, `JAX_PLATFORMS`, no preallocation, compilation cache, `PYTHONPATH`) |
| `check_env.py`, `check_env.sbatch` | JAX >= 0.4.36, an accelerator, float64, then a 40-step level-set circle with shape terms (`--quick`: devices only) |
| `preflight.sh`, `collect.py`, `pull_results.sh`, `adopt_untracked.sh` | as above |
| `bad_nodes.txt` | nodes that failed the device check (untracked) |

Sizing note: the finer circle (240 k slots x 300 captured steps; `mode2`, `opt`) needs
> 18.7 GB in the captured forward alone and ran out of memory on the 24 GB rtx slices
(2026-09-23), so `submit_curc.sh` offers only the groups that fit there.  The seed-parallel
design (`--seed-start 32+i --seeds 1`, the RNG contract of `ensemble_mesh`) makes the merged
artifact identical to a serial run.  `optimize.py --workers W` evaluates an iterate's seeds in
W subprocesses pinned to the job's GPUs and combines the per-seed gradients into the serial
mean (checked: next-iterate coefficients to 3e-15).

## What the study answers

(1) the 3-D shape gradient passes the circle's gate (`sphere`, `mode2`); (2) how many seeds an
evaluation needs (SE vs S); (3) which optimiser / step rule converges per unit cost; (4) where
the adjoint is biased (the gates).  The cluster jobs are all **specular** bodies with curvature
everywhere; the diffuse circle and flat-faced bodies are separate workstation questions (docs:
`ADJOINT_BRANCH_REVIEW.md` §7.9).  What `optimize.py` calls `gd` is projected steepest descent
with an H1-smoothed gradient and a step normalised to a maximum normal displacement, not a
Newton method.

## Groups

Settings shared by the finer-circle jobs (`study.FINE`): `levelset_samples 16`, `grazing 0.05`
(PSEUDOCODE.md §4.4), collisionless `--substeps 1 --partners 2`; collisional jobs use
`--substeps 4 --partners 8`.  The backward's field fit is `ensemble_backward.CELL_BASIS`
(`cell:cubicv:adapt1.5:guard32:align0.9:both` since 2026-10-19; PSEUDOCODE.md §9.7.2), a
guarded cubic per collision cell: a Kn 0.5 squash seed took 985 s -> 95 s on the workstation
(2026-10-11), so collisional runs afford ~10x the iterates.

### Gradient jobs

| group | jobs | purpose / result |
|---|---|---|
| `sphere` | `seed{0..7}` | 3-D level-set sphere (`h_phi = 0.1`, `fnum x 2`), radius gradient vs CRN FD (h = 0.05, 0.02): the first 3-D shape-gradient gate |
| `mode2` | `seed{0..31}` | finer circle (`fnum x 1`, 40 x 40 grid, `h_phi = 0.0625`), `cos 2 theta` gradient vs CRN FD; 32 seeds give the SE vs S curve (`gradient_seed_convergence.png`) |
| `squash` | `grad_s{0.35,0.6,1}`, `opt_S{1,4}` | aspect-ratio problem (2-D ellipse, analytic free-molecular reference): gradient vs FD and `optimize_squash` runs |
| `squashkn` | `kn0.5[_M10]_S{1,4}` | the squash problem at Kn 0.5 / + M 10 with the per-query fit; the 2026-10-01 runs reached only 10 and 3 iterates in 24 h (superseded by `squashkn2`) |
| `squashkn2` | `kn0.5_S{4,8}`, `kn0.5_M10_S{2,4}` | `squashkn` with the per-cell fit, to convergence (`s` of the last 5 iterates within 0.5 %, cap 300); tags carry the basis so they never resume per-query runs |

### Optimisations

| group | jobs | purpose / result |
|---|---|---|
| `opt`, `optc` | `opt_S{1,2,4,8}_{gd,adam}` | finer-circle (`opt`) / SPARTA-grid (`optc`, 60 k slots, fits an rtx slice) drag minimisation at fixed area, S CRN seeds per iterate, gd vs Adam, collisionless; cap 300 iterates.  Measured `optc`: ~4.3-4.9 min per seed evaluation.  Tags `circle_{fine,coarse}_S{S}_{opt}`.  The optc bodies grew through the outflow face -> the `--keepout` runs below |
| `optckn` | `opt_S{1,2,4,8}_adam`, `opt_S4_gd` | `optc` at Kn 0.5 in the 3x longer box (`circle_long`, x in [-10, 20] m, `fnum x 2`, warm-up 2800), bodies kept 1 m from every face (`--keepout 1.0`), `--area-mode tangent` (step projected onto the constant-area set; the augmented-Lagrangian `_al` runs wound up), step smoothed over 8 knots (`--smooth 8`); tags `circle_long_kn0.5_S{S}_{opt}_tan_sm8`; ~10 min / seed |
| `optc2` | `opt_S{4,8}_adam` | the same free-molecular; tags `circle_long_S{S}_adam_tan_sm8` (earlier projection runs `circle_long_S{S}_adam` stay) |
| `optab` | `ctl`, `smooth{4,8}`, `lsh{0.125,0.25}` | A/B of control resolution against sub-cell wiggles on `optc2`'s S = 4 Adam run.  **Result (round 2, 2026-10-15, tangent area; round 1 with the Lagrangian area is void: the multiplier wound up)**: area exact to 1e-12; 50 cm smoothing (`smooth8`) gives the smoothest shapes and the best adjoint check (observed/predicted step 0.88), drag 0.23-0.26 of the circle's; chosen for the long runs (`scratch/optab/analysis_tan.py`, `figures/optc/optab/optab_tan_*.png`) |
| `optm` | `kn0.5`, `M10`, `kn0.5_M10` | long S = 4 Adam optimisations in three regimes, **in free space** (`--boundary free`: the specular channel choked at Kn 0.5 + M 10 and holds a half-range stream; free boundaries reproduce the free-molecular Cd 12.25 +- 0.04 vs 12.22, `scratch/freebox/`), one GPU each.  Tags `circle_long_{kn0.5_,M10_,kn0.5_M10_}S4_adam_tan_sm8_free`.  **Result (2026-10-16/17, `scratch/optm/analysis_optm.py`)**: Kn 0.5 reaches 0.21 of the starting drag and plateaus; free-molecular M 10 reaches 0.104 until the body hits the keep-out; Kn 0.5 + M 10 saw-tooths 0.16-0.31 because of heavy-tailed nose gradients (spikes to 14 000x) that Adam's momentum carries for 5-25 iterates, and the nose grew a sub-2-cell sting where the old `cubicv` fit broke down (see `thingate`).  The Adam runs are the baseline for `optrules` |
| `optrules` | `{kn0.5,kn0.5_M10}_{adam,tr,lbfgs,mma}` | step rules (`examples/mesh/optimizers.py`; `tr` trust region, `lbfgs` SQP at constant area, `mma` Svanberg) on the `optm` problems with the guarded backward (tags `*_g32`).  **Result (2026-10-20)**: best J/J0 Adam 0.203 / 0.105 (Kn 0.5 / M 10) vs tr 0.311 / 0.222 (fastest per iterate, then stalled: the radius random-walked to its floor and a rejection repeated the identical trial), lbfgs 0.80 / 0.41 (curvature pairs are seed noise), mma 0.65 / 0.45.  Reference runs, not to be resubmitted (the tags resume).  Analysis `scratch/optrules/analysis.py`, `figures/optc/optrules/optrules_*.png`.  The old `trper` perimeter arm was dropped |
| `optrules2` | `{kn0.5,kn0.5_M10}_tr2` | the reworked `tr` (2026-10-20): rho pooled over 5 trials from paired per-seed differences, radius changes only beyond rho's error, rejection only when worse beyond noise, a rejection at the floor re-evaluates the base on new seeds.  S = 4, warm-started `--restart-steps 500`; tags `*_S4_tr2_*_w500`.  **Result (2026-10-21)**: no longer loops, but still parks at its floor (58-86 % of iterates); best 0.254 (Kn 0.5) / 0.238 (M 10), then drifts up at M 10.  Done |
| `optrules1seed` | `{kn0.5,kn0.5_M10}_S1_{adam,tr,lbfgs,mma}` | every rule with ONE seed per iterate (the noisy-gradient regime of complex 3-D flows), warm-started, up to 1200 iterates; tags `*_S1_<arm>_*_w500`.  **Result (2026-10-21, `scratch/optrules/noise.py`, `roughness.py`; figures `optc/optrules/optrules_noise*.png`, movies `movie/optrules/optrules_*_noise.mp4`)**: S1 Adam matches S4 Adam per iterate at 1/4 the cost (noise is not its limit) and repeats its M 10 overshoot (~0.105 at iterate ~150, ~0.19 by 330: the fixed step size is); S1 `tr` / L-BFGS drift up after ~50; S1 MMA fragments shapes (15 pieces at M 10).  Done; not resubmitted |
| `optax1seed` | `{kn0.5,kn0.5_M10}_S1_ox_{adam,ademamix,adamcos,sfadam,prodigy}` | the optax step rules of `examples/mesh/optax_rules.py` (2026-10-21) with one seed per iterate, warm-started, 400 iterates: `ox_adam` (the shipped Adam as an optax chain: control), `ox_adamcos` (Adam with a cosine step schedule 0.1 -> 0.0125 m over 300 iterates), `ox_sfadam` (schedule-free Adam), `ox_prodigy` (learning-rate free, capped at 0.2 m), `ox_ademamix` (slow + fast momentum).  Needs `optax` in the cluster venv.  Tags `*_S1_ox_<arm>_*_w500` |
| `optcm` | `{kn0.5,kn0.5_M10}_S1_ox_{adam,adamcos,prodigy,prodigycos}_cm` | (2026-10-23) the body's area centroid held at its start (`--fix-centroid 1`: every Adam-type M 10 run of optax1seed / optrules1seed slid ~10 m downstream onto the outflow face) for Adam, Adam + cosine step, Prodigy and Prodigy with its 4-step cap decaying on a cosine to step / 4 (`ox_prodigycos`: plain Prodigy's step sat at the cap from iterate ~25 on); S = 1, warm-started, 400 iterates (~10-13 h at the optax1seed pace).  Tags `*_S1_ox_<rule>_cm_*_w500`; `scratch/optax/` for the comparison figures and movies |
| `optct` | `{kn0.5,kn0.5_M10}_S1_ox_{adam,adamcos,prodigy,prodigycos}_ct` | (2026-10-23) `optcm` again with the body started 2 m downstream (`--start-shift 2.0`, centroid x = 7: the `optcm` M 10 noses stopped at x ~ -0.6, ~9.4 m from the inflow face) and the centroid held by rigid translations: the gradient's translation part removed, the step corrected along translations, the body moved back after it.  The `optcm` version reshaped the nose and base by the centroid's coefficient gradient (ragged bases, roughness 13-25 at M 10); its runs are done, and resubmitting `optcm` would only resume them.  Tags `*_S1_ox_<rule>_ct_*_w500` |
| `optrules3` | `{kn0.5,kn0.5_M10}_S1_{lbfgs_fs,mma_s}` | (2026-10-24) L-BFGS and MMA with their step-length controls replaced: `lbfgs_fs` (the L-BFGS direction at a fixed largest step 0.05, no trust region -- `lbfgs`'s radius sat at its floor from iterate ~60, rho swamped by one seed's noise) and `mma_s` (MMA on Adam's averaged gradient, its step smoothed -- `mma`'s asymptotes shrank on noisy sign flips and its separable step was rough). One seed, warm-started, 400 iterates, optrules1seed's box and settings. ~7.5 h (Kn 0.5 on rtx, M 10 on H200; M 10 on rtx ~10 h). Tags `*_S1_{lbfgs_fs,mma_s}_*_w500` |

### Gate groups (scratch scripts, `bash scratch/<probe>/cluster_job.sh <job>`)

| group | jobs | purpose / result |
|---|---|---|
| `biasgate` (2026-10-04, `scratch/bias/`) | `kn_seed{0..7}`, `kn_fd{0..3}`, `kn_hor_seed{0..3}`, `fm_hor_s1.0`, `fm_hor_s0.35` | the bias study's Kn 0.5 gate at production settings, the FD arbiter, and the 300-step horizon vs the whole run.  Only `ONLY='^kn_fd\|hor'` is still of interest (`kn_seed*` is superseded by `cellgate`).  Summaries: `scratch/bias/summarise.py`, `horizon_summary.py` |
| `cellgate` | `sq_s1_seed*`, `sq_s0.6_seed*`, `optc_{c0,a50,a100,a200,s8a30,g200}_kn_b*`, `_fm_b*` | per-cell `CELL_BASIS` (quad and cubic) against the per-query k = 32 fit on the Kn 0.5 squash and the optc iterates, with FDs |
| `gapgate` (2026-10-12, `scratch/gapgate/analysis.py`) | `sqh_*`, `circh_*`, `cub_*`, `fmgz_*`, `sqfm_*`, `fmsm_*` | the Kn 0.5 gap: horizons, cubic vs quadratic, free-molecular grazing cut-off, smoothed iterates.  **Result**: the default basis became `cell:cubicv:adapt1.5:align0.9:both` (later guarded, see `guardgate`) |
| `thingate` (2026-10-17/18, `scratch/thingate/summarise.py`) | `sq_M10_s*`, `sq_M10off_*`, `sqg2_*`, `sq_M032_*`, `sq_FM10_*`, `om_it*` | the thin-body adjoint error: five backward fits on one forward vs FD.  **Result**: the unguarded cubic-in-v fit is the error, and only at M 10 (squash s = 0.193: -19.6 +- 8.6 against FD +1.48; wrong at every `optm` spike iterate); the quadratic fit is within 3 %; M 0.32 and free-molecular agree for every fit.  `sqg2`/`omg2` (`grid_scale 2`) need a full H200 (`GRES_TYPE=h200`; the 35 GB slice ran out of memory) |
| `guardgate` (2026-10-18/19, `scratch/guardgate/analysis.py`, `figures/optc/gates/guardgate.png`) | `sq_M032_*`, `sq_M10_*`, `om_it*`, `unit` | leverage-guarded cubic, h = 8 / 32 / 128, paired with the unguarded cubic and the quadratic.  **Result**: the guard removes the blow-up everywhere (max lambda 1e-4..2e-3, thin and `optm` cases within ~2 s.e. of the quadratic); on the blunt squash G = C (-1.4 +- 1.1 % vs FD).  Residual: `optm` iterate 58 reads -3 to -6 % along the nose, the thinnest squash -3.5 to -4.2 % in every fit.  Shipped default: guard 32 |
| `shapeprof` | `optm_it{30,156}` | per-phase seconds of the shape gradient on `optm` iterates (perfetto trace of one seed; H200); `scratch/shapeprof/profile.py` |
| `speedprof` | `m10_it156`, `kn05_it200` | one seed's phases on a production iterate (M 10 optm, Kn 0.5 optrules1seed) with XLA command buffers off (the old WSL guard) / XLA default (env.sh since 2026-10-22) / + WHILE, CONDITIONAL, and `shape_batch` 1 / 4 / 12; peak memory and gradients compared; the 2026-10-22 shape-step speed-ups included. ~2 h, one GPU (H200 to compare with the runs' histories); `scratch/shapefast/` |
| `boxcheck` | `{kn0.5,kn0.5_M10}_{long,xlong}` | (2026-10-24) does the drag depend on where the body sits? The adjoint's dJ/dx along a rigid translation and a paired 1 m finite difference (six seeds, J-normalised) for the circle and the free M 10 Adam body at iterate 100, in `circle_long` and in `circle_xlong` (inflow face 15 m further upstream, warm-up 4200). If the long box's dJ/dx is at the noise, the shape runs move there without a centroid constraint. ~1 h per job (24 cold seeds), one GPU; `scratch/boxcheck/translation.py` -> `scratch/boxcheck/cluster/<job>.json` |
| `opt3dcheck` | `sphere_{kn0.5,kn0.5_M10}` | (2026-10-24) the first 3-D optimiser runs: the `sphere` case (r = 1 m, +-2 m box, 0.2 m cells), Adam at fixed volume, one seed, 5 iterates -- every step of the loop in 3-D. Lengths scaled from the circle's (step 0.02, smoothing 3 x 0.1 m knots, band 0.4 m, margin 0.3 m). The forward needs ~10 GB. Tags `sphere_<regime>_S1_ox_adam_check` |
| `adjbias` | `fm`, `fm_gz0.01`, `fm_gz0.001`, `kn05`, `kn05_gz0.001`, `kn05_bn`, `long_kn05`, `long_m10` | (2026-10-24) which shape-gradient term carries the translation bias (`boxcheck`: adjoint dJ/dx +7.3 % J/m at Kn 0.5 against a zero FD): every term projected on the spline's rigid x-translation, 16 seeds per rung. Free-molecular (true value 0) at grazing floors 0.05 / 0.01 / 0.001 of the hit-time derivative; Kn 0.5 at 0.05, 0.001 and with one-sided wall fits; the production long box at Kn 0.5 and M 10. Laptop (6 FM seeds): pathwise_tally +1.8 +- 0.6, total +2.5 +- 0.5 % J/m. `scratch/adjbias/terms.py` -> `scratch/adjbias/cluster/<rung>.json`; the circle rungs fit an rtx slice |
| `adjfix` | `check_{kn0.5_fix,kn0.5_M10_bn,kn0.5_M10_gz,kn0.5_M10_fix}`, `{kn0.5,kn0.5_M10}_S1_ox_adam_fix` | (2026-10-24) the translation-bias fix found by `adjbias`: one-sided wall fits (`--backward-normals true`, removes -5.7 of the circle's +5.9 % J/m at Kn 0.5) and grazing floor 0.01 (removes the +1.3 % J/m free-molecular part). `check_*`: `boxcheck`'s adjoint-vs-paired-FD on the circle and the free M 10 Adam body (iterate 100), 8 seeds, circle_long (M 10 with each change alone and both) -> `scratch/boxcheck/cluster/adjfix_<name>.json`, ~3-4 h each. `*_ox_adam_fix`: optax1seed's free ox_adam with the fix, 400 iterates, warm-started; does the body stay put? Kn 0.5 jobs fit rtx; M 10 needs an H200 slice (boxcheck M 10 ran out of memory on rtx) |
| `adjm10` | `terms_{body,circle,body_fm,body_vec,body_fm_vec}`, `check_{g16,g64,ppc2,ppc4,fine,fine_half}` | (2026-10-25) the M 10 translation bias `adjfix` left (free M 10 Adam body -9 % J/m against a ~zero FD with both fixes, circle -1.1), all at Kn 0.5 M 10 with `--backward-normals true --grazing 0.01`. `terms_*`: `adjbias`'s term breakdown (16 seeds) on the M 10 body, the circle and the body free-molecular at M 10 (true value 0: collision adjoint or wall terms?) -> `scratch/adjbias/cluster/adjm10_<name>.json`; `*_vec` (8 seeds, rtx) also save the per-seed term vectors (`adjm10_<name>_vectors.npz`) for offline projections (the body's translation number flips sign between two translation discretisations). `check_*`: `boxcheck`'s adjoint-vs-FD (8 seeds) with fit neighbours 16 / 64, 2x / 4x particles per cell, and 0.25 m DSMC cells (`--grid-scale 2`; 0.5 m is ~lambda / 1.5 behind the bow shock) at the same / half the particles per cell -> `scratch/boxcheck/cluster/adjm10_<name>.json`. `terms_*`, `check_g*` fit an rtx slice (M 10 circle_long does); `check_ppc*`, `check_fine*` (2-4x the particles) take whole H200s (the two submit lines are in `study.py`'s adjm10 comment) |

Formulation checks behind the long box (2026-10-11): the empty `circle_long` box's windowed drag
is steady after ~2100-2800 warm-up steps (`scratch/cellfit/warmup_check*.py`); the NTC attempt
budget never clips at `a_max 64` on the optc shapes (`scratch/cellfit/attempts_check.py`);
`keep_inside` + `project_area` hold the margin and the area exactly on a shape that had crossed
a face.  Every optimisation runs to convergence, capped at `study.MAX_ITERATIONS = 300`
(`optimize.window_converged`; `collect` counts a converged run as done).
