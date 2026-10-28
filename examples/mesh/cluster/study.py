"""The cluster study's job matrix (plan phase E, 2026-09-21), one group per task.

    python3 -m examples.mesh.cluster.study --group sphere --list   # one command per seed
    python3 -m examples.mesh.cluster.study --group mode2 --list    # one command per seed
    python3 -m examples.mesh.cluster.study --group opt --list      # one command per (S, opt)
    python3 -m examples.mesh.cluster.study --group opt --gpus      # GPUs per opt job

Every command is a plain ``python3 -m examples....`` invocation that resumes from its own
artifact (rows / iterates already done are skipped), so a job can be re-submitted after a
time limit.  The optimisations (``squash*``, ``opt*``) iterate to convergence -- the last 5
iterates' objective (``optimize``) or ``s`` (``optimize_squash``) within a tolerance of
their mean, ``optimize.window_converged`` -- capped at :data:`MAX_ITERATIONS`; a converged
run resubmitted exits at once.  The groups (``submit_curc.sh`` sizes one SLURM job per line):

* ``sphere``: the 3-D level-set sphere, radius gradient vs CRN FD, 8 seeds -- **one job
  per seed** (``--seed-start 32+i --seeds 1 --output ..._seed{i}.json``; ``collect.py``
  merges the per-seed files into the usual artifact) -- the first 3-D shape gradient
  (``fnum x 2``, ``h_phi = 0.1`` on the 0.2 grid).
* ``mode2``: the finer circle (``fnum x 1`` = 48 k particles, DSMC grid 40 x 40,
  ``h_phi = 0.0625``), the ``cos 2 theta`` mode gradient on 32 seeds, one job per seed --
  the seed-count convergence of the gradient itself (bootstrap SE vs seeds).
* ``optc``: ``opt`` on the SPARTA 20 x 20 DSMC grid (``fnum x 1``, same level set), tags
  ``circle_coarse_S{S}_{opt}``: the variant that fits the 24 GB rtx MIG slices.
* ``squash``: the aspect-ratio test with an analytic reference
  (``examples/mesh/ellipsoid_reference.py``): the ``ellipse`` case (circle box, R = 1.5,
  semi-axes ``(R, s R)``) on :data:`COARSE` settings.  ``grad_s{s}``: ``dD/ds`` (control
  ``squash_y``) at ``s`` in :data:`SQUASH_POINTS` vs CRN FD and the reference, 8 seeds, the
  unstratified fit (``--value-class none``: the hit-ahead class biased mode2,
  2026-09-25); ``opt_S{S}``: ``examples.mesh.optimize_squash`` (Adam, learning rate 0.05)
  from the circle (``s = 1``) on ``J = D - lam V`` with the reference optimum at
  ``s* = 0.35``, to convergence: the optimiser on DSMC gradients against the analytic
  optimum.  Fits a 24 GB rtx slice (80 k slots, like ``optc``).
* ``squashkn``: the same optimisation in the transitional regime, where the free-molecular
  reference no longer holds: the gas density raised to ``Kn = lambda / 2R = 0.5`` at the
  same particle count (``--knudsen 0.5``, NTC collides every step; ``lam`` still from the
  free-molecular reference), ``kn0.5_S{S}`` at the deck's stream and ``kn0.5_M10_S{S}`` at
  M 10 (:data:`KN05_M10`), ``S`` in :data:`SQUASH_KN_SEEDS` (no 8-seed runs: these cost more), the shipped
  ``4 x 8`` virtual pull-back.
* ``biasgate``: the 2026-10-03 bias study's gates at production settings, which do not fit
  the 4 GB workstation GPU (``scratch/bias/``; the jobs are ``bash scratch/bias/cluster_job.sh
  <name>``, whose header documents each): the Kn 0.5 paired gate (``kn_seed{i}``, 8 jobs),
  its FD-only arbiter (``kn_fd{j}``, 4 x 16 seeds), the Kn 0.5 and free-molecular horizon
  tests against the whole run captured (``kn_hor_seed{i}``, ``fm_hor_s{s}``).  Results in
  ``scratch/bias/cluster/<name>.jsonl``.
* ``cellgate`` (2026-10-11): FD gates of the per-cell backward fit (the shipped
  ``ensemble_backward.CELL_BASIS``), ``bash scratch/cellfit/cluster_job.sh <name>`` (its
  header documents each; results in ``scratch/cellfit/cluster/<name>.jsonl``): the Kn 0.5
  squash at production density, every fit paired with the CRN FD (``sq_s1_seed{i}``, 16
  seeds; ``sq_s0.6_seed{i}``, 8), and six optc iterates (:data:`OPTC_SHAPES`) with 16 seeds
  each at Kn 0.5 (``optc_{shape}_kn_b{j}``, 4 seeds a job) and free-molecular
  (``optc_{shape}_fm_b{j}``, 8 a job), FDs along the uniform shift and the cos 2 theta
  mode.
* ``optckn`` (2026-10-11): ``optc`` in the transitional regime (``--knudsen 0.5``, the 4 x 8
  pull-back, the per-cell fit that makes it affordable) in a 3x longer box (:data:`LONG`:
  15 m each side of the body; the free-molecular Adam runs grew through the outflow face
  of the 10 m box and one
  reached the inflow face) with the bodies also kept :data:`KEEPOUT` from every face
  (``optimize.keep_inside``), Adam at ``S in OPTCKN_SEEDS`` and GD at 4,
  to convergence (resumed across jobs).
* ``optc2``: the free-molecular ``optc`` Adam runs again in the long box with the margin,
  S = 4, 8.
* ``gapgate`` (2026-10-12): the hypothesis gates after ``cellgate`` (``bash
  scratch/gapgate/cluster_job.sh <name>``, whose header documents each; results in
  ``scratch/gapgate/cluster/<name>.jsonl``): the Kn 0.5 gap (cubic-in-``v`` fit and the
  whole-run horizon paired, in the production run and a filled box, squash ``sqh_*`` and
  circle ``circh_*``; the cubic fit on the optc iterates ``cub_*``), the free-molecular
  remainder and the S8 ``cos 2 theta`` bias (grazing cut-off sweeps ``fmgz_*`` / ``sqfm_*``),
  the elongated iterates' heavy-tailed gradients (smoothed iterates ``fmsm_*``).
* ``squashkn2``: ``squashkn`` with the per-cell fit at ~10x the iterates per hour: Kn 0.5 at
  S = 4, 8, Kn 0.5 + M 10 at S = 2, 4, to convergence.
  Its tags carry the basis (``optimize_squash``), so they never resume the per-query runs.
* ``optab`` (2026-10-13): A/B of the control's resolution against the sub-cell wiggles
  (gapgate: features of 3.5-14 cm radius, under a quarter of the 0.5 m DSMC cell, carry the
  per-seed gradient spikes): ``optc2``'s S = 4 Adam run (free-molecular, long box, margin,
  Lagrangian area) with the step smoothed wider (``smooth{n}``: ``--smooth n`` knots of
  6.25 cm, control ``ctl`` = 2 knots = 12.5 cm) or on a coarser level set (``lsh{h}``:
  ``--levelset-h h`` with ``--smooth 2``, i.e. 25 / 50 cm, the same physical widths as
  ``smooth4`` / ``smooth8``: representation vs smoothing at equal width).  Tags
  ``circle_long_S4_adam_tan_ab_<name>``; one GPU each (fits an rtx MIG slice).  The first round
  (2026-10-13, tags ``..._al_ab_<name>``, ``--area-mode lagrangian``) is confounded: the
  multiplier wound up and the area swung 4-220 % of ``A0`` (``scratch/optab/analysis.py``), so
  the group now runs ``--area-mode tangent`` (exact area: a Lagrange multiplier per iterate).
* ``optm`` (2026-10-15): the long optimisations across regimes, one GPU each (an H200: a
  multi-GPU request waits in the queue), S = 4 Adam, 50 cm step smoothing, tangent area:
  ``kn0.5`` (the deck's stream, M 0.32), ``M10`` (free-molecular) and ``kn0.5_M10``, in free
  space (``--boundary free``: every face a free-stream face), tags
  ``circle_long_{kn0.5_,M10_,kn0.5_M10_}S4_adam_tan_sm8_free``.
* ``thingate`` (2026-10-17): the adjoint of bodies under ~2 DSMC cells thick at Kn 0.5 + M 10
  (the squashkn2 cubic spike, the optm nose spikes), ``bash scratch/thingate/cluster_job.sh
  <name>`` (its header documents each; results in ``scratch/thingate/cluster/``): the backward
  fits (quadratic, cubic-in-v and three guarded cubics) paired on one forward against CRN FDs
  -- squash at s = 0.193 .. 0.45 (``sq_M10_*``; a quarter cell off the cell faces ``sq_M10off_*``; a 2x finer
  grid ``sqg2_*``; subsonic ``sq_M032_*``; free-molecular ``sq_FM10_*``) and the optm Kn 0.5
  M 10 iterates with the optimiser's own seeds (``om_it*``, ``omg2_it*`` on the finer grid).
  ``sq_*``: an rtx slice; ``sqg2_*`` / ``om*``: one H200 each.
* ``guardgate`` (2026-10-18): the leverage-guarded cubic (``cell:cubicv:...:guard<h>``, h = 8 /
  32 / 128) against the shipped cubic and the quadratic, ``bash scratch/guardgate/cluster_job.sh
  <name>`` (results in ``scratch/guardgate/cluster/``): the blunt Kn 0.5 squash at s = 0.6
  (``sq_M032_s0.6_b*``, 16 seeds), thingate's thin M 10 squash (``sq_M10_*``), optm iterates
  30 / 58 / 156 / 246 (``om_it*``, one H200 each) and the stored device gates (``unit``).
* ``optrules`` (2026-10-19): the optimiser's step rules compared (``examples/mesh/optimizers.py``):
  Adam, the trust region ``tr``, L-BFGS in the trust region ``lbfgs`` (the quasi-Newton SQP
  step at constant area) and MMA ``mma``, each on the optm Kn 0.5 and Kn 0.5 + M 10
  free-space problems with the guarded cubic backward (tags ``*_g32``).  One H200 each, 24 h.
  Done 2026-10-20 (scratch/optrules/analysis.py): the reference runs, not to be resubmitted
  (the tags would resume); ``trper`` (a 5 % perimeter penalty) dropped from the group.
* ``optrules2`` (2026-10-20): the reworked trust region (``tr``: rho pooled over 5 trials from
  paired per-seed differences, x2 / x0.5, re-evaluation on new seeds at the floor) on the same
  two problems, S = 4, warm-started (``--restart-steps 500``); tags ``*_S4_tr2_*_w500``.
* ``optax1seed`` (2026-10-21): the optax step rules (``examples/mesh/optax_rules.py``: Adam as
  an optax chain, AdEMAMix, Adam with a cosine step schedule, schedule-free Adam, Prodigy) with
  one seed per iterate, warm-started, 400 iterates; tags ``*_S1_ox_<arm>_*_w500``.
* ``optcm`` (2026-10-23): the body's area centroid held at its start (``--fix-centroid 1``: the
  Adam-type M 10 runs slid ~10 m downstream onto the outflow face) for Adam, Adam + cosine
  step, Prodigy and Prodigy with a cosine-decaying cap (``ox_prodigycos``), S = 1,
  warm-started, 400 iterates; tags ``*_S1_ox_<rule>_cm_*_w500``.  Done; the constraint then
  reshaped the body's nose and base by the centroid's coefficient gradient (ragged bases).
* ``optct`` (2026-10-23): ``optcm`` again with the centroid held by rigid translations (the
  gradient's translation part removed, the step corrected along translations, the body moved
  back) and the body started 2 m downstream (centroid x = 7: the ``_cm`` M 10 noses stopped
  ~9.4 m from the inflow face); tags ``*_S1_ox_<rule>_ct_*_w500``.
* ``optrules1seed`` (2026-10-20): Adam, ``tr``, ``lbfgs`` and ``mma`` with ONE seed per iterate
  (gradient and objective noise x2) on the same two problems, warm-started, up to 1200
  iterates (the S = 4 runs' 300 x 4 seed evaluations); tags ``*_S1_<arm>_*_w500``.
* ``shapeprof`` (2026-10-19): where a seed's time goes (``bash scratch/shapeprof/cluster_job.sh
  <name>``): the optm Kn 0.5 M 10 iterates 30 / 156 with the run's own config, per phase, the
  unguarded cubic and the plain k32 slope fit for comparison, and a perfetto trace.  One H200.
* ``boxcheck`` (2026-10-24): does the drag depend on where the body sits?  The adjoint's
  dJ/dx along a rigid translation and a paired 1 m finite difference (six seeds) for the circle
  and the free M 10 Adam body at iterate 100, in ``circle_long`` and in ``circle_xlong`` (the
  inflow face 15 m further upstream); ``scratch/boxcheck/translation.py`` ->
  ``scratch/boxcheck/cluster/<regime>_<box>.json``.
* ``adjbias`` (2026-10-24): the translation bias of the shape gradient term by term
  (``scratch/adjbias/terms.py`` -> ``scratch/adjbias/cluster/<rung>.json``): free-molecular at
  grazing floors 0.05 / 0.01 / 0.001, Kn 0.5 at 0.05 / 0.001 / with one-sided wall fits, and the
  production long box at Kn 0.5 and M 10; 16 seeds each.
* ``adjfix`` (2026-10-24): the translation-bias fix (one-sided wall fits + grazing floor
  0.01): boxcheck's adjoint-vs-FD at Kn 0.5 and M 10 (M 10 also with each change alone) and the
  free ox_adam on both optrules problems with the fix; tags ``*_S1_ox_adam_fix_*_w500``.
* ``adjm10`` (2026-10-25): the M 10 translation bias adjfix left (body -9 % J/m with the fix):
  the term breakdown on the M 10 Adam body (also free-molecular) and the circle, and the
  adjoint-vs-FD with the fit's neighbour count 16 / 64, 2x / 4x the particles per cell and
  0.25 m DSMC cells; ``scratch/{adjbias,boxcheck}/cluster/adjm10_<name>.json``.
* ``stepfd`` (2026-10-25): the adjoint against paired finite differences along the free M 10
  runs' own steps (ox_adam_fix iterates 20 / 100 / 150 / 200, mma_s 150, and 150 on knots h / 2),
  whole step / nose part / rest / rigid shift; ``scratch/adjbias/cluster/stepfd_<name>.json``.
* ``optrc`` (2026-10-25): the free M 10 runs with the slide removed (``--recentre 1``: the
  body moved back by whole knots, the coefficients and optimiser state shifted exactly) and the
  adjfix flags, ox_adam and mma_s; tags ``*_M10_S1_{ox_adam,mma_s}_rc_*_w500``.
* ``ldopt`` (2026-10-25): maximum lift-to-drag (``--objective lift_drag``, J = -L / D) with no
  area constraint (``--area-mode none``) and no centroid constraint, ox_adam and mma_s on both
  optrules problems with the adjfix flags; tags ``*_S1_{ox_adam,mma_s}_ld_none_*_w500``.
* ``sph3d`` (2026-10-25): the first 3-D production runs, the sphere in ``sphere_long`` at both
  optrules conditions, ox_adam on one full H200 each: minimum drag at fixed volume, and maximum
  L / D with no volume constraint; tags ``sphere_long_<condition>_S1_ox_adam_{drag,ld}_*``.
* ``optrules3`` (2026-10-24): ``lbfgs_fs`` (the L-BFGS direction at a fixed largest step, no
  trust region) and ``mma_s`` (MMA on the averaged gradient with a smoothed step), one seed,
  warm-started, 400 iterates, optrules1seed's settings; tags ``*_S1_{lbfgs_fs,mma_s}_*_w500``.
* ``opt3dcheck`` (2026-10-24): the first 3-D optimiser runs, the sphere case at Kn 0.5 and
  Kn 0.5 + M 10, Adam at fixed volume, one seed, 5 iterates; tags
  ``sphere_<regime>_S1_ox_adam_check``.
* ``speedprof`` (2026-10-22): one optimiser seed's phases on a production iterate (the optm
  M 10 iterate 156, the optrules1seed Kn 0.5 iterate 200) under XLA command buffers off (the
  old WSL guard) / XLA's default (env.sh now) / with WHILE, CONDITIONAL, and the shape VJP's
  ``shape_batch`` 1 / 4 / 12; peak memory and the gradients compared (``bash
  scratch/shapefast/cluster_job.sh <name>``).  One GPU, ~2 h.
* ``opt``: ``opt_S{S}_{gd|adam}``, the finer circle's drag minimisation at fixed area, to
  convergence (resumed across jobs), ``S`` CRN seeds per iterate for
  ``S in SEED_COUNTS`` and both optimisers (same step, smoothing and band; Adam's step is
  its learning rate); the ``S`` seeds of an iterate run concurrently on
  ``min(S, gpus_per_node)`` GPUs (``optimize.py --workers``; ``--gpus-per-node`` = the
  partition's GPUs per node, default ``GPUS_PER_NODE``).

Settings shared by the finer circle jobs: ``levelset_samples = 16`` so the hit search
reach ``16 h_phi / 2 = 0.5`` still covers the fastest moves, ``grazing = 0.05`` (the
cut-off of the hit-time derivative and of the diffuse score; review section 7.8),
``--substeps 1 --partners 2`` (collisionless).  Every body here is specular, so the
diffuse normal term and the silhouette term are not exercised.
"""

from __future__ import annotations

import argparse
import functools

SEED_COUNTS = (1, 2, 4, 8)
OPTIMIZERS = ("gd", "adam")
SPHERE_SEEDS = 8
MODE2_SEEDS = 32
SEED_START = 32
GPUS_PER_NODE = 3  # CURC Alpine aa100 / ami100 nodes (ah200 has 4; submit_curc.sh passes it)
FINE = "--fnum-scale 1 --grid-scale 2 --levelset-h 0.0625 --levelset-samples 16 --grazing 0.05"
# the optc group: FINE on the 20 x 20 DSMC grid (60 k slots, a quarter of FINE's memory), which
# fits a 24 GB rtx MIG slice; FINE's captured forward alone is 18.7 GB (2026-09-23)
COARSE = "--fnum-scale 1 --grid-scale 1 --levelset-h 0.0625 --levelset-samples 16 --grazing 0.05"
PULLBACK = "--substeps 1 --partners 2"
RESULTS = "examples/gradient_bench/results"
SQUASH_POINTS = (0.35, 0.6, 1.0)
SQUASH_SEEDS = 8
SQUASH_OPT_SEEDS = (1, 4)
SQUASH_KN_SEEDS = (1, 4)
# every optimisation's iterate cap: they stop earlier when converged (module doc)
MAX_ITERATIONS = 300
# the Kn = 0.5 settings (measured 2026-10-01: at most 21 NTC attempts per cell at s = 1)
KN05 = "--knudsen 0.5 --a-max 64"
# M = 10 (3079 m/s), scaled from the Kn = 0.5 measurement rather than measured: the step
# keeps a free-stream move at 0.15 cell; the slots hold the full box (1e5 at this fnum) plus
# a shock layer at ~4x density; NTC attempts go as n^2 (sigma g)_max dt, ~16 x 1.9 x 1/4
# ~ 8x the measured 21 behind the shock, so A_max 256 at Kn = 0.5.  The slots fail loudly when
# exceeded (inflow_shortfall stops the job); A_max is only the first guess since 2026-10-26 (the
# mesh forward regrows it on a clipped step, ensemble_mesh._run_grown)
MACH10 = "--mach 10 --dt 2.5e-5 --slots 160000"
KN05_M10 = f"--knudsen 0.5 {MACH10} --a-max 256"
# cellgate: the optc iterates whose gradients are FD-checked (cluster_job.sh SHAPE): the
# circle (control), S1-adam 50 / 100 / 200 (complex cut cells, 200 protrudes through the
# outflow face), S8-adam 30 (before its inflow failure) and S2-gd 200
OPTC_SHAPES = ("c0", "a50", "a100", "a200", "s8a30", "g200")
KEEPOUT = 1.0  # m: two DSMC cells between the body and every box face
# optckn / optc2 run in the 3x longer box centred on the body (forward.CASES["circle_long"]:
# 15 m each side of the body centre) at fnum x2 (120 k slots for ~95 k particles at Kn 0.5), with a
# longer empty-box warm-up: with the inflow 10 m from the centre the windowed drag was steady
# from 1400 steps (700 read 2 % low free-molecular, 5 % at Kn 0.5;
# scratch/cellfit/warmup_check.py); centred (15 m of inflow) 2100 / 2800 / 3500 steps read
# 3.631 / 3.682 / 3.692 e-20 free-molecular: 2800
LONG = "--case circle_long --fnum-scale 2 --warmup 2800"
OPTCKN_SEEDS = (1, 2, 4, 8)
# optm runs in free space (--boundary free, 2026-10-15): every face a free-stream face.  The
# specular channel choked at Kn 0.5 + M 10 (a normal shock spanning the 10 m height moved
# upstream; scratch/optm10/m10_gate.py) and held a half-range stream (free-molecular Cd 17.2
# against free space's 12.2 at M 0.32; scratch/freebox/fm_free_gate.py).  The free box's default
# slots (1.3x the free stream, 195 k here) cover all three: 117 k alive at Kn 0.5 + M 10
KNPULLBACK = "--substeps 4 --partners 8"
SPHERE_ARTIFACT = "ensemble_mesh_sphere_levelset0.1_gz0.05_radius_f2_H700_M300_W200_virt1x2st_dev"
MODE2_ARTIFACT = (
    "ensemble_mesh_circle_levelset0.0625_gz0.05_grid2_mode2_f1_H700_M300_W200_virt1x2st_dev"
)


def seed_jobs(base: str, artifact: str, n_seeds: int) -> dict[str, str]:
    """One command per seed writing ``<artifact>_seed{i}.json`` (merged by ``collect``)."""
    return {
        f"seed{i}": (
            f"{base} --seed-start {SEED_START + i} --seeds 1 "
            f"--output {RESULTS}/{artifact}_seed{i}.json"
        )
        for i in range(n_seeds)
    }


# optimisation groups: name -> (tag label, resolution settings)
OPT_GROUPS = {"opt": ("fine", FINE), "optc": ("coarse", COARSE)}


def _cellgate_jobs(gpus_per_node: int) -> dict[str, str]:
    del gpus_per_node
    names = [f"sq_s1_seed{i}" for i in range(16)] + [f"sq_s0.6_seed{i}" for i in range(8)]
    for shape in OPTC_SHAPES:
        names += [f"optc_{shape}_kn_b{j}" for j in range(4)]
        names += [f"optc_{shape}_fm_b{j}" for j in range(2)]
    return {name: f"bash scratch/cellfit/cluster_job.sh {name}" for name in names}


def _gapgate_jobs(gpus_per_node: int) -> dict[str, str]:
    del gpus_per_node
    names = [f"sqh_t{t}_seed{i}" for t in (1000, 2000) for i in range(16)]
    names += [f"circh_t{t}_b{j}" for t in (1000, 2000) for j in range(4)]
    names += [f"cub_{shape}_b{j}" for shape in OPTC_SHAPES[1:] for j in range(2)]
    names += [f"fmgz_{shape}_b{j}" for shape in OPTC_SHAPES for j in range(2)]
    names += [f"fmsm_{shape}_b{j}" for shape in ("a50", "a100", "a200", "g200") for j in range(2)]
    names += [f"sqfm_s{s}_b{j}" for s in ("0.35", "1") for j in range(2)]
    return {name: f"bash scratch/gapgate/cluster_job.sh {name}" for name in names}


# thingate: the squash points (labels of cluster_job.sh's s_value) and optm iterates
THIN_S = ("0.193", "0.25", "0.3", "0.45")
THIN_ITERATES = (30, 58, 109, 156, 232, 246)


def _thingate_jobs(gpus_per_node: int) -> dict[str, str]:
    del gpus_per_node
    names = [f"sq_M10_s{s}_b{j}" for s in THIN_S for j in range(2)]
    names += [f"sq_M10off_s0.193_b{j}" for j in range(2)]
    names += [f"sqg2_M10_s{s}_b{j}" for s in ("0.193", "0.3") for j in range(2)]
    names += [f"sq_M032_s{s}_b{j}" for s in ("0.193", "0.3") for j in range(2)]
    names += ["sq_FM10_s0.193_b0"]
    names += [f"om_it{k}_b0" for k in THIN_ITERATES] + ["omg2_it156_b0"]
    return {name: f"bash scratch/thingate/cluster_job.sh {name}" for name in names}


def _guardgate_jobs(gpus_per_node: int) -> dict[str, str]:
    del gpus_per_node
    names = [f"sq_M032_s0.6_b{j}" for j in range(4)]
    names += [f"sq_M10_s{s}_b{j}" for s in THIN_S for j in range(2)]
    names += [f"om_it{k}_b0" for k in (30, 58, 156, 246)] + ["unit"]
    return {name: f"bash scratch/guardgate/cluster_job.sh {name}" for name in names}


def _shapeprof_jobs(gpus_per_node: int) -> dict[str, str]:
    del gpus_per_node
    return {f"optm_it{k}": f"bash scratch/shapeprof/cluster_job.sh optm_it{k}" for k in (30, 156)}


def _speedprof_jobs(gpus_per_node: int) -> dict[str, str]:
    del gpus_per_node
    return {
        job: f"bash scratch/shapefast/cluster_job.sh {job}" for job in ("m10_it156", "kn05_it200")
    }


# boxcheck (2026-10-24): the box's position gradient -- the adjoint's dJ/dx along a rigid
# translation and a paired 1 m finite difference, six seeds, for the circle and the free M 10
# Adam body at iterate 100 -- in circle_long and in circle_xlong (inflow 15 m further up; a
# 1.5x warm-up for the 1.5x box)
BOXES = {"long": LONG, "xlong": "--case circle_xlong --fnum-scale 2 --warmup 4200"}


def _boxcheck_jobs(gpus_per_node: int) -> dict[str, str]:
    del gpus_per_node
    regimes = {"kn0.5": f"{KN05} {KNPULLBACK}",
               "kn0.5_M10": f"--knudsen 0.5 --mach 10 --dt 2.5e-5 --a-max 256 {KNPULLBACK}"}  # fmt: skip
    return {
        f"{regime}_{box}": (
            f"python3 scratch/boxcheck/translation.py --name {regime}_{box} {COARSE} {flags} "
            f"--step 0.05 --smooth 8 --band 1.0 --keepout {KEEPOUT} --area-mode tangent "
            f"--boundary free --seeds-per-iterate 1 {extra}"
        )
        for regime, extra in regimes.items()
        for box, flags in BOXES.items()
    }


# opt3dcheck (2026-10-24): the first 3-D optimiser runs -- the sphere case (r = 1 m in the
# +-2 m box, 0.2 m cells), Adam at fixed volume, one seed, 5 iterates: every step of the loop
# (volume / centroid integrals, reinitialisation, smoothing, warm start) in 3-D.  The lengths
# scaled from the circle's (r = 3 m, 0.5 m cells): step 0.02, smoothing 3 knots of 0.1 m,
# band 0.4 m, face margin 0.3 m.  The forward needs ~10 GB (the laptop's 4 GB ran out)
SPHERE3D = (
    "--case sphere --levelset-h 0.1 --levelset-samples 16 --grazing 0.05 --step 0.02 --smooth 3 "
    "--band 0.4 --keepout 0.3 --area-mode tangent --boundary free --seeds-per-iterate 1 "
    "--workers 1 --restart-steps 300"
)


def _opt3dcheck_jobs(gpus_per_node: int) -> dict[str, str]:
    del gpus_per_node
    regimes = {"kn0.5": f"{KN05} {KNPULLBACK}",
               "kn0.5_M10": f"--knudsen 0.5 --mach 10 --dt 2.5e-5 --a-max 256 {KNPULLBACK}"}  # fmt: skip
    return {
        f"sphere_{regime}": (
            f"python3 -m examples.mesh.optimize --tag sphere_{regime}_S1_ox_adam_check "
            f"{SPHERE3D} --optimizer ox_adam --iterations 5 {extra}"
        )
        for regime, extra in regimes.items()
    }


# adjbias (2026-10-24): which shape-gradient term carries the translation bias (boxcheck: the
# adjoint's dJ/dx +7.3 % of J per metre at Kn 0.5 against a zero FD).  Every term of
# shape_terms projected on the spline's rigid x-translation, 16 seeds per rung
# (scratch/adjbias/terms.py): free-molecular (true 0) at three grazing floors of the hit-time
# derivative; Kn 0.5 at the production floor, at 0.001 and with one-sided wall fits
# (--backward-normals); the production long box at Kn 0.5 and M 10
ADJBIAS_CIRCLE = "--case circle --levelset-h 0.0625 --levelset-samples 16 --boundary free"
ADJBIAS_LONG = (
    "--case circle_long --fnum-scale 2 --warmup 2800 --levelset-h 0.0625 --levelset-samples 16 "
    "--grazing 0.05 --boundary free"
)
ADJBIAS = {
    "fm": f"{ADJBIAS_CIRCLE} --grazing 0.05",
    "fm_gz0.01": f"{ADJBIAS_CIRCLE} --grazing 0.01",
    "fm_gz0.001": f"{ADJBIAS_CIRCLE} --grazing 0.001",
    "kn05": f"{ADJBIAS_CIRCLE} --grazing 0.05 --knudsen 0.5 --a-max 64",
    "kn05_gz0.001": f"{ADJBIAS_CIRCLE} --grazing 0.001 --knudsen 0.5 --a-max 64",
    "kn05_bn": f"{ADJBIAS_CIRCLE} --grazing 0.05 --knudsen 0.5 --a-max 64 --backward-normals true",
    "long_kn05": f"{ADJBIAS_LONG} --knudsen 0.5 --a-max 64",
    "long_m10": f"{ADJBIAS_LONG} --knudsen 0.5 --mach 10 --dt 2.5e-5 --a-max 256",
}


def _adjbias_jobs(gpus_per_node: int) -> dict[str, str]:
    del gpus_per_node
    return {
        name: f"python3 scratch/adjbias/terms.py {name} --seeds 16 {flags}"
        for name, flags in ADJBIAS.items()
    }


def _squashkn2_jobs(gpus_per_node: int) -> dict[str, str]:
    del gpus_per_node
    out = {}
    for label, extra, seeds in (("kn0.5", KN05, (4, 8)), ("kn0.5_M10", KN05_M10, (2, 4))):
        for s in seeds:
            out[f"{label}_S{s}"] = (
                f"python3 -m examples.mesh.optimize_squash --seeds-per-iterate {s} "
                f"--iterations {MAX_ITERATIONS} {extra}"
            )
    return out


def _keepout_jobs(gpus_per_node: int, collisional: bool) -> dict[str, str]:
    """``optckn`` (``collisional``) / ``optc2``: the circle's optimisations in the long box
    (:data:`LONG`) with the face margin and the area in the differentiated objective
    -- since 2026-10-14 ``--area-mode tangent``: the step projected onto the constant-area set
    (a Lagrange multiplier per iterate), the shift restoring only the second-order remainder
    (0.5 mm against the projection runs' 1-2 cm per Adam iterate; the augmented Lagrangian of
    the ``_al`` runs wound up in ``optab``), tags ``circle_long_[kn0.5_]S{S}_{opt}_tan`` (the
    ``_al`` and projection runs keep theirs).  The step is smoothed over 8 knots (50 cm) since
    2026-10-15 (``optab`` round 2: the smoothest shapes and the closest adjoint check, drag within
    a few % of the best), tags ``..._tan_sm8``."""
    if collisional:
        runs = [(s, "adam") for s in OPTCKN_SEEDS] + [(4, "gd")]
        regime, extra = "kn0.5_", f"{KN05} {KNPULLBACK}"
    else:
        runs, regime, extra = [(4, "adam"), (8, "adam")], "", PULLBACK
    out = {}
    for s, opt in runs:
        out[f"opt_S{s}_{opt}"] = (
            f"python3 -m examples.mesh.optimize --tag circle_long_{regime}S{s}_{opt}_tan_sm8 "
            f"--optimizer {opt} --iterations {MAX_ITERATIONS} --seeds-per-iterate {s} "
            f"--workers {min(s, gpus_per_node)} {COARSE} {LONG} --step 0.05 --smooth 8 "
            f"--band 1.0 --keepout {KEEPOUT} --area-mode tangent {extra}"
        )
    return out


def _optm_jobs(gpus_per_node: int) -> dict[str, str]:
    """``optm``: one S = 4 Adam run per regime on one GPU (module doc)."""
    del gpus_per_node
    m10 = "--mach 10 --dt 2.5e-5"
    regimes = {
        "kn0.5": f"{KN05} {KNPULLBACK}",
        "M10": f"{m10} {PULLBACK}",
        "kn0.5_M10": f"--knudsen 0.5 {m10} --a-max 256 {KNPULLBACK}",
    }
    return {
        name: (
            f"python3 -m examples.mesh.optimize --tag circle_long_{name}_S4_adam_tan_sm8_free "
            f"--optimizer adam --iterations {MAX_ITERATIONS} --seeds-per-iterate 4 --workers 1 "
            f"{COARSE} {LONG} --step 0.05 --smooth 8 --band 1.0 --keepout {KEEPOUT} "
            f"--area-mode tangent --boundary free {extra}"
        )
        for name, extra in regimes.items()
    }


# optrules (2026-10-19): the step rules compared on the optm free-space problems, every arm with
# the guarded cubic backward (CELL_BASIS since 2026-10-19) -- arm -> optimiser flags
OPTRULES = {
    "adam": "--optimizer adam",
    "tr": "--optimizer tr",
    "lbfgs": "--optimizer lbfgs",
    "mma": "--optimizer mma",
}
OPTRULES_REGIMES = ("kn0.5", "kn0.5_M10")


def _optrules_tagged(name, arm, flags, seeds=4, suffix="", extra=""):
    """One optrules-family job: the optm problem ``name`` (``OPTRULES_REGIMES``), arm tag
    ``arm``, optimiser ``flags``."""
    m10 = "--mach 10 --dt 2.5e-5"
    regimes = {"kn0.5": f"{KN05} {KNPULLBACK}",
               "kn0.5_M10": f"--knudsen 0.5 {m10} --a-max 256 {KNPULLBACK}"}  # fmt: skip
    return (
        f"python3 -m examples.mesh.optimize --tag circle_long_{name}_S{seeds}_{arm}"
        f"_tan_sm8_free_g32{suffix} --seeds-per-iterate {seeds} --workers 1 "
        f"{COARSE} {LONG} --step 0.05 --smooth 8 --band 1.0 --keepout {KEEPOUT} "
        f"--area-mode tangent --boundary free {flags} {regimes[name]} {extra}"
    ).strip()


def _optrules_jobs(gpus_per_node: int) -> dict[str, str]:
    """``optrules``: each arm of :data:`OPTRULES` on the optm Kn 0.5 and Kn 0.5 + M 10
    problems (one H200 each, S = 4, the same seeds and settings as ``optm``)."""
    del gpus_per_node
    return {
        f"{name}_{arm}": _optrules_tagged(name, arm, flags, extra=f"--iterations {MAX_ITERATIONS}")
        for name in OPTRULES_REGIMES
        for arm, flags in OPTRULES.items()
    }


# the warm start of every later optrules group (2026-10-19 relaxation / CRN checks)
WARM = "--restart-steps 500"


def _optrules2_jobs(gpus_per_node: int) -> dict[str, str]:
    """``optrules2``: the reworked trust region (2026-10-20), S = 4, warm-started."""
    del gpus_per_node
    return {
        f"{name}_tr2": _optrules_tagged(
            name,
            "tr2",
            "--optimizer tr",
            suffix="_w500",
            extra=f"--iterations {MAX_ITERATIONS} {WARM}",
        )
        for name in OPTRULES_REGIMES
    }


# one seed per iterate: 4x the iterates of the S = 4 runs for the same seed evaluations
ONE_SEED_ITERATIONS = 4 * MAX_ITERATIONS


def _optrules1seed_jobs(gpus_per_node: int) -> dict[str, str]:
    """``optrules1seed``: every rule but the perimeter one with one seed per iterate."""
    del gpus_per_node
    return {
        f"{name}_S1_{arm}": _optrules_tagged(
            name,
            arm,
            flags,
            seeds=1,
            suffix="_w500",
            extra=f"--iterations {ONE_SEED_ITERATIONS} {WARM}",
        )
        for name in OPTRULES_REGIMES
        for arm, flags in OPTRULES.items()
    }


# optax1seed (2026-10-21): the optax step rules of examples/mesh/optax_rules.py with one seed per
# iterate, warm-started -- ox_adam (the shipped Adam as an optax chain: the control),
# ox_ademamix (slow+fast momentum), ox_adamcos (cosine schedule on the step, 2 -> 1/4 x step
# over 300 iterates), ox_sfadam (schedule-free Adam), ox_prodigy (learning-rate free, capped at
# 4 x step).  400 iterates: one-seed Adam's best came at ~150 (M 10) / ~250 (deck).
OPTAX_ARMS = ("ox_adam", "ox_ademamix", "ox_adamcos", "ox_sfadam", "ox_prodigy")
OPTAX_ITERATIONS = 400


def _optax1seed_jobs(gpus_per_node: int) -> dict[str, str]:
    """``optax1seed``: every arm of :data:`OPTAX_ARMS`, one seed per iterate, warm-started."""
    del gpus_per_node
    return {
        f"{name}_S1_{arm}": _optrules_tagged(
            name,
            arm,
            f"--optimizer {arm}",
            seeds=1,
            suffix="_w500",
            extra=f"--iterations {OPTAX_ITERATIONS} {WARM}",
        )
        for name in OPTRULES_REGIMES
        for arm in OPTAX_ARMS
    }


# optrules3 (2026-10-24): the one-seed L-BFGS and MMA with their step-length controls replaced
# (examples/mesh/optimizers.py): lbfgs_fs (L-BFGS direction, fixed largest step, no trust
# region) and mma_s (MMA on the averaged gradient, smoothed step); optrules1seed's box and
# settings, so they compare with its lbfgs / mma directly
OPTRULES3 = ("lbfgs_fs", "mma_s")


def _optrules3_jobs(gpus_per_node: int) -> dict[str, str]:
    """``optrules3``: each rule of :data:`OPTRULES3`, one seed per iterate, warm-started."""
    del gpus_per_node
    return {
        f"{name}_S1_{arm}": _optrules_tagged(
            name,
            arm,
            f"--optimizer {arm}",
            seeds=1,
            suffix="_w500",
            extra=f"--iterations {OPTAX_ITERATIONS} {WARM}",
        )
        for name in OPTRULES_REGIMES
        for arm in OPTRULES3
    }


# adjfix (2026-10-24): the translation-bias fix.  adjbias split the circle's bias (Kn 0.5
# +5.9 % J/m, FD ~0) into the grazing floor of the hit-time derivative (+1.2-1.3 % J/m, gone at
# 0.01 in free-molecular flow) and the pooled wall fits of the backward (-5.7 % J/m of the
# collision-order / cut-volume terms with --backward-normals).  FIX holds both.  Checks:
# boxcheck's adjoint-vs-FD on the circle and the free M 10 Adam body (8 seeds; M 10 is untested
# with either change: its body's bias is -7 % J/m against a zero FD) and ox_adam with the fix
# and no centroid constraint, 400 iterates (optax1seed's free ox_adam drifted 2.4 m upstream at
# Kn 0.5 and 11 m downstream at M 10); tags *_S1_ox_adam_fix_*_w500
FIX = "--backward-normals true --grazing 0.01"
ADJFIX_CHECKS = {
    "kn0.5_fix": FIX,
    "kn0.5_M10_bn": "--backward-normals true",
    "kn0.5_M10_gz": "--grazing 0.01",
    "kn0.5_M10_fix": FIX,
}


def _adjfix_jobs(gpus_per_node: int) -> dict[str, str]:
    """``adjfix``: the translation checks of :data:`ADJFIX_CHECKS` and the free ox_adam runs
    with :data:`FIX` on both optrules problems."""
    del gpus_per_node
    regimes = {"kn0.5": f"{KN05} {KNPULLBACK}",
               "kn0.5_M10": f"--knudsen 0.5 --mach 10 --dt 2.5e-5 --a-max 256 {KNPULLBACK}"}  # fmt: skip
    jobs = {
        f"check_{name}": (
            f"python3 scratch/boxcheck/translation.py --name adjfix_{name} --seeds 8 {COARSE} "
            f"{LONG} --step 0.05 --smooth 8 --band 1.0 --keepout {KEEPOUT} --area-mode tangent "
            f"--boundary free --seeds-per-iterate 1 {regimes[name.rsplit('_', 1)[0]]} {flags}"
        )
        for name, flags in ADJFIX_CHECKS.items()
    }
    for name in OPTRULES_REGIMES:
        jobs[f"{name}_S1_ox_adam_fix"] = _optrules_tagged(
            name,
            "ox_adam_fix",
            f"--optimizer ox_adam {FIX}",
            seeds=1,
            suffix="_w500",
            extra=f"--iterations {OPTAX_ITERATIONS} {WARM}",
        )
    return jobs


# adjm10 (2026-10-25): the M 10 translation bias that adjfix left (circle -1.1, free M 10 Adam
# body -8 to -9 % J/m against a ~zero FD, the same with each FIX change alone and both).  All at
# Kn 0.5 M 10 with FIX.  terms_*: adjbias's term breakdown (16 seeds) of the body, the circle and
# the body free-molecular at M 10 (true value 0: is it the collision adjoint?).  check_*:
# boxcheck's adjoint-vs-FD (8 seeds) with the backward fit's neighbour count 16 / 64 (g32 is
# the baseline), 2x / 4x the particles per cell (finite-sample fit bias shrinks with them; A_max
# scaled with the NTC attempts, ~ppc) and the DSMC cells halved to 0.25 m (behind the M 10 bow
# shock the gas is ~4x denser, lambda ~0.75 m against the free stream's 3 m: the 0.5 m cells
# are ~lambda / 1.5 there) at the same particles per cell (4x the particles) and at half of
# them (2x).  M 10 in circle_long fits a 24 GB rtx slice (boxcheck_kn0.5_M10_long, adjbias
# long_m10; circle_xlong's 1.5x box ran out of it), so the baseline-size jobs go to rtx (no
# H200 cap); the ppc / fine checks (2-4x the particles) take whole H200s, the cap's 4 at once:
#   GPU_KIND=rtx ONLY='^(terms|check_g)' bash examples/mesh/cluster/submit_curc.sh adjm10
#   GPU_KIND=h200_full ONLY='ppc|fine' bash examples/mesh/cluster/submit_curc.sh adjm10
ADJM10_REGIME = f"--knudsen 0.5 --mach 10 --dt 2.5e-5 --a-max 256 {FIX}"
ADJM10_TERMS = {
    "body": f"--shape m10_adam_it100 {ADJM10_REGIME}",
    "circle": ADJM10_REGIME,
    "body_fm": f"--shape m10_adam_it100 --mach 10 --dt 2.5e-5 {FIX}",
    # the per-seed term vectors as well (8 seeds): on the body the two translation
    # discretisations (terms.py +9.4, optimize.translation_modes -9.0 % J/m) disagree in sign,
    # so the projections and where they sit on the surface are computed offline.  rtx:
    #   GPU_KIND=rtx ONLY='_vec' bash examples/mesh/cluster/submit_curc.sh adjm10
    "body_vec": f"--shape m10_adam_it100 {ADJM10_REGIME} --save-vectors 1",
    "body_fm_vec": f"--shape m10_adam_it100 --mach 10 --dt 2.5e-5 {FIX} --save-vectors 1",
}
ADJM10_CHECKS = {
    "g16": "--neighbors 16",
    "g64": "--neighbors 64",
    "ppc2": "--fnum-scale 1 --a-max 512",
    "ppc4": "--fnum-scale 0.5 --a-max 1024",
    "fine": "--grid-scale 2",
    "fine_half": "--grid-scale 2 --fnum-scale 4",
}


def _adjm10_jobs(gpus_per_node: int) -> dict[str, str]:
    """``adjm10``: the term breakdowns of :data:`ADJM10_TERMS` and the translation checks of
    :data:`ADJM10_CHECKS`, Kn 0.5 M 10 with :data:`FIX`."""
    del gpus_per_node
    jobs = {
        f"terms_{name}": (
            f"python3 scratch/adjbias/terms.py adjm10_{name} --seeds {8 if 'vec' in name else 16} "
            f"{ADJBIAS_LONG} {flags}"
        )
        for name, flags in ADJM10_TERMS.items()
    }
    for name, flags in ADJM10_CHECKS.items():
        jobs[f"check_{name}"] = (
            f"python3 scratch/boxcheck/translation.py --name adjm10_{name} --seeds 8 {COARSE} "
            f"{LONG} --step 0.05 --smooth 8 --band 1.0 --keepout {KEEPOUT} --area-mode tangent "
            f"--boundary free --seeds-per-iterate 1 {ADJM10_REGIME} {KNPULLBACK} {flags}"
        )
    return jobs


# stepfd (2026-10-25): is the descent the free M 10 runs keep predicting real?  In the slide
# phase (ox_adam_fix iterates 150-250, mma_s alike) the adjoint predicts ~-0.01 per step (~4 %
# of J; independent seeds agree) and ~0 is realised, the predicted decrease at the needle nose
# in pathwise_tally (scratch/adjbias/drift_pipeline.py).  scratch/adjbias/stepfd.py: along the
# step a run took at iterate K (and its nose part, the rest, a rigid 0.04 m shift), the
# adjoint g . delta against paired central differences at eps 0.5 and 1, 8 seeds; it20 is the
# control (realised ~0.9 of predicted there), mma_it150 the other constraint treatment (MMA's
# own area subproblem, no tangent projection), adam_it150_h2 the same shapes on knots h / 2.
# One gradient + 14 forwards per seed: ~5 h on an rtx slice (h2 longer).
#   GPU_KIND=rtx bash examples/mesh/cluster/submit_curc.sh stepfd
STEPFD_ADAM = "circle_long_kn0.5_M10_S1_ox_adam_fix_tan_sm8_free_g32_w500"
STEPFD_MMA = "circle_long_kn0.5_M10_S1_mma_s_tan_sm8_free_g32_w500"
STEPFD = {
    "adam_it20": f"--run {STEPFD_ADAM} --iterate 20",
    "adam_it100": f"--run {STEPFD_ADAM} --iterate 100",
    "adam_it150": f"--run {STEPFD_ADAM} --iterate 150",
    "adam_it200": f"--run {STEPFD_ADAM} --iterate 200",
    "mma_it150": f"--run {STEPFD_MMA} --iterate 150",
    "adam_it150_h2": f"--run {STEPFD_ADAM} --iterate 150 --refine 2",
}


def _stepfd_jobs(gpus_per_node: int) -> dict[str, str]:
    """``stepfd``: :data:`STEPFD`, the adjoint against finite differences along the free M 10
    runs' own steps (``scratch/adjbias/stepfd.py``; the runs' coefficients on the cluster)."""
    del gpus_per_node
    return {
        name: f"python3 scratch/adjbias/stepfd.py {name} --seeds 8 {flags}"
        for name, flags in STEPFD.items()
    }


# optrc (2026-10-25): stepfd found the adjoint right along the free M 10 runs' own steps
# (FD within ~10 % at iterates 20 / 100 / 150, on knots h / 2 too, mma_s alike): the slide is
# not a gradient error -- the drag barely sees a rigid translation, and the smoothed / Adam
# steps carry one.  --recentre 1 moves the body back by whole knots whenever its centroid
# drifts half a knot (exact: the coefficients and the optimiser state shift by an integer;
# optct's projections and resamples roughened the bodies).  M 10 only (the Kn 0.5 fix run
# stayed put); rtx, ~14 h at 400 iterates (ox_adam_fix: 123 s/iterate):
#   GPU_KIND=rtx bash examples/mesh/cluster/submit_curc.sh optrc
OPTRC = ("ox_adam", "mma_s")
# 2026-10-26: stop when the 25-iterate running mean of J has not improved by 2 % in 100
# iterates (optimize.patience_converged; offline on the drag histories the M 10 runs stopped
# 0-3 % short of their eventual best and the slow mma_s tails kept going).  optrc, ldopt, sph3d
CONVERGE = "--converge-patience 100"


def _optrc_jobs(gpus_per_node: int) -> dict[str, str]:
    """``optrc``: :data:`OPTRC` on the M 10 optrules problem with ``--recentre 1`` and
    :data:`FIX`, one seed, warm-started, 400 iterates; tags ``*_S1_<arm>_rc_*_w500``."""
    del gpus_per_node
    return {
        f"kn0.5_M10_S1_{arm}_rc": _optrules_tagged(
            "kn0.5_M10",
            f"{arm}_rc",
            f"--optimizer {arm} --recentre 1 {CONVERGE} {FIX}",
            seeds=1,
            suffix="_w500",
            extra=f"--iterations {OPTAX_ITERATIONS} {WARM}",
        )
        for arm in OPTRC
    }


# ldopt (2026-10-25): maximum lift-to-drag, J = -L / D (ensemble_mesh.objective_weights: one
# backward per seed, the per-hit momentum weighted by (L / D^2, -1 / D) at the seed's forces),
# no area constraint and no centroid constraint (the box keep-out stays), from the circle
# (L = 0: the first steps are the lift gradient's), both optrules problems, the two rules that
# did best on drag; rtx, ~8-14 h at 400 iterates:
#   GPU_KIND=rtx bash examples/mesh/cluster/submit_curc.sh ldopt
# 2026-10-26: the unconstrained runs (_ld_, _ldrc_) are degenerate -- all four thinned the circle
# into a ~35 deg plate (peak L / D 1.46 Kn 0.5, 1.25 M 10 by iterate 60-130), then on below the
# 0.5 m cells, where the predicted decrease stopped holding, until the body vanished (three
# crashed in reinitialise).  Now with an area floor at half the circle's (--area-mode floor:
# a ~1.8 m thick plate at an 8 m chord, >= 3 cells), tags _ldfl_floor_
LDOPT = ("ox_adam", "mma_s")
LDOPT_FLOOR = 0.5


def _ldopt_jobs(gpus_per_node: int) -> dict[str, str]:
    """``ldopt``: :data:`LDOPT` maximising L / D with the area at least :data:`LDOPT_FLOOR` of
    the circle's, ``--recentre 1`` and :data:`FIX` on both optrules problems, one seed,
    warm-started, 400 iterates; tags ``circle_long_<problem>_S1_<arm>_ldfl_floor_sm8_free_g32_w500``
    (the unconstrained runs: ``_ld_none_`` without recentring, ``_ldrc_none_`` with it)."""
    del gpus_per_node
    jobs = {}
    for name in OPTRULES_REGIMES:
        for arm in LDOPT:
            cmd = _optrules_tagged(
                name,
                f"{arm}_ldfl",
                # --recentre 1 (2026-10-26): the first runs' M 10 Adam body slid 0.05 m /
                # iterate in -y towards the bottom face's keep-out; restarted from the circle
                # under new tags (_ldrc_), the first runs (_ld_) kept for comparison
                f"--optimizer {arm} --objective lift_drag --area-mode floor "
                f"--area-floor {LDOPT_FLOOR} --recentre 1 {CONVERGE} {FIX}",
                seeds=1,
                suffix="_w500",
                extra=f"--iterations {OPTAX_ITERATIONS} {WARM}",
            )
            # the tag names the area mode: "_tan_" is _optrules_tagged's, overridden here
            jobs[f"{name}_S1_{arm}_ldfl"] = cmd.replace("_ldfl_tan_sm8", "_ldfl_floor_sm8", 1)
    return jobs


# sph3d (2026-10-25): the 3-D counterparts of the 2-D problems on both optrules conditions
# (Kn 0.5 at 100 m/s and at M 10), ox_adam, one seed, the adjfix flags: minimum drag at fixed
# volume (tangent, free, no centroid constraint) and maximum L / D (lift along y) with neither
# volume nor centroid constraint.  sphere_long: circle_long's proportions for the r = 1 m
# sphere (opt3dcheck's lengths: knots 0.1 m, step 0.02, smoothing 3 knots, band 0.4 m, face
# margin 0.3 m); the warm-up scaled from circle_long's 2800 steps for 3x the box length.
# --fnum-scale 1: 16 particles per free-stream cell (~320 k particles, ~420 k slots).
# opt3dcheck ran at the default 4.0: 4 per cell, 5 % of the gas cells at 1 -- far below the
# usual DSMC 10-20 and the 2-D production's ~125 (its frames: dx = lambda / 5 in the free
# stream, median lambda / 3.6 in the M 10 shock layer at 0.2 m cells).  One full H200 each
# (opt3dcheck: ~10 GB at 42 k slots; the 2-D M 10 runs fit 195 k slots in a 24 GB slice;
# expect tens of GB -- the job's GPU monitor log has the peak), resubmit to resume:
#   GPU_KIND=h200_full bash examples/mesh/cluster/submit_curc.sh sph3d
# or (2026-10-26) on 3g.71gb slices, ~40-50 GB estimated (the 2-D M 10 runs: ~18 GB at ~195 k
# slots; an out-of-memory job fails in its first iterate and resumes on a full card):
#   GPU_KIND=h200_71 bash examples/mesh/cluster/submit_curc.sh sph3d
# --recentre 1 (2026-10-26) as the 2-D ldopt / optrc runs: the body held within half a knot of
# its start by exact whole-knot shifts.  --a-max 16: a low first guess the forward grows on a
# clipped step (ensemble_mesh._run_grown); the record's collision pairs are (A_max, cells) per
# step, so 2-D's 256 over 20 k cells would hold ~10x what 16 particles per cell need
SPHERE_LONG = (
    "--case sphere_long --fnum-scale 1 --levelset-h 0.1 --levelset-samples 16 --warmup 1000 "
    "--step 0.02 --smooth 3 --band 0.4 --keepout 0.3 --boundary free --restart-steps 300 "
    "--iterations 400 --seeds-per-iterate 1 --workers 1 --recentre 1"
)
SPH3D = {
    "drag": "--area-mode tangent",
    # 2026-10-26: with a volume floor at half the sphere's, as the 2-D ldopt (the unconstrained
    # 2-D L / D runs thinned the body below the cells until it vanished)
    "ldfl": f"--objective lift_drag --area-mode floor --area-floor {LDOPT_FLOOR}",
}


def _sph3d_jobs(gpus_per_node: int) -> dict[str, str]:
    """``sph3d``: :data:`SPH3D` on ``sphere_long`` at both optrules conditions, ox_adam,
    :data:`FIX`; tags ``sphere_long_<condition>_S1_ox_adam_<objective>_free_w300``."""
    del gpus_per_node
    regimes = {"kn0.5": f"--knudsen 0.5 {KNPULLBACK}",
               "kn0.5_M10": f"--knudsen 0.5 --mach 10 --dt 2.5e-5 {KNPULLBACK}"}  # fmt: skip
    return {
        f"sphere_long_{regime}_{name}": (
            f"python3 -m examples.mesh.optimize "
            f"--tag sphere_long_{regime}_S1_ox_adam_{name}_free_w300 {SPHERE_LONG} "
            f"--optimizer ox_adam {flags} {extra} --a-max 16 {CONVERGE} {FIX}"
        )
        for regime, extra in regimes.items()
        for name, flags in SPH3D.items()
    }


# lsfilter (2026-10-26): the filtered level-set formulation (van Dijk et al. 2013, Kreissl &
# Maute 2012): the design a field s on the box, phi = F s with F the Helmholtz PDE filter of
# radius smooth * h / sqrt 2 = 0.35 m (the band Gaussian's second moment), the rule on dJ/ds =
# F^T dJ/dphi, its step mapped by F (--filter helmholtz, jax_dsmc.mesh.helmholtz) -- in place of
# the Gaussian on the coefficient band before and after the rule.  Every run without
# reinitialisation (--reinit-every 0, see lsfilter3d: reinitialising every iterate grows
# knot-lattice lobes on features thinner than ~2-3 knots and breaks them up -- the 2-D M 10
# needle nose's tip droplets), band (mark bd0) and Helmholtz (hz0) alike, so each pair differs
# in the filter only; the earlier band runs (optrc, ldopt _ldfl_: reinitialised) are the old
# method's reference.  Every run recentred, stopped by CONVERGE, its gradients saved (the
# offline step replay, scratch/lsfilter):
#   * drag at fixed area, M 10: ox_adam, mma_s, and ox_gd (Helmholtz only: steepest descent,
#     the Hilbertian velocity method itself);
#   * drag at fixed area, Kn 0.5: ox_adam, mma_s;
#   * L / D with the area floor, both conditions: ox_adam, mma_s (marks ldbd0 / ldhz0);
#   * sharp-edged starts (--initial, the edge cases of sharp bodies): M 10 drag at fixed area
#     from a diamond (t / c 0.3, the circle's area) and M 10 L / D with the floor from a thin
#     plate (8 m chord, 1 m thick, 10 deg; the floor at its own area), ox_adam (marks dm / pl
#     + bd0 / hz0);
#   * one job of deterministic canonical tests (--surrogate newtonian: the Newtonian drag at
#     fixed area -- a sharp-nosed optimum -- and the Newtonian L / D with the floor) for both
#     filters and both rules, the sharp starts, and the Helmholtz drag with a perimeter
#     penalty, serially, 300 iterates each, the step on a cosine to step / 16 at 300 (with a
#     fixed step band Adam cycled between J 1.14 and 1.53 around the drag optimum locally);
#     the job is the group's last, so the DSMC jobs queue first.
# rtx (the DSMC ones are optrc / ldopt-sized; the Newtonian job is light):
#   GPU_KIND=rtx bash examples/mesh/cluster/submit_curc.sh lsfilter
HELM = "--filter helmholtz --save-gradient 1"
NO_REINIT = "--reinit-every 0"
LSFILTER_FILTERS = {"bd0": "--save-gradient 1", "hz0": HELM}
# regime -> {filter mark: arms}
LSFILTER_DRAG = {
    "kn0.5_M10": {"bd0": ("ox_adam", "mma_s"), "hz0": ("ox_adam", "mma_s", "ox_gd")},
    "kn0.5": {"bd0": ("ox_adam", "mma_s"), "hz0": ("ox_adam", "mma_s")},
}
LSFILTER_LD = ("ox_adam", "mma_s")
# iterate caps (2026-10-26, from every >= 200-iterate history: drag runs had 95-99 % of their
# eventual gain by 150; the Kn 0.5 MMA drag runs 38-56 % and the L / D floor runs 84 %, so
# 250 there).  A run resumes from its history: resubmit with a larger --iterations to extend.
# 2026-10-27: 250 for all -- the Helmholtz Adam drag runs were still descending at 150 (M 10
# J/J0 0.058, Kn 0.5 0.178), unlike the band runs the 150 came from; the patience test stops
# a converged run (resubmit: ONLY='^kn0.5(_M10)?_S1_ox_adam_hz0$')
LSFILTER_ITERATIONS = 250
LSFILTER_SLOW_ITERATIONS = 250
NEWTON = (
    "--surrogate newtonian --case circle_long --levelset-h 0.0625 --levelset-samples 16 "
    "--iterations 300 --step 0.05 --smooth 8 --band 1.0 --keepout 1.0 --boundary free "
    f"--recentre 1 --frames-every 0 --save-gradient 1 --step-decay 300 {NO_REINIT}"
)
NEWTON_RUNS = {
    "drag": "--area-mode tangent",
    "ld": f"--objective lift_drag --area-mode floor --area-floor {LDOPT_FLOOR}",
}
DIAMOND = "--initial diamond --initial-thickness 0.3"
PLATE = "--initial plate --initial-thickness 0.125 --initial-aoa 10"
# name -> (problem flags, start); ox_adam, band and Helmholtz
# the plate keeps at least its own area (8 m^2: 1 m = 2 cells thick at its chord; half of it
# would be one cell, below which the unconstrained L / D runs broke up)
SHARP = {
    "dm": (NEWTON_RUNS["drag"], DIAMOND),
    "pl": ("--objective lift_drag --area-mode floor --area-floor 1.0", PLATE),
}
NEWTON_PERIMETER = 0.005  # J0 x P / P0 weight of the perimeter-penalised Helmholtz run


def _lsfilter_run(regime: str, arm: str, mark: str, flags: str, floor: bool) -> str:
    """One lsfilter DSMC run: ``regime``'s optrules problem, optimiser ``arm``, tag arm
    ``<arm>_<mark>`` (``_floor_`` in place of ``_tan_`` for an area-floor problem)."""
    flt = LSFILTER_FILTERS[mark[-3:]]
    slow = floor or (regime == "kn0.5" and arm.startswith("mma"))
    iterations = LSFILTER_SLOW_ITERATIONS if slow else LSFILTER_ITERATIONS
    cmd = _optrules_tagged(
        regime, f"{arm}_{mark}",
        f"--optimizer {arm} {flags} --recentre 1 {NO_REINIT} {flt} {CONVERGE} {FIX}",
        seeds=1, suffix="_w500", extra=f"--iterations {iterations} {WARM}",
    )  # fmt: skip
    return cmd.replace(f"_{arm}_{mark}_tan_sm8", f"_{arm}_{mark}_floor_sm8", 1) if floor else cmd


def _lsfilter_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lsfilter``: the battery's DSMC runs, band and Helmholtz without reinitialisation
    (tags ``*_<arm>_<mark>_*``, marks ``bd0`` / ``hz0``, ``ld`` / ``dm`` / ``pl`` before them
    for the L / D, diamond and plate problems) and one job of the Newtonian surrogate runs
    (``lsf_newton_<problem>_<filter>_<arm>``)."""
    del gpus_per_node
    jobs = {}
    for regime, by_filter in LSFILTER_DRAG.items():
        for mark, arms in by_filter.items():
            for arm in arms:
                jobs[f"{regime}_S1_{arm}_{mark}"] = _lsfilter_run(regime, arm, mark, "", False)
    ld = f"--objective lift_drag --area-mode floor --area-floor {LDOPT_FLOOR}"
    for regime in OPTRULES_REGIMES:
        for mark in LSFILTER_FILTERS:
            for arm in LSFILTER_LD:
                jobs[f"{regime}_S1_{arm}_ld{mark}"] = _lsfilter_run(
                    regime, arm, f"ld{mark}", ld, True
                )
    for shape, (flags, start) in SHARP.items():
        for mark in LSFILTER_FILTERS:
            jobs[f"kn0.5_M10_S1_ox_adam_{shape}{mark}"] = _lsfilter_run(
                "kn0.5_M10", "ox_adam", f"{shape}{mark}", f"{flags} {start}", shape == "pl"
            )
    newton = "python3 -m examples.mesh.optimize --tag lsf_newton_"
    runs = [
        f"{newton}{problem}_{flt}_{arm} --filter {flt} --optimizer {arm} {flags} {NEWTON} "
        f"{CONVERGE}"
        for problem, flags in NEWTON_RUNS.items()
        for flt in ("band", "helmholtz")
        for arm in ("ox_adam", "mma_s", "ox_gd")
        if not (flt == "band" and arm == "ox_gd")
    ]
    runs += [
        f"{newton}{mark}_{flt}_ox_adam --filter {flt} --optimizer ox_adam {flags} {start} "
        f"{NEWTON} {CONVERGE}"
        for mark, (flags, start) in SHARP.items()
        for flt in ("band", "helmholtz")
    ]
    runs.append(
        f"{newton}drag_helmholtz_ox_adam_per --filter helmholtz --optimizer ox_adam "
        f"{NEWTON_RUNS['drag']} --perimeter {NEWTON_PERIMETER} {NEWTON} {CONVERGE}"
    )
    jobs["newton"] = " ; ".join(runs)
    return jobs


# lsfilter3d (2026-10-26): the 3-D sphere in the filtered formulation without
# reinitialisation, minimum drag at fixed volume and maximum L / D with the volume floor, both
# conditions, ox_adam: --filter helmholtz (radius smooth * h / sqrt 2 = 0.21 m) and
# --reinit-every 0.  sph3d's band runs (reinitialised every iterate) grew knot-lattice lobes and
# broke up once the nose / plate thinned below ~2-3 knots: the redistance-then-interpolate
# map is not idempotent there (scratch/sph3d/reinit_change*.py).  The deterministic 3-D
# Newtonian battery (scratch/reinit, h = 0.2, the same knot units, 150 iterates) settled it:
# reinitialised, band and Helmholtz alike broke into 5-19 pieces and lost ground (drag J/J0
# 0.27 -> 0.36 / 0.39, L / D 0.74 -> 0.56); without, one piece throughout, drag J/J0 0.076
# (band; its |grad phi| drifts to 2.9) and 0.023 (Helmholtz; |grad phi| stays 0.9-1.0),
# L / D 1.90 (band).  A doubled radius (0.85 m) neither thickened the nose nor improved J,
# so the default radius.  150 iterates (the 3-D Newtonian runs had converged by ~100; at
# 200-490 s an iterate, 8-20 h: one 24 h job) with --step-decay 150 (a cosine to step / 16:
# the fixed step leaves the single-seed runs wandering in a noise ball once converged); to
# extend, resubmit with larger --iterations and --step-decay (the run resumes).
# A 3g.71gb H200 slice each (~40-50 GB estimated, see sph3d):
#   GPU_KIND=h200_71 bash examples/mesh/cluster/submit_curc.sh lsfilter3d
# 2026-10-27: extended to 250 (the Kn 0.5 drag and L / D runs were still improving at 150;
# resumed, the cosine restarts from step x 0.38 at 150 and decays to step / 16 at 250):
#   GPU_KIND=h200_71 ONLY='^sphere_long_kn0.5_(drag|ldfl)_hz0$' bash \
#     examples/mesh/cluster/submit_curc.sh lsfilter3d
LSFILTER3D_ITERATIONS = 250
LSFILTER3D = f"{HELM} --reinit-every 0 --step-decay {LSFILTER3D_ITERATIONS}"


def _lsfilter3d_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lsfilter3d``: :func:`_sph3d_jobs`' runs with :data:`LSFILTER3D`; tags
    ``sphere_long_<condition>_S1_ox_adam_<objective>_hz0_free_w300``."""
    return {
        f"{name}_hz0": cmd.replace("_free_w300", "_hz0_free_w300", 1).replace(
            "--iterations 400", f"--iterations {LSFILTER3D_ITERATIONS}", 1
        )
        + f" {LSFILTER3D}"
        for name, cmd in _sph3d_jobs(gpus_per_node).items()
    }


# ld3dwide (2026-10-27): lsfilter3d's M 10 L / D run (L/D 2.37 @150, the best 3-D L / D) on
# sphere_wide, twice sphere_long's span (z faces at +-4 m): its body pressed the z face
# margin from iterate 46, its span box-limited.  400 iterates on a cosine step decay to 400
# (2x the cells: ~600 s an iterate, ~140 a 24 h job; resubmit to resume).  A full H200 (the
# sphere_long run fit a 3g.71gb slice; this one has twice the particles):
#   GPU_KIND=h200_full bash examples/mesh/cluster/submit_curc.sh ld3dwide
LD3DWIDE_ITERATIONS = 400


def _ld3dwide_jobs(gpus_per_node: int) -> dict[str, str]:
    """``ld3dwide``: :func:`_lsfilter3d_jobs`' M 10 L / D run on ``sphere_wide``; tag
    ``sphere_wide_kn0.5_M10_S1_ox_adam_ldfl_hz0_free_w300``."""
    cmd = _lsfilter3d_jobs(gpus_per_node)["sphere_long_kn0.5_M10_ldfl_hz0"]
    cmd = cmd.replace("sphere_long", "sphere_wide").replace(
        f"--iterations {LSFILTER3D_ITERATIONS}", f"--iterations {LD3DWIDE_ITERATIONS}", 1
    )
    return {
        "sphere_wide_kn0.5_M10_ldfl_hz0": cmd.replace(
            f"--step-decay {LSFILTER3D_ITERATIONS}", f"--step-decay {LD3DWIDE_ITERATIONS}", 1
        )
    }


# optcm (2026-10-23): the area centroid held (every Adam-type M 10 run of optax1seed /
# optrules1seed slid ~10 m downstream onto the outflow face), one seed per iterate, for the
# rules that descended fastest: Adam, Adam + cosine step, Prodigy and Prodigy with a decaying
# cap (its step sat at the 4-step cap from iterate ~25 on)
OPTCM = ("ox_adam", "ox_adamcos", "ox_prodigy", "ox_prodigycos")


def _optcm_jobs(gpus_per_node: int, mark: str = "cm", extra: str = "") -> dict[str, str]:
    """``optcm``: each rule of :data:`OPTCM` with ``--fix-centroid 1`` on both optrules
    problems, S = 1, warm-started, 400 iterates; tags ``*_ox_<rule>_<mark>_*``.  ``optct``
    (``mark="ct"``) is the same with the centroid held by rigid translations and the body
    started 2 m downstream (``extra``; 2026-10-23: the ``_cm`` runs reshaped the body for the
    constraint, the first version of it, and their M 10 noses stopped ~9.4 m from the inflow
    face; they are done: resubmitting ``optcm`` would resume them)."""
    del gpus_per_node
    return {
        f"{name}_S1_{arm}_{mark}": _optrules_tagged(
            name,
            f"{arm}_{mark}",
            f"--optimizer {arm} --fix-centroid 1{extra}",
            seeds=1,
            suffix="_w500",
            extra=f"--iterations {OPTAX_ITERATIONS} {WARM}",
        )
        for name in OPTRULES_REGIMES
        for arm in OPTCM
    }


# optab: name -> extra flags (the later flag wins over COARSE's --levelset-h)
OPTAB = {
    "ctl": "--smooth 2",
    "smooth4": "--smooth 4",
    "smooth8": "--smooth 8",
    "lsh0.125": "--smooth 2 --levelset-h 0.125",
    "lsh0.25": "--smooth 2 --levelset-h 0.25",
}


def _optab_jobs(gpus_per_node: int) -> dict[str, str]:
    del gpus_per_node
    return {
        name: (
            f"python3 -m examples.mesh.optimize --tag circle_long_S4_adam_tan_ab_{name} "
            f"--optimizer adam --iterations {MAX_ITERATIONS} --seeds-per-iterate 4 --workers 1 "
            f"{COARSE} {LONG} --step 0.05 --band 1.0 --keepout {KEEPOUT} "
            f"--area-mode tangent {PULLBACK} {extra}"
        )
        for name, extra in OPTAB.items()
    }


NEW_GROUPS = {
    "optab": _optab_jobs,
    "optm": _optm_jobs,
    "cellgate": _cellgate_jobs,
    "gapgate": _gapgate_jobs,
    "thingate": _thingate_jobs,
    "guardgate": _guardgate_jobs,
    "shapeprof": _shapeprof_jobs,
    "speedprof": _speedprof_jobs,
    "boxcheck": _boxcheck_jobs,
    "opt3dcheck": _opt3dcheck_jobs,
    "optrules3": _optrules3_jobs,
    "adjbias": _adjbias_jobs,
    "adjfix": _adjfix_jobs,
    "adjm10": _adjm10_jobs,
    "stepfd": _stepfd_jobs,
    "optrc": _optrc_jobs,
    "ldopt": _ldopt_jobs,
    "sph3d": _sph3d_jobs,
    "lsfilter": _lsfilter_jobs,
    "lsfilter3d": _lsfilter3d_jobs,
    "ld3dwide": _ld3dwide_jobs,
    "optrules": _optrules_jobs,
    "optrules2": _optrules2_jobs,
    "optrules1seed": _optrules1seed_jobs,
    "optax1seed": _optax1seed_jobs,
    "optcm": _optcm_jobs,
    "optct": functools.partial(_optcm_jobs, mark="ct", extra=" --start-shift 2.0"),
    "squashkn2": _squashkn2_jobs,
    "optckn": functools.partial(_keepout_jobs, collisional=True),
    "optc2": functools.partial(_keepout_jobs, collisional=False),
}


def jobs(group: str, gpus_per_node: int = GPUS_PER_NODE) -> dict[str, str]:
    """``{job name: command}`` of a group."""
    if group in NEW_GROUPS:
        return NEW_GROUPS[group](gpus_per_node)
    return _jobs(group, gpus_per_node)


def _jobs(group: str, gpus_per_node: int) -> dict[str, str]:
    if group == "sphere":
        base = (
            "python3 -m examples.ensemble_mesh --case sphere --body levelset --parameter radius "
            f"--fnum-scale 2 --levelset-h 0.1 --levelset-samples 8 --grazing 0.05 {PULLBACK} "
            "--fd-steps 0.05 0.02"
        )
        return seed_jobs(base, SPHERE_ARTIFACT, SPHERE_SEEDS)
    if group == "mode2":
        base = (
            f"python3 -m examples.ensemble_mesh --body levelset --parameter mode2 {FINE} "
            f"{PULLBACK} --fd-steps 0.1 0.05"
        )
        return seed_jobs(base, MODE2_ARTIFACT, MODE2_SEEDS)
    if group == "squash":
        out = {
            f"grad_s{p:g}": (
                "python3 -m examples.ensemble_mesh --case ellipse --body levelset "
                f"--parameter squash_y --squash {p} {COARSE} --value-class none {PULLBACK} "
                f"--fd-steps 0.05 0.02 --seeds {SQUASH_SEEDS}"
            )
            for p in SQUASH_POINTS
        }
        for s in SQUASH_OPT_SEEDS:
            out[f"opt_S{s}"] = (
                f"python3 -m examples.mesh.optimize_squash --seeds-per-iterate {s} "
                f"--iterations {MAX_ITERATIONS} {PULLBACK}"
            )
        return out
    if group == "squashkn":
        out = {}
        for label, extra in (("kn0.5", KN05), ("kn0.5_M10", KN05_M10)):
            for s in SQUASH_KN_SEEDS:
                out[f"{label}_S{s}"] = (
                    f"python3 -m examples.mesh.optimize_squash --seeds-per-iterate {s} "
                    f"--iterations {MAX_ITERATIONS} {extra}"
                )
        return out
    if group == "biasgate":
        names = [f"kn_seed{i}" for i in range(8)] + [f"kn_fd{j}" for j in range(4)]
        names += [f"kn_hor_seed{i}" for i in range(4)] + ["fm_hor_s1.0", "fm_hor_s0.35"]
        return {name: f"bash scratch/bias/cluster_job.sh {name}" for name in names}
    if group in OPT_GROUPS:
        label, settings = OPT_GROUPS[group]
        out = {}
        for s in SEED_COUNTS:
            for opt in OPTIMIZERS:
                out[f"opt_S{s}_{opt}"] = (
                    f"python3 -m examples.mesh.optimize --tag circle_{label}_S{s}_{opt} "
                    f"--optimizer {opt} --iterations {MAX_ITERATIONS} --seeds-per-iterate {s} "
                    f"--workers {min(s, gpus_per_node)} {settings} --step 0.05 --smooth 2 "
                    f"--band 1.0 --area-mode projection {PULLBACK}"
                )
        return out
    raise ValueError(group)


def gpus(group: str, gpus_per_node: int = GPUS_PER_NODE) -> dict[str, int]:
    """GPUs per job: one per seed job; ``min(S, gpus_per_node)`` per optimisation."""
    if group in ("optckn", "optc2"):
        return {
            name: min(int(name.split("_S")[1].split("_")[0]), gpus_per_node)
            for name in jobs(group, gpus_per_node)
        }
    if group not in OPT_GROUPS:
        return dict.fromkeys(jobs(group), 1)
    return {f"opt_S{s}_{opt}": min(s, gpus_per_node) for s in SEED_COUNTS for opt in OPTIMIZERS}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--group",
        choices=("sphere", "mode2", "squash", "squashkn", "biasgate", *NEW_GROUPS, *OPT_GROUPS),
        required=True,
    )
    ap.add_argument("--list", action="store_true", help="print the job commands, one per line")
    ap.add_argument("--names", action="store_true", help="print the job names")
    ap.add_argument("--gpus", action="store_true", help="print `name gpus` per job")
    ap.add_argument("--gpus-per-node", type=int, default=GPUS_PER_NODE)
    args = ap.parse_args(argv)
    for name, cmd in jobs(args.group, args.gpus_per_node).items():
        if args.names:
            print(name)
        elif args.gpus:
            print(name, gpus(args.group, args.gpus_per_node)[name])
        else:
            print(cmd)


if __name__ == "__main__":
    main()
