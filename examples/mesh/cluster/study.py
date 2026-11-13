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
* ``fitgate`` (2026-10-29): the shape adjoint's grad-lambda fits (k-nearest and per-cell
  polynomial variants, the pooled collision field, dt and particle-count gates, the airfoil
  seed-4 outlier) against central differences, 2-D specular / diffuse pairs at Kn 0.5 and
  M 0.32 / 10 plus the paper's airfoil (:func:`_fitgate_jobs`; ``scratch/fitgate/``).
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
import json
from pathlib import Path

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
# 2026-10-28 (before its first start): with the thickness constraint (thin share T <= 0.02 at
# 2 knots, multiplier from 1 at rate 10) and the topology guard, mark ``hz0th``: the 3-D L / D
# bodies thin into sub-knot plates that split (sphere_long Kn 0.5 L / D: 2 pieces from ~215);
# in the 3-D Newtonian battery (scratch/topo) the constraint held T ~0.03 in one piece at
# ~15 % of L / D, and the guard (a backstop) never fired beside it.
LD3DWIDE_ITERATIONS = 400
LD3DWIDE = "--thickness 1 --thickness-rate 10 --topology-guard 1"


def _ld3dwide_jobs(gpus_per_node: int) -> dict[str, str]:
    """``ld3dwide``: :func:`_lsfilter3d_jobs`' M 10 L / D run on ``sphere_wide``; tag
    ``sphere_wide_kn0.5_M10_S1_ox_adam_ldfl_hz0th_free_w300`` (with :data:`LD3DWIDE`)."""
    cmd = _lsfilter3d_jobs(gpus_per_node)["sphere_long_kn0.5_M10_ldfl_hz0"]
    cmd = cmd.replace("sphere_long", "sphere_wide").replace("_hz0_free_w300", "_hz0th_free_w300")
    cmd = cmd.replace(
        f"--iterations {LSFILTER3D_ITERATIONS}", f"--iterations {LD3DWIDE_ITERATIONS}", 1
    ).replace(f"--step-decay {LSFILTER3D_ITERATIONS}", f"--step-decay {LD3DWIDE_ITERATIONS}", 1)
    return {"sphere_wide_kn0.5_M10_ldfl_hz0th": f"{cmd} {LD3DWIDE}"}


# dlh3d (2026-11-02): minimum drag subject to a minimum lift (L >= 0.15 D_0, D_0 the sphere's
# drag; --objective drag_lift, its multiplier from 1 at rate 2), a peak-heating cap (the
# heat p-mean no higher than the sphere's) and a volume floor (0.8 of the sphere's) -- better
# posed than max L / D, whose bodies thinned into plates.  The lift: diffuse walls cap L / D
# far below the specular runs' 2.4 (free-molecular flat plate at M 10: 0.13; the transitional
# shock layer raises it to a few tenths), so L >= 0.5 D_0 would need D >~ D_0 -- no drag to
# remove; at 0.15 the lift costs ~0.2-0.5 D_0 of drag, the slender minimum-drag body's own
# order (friction).  The heat: the sphere's stagnation heating sits below the free-molecular
# value (shock layer) while a sub-lambda sharp nose approaches it, so a sharpened nose can
# exceed the sphere's p-mean although its area is small -- the cap at 1.0 binds when that
# happens; 0.8 would bind from the first iterate on the sphere itself.  The wall fit
# cell_along_pool_plane (2026-11-03: with cell_pool_plane the diffuse radius gates' unbiased
# pair at Kn 0.5 / 2 / 10, but cell_pool_plane reads 0.87 on the diffuse squash and +20 % on the
# specular circle; the default along_pool_plane reads 0.91 at Kn 10 on diffuse walls; the cos 2
# theta gates fail every fit, m2chk).  The first run (cell_pool_plane) OOMed on h200_71 at
# iterate 23 (A_max 72 -> 132) and is to be restarted from 0, not resumed.  The
# M 10 Kn 0.5 sphere_long of lsfilter3d (Helmholtz, no reinitialisation) with diffuse walls
# (the heat flux needs them; the silhouette term's cost fixed 2026-11-02), plus ld3dwide's
# thickness constraint and topology guard.  Overnight size: --fnum-scale 2 (half
# sphere_long's particles, ~8 per free-stream cell) and 150 iterates on a cosine step decay
# (lsfilter3d's specular M 10 runs: ~190 s an iterate at fnum-scale 1; diffuse backwards
# are ~2x, half the particles ~0.5x).  Resubmit to resume / extend:
#   GPU_KIND=h200_full bash examples/mesh/cluster/submit_curc.sh dlh3d   (71 GB OOMed)
DLH3D_ITERATIONS = 150
DLH3D = (
    "--acc 1 --silhouette-band 0.03 --objective drag_lift --lift-min 0.15 --lift-rate 2 "
    "--lift-multiplier 1 --heat-patch 0.2 --heat-cap 1.0 --heat-cap-relative 1 "
    "--heat-multiplier 1 --heat-rate 2 --area-mode floor --area-floor 0.8 "
    "--thickness 1 --thickness-rate 10 --topology-guard 1 --variant cell_along_pool_plane"
)


def _dlh3d_jobs(gpus_per_node: int) -> dict[str, str]:
    """``dlh3d``: :data:`DLH3D` on lsfilter3d's M 10 sphere_long at half the particles; tag
    ``sphere_long_kn0.5_M10_S1_ox_adam_dlh_hz0_f2_w300``."""
    del gpus_per_node
    cmd = (
        f"python3 -m examples.mesh.optimize --tag sphere_long_kn0.5_M10_S1_ox_adam_dlh_hz0_f2_w300 "
        f"{SPHERE_LONG} --knudsen 0.5 --mach 10 --dt 2.5e-5 {KNPULLBACK} --optimizer ox_adam "
        f"--a-max 16 {CONVERGE} {FIX} {HELM} {NO_REINIT} --step-decay {DLH3D_ITERATIONS} {DLH3D}"
    )
    cmd = cmd.replace("--fnum-scale 1 ", "--fnum-scale 2 ", 1).replace(
        "--iterations 400", f"--iterations {DLH3D_ITERATIONS}", 1
    )
    return {"sphere_long_kn0.5_M10_dlh": cmd}


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


def _fitgate_blocks(gate: str, short: str, seeds: int, bases, extra: str = "") -> dict[str, str]:
    """``fitgate``'s jobs of one gate: blocks of 4 seeds, one job per block and entry of
    ``bases`` (a basis, or a tuple of bases run in one job: one forward and one set of FD
    runs per seed shared by its bases)."""
    run = "python3 scratch/fitgate/variant_gate.py"
    return {
        f"{short}_s{first}_{'-'.join(_tuple(basis))}": (
            f"{run} --gate {gate} --seeds {first} {first + 4} --bases {' '.join(_tuple(basis))}"
            + extra
        )
        for first in range(0, seeds, 4)
        for basis in bases
    }


def _tuple(basis) -> tuple[str, ...]:
    return (basis,) if isinstance(basis, str) else tuple(basis)


FITGATE_CANDIDATES = (
    "along_pool_plane",
    "cell_pool_plane",
    "cell_along_pool_plane",
    "cellxv_along_pool_plane",
    "one_pool_plane",
)
"""The fits carried by the memory-bound fitgate gates (2026-11-01): the leaders at half-size
cells plus the k-nearest fit that did not respond to the cell size (the control)."""


def _diffchk_jobs(gpus_per_node: int) -> dict[str, str]:
    """``diffchk`` (2026-11-03): the production diffuse wall checked against central
    differences on the dlh3d objective's three pieces (variant_gate's c10_diff_m2: drag under
    the cos 2 theta elongation; c10_diff_heat_m2: the total heat rate under it; and
    c10_diff_lift_m2s: J = D - L under the sin 2 theta inclination, dD = 0 by symmetry), the
    M 10 Kn 0.5 circle, 8 seeds each, the per-cell fits beside the old default.  As c10_diff
    (~23 GB): GPU_KIND=rtx_48 bash examples/mesh/cluster/submit_curc.sh diffchk"""
    del gpus_per_node
    chk = " --variants cell_pool_plane cell_along_pool_plane along_pool_plane"
    jobs = {}
    for gate, short in (("c10_diff_m2", "dchk_drag"), ("c10_diff_heat_m2", "dchk_heat"),
                        ("c10_diff_lift_m2s", "dchk_lift")):  # fmt: skip
        jobs |= _fitgate_blocks(gate, short, 8, ("base",), chk)
    return jobs


def _m2chk_jobs(gpus_per_node: int) -> dict[str, str]:
    """``m2chk`` (2026-11-03): where the diffuse cos 2 theta failure lives (c10_diff_m2 0.73,
    the radius passing) -- the diffuse circle at Kn 10 (no collisional terms), the specular
    circle (no diffuse-only terms), and c10_diff_m2's forward and FD with the grazing floor
    0.01 -> 0.001 and the silhouette band 0.03 -> 0.06; 4 seeds each, the diffchk variants.
    ``GPU_KIND=rtx_48``."""
    del gpus_per_node
    chk = " --variants cell_pool_plane cell_along_pool_plane along_pool_plane"
    jobs = {}
    for gate, short in (
        ("c10_diff_kn10_m2", "m2_dkn10"),
        ("c10_spec_m2", "m2_spec"),
        ("c10_diff_m2_gz001", "m2_gz001"),
        ("c10_diff_m2_sil2", "m2_sil2"),
    ):
        jobs.update(_fitgate_blocks(gate, short, 4, ("base",), chk))
    return jobs


def _m2split_jobs(gpus_per_node: int) -> dict[str, str]:
    """``m2split`` (2026-11-03, after m2chk put the cos 2 theta failure in the diffuse wall's
    collisional terms): the pathwise_x split of c10_diff under cos 2 theta (seeds 0-3) and the
    radius (seeds 0, 4, 8, 12: c10_diff's), and the order term's fit at k = 128, each paired
    with its parent's forward and FD.  ``GPU_KIND=rtx_48`` (diffuse Kn 0.5 OOMs 24 GB)."""
    del gpus_per_node
    run = "python3 scratch/fitgate/variant_gate.py"
    pair = " --variants along_pool_plane cell_along_pool_plane cellxv_along_pool_plane"
    return {
        "m2s_xs_m2": f"{run} --gate c10_diff_m2_xsplit --seeds 0 4 --split-x{pair}",
        "m2s_xs_r": f"{run} --gate c10_diff_xsplit --seeds 0 16 --stride 4 --split-x{pair}",
        "m2s_ord": f"{run} --gate c10_diff_m2_ord128 --seeds 0 4{pair}",
    }


def _silfix_jobs(gpus_per_node: int) -> dict[str, str]:
    """``silfix`` (2026-11-03): the diffuse M 10 Kn 0.5 checks after the silhouette search
    fix (levelset._culled bucket order) -- the radius (c10_diff's seeds 0, 4, 8, 12) and the
    cos 2 theta drag / sin 2 theta lift / cos 2 theta heat (seeds 0-3), FD reused from the
    original gates.  ``GPU_KIND=rtx`` (24 GB with the compact pair record)."""
    del gpus_per_node
    run = "python3 scratch/fitgate/variant_gate.py"
    var = " --variants cell_along_pool_plane cell_pool_plane along_pool_plane"
    return {
        "sf_radius": f"{run} --gate c10_diff_sf --seeds 0 16 --stride 4{var}",
        "sf_drag": f"{run} --gate c10_diff_m2_sf --seeds 0 4{var}",
        "sf_lift": f"{run} --gate c10_diff_lift_m2s_sf --seeds 0 4{var}",
        "sf_heat": f"{run} --gate c10_diff_heat_m2_sf --seeds 0 4{var}",
    }


# bitchk (2026-11-03): the compact pair record's bitwise check on one optimiser iterate -- the
# lsfilter specular M 10 Kn 0.5 drag run's command (its iterate 0 ran on an RTX 1g.24gb slice,
# 10-27), one iterate, the lean record compact against dense; grad_0 / coef_1 / the history's
# objective must agree bit for bit (the stored 10-27 grad_0 is a looser reference: code has moved)
BITCHK = (
    "python3 -m examples.mesh.optimize --seeds-per-iterate 1 --workers 1 --fnum-scale 1 "
    "--grid-scale 1 --levelset-h 0.0625 --levelset-samples 16 --grazing 0.05 --case circle_long "
    "--fnum-scale 2 --warmup 2800 --step 0.05 --smooth 8 --band 1.0 --keepout 1.0 "
    "--area-mode tangent --boundary free --optimizer ox_adam --area-mode tangent --initial diamond "
    "--initial-thickness 0.3 --recentre 1 --reinit-every 0 --filter helmholtz --save-gradient 1 "
    "--converge-patience 100 --backward-normals true --grazing 0.01 --knudsen 0.5 --mach 10 "
    "--dt 2.5e-5 --a-max 256 --substeps 4 --partners 8 --iterations 1 --restart-steps 500"
)


def _bitchk_jobs(gpus_per_node: int) -> dict[str, str]:
    """``bitchk``: :data:`BITCHK` with ``--record lean`` (compact pairs) and ``lean_dense``,
    tags ``bitchk_dmhz0_{compact,dense}``.  ``GPU_KIND=rtx`` (the original's slice kind)."""
    del gpus_per_node
    return {
        f"bitchk_{name}": f"{BITCHK} --tag bitchk_dmhz0_{name} --record {record}"
        for name, record in (("compact", "lean"), ("dense", "lean_dense"))
    }


# ellchk (2026-11-03): two iterates of the dlh3d setup under the constraint modes (the second
# has the heat functional's own DSMC gradient: dG/dQ lags one iterate), from the
# start ellipsoid (optimize --initial ellipsoid: semi-axes 1.6 x 0.8 x 0.8, the sphere's
# volume) at 10 / 15 / 20 degrees and from the sphere (lift violated: the projection's
# active rows and restoration step), the constraints in the sphere's units -- its iterate-0
# drag and heat measure from the first dlh3d run (forward quantities: the silhouette fix
# does not move them) and its body volume
SPHERE_REFS = (
    "--lift-ref 2.443238526935362 --heat-g-ref 0.6120237258384549 --area-ref 4.214628770157608"
)


def _constrained_dlh3d(tag: str, extra: str, iterations: int = 2, case: str = "") -> str:
    """:func:`_dlh3d_jobs`' command with ``tag``, two iterates, the sphere's references and
    ``extra`` (the start body and the constraint mode)."""
    cmd = _dlh3d_jobs(1)["sphere_long_kn0.5_M10_dlh"]
    cmd = cmd.replace("--tag sphere_long_kn0.5_M10_S1_ox_adam_dlh_hz0_f2_w300", f"--tag {tag}", 1)
    cmd = cmd.replace(f"--iterations {DLH3D_ITERATIONS}", f"--iterations {iterations}", 1)
    if case:
        cmd = cmd.replace("--case sphere_long ", f"--case {case} ", 1)
        assert f"--case {case} " in cmd, cmd
    return f"{cmd} {SPHERE_REFS} {extra}"


def _ellchk_jobs(gpus_per_node: int) -> dict[str, str]:
    """``ellchk``: four two-iterate jobs (``GPU_KIND=h200_71``; the compact pair record keeps
    the 3-D seed well under 71 GB): projected Adam from the ellipsoid at 10 and 15 degrees,
    MMA at 20 degrees, projected Adam from the sphere."""
    del gpus_per_node
    ell = "--initial ellipsoid --initial-thickness 0.5 --initial-aoa"
    return {
        "ell_a10_proj": _constrained_dlh3d("ellchk_a10_proj", f"{ell} 10 --constraints project"),
        "ell_a15_proj": _constrained_dlh3d("ellchk_a15_proj", f"{ell} 15 --constraints project"),
        "ell_a20_mma": _constrained_dlh3d(
            "ellchk_a20_mma", f"{ell} 20 --constraints mma --optimizer mma_s"
        ),
        "sph_proj": _constrained_dlh3d("ellchk_sphere_proj", "--constraints project"),
    }


def _ellprobe_jobs(gpus_per_node: int) -> dict[str, str]:
    """``ellprobe`` (2026-11-03): one iterate each in ``sphere_big`` (side faces 3 radii out)
    -- the sphere (its drag and heat measure in the larger box: the references of the runs)
    and the start ellipsoid at 35 / 45 / 55 degrees (where its lift peaks; the 10-20 degree
    ellchk values fit L ~ 0.226 sin 2 alpha), projected Adam, ``GPU_KIND=h200_71``."""
    del gpus_per_node
    ell = "--initial ellipsoid --initial-thickness 0.5 --initial-aoa"
    jobs = {"probe_sphere": _constrained_dlh3d("ellprobe_sphere", "--constraints project", 1,
                                               "sphere_big")}  # fmt: skip
    for aoa in (35, 45, 55):
        jobs[f"probe_a{aoa}"] = _constrained_dlh3d(
            f"ellprobe_a{aoa}", f"{ell} {aoa} --constraints project", 1, "sphere_big"
        )
    return jobs


# dlhe (2026-11-04, after ellprobe): the constrained dlh3d runs from the 45-degree ellipsoid in
# sphere_big (lift peaks there: L 0.238 = 0.096 D_s; 35 / 55 degrees 0.220 / 0.227), the lift
# floor halved to 0.075 D_s = 0.185 (the start 29 % above it; 0.15 D_s is beyond this body's
# reach) and the references re-measured in the big box (the sphere's drag +1.1 %, heat +0.6 %
# over sphere_long's; the 45-degree start's heat 0.600 = 0.974 of the cap)
SPHERE_BIG_REFS = (
    "--lift-ref 2.470640398729704 --heat-g-ref 0.6157509902191032 --area-ref 4.2146287701575575"
)
DLHE_START = "--initial ellipsoid --initial-thickness 0.5 --initial-aoa 45"
DLHE_LIFT_MIN = 0.075


def _dlhe_jobs(gpus_per_node: int) -> dict[str, str]:
    """``dlhe``: :data:`DLH3D` from :data:`DLHE_START` in ``sphere_big`` with the lift floor
    :data:`DLHE_LIFT_MIN` -- the penalty (multiplier) form, MMA and projected Adam, the full
    :data:`DLH3D_ITERATIONS` (``GPU_KIND=h200_71``; resubmit to resume)."""
    del gpus_per_node
    modes = {
        "pen": "--constraints penalty",
        "mma": "--constraints mma --optimizer mma_s",
        "proj": "--constraints project",
    }
    jobs = {}
    for name, mode in modes.items():
        cmd = _constrained_dlh3d(f"dlhe_a45_{name}", "", DLH3D_ITERATIONS, "sphere_big")
        cmd = cmd.replace(SPHERE_REFS, SPHERE_BIG_REFS, 1).replace(
            "--lift-min 0.15 ", f"--lift-min {DLHE_LIFT_MIN} ", 1
        )
        assert SPHERE_BIG_REFS in cmd and f"--lift-min {DLHE_LIFT_MIN} " in cmd, cmd
        jobs[f"dlhe_{name}"] = f"{cmd.rstrip()} {DLHE_START} {mode}"
    return jobs


def _dlhp_jobs(gpus_per_node: int) -> dict[str, str]:
    """``dlhp`` (2026-11-05, a presentation run due in ~22 h): :data:`DLH3D` with the
    penalty (multiplier) constraints and Adam -- the path dlh3d descended on for 22 iterates
    -- from the 20 degree ellipsoid in ``sphere_long`` (ellchk's tested start: sphere_big's
    45 degree start costs ~1200 s per iterate) with the lift floor :data:`DLHE_LIFT_MIN`, 120
    iterates, at two particle counts: ``dlhp_f2`` (dlh3d's; ellchk at f2 after the lean
    collision record peaked at 34.85 GiB on the 71 GB slice, ~470 s per iterate: ~14 h) and
    ``dlhp_f4`` (half: ~1.5-2x faster, a ~20 GB peak -- the safety net, or a 35 GB slice).
    ``GPU_KIND=h200_71`` (``ONLY=f4 GPU_KIND=h200`` for the small slice); resubmit to resume.
    The production grad-lambda fit (cell_along_pool_plane): the fit lab's near-wall fits are
    not in production."""
    del gpus_per_node
    jobs = {}
    for fnum in (2, 4):
        cmd = _constrained_dlh3d(
            f"dlhp_a20_pen_f{fnum}", "--initial ellipsoid --initial-thickness 0.5 "
            "--initial-aoa 20 --constraints penalty", 120,
        )  # fmt: skip
        cmd = cmd.replace("--fnum-scale 2 ", f"--fnum-scale {fnum} ", 1).replace(
            "--lift-min 0.15 ", f"--lift-min {DLHE_LIFT_MIN} ", 1
        )
        assert f"--fnum-scale {fnum} " in cmd and f"--lift-min {DLHE_LIFT_MIN} " in cmd, cmd
        jobs[f"dlhp_f{fnum}"] = cmd
    # MMA (2026-11-05): the constraints inside each step (its own subproblem on the lift /
    # heat / volume gradients from one multi-column backward) -- the Adam runs spent ~45
    # iterates winding the lift multiplier up and the rest unwinding its overshoot.  f2: the
    # f2 Adam run was the cleaner (less noise).  Started FEASIBLE: the 35 degree ellipsoid
    # (ellprobe: L 0.220 against the floor 0.075 x 2.443 = 0.183; y extent ~1.13 m, ~0.9 m
    # from sphere_long's side faces) -- the 20 degree start (L 0.145) would spend MMA's first
    # iterates restoring feasibility through its artificial variables
    mma = _constrained_dlh3d(
        "dlhp_a35_mma_f2", "--initial ellipsoid --initial-thickness 0.5 --initial-aoa 35 "
        "--constraints mma --optimizer mma_s", 120,
    ).replace("--lift-min 0.15 ", f"--lift-min {DLHE_LIFT_MIN} ", 1)  # fmt: skip
    assert "--fnum-scale 2 " in mma and f"--lift-min {DLHE_LIFT_MIN} " in mma, mma
    jobs["dlhp_mma_f2"] = mma
    return jobs


def _multichk_jobs(gpus_per_node: int) -> dict[str, str]:
    """``multichk``: drag, lift and heat from one 6-column backward against the three
    single-objective backwards, the diffuse M 10 circle, seed 0 (``GPU_KIND=rtx_48``)."""
    del gpus_per_node
    return {"multichk_s0": "python3 scratch/multiobj/check_multi.py --seed 0"}


def _termchk_jobs(gpus_per_node: int) -> dict[str, str]:
    """``termchk`` (2026-11-03): which Kn 0.5 diffuse term misplaces its surface weight under
    cos 2 theta (drag 1.22-1.27, heat 1.7-1.9 after silfix) -- the heat base on the current
    code, the silhouette band at 0.015 / 0.06 and the order fit at k 128 for drag and heat,
    seeds 0-3, FD reused (``GPU_KIND=rtx``)."""
    del gpus_per_node
    run = "python3 scratch/fitgate/variant_gate.py"
    var = " --variants cell_along_pool_plane cell_pool_plane"
    gates = {
        "tc_heat_base": "c10_diff_heat_m2_sf2",
        "tc_drag_sil15": "c10_diff_m2_sil15",
        "tc_drag_sil60": "c10_diff_m2_sil60",
        "tc_heat_sil15": "c10_diff_heat_m2_sil15",
        "tc_heat_sil60": "c10_diff_heat_m2_sil60",
        "tc_drag_ord": "c10_diff_m2_ord128b",
        "tc_heat_ord": "c10_diff_heat_m2_ord128",
    }
    return {name: f"{run} --gate {gate} --seeds 0 4{var}" for name, gate in gates.items()}


def _termvol_jobs(gpus_per_node: int) -> dict[str, str]:
    """``termvol``: the heat objective's cut-volume weights in situ (volume_gate on the
    600-step heat gate, the cut cells and one layer about them), 2 seeds a job, own FD
    (``GPU_KIND=rtx_48``)."""
    del gpus_per_node
    run = "python3 scratch/fitgate/volume_gate.py --gate c10_diff_heat_h600 --sets cut near1"
    return {f"tv_heat_s{a}": f"{run} --seeds {a} {a + 2}" for a in (0, 2)}


def _bumpchk_jobs(gpus_per_node: int) -> dict[str, str]:
    """``bumpchk`` (2026-11-03): where the Kn 0.5 diffuse excess sits -- the bump gate (nose /
    side / base controls, every term projected on each from one backward, own FD at step
    0.1; 8 seeds for drag and heat in blocks of 4) and the heat objective's pathwise_x split
    (paired, FD reused).  ``GPU_KIND=rtx``."""
    del gpus_per_node
    bump = "python3 scratch/fitgate/bump_gate.py"
    jobs = {
        f"bump_{obj}_s{a}": f"{bump} --objective {obj} --seeds {a} {a + 4}"
        for obj in ("drag", "heat")
        for a in (0, 4)
    }
    jobs["xsplit_heat"] = (
        "python3 scratch/fitgate/variant_gate.py --gate c10_diff_heat_m2_xsplit --seeds 0 4 "
        "--split-x --variants along_pool_plane cell_along_pool_plane cellxv_along_pool_plane"
    )
    return jobs


def _fitlab_jobs(gpus_per_node: int) -> dict[str, str]:
    """``fitlab`` (2026-11-04, stage 1): the fit lab's data (scratch/fitlab/dump.py: every
    captured step's near-wall particles and body hits, with the hits' d x_post / d theta for
    radius / cos 2 / sin 2 / the three bumps and the production fits at the queries) on the
    seeds whose FDs exist -- the failing diffuse M 10 Kn 0.5 circle (drag: the bump gate's
    seeds 0-7; heat 0-7), the specular one (+8-20 %, 32-seed radius FD; seeds 0-7) and the
    passing diffuse Kn 10 control (cos 2 theta 1.02-1.04; seeds 0-3).  ``GPU_KIND=rtx``."""
    del gpus_per_node
    dump = "python3 scratch/fitlab/dump.py"
    jobs = {
        f"lab_{name}_s{a}": f"{dump} --gate {gate} --seeds {a} {a + 4}"
        for name, gate in (("drag", "c10_diff_m2"), ("heat", "c10_diff_heat_m2"),
                           ("spec", "c10_spec"))
        for a in (0, 4)
    }  # fmt: skip
    jobs["lab_kn10_s0"] = f"{dump} --gate c10_diff_kn10_m2 --seeds 0 4"
    return jobs


def _fitlab2_jobs(gpus_per_node: int) -> dict[str, str]:
    """``fitlab2`` (2026-11-05, the transfer cases): the near-wall fit's settings must not be
    tuned to M 10 with T_w = T_inf -- the same circle and box at M 25 and M 0.32 (diffuse and
    specular) and the M 10 diffuse wall at 1000 K, each dump with its own radius / cos 2 theta
    central differences on the same seeds (``dump.py --fd``), 8 seeds in blocks of 4.
    ``GPU_KIND=rtx``."""
    del gpus_per_node
    dump = "python3 scratch/fitlab/dump.py"
    return {
        f"lab_{gate}_s{a}": f"{dump} --gate {gate} --seeds {a} {a + 4} --fd radius mode2"
        for gate in ("c25_diff_m2", "c25_spec_m2", "c03_diff_m2", "c03_spec_m2", "c10_hot_m2")
        for a in (0, 4)
    }


def _shiftfd_jobs(gpus_per_node: int) -> dict[str, str]:
    """``shiftfd`` (2026-11-05): the fit-free truth of pathwise_x (scratch/fitlab/shiftfd.py:
    the body hits' post-move positions shifted by eps w (v_out - v_in) / |v_in . n|, one
    warm-up per seed, a forward difference per control / mode / eps from the same state and
    keys; drag and heat from the same runs) on the fit lab's diffuse seeds 0-7, against which
    every fit's sum grad_x lambda . shift is scored.  ``GPU_KIND=rtx``."""
    del gpus_per_node
    return {f"shift_s{a}": f"python3 scratch/fitlab/shiftfd.py --seeds {a} {a + 4}" for a in (0, 4)}


def _fitlab3_jobs(gpus_per_node: int) -> dict[str, str]:
    """``fitlab3`` (2026-11-05, the refinement gate): the M 10 Kn 0.5 diffuse circle's dumps and
    own radius / cos 2 theta FDs at grid_scale 2 and 4 (the same particles per cell: 4x / 16x
    the particles) -- does the production grad-lambda bias fall as the near-wall cells approach
    the local mean free path (~7-9 cells per mfp at k = 1)?  8 seeds in blocks of 4 (k = 4 in
    blocks of 2).  ``GPU_KIND=h200_71``; slim on the cluster before the pull."""
    del gpus_per_node
    dump = "python3 scratch/fitlab/dump.py"
    jobs = {}
    for gate, block in (("c10_diff_m2_g2", 4), ("c10_diff_heat_m2_g2", 4), ("c10_diff_m2_g4", 2)):
        for a in range(0, 8, block):
            # the dump's near-wall band in cells: ~0.6 m at every k (the lab's d < 0.6 m)
            width = 2.5 if gate.endswith("_g2") else 5.0
            jobs[f"lab_{gate}_s{a}"] = (
                f"{dump} --gate {gate} --seeds {a} {a + block} --fd radius mode2 --width {width}"
            )
    return jobs


def _fitgate_jobs(gpus_per_node: int) -> dict[str, str]:
    """``fitgate`` (2026-10-29): the shape adjoint's fits against central differences at
    transitional conditions, in 2-D (scratch/fitgate/variant_gate.py; results in
    examples/mesh/results/fitgate/).  Every grad-lambda estimator of jax_dsmc.shape.VARIANTS
    (k-nearest and per-cell polynomial; pooled or not; with or without the along-flight
    identity), each with its own silhouette, on the same realisations, under the shipped
    collision field (``base``) and the pooled one (``pool1``):

    * ``sq_spec`` / ``sq_diff``: the Kn 0.5 M 0.32 squash at s = 0.6 that passed specular, and
      its diffuse twin; ``c10_spec`` / ``c10_diff``: the R = 3 circle at Kn 0.5 M 10 (the L / D
      box), specular and diffuse; ``af_r`` / ``af_m2``: the paper's airfoil gates (FD reused
      for seeds 0-15, computed beyond; seed 4 the outlier); 32 seeds in blocks of 4, the
      5^D-pooled collision field ``pool2`` too on ``af_r``;
    * ``c10_diff_kn0.2`` / ``c10_diff_kn2``: the diffuse M 10 circle's Kn sweep (16 seeds);
    * ``c10_diff_dt2`` / ``c10_diff_f2`` (``base`` only, 16 seeds): the diffuse M 10 circle with dt
      halved (the along-flight identity's O(nu dt) error) and with half the particles (the
      fits' kernel width);
    * ``c10_spec_dt2`` (``base``, 16 seeds): the specular M 10 circle with dt halved (its
      variant-independent +11-14 %, 2026-10-30);
    * ``c10_spec_h1000`` / ``c10_diff_h1000`` (``base``, 8 seeds, h200_71): the M 10 circles'
      backward over the last 1000 captured steps instead of 300, the same forwards (2026-11-01);
    * ``vol_*`` (2026-11-01): the collision pull-back's volume weights against FD of scaled
      collision volumes (every cell / the cut cells, eps 0.2 and 0.1) on c10_spec, c10_diff
      and the sq_spec control;
    * ``c10_spec_g2n`` / ``c10_spec_g2`` (``base``): the specular M 10 circle on collision
      cells half the size, same particle count / same particles per cell (2026-11-01);
    * ``af_seed4_diag``: the airfoil seed 4 with every slope fit's node count and conditioning
      (scratch/fitgate/seed_outlier.py; the realisation needs the cluster GPU);
    * ``smoke`` first.  All 2-D: rtx slices, except the diffuse M 10 circle's ``c10d_*`` jobs
      (``GPU_KIND=h200 ONLY='^c10d_(s|kn2)'``, Kn 0.2 ``GPU_KIND=h200_71
      ONLY='^c10d_kn0'``: more than a 24 GB slice); a chosen configuration gets one 3-D check
      (the paper's Newton box) before the 3-D showcases."""
    del gpus_per_node
    run = "python3 scratch/fitgate/variant_gate.py"
    out = "examples/mesh/results/fitgate"
    jobs = {"smoke": f"{run} --gate af_drag_radius --seeds 0 1 --bases base pool1 --smoke"}

    # 32 seeds in blocks of 4 (one job ~30-60 min on an rtx slice)
    for gate, short in (("sq_spec", "sq_spec"), ("sq_diff", "sq_diff"), ("c10_spec", "c10_spec"),
                        ("af_drag_mode2", "af_m2")):  # fmt: skip
        jobs |= _fitgate_blocks(gate, short, 32, ("base", "pool1"))
    jobs |= _fitgate_blocks("af_drag_radius", "af_r", 32, ("base", "pool1", "pool2"))
    # the diffuse M 10 circle: the cold wall's compressed layer (peak ~500 NTC attempts per
    # cell-step, A_max 256 -> ~700) needs ~23 GB with base, past the 24 GB rtx slice (every
    # c10_diff job OOMed, 2026-10-30) -> GPU_KIND=h200 (35 GB slices); base only: pool1's
    # backward needs ~31 GB + a 25 GB buffer (OOMed on the 35 GB slice after seed 0's base row,
    # 2026-10-31) and reads 3 % low on the squash, pool2 ~81 GB.  Seeds 0, 4, ..., 28 are done
    # (c10_diff_base-pool1_s*.jsonl); each block's other three, ~2.7 h a seed
    jobs |= {
        f"c10d_s{first + 1}_base": (
            f"{run} --gate c10_diff --seeds {first + 1} {first + 4} --bases base"
        )
        for first in range(0, 32, 4)
    }
    # Kn 0.2: base OOMed on the 35 GB slice (an 18 GB buffer) -> GPU_KIND=h200_71
    jobs |= _fitgate_blocks("c10_diff_kn0.2", "c10d_kn0.2", 16, ("base",))
    jobs |= _fitgate_blocks("c10_diff_kn2", "c10d_kn2", 16, ("base",))
    for gate in ("c10_diff_dt2", "c10_diff_f2"):
        jobs |= _fitgate_blocks(gate, gate.replace("c10_diff", "c10d"), 16, ("base",))
    # the specular M 10 circle with dt halved (fits a 24 GB rtx slice: c10_spec never outgrew
    # A_max 256)
    jobs |= _fitgate_blocks("c10_spec_dt2", "c10_spec_dt2", 16, ("base",))
    # the captured horizon, 1000 steps instead of 300, paired with c10_spec / c10_diff (their
    # FD reused: c10_spec seeds 0-7, c10_diff's done seeds 0, 4, ..., 28): records ~3.3x ->
    # GPU_KIND=h200_71 ONLY='^c10h_'
    jobs |= _fitgate_blocks("c10_spec_h1000", "c10h_spec", 8, ("base",))
    # 600 steps on the specular circle (2026-11-02): the horizon trend 300 / 600 / 1000, paired
    # on seeds 0-7, FD reused; ~2x c10_spec's record -> GPU_KIND=rtx_48 ONLY='^c10h600_spec'
    jobs |= _fitgate_blocks("c10_spec_h600", "c10h600_spec", 8, ("base",))
    # pathwise_x split into its fitted-perpendicular, timing and collision parts (2026-11-02:
    # the specular bias lives in the one variant-dependent term), Kn 0.5 and Kn 2, seeds 0-7,
    # FD reused: GPU_KIND=rtx ONLY='^c10xs_', GPU_KIND=rtx_48 ONLY='^c10xs2_'
    split = " --variants " + " ".join(FITGATE_CANDIDATES) + " --split-x"
    # and its controls: Kn 10 (the collision part should vanish) and 1000 captured steps
    # (which part carries the horizon's -3 %): GPU_KIND=rtx ONLY='^c10xs10_',
    # GPU_KIND=h200_71 ONLY='^c10xsh_'.  2026-11-03: Kn 1, 32 seeds with its own FD (Kn 2's
    # FD noise, +-2.2 %, could not tell an excess linear in the collisional part from a
    # quadratic one; Kn 1 predicts ~5 % against ~2.5 %): GPU_KIND=rtx ONLY='^c10xs1_'
    for gate, prefix, seeds in (("c10_spec_xsplit", "c10xs", 8),
                                ("c10_spec_kn2_xsplit", "c10xs2", 8),
                                ("c10_spec_kn10_xsplit", "c10xs10", 8),
                                ("c10_spec_h1000_xsplit", "c10xsh", 8),
                                ("c10_spec_kn1_xsplit", "c10xs1", 32)):  # fmt: skip
        jobs |= _fitgate_blocks(gate, prefix, seeds, ("base",), split)
    # the diffuse one OOMed on the 71 GB slice (2026-11-01: ~66 GB in use, then the backward's
    # [1000, N, 2] reward columns, 3.1 GB) -> GPU_KIND=rtx_full ONLY='^c10h_diff', two seeds a
    # job (the diffuse backward is the slow one)
    jobs |= {
        f"c10h_diff_s{first}_base": (
            f"{run} --gate c10_diff_h1000 --seeds {first} {first + 8} --stride 4 --bases base"
        )
        for first in (0, 8, 16, 24)
    }
    # collision cells half the size on the specular M 10 circle: at the same particle count
    # (c10_spec_g2n, 16 seeds, GPU_KIND=h200 or rtx_48 ONLY='^c10g2n_') and at the same
    # particles per cell (c10_spec_g2, 8 seeds, 4x the particles: GPU_KIND=h200_full or
    # rtx_full ONLY='^c10g2_')
    jobs |= _fitgate_blocks("c10_spec_g2n", "c10g2n", 16, ("base",))
    jobs |= _fitgate_blocks("c10_spec_g2", "c10g2", 8, ("base",))
    # the cell scaling (a quarter cell, same particles) and the diffuse circle on half cells:
    # GPU_KIND=h200_71 ONLY='^c10(g4n|dg2n)_'.  Both OOMed there with all 18 variants
    # (2026-11-01; c10g2n used 66 of 71 GB, its 35 GB attempt died in the shape VJP, which
    # carries every estimator's fit tables -- cells x npc_max, padded to the densest cell, 3^D
    # that for the pooled fits): the five candidates only
    few = " --variants " + " ".join(FITGATE_CANDIDATES)
    jobs |= _fitgate_blocks("c10_spec_g4n", "c10g4n", 16, ("base",), few)
    jobs |= _fitgate_blocks("c10_diff_g2n", "c10dg2n", 16, ("base",), few)
    # 2026-11-01 (cell size ruled out): the specular circle at Kn 10 / 2 (is the bias
    # collisional on this body?), 16 seeds, GPU_KIND=rtx_48 ONLY='^c10kn'; the derivative-only
    # settings paired with c10_spec (FD reused, seeds 0-7): GPU_KIND=rtx ONLY='^c10p_'
    for gate, short in (("c10_spec_kn10", "c10kn10"), ("c10_spec_kn2", "c10kn2")):
        jobs |= _fitgate_blocks(gate, short, 16, ("base",))
    # the diffuse circle's collisionless anchor (with c10d_kn2 above: the collisions' share of
    # the diffuse dJ/dR, as for the specular one), GPU_KIND=rtx_48 ONLY='^c10dkn10'
    jobs |= _fitgate_blocks("c10_diff_kn10", "c10dkn10", 16, ("base",))
    # the diffuse captured horizon at 600 steps beside c10h_diff's 1000 (2026-11-01: the cold
    # re-emitted gas may outlive 300 steps), the same seeds and FD (c10_diff's done seeds 0, 4,
    # ..., 28): ~2x c10_diff's 23 GB -> GPU_KIND=h200_71 ONLY='^c10h600_diff'
    jobs |= {
        f"c10h600_diff_s{first}_base": (
            f"{run} --gate c10_diff_h600 --seeds {first} {first + 8} --stride 4 --bases base"
        )
        for first in (0, 8, 16, 24)
    }
    for gate, short in (("c10_spec_gz001", "c10p_gz"), ("c10_spec_two", "c10p_two"),
                        ("c10_spec_v816", "c10p_v816")):  # fmt: skip
        jobs |= _fitgate_blocks(gate, short, 8, ("base",), few)
    # the cell-membership term (paired with c10_spec seeds 0-7 / c10_diff's done seeds 0, 4,
    # ..., 28; FD reused) and the specular circle at 2 dt (16 seeds, the term on): rtx (24 GB)
    # for the specular ones, the diffuse one as c10_diff (~23 GB + the term: rtx_48);
    # GPU_KIND=rtx ONLY='^c10(mem|_2dt)_', GPU_KIND=rtx_48 ONLY='^c10dmem_'
    jobs |= _fitgate_blocks("c10_spec_mem", "c10mem", 8, ("base",), few)
    jobs |= {
        f"c10dmem_s{first}_base": (
            f"{run} --gate c10_diff_mem --seeds {first} {first + 8} --stride 4 --bases base{few}"
        )
        for first in (0, 8, 16, 24)
    }
    jobs |= _fitgate_blocks("c10_spec_2dt", "c10_2dt", 16, ("base",), few)
    # the wall-probe gate (scratch/fitgate/probe_gate.py, 2026-11-01): the fitted grad lambda
    # at the body-hit particles against FD of imposed post-move perturbations (normal push,
    # speed scale, rotation), geometry unchanged; the M 10 specular circle at Kn 0.5 and at
    # Kn 10 (the gate's own control: nearly collisionless, same body and free box -- the
    # squash's channel has reflecting walls, which the probe's hit test cannot tell from the
    # body).  The laptop smoke ran out of host memory after the predictions.
    # GPU_KIND=rtx ONLY='^probe_c10_spec_s', GPU_KIND=rtx_48 ONLY='^probe_c10_spec_kn10'
    probe = "python3 scratch/fitgate/probe_gate.py"
    for gate in ("c10_spec", "c10_spec_kn10"):
        jobs |= {
            f"probe_{gate}_s{first}": f"{probe} --gate {gate} --seeds {first} {first + 4}"
            for first in (0, 4)
        }
    # the collision pull-back in situ (scratch/fitgate/volume_gate.py): the collision volumes
    # of every / the cut cells scaled by 1 +- eps, FD against the backward's prediction; the
    # squash as the low-Mach control.  rtx slices, except the diffuse M 10 circle (> 24 GB):
    # GPU_KIND=rtx ONLY='^vol_(c10_spec|sq)', GPU_KIND=rtx_48 ONLY='^vol_c10_diff'
    vol = "python3 scratch/fitgate/volume_gate.py"
    for gate, seeds in (("c10_spec", 16), ("c10_diff", 8), ("sq_spec", 8)):
        jobs |= {
            f"vol_{gate}_s{first}": f"{vol} --gate {gate} --seeds {first} {first + 4}"
            for first in range(0, seeds, 4)
        }
    # 2026-11-03 (the specular split: the bias is in the collisional part of grad_x lambda at
    # the wall): the c10_spec volume gate on the cells within 1 / 2 lattice layers of the cut
    # cells (120 / 188 of 1112), 32 seeds -- the backward's collision sensitivity where the
    # wall's L* lambda reads it.  GPU_KIND=rtx ONLY='^volw_c10_spec_'
    jobs |= {
        f"volw_c10_spec_s{first}": (
            f"{vol} --gate c10_spec --seeds {first} {first + 4} --sets near1 near2"
        )
        for first in range(0, 32, 4)
    }
    # the same gate's prediction under other collision-field fits, the FD reused (one backward
    # per seed; 2026-11-01: base read pred / FD 1.4 diffuse, 0.94 specular):
    # GPU_KIND=rtx ONLY='^volf_c10_spec', GPU_KIND=h200 ONLY='^volf_c10_diff_(noguard|guard8|cubic|quad)',
    # GPU_KIND=h200_71 ONLY='^volf_c10_diff_(pool1|knn)'
    for gate in ("c10_spec", "c10_diff"):
        for field in ("noguard", "guard8", "cubic", "quad", "pool1", "knn"):
            jobs |= {
                f"volf_{gate}_{field}_s{first}": (
                    f"{vol} --gate {gate} --seeds {first} {first + 4} --field {field} --reuse-fd"
                )
                for first in (0, 4)
            }
    # the same gate under more virtual substeps / partners, FD reused (2026-11-01: every field
    # read the diffuse circle's 1.48, so the shared virtual pull-back's O(rate dt / S)
    # linearisation is next; specular 8 x 16 moved the shape gradient 0 +- 1 %).  Every row
    # also maps the collision rate per particle per step (v4x8: the default, for the maps):
    # GPU_KIND=rtx_48 ONLY='^volp_'  (h200 if rtx_48 is full)
    for gate, virtuals in (("c10_diff", ("4 8", "8 16", "16 8", "32 8")), ("c10_spec", ("4 8",))):
        for virtual in virtuals:
            name = "v" + virtual.replace(" ", "x")
            for first in (0, 2) if virtual != "4 8" else (0,):
                jobs[f"volp_{gate}_{name}_s{first}"] = (
                    f"{vol} --gate {gate} --seeds {first} {first + 2} --virtual {virtual} "
                    "--reuse-fd"
                )
    # the volume gate over a longer captured horizon, the same forward, c10_diff's FD (its FD
    # covers the collisions before the capture, the prediction only those inside it):
    # GPU_KIND=h200_71 ONLY='^volh_c10_diff_h600', GPU_KIND=rtx_full ONLY='^volh_c10_diff_h1000'
    # and the specular one (2026-11-02: does its 0.95 go to 1 as the diffuse 1.48 did?), FD of
    # c10_spec: GPU_KIND=rtx_48 ONLY='^volh_c10_spec_h600', GPU_KIND=h200_71
    # ONLY='^volh_c10_spec_h1000'
    jobs |= {
        f"volh_c10_{wall}_{horizon}_s{first}": (
            f"{vol} --gate c10_{wall}_{horizon} --seeds {first} {first + 2} --reuse-fd"
        )
        for wall in ("diff", "spec")
        for horizon in ("h600", "h1000")
        for first in (0, 2)
    }
    jobs["af_seed4_diag"] = f"python3 scratch/fitgate/seed_outlier.py 4 {out}/af_seed4_diag.npz"
    return jobs


# -- learned collision, round 1 (branch neural-collision; plan 2026-11-11) -----------------
# Run from a checkout of the neural-collision branch.  Every job fits one MIG slice; the
# coarse and training jobs are tiny (<= ~15 k particles per window), the fine references
# 0.4-2 M particles forward-only:
#   GPU_KIND=rtx bash examples/mesh/cluster/submit_curc.sh lcref        (or GPU_KIND=h200)
#   GPU_KIND=rtx_48 ONLY=fine bash examples/mesh/cluster/submit_curc.sh lccirc
#   GPU_KIND=rtx ONLY=coarse bash examples/mesh/cluster/submit_curc.sh lccirc
#   GPU_KIND=rtx bash examples/mesh/cluster/submit_curc.sh lcbase     (after lcref)
#   GPU_KIND=rtx bash examples/mesh/cluster/submit_curc.sh lcsnr      (after lcref)
#   GPU_KIND=rtx bash examples/mesh/cluster/submit_curc.sh lctrain    (after lcbase, lcsnr)
#   GPU_KIND=rtx bash examples/mesh/cluster/submit_curc.sh lceval     (after lctrain)
LC = "python3 -m examples.learned_collision"
LC_REFS = ("shear_a0.1_L32", "shear_a0.1_L128", "shear_a0.5_L32", "shear_a0.5_L128",
           "shock_M1.5", "shock_M3", "shock_M6", "shock_M10")  # fmt: skip
LC_BASE_BLOCKS = {"shear": "S1 S2 S3 S4", "shockA": "K1 K2 K3 K4", "shockB": "K5 K6 K7 K8"}
LC_TEST = "S1 S2 K2 K3 K5 K6 K7"  # K4 joined training 2026-11-12
LC_MODELS = ("scalar", "linear", "mlp")
LC_CIRCLE_BASE = ("ntc", "esfp_recon", "esbgk_recon")


def _lcref_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lcref``: the eight fine 2-D references (examples.learned_collision.refdata)."""
    del gpus_per_node
    return {f"lcref_{ref}": f"{LC}.refdata --ref {ref} --tag v1" for ref in LC_REFS}


LC_CIRCLE_CELLS = ("c8", "c16")
LC_WALLS = ("specular", "diffuse")


def _circle_chain(cells, wall, runs):
    """One job running several coarse circles in turn (``runs``: ``--operator x`` /
    ``--theta y.json`` flags)."""
    return " && ".join(f"{LC}.circle --cells {cells} --wall {wall} {r} --tag v2" for r in runs)


def _lccirc_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lccirc`` (v2, 2026-11-11): the M 10 Kn 0.042 circle on both walls -- fine NTC
    (``ONLY=fine``, ``GPU_KIND=rtx_48`` or ``h200_71``) and the coarse baselines at 8x and 16x
    (``ONLY=coarse``, ``GPU_KIND=rtx``; one job per cell size and wall, the three operators in
    turn)."""
    del gpus_per_node
    jobs = {f"lccirc_fine_{w}": f"{LC}.circle --cells fine --wall {w} --operator ntc --tag v2"
            for w in LC_WALLS}  # fmt: skip
    base = [f"--operator {op}" for op in LC_CIRCLE_BASE]
    for cells in LC_CIRCLE_CELLS:
        for w in LC_WALLS:
            jobs[f"lccirc_coarse_{cells}_{w}"] = _circle_chain(cells, w, base)
    return jobs


def _lcbase_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lcbase``: the 12 cases x 8 baseline operators, forward (three case blocks)."""
    del gpus_per_node
    jobs = {f"lcbase_{name}": f"{LC}.evaluate --cases {cases} --tag base_{name}"
            for name, cases in LC_BASE_BLOCKS.items()}  # fmt: skip
    # (2026-11-11) S1 / S2 at 64 seeds gave +-0.13-0.32 / +-0.05 on mu: the low-amplitude
    # waves' decay is noisy; 512 seeds (S1 ~5 s, S2 ~100 s per operator at 64)
    jobs["lcbase_shear_hi"] = f"{LC}.evaluate --cases S1 S2 --seeds 512 --tag base_shear_hi"
    return jobs


def _lcsnr_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lcsnr``: both window-gradient estimators' SNR, the running bias's noise included
    (independent table fills).  v2 (2026-11-11): S3 S4 K1 K8 at 5 and 10 blocks -- gradients
    resolved, log_pr only on K8.  v3 (2026-11-12): K4 (joined training for a second heat-flux
    case) and S3 (the per-block shear bias table: pair / bias should come back to 2) at the
    training window length."""
    del gpus_per_node
    return {"lcsnr": f"{LC}.grad_snr2d --cases K4 S3 --blocks-list 5 --tag v3"}


def _lctrain_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lctrain``: scalar, linear and MLP corrections on the five training cases."""
    del gpus_per_node
    return {f"lctrain_{m}": f"{LC}.train --model {m} --tag v1" for m in LC_MODELS}


def _lceval_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lceval``: each trained model on the eight held-out cases and the coarse circle."""
    del gpus_per_node
    jobs = {}
    for m in LC_MODELS:
        theta = f"train_{m}_v1.json"
        jobs[f"lceval_{m}"] = (f"{LC}.evaluate --cases {LC_TEST} --operators --theta {theta} "
                               f"--seeds 512 --tag test_{m}")  # fmt: skip
        jobs[f"lceval_circle_{m}"] = " && ".join(
            _circle_chain(c, w, [f"--theta {theta}"]) for c in LC_CIRCLE_CELLS for w in LC_WALLS
        )
    return jobs


def _lctime_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lctime``: brief training for the sweep's timings -- the linear model on the five
    training cases with every case validated forward (examples.learned_collision.sweep reads
    train_linear_lctime.json), and the largest sweep network's iteration time.  2026-11-12:
    40 iterations at the fan-in-normalised default lr with features_v2 (after lcfeat) -- also
    the check that the loss now falls (v1's 6 steps of plain Adam at 0.02 overshot to mu
    0.5x).  Tags lctime3 (2026-11-12): the rate-basis correction ``(dlog r_s, dlog r_q)``; ran
    with a tenth ``bound`` feature (features_v3) since dropped -- its timings size the sweep
    (one input more or less is immaterial), its training result is not the 9-feature model's."""
    del gpus_per_node
    common = "--iters 40 --tag"
    return {"lctime_linear": f"{LC}.train --model linear --validate-train {common} lctime3",
            "lctime_mlp": f"{LC}.train --model mlp --depth 3 --width 128 {common} lctime3_mlp"}  # fmt: skip


def _lcburn_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lcburn`` (2026-11-12): does the window objective agree with the steady shocks' heat
    flux?  lctime (burn 2 blocks, ~2 mean free times before a 5-block window) lowered the shock
    stress errors but raised the heat-flux ones on all three shocks (int qx 0.97 -> 0.87 on
    K4, 0.94 -> 0.82 on K8): short windows from fine snapshots see the coarse operator acting
    on a fine profile, not on its own.  The same 40 iterations with 10 and 30 blocks of
    forward-only burn-in (the window then starts from the coarse model's own relaxed state)."""
    del gpus_per_node
    return {f"lcburn_{b}": f"{LC}.train --model linear --validate-train --iters 40 --burn {b} "
            f"--tag lcburn{b}" for b in (10, 30)}  # fmt: skip


def _lcsteadyfd_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lcsteadyfd`` (2026-11-12): central differences of the steady shock errors in
    ``log_tau`` / ``log_pr`` at the base operator (examples.learned_collision.steady_fd) --
    the steady side of the window objective's heat-flux disagreement (lcburn)."""
    del gpus_per_node
    return {"lcsteadyfd": f"{LC}.steady_fd --tag v1"}


LC_LOSSES = {
    "ctl": "",
    "q3": "--qx-weight 3",
    "q10": "--qx-weight 10",
    "m1": "--shock-modes 1",
    "m2": "--shock-modes 2",
    "m4": "--shock-modes 4",
    "m2q3": "--shock-modes 2 --qx-weight 3",
}
"""``lcloss`` variants of the shock loss (train.py ``--shock-modes`` / ``--qx-weight``)."""


def _lcloss_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lcloss`` (2026-11-12): which shock loss stops training from trading heat flux for
    stress?  lctime3's per-cell loss lowered int tau_xx (K8 1.17 -> 1.07) but also int q_x
    (0.95 -> 0.87, the heat-flux rate sped up), while the steady integrals' 2x2 Jacobian in
    (log r_s, log r_q) (lcsteadyfd, condition 1.1-3.5) asks for a faster stress rate and a
    *slower* heat-flux rate.  Two hypotheses, seven linear runs of lctime's size (40
    iterations, forward validation on the five training cases): profile shape errors the
    rates cannot fix drive the loss (``m*``: the low cosine modes, ``m1`` the integrals
    alone), or qx is under-weighted (``q*``); ``ctl`` the per-cell loss on the 9-feature
    rate-basis model.  Pass: int tau_xx and int q_x both closer to 1 on K4 and K8, K1 and
    S3/S4 mu no worse than the base."""
    del gpus_per_node
    return {f"lcloss_{k}": f"{LC}.train --model linear --validate-train --iters 40 {v} "
            f"--tag lcloss_{k}".replace("  ", " ") for k, v in LC_LOSSES.items()}  # fmt: skip


def _lctransient_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lctransient`` (2026-11-12): every lcloss variant (qx x10 included, where the steady
    sensitivities ask for a slower heat-flux rate on every shock) trained the heat-flux rate
    up -- are the 2-block-burn windows still relaxing from the fine snapshot?  Forward-only
    integrals per block from the training starts (examples.learned_collision.transient), and
    the same lcloss runs with a burn-in past the shocks' relaxation (120 blocks; eval relaxes
    78-126): does the heat-flux row turn negative once the windows see the steady state?"""
    del gpus_per_node
    jobs = {"lctransient": f"{LC}.transient --tag v1"}
    for k in ("ctl", "q3", "m1"):
        jobs[f"lcburnlong_{k}"] = (f"{LC}.train --model linear --validate-train --iters 40 "
                                   f"--shock-burn 120 {LC_LOSSES[k]} --tag lcburnlong_{k}").replace(
                                       "  ", " ")  # fmt: skip
    return jobs


def _lcwarm_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lcwarm`` (2026-11-12): the warm pools of the training shocks' persistent chains
    (examples.learned_collision.warm_pool: 128 coarse runs per case from the training
    starts, relaxed 160 blocks at the base operator -- lctransient's plateau).  Minutes; run
    before lcchain / lccv."""
    del gpus_per_node
    return {"lcwarm": f"{LC}.warm_pool --blocks 160 --tag v1"}


def _lcchain_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lcchain`` (2026-11-12, after lcwarm): lcloss again with the shock windows on
    persistent chains at the model's own steady state (train.py --warm-pool).  The test is
    ``q10``: at x10 the steady sensitivities ask for a slower heat-flux rate on every shock
    from the first step, which the snapshot windows trained the other way.  ``ctl_long``: the
    per-cell loss 3x as many iterations at lr 0.01 (~10x the movement) -- past the K8 path's
    heat-flux flip (~0.04 of movement), does int q_x recover?  Pass as lcloss: int tau_xx
    and int q_x closer to 1 on K4 and K8, K1 and S3/S4 mu no worse than the base."""
    del gpus_per_node
    chain = "--warm-pool v1 --model linear --validate-train"
    jobs = {f"lcchain_{k}": f"{LC}.train {chain} --iters 40 {LC_LOSSES.get(k, '')} "
            f"--tag lcchain_{k}".replace("  ", " ") for k in ("ctl", "q3", "q10", "m1", "m2q3")}  # fmt: skip
    jobs["lcchain_ctl_long"] = f"{LC}.train {chain} --iters 120 --lr 0.01 --tag lcchain_ctl_long"
    return jobs


def _lcconv_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lcconv`` (2026-11-12, after lcchain): converged runs on the chains before lccv.  At
    40 iterations the qx weight moved the model along a frontier (ctl: K8 int txx 1.105 /
    int qx 0.901; q3: 1.183 / 0.972; q10: 1.228 / 1.009) and nothing passed; ctl_long (120
    iterations at lr 0.01) fixed K8's stress but over-shot K4's heat flux (1.076) -- the shared
    coefficients serve K4 and K8 differently.  Do 300 iterations at lccv's middle lr (the
    linear model at qx weights 1, 2, 3; the smallest sweep-like MLP at 2) reach a model that
    corrects both integrals on both shocks?  Pass as lcloss."""
    del gpus_per_node
    common = f"{LC}.train --warm-pool v1 --validate-train --iters 300 --lr 0.003"
    jobs = {f"lcconv_q{w}": f"{common} --model linear --qx-weight {w} --tag lcconv_q{w}"
            for w in (1, 2, 3)}  # fmt: skip
    jobs["lcconv_mlp_q2"] = (f"{common} --model mlp --depth 2 --width 32 --qx-weight 2 "
                             "--tag lcconv_mlp_q2")  # fmt: skip
    return jobs


def _lcbound_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lcbound`` (2026-11-12): where ES-FP's positive-definiteness bound (Mathiaud &
    Mieussens eq. 35; at Pr = 2/3 it acts past ``pd_max = 0.8 theta``) slows the stress
    relaxation in the coarse ES-FP+recon steady shocks -- the active fraction per cell, its
    ``|tau_xx|``-weighted share and the mean slowdown, all eight shocks (examples
    .learned_collision.evaluate --bound).  Does it track the operator's Mach-growing int txx
    excess (1.05 at M1.5 to 1.17 at M10)?"""
    del gpus_per_node
    common = f"{LC}.evaluate --operators esfp_recon --bound --starts 32"
    return {
        "lcbound_lo": f"{common} --cases K1 K2 K3 K4 --tag bound_lo",
        "lcbound_hi": f"{common} --cases K5 K6 K7 K8 --tag bound_hi",
    }


def _lcfeat_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lcfeat``: the features' training spread and correlations (features_v2.json, the five
    training cases; the training, the sweep and the Lasso standardise by it -- run before
    them).  (v3, 2026-11-12, added a tenth ``bound`` feature, dropped after lcbound.)"""
    del gpus_per_node
    return {"lcfeat": f"{LC}.features --tag v2"}


def _lclasso_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lclasso``: the Gauss-Newton Lasso path of the linear correction, CV-chosen penalty and
    stability selection -- free signs, and with the cell-size / time-step terms of
    dlog_tau held <= 0."""
    del gpus_per_node
    return {
        "lclasso_free": f"{LC}.lasso --tag free",
        "lclasso_signs": f"{LC}.lasso --signs --tag signs",
    }


def _lccv_jobs(gpus_per_node: int) -> dict[str, str]:
    """``lccv``: the MLP orthogonal-array sweep, cross-validated (examples.learned_collision
    .sweep -- run it on the login node first: it writes the plan this group reads, sized to
    the GPU-hour budget from lctime's timings).  One job per chain of runs."""
    del gpus_per_node
    path = Path(__file__).resolve().parents[3] / "examples/learned_collision/results"
    plan_path = path / "sweep_plan_v2.json"
    if not plan_path.exists():
        raise SystemExit(f"no {plan_path}: run `python3 -m examples.learned_collision.sweep`")
    plan = json.loads(plan_path.read_text())
    jobs = {}
    for j, members in enumerate(plan["jobs"]):
        cmds = []
        for i in members:
            r = plan["runs"][i]
            val = f"--validate {' '.join(r['validate'])} " if r["validate"] else ""
            cmds.append(
                f"{LC}.train --model mlp --depth {r['depth']} --width {r['width']} "
                f"--weight-decay {r['weight_decay']} --lr {r['lr']} --iters {plan['iters']} "
                f"--cases {' '.join(r['cases'])} {val}--validate-train --seed {r['seed']} "
                f"--tag {r['tag']} {plan.get('train_args', '')}".rstrip()
            )
        jobs[f"lccv_{j}"] = " ; ".join(cmds)
    return jobs


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
    "fitgate": _fitgate_jobs,
    "optrc": _optrc_jobs,
    "ldopt": _ldopt_jobs,
    "sph3d": _sph3d_jobs,
    "lsfilter": _lsfilter_jobs,
    "lsfilter3d": _lsfilter3d_jobs,
    "ld3dwide": _ld3dwide_jobs,
    "dlh3d": _dlh3d_jobs,
    "diffchk": _diffchk_jobs,
    "m2chk": _m2chk_jobs,
    "m2split": _m2split_jobs,
    "silfix": _silfix_jobs,
    "bitchk": _bitchk_jobs,
    "ellchk": _ellchk_jobs,
    "ellprobe": _ellprobe_jobs,
    "dlhe": _dlhe_jobs,
    "dlhp": _dlhp_jobs,
    "multichk": _multichk_jobs,
    "termchk": _termchk_jobs,
    "termvol": _termvol_jobs,
    "bumpchk": _bumpchk_jobs,
    "fitlab": _fitlab_jobs,
    "fitlab2": _fitlab2_jobs,
    "fitlab3": _fitlab3_jobs,
    "shiftfd": _shiftfd_jobs,
    "optrules": _optrules_jobs,
    "optrules2": _optrules2_jobs,
    "optrules1seed": _optrules1seed_jobs,
    "optax1seed": _optax1seed_jobs,
    "optcm": _optcm_jobs,
    "optct": functools.partial(_optcm_jobs, mark="ct", extra=" --start-shift 2.0"),
    "squashkn2": _squashkn2_jobs,
    "optckn": functools.partial(_keepout_jobs, collisional=True),
    "optc2": functools.partial(_keepout_jobs, collisional=False),
    "lcref": _lcref_jobs,
    "lccirc": _lccirc_jobs,
    "lcbase": _lcbase_jobs,
    "lcsnr": _lcsnr_jobs,
    "lctrain": _lctrain_jobs,
    "lceval": _lceval_jobs,
    "lctime": _lctime_jobs,
    "lcfeat": _lcfeat_jobs,
    "lcburn": _lcburn_jobs,
    "lcsteadyfd": _lcsteadyfd_jobs,
    "lcbound": _lcbound_jobs,
    "lcloss": _lcloss_jobs,
    "lctransient": _lctransient_jobs,
    "lcwarm": _lcwarm_jobs,
    "lcchain": _lcchain_jobs,
    "lcconv": _lcconv_jobs,
    "lclasso": _lclasso_jobs,
    "lccv": _lccv_jobs,
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
