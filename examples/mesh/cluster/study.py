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
* ``guardgate`` (2026-10-18): the leverage-guarded cubic (``cell:cubicv:...:guard<h>``, h = 2 /
  8 / 32) against the shipped cubic and the quadratic, ``bash scratch/guardgate/cluster_job.sh
  <name>`` (results in ``scratch/guardgate/cluster/``): the blunt Kn 0.5 squash at s = 0.6
  (``sq_M032_s0.6_b*``, 16 seeds), thingate's thin M 10 squash (``sq_M10_*``), optm iterates
  30 / 58 / 156 / 246 (``om_it*``, one H200 each) and the stored device gates (``unit``).
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
# ~ 8x the measured 21 behind the shock, so A_max 256 at Kn = 0.5.  Both limits fail loudly
# when exceeded (attempts_clipped / inflow_shortfall stop the job) -- nothing regrows them
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
                    f"--band 1.0 {PULLBACK}"
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
