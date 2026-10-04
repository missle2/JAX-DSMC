"""The cluster study's job matrix (plan phase E, 2026-09-21), one group per task.

    python3 -m examples.mesh.cluster.study --group sphere --list   # one command per seed
    python3 -m examples.mesh.cluster.study --group mode2 --list    # one command per seed
    python3 -m examples.mesh.cluster.study --group opt --list      # one command per (S, opt)
    python3 -m examples.mesh.cluster.study --group opt --gpus      # GPUs per opt job

Every command is a plain ``python3 -m examples....`` invocation that resumes from its own
artifact (rows / iterates already done are skipped), so a job can be re-submitted after a
time limit.  The groups (``submit_curc.sh`` sizes one SLURM job per line):

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
  ``s* = 0.35``, 25 iterates: the optimiser on DSMC gradients against the analytic
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
* ``opt``: ``opt_S{S}_{gd|adam}``, the finer circle's drag minimisation at fixed area, 250
  iterates (``--iterations 250``, resumed across jobs), ``S`` CRN seeds per iterate for
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
SQUASH_ITERATIONS = 25
# the Kn = 0.5 settings (measured 2026-10-01: at most 21 NTC attempts per cell at s = 1)
KN05 = "--knudsen 0.5 --a-max 64"
# M = 10 (3079 m/s), scaled from the Kn = 0.5 measurement rather than measured: the step
# keeps a free-stream move at 0.15 cell; the slots hold the full box (1e5 at this fnum) plus
# a shock layer at ~4x density; NTC attempts go as n^2 (sigma g)_max dt, ~16 x 1.9 x 1/4
# ~ 8x the measured 21 behind the shock, so A_max 256 at Kn = 0.5.  Both limits fail loudly
# when exceeded (attempts_clipped / inflow_shortfall stop the job) -- nothing regrows them
MACH10 = "--mach 10 --dt 2.5e-5 --slots 160000"
KN05_M10 = f"--knudsen 0.5 {MACH10} --a-max 256"
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


def jobs(group: str, gpus_per_node: int = GPUS_PER_NODE) -> dict[str, str]:
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
                f"--iterations {SQUASH_ITERATIONS} {PULLBACK}"
            )
        return out
    if group == "squashkn":
        out = {}
        for label, extra in (("kn0.5", KN05), ("kn0.5_M10", KN05_M10)):
            for s in SQUASH_KN_SEEDS:
                out[f"{label}_S{s}"] = (
                    f"python3 -m examples.mesh.optimize_squash --seeds-per-iterate {s} "
                    f"--iterations {SQUASH_ITERATIONS} {extra}"
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
                    f"--optimizer {opt} --iterations 250 --seeds-per-iterate {s} "
                    f"--workers {min(s, gpus_per_node)} {settings} --step 0.05 --smooth 2 "
                    f"--band 1.0 {PULLBACK}"
                )
        return out
    raise ValueError(group)


def gpus(group: str, gpus_per_node: int = GPUS_PER_NODE) -> dict[str, int]:
    """GPUs per job: one per seed job; ``min(S, gpus_per_node)`` per optimisation."""
    if group not in OPT_GROUPS:
        return dict.fromkeys(jobs(group), 1)
    return {f"opt_S{s}_{opt}": min(s, gpus_per_node) for s in SEED_COUNTS for opt in OPTIMIZERS}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--group",
        choices=("sphere", "mode2", "squash", "squashkn", "biasgate", *OPT_GROUPS),
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
