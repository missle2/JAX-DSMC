"""The cluster runs of the paper "Shape optimization with adjoint DSMC" (2026-10-28), one group
per set of figures.  Submitted through the mesh study's script with this module named:

    STUDY_MODULE=examples.paper.cluster.study GPU_KIND=rtx \
        bash examples/mesh/cluster/submit_curc.sh paper_fd

Every optimisation uses the paper's method: the Helmholtz-filtered level set without
reinitialisation (``--filter helmholtz --reinit-every 0``), Adam (``ox_adam``) on a cosine step
decay, one seed per iterate warm-started from the previous iterate's gas, the patience stopping
rule, the one-sided backward fits and the grazing floor (``--backward-normals true --grazing
0.01``).  Knudsen numbers are ours (the VHS mean free path over the case's 2 R; the BGK papers'
Kn is 1.296 x ours for argon, examples/mesh/forward.py).

Groups (figure numbers of docs/paper/outline.md):

* ``paper_fd`` (Appendix A, gate G2): adjoint against paired central differences (8 seeds,
  ``examples.ensemble_mesh``) on the paper's conditions -- the Ma 2 airfoil box (diffuse,
  T_w = T_inf; drag and the body's heat rate, the heat path's gate), the Newton box at M 10
  (Kn 0.5 diffuse drag and heat; free-molecular specular drag).  rtx / h200_71.
* ``paper_fwd`` (Fig. 8, gate G3): NACA 0012 at Ma 2 and the three Knudsen numbers, 8 seeds
  of one iterate each (the drag the BGK papers report for their start).  rtx.
* ``paper_airfoil`` (Fig. 8): minimum drag at Ma 2, diffuse, T_w = T_inf, fixed chord (design
  region a 6 m slab), area at least NACA 0012's, from the ellipse of that area and chord, at
  the BGK Kn 0.2 / 0.5 / 1 (ours 0.154 / 0.386 / 0.772).  rtx.
* ``paper_newton`` (Fig. 9): Newton's problem in free-molecular flow -- specular, M 10, Kn 10
  (collisions on), the body inside the cylinder of radius 1 and height H (H / R = 2 and 1)
  containing its base disk, minimum drag, from the unit sphere.  h200_71.
* ``paper_heat`` (Fig. 13): the same cylinder (H / R = 2) at Kn 0.5 M 10 with diffuse walls,
  minimum drag uncapped and with the peak heating capped at 0.8 and 0.6 of the start body's
  (``--heat-cap-relative 1``, gate G1: examples/paper/reference/newton_fm.py).  h200_71.
* ``paper_sph`` (Fig. 10): Zhang, Yuan & Wu 2025's sphere at Ma 2, their Kn 0.1 (ours 0.077),
  diffuse at T_inf, volume at least the sphere's: ``pilot`` (2 iterates: time and memory,
  gate G4) then ``full``.  h200_full or a 4 x A100 node.
* ``paper_ld2d`` (Fig. 11): the lsfilter M 10 L / D run (Helmholtz, Adam; L / D 3.19 at 250,
  still rising) resumed to 400 iterates under its own tag.  rtx.

Fig. 12 (3-D L / D) is examples/mesh/cluster's ``ld3dwide``; Fig. 5 (cost) is read from the
runs' own ``phase_seconds``.
"""

from __future__ import annotations

import argparse

ITERATE = (
    "--seeds-per-iterate 1 --workers 1 --restart-steps 300 --converge-patience 100 "
    "--backward-normals true --grazing 0.01 --substeps 4 --partners 8 --filter helmholtz "
    "--reinit-every 0 --optimizer ox_adam --save-gradient 1 --frames-every 25"
)
# (BGK Kn, ours): ours = BGK / 1.296 (examples/mesh/forward.py, the airfoil case)
AIRFOIL_KN = {"0.2": 0.154, "0.5": 0.386, "1.0": 0.772}
# diffuse bodies: the hit <-> miss (silhouette) term is about half their derivative
# (docs/math/shape_derivation_body.tex, term 4); off for specular ones (O(band) there)
DIFFUSE = "--acc 1 --silhouette-band 0.03"
AIRFOIL_FLOW = f"--case airfoil {DIFFUSE} --t-wall 273.15 --mach 2 --dt 1e-4 --boundary free"
AIRFOIL_BODY = "--levelset-h 0.1 --levelset-samples 16 --smooth 3 --band 0.4 --keepout 0.3"
AIRFOIL_ITERATIONS = 200
M10 = "--mach 10 --dt 2.5e-5 --boundary free --a-max 16"
NEWTON_BODY = (
    "--case newton --levelset-h 0.1 --levelset-samples 16 --smooth 3 --band 0.4 --keepout 0.3 "
    "--fnum-scale 1 --warmup 1000"
)
# H / R -> (design cylinder, base slab two knots deep)
NEWTON_REGIONS = {
    "h2": "--design-region cylinder:-1_1_1 --solid-region cylinder:0.8_1_1",
    "h1": "--design-region cylinder:0_1_1 --solid-region cylinder:0.8_1_1",
}
NEWTON_ITERATIONS = 300
HEAT_CAPS = {"cap0": "", "cap0.8": 0.8, "cap0.6": 0.6}
SPH_ITERATIONS = 150


def _optimize(tag: str, *flags: str) -> str:
    return " ".join(("python3 -m examples.mesh.optimize --tag", tag, *flags))


def _fd(name: str, *flags: str) -> str:
    return " ".join((
        "python3 -m examples.ensemble_mesh --body levelset --seeds 8 --substeps 4 --partners 8",
        "--backward-normals true --grazing 0.01 --timing-band 10",
        f"--output examples/paper/results/fd/{name}.json", *flags,
    ))  # fmt: skip


def _fd_jobs(gpus_per_node: int) -> dict[str, str]:
    del gpus_per_node
    airfoil = f"{AIRFOIL_FLOW} --knudsen 0.386 --warmup 1500 --levelset-h 0.1"
    newton = f"--case newton --levelset-h 0.1 --fnum-scale 1 --warmup 1000 {M10}"
    return {
        "af_drag_radius": _fd("af_drag_radius", airfoil, "--parameter radius --fd-steps 0.1 0.05"),
        "af_drag_mode2": _fd("af_drag_mode2", airfoil, "--parameter mode2 --fd-steps 0.1 0.05"),
        "af_heat_radius": _fd("af_heat_radius", airfoil, "--objective heat --heat-patch 0.6",
                              "--parameter radius --fd-steps 0.1 0.05"),
        "nw_drag_radius": _fd("nw_drag_radius", newton, f"{DIFFUSE} --knudsen 0.5",
                              "--parameter radius --fd-steps 0.05 0.02"),
        "nw_heat_radius": _fd("nw_heat_radius", newton, f"{DIFFUSE} --knudsen 0.5 --objective heat",
                              "--heat-patch 0.2 --parameter radius --fd-steps 0.05 0.02"),
        "nw_fm_radius": _fd("nw_fm_radius", newton, "--knudsen 10",
                            "--parameter radius --fd-steps 0.05 0.02"),
    }  # fmt: skip


def _fwd_jobs(gpus_per_node: int) -> dict[str, str]:
    del gpus_per_node
    return {
        f"naca_kn{bgk}": _optimize(
            f"paper_fwd_naca0012_kn{bgk}", AIRFOIL_FLOW, AIRFOIL_BODY, f"--knudsen {ours}",
            "--initial naca0012 --initial-thickness 0.12 --warmup 1500 --iterations 1",
            "--seeds-per-iterate 8 --workers 1 --frames-every 0 --backward-normals true",
            "--grazing 0.01 --substeps 4 --partners 8 --filter helmholtz --reinit-every 0",
            "--optimizer ox_adam --area-mode none --step 0.0",
        )
        for bgk, ours in AIRFOIL_KN.items()
    }


def _airfoil_jobs(gpus_per_node: int) -> dict[str, str]:
    del gpus_per_node
    return {
        f"kn{bgk}": _optimize(
            f"paper_airfoil_kn{bgk}", AIRFOIL_FLOW, AIRFOIL_BODY, f"--knudsen {ours}", ITERATE,
            "--initial ellipse --initial-thickness 0.1040 --design-region box:-3_3_-inf_inf",
            "--area-mode floor --area-floor 1.0 --recentre 0 --warmup 1500 --step 0.01",
            f"--iterations {AIRFOIL_ITERATIONS} --step-decay {AIRFOIL_ITERATIONS}",
        )
        for bgk, ours in AIRFOIL_KN.items()
    }


def _newton_jobs(gpus_per_node: int) -> dict[str, str]:
    del gpus_per_node
    return {
        f"fm_{name}": _optimize(
            f"paper_newton_fm_{name}", NEWTON_BODY, M10, "--knudsen 10 --acc 0", regions,
            ITERATE, "--area-mode none --recentre 0 --step 0.02",
            f"--iterations {NEWTON_ITERATIONS} --step-decay {NEWTON_ITERATIONS}",
        )
        for name, regions in NEWTON_REGIONS.items()
    }


def _heat_jobs(gpus_per_node: int) -> dict[str, str]:
    del gpus_per_node
    out = {}
    for name, cap in HEAT_CAPS.items():
        heat = "--heat-patch 0.2"
        if cap:
            heat += f" --heat-cap {cap} --heat-cap-relative 1 --heat-multiplier 1 --heat-rate 2"
        out[name] = _optimize(
            f"paper_heat_kn0.5_M10_{name}", NEWTON_BODY, M10, f"--knudsen 0.5 {DIFFUSE}",
            NEWTON_REGIONS["h2"], ITERATE, heat, "--area-mode none --recentre 0 --step 0.02",
            f"--iterations {NEWTON_ITERATIONS} --step-decay {NEWTON_ITERATIONS}",
        )
    return out


def _sph_jobs(gpus_per_node: int) -> dict[str, str]:
    del gpus_per_node
    flow = f"--case sphere_ma2 --mach 2 --dt 1e-4 --boundary free {DIFFUSE} --t-wall 273.15"
    body = "--levelset-h 0.1 --levelset-samples 16 --smooth 3 --band 0.4 --keepout 0.3"
    common = (flow, body, "--knudsen 0.077 --fnum-scale 1 --warmup 1500 --a-max 16", ITERATE,
              "--area-mode floor --area-floor 1.0 --recentre 1 --step 0.02")  # fmt: skip
    return {
        "pilot": _optimize("paper_sph_pilot", *common, "--iterations 2"),
        "full": _optimize("paper_sph_kn0.1", *common,
                          f"--iterations {SPH_ITERATIONS} --step-decay {SPH_ITERATIONS}"),
    }  # fmt: skip


def _ld2d_jobs(gpus_per_node: int) -> dict[str, str]:
    """The lsfilter M 10 L / D Helmholtz Adam run (examples/mesh/cluster/study.py, group
    lsfilter, job kn0.5_M10_S1_ox_adam_ldhz0) to 400 iterates: the same command, resumed."""
    from examples.mesh.cluster import study as mesh_study  # noqa: PLC0415

    cmd = mesh_study.jobs("lsfilter", gpus_per_node)["kn0.5_M10_S1_ox_adam_ldhz0"]
    old = f"--iterations {mesh_study.LSFILTER_SLOW_ITERATIONS}"
    if old not in cmd:
        raise ValueError(f"the lsfilter command changed: {cmd}")
    return {"m10_ld": cmd.replace(old, "--iterations 400", 1)}


GROUPS = {
    "paper_fd": _fd_jobs,
    "paper_fwd": _fwd_jobs,
    "paper_airfoil": _airfoil_jobs,
    "paper_newton": _newton_jobs,
    "paper_heat": _heat_jobs,
    "paper_sph": _sph_jobs,
    "paper_ld2d": _ld2d_jobs,
}


def jobs(group: str, gpus_per_node: int = 1) -> dict[str, str]:
    return GROUPS[group](gpus_per_node)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", choices=list(GROUPS), required=True)
    ap.add_argument("--list", action="store_true", help="print the job commands, one per line")
    ap.add_argument("--names", action="store_true", help="print the job names")
    ap.add_argument("--gpus", action="store_true", help="print `name gpus` per job")
    ap.add_argument("--gpus-per-node", type=int, default=1)
    args = ap.parse_args(argv)
    for name, cmd in jobs(args.group, args.gpus_per_node).items():
        if args.names:
            print(name)
        elif args.gpus:
            print(name, 1)
        else:
            print(cmd)


if __name__ == "__main__":
    main()
