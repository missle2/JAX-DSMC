"""Gate for the M 10 optimisations in the long box (2026-10-15): does the starting body choke the
channel?  One uncaptured forward of the optimiser's first iterate (the initial level set) with
ample slots, then: particles alive against the production slot count, the inflow diagnostics
(the run stops on any shortfall / clipping), and the number density along x (|y - 5| < 1 m and
over the full height) against the free stream -- a bow shock that stands near the body is
fine, one that reaches the inflow face (x = -10 m) is the choked channel.

    run_locked.sh gpu env PYTHONPATH=. python3 scratch/optm10/m10_gate.py --radius 3.0 \
        [optimize args, e.g. --knudsen 0.5 --mach 10 --dt 2.5e-5 --a-max 256]

``--case`` picks the box (default circle_long), ``--radius`` overrides the case's starting
circle (circle_long: 3 m in a 10 m channel); GATE_SLOTS the slot count (default 400 k).

Results (2026-10-15, n / n_inf against the deck's free stream, 500 particles / m^2 at fnum x2):
free-molecular M 10, R 3: 170 k alive, 1.9 n_inf ahead of the body over the whole upstream (the
front's specular reflections stream back upstream -- physical in free-molecular flow, not a
reference error).  Kn 0.5 + M 10: R 3 and R 1.5 choke the channel (the specular y faces make it
a cascade of bodies 10 m apart): a channel-wide region at the strong-shock ratio ~4.4 grows
upstream -- with 30 m of upstream box (a temporary circle_xlong case, x from -25 m) it spanned
x -14..0 m and overflowed 600 k slots.  R 1 (D/H 0.2): started, 1.01 n_inf at the inflow face,
a thick standing shock over the 8 m ahead of the body, 249 k alive.
"""

import json
import os
import sys
import time

import jax
import numpy as np

sys.path.insert(0, "examples/mesh")
import forward as FWD  # noqa: E402
import optimize as OPT  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402

FIELDS = ("inflow_shortfall", "inflow_storage_overflow", "inflow_candidate_overflow",
          "attempts_clipped", "surface_stuck")  # fmt: skip
SLOTS = int(os.environ.get("GATE_SLOTS", 400_000))


def main():
    args = sys.argv[1:]
    radius = None
    case_name = "circle_long"
    if "--case" in args:
        i = args.index("--case")
        case_name = args[i + 1]
        del args[i : i + 2]
    if "--radius" in args:
        i = args.index("--radius")
        radius = float(args[i + 1])
        del args[i : i + 2]
    cfg = OPT.default_config()
    cfg.update(case=case_name, fnum_scale=2.0, warmup=2800, levelset_h=0.0625,
               levelset_samples=16, grazing=0.05)  # fmt: skip
    for flag, val in zip(args[::2], args[1::2], strict=True):
        k = flag[2:].replace("-", "_")
        cfg[k] = type(cfg[k])(val)
    cfg["slots"] = SLOTS
    if radius is not None:
        FWD.CASES[case_name] = dict(FWD.CASES[case_name], radius=radius)
    _grid, ls = OPT.initial_levelset(cfg)
    t0 = time.time()
    _, final, diag, _, _, s = EM.forward(
        cfg, "radius", EM.nominal(cfg, "radius"), jax.random.key(cfg["seed_start"]), False, ls
    )
    alive = np.asarray(final.alive)
    x = np.asarray(final.x)[alive]
    grid = s["grid"]
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    case = FWD.CASES[case_name]
    # free stream: particles per m^2 (density and fnum scale alike with the Knudsen number)
    per_m2 = FWD.NRHO / (case["fnum"] * cfg["fnum_scale"])
    height = hi[1] - lo[1]
    empty = per_m2 * height * (hi[0] - lo[0])
    edges = np.arange(lo[0], hi[0] + 1e-9, 1.0)
    full, _ = np.histogram(x[:, 0], edges)
    core, _ = np.histogram(x[np.abs(x[:, 1] - 5.0) < 1.0, 0], edges)
    ref_full = per_m2 * height * 1.0
    ref_core = per_m2 * 2.0 * 1.0
    out = {
        "cfg": {k: cfg[k] for k in ("knudsen", "mach", "dt", "a_max", "warmup", "steps")},
        "case": case_name,
        "radius": radius if radius is not None else case["radius"],
        "alive": int(alive.sum()),
        "empty_box": empty,
        "mach10_slots": 160_000,
        "seconds": time.time() - t0,
        "diag": {f: float(np.asarray(getattr(diag, f)).sum()) for f in FIELDS},
        "x": edges[:-1].tolist(),
        "density_full": (full / ref_full).tolist(),
        "density_core": (core / ref_core).tolist(),
    }
    print(
        f"radius {out['radius']}: alive {out['alive']} = {out['alive'] / empty:.2f}x the empty box "
        f"({empty:.0f}), {out['alive'] / 160_000:.2f}x MACH10's 160 k slots; "
        f"{out['seconds']:.0f} s; {out['diag']}"
    )
    for xe, f, c in zip(edges[:-1], out["density_full"], out["density_core"], strict=True):
        if xe < 4.0:  # upstream of the body's centre
            print(f"  x {xe:5.1f}  n/n_inf full height {f:5.2f}  |y-5|<1 {c:5.2f}")
    tag = "_".join(f"{a.lstrip('-')}{b}" for a, b in zip(args[::2], args[1::2], strict=True))
    with open(f"scratch/optm10/gate_{case_name}_r{out['radius']:g}_{tag}.json", "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
