"""Does the drag depend on where the body sits in the box (2026-10-24)?  Every free Adam-type
Kn 0.5 M 10 run slid ~10 m downstream, and holding the centroid by projection roughened the
bodies (scratch/optax/constraint_replay.py).  A body in a uniform stream has no position
gradient; in the box the inflow face ~4 mean free paths ahead of the nose may make one.

Per shape, on SEEDS seeds: the adjoint's dJ/dx (the gradient along a rigid x-translation,
optimize.translation_modes) and a paired finite difference (the same seeds with the body moved
SHIFT m downstream), as fractions of J per metre.  Shapes: the initial circle and the free M 10
ox_adam body at iterate 100 (sharp nose ~2 m ahead of the start), resampled onto this box.
Run once in circle_long and once in circle_xlong (inflow face 15 m further upstream).

    python3 scratch/boxcheck/translation.py --name <label> <optimize.py flags>
        -> scratch/boxcheck/cluster/<label>.json

adjfix (2026-10-24) reuses it with --seeds 8 and the candidate fix (--backward-normals true,
--grazing 0.01) under adjfix_<label> names.
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

from jax_dsmc import ensemble_backward  # noqa: E402

SEEDS = 6
SHIFT = 1.0
SOURCE = ("circle_long_kn0.5_M10_S1_ox_adam_tan_sm8_free_g32_w500", 100, "circle_long")


def parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--seeds", type=int, default=SEEDS)
    cfg = O.default_config()
    for k, v in cfg.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=O.cli_type(v), default=v)
    p.add_argument("--substeps", type=int, default=1)
    p.add_argument("--partners", type=int, default=2)
    args = p.parse_args(argv)
    return args, {k: getattr(args, k) for k in cfg}


def resampled(cfg, ls):
    """The source run's body (coefficients on its own box's grid) on this box's grid."""
    tag, k, case = SOURCE
    _g, src0 = O.initial_levelset(dict(cfg, case=case))
    src = dataclasses.replace(src0, coef=jnp.asarray(np.load(f"{O.RESULTS}/{tag}/coef_{k}.npy")))
    values = O.L.value(src, O.L.knots(ls)).reshape(ls.coef.shape)
    return O.L.reinitialise(dataclasses.replace(ls, coef=O.L.interpolate(values)))


def measure(cfg, ls, lo, hi, n_seeds):
    seeds = list(range(n_seeds))
    tx = O.translation_modes(ls, cfg["band"] + ls.h)[0]
    base = O.seed_terms(cfg, ls, seeds)
    moved = O.seed_terms(cfg, O.translate(ls, np.array([SHIFT, 0.0])), seeds)
    j0, j1 = base["objective"], moved["objective"]
    adj = np.array([np.sum(g * tx) for g in base["gradient"]]) / j0  # per seed, per metre
    fd = (j1 - j0) / SHIFT / j0
    c = np.asarray(O._centroid_of_coef(ls.coef, ls, lo, hi))
    return {
        "centroid": c.tolist(),
        "nose_x": float(np.min(O.surface_polyline(ls, lo, hi)[0][:, 0])),
        "J": j0.tolist(),
        "J_moved": j1.tolist(),
        "adjoint_dJdx_over_J": adj.tolist(),
        "fd_dJdx_over_J": fd.tolist(),
        "summary": {
            "adjoint": [float(adj.mean()), float(adj.std(ddof=1) / np.sqrt(len(adj)))],
            "fd": [float(fd.mean()), float(fd.std(ddof=1) / np.sqrt(len(fd)))],
        },
    }


def main(argv=None):
    args, cfg = parse(sys.argv[1:] if argv is None else argv)
    ensemble_backward.configure("virtual", args.substeps, args.partners, "staged")
    grid, ls = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    out = {"case": cfg["case"], "box_x": [float(lo[0]), float(hi[0])], "seeds": args.seeds,
           "shift": SHIFT, "shapes": {}}  # fmt: skip
    path = Path(f"scratch/boxcheck/cluster/{args.name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    for label, body in (("circle", ls), ("m10_adam_it100", resampled(cfg, ls))):
        t0 = time.time()
        out["shapes"][label] = measure(cfg, body, lo, hi, args.seeds)
        out["shapes"][label]["seconds"] = time.time() - t0
        print(label, out["shapes"][label]["summary"], flush=True)
        path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
