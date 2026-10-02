"""Gate before the squash_y gradient: the in-box drag of the level-set ellipse (R = 1.5 in
the 10 x 10 circle box, semi-axes (R, sR)) against the free-molecular half-range reference
of examples/mesh/ellipsoid_reference.py (blockage bias vs s), in the gradient-run
configuration (fnum x 1, 700 warm-up + 300 steps, window 200, h_phi 0.0625, 16 samples).

    XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false \
        PYTHONPATH=. python3 scratch/mesh/ellipse_drag_gate.py --s 0.35 0.6 1.0 --seeds 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "examples", "mesh"))
import ellipsoid_reference as E  # noqa: E402
import forward as F  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402

OUT = os.path.join(ROOT, "scratch", "mesh", "ellipse_drag_gate.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=float, nargs="+", default=[0.35, 0.6, 1.0])
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    cfg = dict(
        EM.default_config(),
        case="ellipse",
        body="levelset",
        fnum_scale=1.0,
        levelset_h=0.0625,
        levelset_samples=16,
        grazing=0.05,
    )
    c = F.CASES["ellipse"]
    result = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for s in args.s:
        key = f"{s:.3f}"
        if key in result and len(result[key]["drag"]) >= args.seeds:
            continue
        t0 = time.time()
        vals = []
        for i in range(args.seeds):
            _s0, _f, _d, _r, tallies, st = EM.forward(
                dict(cfg, squash=s), "squash_y", s, jax.random.key(32 + i), False
            )
            scale = float(st["params"].fnum[0]) / (cfg["window"] * float(st["params"].dt))
            vals.append(EM.objective_value(cfg, tallies, st["static"].D, scale))
        ref, dref = E.drag_2d(s, c["radius"], F.NRHO, E.MASS_AR, F.TEMP, F.VSTREAM)
        v = np.array(vals)
        result[key] = {"s": s, "drag": vals, "reference": ref, "dreference_ds": dref}
        print(
            f"s={s}: in-box {v.mean():.4e} ± {v.std(ddof=1) / np.sqrt(len(v)):.1e}  "
            f"reference {ref:.4e}  bias {(v.mean() / ref - 1) * 100:+.2f} %  {time.time() - t0:.0f}s",
            flush=True,
        )
        with open(OUT, "w") as fh:
            json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
