"""Held-out drag of the optc shapes: the circle (coef_0) and every run's last evaluated shape
(coef_{n-1}, the shape of the last history row) on seeds the optimiser never used.

    XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false \
        PYTHONPATH=. python3 scratch/mesh/heldout_drag.py --seeds 4

The optimiser's J is measured on its own common-random-number seeds (keys 32 .. 32+S-1),
the same seeds it descends on, so its history can partly fit that sample.  Here each shape
is re-run forward only (no capture) on keys 1000 + i.  The slot capacity is raised to
80 000 (the Adam shapes hold ~56-60 k particles against the default 60 256; a capacity
change is not physics).  Output: examples/mesh/results/optimize/heldout_drag.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples", "mesh"))
import forward as F  # noqa: E402
import optimize as O  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402

KEY0 = 1000


def drag(cfg, ls, seed):
    key = jax.random.key(KEY0 + seed)
    _s0, _final, diag, _rec, tallies, s = EM.forward(cfg, "radius", None, key, False, body=ls)
    scale = float(s["params"].fnum[0]) / (cfg["window"] * float(s["params"].dt))
    return EM.objective_value(cfg, tallies, s["static"].D, scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    F.CASES["circle"]["capacity"] = 80_000
    out_path = os.path.join(O.RESULTS, "heldout_drag.json")
    result = json.load(open(out_path)) if os.path.exists(out_path) else {}
    tags = sorted(os.path.basename(p) for p in glob.glob(os.path.join(O.RESULTS, "circle_coarse_S*")))
    jobs = [("circle", tags[0], 0)] + [
        (t, t, len(json.load(open(os.path.join(O.RESULTS, t, "history.json")))["rows"]) - 1)
        for t in tags
    ]
    for name, tag, k in jobs:
        if name in result and len(result[name]["drag"]) >= args.seeds:
            continue
        hist = json.load(open(os.path.join(O.RESULTS, tag, "history.json")))
        cfg = dict(O.default_config(), **hist["config"])
        _grid, ls0 = O.initial_levelset(cfg)
        ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(os.path.join(O.RESULTS, tag, f"coef_{k}.npy"))))
        t0 = time.time()
        values = [drag(cfg, ls, i) for i in range(args.seeds)]
        result[name] = {
            "iterate": k,
            "drag": values,
            "in_sample": hist["rows"][k]["objective"] if name != "circle" else hist["rows"][0]["objective"],
            "seconds": time.time() - t0,
        }
        v = np.array(values)
        print(f"{name} (coef_{k}): held-out {v.mean():.4e} ± {v.std(ddof=1) / np.sqrt(len(v)):.1e}"
              f"  in-sample {result[name]['in_sample']:.4e}  {time.time() - t0:.0f}s", flush=True)
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
