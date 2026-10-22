"""A/B timings of the optimiser's gradient phases on the local GPU at reduced size (the optm
M 10 circle, fnum x 8, a short captured window, warm-started between calls), 2026-10-22:

* ``shipped``   -- as the optimiser runs it (the traced replay at the record's sizes,
  ``EM.replay_static``);
* ``fullsize``  -- the traced replay at the forward's sizes (32 passes, lane N / 32).

(Its first version, 2026-10-22, also timed ``combined`` -- the shape step's pull-backs summed
into one cotangent: 9.8 s against 10.1, gradient within 5e-16 -- and ``unguarded``.)

Each setting: one call that compiles, one timed.  Prints the phases and the relative
difference of the summed gradient against ``shipped``.

    PYTHONPATH=. python3 scratch/shapefast/bench.py [overrides-json]
"""

import functools
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as OPT  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward, shape  # noqa: E402

PHASES = ("warmup", "capture", "replay", "backward", "shape")
SHIPPED_PULLBACKS = shape._pullbacks


def combined_pullbacks(vjp, cotangents):
    """Cost model of one summed pull-back: the whole sum into the first slot, zeros after."""
    parts = [[a for a in jax.tree.leaves(c) if a.dtype != jax.dtypes.float0] for c in cotangents]
    (one,) = vjp([functools.reduce(jnp.add, column) for column in zip(*parts, strict=True)])
    zero = jax.tree.map(jnp.zeros_like, one)
    return [one] + [zero] * (len(cotangents) - 1)


def total_gradient(terms, s):
    g = terms.pathwise + terms.score + terms.silhouette + terms.timing
    if terms.order is not None:
        g = g + terms.order
    if s["volume"] is not None:
        g = g + s["volume"]
    return np.asarray(jax.tree.leaves(g)[0])


def run(cfg, ls, key, restart):
    t0 = time.time()
    all_terms, info, s = EM.shape_terms(
        cfg, key, ls, "radius", cfg["neighbors"], cfg["timing_band"], (cfg["variant"],), 0,
        restart,
    )  # fmt: skip
    info["total"] = time.time() - t0
    return total_gradient(all_terms[cfg["variant"]], s), info, s["final"]


def main():
    over = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    cfg = dict(OPT.default_config(), case="circle_long", fnum_scale=8.0, levelset_h=0.0625,
               levelset_samples=16, grazing=0.05, boundary="free", knudsen=0.5, mach=10.0,
               dt=2.5e-5, a_max=256, warmup=2800, steps=120, window=80, smooth=8.0,
               variant="along_pool_plane", neighbors=32, timing_band=10.0)  # fmt: skip
    cfg.update(over)
    _grid, ls = OPT.initial_levelset(cfg)
    key = jax.random.key(32)
    shipped = ensemble_backward.CELL_BASIS
    ensemble_backward.configure("virtual", 4, 8, "staged", shipped)
    print("cold start ...", flush=True)
    _g, info, final0 = run(cfg, ls, key, None)
    print("cold", {p: round(info[f"seconds_{p}"], 1) for p in PHASES}, flush=True)
    warm = dict(cfg, restart_steps=100)
    ref = None
    full_size = EM.replay_static
    for name, sizes in (
        ("shipped", full_size),
        ("fullsize", lambda static, walls, floor=(0, 0): static),
    ):
        ensemble_backward.configure("virtual", 4, 8, "staged", shipped)
        EM.replay_static = sizes
        for label in ("compile", "timed"):
            # every call restarts from the cold run's gas: identical inputs, so the settings'
            # gradients compare exactly
            g, info, _final = run(warm, ls, key, final0)
            row = {p: round(info[f"seconds_{p}"], 2) for p in PHASES}
            print(f"{name:10s} {label:8s} total {info['total']:7.1f}", row, flush=True)
        if ref is None:
            ref = g
        print(f"  |g| {np.linalg.norm(g):.6g}  rel diff vs shipped "
              f"{np.linalg.norm(g - ref) / np.linalg.norm(ref):.3g}", flush=True)  # fmt: skip
    EM.replay_static = full_size


if __name__ == "__main__":
    main()
