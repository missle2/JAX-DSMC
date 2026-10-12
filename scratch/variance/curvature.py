"""Surface curvature of optc iterates: the share of the surface whose radius of curvature is
below 12.5 cm (2 knots; the DSMC cell is 50 cm) and the largest |curvature| -- where the
stored gradients' per-seed spikes sit (scratch/variance/spikes.py).

    PYTHONPATH=. python3 scratch/variance/curvature.py
"""

import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as OPT  # noqa: E402

from jax_dsmc.mesh import levelset as L  # noqa: E402

sys.path.insert(0, "scratch/variance")
from spikes import curvature  # noqa: E402

OPT_DIR = "examples/mesh/results/optimize"
RUNS = [("circle", "circle_coarse_S1_adam", 0), ("circle", "circle_coarse_S1_adam", 50),
        ("circle", "circle_coarse_S1_adam", 100), ("circle", "circle_coarse_S1_adam", 200),
        ("circle", "circle_coarse_S8_adam", 30), ("circle", "circle_coarse_S2_gd", 200),
        ("circle_long", "circle_long_S4_adam", 45), ("circle_long", "circle_long_S8_adam", 40),
        ("circle_long", "circle_long_S8_adam", 88)]


def main():
    for case, run, it in RUNS:
        cfg = OPT.default_config()
        cfg.update(case=case, levelset_h=0.0625)
        grid, ls0 = OPT.initial_levelset(cfg)
        ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"{OPT_DIR}/{run}/coef_{it}.npy")))
        lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
        axes = [np.arange(lo[d], hi[d], 0.01) for d in range(2)]
        x = np.stack([a.ravel() for a in np.meshgrid(*axes, indexing="ij")], axis=1)
        phi = np.asarray(L.value(ls, jnp.asarray(x)))
        on = x[np.abs(phi) < 0.005]
        k = np.asarray(curvature(ls, on))
        sharp = np.abs(k) > 8.0
        span = on.max(0) - on.min(0)
        print(f"{run:24s} it {it:3d}: extent {span[0]:5.2f} x {span[1]:5.2f} m, |k| > 8 /m on "
              f"{sharp.mean() * 100:4.1f} % of the surface (concave {np.mean(k < -8) * 100:4.1f} %),"
              f" max |k| {np.abs(k).max():5.1f} /m")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()
