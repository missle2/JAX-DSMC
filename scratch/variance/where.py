"""Where does the per-seed variance of the stored optc gradients (cellgate geom_gate rows)
live?  Per shape: the coefficients' distance to the surface and the local surface
orientation (|n_x|: 1 = facing the stream, 0 = parallel to it), variance-weighted, and how
concentrated each seed's deviation is.

    PYTHONPATH=. python3 scratch/variance/where.py
"""

import dataclasses
import glob
import json
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as OPT  # noqa: E402

from jax_dsmc.mesh import levelset as L  # noqa: E402

OPT_DIR = "examples/mesh/results/optimize"
SHAPES = {"c0": ("circle_coarse_S1_adam", 0), "a50": ("circle_coarse_S1_adam", 50),
          "a100": ("circle_coarse_S1_adam", 100), "a200": ("circle_coarse_S1_adam", 200),
          "s8a30": ("circle_coarse_S8_adam", 30), "g200": ("circle_coarse_S2_gd", 200)}
CELL = "cell:quad:adapt1.5:align0.9:both_k32"


def main():
    cfg = OPT.default_config()
    cfg.update(fnum_scale=1, grid_scale=1, levelset_h=0.0625, levelset_samples=16)
    _g, ls0 = OPT.initial_levelset(cfg)
    for shape, (run, it) in SHAPES.items():
        ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"{OPT_DIR}/{run}/coef_{it}.npy")))
        x = np.asarray(L.knots(ls)).reshape(-1, 2)
        phi, grad = L.value_and_grad(ls, jnp.asarray(x))
        phi, grad = np.asarray(phi), np.asarray(grad)
        gn = np.linalg.norm(grad, axis=1)
        dist = phi / np.maximum(gn, 1e-12)
        nx = np.abs(grad[:, 0]) / np.maximum(gn, 1e-12)
        for reg in ("fm", "kn"):
            rows = [json.loads(line) for f in sorted(glob.glob(
                f"scratch/cellfit/cluster/optc_{shape}_{reg}_b*.jsonl")) for line in open(f)]
            g = np.array([np.ravel(r["runs"][CELL]["gradient"]) for r in rows])
            dev = g - g.mean(0)
            var = (dev ** 2).sum(0)
            near = np.abs(dist) < 0.5
            w = var / var.sum()
            # share of the variance by surface orientation (coefficients within 0.5 m)
            bins = [0, 0.2, 0.5, 0.8, 1.01]
            share = [w[near & (nx >= a) & (nx < b)].sum() for a, b in zip(bins[:-1], bins[1:])]
            count = [(near & (nx >= a) & (nx < b)).sum() for a, b in zip(bins[:-1], bins[1:])]
            # concentration: share of each seed's squared deviation in its top 20 coefficients
            top = np.sort(dev ** 2, axis=1)[:, ::-1]
            conc = top[:, :20].sum(1) / top.sum(1)
            heavy = np.linalg.norm(dev, axis=1)
            print(f"{shape:5s} {reg}: var share by |n_x| bins {np.round(share, 2)} (knots "
                  f"{count}); var outside 0.5 m {w[~near].sum():.2f}; top-20 share median "
                  f"{np.median(conc):.2f} max {conc.max():.2f}; |dev| max/median "
                  f"{heavy.max() / np.median(heavy):.1f}; mean |g| {np.linalg.norm(g.mean(0)):.2e}"
                  f" rms dev {np.sqrt((heavy ** 2).mean()):.2e}")


if __name__ == "__main__":
    main()
