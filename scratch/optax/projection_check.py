"""Was the first centroid constraint (optcm, 2026-10-23: Euclidean projection on, and Newton
restoration along, the centroid's coefficient gradient) reshaping the body?  Per iterate of a
run: the step ``coef_{k+1} - coef_k`` split into its least-squares part along the raw rows
``[g_area, dc/dcoef]`` and the rest; the high-frequency fraction (5-point Laplacian over the
field, coefficient units) of each, and of the rows and the translation modes themselves.

    PYTHONPATH=. python3 scratch/optax/projection_check.py
"""

import dataclasses
import json
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/optax")
import optimize as O  # noqa: E402
from analysis import ROOT, tag  # noqa: E402


def hf(d):
    lap = 4 * d[1:-1, 1:-1] - d[2:, 1:-1] - d[:-2, 1:-1] - d[1:-1, 2:] - d[1:-1, :-2]
    return float(np.sqrt((lap**2).sum() / max((d[1:-1, 1:-1] ** 2).sum(), 1e-300)))


def main():
    for regime, arm in (("kn0.5_M10", "ox_adam_cm"), ("kn0.5_M10", "ox_prodigy_cm"),
                        ("kn0.5", "ox_adam_cm")):  # fmt: skip
        cfg = dict(O.default_config(), **json.load(open(f"{ROOT}/{tag(regime, arm)}/history.json"))["config"])  # fmt: skip
        grid, ls0 = O.initial_levelset(cfg)
        lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
        out = []
        for k in range(100, 200, 10):
            c0 = np.load(f"{ROOT}/{tag(regime, arm)}/coef_{k}.npy")
            c1 = np.load(f"{ROOT}/{tag(regime, arm)}/coef_{k + 1}.npy")
            ls = dataclasses.replace(ls0, coef=jnp.asarray(c0))
            g_area = O.area_and_gradient(ls, lo, hi)[1]
            rows = np.concatenate((g_area[None], O.centroid_and_gradient(ls, lo, hi)[1]))
            modes = O.translation_modes(ls, cfg["band"] + ls.h)
            d = c1 - c0
            # the reinitialisation changes far-field coefficients: compare inside the band
            mask = np.asarray(O.L.coefficient_band(ls, cfg["band"] + ls.h))
            d = d * mask
            r = rows.reshape(3, -1)
            along = (np.linalg.lstsq(r.T, d.ravel(), rcond=None)[0] @ r).reshape(d.shape)
            rest = d - along
            out.append([np.linalg.norm(along) / np.linalg.norm(d), hf(d), hf(along), hf(rest),
                        hf(rows[1]), hf(modes[0])])  # fmt: skip
        m = np.mean(out, axis=0)
        print(f"{regime:10s} {arm:16s} |step along rows|/|step| {m[0]:.2f}; high-frequency "
              f"fraction: step {m[1]:.2f}, its row part {m[2]:.2f}, the rest {m[3]:.2f}; "
              f"centroid row {m[4]:.2f}, translation mode {m[5]:.2f}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
