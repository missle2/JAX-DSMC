"""The localised spikes of the stored optc gradients: for each seed whose deviation puts
>= 50 % in its top 20 coefficients, where the spike sits -- position, distance to the
surface, surface orientation |n_x|, curvature (2-D: div n) -- and how much it moves the
shift / cos 2 theta projections.

    PYTHONPATH=. python3 scratch/variance/spikes.py
"""

import dataclasses
import glob
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as OPT  # noqa: E402

from jax_dsmc.mesh import levelset as L  # noqa: E402

sys.path.insert(0, "scratch/variance")
from where import CELL, OPT_DIR, SHAPES  # noqa: E402


def curvature(ls, x):
    """div(grad phi / |grad phi|) by central differences of the normal (h / 4)."""
    e = 0.25 * float(ls.h) if hasattr(ls, "h") else 0.015
    out = 0.0
    for d in range(2):
        dx = np.zeros(2)
        dx[d] = e
        _, gp = L.value_and_grad(ls, jnp.asarray(x + dx))
        _, gm = L.value_and_grad(ls, jnp.asarray(x - dx))
        npl = np.asarray(gp) / np.linalg.norm(np.asarray(gp), axis=-1, keepdims=True)
        nmi = np.asarray(gm) / np.linalg.norm(np.asarray(gm), axis=-1, keepdims=True)
        out = out + (npl[..., d] - nmi[..., d]) / (2 * e)
    return out


def main():
    cfg = OPT.default_config()
    cfg.update(fnum_scale=1, grid_scale=1, levelset_h=0.0625, levelset_samples=16)
    _g, ls0 = OPT.initial_levelset(cfg)
    for shape, (run, it) in SHAPES.items():
        ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"{OPT_DIR}/{run}/coef_{it}.npy")))
        x = np.asarray(L.knots(ls)).reshape(-1, 2)
        phi, grad = (np.asarray(a) for a in L.value_and_grad(ls, jnp.asarray(x)))
        gn = np.linalg.norm(grad, axis=1)
        for reg in ("fm", "kn"):
            rows = [json.loads(line) for f in sorted(glob.glob(
                f"scratch/cellfit/cluster/optc_{shape}_{reg}_b*.jsonl")) for line in open(f)]
            g = np.array([np.ravel(r["runs"][CELL]["gradient"]) for r in rows])
            dev = g - g.mean(0)
            sd_shift = np.std(g.sum(1))
            for i, r in enumerate(rows):
                d2 = dev[i] ** 2
                top = np.argsort(d2)[::-1][:20]
                share = d2[top].sum() / d2.sum()
                if share < 0.5:
                    continue
                c = top[0]
                xs = x[c] - phi[c] * grad[c] / gn[c] ** 2  # nearest surface point (one step)
                k = float(curvature(ls, xs[None])[0])
                _, gs = L.value_and_grad(ls, jnp.asarray(xs[None]))
                ns = np.asarray(gs)[0] / np.linalg.norm(np.asarray(gs)[0])
                spike_shift = dev[i, top].sum()
                print(f"{shape:5s} {reg} seed {r['seed']:2d}: top-20 share {share:.2f}  spike at "
                      f"({xs[0]:6.2f}, {xs[1]:5.2f})  n ({ns[0]:+.2f}, {ns[1]:+.2f})  curvature "
                      f"{k:+7.2f} /m  spike . 1 = {spike_shift / sd_shift:+.1f} sd(g . 1)")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()
