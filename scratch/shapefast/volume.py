"""The cut-volume term (shape.volume_cotangent) against the earlier whole-grid form (every
Cartesian cell's sub^D samples, one VJP; copied here): the optm grid (2-D, sub 32) and the
3-D sphere (sub 4, the old form's memory allows no more on a laptop), random weights; agreement
and time (2026-10-24: near-surface cells only, chunked).

    PYTHONPATH=. python3 scratch/shapefast/volume.py
"""

import functools
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as OPT  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import shape  # noqa: E402
from jax_dsmc.grid import collision_cell, spacing  # noqa: E402


def old_volume_cotangent(geom, mesh, grid, weights, sub):
    ls = mesh.body
    width = ls.h / 4.0
    d = len(grid.n)
    dx = spacing(grid)
    idx = jnp.stack(jnp.meshgrid(*[jnp.arange(n) for n in grid.n], indexing="ij"), axis=-1)
    offs = jnp.stack(jnp.meshgrid(*[(jnp.arange(sub) + 0.5) / sub] * d, indexing="ij"), axis=-1)
    x = (grid.lo + (idx.reshape(-1, 1, d) + offs.reshape(1, -1, d)) * dx).reshape(-1, d)
    w_sample = weights[collision_cell(x, grid)] * jnp.prod(dx) / sub**d * grid.transverse
    f = functools.partial(shape._weighted_flow_volume, ls=ls, x=x, w_sample=w_sample, width=width)
    _v, vjp = jax.vjp(f, geom)
    return vjp(jnp.float64(1.0))[0]


def compare(label, cfg, sub):
    grid, mesh, *_ = EM.build(cfg, "radius", EM.nominal(cfg, "radius"), 10)
    weights = jnp.asarray(np.random.default_rng(0).normal(size=grid.volume.shape))
    geom = shape.geometry(mesh)
    out = {}
    for name, fn in (("old", old_volume_cotangent), ("new", shape.volume_cotangent)):
        for _ in range(2):  # compile, then time
            t0 = time.time()
            cot = jax.block_until_ready(fn(geom, mesh, grid, weights, sub=sub))
            out[name] = (np.asarray(cot), time.time() - t0)
    ref, new = out["old"][0], out["new"][0]
    print(f"{label}: grid {grid.n}, max |new - old| / max |old| "
          f"{np.abs(new - ref).max() / np.abs(ref).max():.2e}; old {out['old'][1]:.2f} s, "
          f"new {out['new'][1]:.2f} s", flush=True)  # fmt: skip


compare("optm 2-D", dict(OPT.default_config(), case="circle_long", fnum_scale=2.0,
                         levelset_h=0.0625, levelset_samples=16, grazing=0.05, boundary="free",
                         knudsen=0.5, mach=10.0, dt=2.5e-5, a_max=256), 32)  # fmt: skip
compare("sphere 3-D", dict(OPT.default_config(), case="sphere", levelset_h=0.1,
                           levelset_samples=16, grazing=0.05, boundary="free", knudsen=0.5), 4)  # fmt: skip
