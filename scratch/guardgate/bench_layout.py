"""Guarded evaluation at slot-aligned queries: per-query gather vs the cell tables (layout),
against the unguarded cubic.  circle_long-like sizes: 1200 cells, 100 k particles, npc 256."""

import time

import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc import cell_fit, ensemble_device
from jax_dsmc.state import GridSpec

N, NPC = 100_000, 256
rng = np.random.default_rng(0)
grid = GridSpec(jnp.zeros(2), jnp.asarray([60.0, 20.0]), (60, 20))
x = jnp.asarray(rng.uniform(0, 1, (N, 2)) * np.array([60.0, 20.0]))
v = jnp.asarray(rng.normal(0, 1, (N, 3)))
alive = jnp.ones(N, bool)
ids, valid, cell, rank, _count = ensemble_device.cell_nodes(x, alive, grid, NPC)
lam = jnp.asarray(rng.normal(size=N))
scale = jnp.ones(5)
qv = v + jnp.asarray(rng.normal(0, 2.0, (N, 3)))


def run(basis, use_layout):
    f = cell_fit.fit(x[ids], v[ids], valid, lam[ids], scale, basis)
    c = jnp.minimum(cell, ids.shape[0] - 1)
    lay = (ids, rank) if use_layout else None
    return cell_fit.evaluate(f, c, x, qv, scale, basis, layout=lay)


for basis, lay in (("cell:cubicv:adapt1.5", False), ("cell:cubicv:adapt1.5:guard32", False),
                   ("cell:cubicv:adapt1.5:guard32", True)):  # fmt: skip
    fn = jax.jit(run, static_argnums=(0, 1))
    jax.block_until_ready(fn(basis, lay))
    t0 = time.time()
    for _ in range(20):
        out = fn(basis, lay)
    jax.block_until_ready(out)
    print(f"{basis:30s} layout {lay}: {(time.time() - t0) / 20 * 1e3:6.1f} ms (fit + evaluate)")


def fit_only(basis):
    return cell_fit.fit(x[ids], v[ids], valid, lam[ids], scale, basis)


def eval_only(f, basis, use_layout):
    c = jnp.minimum(cell, ids.shape[0] - 1)
    return cell_fit.evaluate(f, c, x, qv, scale, basis, layout=(ids, rank) if use_layout else None)


print("separately:")
for basis, lay in (("cell:cubicv:adapt1.5", False), ("cell:cubicv:adapt1.5:guard32", False),
                   ("cell:cubicv:adapt1.5:guard32", True)):  # fmt: skip
    ff = jax.jit(fit_only, static_argnums=0)
    fe = jax.jit(eval_only, static_argnums=(1, 2))
    f = jax.block_until_ready(ff(basis))
    jax.block_until_ready(fe(f, basis, lay))
    t0 = time.time()
    for _ in range(20):
        g = ff(basis)
    jax.block_until_ready(g)
    t1 = time.time()
    for _ in range(20):
        out = fe(f, basis, lay)
    jax.block_until_ready(out)
    t2 = time.time()
    print(f"{basis:30s} layout {lay}: fit {(t1 - t0) / 20 * 1e3:6.1f} ms  "
          f"evaluate {(t2 - t1) / 20 * 1e3:6.1f} ms")
