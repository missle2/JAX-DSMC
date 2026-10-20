"""Pieces of the guarded evaluation (slot-aligned, the cell tables), each jitted alone."""

import time

import jax
import jax.numpy as jnp

import bench_layout as B  # noqa: F401  (arrays; runs its own benchmark first)
from jax_dsmc import cell_fit

basis = "cell:cubicv:adapt1.5:guard32"
f = jax.block_until_ready(jax.jit(B.fit_only, static_argnums=0)(basis))
c = jnp.minimum(B.cell, B.ids.shape[0] - 1)
sizes = cell_fit._prefix_sizes("cubicv", 2)


def design(f):
    q = jnp.concatenate((B.x, B.qv), axis=-1) / B.scale
    return cell_fit.features(q - f.centre[c], "cubicv")


def leverage(f):
    h_tab, (_n, per, rank) = cell_fit._leverage_tables(f.whiten, design(f), sizes, (B.ids, B.rank))
    return h_tab[c, jnp.clip(rank, 0, per - 1)]


def level(f):
    return cell_fit._guard_level(f, c, design(f), 32.0, jnp.ones(c.shape, bool), sizes,
                                 (B.ids, B.rank))  # fmt: skip


def value(f):
    lv = level(f)
    flat = f.levels.reshape((-1, *f.levels.shape[2:]))
    beta = flat[c * f.levels.shape[1] + lv]
    return jnp.einsum("qi,qi->q", design(f), beta)


for name, fn in (("design", design), ("leverage", leverage), ("level", level), ("value", value)):
    g = jax.jit(fn)
    jax.block_until_ready(g(f))
    t0 = time.time()
    for _ in range(50):
        out = g(f)
    jax.block_until_ready(out)
    print(f"{name:10s} {(time.time() - t0) / 50 * 1e3:6.2f} ms", flush=True)
