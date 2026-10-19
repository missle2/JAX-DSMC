"""Micro-benchmark: one per-cell fit + evaluation at Q queries, plain vs guarded, trace on/off."""

import time

import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc import cell_fit

C, P, Q = 1200, 128, 400_000
rng = np.random.default_rng(0)
x = jnp.asarray(rng.uniform(0, 1, (C, P, 2)))
v = jnp.asarray(rng.normal(0, 1, (C, P, 3)))
valid = jnp.asarray(np.arange(P)[None] < rng.integers(40, P, C)[:, None])
lam = jnp.asarray(rng.normal(size=(C, P)))
scale = jnp.asarray([1.0, 1.0, 1.0, 1.0, 1.0])
c = jnp.asarray(rng.integers(0, C, Q))
qx = jnp.asarray(rng.uniform(0, 1, (Q, 2)))
qv = jnp.asarray(rng.normal(0, 1.5, (Q, 3)))


def run(basis):
    f = cell_fit.fit(x, v, valid, lam, scale, basis)
    return cell_fit.evaluate(f, c, qx, qv, scale, basis, q_valid=jnp.ones(Q, bool))


for basis, trace in (("cell:cubicv:adapt1.5", None), ("cell:cubicv:adapt1.5:guard32", None),
                     ("cell:cubicv:adapt1.5:guard32", {})):  # fmt: skip
    cell_fit.TRACE = trace
    fn = jax.jit(run, static_argnums=0)
    jax.block_until_ready(fn(basis))
    t0 = time.time()
    for _ in range(10):
        out = fn(basis)
    jax.block_until_ready(out)
    jax.effects_barrier()
    print(f"{basis:32s} trace {trace is not None}: {(time.time() - t0) / 10 * 1e3:7.1f} ms")
