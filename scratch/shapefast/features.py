"""cell_fit.features (a stack of the p monomial columns: a minor-axis concatenate, strided
writes, 0.4-0.6 ms per call in the backward's profile) against the same products from a
static-index gather of [offsets, 1] (same multiplication order: bitwise equal), and the
guard's cumsum + gather of prefix leverages against a matmul with the 0/1 prefix matrix,
2026-10-22.

    PYTHONPATH=. python3 scratch/shapefast/features.py
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc import cell_fit


@functools.cache
def index_table(poly, dim):
    """``int[p, 3]``: each monomial's variables, padded with the index of the ones column."""
    terms = cell_fit.monomials(poly, dim)
    width = max(1, max(len(t) for t in terms))
    return np.asarray([list(t) + [dim + 3] * (width - len(t)) for t in terms])


def features_new(offsets, poly):
    table = index_table(poly, offsets.shape[-1] - 3)
    ext = jnp.concatenate((offsets, jnp.ones((*offsets.shape[:-1], 1), offsets.dtype)), axis=-1)
    cols = ext[..., table[:, 0]]
    for k in range(1, table.shape[1]):
        cols = cols * ext[..., table[:, k]]
    return cols


def lev_old(z, sizes):
    return jnp.cumsum(z**2, axis=-1)[..., jnp.asarray(sizes) - 1]


def lev_new(z, sizes):
    p = z.shape[-1]
    prefix = (np.arange(p)[:, None] < np.asarray(sizes)[None, :]).astype(np.float32)
    return (z**2) @ prefix


def bench(fn, *args, n=50):
    f = jax.jit(fn)
    jax.block_until_ready(f(*args))
    t0 = time.perf_counter()
    for _ in range(n):
        out = f(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n * 1e3, out


def main():
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.normal(size=(195000, 5)))
    for name, fn in (("old", lambda o: cell_fit.features(o, "cubicv")),
                     ("new", lambda o: features_new(o, "cubicv"))):  # fmt: skip
        ms, out = bench(fn, q)
        ms32, _ = bench(fn, q.astype(jnp.float32))
        print(f"features {name}: f64 {ms:.3f} ms  f32 {ms32:.3f} ms", flush=True)
        if name == "old":
            ref = np.asarray(out)
        else:
            print("  bitwise equal:", bool(np.array_equal(ref, np.asarray(out))))
    # the guard's prefix leverages at the production table size (C 1200, P 320, p 24)
    z = jnp.asarray(rng.normal(size=(1200, 320, 24)).astype(np.float32))
    sizes = cell_fit._prefix_sizes("cubicv", 2)
    a_ms, a = bench(functools.partial(lev_old, sizes=sizes), z)
    b_ms, b = bench(functools.partial(lev_new, sizes=sizes), z)
    rel = float(jnp.max(jnp.abs(a - b) / jnp.maximum(a, 1e-30)))
    print(f"leverage prefix: cumsum {a_ms:.3f} ms  matmul {b_ms:.3f} ms  max rel diff {rel:.2g}")


if __name__ == "__main__":
    main()
