"""GUARD_BATCH sweep of the guarded evaluation (bench_eval.py's sizes)."""

import time

import jax

import bench_eval as B  # noqa: F401  (sizes and arrays)
from jax_dsmc import cell_fit

cell_fit.TRACE = None
for batch in (8192, 32768, 131072):
    cell_fit.GUARD_BATCH = batch
    jax.clear_caches()
    fn = jax.jit(B.run, static_argnums=0)
    basis = "cell:cubicv:adapt1.5:guard32"
    jax.block_until_ready(fn(basis))
    t0 = time.time()
    for _ in range(10):
        out = fn(basis)
    jax.block_until_ready(out)
    print(f"batch {batch}: {(time.time() - t0) / 10 * 1e3:6.1f} ms")
