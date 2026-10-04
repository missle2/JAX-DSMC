"""ranked_slots on XLA:CPU: jnp.searchsorted(method="scan") vs an unrolled branch-free
binary search of static depth (same integers), timed and op-counted at the forward's sizes."""
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def scan_search(rank, want):
    return jnp.searchsorted(rank, want, side="left", method="scan").astype(jnp.int32)


def unrolled_search(rank, want):
    n = rank.shape[0]
    pos = jnp.zeros(want.shape, jnp.int32)
    step = 1 << (n.bit_length() - 1)
    while step:
        probe = jnp.minimum(pos + step, n) - 1
        pos = jnp.where((pos + step <= n) & (rank[probe] < want), pos + step, pos)
        step >>= 1
    return pos


def ops(f, *a):
    txt = jax.jit(f).lower(*a).compile().as_text()
    entry = txt[txt.index("ENTRY"):]
    return sum(1 for l in entry.splitlines() if " = " in l and "parameter(" not in l and "constant(" not in l)


rng = np.random.default_rng(0)
for n, size in ((60000, 1875), (60000, 3750), (60000, 256), (60000, 30000)):
    mask = rng.random(n) < 0.5
    rank = jnp.asarray(np.cumsum(mask) - 1, jnp.int32)
    want = jnp.arange(size, dtype=jnp.int32) + 7
    a, b = jax.jit(scan_search)(rank, want), jax.jit(unrolled_search)(rank, want)
    assert (np.asarray(a) == np.asarray(b)).all()
    for name, f in (("scan", scan_search), ("unrolled", unrolled_search)):
        g = jax.jit(f)
        g(rank, want).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(200):
            r = g(rank, want)
        r.block_until_ready()
        print(f"n {n} size {size:6d} {name:9s} {(time.perf_counter()-t0)/200*1e6:8.1f} us  entry ops {ops(f, rank, want)}")
