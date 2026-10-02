"""1-core CPU microbenchmark of stable cell-order candidates on a steady-state cell array."""
import functools, sys, time
import jax, jax.numpy as jnp, numpy as np
from jax import lax
jax.config.update("jax_enable_x64", True)
d = np.load("/tmp/misc_state_polygon.npz")
c = jnp.asarray(d["cell"]); ref = np.asarray(d["order"])
C = int(d["counts"].shape[0]); N = c.shape[0]

def packed64(c):
    n = c.shape[0]
    p = c.astype(jnp.int64) * n + jnp.arange(n, dtype=jnp.int64)
    return (lax.sort(p, is_stable=False) % n).astype(jnp.int32)

def packed32(c):
    n = c.shape[0]
    p = c * n + jnp.arange(n, dtype=jnp.int32)
    return lax.sort(p, is_stable=False) % n

def shift32(c):
    n = c.shape[0]; s = int(np.ceil(np.log2(n)))
    p = (c << s) | jnp.arange(n, dtype=jnp.int32)
    return lax.sort(p, is_stable=False) & ((1 << s) - 1)

def argsort_stable(c):
    return jnp.argsort(c, stable=True).astype(jnp.int32)

def blocked(c, b):
    n = c.shape[0]; B = -(-n // b); pad = B * b - n
    cp = jnp.concatenate([c, jnp.full(pad, C, c.dtype)]) if pad else c
    cb = cp.reshape(B, b)
    # local rank within block among same-cell earlier slots
    same = (cb[:, :, None] == cb[:, None, :]) & (jnp.arange(b)[None, :] < jnp.arange(b)[:, None])[None]
    local = jnp.sum(same, axis=2, dtype=jnp.int32)  # (B, b)
    H = jnp.zeros((B, C + 1), jnp.int32).at[jnp.arange(B)[:, None], cb].add(1)
    P = jnp.cumsum(H, axis=0) - H
    counts = jnp.sum(H, axis=0)
    starts = jnp.cumsum(counts) - counts
    dest = starts[cb] + P[jnp.arange(B)[:, None], cb] + local
    return jnp.zeros(B * b, jnp.int32).at[dest.reshape(-1)].set(jnp.arange(B * b, dtype=jnp.int32), unique_indices=True)[:n]

cands = {"packed64": packed64, "packed32": packed32, "shift32": shift32, "argsort": argsort_stable}
for b in (16, 32, 64, 128):
    cands[f"blocked{b}"] = functools.partial(blocked, b=b)
# shuffle-free real data, and a random permutation of slots (worst case)
for name, f in cands.items():
    g = jax.jit(f)
    out = np.asarray(g(c))
    ok = np.array_equal(out, ref)
    g(c).block_until_ready()
    ts = []
    for _ in range(20):
        t0 = time.perf_counter(); g(c).block_until_ready(); ts.append(time.perf_counter() - t0)
    print(f"{name:12s} {np.median(ts)*1e3:7.3f} ms  min {min(ts)*1e3:7.3f}  same={ok}", flush=True)
