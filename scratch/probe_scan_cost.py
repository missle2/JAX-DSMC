"""Probe F: careful timing of the clamp+scratch fix."""
import jax, jax.numpy as jnp, time
jax.config.update("jax_enable_x64", True)
from jax import random
import numpy as np

def build(fixed, C, N, A_max, seed=0):
    key = random.key(seed)
    def body(carry, a):
        v, order, starts, counts, n = carry
        active = (a < n) & (counts >= 2)
        npc = jnp.maximum(counts, 2) if fixed else counts
        k1, k2 = random.split(random.fold_in(key, a))
        i = random.randint(k1, npc.shape, 0, npc)
        r = random.randint(k2, npc.shape, 0, npc - 1)
        j = (i + 1 + r) % npc
        if fixed:
            pi = jnp.where(active, order[starts+i], N)
            pj = jnp.where(active, order[starts+j], N)
        else:
            pi = order[starts+i]; pj = order[starts+j]
        vi, vj = v[pi], v[pj]
        v = v.at[pi].set(jnp.where(active, vj, vi))
        v = v.at[pj].set(jnp.where(active, vi, vj))
        return (v, order, starts, counts, n), None
    @jax.jit
    def f(v, order, starts, counts, n):
        (v, *_), _ = jax.lax.scan(body, (v, order, starts, counts, n), jnp.arange(A_max))
        return v
    return f

def bench(f, args, reps=200):
    f(*args).block_until_ready()
    ts = []
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(reps): out = f(*args)
        out.block_until_ready()
        ts.append((time.perf_counter()-t0)/reps*1e3)
    return min(ts)

print(f"{'C':>7} {'N':>9} {'A_max':>6} {'spec ms':>10} {'fixed ms':>10} {'ratio':>7}")
for (Cn, Nn, Am) in [(4096, 262144, 8), (16384, 1048576, 8), (65536, 4194304, 16)]:
    cell = jnp.repeat(jnp.arange(Cn, dtype=jnp.int32), Nn // Cn)
    order = jnp.argsort(cell); cnt = jnp.bincount(cell, length=Cn+1)
    starts = (jnp.cumsum(cnt)-cnt)[:Cn]; counts = cnt[:Cn]
    n = jnp.full((Cn,), Am, jnp.int32)
    ts = bench(build(False, Cn, Nn, Am), (jnp.arange(float(Nn)), order, starts, counts, n))
    tf = bench(build(True,  Cn, Nn, Am), (jnp.arange(float(Nn)+1), order, starts, counts, n))
    print(f"{Cn:>7} {Nn:>9} {Am:>6} {ts:>10.3f} {tf:>10.3f} {tf/ts:>7.3f}")
