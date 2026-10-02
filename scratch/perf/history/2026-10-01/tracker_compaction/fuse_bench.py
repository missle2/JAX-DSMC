import time, functools
import jax, jax.numpy as jnp, numpy as np
from jax import lax
jax.config.update("jax_enable_x64", True)
n, C = 60000, 400
rng = np.random.default_rng(0)
cell = jnp.asarray(rng.integers(0, C, n).astype(np.int32))
alive = jnp.asarray(rng.random(n) < 0.8)
f = jnp.asarray(rng.random(C) * 3)
u = jnp.asarray(rng.random(C))
logh = jnp.asarray(rng.random(n + 1))

def logp_of(f, u):
    lower = jnp.floor(f); frac = f - lower
    up = u < frac
    return jnp.where(up, jnp.log(jnp.maximum(frac, 1e-30)), jnp.log(jnp.maximum(1 - frac, 1e-30)))

def base(logh, f, u, cell, alive):
    lp = logp_of(f, u)
    return logh.at[:n].add(jnp.where(alive, lp[cell], 0.0))[:n]
def red(logh, f, u, cell, alive):
    lp = logp_of(f, u)
    lp = jnp.max(jnp.stack([lp, lp]), axis=0)
    return logh.at[:n].add(jnp.where(alive, lp[cell], 0.0))[:n]
def cnd(logh, f, u, cell, alive):
    lp = lax.cond(jnp.all(jnp.isfinite(f)) | True, logp_of, logp_of, f, u)
    return logh.at[:n].add(jnp.where(alive, lp[cell], 0.0))[:n]
def whl(logh, f, u, cell, alive):
    lp = lax.fori_loop(0, 1, lambda i, a: logp_of(*a) if False else a, (f, u))
    lp = logp_of(f, u)
    lp = lax.while_loop(lambda c: c[0] < 1, lambda c: (c[0] + 1, c[1]), (0, lp))[1]
    return logh.at[:n].add(jnp.where(alive, lp[cell], 0.0))[:n]
ref = None
for name, fn in [("base", base), ("reduce", red), ("cond", cnd), ("while", whl)]:
    g = jax.jit(fn)
    r = g(logh, f, u, cell, alive).block_until_ready()
    if ref is None: ref = r
    t = time.perf_counter()
    for _ in range(200): r = g(logh, f, u, cell, alive)
    r.block_until_ready()
    print(name, f"{(time.perf_counter()-t)/200*1e3:.3f} ms", bool(jnp.all(r == ref)))


def logp1(f, u):
    lower = jnp.floor(f); frac = f - lower
    up = u < frac
    return jnp.log(jnp.maximum(jnp.where(up, frac, 1 - frac), 1e-30))
def w1(logh, f, u, cell, alive):
    lp = logp1(f, u)
    return logh.at[:n].add(jnp.where(alive, lp[cell], 0.0))[:n]
def w2(logh, f, u, cell, alive):
    lp = lax.cond(jnp.any(cell >= 0), logp_of, logp_of, f, u)
    return logh.at[:n].add(jnp.where(alive, lp[cell], 0.0))[:n]
def w3(logh, f, u, cell, alive):
    lp = logp_of(f, u)
    lp = lax.sort(lp + 0 * f)  # placeholder
    return logh.at[:n].add(jnp.where(alive, lp[cell], 0.0))[:n]
def w4(logh, f, u, cell, alive):
    lp = logp_of(f, u)
    lp = jax.ops.segment_sum(lp, jnp.arange(C), C)
    return logh.at[:n].add(jnp.where(alive, lp[cell], 0.0))[:n]
for name, fn in [("w1 onelog", w1), ("w2 runtime cond", w2), ("w3 sort", w3), ("w4 segsum", w4)]:
    g = jax.jit(fn)
    r = g(logh, f, u, cell, alive).block_until_ready()
    t = time.perf_counter()
    for _ in range(200): r = g(logh, f, u, cell, alive)
    r.block_until_ready()
    print(name, f"{(time.perf_counter()-t)/200*1e3:.3f} ms", bool(jnp.all(r == ref)))
