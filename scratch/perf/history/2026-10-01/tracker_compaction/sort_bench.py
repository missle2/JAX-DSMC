import time, sys
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
n, C = 60000, 401
rng = np.random.default_rng(0)
c = jnp.asarray(np.where(rng.random(n) < 0.78, np.sort(rng.integers(0, 400, n)) , 400).astype(np.int32))
c_shuf = jnp.asarray(rng.permutation(np.asarray(c)))

def s64(c):
    p = c.astype(jnp.int64) * n + jnp.arange(n, dtype=jnp.int64)
    return (jax.lax.sort(p, is_stable=False) % n).astype(jnp.int32)
def s32(c):
    p = c * n + jnp.arange(n, dtype=jnp.int32)
    return jax.lax.sort(p, is_stable=False) % n
def s32u(c):
    p = (c * n + jnp.arange(n, dtype=jnp.int32)).astype(jnp.uint32)
    return (jax.lax.sort(p, is_stable=False) % n).astype(jnp.int32)
def argst(c):
    return jnp.argsort(c, stable=True).astype(jnp.int32)
def f32key(c):
    p = (c * n + jnp.arange(n, dtype=jnp.int32)).astype(jnp.float32)  # exact below 2^24? no
    return p
for name, f in [("s64", s64), ("s32", s32), ("s32u", s32u), ("argsort", argst)]:
    g = jax.jit(f)
    for cc, lab in ((c, "near-sorted"), (c_shuf, "shuffled")):
        r = g(cc).block_until_ready()
        t = time.perf_counter()
        for _ in range(50):
            r = g(cc)
        r.block_until_ready()
        print(name, lab, f"{(time.perf_counter()-t)/50*1e3:.3f} ms", bool(jnp.all(r == s64(cc))))
