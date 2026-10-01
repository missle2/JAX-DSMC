import time, jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
N = 60000
rng = np.random.default_rng(0)
keys = np.sort(rng.integers(0, 400, N)).astype(np.int32)
m = rng.random(N) < 0.05; keys[m] = rng.integers(0, 401, m.sum())
k = jnp.asarray(keys)
x = jnp.asarray(rng.random((6, N)))
def f_argsort(k, x):
    o = jnp.argsort(k, stable=True)
    return k[o], x[:, o]
def f_packed(k, x):
    s = jnp.sort(k * 65536 + jnp.arange(N, dtype=jnp.int32))
    o = s & 0xFFFF
    return s >> 16, x[:, o]
def f_multi(k, x):
    out = jax.lax.sort((k,) + tuple(x[i] for i in range(6)), num_keys=1)
    return out[0], jnp.stack(out[1:])
def f_cumsum(k, x):
    return jnp.cumsum(k), x * 2.0
for name, f in [("argsort", f_argsort), ("packed", f_packed), ("multi", f_multi), ("cumsum", f_cumsum)]:
    g = jax.jit(lambda k, x, f=f: jax.lax.fori_loop(0, 100, lambda i, c: f(c[0], c[1]), (k, x)))
    jax.block_until_ready(g(k, x))
    t = time.perf_counter(); jax.block_until_ready(g(k, x)); dt = (time.perf_counter() - t) / 100
    print(jax.devices()[0].platform, name, f"{dt*1e3:.3f} ms", flush=True)
