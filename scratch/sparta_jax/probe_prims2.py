import time, jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
N = 61440
rng = np.random.default_rng(0)
m = jnp.asarray(rng.random(N) < 0.1)
k = jnp.asarray(rng.integers(0, 400, 8192).astype(np.int32))

def cumsum2(m, C):
    a = m.astype(jnp.float32).reshape(-1, C)
    tri = jnp.triu(jnp.ones((C, C), jnp.float32))
    inner = a @ tri
    rows = inner[:, -1]
    off = jnp.cumsum(rows) - rows
    return (inner + off[:, None]).reshape(-1).astype(jnp.int32)
def cumsum3(m, C):
    a = m.astype(jnp.float32).reshape(-1, C)
    tri = jnp.triu(jnp.ones((C, C), jnp.float32))
    inner = a @ tri
    rows = inner[:, -1]
    r = rows.shape[0]; pad = (-r) % C
    rr = jnp.pad(rows, (0, pad)).reshape(-1, C)
    i2 = rr @ tri
    r2 = i2[:, -1]; o2 = jnp.cumsum(r2) - r2
    rowcs = (i2 + o2[:, None]).reshape(-1)[:r]
    off = rowcs - rows
    return (inner + off[:, None]).reshape(-1).astype(jnp.int32)
ref = np.cumsum(np.asarray(m))
for C in (16, 32, 64):
    assert (np.asarray(cumsum2(m, C)) == ref).all()
N2 = 60000
tests = {
 "cumsum_i32": lambda m: jnp.cumsum(m.astype(jnp.int32)),
 "cumsum2_16": lambda m: cumsum2(m, 16),
 "cumsum2_32": lambda m: cumsum2(m, 32),
 "cumsum2_64": lambda m: cumsum2(m, 64),
 "cumsum3_32": lambda m: cumsum3(m, 32),
 "topk4096": lambda m: jax.lax.top_k(jnp.where(m, -jnp.arange(N, dtype=jnp.int32), -N), 4096)[1],
 "nonzero": lambda m: jnp.nonzero(m, size=8192, fill_value=N)[0],
 "sort8192": lambda m: jnp.sort(k * 16384 + jnp.arange(8192, dtype=jnp.int32)),
 "sort2048": lambda m: jnp.sort(k[:2048] * 16384 + jnp.arange(2048, dtype=jnp.int32)),
}
for name, f in tests.items():
    g = jax.jit(lambda m, f=f: jax.lax.fori_loop(0, 200, lambda i, c: (c[0] ^ (i < 0), c[1] + f(c[0]).sum().astype(jnp.float64)), (m, jnp.float64(0))))
    jax.block_until_ready(g(m))
    t = time.perf_counter(); jax.block_until_ready(g(m)); dt = (time.perf_counter() - t) / 200
    print(jax.devices()[0].platform, name, f"{dt*1e6:.1f} us", flush=True)
