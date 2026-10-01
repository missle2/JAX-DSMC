import time, jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
N = 60000
rng = np.random.default_rng(0)
k = jnp.asarray(np.sort(rng.integers(0, 400, N)).astype(np.int32))
x = jnp.asarray(rng.random(N)); y = jnp.asarray(rng.random(N))
perm = jnp.asarray(rng.permutation(N).astype(np.int32))
m = jnp.asarray(rng.random(N) < 0.1)
tests = {
 "cumsum_i32": lambda k, x, y: (jnp.cumsum(k), x, y),
 "cumsum_assoc": lambda k, x, y: (jax.lax.associative_scan(jnp.add, k), x, y),
 "gather6": lambda k, x, y: (k[perm], x[perm] + y[perm], y[perm]*x[perm]),
 "scatter_set": lambda k, x, y: (k, jnp.zeros(N).at[perm].set(x), y),
 "segsum400": lambda k, x, y: (k, x, y + jax.ops.segment_sum(x, k, 401)[k]),
 "bincount": lambda k, x, y: (k + jnp.bincount(k, length=401)[k], x, y),
 "elementwise": lambda k, x, y: (k, x + 1e-4 * y, jnp.where(x > 0.5, y * 2, y) ),
 "nonzero": lambda k, x, y: (k, x, y.at[jnp.nonzero(m, size=8192, fill_value=N)[0]].add(1.0, mode="drop")),
 "cumsum_compact": lambda k, x, y: (k, x, jnp.zeros(8192).at[jnp.where(m, jnp.cumsum(m) - 1, 8192)].set(y, mode="drop").sum() + y),
 "floor_cell": lambda k, x, y: ((jnp.floor(x * 20).astype(jnp.int32) * 20 + jnp.floor(y * 20).astype(jnp.int32)), x, y),
 "rng_uniform": lambda k, x, y: (k, jax.random.uniform(jax.random.key(k[0]), (N,)), y),
}
for name, f in tests.items():
    g = jax.jit(lambda k, x, y, f=f: jax.lax.fori_loop(0, 200, lambda i, c: f(*c), (k, x, y)))
    jax.block_until_ready(g(k, x, y))
    t = time.perf_counter(); jax.block_until_ready(g(k, x, y)); dt = (time.perf_counter() - t) / 200
    print(jax.devices()[0].platform, name, f"{dt*1e6:.1f} us", flush=True)
