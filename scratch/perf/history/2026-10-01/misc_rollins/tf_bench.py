import sys, time
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
if sys.argv[1] == "unrolled":
    from jax._src import prng
    from jax.interpreters import mlir
    mlir.register_lowering(prng.threefry2x32_p, prng._threefry2x32_lowering_rule, platform="cpu")
@jax.jit
def f(key):
    k1, k2 = jax.random.split(key)
    return jax.random.uniform(k1, (60000, 3), dtype=jnp.float64).sum(1) * jax.random.normal(k2, (60000,), dtype=jnp.float64)
t0 = time.perf_counter(); out = f(jax.random.key(3)).block_until_ready(); tc = time.perf_counter() - t0
ts = []
for i in range(30):
    t0 = time.perf_counter(); f(jax.random.key(i)).block_until_ready(); ts.append(time.perf_counter() - t0)
np.save(f"/tmp/tf_{sys.argv[1]}.npy", np.asarray(out))
print(sys.argv[1], f"compile+run {tc:.2f}s  run {np.median(ts)*1e3:.3f} ms")
