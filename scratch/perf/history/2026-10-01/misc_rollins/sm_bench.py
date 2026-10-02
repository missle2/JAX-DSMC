import sys, time
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/home/aakash/jax_dsmc/JAX-DSMC-sparta")
from sparta_jax import rng as R
@jax.jit
def f(seed):
    k = R.step_key(seed, 3, 1)
    u = R.uniform(k, jnp.arange(60000 * 3).reshape(60000, 3))
    k2 = R.step_key(seed, 3, 2)
    z = R.uniform(k2, jnp.arange(60000))
    return u.sum(1) * z
@jax.jit
def g(key):
    return jax.random.uniform(key, (60000, 3), dtype=jnp.float64).sum(1) * jax.random.uniform(jax.random.fold_in(key, 1), (60000,), dtype=jnp.float64)
for name, h, a in (("splitmix", f, lambda i: jnp.uint64(i)), ("threefry", g, lambda i: jax.random.key(i))):
    h(a(0)).block_until_ready(); ts = []
    for i in range(30):
        t0 = time.perf_counter(); h(a(i)).block_until_ready(); ts.append(time.perf_counter() - t0)
    print(name, f"{np.median(ts)*1e3:.3f} ms")
