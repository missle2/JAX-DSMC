"""Does XLA:CPU's sort run faster on nearly-sorted input (keys gathered in last step's order)?"""
import time
import jax, jax.numpy as jnp, numpy as np
from jax import lax
jax.config.update("jax_enable_x64", True)
d = np.load("/tmp/misc_state_polygon.npz")
c = np.asarray(d["cell"]); ref = np.asarray(d["order"]); N = c.shape[0]
rng = np.random.default_rng(0)
def keys64(c): return c.astype(np.int64) * N + np.arange(N)
def keys32(c): return (c.astype(np.int32) * N + np.arange(N, dtype=np.int32))
f64 = jax.jit(lambda p: lax.sort(p, is_stable=False))
def t(f, p):
    f(p).block_until_ready(); ts = []
    for _ in range(20):
        t0 = time.perf_counter(); f(p).block_until_ready(); ts.append(time.perf_counter() - t0)
    return np.median(ts) * 1e3
for frac in (None, 0.0, 0.05, 0.1, 0.2):
    cc = c.copy()
    if frac:
        m = rng.random(N) < frac
        cc[m] = rng.integers(0, 400, m.sum())
    for kind, kf in (("i64", keys64), ("i32", keys32)):
        p = kf(cc)
        arr = p if frac is None else p[ref]  # slot order (shuffled) or last order
        print(f"{'slot order' if frac is None else f'prev order, {frac:.0%} moved':22s} {kind}: {t(f64, jnp.asarray(arr)):.3f} ms", flush=True)
