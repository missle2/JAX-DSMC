"""XLA:CPU per-primitive cost at the deck's size (N = 53248), 1 core.  Each case is a loop body
whose result feeds the carry; the 'identity' case (one f64 pass) is the baseline to subtract."""
import time
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
N = 53248
r = np.random.default_rng(0)
x = jnp.asarray(r.random(N))
perm = jnp.asarray(r.permutation(N).astype(np.int32))
near_perm = jnp.asarray(np.argsort(np.arange(N) + r.integers(-50, 50, N)).astype(np.int32))
cell = jnp.asarray(r.integers(0, 400, N).astype(np.int32))
tab = jnp.asarray(r.random((401, 4)))
tab1 = jnp.asarray(r.random(401))
m = jnp.asarray(r.random(N) < 0.1)
k8 = jnp.asarray(r.integers(0, 400, 8192).astype(np.int32))
REP = 400

def timeit(name, f, *args):
    """f(carry_array, *rest) -> array of the same shape/dtype as the carry (fed back)."""
    def body(i, c):
        return f(c, *args[1:])
    g = jax.jit(lambda: jax.lax.fori_loop(0, REP, body, args[0]))
    jax.block_until_ready(g())
    best = 1e9
    for _ in range(5):
        t = time.perf_counter(); jax.block_until_ready(g()); best = min(best, time.perf_counter() - t)
    print(f"{name:34s} {best / REP * 1e6:8.1f} us", flush=True)

xi = (x * 400).astype(jnp.int32)
timeit("identity (x*0.999+1e-3)", lambda a: a * 0.999 + 1e-3, x)
timeit("5 inputs elementwise", lambda a, b: a * 0.999 + 1e-3 * (b * b - b + jnp.sqrt(b)), x, x)
timeit("divide f64 N", lambda a: 1.0 / (a + 1.0), x)
timeit("floor->int->f64 N", lambda a: jnp.floor(a * 20).astype(jnp.int32).astype(jnp.float64) * 1e-3 + a * 0.5, x)
timeit("int div/mod by 20 N", lambda a: (a // 20 + (a % 20)) % 400, xi)
timeit("sum f64 N (+bcast)", lambda a: a * 0.5 + jnp.sum(a) * 1e-9, x)
timeit("sum i32 compare N (+bcast)", lambda a: a * 0.5 + jnp.sum(a > 0.5, dtype=jnp.int32) * 1e-9, x)
timeit("gather f64 random N", lambda a, p: a[p], x, perm)
timeit("gather f64 near-identity N", lambda a, p: a[p], x, near_perm)
timeit("gather 401-table f64 N", lambda a, c: tab1[c] * a, x, cell)
timeit("gather 401-table, idx from data", lambda a: tab1[(a * 400).astype(jnp.int32) % 401] * 0.5 + a * 0.5, x)
timeit("gather 401x4 rows, idx from data", lambda a: tab[(a * 400).astype(jnp.int32) % 401].sum(1) * 0.1 + a * 0.5, x)
timeit("scatter set f64 random N", lambda a, p: jnp.zeros(N).at[p].set(a), x, perm)
timeit("histogram 401 bins (N)", lambda a, c: a + jnp.zeros(401).at[c].add(a)[0] * 1e-9, x, cell)
timeit("cumsum i32 N", lambda a: jnp.cumsum(a % 2), xi)
timeit("sort i32 N", lambda a: jnp.sort(a ^ 7), xi)
timeit("sort i32 8192", lambda a: jnp.sort(a ^ 7), xi[:8192])
timeit("sort i32 4096", lambda a: jnp.sort(a ^ 7), xi[:4096])
timeit("sort i32 1024", lambda a: jnp.sort(a ^ 7), xi[:1024])
timeit("scalar op (per-op floor)", lambda a: a * 0.5 + 1.0, x[:1])
