import sys
sys.argv = sys.argv[:1]
exec(open("/home/aakash/jax_dsmc/JAX-DSMC-sparta/scratch/sparta_jax/acct/prims_cpu.py").read().split('xi = (x')[0])
from jax import lax
xi = (x * 400).astype(jnp.int32)
mi = (x < 0.1).astype(jnp.int32)
MAGIC = 6755399441055744.0

def floor_magic(v):
    t = v + MAGIC
    r = t - MAGIC
    k = lax.bitcast_convert_type(t, jnp.int64).astype(jnp.int32)
    return k - (r > v).astype(jnp.int32)

def cumsum_blocked(m, B=16):
    n = m.shape[0]
    a = m.reshape(-1, B)
    y = a
    s = 1
    while s < B:
        y = y + jnp.pad(y[:, :-s], ((0, 0), (s, 0)))
        s *= 2
    tot = y[:, -1]
    if tot.shape[0] > 256:
        off = cumsum_blocked(tot, B) - tot
    else:
        off = jnp.cumsum(tot) - tot
    return (y + off[:, None]).reshape(n)

assert np.array_equal(np.asarray(cumsum_blocked(mi)), np.cumsum(np.asarray(mi)))
v = np.asarray(x) * 20
for t in [0.0, 1.0, 2.0, 19.999999999, 3.0000001, 7.5]:
    v[0] = t
    assert int(floor_magic(jnp.asarray(v))[0]) == int(np.floor(t)), t
w = x * 20
timeit("identity (x*0.999+1e-3)", lambda a: a * 0.999 + 1e-3, x)
timeit("i32 identity (a ^ 1)", lambda a: a ^ 1, mi)
timeit("cumsum i32 N", lambda a: jnp.cumsum(a) & 1, mi)
timeit("cumsum blocked4 i32 N", lambda a: cumsum_blocked(a, 4) & 1, mi)
timeit("cumsum blocked8 i32 N", lambda a: cumsum_blocked(a, 8) & 1, mi)
timeit("cumsum blocked16 i32 N", lambda a: cumsum_blocked(a, 16) & 1, mi)
timeit("sum i32 N (+bcast)", lambda a: a ^ (jnp.sum(a, dtype=jnp.int32) & 1), mi)
timeit("sum i32 blocked 128", lambda a: a ^ (jnp.sum(jnp.sum(a.reshape(-1, 128), axis=1, dtype=jnp.int32), dtype=jnp.int32) & 1), mi)
timeit("sum i32 blocked rows of 8", lambda a: a ^ (jnp.sum(jnp.sum(a.reshape(-1, 8), axis=1, dtype=jnp.int32), dtype=jnp.int32) & 1), mi)
timeit("sum f64 N (+bcast)", lambda a: a * 0.5 + jnp.sum(a) * 1e-9, x)
timeit("gather f64 401-table, int idx", lambda a, c: tab1[c] * 0.5 + a * 0.5, x, cell)
timeit("4 gathers 401-tables, int idx", lambda a, c: (tab[:, 0][c] + tab[:, 1][c] + tab[:, 2][c] + tab[:, 3][c]) * 0.1 + a * 0.5, x, cell)
timeit("gather f64 random N", lambda a, p: a[p], x, perm)
timeit("scatter set f64 random N", lambda a, p: jnp.zeros(N).at[p].set(a), x, perm)
timeit("scatter set f64 random N, mode drop", lambda a, p: jnp.zeros(N).at[p].set(a, mode="drop"), x, perm)
timeit("scatter i32 into 8192, mostly dropped", lambda a, p: jnp.zeros(N, jnp.int32).at[:8192].set(jnp.zeros(8192, jnp.int32).at[jnp.where(p < 4000, p, 8192)].set(a, mode="drop")) + a, mi, perm)
