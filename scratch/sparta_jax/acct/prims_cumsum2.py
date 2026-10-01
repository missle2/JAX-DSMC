import sys
exec(open("/home/aakash/jax_dsmc/JAX-DSMC-sparta/scratch/sparta_jax/acct/prims_cpu.py").read().split('xi = (x')[0])
sys.path.insert(0, "/home/aakash/jax_dsmc/JAX-DSMC-sparta")
from sparta_jax.sim import materialise
ci = (x * 801).astype(jnp.int32)

def cs_rw(m, B=8, mat=True):
    n = m.shape[0]
    if n <= 4 * B or n % B:
        return jnp.cumsum(m, dtype=jnp.int32)
    inner = jnp.cumsum(m.reshape(-1, B), axis=1, dtype=jnp.int32)
    tot = inner[:, -1]
    off = cs_rw(tot, B, mat) - tot
    if mat:
        off = materialise(off)
    return (inner + off[:, None]).reshape(n)

def cs_log(m, B=8, mat=True):
    n = m.shape[0]
    if n <= 4 * B or n % B:
        return jnp.cumsum(m, dtype=jnp.int32)
    y = m.reshape(-1, B)
    s = 1
    while s < B:
        y = y + jnp.pad(y[:, :-s], ((0, 0), (s, 0)))
        s *= 2
    tot = y[:, -1]
    off = cs_log(tot, B, mat) - tot
    if mat:
        off = materialise(off)
    return (y + off[:, None]).reshape(n)

ref = np.cumsum(np.asarray(ci) > 400)
for nm, f in [("rw8", cs_rw), ("log8", cs_log), ("rw16", lambda m: cs_rw(m, 16)), ("rw32", lambda m: cs_rw(m, 32))]:
    assert np.array_equal(np.asarray(jax.jit(lambda c: f((c > 400).astype(jnp.int32)))(ci)), ref), nm
# realistic: mask from a compare, result used as a scatter index and the total
def use(f):
    def g(c):
        m = c > 400
        r = f(m.astype(jnp.int32)) - 1
        idx = jnp.full(5120, N, jnp.int32).at[jnp.where(m & (r < 5120), r, 5120)].set(jnp.arange(N, dtype=jnp.int32), mode="drop")
        return (c + idx[c % 5120 if False else 0] + r[-1]) % 801
    return g
timeit("identity", lambda c: (c + 1) % 801, ci)
timeit("use jnp.cumsum", use(lambda m: jnp.cumsum(m, dtype=jnp.int32)), ci)
for B in (8, 16, 32):
    timeit(f"use rw{B} mat", use(lambda m, B=B: cs_rw(m, B)), ci)
    timeit(f"use rw{B} nomat", use(lambda m, B=B: cs_rw(m, B, False)), ci)
timeit("use log8 mat", use(lambda m: cs_log(m, 8)), ci)
timeit("use log8 nomat", use(lambda m: cs_log(m, 8, False)), ci)
