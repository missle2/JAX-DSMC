import sys
exec(open("/home/aakash/jax_dsmc/JAX-DSMC-sparta/scratch/sparta_jax/acct/prims_cpu.py").read().split('xi = (x')[0])
from jax import lax
mi = (x < 0.1).astype(jnp.int32)

def f32_to_i32(v):  # exact for integer-valued v in [0, 2^22)
    return lax.bitcast_convert_type(v + 12582912.0, jnp.int32) - 0x4B400000

def cs_matmul(m, B=32):
    n = m.shape[0]
    a = m.astype(jnp.float32).reshape(-1, B)
    tri = jnp.triu(jnp.ones((B, B), jnp.float32))
    inner = f32_to_i32(a @ tri)
    tot = inner[:, -1]
    off = jnp.cumsum(tot) - tot
    return (inner + off[:, None]).reshape(n)

def cs_matmul2(m, B=32):
    n = m.shape[0]
    a = m.astype(jnp.float32).reshape(-1, B)
    tri = jnp.triu(jnp.ones((B, B), jnp.float32))
    inner = f32_to_i32(a @ tri)
    tot = inner[:, -1]
    R = tot.shape[0]
    if R % B == 0 and R > 4 * B:
        off = cs_matmul2(tot, B) - tot
    else:
        off = jnp.cumsum(tot) - tot
    return (inner + off[:, None]).reshape(n)

def cs_concat(m, B=8):
    n = m.shape[0]
    a = m.reshape(-1, B)
    cols = [a[:, 0]]
    for k in range(1, B):
        cols.append(cols[-1] + a[:, k])
    inner = jnp.stack(cols, axis=1)
    tot = inner[:, -1]
    off = (cs_concat(tot, B) if tot.shape[0] % B == 0 and tot.shape[0] > 64 else jnp.cumsum(tot)) - tot
    return (inner + off[:, None]).reshape(n)

def cs_rowsum(m, B=16):
    """block totals by a reduce, offsets by small cumsum, inner prefix by cumsum along axis 1"""
    n = m.shape[0]
    a = m.reshape(-1, B)
    inner = jnp.cumsum(a, axis=1)
    tot = inner[:, -1]
    off = jnp.cumsum(tot) - tot
    return (inner + off[:, None]).reshape(n)

ref = np.cumsum(np.asarray(mi))
for name, f in [("matmul32", cs_matmul), ("matmul2_32", cs_matmul2), ("concat8", cs_concat), ("rowsum16", cs_rowsum),
                ("matmul64", lambda m: cs_matmul(m, 64)), ("concat16", lambda m: cs_concat(m, 16))]:
    assert np.array_equal(np.asarray(jax.jit(f)(mi)), ref), name
for n in (53248, 5120):
    a = mi[:n]
    timeit(f"[{n}] i32 identity", lambda a: a ^ 1, a)
    timeit(f"[{n}] jnp.cumsum", lambda a: jnp.cumsum(a) & 1, a)
    timeit(f"[{n}] assoc scan", lambda a: lax.associative_scan(jnp.add, a) & 1, a)
    timeit(f"[{n}] matmul32", lambda a: cs_matmul(a) & 1, a)
    timeit(f"[{n}] matmul2_32", lambda a: cs_matmul2(a) & 1, a)
    timeit(f"[{n}] matmul64", lambda a: cs_matmul(a, 64) & 1, a)
    timeit(f"[{n}] matmul128", lambda a: cs_matmul(a, 128) & 1, a)
    timeit(f"[{n}] concat8", lambda a: cs_concat(a) & 1, a)
    timeit(f"[{n}] concat16", lambda a: cs_concat(a, 16) & 1, a)
    timeit(f"[{n}] rowsum16", lambda a: cs_rowsum(a) & 1, a)
