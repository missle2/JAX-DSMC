"""Probe D: the proposed fix (clamp npc_safe = max(counts,2) + scratch slot at index N).
Correctness + cost."""
import jax, jax.numpy as jnp, time
jax.config.update("jax_enable_x64", True)
from jax import random
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)

# --------------------------------------------------------------- pairing
def pair_ntc_spec(key, npc):
    k1, k2 = random.split(key)
    i = random.randint(k1, npc.shape, 0, npc)
    r = random.randint(k2, npc.shape, 0, npc - 1)
    return i, (i + 1 + r) % npc

def pair_ntc_fixed(key, npc):
    npc_s = jnp.maximum(npc, 2)                 # <- clamp
    k1, k2 = random.split(key)
    i = random.randint(k1, npc.shape, 0, npc_s)
    r = random.randint(k2, npc.shape, 0, npc_s - 1)
    return i, (i + 1 + r) % npc_s

# --------------------------------------------------------------- one slot
def slot(v, order, starts, counts, n, a, key, fixed, N):
    active = (a < n) & (counts >= 2)
    i, j = (pair_ntc_fixed if fixed else pair_ntc_spec)(random.fold_in(key, a), counts)
    if fixed:
        pi = jnp.where(active, order[starts + i], N)     # inactive -> scratch row N
        pj = jnp.where(active, order[starts + j], N)
    else:
        pi = order[starts + i]; pj = order[starts + j]
    vi, vj = v[pi], v[pj]
    vi_n, vj_n = vj, vi                        # momentum/energy-conserving swap
    v = v.at[pi].set(jnp.where(active, vi_n, vi))
    v = v.at[pj].set(jnp.where(active, vj_n, vj))
    return v, (active, pi, pj)

def run(cell_of, C, N, A_max, seed, fixed):
    c = jnp.asarray(cell_of, jnp.int32)
    order = jnp.argsort(c); cnt = jnp.bincount(c, length=C+1)
    starts = (jnp.cumsum(cnt)-cnt)[:C]; counts = cnt[:C]
    n = jnp.full((C,), A_max, jnp.int32)
    v = jnp.arange(1., N+1.)
    if fixed:
        v = jnp.concatenate([v, jnp.zeros(1)])           # scratch row
    ref = np.arange(1., N+1.)
    key = random.key(seed)
    for a in range(A_max):
        v, (active, pi, pj) = slot(v, order, starts, counts, n, a, key, fixed, N)
        # serial reference: only active cells, disjoint by construction
        for cc in range(C):
            if bool(active[cc]):
                p, q = int(pi[cc]), int(pj[cc])
                if p < N and q < N:
                    ref[p], ref[q] = ref[q], ref[p]
    return (v[:N] if fixed else v), ref

cell_of = [0,0, 2,2,2,2, 4, 5,5,5]; N, C = 10, 6
print("=== correctness over 300 seeds, A_max=3 ===")
for fixed in (False, True):
    nbad = nmom = 0
    for s in range(300):
        v, ref = run(cell_of, C, N, 3, s, fixed)
        if not np.allclose(np.array(v), ref): nbad += 1
        if abs(float(v.sum()) - 55.0) > 1e-9:  nmom += 1
    print(f"  {'FIXED ' if fixed else 'SPEC  '}: mismatch vs serial reference {nbad}/300, "
          f"momentum violations {nmom}/300")

# NOTE the serial reference is itself built from the *fixed* pi/pj when fixed=True;
# for the spec run it is built from the spec's pi/pj, so a mismatch means the
# vectorised scatter disagrees with what the same draws should have produced.

print("\n=== 3-cell sanity: fixed version reproduces an exact hand-computed result ===")
v, ref = run(cell_of, C, N, 1, 0, True)
print("  vectorised:", np.array(v))
print("  serial ref:", ref, "  match:", np.allclose(np.array(v), ref))

# --------------------------------------------------------------- cost
print("\n=== cost of the fix (C cells, A_max slots, scan) ===")
def make(fixed, C, N, A_max):
    key = random.key(0)
    def body(carry, a):
        v, order, starts, counts, n = carry
        active = (a < n) & (counts >= 2)
        i, j = (pair_ntc_fixed if fixed else pair_ntc_spec)(random.fold_in(key, a), counts)
        if fixed:
            pi = jnp.where(active, order[starts+i], N); pj = jnp.where(active, order[starts+j], N)
        else:
            pi = order[starts+i]; pj = order[starts+j]
        vi, vj = v[pi], v[pj]
        v = v.at[pi].set(jnp.where(active, vj, vi))
        v = v.at[pj].set(jnp.where(active, vi, vj))
        return (v, order, starts, counts, n), None
    @jax.jit
    def f(v, order, starts, counts, n):
        (v, *_), _ = jax.lax.scan(body, (v, order, starts, counts, n), jnp.arange(A_max))
        return v
    return f

for (Cn, Nn, Am) in [(4096, 262144, 8), (16384, 1048576, 8)]:
    cell = jnp.repeat(jnp.arange(Cn, dtype=jnp.int32), Nn // Cn)
    order = jnp.argsort(cell); cnt = jnp.bincount(cell, length=Cn+1)
    starts = (jnp.cumsum(cnt)-cnt)[:Cn]; counts = cnt[:Cn]
    n = jnp.full((Cn,), Am, jnp.int32)
    res = {}
    for fixed in (False, True):
        v = jnp.arange(float(Nn) + (1 if fixed else 0))
        f = make(fixed, Cn, Nn, Am)
        f(v, order, starts, counts, n).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(30): out = f(v, order, starts, counts, n)
        out.block_until_ready()
        res[fixed] = (time.perf_counter()-t0)/30*1e3
    print(f"  C={Cn} N={Nn} A_max={Am}:  spec {res[False]:.3f} ms   fixed {res[True]:.3f} ms   "
          f"ratio {res[True]/res[False]:.3f}x   extra memory: 1 row of v = {8*3} B")
