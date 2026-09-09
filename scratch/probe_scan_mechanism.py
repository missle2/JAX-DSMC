"""Probe C: (1) JAX duplicate-index .at[].set semantics, (2) which of the two
scatters actually causes the 8.4 corruption."""
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax import random
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)

print("=== 1. duplicate-index .at[].set : who wins? ===")
base = jnp.zeros(4)
idx  = jnp.array([1, 1, 1, 2])
upd  = jnp.array([10., 20., 30., 40.])
print("  eager  :", base.at[idx].set(upd))
print("  jit    :", jax.jit(lambda b,i,u: b.at[i].set(u))(base, idx, upd))
print("  cpu    :", jax.jit(lambda b,i,u: b.at[i].set(u), device=jax.devices('cpu')[0])(base, idx, upd))
# is it stable across repeats / sizes?
outs = {tuple(np.array(jax.jit(lambda b,i,u: b.at[i].set(u))(base, idx, upd))) for _ in range(50)}
print("  distinct results over 50 identical jit calls:", outs)
big_i = jnp.zeros(4096, dtype=jnp.int32)
big_u = jnp.arange(4096.)
r = jax.jit(lambda i,u: jnp.zeros(1).at[i].set(u))(big_i, big_u)
print("  4096 duplicate writes to index 0 -> winner =", float(r[0]), "(0=first, 4095=last)")
print("  docs: jnp .at[].set with duplicate indices -> UNSPECIFIED which update wins")
print("        (jax.ops / .at docs: 'duplicate indices ... the result is nondeterministic'")
print("         only .add/.mul/.min/.max are well defined under duplicates)")

print("\n=== 2. mechanism: S1 vs S2 ===")
cell_of = [0,0, 2,2,2,2, 4, 5,5,5]; N, C = 10, 6
c = jnp.asarray(cell_of, jnp.int32)
order = jnp.argsort(c); cnt = jnp.bincount(c, length=C+1)
starts = (jnp.cumsum(cnt)-cnt)[:C]; counts = cnt[:C]
v0 = jnp.arange(1., N+1.)
i_f = jnp.array([0,0,0,0,0,0], jnp.int32); j_f = jnp.array([1,0,1,0,0,1], jnp.int32)
active = (0 < jnp.ones(C, jnp.int32)) & (counts >= 2)
pi = order[starts + i_f]; pj = order[starts + j_f]
vi, vj = v0[pi], v0[pj]
vi_n, vj_n = vj, vi
print("  pi=", pi, " pj=", pj, " active=", active)
s1 = v0.at[pi].set(jnp.where(active, vi_n, vi))
print("  after S1 (v.at[pi].set):", s1, "   <- v[2]:", float(s1[2]), "(cell2 wants 4, cell1 wants 3)")
s2 = s1.at[pj].set(jnp.where(active, vj_n, vj))
print("  after S2 (v.at[pj].set):", s2, "   <- v[2]:", float(s2[2]))
print("  NOTE: cell1 (empty) has pj=2 and its vj was gathered from v BEFORE S1,")
print("        so S2 writes the STALE 3.0 over whatever S1 left at index 2.")
print("        -> the clobber is DETERMINISTIC, not merely a duplicate-index race.")
