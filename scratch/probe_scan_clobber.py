"""Probe B: does an inactive (empty / singleton) cell clobber an active cell's
collision write in the spec 8.4 attempt-slot scan?  Concrete 6-cell case."""

import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from jax import random
import numpy as np

np.set_printoptions(precision=3, suppress=True, linewidth=200)


# ---------------------------------------------------------------- sort table (spec 6)
def build_table(cell_of, C, N):
    c = jnp.asarray(cell_of, dtype=jnp.int32)
    order = jnp.argsort(c)
    counts = jnp.bincount(c, length=C + 1)
    starts = jnp.cumsum(counts) - counts
    return order, starts[:C], counts[:C]


# 6 cells, 10 particles.  cell 1 EMPTY, cell 3 EMPTY, cell 4 singleton.
#   cell 0: particles 0,1        (count 2)
#   cell 1: -                    (count 0)   <-- empty, starts[1] == starts[2]
#   cell 2: particles 2,3,4,5    (count 4)
#   cell 3: -                    (count 0)   <-- empty
#   cell 4: particle 6           (count 1)
#   cell 5: particles 7,8,9      (count 3)
cell_of = [0, 0, 2, 2, 2, 2, 4, 5, 5, 5]
N, C = 10, 6
order, starts, counts = build_table(cell_of, C, N)
print("order  =", order)
print("starts =", starts)
print("counts =", counts)
for c in range(C):
    print(
        f"  cell {c}: count={int(counts[c])} start={int(starts[c])} "
        f"order[start]={int(order[jnp.minimum(starts[c], N - 1)])}"
    )


# ---------------------------------------------------------------- spec 8.3 pair_ntc
def pair_ntc(key, npc):
    k1, k2 = random.split(key)
    i = random.randint(k1, npc.shape, 0, npc)
    r = random.randint(k2, npc.shape, 0, npc - 1)
    j = (i + 1 + r) % npc
    return i, j


# ---------------------------------------------------------------- one slot, spec 8.4 verbatim
def slot_spec(v, order, starts, counts, n, a, key, force_ij=None):
    active = (a < n) & (counts >= 2)
    if force_ij is None:
        i, j = pair_ntc(random.fold_in(key, a), counts)
    else:
        i, j = force_ij
    pi = order[starts + i]
    pj = order[starts + j]
    vi, vj = v[pi], v[pj]
    accept = active  # force p = 1 so every active slot collides
    # momentum- and energy-conserving "scatter": swap the two velocities
    vi_n, vj_n = vj, vi
    v = v.at[pi].set(jnp.where(accept, vi_n, vi))
    v = v.at[pj].set(jnp.where(accept, vj_n, vj))
    return v, (active, i, j, pi, pj)


v0 = jnp.arange(1.0, N + 1.0)  # velocity of particle p is p+1, easy to track
n = jnp.array([1, 1, 1, 1, 1, 1], dtype=jnp.int32)  # 1 attempt everywhere

print("\n================= DEMO 1: forced pairing, cell 2 picks its particle 0 =================")
# cell 1 is empty -> starts[1]=2 -> pi=pj=order[2]=particle 2 (cell 2's particle 0)
# make cell 2 pick i=0 (particle 2) and j=1 (particle 3) -> a legitimate collision on particle 2
i_f = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.int32)
j_f = jnp.array([1, 0, 1, 0, 0, 1], dtype=jnp.int32)
v1, info = slot_spec(v0, order, starts, counts, n, 0, random.key(0), (i_f, j_f))
active, i, j, pi, pj = info
print("active =", active)
print("pi     =", pi)
print("pj     =", pj)
print("v before =", v0)
print("v after  =", v1)
print("\nExpected if cells were disjoint (only active cells 0,2,5 swap their pairs):")
exp = np.array(v0)
for c in [0, 2, 5]:
    a_, b_ = int(pi[c]), int(pj[c])
    exp[a_], exp[b_] = exp[b_], exp[a_]
print("v expected =", exp)
print("MATCH      =", bool(jnp.allclose(v1, exp)))
print("momentum before/after:", float(v0.sum()), float(v1.sum()), " expected:", float(exp.sum()))
print(
    "energy   before/after:",
    float((v0**2).sum()),
    float((v1**2).sum()),
    " expected:",
    float((exp**2).sum()),
)

print("\n================= DEMO 2: real random pairing, 200 seeds =================")
bad_mom = 0
bad_any = 0
for s in range(200):
    v1, info = slot_spec(v0, order, starts, counts, n, 0, random.key(s))
    active, i, j, pi, pj = info
    exp = np.array(v0)
    for c in range(C):
        if bool(active[c]):
            a_, b_ = int(pi[c]), int(pj[c])
            exp[a_], exp[b_] = exp[b_], exp[a_]
    if not bool(jnp.allclose(v1, exp)):
        bad_any += 1
    if abs(float(v1.sum()) - float(v0.sum())) > 1e-9:
        bad_mom += 1
print(f"  slots where result != disjoint-cell expectation : {bad_any}/200")
print(f"  slots where TOTAL MOMENTUM is not conserved     : {bad_mom}/200")

# show one failing seed in full
for s in range(200):
    v1, info = slot_spec(v0, order, starts, counts, n, 0, random.key(s))
    if abs(float(v1.sum()) - float(v0.sum())) > 1e-9:
        active, i, j, pi, pj = info
        print(f"\n  --- failing seed {s} ---")
        print("   active=", active, " i=", i, " j=", j)
        print("   pi=", pi, " pj=", pj)
        print("   v before =", v0)
        print("   v after  =", v1)
        print("   momentum", float(v0.sum()), "->", float(v1.sum()))
        break

print("\n================= DEMO 3: trailing empty cells -> out-of-bounds index =================")
cell_of2 = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]  # cells 3,4,5 all empty and TRAILING
order2, starts2, counts2 = build_table(cell_of2, C, N)
print("counts=", counts2, " starts=", starts2, "  (starts[3..5] ==", int(starts2[3]), "== N)")
print(
    "order2[N] out-of-bounds gather ->",
    int(order2[starts2[3]]),
    " (JAX clamps to order[N-1] =",
    int(order2[N - 1]),
    ")",
)
i_f = jnp.zeros(C, dtype=jnp.int32)
j_f = jnp.array([1, 1, 1, 0, 0, 0], dtype=jnp.int32)
v2, info2 = slot_spec(v0, order2, starts2, counts2, n, 0, random.key(0), (i_f, j_f))
print("pi=", info2[3], " pj=", info2[4])
print("v before =", v0)
print("v after  =", v2)
print("momentum", float(v0.sum()), "->", float(v2.sum()))
