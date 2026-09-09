"""Probe A: what does jax.random.randint do with degenerate / inverted ranges in JAX 0.6.2?"""

import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from jax import random

print("JAX", jax.__version__)
k = random.key(0)


def show(lo, hi, n=12):
    try:
        r = random.randint(k, (n,), lo, hi)
        print(f"  randint(shape={n}, minval={lo}, maxval={hi}) -> {r}  dtype={r.dtype}")
    except Exception as e:
        print(f"  randint(minval={lo}, maxval={hi}) RAISED {type(e).__name__}: {e}")


print("\n-- concrete (static) bounds --")
for lo, hi in [(0, 5), (0, 1), (0, 0), (0, -1), (0, -3)]:
    show(lo, hi)

print("\n-- traced/vectorised bounds (the spec's actual usage: bounds are arrays) --")
counts = jnp.array([0, 1, 2, 3, 5, 0], dtype=jnp.int32)
i = random.randint(random.fold_in(k, 1), counts.shape, 0, counts)
r = random.randint(random.fold_in(k, 2), counts.shape, 0, counts - 1)
print("  counts      =", counts)
print("  i=randint(0,counts)     =", i)
print("  r=randint(0,counts-1)   =", r)
# spec's j formula
npc = counts
j = (i + 1 + r) % jnp.where(npc == 0, 1, npc)  # guard the mod to avoid div-by-zero display
print("  j=(i+1+r)%npc (0-guard) =", j)
try:
    j_raw = (i + 1 + r) % npc
    print("  j RAW (i+1+r)%npc       =", j_raw)
except Exception as e:
    print("  j RAW raised", e)

print("\n-- repeat over many keys, does randint(0,0)/randint(0,-1) ever leave 0? --")
vals_00, vals_0m1 = [], []
for s in range(2000):
    kk = random.fold_in(k, s)
    vals_00.append(int(random.randint(kk, (), 0, 0)))
    vals_0m1.append(int(random.randint(kk, (), 0, -1)))
import collections

print("  randint(0,0) value histogram :", collections.Counter(vals_00))
print("  randint(0,-1) value histogram:", collections.Counter(vals_0m1))

print("\n-- inside jit (traced bounds), same question --")


@jax.jit
def f(key, counts):
    i = random.randint(key, counts.shape, 0, counts)
    r = random.randint(random.fold_in(key, 7), counts.shape, 0, counts - 1)
    return i, r


for s in range(5):
    ii, rr = f(random.fold_in(k, 100 + s), counts)
    print(f"   seed {s}: i={ii}  r={rr}")
