"""Does the scan carry get copied?  Sweep BOTH N and C, one harness.

`perf_audit.py` finding A used C=1 and saw the scatter cost grow 25x with N.
`perf_inplace.py` used C=64 and saw every variant flat.  Those cannot both be
the whole story, and C=1 is the unrepresentative 0-D shape that has already
misled this project twice -- so measure the two axes together.
"""
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

jax.config.update("jax_enable_x64", True)

A = 256


def b_write(carry, xs):
    v, acc = carry
    idx, u = xs
    vi = v[idx]
    v = v.at[idx].set(vi + u[:, None])
    return (v, acc + jnp.sum(vi)), None


def b_read(carry, xs):
    v, acc = carry
    idx, u = xs
    return (v, acc + jnp.sum(v[idx]) + jnp.sum(u)), None


def run(N, C, body):
    v = jnp.zeros((N + 1, 3))
    idx = jnp.zeros((A, C), jnp.int32)
    u = jnp.ones((A, C))
    return lax.scan(body, (v, jnp.float64(0.0)), (idx, u))[0][1]


J = jax.jit(run, static_argnums=(0, 1, 2))


def bench(N, C, body, reps=5):
    jax.block_until_ready(J(N, C, body))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(J(N, C, body))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / A * 1e6


NS = (64_000, 512_000, 2_000_000, 8_000_000)
CS = (1, 8, 64, 512, 4096)

print(f"A_max = {A}.  us per slot for the WRITING body (v.at[idx].set).")
print("A row that grows with N means the (N+1,3) carry is copied per iteration.\n")
print(f"{'C':>7} " + " ".join(f"{n:>12,}" for n in NS) + f" {'growth':>9}")
print("-" * (8 + 13 * len(NS) + 10))
for C in CS:
    row = [bench(N, C, b_write) for N in NS]
    print(f"{C:>7,} " + " ".join(f"{t:>12.1f}" for t in row) + f" {row[-1] / row[0]:>8.1f}x")

print(f"\n{'C':>7} " + " ".join(f"{n:>12,}" for n in NS) + f" {'growth':>9}   (READ-ONLY control)")
print("-" * (8 + 13 * len(NS) + 10))
for C in CS:
    row = [bench(N, C, b_read) for N in NS]
    print(f"{C:>7,} " + " ".join(f"{t:>12.1f}" for t in row) + f" {row[-1] / row[0]:>8.1f}x")
