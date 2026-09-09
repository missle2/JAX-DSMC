"""Can the O(N)-per-slot carry copy be avoided?

Audit finding A: a `lax.scan` whose carry is an (N+1,3) array updated by
`.at[idx].set(...)` costs O(N) per iteration, not O(C).  At N=2e6 that is 25x
the cost of the same loop without the write, and the gap grows linearly.

This tries every formulation that might let XLA update the buffer in place,
so the fix is chosen by measurement rather than by guesswork.
"""
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

jax.config.update("jax_enable_x64", True)

A, C = 256, 64


def bench(fn, *a, reps=5, **kw):
    jax.block_until_ready(fn(*a, **kw))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*a, **kw))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# --- 1. scan + .at[].set  (what collide does today) -------------------------
def b_scan_set(carry, xs):
    v, acc = carry
    idx, u = xs
    vi = v[idx]
    v = v.at[idx].set(vi + u[:, None])
    return (v, acc + jnp.sum(vi)), None


def f_scan_set(N):
    v = jnp.zeros((N + 1, 3)); idx = jnp.zeros((A, C), jnp.int32); u = jnp.ones((A, C))
    return lax.scan(b_scan_set, (v, jnp.float64(0.0)), (idx, u))[0][1]


# --- 2. fori_loop + .at[].set ----------------------------------------------
def f_fori(N):
    v0 = jnp.zeros((N + 1, 3)); idx = jnp.zeros((A, C), jnp.int32); u = jnp.ones((A, C))

    def body(a, carry):
        v, acc = carry
        vi = v[idx[a]]
        return (v.at[idx[a]].set(vi + u[a][:, None]), acc + jnp.sum(vi))

    return lax.fori_loop(0, A, body, (v0, jnp.float64(0.0)))[1]


# --- 3. scan + .at[].add ----------------------------------------------------
def b_scan_add(carry, xs):
    v, acc = carry
    idx, u = xs
    vi = v[idx]
    v = v.at[idx].add(u[:, None])
    return (v, acc + jnp.sum(vi)), None


def f_scan_add(N):
    v = jnp.zeros((N + 1, 3)); idx = jnp.zeros((A, C), jnp.int32); u = jnp.ones((A, C))
    return lax.scan(b_scan_add, (v, jnp.float64(0.0)), (idx, u))[0][1]


# --- 4. COMPACT WORKSPACE: gather the cell's particles once, scan over a
#        (C, K, 3) buffer with cell-local indices, scatter back once.
#        K is the static per-cell capacity, so the carry is C*K*3, not N*3.
def b_workspace(carry, xs, C_):
    w, acc = carry            # w: (C, K, 3)
    loc, u = xs               # loc: (C,) local slot in [0, K)
    rows = jnp.arange(C_)
    vi = w[rows, loc]
    w = w.at[rows, loc].set(vi + u[:, None])
    return (w, acc + jnp.sum(vi)), None


def f_workspace(N, K=64):
    w = jnp.zeros((C, K, 3))
    loc = jnp.zeros((A, C), jnp.int32); u = jnp.ones((A, C))
    body = functools.partial(b_workspace, C_=C)
    out = lax.scan(body, (w, jnp.float64(0.0)), (loc, u))[0]
    # one scatter back into the global array, outside the loop
    v = jnp.zeros((N + 1, 3))
    v = v.at[jnp.arange(C * K) % (N + 1)].set(out[0].reshape(-1, 3))
    return out[1] + jnp.sum(v)


VARIANTS = {
    "scan + .at[].set (current)": f_scan_set,
    "fori_loop + .at[].set": f_fori,
    "scan + .at[].add": f_scan_add,
    "compact (C,K,3) workspace": f_workspace,
}

print(f"A_max={A}, C={C}.  us per slot; the ideal is flat in N.\n")
print(f"{'variant':<30} " + " ".join(f"{n:>12,}" for n in (64_000, 512_000, 2_000_000)))
print("-" * 74)
for name, fn in VARIANTS.items():
    j = jax.jit(fn, static_argnums=(0,))
    cells = []
    for N in (64_000, 512_000, 2_000_000):
        try:
            cells.append(f"{bench(j, N) / A * 1e6:>12.1f}")
        except Exception as e:  # noqa: BLE001
            cells.append(f"{type(e).__name__:>12}")
    print(f"{name:<30} " + " ".join(cells))

print("\nflat in N  => the carry is updated in place")
print("grows in N => the carry is copied each iteration (O(N) per slot)")
