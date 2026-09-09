"""How large can the BKW gate go if the cell count scales WITH N?

The gate currently pins `cells-per-side`, so `npc = N / C` grows with `N` and
`A_max ~ npc` grows with it -- the attempt-slot scan gets longer and longer while
the parallel (cell) axis stays fixed at 8.  That is backwards: the scan is the
SEQUENTIAL axis and cells are the PARALLEL one.

Holding `npc` fixed instead makes `A_max` constant and puts all the growth on the
cell axis, which is the production regime.  Total attempts stay ~ N either way,
so this is the same amount of physics arranged for the hardware.
"""
import math
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from examples.bkw import _relax, initial_state, measured_excess, setup, rate_ols


def side_for(n_part, npc_target):
    """Cells per axis that gets closest to `npc_target` particles per cell."""
    return max(1, round((n_part / npc_target) ** (1.0 / 3.0)))


def one(n_part, npc_target, n_steps=30, seeds=3):
    side = side_for(n_part, npc_target)
    sp, grid, static, params, cstate, nu, dt, temp = setup(
        n_part, n_steps, steps_per_tau=0.15, side=side
    )
    fn = jax.jit(_relax, static_argnames=("static",))
    rates, ts = [], []
    t_compile = None
    for s in range(seeds):
        key = jax.random.PRNGKey(s)
        state = initial_state(key, static, grid, temp)
        t0 = time.perf_counter()
        (_f, _), (em, v, c) = fn(state, cstate, grid, sp, params, static,
                                 jax.random.fold_in(key, 99))
        em.block_until_ready()
        el = time.perf_counter() - t0
        if t_compile is None:
            t_compile = el
        else:
            ts.append(el)
        e = np.concatenate([[float(measured_excess(state.v))], np.asarray(em)])
        rates.append(rate_ols(e, dt * nu, 0.5))
        viol, clip = int(np.sum(v)), int(np.sum(c))
    return dict(side=side, C=side ** 3, npc=n_part // side ** 3, A_max=static.A_max,
                compile=t_compile, run=float(np.median(ts)) if ts else float("nan"),
                rate=float(np.mean(rates)), viol=viol, clip=clip)


target = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
print(f"backend: {jax.default_backend()}   devices: {jax.devices()}")
print(f"\n=== npc held near {target}: A_max stays flat, cells carry the growth ===")
print(f"{'N':>12} {'cells':>8} {'npc':>7} {'A_max':>7} {'compile s':>10} {'run s':>9} "
      f"{'us/particle':>12} {'rate':>9} {'viol':>5}")
for N in (10_000, 40_000, 160_000, 640_000, 2_560_000, 10_240_000):
    try:
        r = one(N, target)
    except Exception as e:  # noqa: BLE001
        print(f"{N:>12,}  FAILED: {type(e).__name__}: {str(e)[:90]}")
        break
    print(f"{N:>12,} {r['C']:>8,} {r['npc']:>7,} {r['A_max']:>7,} {r['compile']:>10.2f} "
          f"{r['run']:>9.3f} {r['run'] / N * 1e6:>12.3f} {r['rate']:>9.4f} {r['viol']:>5}")

print(f"\n=== for contrast: cells PINNED at 8, so A_max ~ N (what the gate does now) ===")
print(f"{'N':>12} {'cells':>8} {'npc':>9} {'A_max':>8} {'compile s':>10} {'run s':>9}")
for N in (10_000, 40_000, 160_000, 640_000):
    try:
        r = one(N, N / 8)
    except Exception as e:  # noqa: BLE001
        print(f"{N:>12,}  FAILED: {type(e).__name__}: {str(e)[:90]}")
        break
    print(f"{N:>12,} {r['C']:>8,} {r['npc']:>9,} {r['A_max']:>8,} {r['compile']:>10.2f} "
          f"{r['run']:>9.3f}")
