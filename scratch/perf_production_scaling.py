"""Is the collide step O(N) in the regime we actually run in?

The 0D milestone-1 case is the WORST case for the section-8.4 design and is not
representative:  C = 1, and the attempt count per cell is
``f = 0.5 npc (npc-1) Sigma dt fnum / V`` with ``npc = N``, so ``A_max ~ N``.
Every scan iteration touches the full ``(N+1, 3)`` velocity array, so with only
one cell doing one useful update per iteration the step is **O(N^2)**.

The production regime pins *particles per cell* instead and grows the cell count:
``npc`` fixed => ``A_max`` fixed, ``C = N / npc``.  Now each scan iteration still
touches O(N) memory but performs ``C`` useful updates, so the full-array cost is
amortised and the step should be **O(N)**.

This script measures both, which is the only way to tell an implementation defect
from a property of the configuration.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from scratch.perf_collide_scaling import FN, case  # noqa: E402
from jax_dsmc.sort import build_table  # noqa: E402


def timeit(N, n, A_max, reps=5):
    state, cs, g, sp, pr, st, T, k, C = case(N, n, A_max)
    tb = build_table(state, g)
    FN(state, tb, cs, g, sp, pr, st, T, 0, k)[0].v.block_until_ready()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        FN(state, tb, cs, g, sp, pr, st, T, 0, k)[0].v.block_until_ready()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), C


print("=== PRODUCTION REGIME: particles-per-cell FIXED at ~32, A_max fixed at 64 ===")
print("If the design is sound this is linear in N: 8x the particles, ~8x the time.\n")
print(f"{'N':>10} {'cells C':>9} {'npc':>6} {'run s':>9} {'ns/particle':>12} {'vs first':>9}")
base = None
for side in (4, 8, 16, 32):
    C = side ** 3
    N = C * 32
    tr, Cc = timeit(N, (side, side, side), 64)
    per = tr / N * 1e9
    base = per if base is None else base
    print(f"{N:>10,} {Cc:>9,} {32:>6} {tr:>9.4f} {per:>12.1f} {per / base:>9.2f}x")

print("\n=== 0D REGIME (milestone 1): one cell, A_max ~ N.  Expect O(N^2). ===")
print(f"{'N':>10} {'A_max':>8} {'run s':>9} {'vs first':>9} {'N^2 predicts':>13}")
base_t = base_n = None
for N in (4_000, 8_000, 16_000, 32_000):
    A = max(8, int(0.195 * N))
    tr, _ = timeit(N, (1, 1, 1), A)
    if base_t is None:
        base_t, base_n = tr, N
    print(f"{N:>10,} {A:>8,} {tr:>9.4f} {tr / base_t:>9.2f}x {(N / base_n) ** 2:>13.2f}x")

print("\n=== the A_max cliff seen at 8192 ===")
print(f"{'A_max':>8} {'run s':>9} {'us/slot':>9}")
for A in (1024, 2048, 4096, 8192, 16384):
    tr, _ = timeit(32_000, (1, 1, 1), A)
    print(f"{A:>8,} {tr:>9.4f} {tr / A * 1e6:>9.1f}")
