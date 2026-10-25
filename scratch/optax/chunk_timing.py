"""The chunked body integrals (2026-10-24) against one chunk (the earlier whole-grid form):
the optimiser's per-iterate calls on the circle_long level set (levelset_h 0.0625).

    PYTHONPATH=. python3 scratch/optax/chunk_timing.py
"""

import sys
import time

import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402


def timed(fn, n=3):
    fn()  # compile
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


cfg = dict(O.default_config(), case="circle_long", levelset_h=0.0625, levelset_samples=16)
grid, ls = O.initial_levelset(cfg)
lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
points = int(np.prod([np.ceil((h - l) / ls.h) * 4 for l, h in zip(lo, hi, strict=True)]))
for chunk in (1 << 16, 1 << 24):
    O.L.SUBGRID_CHUNK = chunk
    a = timed(lambda: O.area_and_gradient(ls, lo, hi))
    c = timed(lambda: O.centroid_and_gradient(ls, lo, hi))
    print(f"chunk {chunk:>9d} ({-(-points // chunk)} chunks of {points} points): "
          f"area + gradient {a:.3f} s, centroid + Jacobian {c:.3f} s", flush=True)  # fmt: skip
