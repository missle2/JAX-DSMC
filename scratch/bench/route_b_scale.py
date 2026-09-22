"""Route (b) on P3's model at the paper's setting, large N0: the interpolation-free answer.

Same objective and reservoirs as P3 Test 3 (T = 4 all components, dt = 0.025, 40 cells,
20 steps, nominal counts).  Writes per-seed rows with the fixed-count CRN FD."""

import json
import sys
import time

import numpy as np

from jax_dsmc import ensemble_inflow as ei
from jax_dsmc import ensemble_particle as ep

N0 = int(sys.argv[1])
SEEDS = int(sys.argv[2])
out = sys.argv[3]
cfg = ei.Config(n_initial=N0, cells=40, dt=0.025, steps=20, retained=8, source_samples=8, partners=1, neighbors=32)
T = np.full((2, 3), 4.0)
counts = ei.nominal_counts(cfg, T)
rows = []
START = int(sys.argv[4]) if len(sys.argv) > 4 else 0
for seed in range(START, START + SEEDS):
    t0 = time.time()
    tb = ep.forward(cfg, T, seed=seed, injection_counts=counts)
    g = ep.backward(tb)[:, 0]
    fd = []
    for side in (0, 1):
        plus, minus = T.copy(), T.copy()
        plus[side, 0] += 0.1
        minus[side, 0] -= 0.1
        fd.append((ep.forward(cfg, plus, seed=seed, injection_counts=counts).objective
                   - ep.forward(cfg, minus, seed=seed, injection_counts=counts).objective) / 0.2)
    rows.append({"seed": seed, "b": g.tolist(), "fd": fd, "objective": tb.objective, "seconds": time.time() - t0})
    print(f"seed {seed}: b={g.round(4)} fd={np.round(fd, 4)} ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"n_initial": N0, "counts": counts, "normalization": tb.normalization, "rows": rows}, open(out, "w"), indent=1)
b = np.array([r["b"] for r in rows]); f = np.array([r["fd"] for r in rows]); n = len(rows)
print(f"N0={N0} seeds={n}: b {b.mean(0).round(4)} +- {(b.std(0, ddof=1)/np.sqrt(n)).round(4)}  fd {f.mean(0).round(4)} +- {(f.std(0, ddof=1)/np.sqrt(n)).round(4)}  paper 0.1517 / 0.1516")
