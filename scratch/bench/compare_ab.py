"""Paired like-to-like: route (a) P3 retained frames vs route (b) particle-carried, on
identical forward trajectories, plus the fixed-count CRN finite difference.  Stress
setting: Delta t * rho * Q0 near the disjoint-pair limit (density grows to ~16 with
dt = 0.025, so the rate reaches ~0.4-0.6), sharp reward window, 20 steps."""

import json
import sys
import time

import numpy as np

from jax_dsmc import ensemble_inflow as ei
from jax_dsmc import ensemble_particle as ep

N0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
R = int(sys.argv[3]) if len(sys.argv) > 3 else 512
out = sys.argv[4] if len(sys.argv) > 4 else f"scratch/bench/compare_ab_N{N0}_R{R}.json"
cfg = ei.Config(n_initial=N0, cells=40, dt=0.025, steps=20, retained=R, source_samples=100000,
                partners=8, neighbors=32, reconstruction="ballistic")
T = np.full((2, 3), 4.0)
counts = ei.nominal_counts(cfg, T)
h = 0.1
rows = []
for seed in range(SEEDS):
    t0 = time.time()
    tb = ep.forward(cfg, T, seed=seed, injection_counts=counts)
    gb = ep.backward(tb)
    tb_time = time.time() - t0
    t0 = time.time()
    ta = ei.forward(cfg, T, seed=seed, injection_counts=counts)
    ga = ei.backward(ta, seed=1000 + seed)
    ta_time = time.time() - t0
    assert ta.objective == tb.objective
    fd = []
    for side in (0, 1):
        plus, minus = T.copy(), T.copy()
        plus[side, 0] += h
        minus[side, 0] -= h
        fd.append((ep.forward(cfg, plus, seed=seed, injection_counts=counts).objective
                   - ep.forward(cfg, minus, seed=seed, injection_counts=counts).objective) / (2 * h))
    rows.append({"seed": seed, "a": ga[:, 0].tolist(), "b": gb[:, 0].tolist(), "fd": fd,
                 "objective": tb.objective, "s_a": ta_time, "s_b": tb_time,
                 "pairs_per_step": float(np.mean([len(s.pairs) for s in tb.steps])),
                 "particles_per_step": float(np.mean([len(s.pre) for s in tb.steps]))})
    print(f"seed {seed}: a={np.round(ga[:,0],4)} b={np.round(gb[:,0],4)} fd={np.round(fd,4)}  ({ta_time:.0f}s / {tb_time:.1f}s)", flush=True)
    json.dump({"config": cfg.__dict__, "counts": counts, "fd_step": h, "rows": rows}, open(out, "w"), indent=1)
a, b, f = (np.array([r[k] for r in rows]) for k in ("a", "b", "fd"))
n = len(rows)
se = lambda x: x.std(0, ddof=1) / np.sqrt(n)  # noqa: E731
print(f"\nN0={N0} R={R} seeds={n}")
print(f"  a   {a.mean(0).round(4)} +- {se(a).round(4)}")
print(f"  b   {b.mean(0).round(4)} +- {se(b).round(4)}")
print(f"  fd  {f.mean(0).round(4)} +- {se(f).round(4)}")
print(f"  b-a paired {(b-a).mean(0).round(4)} +- {se(b-a).round(4)}")
print(f"  b-fd paired {(b-f).mean(0).round(4)} +- {se(b-f).round(4)}   a-fd paired {(a-f).mean(0).round(4)} +- {se(a-f).round(4)}")
print(f"  paper 0.1517 / 0.1516;  cost a {np.mean([r['s_a'] for r in rows]):.0f}s  b {np.mean([r['s_b'] for r in rows]):.1f}s per seed")
