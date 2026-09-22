"""Two-step arbiter: fixed-count CRN finite differences over many forward seeds against
route (b) (interpolation-free via exact_last, and interpolated) and route (a)."""

import json
import sys

import numpy as np

from jax_dsmc import ensemble_inflow as ei
from jax_dsmc import ensemble_particle as ep

SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 48
A_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
cfg = ei.Config(n_initial=20000, cells=20, dt=0.05, steps=2, retained=4000, source_samples=100000,
                partners=8, neighbors=32, q0=4.0, reconstruction="ballistic")
T = np.full((2, 3), 4.0)
counts = (5000, 5000)
rows = []
for seed in range(SEEDS):
    tb = ep.forward(cfg, T, seed=seed, injection_counts=counts)
    row = {"seed": seed, "b_exact": ep.backward(tb, exact_last=True)[:, 0].tolist(),
           "b": ep.backward(tb)[:, 0].tolist(), "fd": {}}
    for h in (0.2, 0.1, 0.05):
        fd = []
        for side in (0, 1):
            plus, minus = T.copy(), T.copy()
            plus[side, 0] += h
            minus[side, 0] -= h
            fd.append((ep.forward(cfg, plus, seed=seed, injection_counts=counts).objective
                       - ep.forward(cfg, minus, seed=seed, injection_counts=counts).objective) / (2 * h))
        row["fd"][str(h)] = fd
    if seed < A_SEEDS:
        row["a"] = ei.backward(ei.forward(cfg, T, seed=seed, injection_counts=counts), seed=seed)[:, 0].tolist()
    rows.append(row)
    print(f"seed {seed}: b_exact={np.round(row['b_exact'],4)} fd0.1={np.round(row['fd']['0.1'],4)}" + (f" a={np.round(row['a'],4)}" if "a" in row else ""), flush=True)
    json.dump({"config": cfg.__dict__, "rows": rows}, open("scratch/bench/arbiter_two_step.json", "w"), indent=1)

def summ(key, sub=None):
    x = np.array([r[key] if sub is None else r[key][sub] for r in rows if key in r])
    return x.mean(0), x.std(0, ddof=1) / np.sqrt(len(x)), x

bx, bxse, BX = summ("b_exact"); b, bse, B = summ("b")
print(f"\nseeds={SEEDS}")
print(f"  b exact-last  {bx.round(4)} +- {bxse.round(4)}")
print(f"  b interpolated {b.round(4)} +- {bse.round(4)}")
for h in ("0.2", "0.1", "0.05"):
    f, fse, F = summ("fd", h)
    d = BX - F
    print(f"  fd h={h:<4} {f.round(4)} +- {fse.round(4)}   b_exact-fd paired {d.mean(0).round(4)} +- {(d.std(0,ddof=1)/np.sqrt(len(d))).round(4)}")
a, ase, A = summ("a")
d = BX[: len(A)] - A
print(f"  a ({len(A)} seeds) {a.round(4)} +- {ase.round(4)}   b_exact-a paired {d.mean(0).round(4)} +- {(d.std(0,ddof=1)/np.sqrt(len(d))).round(4)}")
f = np.array([r["fd"]["0.1"] for r in rows[: len(A)]]); d = A - f
print(f"  a-fd(0.1) paired ({len(A)} seeds) {d.mean(0).round(4)} +- {(d.std(0,ddof=1)/np.sqrt(len(d))).round(4)}")
