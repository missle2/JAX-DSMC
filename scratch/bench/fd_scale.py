"""Fixed-count CRN finite differences of the P3 Test-3 objective on our host forward,
many seeds in parallel: the arbiter for route (b) at N0 = 16000."""

import json
import sys
from multiprocessing import Pool

import numpy as np

from jax_dsmc import ensemble_inflow as ei
from jax_dsmc import ensemble_particle as ep

N0 = int(sys.argv[1])
SEEDS = int(sys.argv[2])
WORKERS = int(sys.argv[3])
out = sys.argv[4]
cfg = ei.Config(n_initial=N0, cells=40, dt=0.025, steps=20, retained=8, source_samples=8, partners=1)
T = np.full((2, 3), 4.0)
counts = ei.nominal_counts(cfg, T)


def one(seed):
    fd = []
    for side in (0, 1):
        plus, minus = T.copy(), T.copy()
        plus[side, 0] += 0.1
        minus[side, 0] -= 0.1
        fd.append((ep.forward(cfg, plus, seed=seed, injection_counts=counts).objective
                   - ep.forward(cfg, minus, seed=seed, injection_counts=counts).objective) / 0.2)
    return {"seed": seed, "fd": fd}


if __name__ == "__main__":
    with Pool(WORKERS) as pool:
        rows = []
        for row in pool.imap_unordered(one, range(SEEDS), chunksize=4):
            rows.append(row)
            if len(rows) % 50 == 0:
                f = np.array([r["fd"] for r in rows])
                print(f"{len(rows)} seeds: fd {f.mean(0).round(4)} +- {(f.std(0, ddof=1)/np.sqrt(len(f))).round(4)}", flush=True)
                json.dump({"n_initial": N0, "rows": rows}, open(out, "w"))
    f = np.array([r["fd"] for r in rows])
    json.dump({"n_initial": N0, "rows": rows}, open(out, "w"))
    print(f"FINAL N0={N0} seeds={len(f)}: fd {f.mean(0).round(4)} +- {(f.std(0, ddof=1)/np.sqrt(len(f))).round(4)}")
