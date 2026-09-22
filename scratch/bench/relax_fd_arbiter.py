"""FD-only arbiter for the relaxation case at larger N (fnum = 1/N keeps the physics N-independent).

    JAX_PLATFORMS=cpu python3 scratch/bench/relax_fd_arbiter.py --particles 20000 --attempts 20480 --seeds 256
"""

import argparse
import json
import time

import jax
import numpy as np

from examples import ensemble_relaxation as er

p = argparse.ArgumentParser()
p.add_argument("--parameter", default="diameter")
p.add_argument("--particles", type=int, default=20000)
p.add_argument("--attempts", type=int, default=20480)
p.add_argument("--steps", type=int, default=10)
p.add_argument("--dt", type=float, default=0.2)
p.add_argument("--seeds", type=int, default=256)
p.add_argument("--seed-start", type=int, default=0)
p.add_argument("--output", default="scratch/bench/relax_fd_arbiter.json")
a = p.parse_args()
cfg = dict(particles=a.particles, steps=a.steps, attempts=a.attempts, dt=a.dt, majorant="certified")
hs = [0.1, 0.05, 0.025]
rows = []
t0 = time.time()
for s in range(a.seed_start, a.seed_start + a.seeds):
    fd = er.finite_differences(a.parameter, jax.random.key(s), cfg, hs)
    rows.append({"seed": s, "fd": fd})
    print(f"seed={s} FD={fd} ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"config": cfg, "parameter": a.parameter, "fd_steps": hs, "rows": rows}, open(a.output, "w"))
fd = np.array([r["fd"] for r in rows])
print("mean", fd.mean(0), "se", fd.std(0, ddof=1) / len(fd) ** 0.5)
