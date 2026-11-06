"""Slim the fit lab's dumps for the pull (2026-11-05): the transfer cases' dumps are ~70 GB.
Keeps what the lab's window / kNN / closure scripts read -- the nodes with v_n > 0 within
``--d-max`` of the wall (step, slot, x, v, lambda) and every hit array -- and writes
``scratch/fitlab/cache_slim/<stem>.npz`` + the json, for ``pull_results.sh`` to fetch instead
of the full cache.  numpy only (no JAX, no GPU): run it on the cluster.

    python3 scratch/fitlab/slim.py [--gates c25_diff_m2 ...] [--d-max 0.6]
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scratch/fitlab")

CACHE = Path("scratch/fitlab/cache")
SLIM = Path("scratch/fitlab/cache_slim")
CENTRE = np.array([5.0, 5.0])
RADIUS = 3.0
NODE_KEEP = ("node_step", "node_slot", "node_x", "node_v", "node_lam")
GATES = (
    "c25_diff_m2",
    "c25_spec_m2",
    "c03_diff_m2",
    "c03_spec_m2",
    "c10_hot_m2",
    "c10_diff_m2_g2",
    "c10_diff_heat_m2_g2",
    "c10_diff_m2_g4",
)


def slim(path, d_max):
    out = SLIM / path.name
    if out.exists() and (SLIM / path.with_suffix(".json").name).exists():
        return
    with np.load(path) as z:
        x, v = z["node_x"], z["node_v"]
        rel = x - CENTRE
        r = np.hypot(rel[:, 0], rel[:, 1])
        vn = (v[:, 0] * rel[:, 0] + v[:, 1] * rel[:, 1]) / r
        keep = (vn > 0) & (r - RADIUS < d_max)
        arrays = {k: z[k][keep] for k in NODE_KEEP}
        arrays.update({k: z[k] for k in z.files if k.startswith("hit_")})
    SLIM.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    shutil.copy(path.with_suffix(".json"), SLIM / path.with_suffix(".json").name)
    print(f"{path.name}: {path.stat().st_size / 1e9:.2f} GB -> {out.stat().st_size / 1e9:.2f} GB "
          f"({keep.mean():.0%} of the nodes)", flush=True)  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", nargs="+", default=list(GATES))
    ap.add_argument("--d-max", type=float, default=0.6)
    args = ap.parse_args()
    for gate in args.gates:
        for path in sorted(CACHE.glob(f"{gate}_s*.npz")):
            if "_smoke" not in path.name and path.with_suffix(".json").exists():
                slim(path, args.d_max)


if __name__ == "__main__":
    main()
