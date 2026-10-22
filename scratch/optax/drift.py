"""Where the body sits (2026-10-23): area centroid and x extent of the zero contour every 10
iterates for every run of scratch/optax/analysis.py; adds ``position`` to its json.  The
drag of a body in a uniform stream does not depend on where it sits, so any drift is the
gradient's translation component (noise or a box effect), which Adam's per-coefficient
normalisation turns into full-size steps.

    PYTHONPATH=. python3 scratch/optax/drift.py
"""

import dataclasses
import json
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/optax")
import optimize as O  # noqa: E402
from analysis import ROOT, tag  # noqa: E402


def centroid(seg):
    x, y = seg[:, 0], seg[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2
    return (np.sum((x + np.roll(x, -1)) * cross) / (6 * a),
            np.sum((y + np.roll(y, -1)) * cross) / (6 * a))  # fmt: skip


def main():
    data = json.load(open("scratch/optax/analysis.json"))
    for regime, arms in data.items():
        cfg = dict(O.default_config(),
                   **json.load(open(f"{ROOT}/{tag(regime, 'S4_adam')}/history.json"))["config"])  # fmt: skip
        grid, ls0 = O.initial_levelset(cfg)
        lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
        for arm, d in arms.items():
            pos = {}
            n = len(d["J"])
            for k in sorted({*range(0, n + 1, 10), n}):
                try:
                    c = np.load(f"{ROOT}/{tag(regime, arm)}/coef_{k}.npy")
                except FileNotFoundError:
                    continue
                seg = max(O.surface_polyline(dataclasses.replace(ls0, coef=jnp.asarray(c)), lo, hi),
                          key=len)  # fmt: skip
                cx, cy = centroid(seg)
                pos[str(k)] = [float(cx), float(cy), float(seg[:, 0].min()), float(seg[:, 0].max())]
            d["position"] = pos
            last = pos[max(pos, key=int)]
            print(
                f"{regime:10s} {arm:12s} centroid x 5.00 -> {last[0]:6.2f}   x extent "
                f"[{last[2]:6.2f}, {last[3]:6.2f}]  (box [{lo[0]:g}, {hi[0]:g}])",
                flush=True,
            )
    json.dump(data, open("scratch/optax/analysis.json", "w"))


if __name__ == "__main__":
    main()
