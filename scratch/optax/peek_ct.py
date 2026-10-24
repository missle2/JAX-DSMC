"""The optct bodies (2026-10-24): zero contours of a few iterates per run, one PNG.

    PYTHONPATH=. python3 scratch/optax/peek_ct.py
"""

import dataclasses
import json
import sys

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/optax")
import optimize as O  # noqa: E402
from analysis import ROOT, tag  # noqa: E402

RUNS = (("kn0.5", "ox_adam_ct"), ("kn0.5", "ox_adamcos_ct"), ("kn0.5_M10", "ox_adam_ct"),
        ("kn0.5_M10", "ox_prodigy_ct"))  # fmt: skip
ITERS = (0, 40, 80, 110, 140, 200, 399)
fig, axes = plt.subplots(len(RUNS), len(ITERS), figsize=(24, 9), constrained_layout=True)
for row, (regime, arm) in enumerate(RUNS):
    cfg = dict(O.default_config(), **json.load(open(f"{ROOT}/{tag(regime, arm)}/history.json"))["config"])  # fmt: skip
    grid, ls0 = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    for col, k in enumerate(ITERS):
        ax = axes[row, col]
        try:
            c = np.load(f"{ROOT}/{tag(regime, arm)}/coef_{k}.npy")
        except FileNotFoundError:
            ax.set_axis_off()
            continue
        for seg in O.surface_polyline(dataclasses.replace(ls0, coef=jnp.asarray(c)), lo, hi):
            ax.plot(seg[:, 0], seg[:, 1], lw=0.8)
        ax.set_xlim(-10, 20)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_title(f"{regime} {arm} @ {k}", fontsize=8)
fig.savefig("scratch/optax/peek_ct.png", dpi=80)
