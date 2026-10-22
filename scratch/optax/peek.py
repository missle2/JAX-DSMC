"""Quick look (2026-10-23): shapes of one run at a few iterates, with the box."""

import dataclasses, json, sys
import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp, matplotlib.pyplot as plt, numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

tag, its = sys.argv[1], [int(k) for k in sys.argv[2].split(",")]
d = f"examples/mesh/results/optimize/{tag}"
cfg = dict(O.default_config(), **json.load(open(f"{d}/history.json"))["config"])
grid, ls0 = O.initial_levelset(cfg)
lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
fig, axes = plt.subplots(1, len(its), figsize=(4 * len(its), 3))
J = [r["objective"] for r in json.load(open(f"{d}/history.json"))["rows"]]
for ax, k in zip(axes, its):
    c = np.load(f"{d}/coef_{k}.npy")
    for seg in O.surface_polyline(dataclasses.replace(ls0, coef=jnp.asarray(c)), lo, hi):
        ax.plot(seg[:, 0], seg[:, 1], lw=1)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_title(f"{k}: J={J[min(k, len(J) - 1)] / J[0]:.3f}")
fig.tight_layout()
fig.savefig("scratch/optax/peek.png", dpi=70)
