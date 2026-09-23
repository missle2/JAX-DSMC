"""Zero sets of the failed optc iterates (coarse circle) + build_mesh outcome."""
import dataclasses, sys, os
import numpy as np
import jax.numpy as jnp
sys.path.insert(0, "examples/mesh")
from examples.mesh import optimize as O
from jax_dsmc.mesh import cut
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

cfg = O.default_config()
cfg.update(fnum_scale=1.0, grid_scale=1, levelset_h=0.0625, levelset_samples=16, grazing=0.05)
grid, ls0 = O.initial_levelset(cfg)
lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
R = "examples/mesh/results/optimize"
runs = [("circle_coarse_S1_gd", 2), ("circle_coarse_S8_gd", 2), ("circle_coarse_S1_adam", 1), ("circle_coarse_S8_adam", 1)]
fig, axes = plt.subplots(1, len(runs), figsize=(4.5 * len(runs), 4.5))
for ax, (tag, k) in zip(axes, runs):
    for j in range(k + 1):
        ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"{R}/{tag}/coef_{j}.npy")))
        for seg in O.surface_polyline(ls, lo, hi, 800):
            ax.plot(seg[:, 0], seg[:, 1], lw=0.8, label=f"coef_{j}")
        dc = np.abs(np.load(f"{R}/{tag}/coef_{j}.npy") - np.asarray(ls0.coef)).max()
        try:
            cut.build_mesh(ls, dataclasses.replace(grid, volume=jnp.zeros(0)), radius=1)
            msg = "ok"
        except ValueError as e:
            msg = str(e)[:60]
        print(tag, j, f"max|dcoef| {dc:.3f}", msg)
    for x in np.arange(lo[0], hi[0] + 1e-9, 0.5):
        ax.axvline(x, color="0.85", lw=0.4); ax.axhline(x, color="0.85", lw=0.4)
    ax.set_xlim(1.5, 8.5); ax.set_ylim(1.5, 8.5); ax.set_aspect("equal"); ax.set_title(tag, fontsize=8)
axes[0].legend(fontsize=6)
fig.tight_layout(); fig.savefig("scratch/mesh/probe_opt_shapes.png", dpi=120)
