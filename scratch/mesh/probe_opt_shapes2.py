"""Latest iterate of every optc run (second submission): zero sets + curvature extremes."""
import dataclasses, glob, sys
import numpy as np
import jax.numpy as jnp
sys.path.insert(0, "examples/mesh")
from examples.mesh import optimize as O
from jax_dsmc.mesh import levelset as L
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

cfg = O.default_config(); cfg.update(levelset_h=0.0625, levelset_samples=16)
grid, ls0 = O.initial_levelset(cfg)
lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
tags = sorted(glob.glob("examples/mesh/results/optimize/circle_coarse_S*"))
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for ax, d in zip(axes.ravel(), tags):
    ks = sorted(int(p.split("_")[-1][:-4]) for p in glob.glob(f"{d}/coef_*.npy"))
    for k in ks:
        ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"{d}/coef_{k}.npy")))
        segs = O.surface_polyline(ls, lo, hi, 1000)
        for j, seg in enumerate(segs):
            ax.plot(seg[:, 0], seg[:, 1], lw=0.8, label=f"{k}" if j == 0 else None)
    # curvature of the last iterate along its zero set
    seg = max(segs, key=len)
    x = jnp.asarray(seg)
    # kappa = div(grad phi / |grad phi|) by finite differences of the normal
    e = 1e-4
    def nrm(p):
        _, g = L.value_and_grad(ls, p); return np.asarray(g) / np.linalg.norm(np.asarray(g), axis=1, keepdims=True)
    kap = sum((nrm(x + e * np.eye(2)[a]) - nrm(x - e * np.eye(2)[a]))[:, a] / (2 * e) for a in range(2))
    i = np.argmin(kap)
    ax.plot(*seg[i], "rx", ms=10)
    print(d.split("/")[-1], "iterates", ks[-1], f"pieces {len(segs)}  kappa min {kap.min():.2f} max {kap.max():.2f} (circle 0.333)  at {seg[i].round(2)}")
    ax.set_aspect("equal"); ax.set_title(d.split("/")[-1], fontsize=8); ax.legend(fontsize=5)
fig.tight_layout(); fig.savefig("scratch/mesh/probe_opt_shapes2.png", dpi=100)
