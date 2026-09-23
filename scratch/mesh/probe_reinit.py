"""reinitialise at n_sample 16: old fixed 8 steps vs the band-sized default, on the exact
circle and on Adam's recomputed first step (optc S1 adam)."""
import dataclasses, sys
import numpy as np
import jax.numpy as jnp
sys.path.insert(0, "examples/mesh")
from examples.mesh import optimize as O
from jax_dsmc.mesh import cut
from jax_dsmc.mesh import levelset as L
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

cfg = O.default_config()
cfg.update(fnum_scale=1.0, grid_scale=1, levelset_h=0.0625, levelset_samples=16, grazing=0.05)
grid, ls0 = O.initial_levelset(cfg)
lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
xs = np.linspace(0, 10, 801)
pts = jnp.asarray(np.stack(np.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2))
true = np.linalg.norm(np.asarray(pts) - 5.0, axis=1) - 3.0
def err(ls):
    phi = np.asarray(L.value(ls, pts))
    near = np.abs(true) < 1.5
    spurious = np.sum((np.sign(phi) != np.sign(true)) & (np.abs(true) > 0.05))
    return np.abs(phi - true)[near].max(), spurious
print("band", 3 * ls0.h + ls0.reach, "h", ls0.h)
print("exact circle          err, spurious:", err(ls0))
a = ls0; b = ls0
for i in range(4):
    a = L.reinitialise(a, n_iter=8); b = L.reinitialise(b)
    print(f"reinit x{i+1}: old(8 steps) {err(a)}   new {err(b)}")
# Adam's first step recomputed from its saved first moment (m = (1 - beta1) dir at k = 0)
R = "examples/mesh/results/optimize/circle_coarse_S1_adam"
z = np.load(f"{R}/adam_state.npz")
d, _ = O.adam_direction(z["m"] / 0.1, (np.zeros_like(z["m"]), np.zeros_like(z["m"])), 0, 0.9, 0.999)
eta = 0.05 / np.abs(d).max()
area0 = float(L.body_volume(ls0, lo, hi, 4))
stepped = dataclasses.replace(ls0, coef=ls0.coef - eta * jnp.asarray(d))
old = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"{R}/coef_1.npy")))
new, shift = O.project_area(L.reinitialise(stepped), area0, lo, hi)
print("adam coef_1 saved vs recomputed(new reinit): max|dcoef|", float(np.abs(new.coef - old.coef).max()))
for name, ls in (("stepped (no reinit)", stepped), ("saved coef_1", old), ("recomputed coef_1", new)):
    phi = np.asarray(L.value(ls, pts)); 
    print(name, "min |phi| on |true|>0.3:", np.abs(phi)[np.abs(true) > 0.3].min())
    cut.build_mesh(ls, dataclasses.replace(grid, volume=jnp.zeros(0)), radius=1)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for name, ls in (("stepped", stepped), ("saved", old), ("recomputed", new)):
    for j, seg in enumerate(O.surface_polyline(ls, lo, hi, 800)):
        for a_ in ax: a_.plot(seg[:, 0], seg[:, 1], lw=0.8, label=name if j == 0 else None)
ax[0].set_xlim(1.5, 8.5); ax[0].set_ylim(1.5, 8.5); ax[1].set_xlim(1.7, 3.2); ax[1].set_ylim(4, 6.5)
for a_ in ax: a_.set_aspect("equal")
ax[0].legend(); fig.savefig("scratch/mesh/probe_reinit.png", dpi=110)
