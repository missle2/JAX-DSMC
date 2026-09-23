"""Adam's first step (optc S1 adam, recomputed from its first moment) with the normalised
direction re-smoothed at sigma 2 and 4 knots; build_mesh + zero sets + curvature proxy."""
import dataclasses, sys
import numpy as np
import jax.numpy as jnp
sys.path.insert(0, "examples/mesh")
from examples.mesh import optimize as O
from jax_dsmc.mesh import cut
from jax_dsmc.mesh import levelset as L
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

cfg = O.default_config(); cfg.update(levelset_h=0.0625, levelset_samples=16)
grid, ls0 = O.initial_levelset(cfg)
lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
area0 = float(L.body_volume(ls0, lo, hi, 4))
R = "examples/mesh/results/optimize/circle_coarse_S1_adam"
g_s = np.load(f"{R}/adam_state.npz")["m"] / 0.1   # the smoothed, banded direction fed to Adam
zero = np.zeros_like(g_s)
d_raw, _ = O.adam_direction(g_s, (zero, zero), 0, 0.9, 0.999)
fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
for name, d in [("adam raw", d_raw)] + [(f"adam + smooth {s}", O.smoothed_direction(d_raw, ls0, s, 1.0 + ls0.h)) for s in (2.0, 4.0)] + [("gd", g_s)]:
    eta = 0.05 / np.abs(d).max()
    new, _ = O.project_area(L.reinitialise(dataclasses.replace(ls0, coef=ls0.coef - eta * jnp.asarray(d))), area0, lo, hi)
    np.save(f"scratch/mesh/adam1_{name.replace(' ', '_').replace('+', '')}.npy", np.asarray(new.coef))
    th = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    # radial displacement of the zero set along rays from the centre
    rr = []
    for t in th[::10]:
        s = np.linspace(2.5, 3.5, 2001); p = jnp.asarray(np.stack([5 + s * np.cos(t), 5 + s * np.sin(t)], 1))
        f = np.asarray(L.value(new, p)); i = np.nonzero(np.diff(np.sign(f)))[0]
        rr.append(s[i[0]] if i.size else np.nan)
    rr = np.array(rr)
    ok = "ok"
    try:
        cut.build_mesh(new, dataclasses.replace(grid, volume=jnp.zeros(0)), radius=1)
    except ValueError as e:
        ok = str(e)[:50]
    print(f"{name:20s} max|dr| {np.nanmax(np.abs(rr - 3)):.3f}  max|d2r/dth2|*... roughness {np.nanmax(np.abs(np.diff(rr, 2))) / (th[10] ** 2):.1f}  build {ok}")
    for j, seg in enumerate(O.surface_polyline(new, lo, hi, 800)):
        for a in ax: a.plot(seg[:, 0], seg[:, 1], lw=0.9, label=name if j == 0 else None)
ax[0].set_xlim(1.5, 8.5); ax[0].set_ylim(1.5, 8.5); ax[1].set_xlim(1.7, 3.2); ax[1].set_ylim(4, 6.5)
for a in ax: a.set_aspect("equal")
ax[0].legend(fontsize=7); fig.savefig("scratch/mesh/probe_adam_smooth.png", dpi=110)
