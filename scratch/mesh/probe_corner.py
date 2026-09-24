"""The optc Adam shapes that raised in outside_corner: build_mesh on them."""
import dataclasses, sys
import numpy as np
import jax.numpy as jnp
sys.path.insert(0, "examples/mesh")
from examples.mesh import optimize as O
from jax_dsmc.mesh import cut
from jax_dsmc.mesh import levelset as L

cfg = O.default_config(); cfg.update(levelset_h=0.0625, levelset_samples=16)
grid, ls0 = O.initial_levelset(cfg)
for tag, k in (("circle_coarse_S1_adam", 56), ("circle_coarse_S8_adam", 9)):
    ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"examples/mesh/results/optimize/{tag}/coef_{k}.npy")))
    try:
        m = cut.build_mesh(ls, dataclasses.replace(grid, volume=jnp.zeros(0)), radius=1)
        print(tag, k, "builds: n_sub", m.n_sub, "cut cells", int((np.asarray(m.cell_type) == 2).sum()))
    except ValueError as e:
        print(tag, k, "RAISES:", str(e)[:120])
        cell = int(str(e).split("[")[1].split("]")[0])
        lo, hi = cut.cell_boxes(grid)
        xs = np.linspace(lo[cell], hi[cell], 5)
        cor = cut.corners(lo[cell:cell + 1], hi[cell:cell + 1])[0]
        print("   cell", cell, "box", lo[cell], hi[cell], "corner phi", np.asarray(L.value(ls, jnp.asarray(cor))).round(4))
        g = np.stack(np.meshgrid(np.linspace(lo[cell][0], hi[cell][0], 41), np.linspace(lo[cell][1], hi[cell][1], 41), indexing="ij"), -1).reshape(-1, 2)
        f = np.asarray(L.value(ls, jnp.asarray(g)))
        print("   flow fraction of cell", (f > 0).mean().round(4), " max phi", f.max().round(4))
