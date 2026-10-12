"""A synthetic snapshot (snapshot.py's format) to smoke-test cv.py: 8 x 6 cells, a smooth
non-polynomial field plus noise."""
import sys

import jax.numpy as jnp
import numpy as np

from jax_dsmc.grid import cell_neighbourhood, collision_cell
from jax_dsmc.state import GridSpec

grid = GridSpec(jnp.zeros(2), jnp.asarray([0.8, 0.6]), (8, 6))
rng = np.random.default_rng(0)
n = 48 * 37
snap = {}
for k in (0, 120, 240):
    x = rng.uniform(0, 1, (n, 2)) * np.array([0.8, 0.6])
    v = rng.normal(0, 1.0, (n, 3))
    lam = np.sin(3 * x[:, 0]) * v[:, 0] + 0.3 * np.exp(-v[:, 1] ** 2) + 0.2 * v[:, 2] ** 3
    lam = lam + 0.3 * rng.normal(size=n)
    snap.update({f"x{k}": x, f"v{k}": v, f"alive{k}": np.ones(n, bool),
                 f"cell{k}": np.asarray(collision_cell(jnp.asarray(x), grid)),
                 f"lam{k}": lam, f"lamt{k}": lam,
                 f"normal{k}": np.where(x[:, :1] < 0.1, np.array([[1.0, 0, 0]]), 0.0)})
nbr, ex = cell_neighbourhood(grid, 1)
snap.update(scale=np.array([0.1, 0.1, 1.0, 1.0, 1.0]), n_cells=48, nbr1=np.asarray(nbr),
            exists1=np.asarray(ex))
np.savez(sys.argv[1], **snap)
