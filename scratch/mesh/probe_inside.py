import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys; sys.path.insert(0, "tests")
import numpy as np, jax, jax.numpy as jnp
from test_mesh_geometry import *
rng = np.random.default_rng(6)
grid = circle_grid(); mesh = circle_mesh(radius=1)
N = 5000
x = rng.uniform(0.0, 10.0, (N, 2))
dist = np.linalg.norm(x - CENTRE, axis=1)
keep = np.abs(dist - RADIUS) > SAGITTA
cell = cell_of(jnp.asarray(x), grid)
got = np.asarray(mesh_geometry.inside_polygon(jnp.asarray(x), cell, mesh, grid))
bad = keep & (got != (dist < RADIUS))
print("bad", bad.sum(), "of", keep.sum())
i = np.nonzero(bad)[0][:5]
print(x[i], dist[i], got[i], np.asarray(cell)[i])
print("cell types", np.asarray(mesh.cell_type)[np.asarray(cell)[i]], "corners", np.asarray(mesh.outside_corner)[np.asarray(cell)[i]])
print("csurfs", np.asarray(mesh.csurfs)[np.asarray(cell)[i]])
lower, upper = cell_bounds(grid)
print(np.asarray(lower)[np.asarray(cell)[i]], np.asarray(upper)[np.asarray(cell)[i]])
from jax_dsmc.mesh.geometry import _cell_corner, _segment_crossings
c = np.asarray(cell)[i]
tgt = _cell_corner(grid, jnp.asarray(c), jnp.maximum(mesh.outside_corner[jnp.asarray(c)], 0))
print("target", np.asarray(tgt))
print("count", np.asarray(_segment_crossings(jnp.asarray(x[i]), tgt, mesh, mesh.csurfs[jnp.asarray(c)])))
