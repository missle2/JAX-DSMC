"""Smoke probe for jax_dsmc.mesh.geometry / analytic (not a test)."""
import os, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
print(jax.__version__)
from jax_dsmc.state import GridSpec
from jax_dsmc.mesh.types import Mesh, Surface, AnalyticBody, OUTSIDE, INSIDE, CUT
from jax_dsmc.mesh import geometry as mg, analytic
from jax_dsmc import geometry as bg

path = "/home/aakash/sparta/sparta/examples/circle/data.circle"
lines = open(path).read().splitlines()
ip = lines.index("Points") + 2
pts = np.array([[float(t) for t in lines[ip + i].split()[1:3]] for i in range(50)])
il = lines.index("Lines") + 2
el = np.array([[int(t) - 1 for t in lines[il + i].split()[1:3]] for i in range(50)])
d = pts[el[:, 1]] - pts[el[:, 0]]
nrm = np.stack([-d[:, 1], d[:, 0]], 1); nrm /= np.linalg.norm(nrm, axis=1)[:, None]
print("normal check outward:", np.all(np.sum(nrm * (pts[el[:, 0]] - 5.0), 1) > 0))
grid = GridSpec(lo=jnp.array([0., 0.]), hi=jnp.array([10., 10.]), n=(20, 20))
C = 400
cand = -np.ones((C, 50), np.int32); cand[:] = np.arange(50)
body = Surface(points=jnp.asarray(pts), elements=jnp.asarray(el, jnp.int32), model=jnp.zeros(50, jnp.int32), group=jnp.zeros(50, jnp.int32))
mesh = Mesh(body=body, cell_type=jnp.full((C,), CUT, jnp.int32), volume=jnp.ones(C), candidates=jnp.asarray(cand), csurfs=jnp.asarray(cand), outside_corner=jnp.zeros(C, jnp.int32), normals=jnp.asarray(nrm), areas=jnp.asarray(np.linalg.norm(d, axis=1)), radius=20, kind="surface")
x = jnp.array([[9.0, 5.0], [5.0, 9.5], [0.5, 0.5]])
v = jnp.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 1.0, 0.0]])
ex = jnp.full((3,), -1, jnp.int32)
f = jax.jit(mg.first_hit)
t0 = time.time(); print(f(x, v, grid, mesh, ex)); print("compile+run", time.time() - t0)
t0 = time.time(); print(f(x, v, grid, mesh, ex)); print("run", time.time() - t0)
print("cache", f._cache_size())
t, s = f(x, v, grid, mesh, ex)
print(mg.surface_normal(mesh, s, x + t[:, None] * v[:, :2], grid))
ab = AnalyticBody(centre=jnp.array([5.0, 5.0]), radius=jnp.array(3.0))
amesh = Mesh(body=ab, cell_type=mesh.cell_type, volume=mesh.volume, candidates=jnp.zeros((C, 1), jnp.int32), csurfs=jnp.zeros((C, 1), jnp.int32), outside_corner=mesh.outside_corner, normals=jnp.zeros((1, 2)), areas=jnp.zeros((1,)), radius=1, kind="circle")
print(jax.jit(mg.first_hit)(x, v, grid, amesh, ex))
print(jax.jit(mg.inside_polygon)(jnp.array([[5.0, 5.0], [9.0, 9.0], [7.9, 5.0]]), jnp.array([210, 380, 310]), mesh, grid))
print(mg.stencil_sufficient(v, jnp.array([0.1, 0.6, 0.3]), grid, mesh))
