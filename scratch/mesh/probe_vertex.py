import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys; sys.path.insert(0, "tests")
import numpy as np, jax.numpy as jnp
from test_mesh_geometry import *
grid = circle_grid(); mesh = circle_mesh(radius=20)
points, elements = read_circle(); normals, _ = element_normals(points, elements)
N = len(points)
# axis ray at vertex 0
x = jnp.array([[9.0, 5.0]]); v = jnp.array([[-1.0, 0.0]])
t = np.asarray(mesh_geometry.hit_lines(x, v, broadcast(points[elements[:, 0]], 1), broadcast(points[elements[:, 1]], 1), broadcast(normals, 1)))
print("axis ray: t[e0]=%.17g t[e49]=%.17g" % (t[0, 0], t[0, 49]), "finite:", np.nonzero(np.isfinite(t[0]))[0])
# rays from random origins aimed exactly (v = p_k - x) at every vertex
rng = np.random.default_rng(0)
xs = CENTRE + 4.5 * np.stack([np.cos(a := rng.uniform(0, 2*np.pi, N)), np.sin(a)], 1)
vs = points - xs
t = np.asarray(mesh_geometry.hit_lines(jnp.asarray(xs), jnp.asarray(vs), broadcast(points[elements[:, 0]], N), broadcast(points[elements[:, 1]], N), broadcast(normals, N)))
hits = np.isfinite(t)
print("vertex-aimed rays: min hits per ray", hits.sum(1).min(), "max", hits.sum(1).max(), "all hit:", hits.any(1).all())
# rays through a vertex that are NOT exactly representable (v rotated by random tiny amounts): still >= 1 hit
vs2 = vs * (1 + 1e-13 * rng.normal(size=(N, 1))) + 1e-13 * rng.normal(size=(N, 2))
t2 = np.asarray(mesh_geometry.hit_lines(jnp.asarray(xs), jnp.asarray(vs2), broadcast(points[elements[:, 0]], N), broadcast(points[elements[:, 1]], N), broadcast(normals, N)))
print("perturbed near-vertex rays: hits per ray min", np.isfinite(t2).sum(1).min(), "-> watertight" if np.isfinite(t2).any(1).all() else "LEAK")
leak = np.nonzero(~np.isfinite(t2).any(1))[0]
print("leaking rays:", leak[:10], "count", len(leak))
from jax_dsmc.mesh.geometry import _lateral_lines, _plane_time
k = leak[0]
xk, vk = jnp.asarray(xs[k:k+1]), jnp.asarray(vs2[k:k+1])
lat = np.asarray(_lateral_lines(xk, vk, broadcast(points[elements[:, 0]], 1), broadcast(points[elements[:, 1]], 1)))[0]
tt, ok = _plane_time(xk, vk, broadcast(points[elements[:, 0]], 1), broadcast(normals, 1))
print("vertex", k, "lateral true for:", np.nonzero(lat)[0], "plane ok for:", np.nonzero(np.asarray(ok)[0])[0])
c = vs2[k, 0] * (points[:, 1] - xs[k, 1]) - vs2[k, 1] * (points[:, 0] - xs[k, 0])
print("c at k-1,k,k+1:", c[(k-1) % N], c[k], c[(k+1) % N])
print("elements containing vertex k:", np.nonzero((elements == k).any(1))[0], elements[(elements == k).any(1)])
for k in leak:
    c = vs2[k, 0] * (points[:, 1] - xs[k, 1]) - vs2[k, 1] * (points[:, 0] - xs[k, 0])
    print("ray", k, "all vertices on one side of the ray line:", (c >= 0).all() or (c <= 0).all(), "-> genuine miss")
# and for every non-leaking perturbed ray the line does cross the polygon
ok = []
for k in range(N):
    c = vs2[k, 0] * (points[:, 1] - xs[k, 1]) - vs2[k, 1] * (points[:, 0] - xs[k, 0])
    crosses = not ((c >= 0).all() or (c <= 0).all())
    ok.append(crosses == np.isfinite(t2[k]).any())
print("perturbed rays: hit <=> line crosses polygon for all rays:", all(ok))
