"""Round-off amplification of a grazing particle on a concave wall: perturb the start by
~1 ulp and compare the ends of one move as the number of reflections grows."""
import sys
import numpy as np
import jax, jax.numpy as jnp
from jax_dsmc import wall_adjoint
from jax_dsmc.boundary import BC, Faces, WallParams
from jax_dsmc.mesh import cut
from jax_dsmc.mesh import levelset as L
from jax_dsmc.state import GridSpec

h, radius, centre = 0.0625, 3.0, np.array([5.0, 5.0])
grid = GridSpec(lo=jnp.array([0.0, 0.0]), hi=jnp.array([10.0, 10.0]), n=(20, 20))
lo, m = L.grid_for_box(grid.lo, grid.hi, h, margin=3)
knots = np.asarray(lo) + h * np.stack(np.meshgrid(*[np.arange(n) for n in m], indexing="ij"), -1)
ls = L.from_values(radius - np.linalg.norm(knots - centre, axis=-1), lo, h, n_sample=16)  # concave wall
mesh = cut.build_mesh(ls, grid, radius=1)
faces = Faces.uniform(BC.SPECULAR, 2)
wall = WallParams.uniform(2, 300.0, acc=0.0, n_models=1)
alpha = 0.3
on = np.asarray(L.project(ls, jnp.asarray((centre + radius * np.array([np.cos(alpha), np.sin(alpha)]))[None]), 20))[0]
tangent = np.array([-np.sin(alpha), np.cos(alpha)]); inward = -np.array([np.cos(alpha), np.sin(alpha)])
x0 = on + 1e-9 * inward
for theta in (0.02, 0.0065):
    speed = 780.0
    v = speed * (np.cos(theta) * tangent + np.sin(theta) * inward)
    print(f"grazing angle {np.degrees(theta):.2f} deg, chord {2 * radius * np.sin(theta):.4f}")
    for dt in (2e-5, 5e-5, 1e-4, 2e-4):
        xs = jnp.asarray(np.stack([x0, x0 * (1 + 2.2e-16)]))
        vs = jnp.asarray(np.stack([np.r_[v, 0.0]] * 2))
        out = wall_adjoint.advance(xs, vs, jnp.ones(2, bool), grid, faces, wall, 6.63e-26, dt, jax.random.PRNGKey(0),
                                   mesh=mesh, tracker="lane", max_bounces=64, return_events=True)
        x1 = np.asarray(out[0]); v1 = np.asarray(out[1])
        n_hits = int(out[3].surface_hits) // 2
        dx = np.abs(x1[0] - x1[1]).max() / 10.0
        dv = np.abs(v1[0] - v1[1]).max() / np.abs(v1[0]).max()
        print(f"   dt {dt:.0e}: {n_hits} reflections, start perturbed 2.2e-16 -> end rel. diff x {dx:.1e} v {dv:.1e}")
