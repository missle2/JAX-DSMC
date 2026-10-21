"""Statistical check of the 2026-10-21 post-loop query fix: a level-set disk under a specular
box face 0.1 above it, fast particles in the gap, one or two bounces allowed.  A reflected particle
whose remaining move ends in a grazing sliver of the disk (chord < the reach sampling's
spacing) was not counted unresolved by the reach-sampled query: its final move ended inside.
Runs the lane tracker with the fixed query and with the old one patched back in."""

import functools
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, "tests")
from jax_dsmc import wall_adjoint  # noqa: E402
from jax_dsmc.boundary import BC, Faces, WallParams  # noqa: E402
from jax_dsmc.mesh import cut  # noqa: E402
from jax_dsmc.mesh import levelset as L  # noqa: E402
from jax_dsmc.state import GridSpec  # noqa: E402

jax.config.update("jax_enable_x64", True)
MASS = 6.63e-26


def old_lane_distance(args, grid, mesh):
    x, v, exclude, slot, _remaining = args
    n = x.shape[0]
    take = jnp.maximum(slot, 0)
    d, _ = wall_adjoint._next_surface(x[take], v[take], grid, mesh, exclude[take], None)
    return jnp.full((n,), jnp.inf).at[wall_adjoint._drop_slot(slot, n)].set(d, mode="drop")


def run(n, seed, h=0.125, top=8.1, tracker="lane", bounces=2, move=0.5):
    grid = GridSpec(lo=jnp.array([0.0, 0.0]), hi=jnp.array([10.0, top]), n=(40, 32))
    lo, m = L.grid_for_box(grid.lo, grid.hi, h, margin=3)
    ls = L.from_analytic(jnp.array([5.0, 5.0]), 3.0, lo, h, m, n_sample=16)
    mesh = cut.build_mesh(ls, grid, radius=1, sub=4)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    x = jnp.stack([jax.random.uniform(k1, (n,), minval=3.5, maxval=6.5),
                   jax.random.uniform(k2, (n,), minval=8.02, maxval=top - 0.01)], axis=1)  # fmt: skip
    ang = -jax.random.uniform(k3, (n,), minval=0.05, maxval=3.09)  # down: disk, face, down
    speed = 3000.0
    v = jnp.stack([speed * jnp.cos(ang), speed * jnp.sin(ang), jnp.zeros(n)], axis=1)
    dt = move / speed
    faces = Faces((BC.SPECULAR,) * 4)
    wall = WallParams.uniform(2, 300.0, acc=0.0, n_models=1)
    out = wall_adjoint.advance(x, v, jnp.ones(n, bool), grid, faces, wall, MASS, dt,
                               jax.random.PRNGKey(1), mesh=mesh, tracker=tracker,
                               max_bounces=bounces, lane_size=n)  # fmt: skip
    return out[3]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    print("fixed stencil, 1 bounce:", run(n, 0, tracker="stencil", bounces=1))
    print("fixed lane, 2 bounces  :", run(n, 0))
    wall_adjoint._lane_distance = old_lane_distance
    jax.clear_caches()
    print("old lane, 2 bounces    :", run(n, 0))
    _ = functools
