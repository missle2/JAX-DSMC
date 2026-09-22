import time
import numpy as np, jax, jax.numpy as jnp
from jax_dsmc.state import GridSpec
from jax_dsmc.boundary import BC, Faces, WallParams
from jax_dsmc.mesh import cut, surface as S
from jax_dsmc.mesh.types import AnalyticBody
from jax_dsmc import wall_adjoint

grid = GridSpec(lo=jnp.array([0.,0.]), hi=jnp.array([10.,10.]), n=(20,20))
faces = Faces((BC.OUTFLOW, BC.OUTFLOW, BC.SPECULAR, BC.SPECULAR))
wall = WallParams.uniform(2, 300.0, acc=0.0, n_models=1)
mass = 6.63e-26
body = AnalyticBody(centre=jnp.array([5.,5.]), radius=jnp.float64(3.0))
mesh_a = cut.build_mesh(body, grid)
poly = S.read_sparta_surf('examples/mesh/sparta/data.circle', 2)
mesh_p = cut.build_mesh(poly, grid)
key = jax.random.PRNGKey(0)
N = 20000
kx, kv = jax.random.split(key)
x = jax.random.uniform(kx, (N, 2), dtype=jnp.float64) * 10.0
v = jax.random.normal(kv, (N, 3), dtype=jnp.float64) * 337.0
v = v.at[:, 0].add(100.0)
# start outside the body
alive = jnp.sum((x - 5.0) ** 2, axis=1) > 9.0
dt = 1e-4
for name, mesh in (("analytic", mesh_a), ("polygon", mesh_p)):
    for tracker in ("stencil", "walk"):
        t0 = time.time()
        out = wall_adjoint.advance(x, v, alive, grid, faces, wall, mass, dt, key, mesh=mesh, tracker=tracker, max_bounces=4, max_walk=16, return_tallies=True, return_events=True)
        xo, vo, ao, counts, logp, events, tally = jax.block_until_ready(out)
        print(name, tracker, "counts", tuple(int(c) for c in counts), "alive", int(ao.sum()), "hits", int(tally.hits[4:].sum()), "t", round(time.time()-t0, 2))
        if tracker == "stencil":
            ref = (xo, vo, ao)
        else:
            print("   walk vs stencil: max|dx|", float(jnp.abs(xo - ref[0]).max()), "max|dv|", float(jnp.abs(vo - ref[1]).max()), "alive same", bool((ao == ref[2]).all()))
