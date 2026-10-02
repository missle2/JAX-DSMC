"""Row counts behind the forward's compactions, on the polygon circle after 1000 steps."""
import dataclasses, os, sys
import jax, jax.numpy as jnp, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "perf"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "examples", "mesh"))
import bench_forward as B
from examples import ensemble_mesh as EM
from jax_dsmc import initialize
from jax_dsmc.grid import cell_of
from jax_dsmc.mesh import geometry as G
from jax_dsmc.mesh.types import CUT
from jax_dsmc.state import FlowParams

body = sys.argv[1] if len(sys.argv) > 1 else "polygon"
grid, mesh, sp, params, static, faces, res, wall, capacity = B.setup(body)
flow = FlowParams(nrho=jnp.float64(1.0), temp=jnp.float64(273.15), vstream=jnp.array([100.0, 0, 0]))
key = jax.random.key(0)
state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
cstate = initialize.initial_collide_state(flow, grid, sp, static)
for k0 in range(0, 1000, 100):
    (state, cstate), _ = EM._run(state, cstate, grid, sp, params, key, faces, wall, res, mesh, static, capacity, k0, False)
np.savez(os.path.join(HERE, f"state1000_{body}.npz"), x=state.x, v=state.v, alive=state.alive)
x, v, alive = state.x, state.v, state.alive
cell = cell_of(x, grid)
print("N", x.shape[0], "alive", int(alive.sum()))
if mesh.kind == "surface":
    cut = (mesh.cell_type[cell] == CUT)
    print("alive in CUT cells", int((cut & alive).sum()), "all in CUT", int(cut.sum()))
    reach = jnp.linalg.norm(v[:, :2], axis=1) * params.dt
    near = reach * (1 + G.POLYGON_MARGIN) >= G.cell_clearance(mesh, grid)[cell]
    print("pass-0 near (all slots)", int(near.sum()), "alive near", int((near & alive).sum()))
t, s = G.first_hit(x, v, grid, mesh, jnp.full(x.shape[0], -1, jnp.int32), jnp.full(x.shape[0], params.dt))
hit = alive & (t <= params.dt)
print("pass-0 hits", int(hit.sum()), "element hits", int((hit & (s >= 4)).sum()))
for f in range(4):
    print(" face", f, int((hit & (s == f)).sum()))
