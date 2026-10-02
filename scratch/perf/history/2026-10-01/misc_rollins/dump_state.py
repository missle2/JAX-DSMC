"""Steady-state polygon state (800 steps) -> /tmp/misc_state.npz (x, v, alive, cell, grid)."""
import os, sys
import jax, jax.numpy as jnp, numpy as np
HERE = "/home/aakash/jax_dsmc/wt_misc/scratch/perf"
sys.path.insert(0, HERE); sys.path.insert(0, "/home/aakash/jax_dsmc/wt_misc/examples/mesh")
import bench_forward as B
from examples import ensemble_mesh as EM
from jax_dsmc import initialize, sort
from jax_dsmc.state import FlowParams
body = sys.argv[1] if len(sys.argv) > 1 else "polygon"
grid, mesh, sp, params, static, faces, res, wall, capacity = B.setup(body)
flow = FlowParams(nrho=jnp.float64(1.0), temp=jnp.float64(273.15), vstream=jnp.array([100.0, 0, 0]))
key = jax.random.key(0)
state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
cstate = initialize.initial_collide_state(flow, grid, sp, static)
common = (grid, sp, params, key, faces, wall, res, mesh, static, capacity)
for k0 in range(0, 800, B.BLOCK):
    (state, cstate), _ = EM._run(state, cstate, *common, k0, False)
t = sort.build_table(state, grid)
np.savez(f"/tmp/misc_state_{body}.npz", x=state.x, v=state.v, alive=state.alive, cell=t.cell,
         order=t.order, counts=t.counts, logh=state.logh_acc)
import pickle
pickle.dump(grid, open(f"/tmp/misc_grid_{body}.pkl", "wb"))
print("saved", int(state.alive.sum()), t.counts.shape)
