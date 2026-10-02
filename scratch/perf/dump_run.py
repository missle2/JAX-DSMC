import sys, os

sys.path.insert(0, "scratch/perf")
sys.path.insert(0, "examples/mesh")
import jax, jax.numpy as jnp
import bench_forward as B
from examples import ensemble_mesh as EM
from jax_dsmc import initialize
from jax_dsmc.state import FlowParams

grid, mesh, sp, params, static, faces, res, wall, capacity = B.setup(sys.argv[1])
flow = FlowParams(nrho=jnp.float64(1.0), temp=jnp.float64(273.15), vstream=jnp.array([100.0, 0, 0]))
state = initialize.initial_state(jax.random.key(0), flow, grid, sp, static, n_alive=0)
cstate = initialize.initial_collide_state(flow, grid, sp, static)
EM._run.lower(
    state,
    cstate,
    grid,
    sp,
    params,
    jax.random.key(0),
    faces,
    wall,
    res,
    mesh,
    static,
    capacity,
    0,
    False,
).compile()
