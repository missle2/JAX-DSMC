"""Steady-state counts of the SPARTA-structured tracker's lanes (near slots, CUT-cell near
slots, pass-0 events) on the SPARTA circle, both bodies."""

import os
import sys

import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "perf"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "examples", "mesh"))
import bench_forward as B  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import initialize, wall_adjoint  # noqa: E402
from jax_dsmc.grid import cell_of  # noqa: E402
from jax_dsmc.mesh import geometry as G  # noqa: E402
from jax_dsmc.mesh.types import CUT  # noqa: E402
from jax_dsmc.state import FlowParams  # noqa: E402

for body in sys.argv[1:] or ["polygon", "levelset"]:
    grid, mesh, sp, params, static, faces, res, wall, capacity = B.setup(body)
    flow = FlowParams(nrho=jnp.float64(1.0), temp=jnp.float64(273.15), vstream=jnp.array([100.0, 0, 0]))
    key = jax.random.key(0)
    state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    common = (grid, sp, params, key, faces, wall, res, mesh, static, capacity)
    for k0 in range(0, 800, B.BLOCK):
        (state, cstate), _ = EM._run(state, cstate, *common, k0, False)
    x, v, alive = state.x, state.v, state.alive
    budget = jnp.full(x.shape[0], params.dt)
    used, hit, face, full = wall_adjoint._box_events(x, v, alive, budget, grid, faces)
    reach = G.near_surface(x, v, budget, grid, mesh)
    near = alive & (full | reach)
    t, s = G.first_hit(x, v, grid, mesh, jnp.full(x.shape[0], -1, jnp.int32), budget)
    ev = alive & (t <= budget)
    cut = mesh.cell_type[cell_of(x, grid)] == CUT if mesh.kind == "surface" else near
    print(body, dict(N=x.shape[0], alive=int(alive.sum()), near=int(near.sum()),
          near_full=int((alive & full).sum()), near_cut=int((near & cut).sum()),
          box_hit=int(hit.sum()), event=int(ev.sum()), elem=int((ev & (s >= 4)).sum()),
          chunk=wall_adjoint.near_lane_size(x.shape[0])), flush=True)
