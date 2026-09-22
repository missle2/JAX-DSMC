"""Bisect the GPU segfault of the sphere forward: run advance() alone on a filled state."""
import sys, time
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "examples/mesh")
from forward import build
from jax_dsmc import wall_adjoint
grid, mesh, sp, params, static, faces, res, wall, target, capacity = build("sphere", sys.argv[1] if len(sys.argv) > 1 else "polygon", "stencil", "Ar", 0.0, 1)
N = static.N
key = jax.random.PRNGKey(0)
kx, kv = jax.random.split(key)
x = jax.random.uniform(kx, (N, 3), dtype=jnp.float64) * 4.0 - 2.0
v = jax.random.normal(kv, (N, 3), dtype=jnp.float64) * 240.0 + jnp.array([100.0, 0.0, 0.0])
alive = jnp.sum(x**2, axis=1) > 1.05
print("alive", int(alive.sum()), flush=True)
for return_tallies in (False, True):
    t0 = time.time()
    out = wall_adjoint.advance(x, v, alive, grid, faces, wall, float(sp.mass[0]), 1e-4, key, mesh=mesh, return_tallies=return_tallies, return_events=True)
    jax.block_until_ready(out)
    print("advance ok tallies=", return_tallies, tuple(int(c) for c in out[3]), time.time() - t0, flush=True)

from jax_dsmc import simulate, initialize
from jax_dsmc.state import ParticleState, FlowParams
flow = FlowParams(nrho=jnp.float64(1.0), temp=jnp.float64(273.15), vstream=jnp.array([100.0, 0.0, 0.0]))
cstate = initialize.initial_collide_state(flow, grid, sp, static)
state = ParticleState(x=jnp.where(alive[:, None], x, grid.lo), v=jnp.where(alive[:, None], v, 0.0), species=jnp.zeros(N, jnp.int32), alive=alive & (jnp.arange(N) < 60000), logh_acc=jnp.zeros(N))
mode = sys.argv[2] if len(sys.argv) > 2 else "full"
kw = dict(faces=faces, wall=wall, res=res, capacity=capacity, emit_moments=False, mesh=mesh, emit_surface=True)
if mode == "nores": kw.pop("res"); kw["capacity"] = 0
if mode == "nosurf": kw["emit_surface"] = False
step = jax.jit(lambda s, c, k0: simulate.run(s, c, grid, sp, params, static, key, step0=k0, **kw))
for k0 in range(0, 30, 10):
    t0 = time.time()
    (state, cstate), (_, diag, *rest) = step(state, cstate, k0)
    jax.block_until_ready(state)
    print("block", k0, "alive", int(state.alive.sum()), "hits", int(diag.surface_hits.sum()), time.time() - t0, flush=True)
