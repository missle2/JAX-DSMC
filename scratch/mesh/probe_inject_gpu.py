import sys, time
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "examples/mesh")
from forward import build
from jax_dsmc import inflow, wall_adjoint
from jax_dsmc.state import ParticleState
body = sys.argv[1] if len(sys.argv) > 1 else "polygon"
grid, mesh, sp, params, static, faces, res, wall, target, capacity = build("sphere", body, "stencil", "Ar", 0.0, 1)
N = static.N
key = jax.random.PRNGKey(0)
state = ParticleState(x=jnp.broadcast_to(grid.lo, (N, 3)), v=jnp.zeros((N, 3)), species=jnp.zeros(N, jnp.int32), alive=jnp.zeros(N, bool), logh_acc=jnp.zeros(N))
inj = jax.jit(lambda s, k: inflow.inject(s, grid, faces, res, sp.mass[0], params.fnum[0], params.dt, k, capacity, wall=wall, max_bounces=4, track=True, return_diagnostics=True, mesh=mesh))
for i in range(40):
    t0 = time.time()
    out = inj(state, jax.random.fold_in(key, i))
    jax.block_until_ready(out)
    print(i, int(out[1]), round(time.time() - t0, 2), flush=True)
