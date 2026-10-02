"""Refinement iteration counts (while_loop trips) over a forward run (bench case)."""
import collections, dataclasses, functools, sys
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "/home/aakash/jax_dsmc/wt_levelset/scratch/perf"); sys.path.insert(0, "/home/aakash/jax_dsmc/wt_levelset/examples/mesh")
import bench_forward as B
from examples import ensemble_mesh as EM
from jax_dsmc import initialize
from jax_dsmc.mesh import levelset as L
from jax_dsmc.state import FlowParams

REC = collections.defaultdict(list)
def _rec(n, k, nf, maxper):
    REC[int(n)].append((int(k), int(nf)))

def _refine(ls, x, vhat, found, s_lo, s_hi, phi_lo, phi_hi, tol):
    s0 = s_lo + (s_hi - s_lo) * (phi_lo / (phi_lo - phi_hi))
    s0 = jnp.where(found, jnp.clip(s0, s_lo, s_hi), 0.0)
    carry = (jnp.int32(0), s0, s_lo, s_hi, s_hi - s_lo, ~found)
    carry = jax.lax.while_loop(
        functools.partial(L._refine_more, max_iter=ls.n_bisect),
        functools.partial(L._refine_step, ls=ls, x=x, vhat=vhat, tol=tol), carry)
    jax.debug.callback(functools.partial(_rec, x.shape[0]), carry[0], jnp.sum(found), 0)
    return carry[1]
L._refine = _refine
grid, mesh, sp, params, static, faces, res, wall, capacity = B.setup("levelset")
static = dataclasses.replace(static, n_steps=100)
flow = FlowParams(nrho=jnp.float64(1.0), temp=jnp.float64(273.15), vstream=jnp.array([100.0, 0, 0]))
key = jax.random.key(0)
state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
cstate = initialize.initial_collide_state(flow, grid, sp, static)
for k0 in range(0, 400, 100):
    (state, cstate), _ = EM._run(state, cstate, grid, sp, params, key, faces, wall, res, mesh, static, capacity, k0, False)
    jax.block_until_ready(state)
for n, v in sorted(REC.items()):
    v = np.asarray(v)
    print(n, "calls", len(v), "iters hist", np.bincount(v[:, 0]), "found mean", v[:, 1].mean())
