"""Candidate counts of the culled level-set calls over a forward run (bench case)."""
import collections, dataclasses, functools, os, sys
import jax, jax.numpy as jnp, numpy as np
HERE = "/home/aakash/jax_dsmc/wt_levelset/scratch/perf"
sys.path.insert(0, HERE); sys.path.insert(0, "/home/aakash/jax_dsmc/wt_levelset/examples/mesh")
import bench_forward as B
from examples import ensemble_mesh as EM
from jax_dsmc import initialize
from jax_dsmc.mesh import levelset as L
from jax_dsmc.state import FlowParams

REC = collections.defaultdict(list)
_orig = L._culled

def _record(name, n, count):
    REC[(str(name), int(n))].append(int(count))

def _wrapped(fn, fill, ls, rows, candidate):
    name = fn.__name__
    jax.debug.callback(functools.partial(_record, name, candidate.shape[0]), jnp.sum(candidate))
    return _orig(fn, fill, ls, rows, candidate)

L._culled = _wrapped
grid, mesh, sp, params, static, faces, res, wall, capacity = B.setup("levelset")
static = dataclasses.replace(static, n_steps=100)
flow = FlowParams(nrho=jnp.float64(1.0), temp=jnp.float64(273.15), vstream=jnp.array([100.0, 0, 0]))
key = jax.random.key(0)
state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
cstate = initialize.initial_collide_state(flow, grid, sp, static)
for k0 in range(0, 400, 100):
    (state, cstate), _ = EM._run(state, cstate, grid, sp, params, key, faces, wall, res, mesh, static, capacity, k0, False)
    jax.block_until_ready(state)
    if k0 in (300,):
        print("block", k0, "alive", int(jnp.sum(state.alive)))
        for k, v in sorted(REC.items()):
            v = np.asarray(v)
            print(f"  {k}: calls {v.size} mean {v.mean():.1f} p99 {np.quantile(v, 0.99):.0f} max {v.max()} q {np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9]).round()}")
        REC.clear()
