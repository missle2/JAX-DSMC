"""Per-block (100-step) wall times of the one-core forward from empty, 3 runs after the
compile: where in the 1000 steps two versions differ.
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false PYTHONPATH=. \
        taskset -c 3 python3 scratch/xla_ops/blocks.py [polygon|levelset]"""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "scratch/perf")
sys.path.insert(0, "examples/mesh")
import bench_forward as B  # noqa: E402
import forward as F  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import initialize  # noqa: E402
from jax_dsmc.state import FlowParams  # noqa: E402

grid, mesh, sp, params, static, faces, res, wall, capacity = B.setup(sys.argv[1] if len(sys.argv) > 1 else "polygon")
flow = FlowParams(nrho=jnp.float64(F.NRHO), temp=jnp.float64(F.TEMP), vstream=jnp.array([F.VSTREAM, 0.0, 0.0]))
runs = []
for rep in range(4):
    key = jax.random.key(0)
    state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    times = []
    for k0 in range(0, B.STEPS, B.BLOCK):
        t0 = time.perf_counter()
        (state, cstate), _ = EM._run(state, cstate, grid, sp, params, key, faces, wall, res, mesh, static, capacity, k0, False)
        jax.block_until_ready(state)
        times.append(time.perf_counter() - t0)
    if rep:
        runs.append(times)
best = np.min(np.array(runs), axis=0)
print("best ms/step per block:", " ".join(f"{t * 10:.2f}" for t in best), f"| total {best.sum():.3f} s")
