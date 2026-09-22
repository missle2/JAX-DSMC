"""The truncated-horizon FD of the plane case: warm-up at theta = 0, the captured steps at
theta in {0, h, 2h} from the SAME warm state (one-sided so the body never sweeps over
particles), second-order one-sided differences.  This is the derivative the truncated
adjoint estimates (the capture-start state held fixed); the full CRN FD of
``ensemble_plane`` includes the warm-up's theta dependence.
"""

import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from examples.mesh import ensemble_plane as P
from jax_dsmc import initialize
from jax_dsmc.state import FlowParams, SimParams
from jax_dsmc.grid import num_cells

objective = sys.argv[1] if len(sys.argv) > 1 else "shear-window"
seeds = [int(a) for a in sys.argv[2:]] or list(range(8))
cfg = P.default_config()
cfg["steps"] = int(os.environ.get("STEPS", cfg["steps"]))
cfg["window"] = int(os.environ.get("WINDOW", cfg["window"]))
hs = [0.04, 0.02]


def captured(cfg, theta, state, cstate, key, capture):
    grid, mesh = P.build(cfg, theta)
    sp = P.species(cfg)
    c = num_cells(grid)
    params = SimParams(
        jnp.float64(cfg["dt"]), jnp.full(c, 1.0 / cfg["particles"]), jnp.float64(0.0)
    )
    common = (grid, sp, params, key, P.faces(), P.wall_params(cfg), mesh)
    (final, _), aux = P._run(state, cstate, *common, P.static_of(cfg), cfg["warmup"], capture)
    P._check(aux[1])
    return P.objective_value(cfg, objective, aux[2] if capture else None, aux[-1])


rows = []
for seed in seeds:
    key = jax.random.key(32 + seed)
    grid, mesh = P.build(cfg, 0.0)
    sp = P.species(cfg)
    c = num_cells(grid)
    params = SimParams(
        jnp.float64(cfg["dt"]), jnp.full(c, 1.0 / cfg["particles"]), jnp.float64(0.0)
    )
    flow = FlowParams(nrho=jnp.float64(1.0), temp=jnp.float64(1.0), vstream=jnp.zeros(3))
    cstate = initialize.initial_collide_state(flow, grid, sp, P.static_of(cfg))
    state = P.initial(cfg, 0.0, key)
    common = (grid, sp, params, key, P.faces(), P.wall_params(cfg), mesh)
    (state, cstate), aux = P._run(state, cstate, *common, P.static_of(cfg, cfg["warmup"]), 0, False)
    cap = objective == "energy-window"
    j0 = captured(cfg, 0.0, state, cstate, key, cap)
    row = {"seed": seed, "J0": j0}
    for h in hs:
        j1 = captured(cfg, h, state, cstate, key, cap)
        j2 = captured(cfg, 2 * h, state, cstate, key, cap)
        row[f"fd1_{h}"] = (j1 - j0) / h
        row[f"fd2_{h}"] = (-3 * j0 + 4 * j1 - j2) / (2 * h)
    rows.append(row)
    print(row, flush=True)
for k in rows[0]:
    if k.startswith("fd"):
        a = np.array([r[k] for r in rows])
        print(f"{k}: {a.mean():+.5f} ± {a.std(ddof=1) / np.sqrt(len(a)):.5f}")
json.dump(
    {"config": cfg, "objective": objective, "rows": rows},
    open(
        f"examples/mesh/results/plane_truncated_fd_{objective}_M{cfg['steps']}_W{cfg['window']}.json",
        "w",
    ),
    indent=1,
)
