"""Lower + compile time of one 100-step forward block (the optimiser's `_run`, uncaptured)
for the polygon and level-set circle, against whichever tree is first on PYTHONPATH
(JD_ROOT): the working tree or a clean checkout of HEAD.

    JD_ROOT=/tmp/jd_head PYTHONPATH=/tmp/jd_head python3 scratch/perf/compile_time.py
"""

import dataclasses
import json
import os
import sys
import time

import jax
import jax.numpy as jnp

ROOT = os.environ.get(
    "JD_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
sys.path.insert(0, os.path.join(ROOT, "examples", "mesh"))
import forward as F  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import initialize, simulate  # noqa: E402
from jax_dsmc.state import FlowParams  # noqa: E402


def objs(body):
    if body == "polygon":
        grid, mesh, sp, params, static, faces, res, wall, _t, capacity = F.build(
            "circle", "polygon", "lane", "Ar", 0.0, 1
        )
    else:
        cfg = dict(EM.default_config(), body="levelset", levelset_h=0.0625, levelset_samples=16)
        cfg["fnum_scale"] = 1.0
        grid, mesh, sp, params, static, faces, res, wall, capacity = EM.build(
            cfg, "radius", 3.0, 100
        )
    return (
        grid,
        mesh,
        sp,
        params,
        dataclasses.replace(static, n_steps=100),
        faces,
        res,
        wall,
        capacity,
    )


rows = {}
for body in ("polygon", "levelset"):
    grid, mesh, sp, params, static, faces, res, wall, capacity = objs(body)
    flow = FlowParams(
        nrho=jnp.float64(1.0), temp=jnp.float64(273.15), vstream=jnp.array([100.0, 0, 0])
    )
    state = initialize.initial_state(jax.random.key(0), flow, grid, sp, static, n_alive=0)
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    fn = jax.jit(
        simulate.run,
        static_argnames=("static", "faces", "capacity", "emit_moments", "capture", "emit_surface"),
    )
    kw = dict(
        faces=faces,
        wall=wall,
        res=res,
        capacity=capacity,
        emit_moments="none",
        step0=0,
        mesh=mesh,
        capture=False,
        emit_surface=True,
    )
    t0 = time.perf_counter()
    lowered = fn.lower(state, cstate, grid, sp, params, static, jax.random.key(0), **kw)
    t1 = time.perf_counter()
    lowered.compile()
    t2 = time.perf_counter()
    rows[body] = {"lower": t1 - t0, "compile": t2 - t1, "hlo_chars": len(lowered.as_text())}
    print(body, rows[body], flush=True)
print(json.dumps({"root": ROOT, "rows": rows}))
