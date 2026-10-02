"""Forward throughput of the SPARTA circle case (argon, 20 x 20, r = 3, fnum 1e-3, 1000
steps from an empty box -- ``examples/mesh/sparta/in.circle_ar``), for the comparison with
SPARTA's own CPU runs (scratch/perf/bench_sparta.sh).

Bodies: ``polygon`` (SPARTA's data.circle, the geometry SPARTA runs) and ``levelset`` (the
optimiser's body: h_phi 0.0625, 16 samples).  Lane tracker, K_bounce 32, collisions on
(NTC; none are accepted at this Kn, but the stage runs).  The run is 10 blocks of 100 steps
(one compile, then a timed repeat); particle moves are counted as SPARTA does, the alive
count summed over steps (trapezoid over each block's ends).

    XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false \
        PYTHONPATH=. python3 scratch/perf/bench_forward.py [--bodies polygon levelset]
    JAX_PLATFORMS=cpu PYTHONPATH=. python3 scratch/perf/bench_forward.py   # CPU
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import jax
import jax.numpy as jnp

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "examples", "mesh"))
import forward as F  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import initialize  # noqa: E402
from jax_dsmc.state import FlowParams  # noqa: E402

BLOCK = 100
STEPS = 1000


def setup(body):
    if body == "polygon":
        grid, mesh, sp, params, static, faces, res, wall, _t, capacity = F.build(
            "circle", "polygon", "lane", "Ar", 0.0, 1
        )
    else:
        cfg = dict(
            EM.default_config(), body="levelset", levelset_h=0.0625, levelset_samples=16,
            fnum_scale=1.0,
        )  # fmt: skip
        grid, mesh, sp, params, static, faces, res, wall, capacity = EM.build(
            cfg, "radius", F.CASES["circle"]["radius"], STEPS
        )
    static = dataclasses.replace(static, n_steps=BLOCK)
    return grid, mesh, sp, params, static, faces, res, wall, capacity


def run_once(objs, key):
    grid, mesh, sp, params, static, faces, res, wall, capacity = objs
    flow = FlowParams(
        nrho=jnp.float64(F.NRHO), temp=jnp.float64(F.TEMP), vstream=jnp.array([F.VSTREAM, 0.0, 0.0])
    )
    state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    moves, alive0 = 0.0, 0
    t0 = time.perf_counter()
    for k0 in range(0, STEPS, BLOCK):
        (state, cstate), _aux = EM._run(
            state,
            cstate,
            grid,
            sp,
            params,
            key,
            faces,
            wall,
            res,
            mesh,
            static,
            capacity,
            k0,
            False,
        )
        alive1 = int(jnp.sum(state.alive))  # blocks: syncs once per 100 steps
        moves += BLOCK * 0.5 * (alive0 + alive1)
        alive0 = alive1
    return time.perf_counter() - t0, moves, alive0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", nargs="*", default=["polygon", "levelset"])
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    platform = jax.devices()[0].platform
    device = jax.devices()[0].device_kind
    rows = []
    for body in args.bodies:
        objs = setup(body)
        t_first, _, _ = run_once(objs, jax.random.key(0))  # compile + run
        times = []
        for r in range(args.repeats):
            t, moves, n_final = run_once(objs, jax.random.key(1 + r))
            times.append(t)
        row = {
            "platform": platform,
            "device": device,
            "body": body,
            "seconds_first_incl_compile": t_first,
            "seconds": min(times),
            "moves": moves,
            "final_particles": n_final,
            "moves_per_second": moves / min(times),
            "slots": objs[4].N,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = os.path.join(ROOT, "scratch", "perf", f"bench_forward_{platform}.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
