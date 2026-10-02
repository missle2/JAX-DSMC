"""Bitwise gate for forward speed-ups: 300 steps of the SPARTA circle case from an empty
box (polygon and level-set bodies, lane tracker), the final state, the per-step surface
tallies and diagnostics saved, or compared against a saved run.

    ... python3 scratch/perf/forward_digest.py --save base_gpu
    ... python3 scratch/perf/forward_digest.py --compare base_gpu
(CPU: JAX_PLATFORMS=cpu, tag e.g. base_cpu)
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "examples", "mesh"))
import bench_forward as B  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import initialize  # noqa: E402
from jax_dsmc.state import FlowParams  # noqa: E402

DIR = os.path.join(HERE, "digest")
STEPS = 300


def run(body):
    grid, mesh, sp, params, static, faces, res, wall, capacity = B.setup(body)
    static = dataclasses.replace(static, n_steps=STEPS)
    flow = FlowParams(
        nrho=jnp.float64(1.0), temp=jnp.float64(273.15), vstream=jnp.array([100.0, 0, 0])
    )
    key = jax.random.key(7)
    state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    t0 = time.perf_counter()
    (state, cstate), aux = EM._run(
        state, cstate, grid, sp, params, key, faces, wall, res, mesh, static, capacity, 0, False
    )
    jax.block_until_ready(state)
    out = {"x": state.x, "v": state.v, "alive": state.alive, "vremax": cstate.vremax}
    tallies = aux[-1]
    for name in ("momentum", "energy", "heat", "hits"):
        value = getattr(tallies, name, None)
        if value is not None:
            out[f"tally_{name}"] = value
    for name, value in dataclasses.asdict(aux[1]).items():
        out[f"diag_{name}"] = value
    print(body, f"{time.perf_counter() - t0:.1f} s (incl. compile)", flush=True)
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--bodies", nargs="*", default=["polygon", "levelset"])
    args = ap.parse_args()
    os.makedirs(DIR, exist_ok=True)
    worst = []
    for body in args.bodies:
        got = run(body)
        if args.save:
            np.savez(os.path.join(DIR, f"{args.save}_{body}.npz"), **got)
        if args.compare:
            ref = np.load(os.path.join(DIR, f"{args.compare}_{body}.npz"))
            for k in sorted(ref.files):
                a, b = got[k], ref[k]
                same = np.array_equal(a, b)
                if not same:
                    diff = np.abs(a.astype(float) - b.astype(float))
                    worst.append((body, k, float(np.nanmax(diff))))
                    print(f"  {body} {k}: DIFFERS, max |diff| {np.nanmax(diff):.3e}")
            print(f"{body}: {len(ref.files)} arrays compared", flush=True)
    if args.compare:
        print("BITWISE IDENTICAL" if not worst else f"DIFFERENCES: {len(worst)}")


if __name__ == "__main__":
    main()
