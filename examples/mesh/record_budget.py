"""G9: the memory of the captured event record for the shape-gradient case, by `eval_shape`.

The shape adjoint replays the wall map from the recorded element ids, so the record is the
existing `EventRecord` (post-move state, births, pair decisions, `WallEvents` with one row
per bounce pass) captured only over the last ``W + H`` steps.  This prints the bytes of one
captured block for the reduced circle case (``fnum x 4``: ~12 k alive, capacity ``--slots``)
without running anything, and writes ``results/record_budget.json``.

    JAX_PLATFORMS=cpu python3 examples/mesh/record_budget.py --slots 16000 --steps 300 --k-bounce 2 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from forward import NRHO, RESULTS, TEMP, VSTREAM, build  # noqa: E402

from jax_dsmc import initialize, simulate  # noqa: E402
from jax_dsmc.state import FlowParams  # noqa: E402


def record_bytes(slots: int, steps: int, k_bounce: int, body: str = "analytic") -> dict:
    grid, mesh, sp, params, static, faces, res, wall, _target, capacity = build(
        "circle", body, "stencil", "Ar", 0.0, 1
    )
    static = dataclasses.replace(static, N=slots, K_bounce=k_bounce, n_steps=steps)
    params = dataclasses.replace(params, fnum=params.fnum * 4.0)
    flow = FlowParams(
        nrho=jnp.float64(NRHO), temp=jnp.float64(TEMP), vstream=jnp.array([VSTREAM, 0.0, 0.0])
    )
    key = jax.random.PRNGKey(0)
    state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    out = jax.eval_shape(
        lambda s, c: simulate.run(
            s,
            c,
            grid,
            sp,
            params,
            static,
            key,
            faces=faces,
            wall=wall,
            res=res,
            capacity=capacity,
            emit_moments=False,
            mesh=mesh,
            capture=True,
        ),
        state,
        cstate,
    )
    record = out[1][2]
    sizes = {}
    total = 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(record):
        b = int(leaf.size) * leaf.dtype.itemsize
        key_name = jax.tree_util.keystr(path)
        sizes[key_name] = b
        total += b
    return {
        "slots": slots,
        "steps": steps,
        "k_bounce": k_bounce,
        "capacity": capacity,
        "bytes": total,
        "gib": total / 2**30,
        "bytes_per_slot_step": total / (slots * steps),
        "leaves": sizes,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--slots", type=int, default=16000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--k-bounce", type=int, nargs="+", default=[2, 4])
    args = ap.parse_args(argv)
    rows = [record_bytes(args.slots, args.steps, k) for k in args.k_bounce]
    for r in rows:
        print(
            f"slots {r['slots']} x {r['steps']} steps, K_bounce {r['k_bounce']}: {r['gib']:.2f} GiB "
            f"({r['bytes_per_slot_step']:.0f} B per slot-step); pass (<= 1 GiB): {r['gib'] <= 1.0}"
        )
    big = sorted(rows[-1]["leaves"].items(), key=lambda kv: -kv[1])[:6]
    print(
        "largest leaves at K_bounce",
        rows[-1]["k_bounce"],
        ":",
        ", ".join(f"{k} {v / 2**20:.0f} MiB" for k, v in big),
    )
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "record_budget.json"), "w") as fh:
        json.dump({"rows": rows, "pass": all(r["gib"] <= 1.0 for r in rows)}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
