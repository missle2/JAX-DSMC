"""Where one forward step of the SPARTA circle case spends its time, by code path: the
stage functions are wrapped in ``jax.named_scope`` (metadata only -- no change to the
program's arithmetic) at their call sites before tracing, and 200 steps (after a compiled
warm-up block) are profiled.  Bodies: ``polygon`` (SPARTA's geometry) and ``levelset``.

    XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false \
        PYTHONPATH=. python3 scratch/perf/profile_forward.py
-> scratch/perf/profile/forward_<body>/ (trace, events.npz via profile_extract.extract)
   and scratch/perf/profile/forward_scopes.json (GPU time and kernel count per scope)
"""

from __future__ import annotations

import functools
import json
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
import profile_extract  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import boundary, inflow, initialize, sample, simulate, sort, wall_adjoint  # noqa: E402
from jax_dsmc.collide import ntc  # noqa: E402
from jax_dsmc.mesh import geometry as mesh_geometry  # noqa: E402
from jax_dsmc.mesh import levelset  # noqa: E402
from jax_dsmc.state import FlowParams  # noqa: E402

# (module, attribute, scope): the attribute is replaced by a scoped wrapper; the name is
# looked up at trace time by its callers, so every call site is covered
SCOPES = [
    (wall_adjoint, "advance", "move"),
    (wall_adjoint, "_dense_pass", "dense_pass"),
    (wall_adjoint, "_near_chunk", "near_chunk"),
    (wall_adjoint, "ranked_slots", "ranked_slots"),
    (wall_adjoint, "_slot_uniforms", "slot_uniforms"),
    (wall_adjoint, "_bounce", "bounce_pass"),
    (wall_adjoint, "_apply_events", "wall_events"),
    (wall_adjoint, "_next_surface", "surface_search"),
    (mesh_geometry, "inside_polygon", "inside_check"),
    (levelset, "hit_body", "levelset_hit"),
    (inflow, "inject", "inflow"),
    (inflow, "advance", "inflow_tracking"),
    (inflow, "sample_face_velocities", "inflow_velocities"),
    (inflow, "sample_face_positions", "inflow_positions"),
    (sort, "build_table", "cell_sort"),
    (sort, "update_table", "cell_sort"),
    (sample, "cell_temperature", "cell_temperature"),
    (simulate, "collide", "collide"),
    (ntc, "certified_majorant", "majorant"),
    (ntc, "_slot", "ntc_slots"),
    (boundary, "fold_index", "fold_checksum"),
]
ORDER = [s for _m, _a, s in SCOPES]


def _scoped(scope, fn, *args, **kwargs):
    with jax.named_scope(scope):
        return fn(*args, **kwargs)


def install():
    for module, attr, scope in SCOPES:
        original = getattr(module, attr)
        wrapped = functools.wraps(original)(functools.partial(_scoped, scope, original))
        setattr(module, attr, wrapped)


def innermost(path):
    """The last of our scopes on an op path (``"unscoped"`` if none)."""
    best, pos = "unscoped", -1
    for scope in ORDER:
        i = path.rfind("/" + scope + "/")
        j = path.rfind("/" + scope) if path.endswith("/" + scope) else -1
        k = max(i, j)
        if k > pos:
            best, pos = scope, k
    return best


def main():
    install()
    out_root = os.path.join(HERE, "profile")
    summary = {}
    for body in ("polygon", "levelset"):
        objs = B.setup(body)
        grid, mesh, sp, params, static, faces, res, wall, capacity = objs
        flow = FlowParams(
            nrho=jnp.float64(1.0), temp=jnp.float64(273.15), vstream=jnp.array([100.0, 0.0, 0.0])
        )
        key = jax.random.key(0)
        state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
        cstate = initialize.initial_collide_state(flow, grid, sp, static)
        # fill to the steady state (8 blocks = 800 steps), then profile 2 blocks
        for k0 in range(0, 800, B.BLOCK):
            (state, cstate), _ = EM._run(
                state, cstate, grid, sp, params, key, faces, wall, res, mesh, static, capacity,
                k0, False,
            )  # fmt: skip
        jax.block_until_ready(state)
        alive = int(jnp.sum(state.alive))
        out = os.path.join(out_root, f"forward_{body}")
        jax.profiler.start_trace(out)
        t0 = time.perf_counter()
        for k0 in (800, 900):
            (state, cstate), _ = EM._run(
                state, cstate, grid, sp, params, key, faces, wall, res, mesh, static, capacity,
                k0, False,
            )  # fmt: skip
        jax.block_until_ready(state)
        wall_s = time.perf_counter() - t0
        jax.profiler.stop_trace()
        profile_extract.extract(out)
        ev = np.load(os.path.join(out, "events.npz"))["events"]
        ops = json.load(open(os.path.join(out, "events.json")))["op"]
        per = {}
        for o, d in zip(ev[:, 3].astype(int), ev[:, 1], strict=True):
            s = innermost(ops[o])
            t, n = per.get(s, (0.0, 0))
            per[s] = (t + d / 1e9, n + 1)
        steps = 200
        summary[body] = {
            "alive": alive,
            "slots": static.N,
            "wall_seconds_200_steps": wall_s,
            "gpu_busy_seconds": float(ev[:, 1].sum() / 1e9),
            "kernels_per_step": ev.shape[0] / steps,
            "scopes": {
                s: {"gpu_ms_per_step": t / steps * 1e3, "kernels_per_step": n / steps}
                for s, (t, n) in sorted(per.items(), key=lambda kv: -kv[1][0])
            },
        }
        print(body, json.dumps(summary[body], indent=1), flush=True)
    with open(os.path.join(out_root, "forward_scopes.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
