"""Jaxpr census of one forward block: every equation on a particle-sized operand or result
(leading dim >= N_MIN), counted by primitive and by the jax_dsmc function it comes from --
the fusion barriers (reductions, cumsums, scatters, gathers, sorts, loop and branch
boundaries) between the full-array passes.
    JAX_PLATFORMS=cpu PYTHONPATH=. python3 scratch/xla_ops/jaxpr_census.py [polygon]"""
import collections
import os
import sys

import jax
import jax.extend.core as jcore
import jax.numpy as jnp

sys.path.insert(0, "scratch/perf")
sys.path.insert(0, "examples/mesh")
import bench_forward as B  # noqa: E402
import forward as F  # noqa: E402
import xla_ops as X  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import initialize  # noqa: E402
from jax_dsmc.state import FlowParams  # noqa: E402

N_MIN = 20000
BARRIERS = {"reduce_sum", "reduce_max", "reduce_min", "reduce_or", "reduce_and", "argmin",
            "argmax", "cumsum", "cumlogsumexp", "cummax", "scatter", "scatter-add",
            "scatter_add", "scatter-max", "gather", "sort", "while", "cond", "scan",
            "dynamic_slice", "dynamic_update_slice", "concatenate", "reduce_precision",
            "select_and_scatter_add", "pjit", "custom_jvp_call", "custom_vjp_call"}


def where(eqn):
    tb = eqn.source_info.traceback
    for fr in (tb.frames if tb else []):
        if "/jax_dsmc/" in fr.file_name:
            return f"{os.path.basename(fr.file_name)}:{fr.function_name}"
    return "?"


def big(eqn):
    for v in list(eqn.invars) + list(eqn.outvars):
        shape = getattr(getattr(v, "aval", None), "shape", ())
        if shape and shape[0] >= N_MIN:
            return True
    return False


def subjaxprs(eqn):
    """Every jaxpr nested in an equation's params (call, remat, loop and branch bodies)."""
    out = []
    for val in eqn.params.values():
        for item in val if isinstance(val, (list, tuple)) else [val]:
            if isinstance(item, jcore.ClosedJaxpr):
                out.append(item.jaxpr)
            elif isinstance(item, jcore.Jaxpr):
                out.append(item)
    return out


def walk(jaxpr, mult, counts, loops):
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        if big(eqn):
            counts[(name, where(eqn))] += mult
        if name in ("cond", "while"):
            loops[f"{name}{' (particle-sized)' if big(eqn) else ''}"] += mult
        trips = eqn.params["length"] if name == "scan" else 1  # while bodies: once
        for sub in subjaxprs(eqn):
            walk(sub, mult * trips, counts, loops)


grid, mesh, sp, params, static, faces, res, wall, capacity = B.setup(sys.argv[1] if len(sys.argv) > 1 else "polygon")
flow = FlowParams(nrho=jnp.float64(F.NRHO), temp=jnp.float64(F.TEMP), vstream=jnp.array([F.VSTREAM, 0.0, 0.0]))
key = jax.random.key(0)
state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
cstate = initialize.initial_collide_state(flow, grid, sp, static)
closed = jax.make_jaxpr(EM._run.__wrapped__, static_argnums=(6, 10, 11, 13))(
    state, cstate, grid, sp, params, key, faces, wall, res, mesh, static, capacity, 0, False)
counts, loops = collections.Counter(), collections.Counter()
walk(closed.jaxpr, 1, counts, loops)
steps = static.n_steps
per = collections.Counter()
for (name, w), c in counts.items():
    per[name] += c / steps
print(f"particle-sized equations per step: {sum(per.values()):.0f}  (N >= {N_MIN}; while bodies counted once)")
for name, c in per.most_common(40):
    print(f"{c:7.1f}  {name}")
print("--- barriers by function (per step)")
byf = collections.Counter()
for (name, w), c in counts.items():
    if name in BARRIERS:
        byf[(w, name)] += c / steps
for (w, name), c in byf.most_common(60):
    print(f"{c:6.1f}  {name:22s} {w}")
print(loops)
