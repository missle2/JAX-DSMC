"""Serial-CPU profile of the SPARTA circle forward (polygon), one core, XLA:CPU threading
off: 100 steady-state steps traced at host-tracer level 3 (one event per executed HLO
op), each op attributed to its source line through the CPU program's optimised HLO dump,
then the same categories as forward_breakdown.py.

    rm -rf /tmp/hlo_cpu
    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false \
        --xla_dump_to=/tmp/hlo_cpu \
        --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*_run.*" PYTHONPATH=. \
        taskset -c 3 python3 scratch/perf/profile_forward_cpu.py
"""

import glob
import json
import os
import re
import sys
import time

import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "examples", "mesh"))
import bench_forward as B  # noqa: E402
import forward_breakdown as FB  # noqa: E402
import xspace  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import initialize  # noqa: E402
from jax_dsmc.state import FlowParams  # noqa: E402

CONTAINERS = ("while", "conditional", "call", "fusion_region")


def op_sources(dump):
    """{op name: 'file:line'} over every instruction of the optimised CPU module."""
    path = glob.glob(os.path.join(dump, "*jit__run*after_optimizations.txt"))[0]
    out = {}
    for line in open(path):
        m = re.match(
            r"\s*(?:ROOT )?%?([\w.\-]+) = .*?source_file=\"([^\"]+)\" source_line=(\d+)", line
        )
        if m:
            out.setdefault(m.group(1), f"{os.path.basename(m.group(2))}:{m.group(3)}")
    return out


def main():
    body = sys.argv[1] if len(sys.argv) > 1 else "polygon"
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    grid, mesh, sp, params, static, faces, res, wall, capacity = B.setup(body)
    flow = FlowParams(
        nrho=jnp.float64(1.0), temp=jnp.float64(273.15), vstream=jnp.array([100.0, 0, 0])
    )
    key = jax.random.key(0)
    state = initialize.initial_state(key, flow, grid, sp, static, n_alive=0)
    cstate = initialize.initial_collide_state(flow, grid, sp, static)
    common = (grid, sp, params, key, faces, wall, res, mesh, static, capacity)
    for k0 in range(0, 800, B.BLOCK):
        (state, cstate), _ = EM._run(state, cstate, *common, k0, False)
    jax.block_until_ready(state)
    out = os.path.join(HERE, "profile", f"forward_cpu_serial_{body}{tag}")
    opts = jax.profiler.ProfileOptions()
    opts.host_tracer_level = 3
    opts.python_tracer_level = 0
    jax.profiler.start_trace(out, profiler_options=opts)
    t0 = time.perf_counter()
    (state, cstate), _ = EM._run(state, cstate, *common, 800, False)
    jax.block_until_ready(state)
    wall_s = time.perf_counter() - t0
    jax.profiler.stop_trace()
    src = op_sources(os.environ.get("HLO_DUMP", "/tmp/hlo_cpu"))
    traces = glob.glob(os.path.join(out, "plugins", "profile", "*", "*.xplane.pb"))
    path = max(traces, key=os.path.getmtime)  # the run just traced, not an older one beside it
    buf, planes = xspace.planes(path)
    host = next(s for n, s in planes if n == "/host:CPU")
    cats, ops, lines, unattributed, total = {}, {}, {}, {}, 0.0
    for _line, _name, _start, dur, stats in xspace.plane_events(buf, host, event_stats=True):
        op = stats.get("hlo_op")
        if op is None or stats.get("hlo_module", "").find("_run") < 0:
            continue
        if op.startswith(CONTAINERS):
            continue
        cat = FB.classify(src.get(op, ""), "")
        ms = dur / 1e6 / B.BLOCK
        cats[cat] = cats.get(cat, 0.0) + ms
        ops[(op, src.get(op, ""))] = ops.get((op, src.get(op, "")), 0.0) + ms
        lines[src.get(op, "?")] = lines.get(src.get(op, "?"), 0.0) + ms
        if op not in src:
            kind = re.sub(r"\.\d+$", "", op)
            unattributed[kind] = unattributed.get(kind, 0.0) + ms
        total += ms
    top = sorted(ops.items(), key=lambda t: -t[1])[:25]
    row = {
        "wall_ms_per_step": wall_s / B.BLOCK * 1e3,
        "leaf_ops_ms_per_step": total,
        "categories": dict(sorted(cats.items(), key=lambda t: -t[1])),
        "top_ops": [[o, s, ms] for (o, s), ms in top],
        "lines": dict(sorted(lines.items(), key=lambda t: -t[1])),
        "unattributed": dict(sorted(unattributed.items(), key=lambda t: -t[1])),
    }
    name = "forward_cpu_serial.json" if body == "polygon" and not tag else f"forward_cpu_serial_{body}{tag}.json"
    with open(os.path.join(HERE, "profile", name), "w") as fh:
        json.dump(row, fh, indent=1)
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
