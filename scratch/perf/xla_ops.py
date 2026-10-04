"""Executed-op census of the serial-CPU forward (SPARTA circle, polygon or level set): every
HLO op the CPU runtime executes in 100 steady-state steps, counted and timed per step and
attributed to the enclosing ``jax_dsmc`` function of its source line, plus the op-size
histogram -- how much of the step is per-op overhead of many small ops rather than work.

    rm -rf /tmp/hlo_cpu
    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false \
        --xla_dump_to=/tmp/hlo_cpu \
        --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*_run.*" PYTHONPATH=. \
        taskset -c 3 python3 scratch/perf/xla_ops.py polygon TAG

-> scratch/perf/profile/xla_ops_<body><tag>.json (the trace folder is git-ignored).
"""

import ast
import glob
import json
import os
import re
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "examples", "mesh"))
import bench_forward as B  # noqa: E402
import xspace  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import initialize  # noqa: E402
from jax_dsmc.state import FlowParams  # noqa: E402

CONTAINERS = ("while", "conditional", "call", "fusion_region")
BUCKETS_US = (1.0, 2.0, 4.0, 8.0, 16.0, 64.0, np.inf)
_SPANS = {}


def _spans(path):
    if path not in _SPANS:
        out = []
        if os.path.exists(path):
            for node in ast.walk(ast.parse(open(path).read())):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append((node.lineno, node.end_lineno, node.name))
        _SPANS[path] = out
    return _SPANS[path]


def function_of(path, line):
    """``file.py:function`` of the innermost function of ``path`` holding ``line``."""
    inside = [(hi - lo, f) for lo, hi, f in _spans(path) if lo <= line <= hi]
    return f"{os.path.basename(path)}:{min(inside)[1] if inside else '<module>'}"


def op_table(dump):
    """{op name: (opcode, 'file.py:function')} over the optimised CPU module.  An op XLA
    created without source metadata (copies, the reduce-windows of a ``cumsum``, broadcasts)
    is attributed to the nearest op with metadata among its operands, then its users, marked
    ``~``."""
    path = glob.glob(os.path.join(dump, "*jit__run*after_optimizations.txt"))[0]
    pat = re.compile(r"\s*(?:ROOT )?%?([\w.\-]+) = \S+ (?:\([^=]*?\) )?([\w\-]+)\((.*)")
    ops, users = {}, {}
    for line in open(path):
        m = pat.match(line)
        if not m or m.group(1) in ops:
            continue
        name, opcode = m.group(1), m.group(2)
        operands = re.findall(r"%([\w.\-]+)", m.group(3).split("), ")[0])
        src = re.search(r'source_file="([^"]+)" source_line=(\d+)', line)
        where = function_of(src.group(1), int(src.group(2))) if src else ""
        ops[name] = (opcode, where, operands)
        for o in operands:
            users.setdefault(o, []).append(name)

    def nearest(name, links):
        seen, frontier = {name}, [name]
        for _ in range(8):
            nxt = []
            for n in frontier:
                for o in links(n):
                    if o in seen or o not in ops:
                        continue
                    if ops[o][1]:
                        return ops[o][1]
                    seen.add(o)
                    nxt.append(o)
            frontier = nxt
        return ""

    out = {}
    for name, (opcode, where, _operands) in ops.items():
        if not where:
            found = nearest(name, lambda n: ops[n][2]) or nearest(name, lambda n: users.get(n, []))
            where = f"~{found}" if found else ""
        out[name] = (opcode, where)
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
    out = os.path.join(HERE, "profile", f"xla_ops_{body}{tag}")
    result = os.path.join(HERE, "profile", f"xla_ops_{body}{tag}.json")
    steps = B.BLOCK  # one _run call
    if os.environ.get("REUSE"):  # re-attribute the last trace (same HLO dump)
        old = json.load(open(result))
        wall_s = old["wall_ms_per_step"] * steps / 1e3
        untraced_s = old.get("untraced_ms_per_step", float("nan")) * steps / 1e3
    else:
        for k0 in range(0, 800, B.BLOCK):
            (state, cstate), _ = EM._run(state, cstate, *common, k0, False)
        jax.block_until_ready(state)
        t0 = time.perf_counter()  # the same block untraced: the tracer's own per-op cost
        jax.block_until_ready(EM._run(state, cstate, *common, 800, False))
        untraced_s = time.perf_counter() - t0
        opts = jax.profiler.ProfileOptions()
        opts.host_tracer_level = 3
        opts.python_tracer_level = 0
        jax.profiler.start_trace(out, profiler_options=opts)
        t0 = time.perf_counter()
        (state, cstate), _ = EM._run(state, cstate, *common, 800, False)
        jax.block_until_ready(state)
        wall_s = time.perf_counter() - t0
        jax.profiler.stop_trace()
    table = op_table(os.environ.get("HLO_DUMP", "/tmp/hlo_cpu"))
    traces = glob.glob(os.path.join(out, "plugins", "profile", "*", "*.xplane.pb"))
    buf, planes = xspace.planes(max(traces, key=os.path.getmtime))
    host = next(s for n, s in planes if n == "/host:CPU")
    durs, rows = [], {}
    for _line, _name, _start, dur, stats in xspace.plane_events(buf, host, event_stats=True):
        op = stats.get("hlo_op")
        if op is None or "_run" not in stats.get("hlo_module", "") or op.startswith(CONTAINERS):
            continue
        us = dur / 1e3  # ns -> us
        opcode, where = table.get(op, (re.sub(r"\.\d+$", "", op), ""))
        kind = re.sub(r"[.\d]+$", "", op) if opcode == "fusion" else opcode
        durs.append(us)
        merged = where.lstrip("~") or "?"
        for key_ in (("function", merged), ("opcode", kind), ("pair", f"{merged} | {kind}")):
            r = rows.setdefault(key_, [0, 0.0])
            r[0] += 1
            r[1] += us
    durs = np.array(durs)
    hist, lo = {}, 0.0
    for hi in BUCKETS_US:
        sel = (durs >= lo) & (durs < hi)
        hist[f"{lo:g}-{hi:g} us"] = {
            "ops_per_step": int(sel.sum()) / steps,
            "ms_per_step": float(durs[sel].sum()) / 1e3 / steps,
        }
        lo = hi

    def group(kind):
        sel = [(k[1], v) for k, v in rows.items() if k[0] == kind]
        return {
            name: {
                "ops_per_step": n / steps,
                "ms_per_step": us / 1e3 / steps,
                "us_per_op": us / max(n, 1),
            }
            for name, (n, us) in sorted(sel, key=lambda t: -t[1][1])
        }

    row = {
        "wall_ms_per_step": wall_s / steps * 1e3,
        "untraced_ms_per_step": untraced_s / steps * 1e3,
        "ops_per_step": len(durs) / steps,
        "op_ms_per_step": float(durs.sum()) / 1e3 / steps,
        "median_op_us": float(np.median(durs)),
        "size_histogram": hist,
        "functions": group("function"),
        "opcodes": group("opcode"),
        "function_opcodes": group("pair"),
    }
    with open(result, "w") as fh:
        json.dump(row, fh, indent=1)
    print(json.dumps({k: row[k] for k in list(row)[:5]}, indent=1))
    print(json.dumps(hist, indent=1))
    for name, r in list(row["functions"].items())[:45]:
        print(f"{r['ms_per_step']:7.3f} ms {r['ops_per_step']:7.1f} ops {r['us_per_op']:6.1f} us  {name}")
    print("--- opcodes")
    for name, r in list(row["opcodes"].items())[:30]:
        print(f"{r['ms_per_step']:7.3f} ms {r['ops_per_step']:7.1f} ops {r['us_per_op']:6.1f} us  {name}")


if __name__ == "__main__":
    main()
