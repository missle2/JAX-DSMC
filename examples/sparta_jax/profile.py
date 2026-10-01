"""Per-stage profile of sparta_jax (emit / move_dense / move_select / move_walk / sort / collide).

Fills the box for ``--warm`` steps, then traces ``--steps`` steps with jax.profiler.  Every executed
HLO op (CPU: host-tracer level 3 op events; GPU: device kernel events) is attributed to the
stage whose ``jax.named_scope`` appears in the op's ``op_name`` metadata in the optimised HLO dump.

    run_locked.sh cpu env XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_dump_to=/tmp/sj_hlo \
        --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*run.*" taskset -c 3 \
        python3 examples/sparta_jax/profile.py --tag cpu1
    run_locked.sh gpu env XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_dump_to=/tmp/sj_hlo_gpu \
        --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*run.*" \
        python3 examples/sparta_jax/profile.py --tag gpu --dump /tmp/sj_hlo_gpu
"""

import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import xspace  # noqa: E402

from sparta_jax import sim  # noqa: E402

STAGES = ("emit", "move_dense", "move_select", "move_walk", "sort", "collide", "tally")
CONTAINERS = ("while", "conditional", "call")


def op_names(dump):
    paths = sorted(glob.glob(os.path.join(dump, "*jit_run*after_optimizations.txt")))
    out = {}
    for path in paths:
        for line in open(path):
            m = re.match(r"\s*(?:ROOT )?%?([\w.\-]+) = .*?op_name=\"([^\"]+)\"", line)
            if m:
                out.setdefault(m.group(1), m.group(2))
    return out


def stage_of(op_name):
    best, pos = "other", -1
    for s in STAGES:
        k = op_name.rfind(s)
        if k > pos:
            best, pos = s, k
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sort", default="partition")
    ap.add_argument("--warm", type=int, default=800)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--dump", default="/tmp/sj_hlo")
    ap.add_argument("--tag", default="cpu")
    ap.add_argument("--set", nargs="*", default=[], help="Config overrides key=value (ints)")
    args = ap.parse_args()
    over = {k: (int(v) if v.isdigit() else v) for k, v in (kv.split("=") for kv in args.set)}
    cfg, consts, _geo = sim.make(sort=args.sort, **over)
    platform = jax.devices()[0].platform
    state = sim.empty_state(cfg)
    key = 7
    state, _ = sim.run(state, key, jnp.int32(0), consts, cfg, args.warm)
    state, _ = sim.run(state, key, jnp.int32(args.warm), consts, cfg, args.steps)  # compile
    jax.block_until_ready(state)
    out = os.path.join(HERE, "results", f"profile_{args.tag}")
    opts = jax.profiler.ProfileOptions()
    opts.host_tracer_level = 3
    opts.python_tracer_level = 0
    jax.profiler.start_trace(out, profiler_options=opts)
    t0 = time.perf_counter()
    state, stats = sim.run(state, key, jnp.int32(args.warm), consts, cfg, args.steps)
    jax.block_until_ready(state)
    wall = time.perf_counter() - t0
    jax.profiler.stop_trace()
    names = op_names(args.dump)
    path = sorted(glob.glob(os.path.join(out, "plugins", "profile", "*", "*.xplane.pb")))[-1]
    buf, planes = xspace.planes(path)
    want = "/device:GPU:0" if platform == "gpu" else "/host:CPU"
    span = next(s for n, s in planes if n == want)
    per, ops, n_ev, total = {}, {}, 0, 0.0
    for _line, ename, _start, dur, st in xspace.plane_events(buf, span, event_stats=True):
        if platform == "gpu":
            op = st.get("hlo_op") or ename
            mod = st.get("hlo_module", "run")
        else:
            op = st.get("hlo_op")
            mod = st.get("hlo_module", "")
            if op is None:
                continue
        if "run" not in mod or op.startswith(CONTAINERS):
            continue
        s = stage_of(names.get(op, ""))
        ms = dur / 1e6 / args.steps
        t, n = per.get(s, (0.0, 0))
        per[s] = (t + ms, n + 1)
        t_op, n_op = ops.get((s, op), (0.0, 0))
        ops[(s, op)] = (t_op + ms, n_op + 1)
        n_ev += 1
        total += ms
    row = {
        "platform": platform,
        "tag": args.tag,
        "sort": args.sort,
        "alive": int(stats.np[-1]),
        "wall_ms_per_step": wall / args.steps * 1e3,
        "op_ms_per_step": total,
        "ops_per_step": n_ev / args.steps,
        "stages": {
            s: {"ms_per_step": t, "ops_per_step": n / args.steps}
            for s, (t, n) in sorted(per.items(), key=lambda kv: -kv[1][0])
        },
        "top_ops": [
            [s, o, t] for (s, o), (t, _n) in sorted(ops.items(), key=lambda kv: -kv[1][0])[:30]
        ],
        "all_ops": [
            [s, o, n / args.steps, t]
            for (s, o), (t, n) in sorted(ops.items(), key=lambda kv: -kv[1][0])
        ],
    }
    with open(os.path.join(HERE, "results", f"profile_{args.tag}.json"), "w") as fh:
        json.dump(row, fh, indent=1)
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
