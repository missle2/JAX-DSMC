#!/usr/bin/env python3
"""Work done BEFORE the GPU allocation (login node), and the JIT-before-allocation gate.

    python3 bench/prep.py geometry --plans bench/plans/pm-gpu   # sphere geometry caches
    python3 bench/prep.py compile-gate --rows $RESULTS_DIR/rows.jsonl
    python3 bench/prep.py aot                                   # can this jaxlib compile for
                                                                 # CUDA without a GPU?

geometry      Every sphere size named in the plan files is built once into
              $SPARTA_JAX_GEO_CACHE (sparta_jax.geometry3d.build_cached), so no rank of a GPU job
              spends node time on host-side cut-cell tables (about a second at 1M particles,
              a minute at the largest sizes).
compile-gate  The share of the charged node time that sparta_jax spent compiling, from the
              rows (compile_s x nodes over the jobs' total).  Below ~3 % the persistent
              compilation cache (JAX_COMPILATION_CACHE_DIR, set by platforms.sh) is all the
              study needs and device-less AOT is not worth building; above, see `aot`.
aot           Lowers the 1-device chunk for the CUDA platform on this (GPU-less) host and tries
              to compile it against a GPU target description; reports what works.  The
              persistent cache key includes the device kind and mesh, so a cache entry can only
              be produced on (or for) the target GPU.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def cmd_geometry(a):
    from sparta_jax import decks as D
    from sparta_jax import geometry3d as G3
    sys.path.insert(0, HERE)
    from run_bench import parse_size as ps

    sizes = set()
    for f in glob.glob(os.path.join(a.plans, "*.plan")):
        for line in open(f):
            m = re.search(r"--deck sphere --particles (\S+)", line)
            if m:
                sizes.add(ps(m.group(1)))
    sizes |= {ps(s) for s in a.sizes}
    for n in sorted(sizes):
        x, y, z = D.size_xyz(n)
        d = D.sphere(x, y, z)
        t = time.perf_counter()
        G3.build_cached(d, a.cache)
        print(f"sphere {n:>12d} {x}x{y}x{z}: {time.perf_counter() - t:.1f} s -> {G3.cache_path(d, a.cache)}", flush=True)


def cmd_compile_gate(a):
    rows = [json.loads(ln) for ln in open(a.rows) if ln.strip()]
    jr = [r for r in rows if r["code"] == "jax" and r.get("rep", 0) == 0]
    if not jr:
        print("no sparta_jax rows yet")
        return
    comp = sum(r.get("compile_s", 0.0) * r.get("nodes", 1) for r in jr)
    total = sum(r.get("wall_s", 0.0) * r.get("nodes", 1) for r in rows if r["code"] == "jax")
    frac = comp / max(total, 1e-9)
    print(f"sparta_jax compile: {comp / 3600:.3f} node-h of {total / 3600:.3f} node-h of sparta_jax runs ({100 * frac:.1f} %)")
    worst = sorted(jr, key=lambda r: -r.get("compile_s", 0))[:5]
    for r in worst:
        print(f"  {r['deck']:8s} {r['particles']:>11d} nodes {r.get('nodes', 1):3d} compile {r.get('compile_s', 0):.1f} s")
    print("verdict:", "the persistent cache suffices" if frac < 0.03 else "worth pursuing device-less AOT (prep.py aot)")


def cmd_aot(a):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax

    import sparta_jax  # noqa: F401
    from sparta_jax import decks as D
    from sparta_jax import driver
    from sparta_jax import sim3d as S

    d = D.collide(20, 20, 20)
    geo, cfg, consts, state, _ = driver.setup(d)
    traced = S.run.trace(state, 0, d.runs[0].dt, 0, consts, cfg, d.stats_every)
    try:
        low = traced.lower(lowering_platforms=("cuda",))
        print(f"lowering for cuda on a CPU host: ok ({len(low.as_text()) // 1000} kB of StableHLO)")
    except Exception as e:  # noqa: BLE001
        print(f"lowering for cuda: FAILED ({type(e).__name__}: {e})")
        return
    try:
        from jax.experimental import topologies

        topo = topologies.get_topology_desc(platform="cuda", topology_name=a.topology)
        print(f"GPU topology description: {topo}")
        print("compile against it: try jax.jit(...).lower(..., _private_parameters) per the jax version; "
              "not wired in (the gate decides if it is needed)")
    except Exception as e:  # noqa: BLE001
        print(f"device-less GPU topology: unavailable in jax {jax.__version__} ({type(e).__name__}: {e})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("geometry")
    g.add_argument("--plans", default=os.path.join(HERE, "plans"))
    g.add_argument("--sizes", nargs="*", default=[])
    g.add_argument("--cache", default=os.environ.get("SPARTA_JAX_GEO_CACHE"))
    c = sub.add_parser("compile-gate")
    c.add_argument("--rows", required=True)
    t = sub.add_parser("aot")
    t.add_argument("--topology", default="a100")
    a = p.parse_args()
    if a.cmd == "geometry":
        if not a.cache:
            p.error("--cache or SPARTA_JAX_GEO_CACHE")
        cmd_geometry(a)
    elif a.cmd == "compile-gate":
        cmd_compile_gate(a)
    else:
        cmd_aot(a)


if __name__ == "__main__":
    main()
