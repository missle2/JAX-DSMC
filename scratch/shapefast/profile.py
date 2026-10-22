"""GPU kernel profile of the backward and the shape step (local GPU, reduced size as
bench.py), 2026-10-22: one traced call of each after a compiling one; the trace's kernels
summed by HLO op and mapped to source lines through the optimised-HLO dump
(XLA_FLAGS=--xla_dump_to=scratch/shapefast/hlo, set by the caller).

    XLA_FLAGS="--xla_dump_to=scratch/shapefast/hlo --xla_dump_hlo_as_text" \
        PYTHONPATH=. python3 scratch/shapefast/profile.py
"""

import collections
import glob
import gzip
import json
import re
import sys

import jax

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/shapefast")
import optimize as OPT  # noqa: E402
from backward import setup  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward, shape  # noqa: E402

OUT = "scratch/shapefast/trace"


def source_map():
    """``{(module, op): "file:line op_name"}`` from every after-optimisation HLO dump."""
    out = {}
    pat = re.compile(r"^\s*(?:ROOT )?%?(\S+) = .*?metadata=\{([^}]*)\}")
    for path in glob.glob("scratch/shapefast/hlo/*after_optimizations.txt"):
        module = re.sub(r"^module_\d+\.", "", path.split("/")[-1]).split(".")[0]
        for line in open(path):
            m = pat.match(line)
            if not m:
                continue
            meta = dict(re.findall(r'(\w+)="?([^" ]*)"?', m.group(2)))
            where = f"{meta.get('source_file', '?').split('/')[-1]}:{meta.get('source_line', '?')}"
            out[(module, m.group(1))] = f"{where} {meta.get('op_name', '')[-80:]}"
    return out


def summarise(label):
    path = sorted(glob.glob(f"{OUT}/{label}/plugins/profile/*/*.trace.json.gz"))[-1]
    ev = json.load(gzip.open(path))["traceEvents"]
    gpu_pids = {e["pid"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"
                and "GPU" in e["args"]["name"]}  # fmt: skip
    kern = [e for e in ev if e.get("ph") == "X" and e["pid"] in gpu_pids]
    total = sum(e["dur"] for e in kern)
    by_op, by_line = collections.Counter(), collections.Counter()
    smap = source_map()
    for e in kern:
        a = e.get("args", {})
        mod, op = a.get("hlo_module", "?"), a.get("hlo_op", e["name"])
        by_op[(mod, op)] += e["dur"]
        src = smap.get((mod.split("(")[0], op), "?")
        by_line[src.split(" ")[0]] += e["dur"]
    print(f"== {label}: {len(kern)} kernels, {total / 1e6:.2f} s GPU busy", flush=True)
    for (mod, op), d in by_op.most_common(25):
        print(f"  {d / total:6.1%} {d / 1e6:7.3f}s {mod[:28]:28s} {op[:28]:28s} "
              f"{smap.get((mod.split('(')[0], op), '?')}")  # fmt: skip
    print("  -- by source line")
    for src, d in by_line.most_common(25):
        print(f"  {d / total:6.1%} {d / 1e6:7.3f}s {src}")


def main():
    cfg = dict(OPT.default_config(), case="circle_long", fnum_scale=16.0, levelset_h=0.0625,
               levelset_samples=16, grazing=0.05, boundary="free", knudsen=0.5, mach=10.0,
               dt=2.5e-5, a_max=256, warmup=2800, steps=80, window=50, smooth=8.0,
               variant="along_pool_plane", neighbors=32, timing_band=10.0)  # fmt: skip
    _grid, ls = OPT.initial_levelset(cfg)
    key = jax.random.key(32)
    ensemble_backward.configure("virtual", 4, 8, "staged", ensemble_backward.CELL_BASIS)
    args, extra = setup(cfg, ls, key, full=True)
    dev, dev_t = EM._backwards(*args)
    jax.block_until_ready(dev.lam)
    jax.profiler.start_trace(f"{OUT}/backward")
    dev, dev_t = EM._backwards(*args)
    jax.block_until_ready(dev.lam)
    jax.profiler.stop_trace()
    kwargs = extra["shape_kwargs"](dev, dev_t)
    call = (dev_t.lam,)
    opts = {"names": (cfg["variant"],), "collision": dev.collision, **kwargs}
    jax.block_until_ready(shape.gradient_variants(*call, **opts))
    jax.profiler.start_trace(f"{OUT}/shape")
    jax.block_until_ready(shape.gradient_variants(*call, **opts))
    jax.profiler.stop_trace()
    summarise("backward")
    summarise("shape")


if __name__ == "__main__":
    main()
