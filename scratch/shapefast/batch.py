"""Batched shape step (shape._gradient(batch=)), local GPU, reduced size as bench.py,
2026-10-22: the shape step at batch 1 (the sequential scan) and the batches in argv[1]
(default 4,10,20), timed after compiling; the gradient's relative difference to batch 1.

    PYTHONPATH=. python3 scratch/shapefast/batch.py [batches] [static-overrides-json] [--mem]
        [--full] [--trace]   (--mem: compiled temp memory only; --full: the forward's tracker
        sizes; --trace: kernel count and GPU-busy time of one more call)
"""

import glob
import gzip
import json
import sys
import time

import jax
import numpy as np

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/shapefast")
import optimize as OPT  # noqa: E402
from backward import setup  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward, shape  # noqa: E402


def total(terms):
    parts = (terms.pathwise, terms.score, terms.silhouette, terms.timing, terms.order)
    return sum(np.asarray(jax.tree.leaves(a)[0]) for a in parts if a is not None)


def busy(label):
    """Kernel count and GPU-busy seconds of the trace written under ``label``."""
    path = sorted(glob.glob(f"scratch/shapefast/trace/{label}/plugins/profile/*/*.trace.json.gz"))
    ev = json.load(gzip.open(path[-1]))["traceEvents"]
    gpu = {e["pid"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"
           and "GPU" in e["args"]["name"]}  # fmt: skip
    kern = [e for e in ev if e.get("ph") == "X" and e["pid"] in gpu]
    return len(kern), sum(e["dur"] for e in kern) / 1e6


def main():
    batches = [int(b) for b in (sys.argv[1] if len(sys.argv) > 1 else "4,10,20").split(",")]
    small = json.loads(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2][0] == "{" else None
    cfg = dict(OPT.default_config(), case="circle_long", fnum_scale=16.0, levelset_h=0.0625,
               levelset_samples=16, grazing=0.05, boundary="free", knudsen=0.5, mach=10.0,
               dt=2.5e-5, a_max=256, warmup=2800, steps=80, window=50, smooth=8.0,
               variant="along_pool_plane", neighbors=32, timing_band=10.0)  # fmt: skip
    _grid, ls = OPT.initial_levelset(cfg)
    key = jax.random.key(32)
    ensemble_backward.configure("virtual", 4, 8, "staged", ensemble_backward.CELL_BASIS)
    if "--full" in sys.argv:  # the forward's tracker sizes (no replay_static cut)
        EM.replay_static = lambda static, walls, floor=(0, 0): static
    args, extra = setup(cfg, ls, key, full=True, static_over=small)
    dev, dev_t = EM._backwards(*args)
    kwargs = extra["shape_kwargs"](dev, dev_t)
    print("sizes", kwargs["static"].K_bounce, kwargs["static"].lane_size, flush=True)
    if "--mem" in sys.argv:
        estimators = (shape.VARIANTS[cfg["variant"]],)
        kw = {k: v for k, v in kwargs.items() if k not in ("pool", "k", "lam_timing")}
        for b in [1, *batches]:
            lowered = shape._gradient.lower(**kw, estimators=estimators, lam_timing=dev_t.lam,
                                            collision=dev.collision, batch=b)  # fmt: skip
            mem = lowered.compile().memory_analysis()
            print(
                f"batch {b:3d} temp {mem.temp_size_in_bytes / 2**30:6.2f} GiB  "
                f"args {mem.argument_size_in_bytes / 2**30:6.2f} GiB",
                flush=True,
            )
        return
    ref = None
    for b in [1, *batches]:
        for label in ("compile", "timed", "timed"):
            t0 = time.time()
            terms = shape.gradient_variants(dev_t.lam, names=(cfg["variant"],),
                                            collision=dev.collision, batch=b, **kwargs)  # fmt: skip
            jax.block_until_ready(terms)
            print(f"batch {b:3d} {label:8s} {time.time() - t0:7.2f} s", flush=True)
        if "--trace" in sys.argv:
            jax.profiler.start_trace(f"scratch/shapefast/trace/batch{b}")
            jax.block_until_ready(shape.gradient_variants(dev_t.lam, names=(cfg["variant"],),
                                  collision=dev.collision, batch=b, **kwargs))  # fmt: skip
            jax.profiler.stop_trace()
            n, gpu_s = busy(f"batch{b}")
            print(f"  traced: {n} kernels, {gpu_s:.2f} s GPU busy", flush=True)
        t = terms[cfg["variant"]]
        g = total(t)
        ref = g if ref is None else ref
        rel = np.linalg.norm(g - ref) / np.linalg.norm(ref)
        print(f"  replay error {float(t.replay_error):.3g}  |g| {np.linalg.norm(g):.6g}  "
              f"rel {rel:.3g}  hits {int(np.sum(t.surface_hits))}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
