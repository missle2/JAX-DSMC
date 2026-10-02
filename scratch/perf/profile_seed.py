"""JAX profile of one optimiser seed (examples.ensemble_mesh.shape_terms, the optc settings
at fnum x 4 as scratch/perf/regress.py, variant along_pool) after a warm-up seed, so the
trace holds no compilation.  Host stages are TraceAnnotations around the chain's pieces
(forward, replay, the two value backwards, the shape gradient); the device side is the
XLA modules and kernels.  Writes the trace to scratch/perf/profile/<tag>/ and the stage
wall times to scratch/perf/profile/<tag>/stages.json (scratch/perf/profile_report.py reads
both).

    XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false \
        PYTHONPATH=. python3 scratch/perf/profile_seed.py --tag gpu_after
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time

import jax

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "examples", "mesh"))
import optimize as O  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward, shape  # noqa: E402

STAGES = {}


TRACE = {"dir": None}
SLUG = {
    "forward (1000 steps, captured 300)": "forward",
    "traced replay": "replay",
    "value backwards (x2)": "backward",
    "shape gradient": "shape",
}


def _timed(name, fn, *args, **kwargs):
    """One stage, timed, and -- in the profiled seed -- traced on its own (one trace per
    stage keeps each under the GPU tracer's ~2 M event cap)."""
    if TRACE["dir"] is not None:
        options = jax.profiler.ProfileOptions()
        options.python_tracer_level = 0
        options.host_tracer_level = 2
        jax.profiler.start_trace(os.path.join(TRACE["dir"], SLUG[name]), profiler_options=options)
    with jax.profiler.TraceAnnotation(f"stage:{name}"):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
        STAGES[name] = STAGES.get(name, 0.0) + time.perf_counter() - t0
    if TRACE["dir"] is not None:
        jax.profiler.stop_trace()
    return out


def wrap(module, attr, name):
    original = getattr(module, attr)
    setattr(module, attr, functools.partial(_timed, name, original))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    ensemble_backward.configure("virtual", 1, 2, "staged")
    cfg = dict(
        O.default_config(), fnum_scale=4.0, levelset_h=0.0625, levelset_samples=16, grazing=0.05
    )
    _grid, ls = O.initial_levelset(cfg)
    wrap(EM, "forward", "forward (1000 steps, captured 300)")
    wrap(shape, "replay", "traced replay")
    wrap(EM, "_backwards", "value backwards (x2)")
    wrap(shape, "gradient_variants", "shape gradient")
    variants = ("along_pool",)
    EM.shape_terms(cfg, jax.random.key(40), ls, "radius", 32, 0.2, variants)  # warm-up
    STAGES.clear()
    out = os.path.join(ROOT, "scratch", "perf", "profile", args.tag)
    os.makedirs(out, exist_ok=True)
    TRACE["dir"] = out
    t0 = time.perf_counter()
    _terms, info, _s = EM.shape_terms(cfg, jax.random.key(32), ls, "radius", 32, 0.2, variants)
    total = time.perf_counter() - t0
    TRACE["dir"] = None
    row = {
        "device": jax.devices()[0].device_kind,
        "total_incl_trace_export": total,
        "stages": STAGES,
        "hits": info["hits"],
        "seconds_backward": info["seconds_backward"],
        "seconds_shape": info["seconds_shape"],
    }
    with open(os.path.join(out, "stages.json"), "w") as fh:
        json.dump(row, fh, indent=1)
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
