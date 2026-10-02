"""M1 (scratch/perf/*.md): per-stage wall time of the optimiser's per-seed chain and the
XLA compiles each seed triggers, three seeds in one process.

    XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false \
        JAX_LOG_COMPILES=1 PYTHONPATH=. python3 scratch/perf/probe_stages.py 2> compiles.log

The optc configuration (level-set circle, h_phi 0.0625, 16 samples, grazing 0.05, lane
tracker, virtual pull-back 1 x 2, along_pool only) at fnum x 4 (15 k slots) so the chain
fits a 4 GB GPU.  Stages: forward + replay (= total - backward - shape), the two value
backwards, the shape gradient.  A marker line on stderr separates the seeds so the compile
log can be split per seed.
"""

from __future__ import annotations

import json
import os
import sys
import time

import jax

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "examples", "mesh"))
import optimize as O  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward  # noqa: E402


def main():
    ensemble_backward.configure("virtual", 1, 2, "staged")
    cfg = dict(
        O.default_config(),
        fnum_scale=4.0,
        levelset_h=0.0625,
        levelset_samples=16,
        grazing=0.05,
    )
    _grid, ls = O.initial_levelset(cfg)
    rows = []
    for seed in range(3):
        print(f"=== SEED {seed} START {time.time():.1f}", file=sys.stderr, flush=True)
        t0 = time.time()
        _terms, info, _s = EM.shape_terms(
            cfg, jax.random.key(32 + seed), ls, "radius", 32, 0.2, ("along_pool",)
        )
        total = time.time() - t0
        row = {
            "seed": seed,
            "total": total,
            "backward": info["seconds_backward"],
            "shape": info["seconds_shape"],
            "forward_replay": total - info["seconds_backward"] - info["seconds_shape"],
            "hits": info["hits"],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        print(f"=== SEED {seed} END {time.time():.1f}", file=sys.stderr, flush=True)
    with open(os.path.join(ROOT, "scratch", "perf", "probe_stages.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
