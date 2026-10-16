"""Untraced per-stage wall times of one optimiser seed (examples.ensemble_mesh.shape_terms), the
regress.py configuration (optc settings at fnum x 4, 15 k slots): forward, traced replay, the
value backwards, the shape gradient's variants and the cut-volume term (2026-10, outside
profile_seed.py's hooks), each blocked on.  One warm-up seed (key 40), then the timed seeds.

    run_locked.sh gpu env PYTHONPATH=. python3 scratch/perf/stage_times.py --tag cellfit \
        [--variants one along_pool]

Writes scratch/perf/regress/stages_<tag>.json.
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

STAGES: dict[str, float] = {}


def _timed(name, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    jax.block_until_ready(out)
    STAGES[name] = STAGES.get(name, 0.0) + time.perf_counter() - t0
    return out


def wrap(module, attr, name):
    if hasattr(module, attr):  # _volume_term is absent before 2026-10
        setattr(module, attr, functools.partial(_timed, name, getattr(module, attr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variants", nargs="+", default=["one", "along_pool"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[32, 33])
    args = ap.parse_args()
    ensemble_backward.configure("virtual", 1, 2, "staged")
    cfg = dict(
        O.default_config(), fnum_scale=4.0, levelset_h=0.0625, levelset_samples=16, grazing=0.05
    )
    _grid, ls = O.initial_levelset(cfg)
    wrap(EM, "forward", "forward")
    wrap(shape, "replay", "replay")
    wrap(EM, "_backwards", "backward")
    wrap(shape, "gradient_variants", "gradient_variants")
    wrap(EM, "_volume_term", "volume_term")
    variants = tuple(args.variants)
    EM.shape_terms(cfg, jax.random.key(40), ls, "radius", 32, 0.2, variants)  # warm-up
    rows = []
    for seed in args.seeds:
        STAGES.clear()
        t0 = time.perf_counter()
        _terms, info, _s = EM.shape_terms(
            cfg, jax.random.key(seed), ls, "radius", 32, 0.2, variants
        )
        total = time.perf_counter() - t0
        row = {
            "seed": seed,
            "device": jax.devices()[0].device_kind,
            "variants": list(variants),
            "total": total,
            **{k: float(v) for k, v in STAGES.items()},
            "host_other": total - sum(STAGES.values()),
            "hits": info["hits"],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(
        os.path.join(ROOT, "scratch", "perf", "regress", f"stages_{args.tag}.json"), "w"
    ) as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
