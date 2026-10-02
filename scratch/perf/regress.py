"""Regression gate for the speed / memory changes: the full per-seed shape chain of the
optimiser (examples.ensemble_mesh.shape_terms) on two seeds in one process, every term of
two variants saved, compared against a stored reference.

    # once, on the unchanged code
    ... PYTHONPATH=. python3 scratch/perf/regress.py --save base
    # after a change
    ... PYTHONPATH=. python3 scratch/perf/regress.py --compare base [--save after_x]

Configuration: the optc settings (level-set circle, h_phi 0.0625, 16 samples, grazing
0.05, lane tracker, virtual pull-back 1 x 2, collisions on) at fnum x 4 (15 k slots) so
the chain fits a 4 GB GPU.  Variants 'one' (unpooled fit) and 'along_pool' (the
optimiser's).  Reported per term: max |a - b| / max |b| over the coefficient grid, and
whether the arrays are bitwise equal; per seed the stage timings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "examples", "mesh"))
import optimize as O  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward  # noqa: E402

DIR = os.path.join(ROOT, "scratch", "perf", "regress")
VARIANTS = ("one", "along_pool")
FIELDS = ("pathwise", "pathwise_x", "pathwise_v", "pathwise_tally", "score", "timing", "silhouette")
SEEDS = (0, 1)


def _array_mass_rewards(mass, dim, first, last, scale):
    """The pre-fix-1 construction: ``mass`` captured as a device array, not a Python float."""
    import functools  # noqa: PLC0415

    import jax.numpy as jnp  # noqa: PLC0415

    m = jnp.asarray(mass)
    return (
        functools.partial(EM._drag_reward, mass=m, dim=dim, first=first, scale=scale),
        functools.partial(EM._drag_jump, mass=m, dim=dim, first=first, last=last, scale=scale),
    )


def run(record="lean", npc_bucket=None, array_mass=False):
    ensemble_backward.configure("virtual", 1, 2, "staged")
    if npc_bucket is not None:
        EM.NPC_BUCKET = npc_bucket
    if array_mass:
        EM.drag_rewards = _array_mass_rewards
    cfg = dict(
        O.default_config(),
        fnum_scale=4.0,
        levelset_h=0.0625,
        levelset_samples=16,
        grazing=0.05,
        record=record,
    )
    _grid, ls = O.initial_levelset(cfg)
    arrays, meta = {}, []
    for seed in SEEDS:
        t0 = time.time()
        terms, info, _s = EM.shape_terms(
            cfg, jax.random.key(32 + seed), ls, "radius", 32, 0.2, VARIANTS
        )
        total = time.time() - t0
        for v in VARIANTS:
            for f in FIELDS:
                arrays[f"s{seed}_{v}_{f}"] = np.asarray(getattr(terms[v], f))
        meta.append(
            {
                "seed": seed,
                "objective": info["objective"],
                "hits": info["hits"],
                "total": total,
                "backward": info["seconds_backward"],
                "shape": info["seconds_shape"],
                "forward_replay": total - info["seconds_backward"] - info["seconds_shape"],
            }
        )
        print(json.dumps(meta[-1]), flush=True)
    return arrays, meta


def compare(arrays, meta, name):
    ref = np.load(os.path.join(DIR, f"{name}.npz"))
    ref_meta = json.load(open(os.path.join(DIR, f"{name}.json")))
    worst = 0.0
    for key in sorted(arrays):
        a, b = arrays[key], ref[key]
        scale = float(np.max(np.abs(b)))
        rel = float(np.max(np.abs(a - b))) / scale if scale > 0 else float(np.max(np.abs(a)))
        bitwise = bool(np.array_equal(a, b))
        worst = max(worst, rel)
        print(f"{key:40s} rel {rel:.3e}  {'bitwise' if bitwise else ''}")
    for m, r in zip(meta, ref_meta, strict=True):
        print(
            f"seed {m['seed']}: objective {m['objective']:.10e} vs {r['objective']:.10e}  "
            f"total {m['total']:.0f}s vs {r['total']:.0f}s  (fwd+replay {m['forward_replay']:.0f} "
            f"vs {r['forward_replay']:.0f}, backward {m['backward']:.0f} vs {r['backward']:.0f}, "
            f"shape {m['shape']:.0f} vs {r['shape']:.0f})"
        )
    print(f"WORST relative difference over all terms: {worst:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--record", default="lean", choices=("lean", "full"))
    ap.add_argument("--npc-bucket", type=int, default=None)
    ap.add_argument("--array-mass", action="store_true")
    args = ap.parse_args()
    os.makedirs(DIR, exist_ok=True)
    arrays, meta = run(args.record, args.npc_bucket, args.array_mass)
    if args.compare:
        compare(arrays, meta, args.compare)
    if args.save:
        np.savez(os.path.join(DIR, f"{args.save}.npz"), **arrays)
        with open(os.path.join(DIR, f"{args.save}.json"), "w") as fh:
            json.dump(meta, fh, indent=1)


if __name__ == "__main__":
    main()
