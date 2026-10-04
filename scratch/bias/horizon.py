"""Horizon dependence of the squash adjoint: the same forward (same key, absolute step
indices) captured for the last M = 300 and M = 600 steps (window 200), all variants, plus
the CRN FD.  The paired difference is the part of dD/ds the 300-step horizon misses.

    PYTHONPATH=. python3 scratch/bias/horizon.py OUT.jsonl S SEED0 SEED1 '{overrides}'
"""

import functools
import json
import sys
import time

import jax

sys.path.insert(0, "examples/mesh")
import optimize_squash as OS  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward, shape  # noqa: E402

TOTAL = 1000


def main():
    path, s = sys.argv[1], float(sys.argv[2])
    seeds = range(int(sys.argv[3]), int(sys.argv[4]))
    over = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    hs = over.pop("fd_steps", [0.05])
    sk = over.pop("virtual", [1, 2])
    horizons = over.pop("horizons", [300, 600])
    tband = over.pop("timing_band", 10)
    base = dict(OS.default_config(), squash=s, **over)
    ensemble_backward.configure("virtual", sk[0], sk[1], "staged")
    variants = tuple(shape.VARIANTS)
    for i in seeds:
        key = jax.random.key(base["seed_start"] + i)
        row = {"s": s, "seed": i, "config": over, "virtual": sk, "horizons": {}}
        t0 = time.time()
        for m in horizons:
            cfg = dict(base, warmup=TOTAL - m, steps=m)
            terms, info, st = EM.shape_terms(cfg, key, None, "squash_y", 32, tband, variants)
            pick = functools.partial(EM.control_gradient, cfg, "squash_y", st["grid"])
            vol = 0.0 if st["volume"] is None else pick(st["volume"])
            row["horizons"][str(m)] = {
                "objective": info["objective"],
                "tally": pick(terms["one"].pathwise_tally),
                "volume": vol,
                "timing": pick(terms["one"].timing),
                "order": 0.0 if terms["one"].order is None else pick(terms["one"].order),
                "variants": {n: pick(t.pathwise) + pick(t.score) + pick(t.silhouette) + vol
                             + (0.0 if t.order is None else pick(t.order))
                             for n, t in terms.items()},
            }
            del terms, st
        row["fd"] = EM.finite_differences(dict(base, warmup=TOTAL - 300, steps=300), "squash_y",
                                          key, hs)
        row["seconds"] = time.time() - t0
        print(json.dumps(row), flush=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
