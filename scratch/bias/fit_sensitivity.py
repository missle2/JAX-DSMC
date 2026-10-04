"""Kn 0.5 squash: sensitivity of the adjoint's grad-lambda parts to the virtual pull-back's
(S, K) and to the fit's neighbour count, on the same forward (paired).  The tally, timing
and volume terms barely depend on them; a fitted part that moves with the settings is
fit- or quadrature-limited.

    PYTHONPATH=. python3 scratch/bias/fit_sensitivity.py OUT.jsonl S SEED0 SEED1 '{overrides}'
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

shape.VARIANTS.update({
    "one_k16": (0, 16, False),
    "one_pool_plane_k48": (1, 48, False, 0.25),
    "one_pool_plane_k192": (1, 192, False, 0.25),
})
NAMES = ("one", "one_k16", "one_pool_plane", "one_pool_plane_k48", "one_pool_plane_k192",
         "along_pool_plane")


def main():
    path, s = sys.argv[1], float(sys.argv[2])
    over = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    settings = over.pop("virtual", [[2, 4], [4, 8]])
    cfg = dict(OS.default_config(), squash=s, **over)
    for i in range(int(sys.argv[3]), int(sys.argv[4])):
        key = jax.random.key(cfg["seed_start"] + i)
        row = {"s": s, "seed": i, "config": over, "runs": {}}
        t0 = time.time()
        for sk in settings:
            ensemble_backward.configure("virtual", sk[0], sk[1], "staged")
            terms, info, st = EM.shape_terms(cfg, key, None, "squash_y", 32, 10, NAMES)
            pick = functools.partial(EM.control_gradient, cfg, "squash_y", st["grid"])
            vol = 0.0 if st["volume"] is None else pick(st["volume"])
            one = terms["one"]
            row["runs"][f"{sk[0]}x{sk[1]}"] = {
                "tally": pick(one.pathwise_tally), "timing": pick(one.timing), "volume": vol,
                "variants": {n: {"x": pick(t.pathwise_x), "v": pick(t.pathwise_v)}
                             for n, t in terms.items()},
            }
            del terms, st
        row["seconds"] = time.time() - t0
        print(json.dumps(row), flush=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
