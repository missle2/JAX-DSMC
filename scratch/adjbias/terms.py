"""Which term carries the translation bias (2026-10-24)?  The level-set circle in a
free-stream box at the production estimator settings; per seed, every shape-gradient term
(shape.ShapeTerms + the cut-volume term) projected on the spline's exact rigid x-translation
(``-d_x phi`` at the knots, interpolated), as % of the seed's J per metre.  The true value is 0
in free-molecular flow (a body in a uniform stream) and ~0 at Kn 0.5 (boxcheck's paired FD).
(examples.ensemble_mesh --parameter centre_x is NaN here: the circle's centre sits on a knot,
where the analytic distance's centre derivative is 0/0.)

    PYTHONPATH=. python3 scratch/adjbias/terms.py NAME [--seeds N] [ensemble_mesh flags]
        -> scratch/adjbias/cluster/NAME.json (the cluster group adjbias)
"""

import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward  # noqa: E402

TERMS = ("pathwise_x", "pathwise_v", "pathwise_tally", "score", "silhouette", "timing", "order")


def translation(ls):
    x = O.L.knots(ls)
    _phi, grad = O.L.value_and_grad(ls, x)
    t = O.L.interpolate((-grad[:, 0]).reshape(ls.coef.shape))
    return np.asarray(t) * np.asarray(O.L.coefficient_band(ls, 1.0))


def main():
    name, argv = sys.argv[1], sys.argv[2:]
    cfg = dict(EM.default_config(), body="levelset")
    flags = dict(zip(argv[::2], argv[1::2], strict=True))
    seeds = int(flags.pop("--seeds", 8))
    for k, v in flags.items():
        key = k.lstrip("-").replace("-", "_")
        cfg[key] = v in ("1", "true", "True") if isinstance(cfg[key], bool) else type(cfg[key])(v)
    path = Path(f"scratch/adjbias/cluster/{name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_backward.configure("virtual", 4, 8, "staged")
    grid, ls = O.initial_levelset(cfg)
    tx = translation(ls)
    rows = []
    for i in range(seeds):
        t0 = time.time()
        all_terms, info, s = EM.shape_terms(cfg, jax.random.key(32 + i), ls, "radius", 32, 10.0,
                                            ("along_pool_plane",))  # fmt: skip
        terms = all_terms["along_pool_plane"]
        j = info["objective"]
        row = {"J": j, "hits": info["hits"]}
        for t in TERMS:
            v = getattr(terms, t)
            row[t] = 0.0 if v is None else float(jnp.sum(v * tx)) / j * 100
        row["volume"] = 0.0 if s["volume"] is None else float(jnp.sum(s["volume"] * tx)) / j * 100
        row["total"] = sum(row[t] for t in (*TERMS, "volume"))
        row["seconds"] = time.time() - t0
        rows.append(row)
        print(i, {k: round(v, 3) for k, v in row.items()}, flush=True)
        path.write_text(json.dumps({"config": cfg, "rows": rows}))
    print("mean +- se, % J / m:")
    for k in (*TERMS, "volume", "total"):
        v = np.array([r[k] for r in rows])
        print(f"  {k:15s} {v.mean():+8.3f} +- {v.std(ddof=1) / np.sqrt(len(v)):.3f}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # the interpreter's teardown hangs some cluster jobs (optimize.py)
