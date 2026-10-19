"""Where a seed's time goes in the optimiser's gradient (EM.shape_terms), on one optm iterate
with the run's own config: per phase (cut-cell build, warm-up, captured steps, replay, the two
backwards, the shape VJP), each setting run twice (the first compiles) and the second timed;
then the shape step's estimator share (the shipped along_pool_plane against the plain k32
own-cell fit) and the backward's guard cost (CELL_BASIS against the unguarded cubic).  The
last timed call of the shipped setting is also traced (jax.profiler, perfetto) into
``<out>.trace/``.

    PYTHONPATH=. python3 scratch/shapeprof/profile.py OUT.json RUN_DIR ITERATE [overrides]
"""

import dataclasses
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as OPT  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward  # noqa: E402

UNGUARDED = "cell:cubicv:adapt1.5:align0.9:both"
PHASES = ("build", "warmup", "capture", "replay", "backward", "shape")


def timed(cfg, ls, key, basis, variant, trace=None):
    """The phases of two calls (compile, then timed) of one setting."""
    ensemble_backward.configure("virtual", 4, 8, "staged", basis)
    out = {}
    for label in ("first", "second"):
        if trace and label == "second":
            jax.profiler.start_trace(trace, create_perfetto_trace=True)
        t0 = time.time()
        _terms, info, _s = EM.shape_terms(
            cfg, key, ls, "radius", cfg["neighbors"], cfg["timing_band"], (variant,)
        )
        total = time.time() - t0
        if trace and label == "second":
            jax.profiler.stop_trace()
        out[label] = {"total": total, **{p: info.get(f"seconds_{p}") for p in PHASES},
                      "hits": info["hits"]}  # fmt: skip
        print(basis, variant, label, json.dumps(out[label]), flush=True)
    return out


def main():
    path, run, it = sys.argv[1], sys.argv[2], int(sys.argv[3])
    over = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    with open(f"{run}/history.json") as fh:
        cfg = dict(OPT.default_config(), **json.load(fh)["config"])
    cfg.update(over)
    _grid, ls = OPT.initial_levelset(cfg)
    ls = dataclasses.replace(ls, coef=jnp.asarray(np.load(f"{run}/coef_{it}.npy")))
    key = jax.random.key(cfg["seed_start"])
    shipped = ensemble_backward.CELL_BASIS
    result = {"iterate": it, "config": cfg, "runs": {}}
    for name, basis, variant, trace in (
        ("shipped", shipped, cfg["variant"], path[: -len(".json")] + ".trace"),
        ("unguarded", UNGUARDED, cfg["variant"], None),
        ("plain_k32", shipped, "one", None),
    ):
        result["runs"][name] = {"basis": basis, "variant": variant,
                                **timed(cfg, ls, key, basis, variant, trace)}  # fmt: skip
        with open(path, "w") as fh:
            json.dump(result, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
