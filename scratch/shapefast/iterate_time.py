"""Seconds per phase of one optimiser seed on a production iterate, under the XLA flags of
the calling process and ``SHAPE_BATCH`` (the shape VJP's batched steps; default the
config's) -- the cluster A/B of scratch/shapefast/cluster_job.sh,
2026-10-22: the run's own config and level set at ``ITERATE``, three calls -- cold (compiles),
then two warm-started from the previous call's gas as the optimiser runs them (``restart``
500 steps) -- and the last call's summed gradient saved beside the timings so the flag
settings' gradients can be compared (command buffers change no arithmetic).

    PYTHONPATH=. [SHAPE_BATCH=b] python3 scratch/shapefast/iterate_time.py OUT.json RUN_DIR ITERATE
"""

import dataclasses
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as OPT  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import ensemble_backward  # noqa: E402

PHASES = ("build", "warmup", "capture", "replay", "backward", "shape")


def main():
    out, run, it = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(f"{run}/history.json") as fh:
        cfg = dict(OPT.default_config(), **json.load(fh)["config"])
    if os.environ.get("SHAPE_BATCH"):
        cfg["shape_batch"] = int(os.environ["SHAPE_BATCH"])
    _grid, ls = OPT.initial_levelset(cfg)
    ls = dataclasses.replace(ls, coef=jnp.asarray(np.load(f"{run}/coef_{it}.npy")))
    key = jax.random.key(cfg["seed_start"])
    ensemble_backward.configure("virtual", 4, 8, "staged", ensemble_backward.CELL_BASIS)
    result = {"xla_flags": os.environ.get("XLA_FLAGS", ""), "run": run, "iterate": it,
              "device": str(jax.devices()[0].device_kind),
              "shape_batch": cfg.get("shape_batch", 1), "calls": []}  # fmt: skip
    final = None
    for label in ("cold", "warm1", "warm2"):
        run_cfg = cfg if final is None else dict(cfg, restart_steps=500)
        t0 = time.time()
        terms, info, s = EM.shape_terms(
            run_cfg, key, ls, "radius", cfg["neighbors"], cfg["timing_band"],
            (cfg["variant"],), 0, final,
        )  # fmt: skip
        total = time.time() - t0
        final = s["final"]
        row = {"call": label, "total": total, **{p: info.get(f"seconds_{p}") for p in PHASES},
               "replay_sizes": info.get("replay_sizes"),
               "peak_gb": jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**30}  # fmt: skip
        result["calls"].append(row)
        print(json.dumps(row), flush=True)
    t = terms[cfg["variant"]]
    g = t.pathwise + t.score + t.silhouette + t.timing
    if t.order is not None:
        g = g + t.order
    if s["volume"] is not None:
        g = g + s["volume"]
    np.save(out[: -len(".json")] + "_grad.npy", np.asarray(jax.tree.leaves(g)[0]))
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
