"""Where the shape step's time goes, by slope-fit variant (local GPU, reduced particles):
plain own-cell k32 ("one"), pooled 3x3 k96 ("one_pool"), pooled with the half-space plane
("one_pool_plane") and the shipped along-flight form ("along_pool_plane").  Each variant run
twice (compile, then timed) on one seed of an optm iterate.

    PYTHONPATH=. python3 scratch/shapeprof/prof_shape.py RUN_DIR ITERATE [overrides]
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

run, it = sys.argv[1], int(sys.argv[2])
over = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
with open(f"{run}/history.json") as fh:
    cfg = dict(OPT.default_config(), **json.load(fh)["config"])
cfg.update(over)
_grid, ls = OPT.initial_levelset(cfg)
ls = dataclasses.replace(ls, coef=jnp.asarray(np.load(f"{run}/coef_{it}.npy")))
key = jax.random.key(cfg["seed_start"])
from jax_dsmc import ensemble_backward  # noqa: E402

ensemble_backward.configure("virtual", 4, 8, "staged", ensemble_backward.CELL_BASIS)
for variant in ("one", "one_pool", "one_pool_plane", "along_pool_plane"):
    for label in ("compile", "timed"):
        t0 = time.time()
        _t, info, _s = EM.shape_terms(cfg, key, ls, "radius", 32, cfg["timing_band"], (variant,))
        print(f"{variant:18s} {label:7s} total {time.time() - t0:6.1f}  "
              + "  ".join(f"{p} {info[f'seconds_{p}']:.1f}"
                          for p in ("warmup", "capture", "replay", "backward", "shape")),
              flush=True)  # fmt: skip
