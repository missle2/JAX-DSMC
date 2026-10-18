"""Wall time of each piece of optimize.run's per-iterate shape update (2026-10-17), on a real
optm iterate, a synthetic Adam-like direction (the band-restricted, smoothed sign of a random
field), each piece timed twice (the first includes compilation).

    run_locked.sh gpu env PYTHONPATH=. python3 scratch/geomstep/time_update.py RUN ITER
"""

import dataclasses
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

from jax_dsmc.mesh import cut  # noqa: E402
from jax_dsmc.mesh import levelset as L  # noqa: E402

run, it = sys.argv[1], int(sys.argv[2])
d = f"examples/mesh/results/optimize/{run}"
cfg = json.load(open(f"{d}/history.json"))["config"]
grid, ls0 = O.initial_levelset(cfg)
lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"{d}/coef_{it}.npy")))
area0 = float(L.body_volume(ls0, lo, hi, 4))
rng = np.random.default_rng(0)
out = {}


def timed(name, fn, *a, **k):
    for rep in range(2):
        t0 = time.perf_counter()
        r = fn(*a, **k)
        jax.block_until_ready(jax.tree_util.tree_leaves(r))
        out.setdefault(name, []).append(time.perf_counter() - t0)
    return r


timed("cut.build_mesh (per iterate + per seed forward)", cut.build_mesh, ls,
      dataclasses.replace(grid, volume=jnp.zeros(0)), radius=cfg["radius_cells"])
a, g_area = timed("area_and_gradient", O.area_and_gradient, ls, lo, hi)
g = rng.normal(size=ls.coef.shape)
direction = timed("smoothed_direction", O.smoothed_direction, g, ls, cfg["smooth"], cfg["band"] + ls.h)
direction = np.sign(direction) * (np.abs(direction) > 0)
direction = timed("smoothed_direction (2nd)", O.smoothed_direction, direction, ls, cfg["smooth"], cfg["band"] + ls.h)
direction, _ = timed("tangent_component", O.tangent_component, direction, g_area)
eta = cfg["step"] / np.abs(direction).max()
stepped = dataclasses.replace(ls, coef=ls.coef - eta * jnp.asarray(direction))
new = timed("reinitialise", L.reinitialise, stepped)
timed("project_area (40 bisections)", O.project_area, new, area0, lo, hi)
timed("keep_inside", O.keep_inside, new, lo, hi, cfg["keepout"])
timed("after_step (all of it)", O.after_step, stepped, cfg["area_mode"], area0, lo, hi, cfg["keepout"])
timed("body_volume", L.body_volume, new, lo, hi, 4)
for k, v in out.items():
    print(f"{k:50s} first {v[0]:7.2f} s   repeat {v[1]:7.2f} s")
json.dump(out, open(f"scratch/geomstep/time_update_{run}_{it}.json", "w"), indent=1)
