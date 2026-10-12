"""How "complicated" are real cut cells for a split per-cell fit?  On level-set iterates of
an optimisation run (optc: circle_coarse_S{S}_{opt}, the 20 x 20 DSMC grid), sample each
collision cell's flow region uniformly, take the nearest surface normal there (the
backward's ``shape.local_normals`` rule: elements and diffuse box faces within one cell),
and per cell the resultant length ``R = |sum n| / #(n != 0)`` -- 1 when one normal
describes every near-surface point of the cell, smaller at corners, high curvature at
the cell scale, two surfaces in one cell.  Reported: the share of near-surface flow volume
(~ particles) in cells with R below thresholds, and the worst cells.

    PYTHONPATH=. python3 scratch/cellfit/geometry_survey.py RUN_DIR it1,it2,.. [OUT.json]
"""

import dataclasses
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize as OPT  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import boundary  # noqa: E402
from jax_dsmc.grid import cell_bounds, collision_cell, n_cartesian, num_cells  # noqa: E402
from jax_dsmc.mesh import geometry as mesh_geometry  # noqa: E402
from jax_dsmc.mesh import levelset as L  # noqa: E402

THRESHOLDS = (0.99, 0.95, 0.9, 0.8, 0.7, 0.5)
SAMPLES = 24  # per axis per Cartesian cell


def config():
    cfg = OPT.default_config()
    cfg.update(fnum_scale=1, grid_scale=1, levelset_h=0.0625, levelset_samples=16, grazing=0.05)
    return cfg


def survey(cfg, coef):
    grid0, ls = OPT.initial_levelset(cfg)
    ls = dataclasses.replace(ls, coef=jnp.asarray(coef))
    grid, mesh, _sp, _params, _static, faces, *_ = EM.build(
        cfg, "radius", EM.nominal(cfg, "radius"), 1, body=ls
    )
    d = len(grid.n)
    lower, upper = cell_bounds(dataclasses.replace(grid, n_sub=0))
    c0 = n_cartesian(grid)
    u = (np.arange(SAMPLES) + 0.5) / SAMPLES
    offs = np.stack(np.meshgrid(*([u] * d), indexing="ij"), -1).reshape(-1, d)
    lo, hi = np.asarray(lower[:c0]), np.asarray(upper[:c0])
    x = (lo[:, None, :] + offs[None] * (hi - lo)[:, None, :]).reshape(-1, d)
    x = jnp.asarray(x)
    flow = ~np.asarray(L.inside(ls, x))
    cell = np.asarray(collision_cell(x, grid))
    diffuse = tuple(f for f in range(2 * d) if faces.bcs[f] is boundary.BC.DIFFUSE)
    normal = np.asarray(mesh_geometry.nearest_normal(x, grid, mesh, diffuse, 1.0))
    c_all = num_cells(grid)
    near = flow & np.any(normal != 0.0, axis=1)
    tot = np.zeros((c_all, 3))
    np.add.at(tot, cell[near], normal[near])
    n_near = np.bincount(cell[near], minlength=c_all)
    n_flow = np.bincount(cell[flow], minlength=c_all)
    r = np.where(n_near > 0, np.linalg.norm(tot, axis=1) / np.maximum(n_near, 1), 1.0)
    # the worst point: smallest cosine to the cell's mean normal
    nc = tot / np.maximum(np.linalg.norm(tot, axis=1, keepdims=True), 1e-300)
    cos = np.sum(normal * nc[cell], axis=1)
    worst = np.ones(c_all)
    np.minimum.at(worst, cell[near], cos[near])
    has = n_near > 0
    w = n_flow[has].astype(float)
    out = {
        "cells_near_surface": int(has.sum()),
        "split_subcells": int(grid.n_sub),
        "flow_share_near_surface": float(n_flow[has].sum() / n_flow.sum()),
        "near_cells_below": {
            str(t): {
                "cells": int((r[has] < t).sum()),
                "flow_share_of_near": float(w[r[has] < t].sum() / w.sum()),
                "flow_share_of_all": float(w[r[has] < t].sum() / n_flow.sum()),
            }
            for t in THRESHOLDS
        },
        "worst_cos_below_0": int((worst[has] < 0.0).sum()),
        "r_min": float(r[has].min()),
    }
    return out


def main():
    run, its = sys.argv[1], [int(i) for i in sys.argv[2].split(",")]
    cfg = config()
    res = {}
    for k in its:
        coef = np.load(f"{run}/coef_{k}.npy")
        s = survey(cfg, coef)
        res[k] = s
        b = s["near_cells_below"]
        print(f"iterate {k:3d}: near-surface cells {s['cells_near_surface']:3d} "
              f"(split sub-cells {s['split_subcells']}), near-surface flow share "
              f"{s['flow_share_near_surface']:.1%}; R min {s['r_min']:.2f}; cells (share of all "
              "flow) with R < " + ", ".join(
                  f"{t}: {b[str(t)]['cells']} ({b[str(t)]['flow_share_of_all']:.2%})"
                  for t in THRESHOLDS) + f"; cells with an opposed normal {s['worst_cos_below_0']}",
              flush=True)
    if len(sys.argv) > 3:
        with open(sys.argv[3], "w") as fh:
            json.dump({"run": run, "iterates": res}, fh, indent=1)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()
