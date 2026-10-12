"""Snapshots of the squash backward's carried value for offline fit studies: one seed's
``shape_terms`` with the backward of the given setting, its value columns ``lam`` (hit
rewards) and ``lam_timing`` at a few captured steps, with the nodes' pre-collision state
``(x, v)``, collision cell, alive mask and the fit scale.

    PYTHONPATH=. python3 scratch/cellfit/snapshot.py OUT.npz SEED '{"S":4,"K":8,"k":64}'
"""

import json
import sys

import jax
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize_squash as OS  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from typing import NamedTuple  # noqa: E402

import jax.numpy as jnp  # noqa: E402

from jax_dsmc import ensemble_backward, shape  # noqa: E402
from jax_dsmc.grid import cell_neighbourhood, collision_cell, num_cells, spacing  # noqa: E402

STEPS = (0, 60, 120, 180, 240, 299)
CAPTURED = {}
_ORIGINAL = EM._backwards
_VARIANTS = shape.gradient_variants


def capturing_backwards(rec, state0, final, grid, v_scale, *args, **kw):
    out = _ORIGINAL(rec, state0, final, grid, v_scale, *args, **kw)
    snap = {}
    for k in STEPS:
        x = rec.x[k]
        snap[f"x{k}"] = np.asarray(x)
        snap[f"v{k}"] = np.asarray(rec.v[k])
        snap[f"alive{k}"] = np.asarray(rec.alive[k])
        snap[f"cell{k}"] = np.asarray(collision_cell(x, grid))
        snap[f"lam{k}"] = np.asarray(out[0].lam[k])
        snap[f"lamt{k}"] = np.asarray(out[1].lam[k])
    dx = np.asarray(spacing(grid))
    snap["scale"] = np.concatenate((dx, np.full(3, v_scale)))
    snap["n_cells"] = num_cells(grid)
    nbr, exists = cell_neighbourhood(grid, 1)
    snap["nbr1"], snap["exists1"] = np.asarray(nbr), np.asarray(exists)
    CAPTURED.update(snap)
    return out


class Positions(NamedTuple):
    """The record's positions only: all that shape.local_normals reads."""

    x: jnp.ndarray


def capturing_variants(*args, **kw):
    """The nearest surface normal (shape.local_normals) of every node at the snapshot steps:
    the one-sided / split fits' side."""
    rec = Positions(kw["records"].x[jnp.asarray(STEPS)])
    normals = shape.local_normals(rec, kw["grid"], kw["mesh"], kw["faces"])
    for i, k in enumerate(STEPS):
        CAPTURED[f"normal{k}"] = np.asarray(normals[i])
    return _VARIANTS(*args, **kw)


def main():
    path, seed = sys.argv[1], int(sys.argv[2])
    st = json.loads(sys.argv[3])
    EM._backwards = capturing_backwards
    shape.gradient_variants = capturing_variants
    ensemble_backward.configure(
        "virtual", st["S"], st["K"], "staged", st.get("basis", "quadratic_full")
    )
    cfg = dict(OS.default_config(), squash=1.0, fnum_scale=4, knudsen=0.5, a_max=64)
    key = jax.random.key(cfg["seed_start"] + seed)
    EM.shape_terms(cfg, key, None, "squash_y", st["k"], 10, ("along_pool_plane",))
    np.savez_compressed(path, **CAPTURED)
    print("saved", path, {k: v.shape for k, v in CAPTURED.items() if k.endswith("0")})


if __name__ == "__main__":
    main()
