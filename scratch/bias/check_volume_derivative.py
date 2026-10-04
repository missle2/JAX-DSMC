"""shape.volume_cotangent (smoothed cut-volume derivative, chained to s) against central
differences of the forward's own cut volumes (cut.build_mesh, stratified MC, fixed seed)
on the squash ellipse, with random per-cell weights.

    PYTHONPATH=. python3 scratch/bias/check_volume_derivative.py
"""

import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
import optimize_squash as OS  # noqa: E402

from examples import ensemble_mesh as EM  # noqa: E402
from jax_dsmc import shape  # noqa: E402
from jax_dsmc.mesh import cut  # noqa: E402

cfg = OS.default_config()
for s0 in (0.35, 1.0):
    cfg = dict(cfg, squash=s0)
    grid, mesh, *_ = EM.build(cfg, "squash_y", s0, 10)
    n = grid.volume.shape[0]
    rng = np.random.default_rng(0)
    w = jnp.asarray(rng.normal(size=n))
    cot = shape.volume_cotangent(shape.geometry(mesh), mesh, grid, w)
    adj = EM.control_gradient(dict(cfg, squash=s0), "squash_y", grid, cot)
    cut_cells = np.asarray(mesh.cell_type) == cut.CUT
    for h in (0.02, 0.01, 0.005):
        vols = []
        for sg in (1, -1):
            g2, m2, *_ = EM.build(cfg, "squash_y", s0 + sg * h, 10)
            assert g2.volume.shape[0] == n, "cell count changed"
            vols.append(np.asarray(g2.volume))
        fd = float(np.dot(np.asarray(w), (vols[0] - vols[1]) / (2 * h)))
        print(f"s={s0} h={h}: smoothed VJP {adj:.6g}  FD of MC volumes {fd:.6g}  "
              f"ratio {adj / fd:.4f}  n_sub={int(grid.n_sub)} cut={cut_cells.sum()}")
    # total area derivative: -dA_body/ds = -pi R^2
    tot = EM.control_gradient(dict(cfg, squash=s0), "squash_y", grid,
                              shape.volume_cotangent(shape.geometry(mesh), mesh, grid,
                                                     jnp.ones(n) / float(grid.transverse)))
    print(f"  d(total flow area)/ds = {tot:.6g} vs -pi R^2 = {-np.pi * 1.5**2:.6g}")
