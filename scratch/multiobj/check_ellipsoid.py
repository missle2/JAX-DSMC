"""The 3-D start ellipsoid (optimize --initial ellipsoid, 2026-11-03): its body volume
against the sphere's, its principal axis' pitch against --initial-aoa, its centroid.

    PYTHONPATH=. python3 scratch/multiobj/check_ellipsoid.py
"""

import json
import sys

import numpy as np

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/sph3d")
import optimize as O  # noqa: E402
import shape3d as S  # noqa: E402

TAG = "sphere_long_kn0.5_M10_S1_ox_adam_dlh_hz0_f2_w300"
cfg = dict(O.default_config(),
           **json.load(open(f"examples/mesh/results/optimize/{TAG}/history.json"))["config"])
grid, sphere = O.initial_levelset(cfg)
lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
v_sphere = float(O.L.body_volume(sphere, lo, hi, 4))
print(f"sphere volume {v_sphere:.4f} (history iterate 0: 4.2146)")
pg = S.PhiGrid(sphere, [(-4.0, 4.0), (-2.0, 2.0), (-2.0, 2.0)])
for aoa in (0.0, 15.0):
    c = {**cfg, "initial": "ellipsoid", "initial_thickness": 0.5, "initial_aoa": aoa}
    _g, ls = O.initial_levelset(c)
    v = float(O.L.body_volume(ls, lo, hi, 4))
    phi = pg.phi(np.asarray(ls.coef))
    idx = np.argwhere(phi < 0)
    x = pg.lo + pg.h * idx
    d = x - x.mean(0)
    w, vec = np.linalg.eigh(d[:, :2].T @ d[:, :2])
    ax = vec[:, np.argmax(w)]
    ax = ax if ax[0] < 0 else -ax  # the nose end (-x)
    pitch = np.degrees(np.arctan2(ax[1], -ax[0]))
    ext = x.max(0) - x.min(0)
    print(f"aoa {aoa:4.1f}: volume {v:.4f} ({v / v_sphere:.3f} sphere)  nose-up pitch "
          f"{pitch:+.2f} deg  centroid {x.mean(0).round(3)}  extent {ext.round(2)}")
