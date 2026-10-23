"""Old (Euclidean on the raw centroid rows) against new (correction along rigid translations)
centroid projection on the real M 10 optcm body at iterate 150: a smooth step (H1-smoothed
noise in the band) plus a translation tendency of the same size, projected both ways; the
correction's size, roughness (high-frequency fraction) and where it sits.

    PYTHONPATH=. python3 scratch/optax/projection_synthetic.py
"""

import dataclasses
import json
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "examples/mesh")
sys.path.insert(0, "scratch/optax")
import optimize as O  # noqa: E402
from analysis import ROOT, tag  # noqa: E402
from projection_check import hf  # noqa: E402

for regime in ("kn0.5_M10", "kn0.5"):
    t = tag(regime, "ox_adam_cm")
    cfg = dict(O.default_config(), **json.load(open(f"{ROOT}/{t}/history.json"))["config"])
    grid, ls0 = O.initial_levelset(cfg)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(f"{ROOT}/{t}/coef_150.npy")))
    band = cfg["band"] + ls.h
    g_area = O.area_and_gradient(ls, lo, hi)[1]
    cons = O.constraint_gradients(ls, lo, hi, g_area, dict(cfg, fix_centroid=1))
    rows, modes = cons
    rng = np.random.default_rng(1)
    for drift in (0.0, 0.5, 1.0):
        d0 = O.smoothed_direction(rng.normal(size=ls.coef.shape), ls, cfg["smooth"], band)
        d0 /= np.abs(d0).max()
        tx = modes[1] / np.abs(modes[1]).max()
        v0 = d0 + drift * tx
        old, _ = O.tangent_component(v0, rows)
        new, _ = O.tangent_component(v0, cons)
        for name, v in (("old", old), ("new", new)):
            corr = v0 - v
            px = np.abs(corr).sum(axis=1)
            x_peak = float(ls.lo[0]) + ls.h * int(px.argmax())
            print(f"{regime:10s} drift {drift:.1f} {name}: |corr|/|v0| "
                  f"{np.linalg.norm(corr) / np.linalg.norm(v0):.2f}, max|corr|/max|v0| "
                  f"{np.abs(corr).max() / np.abs(v0).max():.2f}, HF corr {hf(corr):.2f}, "
                  f"HF step {hf(v):.3f} (input {hf(v0):.3f}), corr peaks at x {x_peak:.1f}",
                  flush=True)  # fmt: skip
