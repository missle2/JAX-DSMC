"""Where the knot-scale part of the M 10 sphere run's late steps comes from (2026-10-26): its
energy against |phi| at the knot (surface vs the band's edge)."""

import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "examples/mesh")
import dataclasses  # noqa: E402

import optimize as O  # noqa: E402

TAG = "sphere_long_kn0.5_M10_S1_ox_adam_drag_free_w300"
out = Path(O.RESULTS) / TAG
hist = json.loads((out / "history.json").read_text())
cfg = dict(O.default_config(), **hist["config"])
grid, ls0 = O.initial_levelset(cfg)
lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
band_w = cfg["band"] + float(ls0.h)
print("band width", band_w, "smooth", cfg["smooth"])
for k in (40, 72, 80, 88):
    if any(hist["rows"][k].get("recentred_knots", [0])):
        print(k, "recentred, skipped")
        continue
    c0, c1 = np.load(out / f"coef_{k}.npy"), np.load(out / f"coef_{k + 1}.npy")
    ls = dataclasses.replace(ls0, coef=jnp.asarray(c0))
    d = c1 - c0
    hp = d - gaussian_filter(d, 1.0, mode="nearest")
    phik = np.abs(np.asarray(O.L.value(ls, O.L.knots(ls)))).reshape(c0.shape)
    mask = np.asarray(O.L.coefficient_band(ls, band_w))
    bins = [0, 0.1, 0.2, 0.3, 0.4, band_w, 9]
    e = hp**2
    share = [e[(phik >= a) & (phik < b)].sum() / e.sum() for a, b in zip(bins[:-1], bins[1:])]
    edge = mask & ~np.asarray(O.L.coefficient_band(ls, band_w - float(ls0.h)))
    print(f"k {k}: hp share by |phi| at knot {[round(float(s), 2) for s in share]} "
          f"(bins {bins[1:-1]}); on the band's outer knot layer {e[edge].sum() / e.sum():.2f}; "
          f"step outside band {np.abs(d[~mask]).max():.2e}"
          f" max {np.abs(d).max():.3f}", flush=True)
