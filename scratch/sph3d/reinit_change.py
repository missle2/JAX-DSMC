"""What one reinitialisation pass does to the M 10 sphere run's nose (2026-10-26): per iterate,
the coefficient change of ``levelset.reinitialise`` alone near the nose (|phi| < 0.3 at the
knot, x < tip + 0.5 m: rms, knot-scale share) against the run's whole per-iterate change
there, and the nose band knots whose projection did not reach the surface (they take the
point-cloud distance, error ~ the cloud's spacing)."""

import dataclasses
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

TAG = "sphere_long_kn0.5_M10_S1_ox_adam_drag_free_w300"
out = Path(O.RESULTS) / TAG
rows = json.loads((out / "history.json").read_text())["rows"]
cfg = dict(O.default_config(), **json.loads((out / "history.json").read_text())["config"])
_g, ls0 = O.initial_levelset(cfg)
xk = float(ls0.lo[0]) + float(ls0.h) * np.arange(ls0.coef.shape[0])
L = O.L

for k in (40, 56, 64, 68, 72, 76, 80, 96):
    c0 = np.load(out / f"coef_{k}.npy")
    ls = dataclasses.replace(ls0, coef=jnp.asarray(c0))
    x = L.knots(ls)
    phi = np.asarray(L.value(ls, x)).reshape(c0.shape)
    tip = xk[np.argwhere(phi < 0)[:, 0].min()]
    sel = (np.abs(phi) < 0.3) & (xk[:, None, None] < tip + 0.5)
    # the projection's convergence (as reinitialise)
    band = 3.0 * ls.h + ls.reach
    nose_band = (np.abs(phi) <= band) & (xk[:, None, None] < tip + 0.5)
    xb = jnp.asarray(np.asarray(x)[nose_band.ravel()])
    proj = np.asarray(L.project(ls, xb, int(np.ceil(band / ls.h)) + 4))
    ok = np.abs(np.asarray(L.value(ls, jnp.asarray(proj)))) <= 1e-6 * ls.h
    r = np.asarray(L.reinitialise(ls).coef) - c0
    hp = r - gaussian_filter(r, 1.0, mode="nearest")
    run = np.load(out / f"coef_{k + 1}.npy") - c0 if k + 1 < len(rows) else np.zeros_like(c0)
    print(f"k {k:3d} tip {tip:+.2f}: reinit alone rms {np.sqrt(np.mean(r[sel] ** 2)):.4f} "
          f"(knot-scale {np.linalg.norm(hp[sel]) / max(np.linalg.norm(r[sel]), 1e-30):.2f})  "
          f"run's change rms {np.sqrt(np.mean(run[sel] ** 2)):.4f}  nose band knots not "
          f"projected {1 - ok.mean():.2f} of {ok.size}", flush=True)
