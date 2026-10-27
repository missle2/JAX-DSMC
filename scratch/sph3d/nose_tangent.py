"""Is the M 10 sphere run's lobed nose the band rule's area-tangent correction (2026-10-26)?
The tangent projection adds ``lambda g_area`` with ``g_area`` the raw (one-knot shell) area
gradient.  Per iterate: the nose's area gradient (the volume of the sub-box x < tip + 0.6 m,
cheap) and the knot-scale (high-pass, Gaussian 1 knot) parts of it and of the coefficient
change ``coef_{k+1} - coef_k`` near the nose (|phi| < 0.3 at the knot); their cosine and the
least-squares multiple of the step explained by the shell."""

import dataclasses
import functools
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

TAG = "sphere_long_kn0.5_M10_S1_ox_adam_drag_free_w300"
out = Path(O.RESULTS) / TAG
hist = json.loads((out / "history.json").read_text())
rows = hist["rows"]
cfg = dict(O.default_config(), **hist["config"])
_g, ls0 = O.initial_levelset(cfg)
xk = float(ls0.lo[0]) + float(ls0.h) * np.arange(ls0.coef.shape[0])


def nose_volume(coef, lo, hi):
    return O.L.body_volume(dataclasses.replace(ls0, coef=coef), lo, hi, 4)


def grad_nose(coef, lo, hi):
    """The sub-box volume's coefficient gradient (``lo``, ``hi`` host arrays: static)."""
    return jax.grad(functools.partial(nose_volume, lo=lo, hi=hi))(coef)


def hp(a):
    return a - gaussian_filter(a, 1.0, mode="nearest")


for k in range(48, len(rows) - 1, 4):
    if any(rows[k].get("recentred_knots", [0])):
        continue
    c0, c1 = np.load(out / f"coef_{k}.npy"), np.load(out / f"coef_{k + 1}.npy")
    tip = xk[np.argwhere(c0 < 0)[:, 0].min()]
    lo, hi = np.array([tip - 0.3, -0.6, -0.6]), np.array([tip + 0.6, 0.6, 0.6])
    ga = np.asarray(grad_nose(jnp.asarray(c0), lo, hi))
    phik = np.abs(np.asarray(O.L.value(dataclasses.replace(ls0, coef=jnp.asarray(c0)),
                                       O.L.knots(ls0)))).reshape(c0.shape)  # fmt: skip
    sel = (phik < 0.3) & (xk[:, None, None] < tip + 0.5)
    d = c1 - c0
    a, b = hp(d)[sel], hp(ga)[sel]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    lam = float(a @ b / (b @ b))
    explained = 1 - np.linalg.norm(a - lam * b) ** 2 / np.linalg.norm(a) ** 2
    print(f"k {k:3d} tip {tip:+.2f}: cos(hp step, hp shell) {cos:+.2f}  explained "
          f"{explained:.2f}  hp step rms {np.sqrt(np.mean(a**2)):.4f}  full step rms "
          f"{np.sqrt(np.mean(d[sel] ** 2)):.4f}  area_multiplier "
          f"{rows[k].get('area_multiplier')}", flush=True)
