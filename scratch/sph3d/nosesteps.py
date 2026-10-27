"""Knot-scale content of the M 10 sphere run's coefficient steps (2026-10-26): per iterate,
the step ``coef_{k+1} - coef_k`` (iterates with a recentring shift skipped) split into its
smooth part (Gaussian, 1 knot) and the rest, near the nose (knots within 0.5 m of the tip) and
over the whole band; and the same split of the coefficients themselves at the nose."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "sphere_long_kn0.5_M10_S1_ox_adam_drag_free_w300"
out = Path(O.RESULTS) / TAG
hist = json.loads((out / "history.json").read_text())
rows = hist["rows"]
cfg = dict(O.default_config(), **hist["config"])
_g, ls0 = O.initial_levelset(cfg)
lo, h = np.asarray(ls0.lo), float(ls0.h)


def rough(a, sel):
    hp = a - gaussian_filter(a, 1.0, mode="nearest")
    return float(np.linalg.norm(hp[sel]) / max(np.linalg.norm(a[sel]), 1e-30))


xk = lo[0] + h * np.arange(ls0.coef.shape[0])
for k in range(0, len(rows) - 1, 8):
    if any(rows[k].get("recentred_knots", [0])):
        continue
    c0, c1 = np.load(out / f"coef_{k}.npy"), np.load(out / f"coef_{k + 1}.npy")
    inside = np.argwhere(c0 < 0)
    tip = xk[inside[:, 0].min()] if len(inside) else 0.0
    band = np.abs(c0) < 0.5
    nose = band & (xk[:, None, None] < tip + 0.5)
    d = c1 - c0
    print(f"k {k:3d} tip {tip:+.2f}  step rough: band {rough(d, band):.2f} nose "
          f"{rough(d, nose):.2f}   coef rough at nose {rough(c0, nose):.3f}  "
          f"step max nose/band {np.abs(d[nose]).max() / np.abs(d[band]).max():.2f}")
