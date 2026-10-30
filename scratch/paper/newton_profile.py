"""Radius profile r(x, theta) of a paper_newton run's body vs the FM reference."""

import dataclasses
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy import ndimage

from examples.mesh import optimize as O
from jax_dsmc.mesh import levelset as L

tag, k = sys.argv[1], sys.argv[2]
run = Path(O.RESULTS) / tag
hist = json.loads((run / "history.json").read_text())
cfg = dict(O.default_config(), **hist["config"])
grid, ls0 = O.initial_levelset(cfg)
ls = dataclasses.replace(ls0, coef=jnp.asarray(np.load(run / f"coef_{k}.npy")))
xs = np.linspace(-1.3, 1.3, 105)
ys = np.linspace(-1.3, 1.3, 105)
X, Y, Z = np.meshgrid(xs, ys, ys, indexing="ij")
pts = jnp.asarray(np.c_[X.ravel(), Y.ravel(), Z.ravel()])
phi = np.asarray(L.value(ls, pts)).reshape(X.shape)
inside = phi < 0
lab, n = ndimage.label(inside)
sizes = ndimage.sum(inside, lab, range(1, n + 1))
dv = (xs[1] - xs[0]) ** 3
print(f"{tag} it {k}: pieces {n}, volumes {np.round(np.sort(sizes)[::-1] * dv, 4)}")
for i in range(n):
    m = lab == i + 1
    xi = X[m]
    print(f"  piece {i}: x {xi.min():.2f}..{xi.max():.2f}, r_max {np.hypot(Y[m], Z[m]).max():.2f}")
main = lab == (np.argmax(sizes) + 1)
th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
print("  x     r_mean  r_min  r_max (radius along 16 rays)")
for xv in np.linspace(-1.0, 1.0, 11):
    rs = []
    for t in th:
        r = np.linspace(0, 1.2, 241)
        p = jnp.asarray(np.c_[np.full_like(r, xv), r * np.cos(t), r * np.sin(t)])
        v = np.asarray(L.value(ls, p))
        out = np.nonzero(v > 0)[0]
        rs.append(0.0 if v[0] > 0 else (r[out[0]] if len(out) else 1.2))
    rs = np.array(rs)
    print(f"  {xv:+.1f}  {rs.mean():.3f}  {rs.min():.3f}  {rs.max():.3f}")
