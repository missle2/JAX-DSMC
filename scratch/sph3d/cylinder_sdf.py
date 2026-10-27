"""The cubic B-spline interpolant of a thin cylinder's signed distance (what
``levelset.reinitialise`` builds from the knot distances), by radius in knots (2026-10-26):
the zero set's deviation from the circle (max |r - R| / R, the 4-fold lobe amplitude) and its
piece count on a y-z section, axis on a knot and between knots."""

import dataclasses

import jax.numpy as jnp
import numpy as np
from scipy import ndimage

from jax_dsmc.mesh import levelset as L

H = 0.1
M = (8, 21, 21)
template = L.LevelSet(jnp.zeros(M), jnp.asarray([-0.4, -1.0, -1.0]), H, 16, 40)
lo = np.asarray(template.lo)
ys = np.linspace(-0.45, 0.45, 181)
pts = np.stack(np.meshgrid([0.0], ys, ys, indexing="ij"), -1).reshape(-1, 3)
for offset, label in ((0.0, "axis on a knot"), (0.5 * H, "axis between knots")):
    print(label)
    for r_knots in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        r = r_knots * H
        idx = np.stack(np.meshgrid(*[np.arange(m) for m in M], indexing="ij"), -1)
        x = lo + H * idx
        sdf = np.hypot(x[..., 1] - offset, x[..., 2] - offset) - r
        ls = dataclasses.replace(template, coef=L.interpolate(jnp.asarray(sdf)))
        phi = np.asarray(L.value(ls, jnp.asarray(pts))).reshape(len(ys), len(ys))
        inside = phi < 0
        yy, zz = np.meshgrid(ys - offset, ys - offset, indexing="ij")
        rr = np.hypot(yy, zz)
        edge = inside ^ ndimage.binary_erosion(inside)
        dev = np.abs(rr[edge] - r).max() / r if edge.any() else float("nan")
        print(f"  R = {r_knots:4.2f} h: max |r - R| / R {dev:.2f}  pieces "
              f"{ndimage.label(inside)[1]}  min phi {phi.min():+.3f} (exact {-r:+.3f})")
