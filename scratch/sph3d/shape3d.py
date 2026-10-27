"""The 3-D level set on a regular grid and orthographic ray-marched renderings of its body
(2026-10-26, for scratch/sph3d/movie3d.py): no mesh library in the env, so the zero
iso-surface is found along each pixel's ray by trilinear sampling of phi and shaded by its
gradient."""

import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy import ndimage

sys.path.insert(0, "examples/mesh")
import optimize as O  # noqa: E402

SPACING = 0.05
CHUNK = 1 << 17


@jax.jit
def _values(ls, pts):
    return O.L.value(ls, pts)


class PhiGrid:
    """phi of the run's level sets on a regular grid over ``window`` ((lo, hi) per axis)."""

    def __init__(self, ls0, window, spacing=SPACING):
        self.ls0 = ls0
        self.lo = np.array([w[0] for w in window], float)
        self.n = tuple(int(round((w[1] - w[0]) / spacing)) + 1 for w in window)
        self.h = spacing
        axes = [self.lo[a] + spacing * np.arange(self.n[a]) for a in range(3)]
        pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        pad = -len(pts) % CHUNK
        self.pts = jnp.asarray(np.concatenate([pts, np.repeat(pts[-1:], pad, 0)]))
        self.count = len(pts)

    def phi(self, coef):
        ls = dataclasses.replace(self.ls0, coef=jnp.asarray(coef))
        out = [np.asarray(_values(ls, self.pts[i:i + CHUNK]))
               for i in range(0, len(self.pts), CHUNK)]  # fmt: skip
        return np.concatenate(out)[: self.count].reshape(self.n)


def body_stats(phi, lo, h):
    """Volume, bounding box, piece count and the centroid of the body ``phi < 0``."""
    inside = phi < 0
    if not inside.any():
        return {"volume": 0.0, "pieces": 0}
    idx = np.argwhere(inside)
    x = lo + h * idx
    return {
        "volume": float(inside.sum() * h**3),
        "pieces": int(ndimage.label(inside)[1]),
        "lo": x.min(0).round(2).tolist(),
        "hi": x.max(0).round(2).tolist(),
        "centroid": x.mean(0).round(2).tolist(),
    }


def view_basis(eye, up_hint):
    """Screen right, screen up and the viewing direction (into the scene) for a camera in the
    direction ``eye`` (from the scene) whose screen up is ``up_hint`` made normal to it."""
    view = -np.asarray(eye, float) / np.linalg.norm(eye)
    right = np.cross(view, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, view)
    return right, up, view


def render(phi, lo, h, centre, half, eye, up_hint, pixel=0.025, reach=4.5, light=None):
    """Shaded image (``float[pixels_v, pixels_h]``, NaN off the body) of ``phi < 0``
    orthographically from ``eye`` (:func:`view_basis`): a ``2 half[0]`` x ``2 half[1]``
    window (screen right x up) on ``centre``, ``pixel`` m pixels; along each ray (depth
    ``+-reach``, samples the grid spacing apart) the first sign change, refined linearly, shaded Lambert +
    ambient by the interpolated gradient."""
    right, up, view = view_basis(eye, up_hint)
    su = np.arange(-half[0], half[0] + 1e-9, pixel)
    sv = np.arange(-half[1], half[1] + 1e-9, pixel)[::-1]
    su, sv = np.meshgrid(su, sv, indexing="xy")
    depth = np.arange(-reach, reach, h)
    origin = centre + su[..., None] * right + sv[..., None] * up  # (V, H, 3)
    ray = origin[:, :, None, :] + depth[None, None, :, None] * view  # (V, H, S, 3)
    coords = ((ray - lo) / h).reshape(-1, 3).T
    big = float(np.abs(phi).max())
    vals = ndimage.map_coordinates(phi, coords, order=1, mode="constant", cval=big)
    vals = vals.reshape(*su.shape, len(depth))
    neg = vals < 0
    hit = neg.any(-1)
    first = np.argmax(neg, -1)
    prev = np.maximum(first - 1, 0)
    v0 = np.take_along_axis(vals, prev[..., None], -1)[..., 0]
    v1 = np.take_along_axis(vals, first[..., None], -1)[..., 0]
    t = np.where(first > 0, v0 / np.maximum(v0 - v1, 1e-30), 0.0)
    d = depth[prev] + t * (depth[1] - depth[0])
    surf = origin + d[..., None] * view
    grads = np.gradient(phi, h)
    c = ((surf - lo) / h).reshape(-1, 3).T
    n = np.stack([ndimage.map_coordinates(g, c, order=1, mode="nearest") for g in grads], -1)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-30)
    n = n.reshape(*su.shape, 3)
    light = view_light(right, up, view) if light is None else light
    shade = 0.25 + 0.75 * np.clip(n @ light, 0.0, 1.0)
    return np.where(hit, shade, np.nan)


def view_light(right, up, view):
    """A key light from upper left, slightly in front of the camera."""
    v = -view + 0.6 * up - 0.5 * right
    return v / np.linalg.norm(v)
