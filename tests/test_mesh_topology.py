"""TP0 -- the topology guard (`jax_dsmc.mesh.topology`) and the thickness measure
(`levelset.body_thinness`): a pinch-off and a droplet are held, a hole and a shrinking body
are not; a thick body has no thin surface, a plate thinner than the depth is partly thin, and
the measure's gradient thickens it.  CPU, seconds."""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc.mesh import levelset as L
from jax_dsmc.mesh import topology as T

jax.config.update("jax_enable_x64", True)


def _box(shape, inside):
    """Knot values: -1 on ``inside`` (a bool mask), +1 elsewhere."""
    return np.where(inside, -1.0, 1.0) * np.ones(shape)


def test_pinch_off_is_held():
    # a dumbbell: two 3 x 3 blocks joined by a one-knot neck; the step opens the neck
    inside = np.zeros((9, 5), dtype=bool)
    inside[0:3, 1:4] = inside[6:9, 1:4] = True
    inside[3:6, 2] = True
    old = _box(inside.shape, inside)
    new = old.copy()
    new[4, 2] = 0.5
    out, counts = T.guard(old, new, 0.01)
    assert out[4, 2] < 0.0
    assert counts == {"held_inside": 1, "held_outside": 0, "pieces": 1}


def test_parallel_removals_cannot_split_together():
    # a two-knot-wide neck: each knot alone is simple, both together split the body
    inside = np.zeros((9, 6), dtype=bool)
    inside[0:3, 1:5] = inside[6:9, 1:5] = True
    inside[3:6, 2:4] = True
    old = _box(inside.shape, inside)
    new = old.copy()
    new[4, 2:4] = 0.5
    out, counts = T.guard(old, new, 0.01)
    assert T.pieces(out) == 1
    assert counts["held_inside"] == 1


def test_droplet_is_held_outside():
    inside = np.zeros((10, 10), dtype=bool)
    inside[2:5, 2:5] = True
    old = _box(inside.shape, inside)
    new = old.copy()
    new[8, 8] = -0.5  # away from the body
    new[5, 3] = -0.5  # growing from it
    out, counts = T.guard(old, new, 0.01)
    assert out[8, 8] > 0.0 and out[5, 3] < 0.0
    assert counts["held_outside"] == 1 and counts["pieces"] == 1


def test_hole_and_shrinking_are_free():
    # 3-D: a 5^3 block loses its centre (a cavity) and a corner layer (shrinking)
    inside = np.zeros((7, 7, 7), dtype=bool)
    inside[1:6, 1:6, 1:6] = True
    old = _box(inside.shape, inside)
    new = old.copy()
    new[3, 3, 3] = 0.5
    new[1, :, :] = 0.5
    out, counts = T.guard(old, new, 0.01)
    np.testing.assert_array_equal(out, new)
    assert counts == {"held_inside": 0, "held_outside": 0, "pieces": 1}


def test_last_knot_is_kept():
    inside = np.zeros((5, 5), dtype=bool)
    inside[2, 2] = True
    old = _box(inside.shape, inside)
    out, counts = T.guard(old, np.ones_like(old), 0.01)
    assert out[2, 2] < 0.0 and counts["pieces"] == 1


def _slab(half_thickness, h=0.1):
    """A 2-D slab ``|y| < half_thickness``, 4 m long with round ends, as knot values."""
    lo = np.array([-3.0, -1.5])
    n = (61, 31)
    x = lo[0] + h * np.arange(n[0])[:, None]
    y = lo[1] + h * np.arange(n[1])[None, :]
    along = np.maximum(np.abs(x) - 2.0, 0.0)
    phi = np.hypot(along, np.maximum(np.abs(y), 0.0)) - half_thickness
    return L.from_values(phi, lo, h), lo, lo + h * (np.array(n) - 1)


def test_thinness_of_thick_and_thin_bodies():
    thick, lo, hi = _slab(0.6)
    thin, _lo, _hi = _slab(0.06)
    depth = 0.2
    t_thick = float(L.body_thinness(thick, lo, hi, depth, 4))
    t_thin = float(L.body_thinness(thin, lo, hi, depth, 4))
    assert t_thick < 0.01
    assert t_thin > 0.3


def _thinness(coef, ls, lo, hi):
    return L.body_thinness(dataclasses.replace(ls, coef=coef), lo, hi, 0.2, 4)


def test_thinness_gradient_thickens():
    ls, lo, hi = _slab(0.08)
    thinness = functools.partial(_thinness, ls=ls, lo=lo, hi=hi)
    g = np.asarray(jax.grad(thinness)(ls.coef))
    assert np.all(np.isfinite(g))
    # lowering phi uniformly thickens the slab: the measure decreases along -1
    assert float(np.sum(g)) > 0.0
    step = 0.02 * jnp.ones_like(ls.coef)
    assert float(thinness(ls.coef - step)) < float(thinness(ls.coef))


def test_banded_thinness_matches_dense():
    ls, lo, hi = _slab(0.08)
    t, g = L.thinness_and_gradient(ls, lo, hi, 0.2, 4, chunk=512)
    thinness = functools.partial(_thinness, ls=ls, lo=lo, hi=hi)
    g_dense = np.asarray(jax.grad(thinness)(ls.coef))
    assert abs(t - float(thinness(ls.coef))) < 1e-3 * abs(t)
    np.testing.assert_allclose(np.asarray(g), g_dense, atol=1e-3 * np.abs(g_dense).max())
