"""PR0 -- the optimiser's design and solid regions (`examples/mesh/optimize.hold_regions`): a body
held inside a box loses what crosses it, a cylinder keeps it within its radius, a solid slab is
added to the body; CPU, seconds."""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "mesh"))
import optimize as O  # noqa: E402

from jax_dsmc.mesh import levelset as L  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _disk(radius, h=0.1, n=61, dim=2):
    """A centred ball of ``radius`` on knots ``h`` apart (``n`` per axis)."""
    lo = np.full(dim, -0.5 * h * (n - 1))
    axes = [lo[a] + h * np.arange(n) for a in range(dim)]
    x = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    phi = np.linalg.norm(x, axis=-1) - radius
    return L.from_values(phi, lo, h), lo, lo + h * (n - 1)


def _inside(ls, x):
    return np.asarray(L.value(ls, jnp.asarray(np.atleast_2d(x)))) < 0.0


def test_parse_region():
    kind, box = O.parse_region("box:-1,1,-inf,inf", [0.0, 0.0])
    assert kind == "box" and box.shape == (2, 2) and np.isinf(box[1]).all()
    kind, (numbers, centre) = O.parse_region("cylinder:-1_1_0.5", [0.0, 0.0, 0.0])
    assert kind == "cylinder" and numbers == [-1.0, 1.0, 0.5] and centre.shape == (3,)
    assert O.parse_region("", [0.0, 0.0]) is None


def test_box_cuts_the_body():
    ls, _lo, _hi = _disk(2.0)
    held, moved = O.hold_regions(ls, O.parse_region("box:-1,1,-inf,inf", [0, 0]), None)
    assert moved > 0
    assert _inside(held, [0.0, 1.5])[0]  # kept inside the slab
    assert not _inside(held, [1.5, 0.0])[0]  # cut beyond x = 1
    assert not _inside(held, [-1.3, 0.0])[0]


def test_inactive_region_moves_nothing():
    ls, _lo, _hi = _disk(0.5)
    held, moved = O.hold_regions(ls, O.parse_region("box:-2,2,-2,2", [0, 0]), None)
    assert moved == 0 and held is ls


def test_cylinder_and_solid_slab():
    ls, _lo, _hi = _disk(0.6, h=0.1, n=41, dim=3)
    design = O.parse_region("cylinder:-1,1,0.8", [0.0, 0.0, 0.0])
    solid = O.parse_region("cylinder:0.8,1,0.8", [0.0, 0.0, 0.0])
    held, moved = O.hold_regions(ls, design, solid)
    assert moved > 0
    # the base slab is now body, the ball's centre still is, nothing outside r = 0.8
    assert _inside(held, [0.9, 0.0, 0.7])[0]
    assert _inside(held, [0.0, 0.0, 0.0])[0]
    assert not _inside(held, [0.9, 0.0, 0.95])[0]
