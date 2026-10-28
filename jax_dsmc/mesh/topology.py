"""Topology control of a level-set body on its knot lattice (2026-10-27).

A shape step may pinch a thin neck off (the 3-D L / D plates broke into pieces) or nucleate a
droplet away from the body.  :func:`guard` is a projection after the step, applied like the
face margin (``examples/mesh/optimize.keep_inside``): on the lattice of ``phi``'s values at the
knots, the inside set ``phi < 0`` with face connectivity (4 in 2-D, 6 in 3-D) may lose a knot
only where the knot is *simple for the inside* (Han, Xu & Prince 2003, "A topology preserving
level set method for geometric deformable models", IEEE TPAMI 25(6); Bertrand's topological
number): the inside knots of its geodesic neighbourhood form exactly one face-connected
component touching it, so removing it neither splits a component nor deletes one.  A knot
that would leave the inside otherwise is held inside (its value set to ``-margin``); a knot
that would enter the inside without joining a component already there is held outside
(``+margin``).  One-sided: holes and tunnels (the outside's topology) may form freely.

The test is on the knot values, the body the flow sees is the cubic spline through them:
face-connected negative knots keep a negative spline between them (the interpolant at a
midpoint is ``(-f_0 + 9 f_1 + 9 f_2 - f_3) / 16``, more negative than its two knots when their
outer neighbours are positive), a corner-only contact would not -- hence face connectivity.
"""

import functools

import numpy as np
from scipy import ndimage


@functools.cache
def _offsets(dim: int) -> tuple[np.ndarray, np.ndarray]:
    """``(geodesic, face)``: the neighbourhood offsets of the topological number for face
    connectivity -- 2-D: the 8 neighbours; 3-D: the 18 sharing a face or an edge -- and the
    face neighbours among them."""
    grid = np.stack(np.meshgrid(*[np.arange(-1, 2)] * dim, indexing="ij"), axis=-1)
    offsets = grid.reshape(-1, dim)
    order = np.abs(offsets).sum(axis=1)
    geodesic = offsets[(order > 0) & (order <= min(dim, 2))]
    return geodesic, offsets[order == 1]


def topological_number(inside: np.ndarray, index: tuple[int, ...]) -> int:
    """Face-connected components of ``inside`` in the geodesic neighbourhood of ``index``
    (centre excluded) that contain a face neighbour of it; ``inside`` is padded with
    ``False`` beyond its edges.  1: the knot is simple for the inside."""
    dim = inside.ndim
    geodesic, face = _offsets(dim)
    block = np.zeros((3,) * dim, dtype=bool)
    for off in geodesic:
        j = tuple(np.asarray(index) + off)
        if all(0 <= j[a] < inside.shape[a] for a in range(dim)) and inside[j]:
            block[tuple(off + 1)] = True
    labels, _n = ndimage.label(block, structure=ndimage.generate_binary_structure(dim, 1))
    touching = {int(labels[tuple(off + 1)]) for off in face} - {0}
    return len(touching)


def guard(old: np.ndarray, new: np.ndarray, margin: float) -> tuple[np.ndarray, dict]:
    """The stepped knot values ``new`` with the inside's components kept as in ``old`` (both
    ``f64[m_1, ..., m_D]``, ``phi`` at the knots): knots entering the inside away from it are
    held at ``+margin``, knots leaving it that are not simple at ``-margin`` (removals tried
    in the order of their new value, the farthest out first, each against the inside as
    updated so far).  Returns the values and the counts ``{"held_inside", "held_outside",
    "pieces"}`` (``pieces``: the inside's face-connected components after)."""
    old_in, new_in = old < 0.0, new < 0.0
    structure = ndimage.generate_binary_structure(old.ndim, 1)
    out = np.array(new, dtype=np.float64)
    entering = new_in & ~old_in
    union = old_in | entering
    labels, _n = ndimage.label(union, structure=structure)
    joined = np.unique(labels[old_in])
    nucleated = entering & ~np.isin(labels, joined)
    out[nucleated] = margin
    inside = old_in | (entering & ~nucleated)
    leaving = np.argwhere(old_in & ~new_in)
    held = 0
    for i in np.argsort(-new[tuple(leaving.T)], kind="stable"):
        index = tuple(int(c) for c in leaving[i])
        if topological_number(inside, index) == 1:
            inside[index] = False
        else:
            out[index] = -margin
            held += 1
    return out, {"held_inside": held, "held_outside": int(nucleated.sum()), "pieces": pieces(out)}


def pieces(values: np.ndarray) -> int:
    """Face-connected components of ``values < 0`` (``phi`` at the knots)."""
    structure = ndimage.generate_binary_structure(values.ndim, 1)
    return int(ndimage.label(values < 0.0, structure=structure)[1])
