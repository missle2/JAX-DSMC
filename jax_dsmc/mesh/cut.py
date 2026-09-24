"""Host-side cut-cell tables: which elements touch which cell, cell types, flow volumes.

Everything here is numpy on concrete arrays (``grid`` is converted with ``np.asarray``); it
runs once at set-up and its results are the static per-cell tables of
:class:`~jax_dsmc.mesh.types.Mesh`.  Three body kinds are supported: a polygonal
:class:`~jax_dsmc.mesh.types.Surface`, an :class:`~jax_dsmc.mesh.types.AnalyticBody`
(circle / sphere, treated as a single element with id 0) and a
:class:`~jax_dsmc.mesh.types.LevelSet` (also a single element; the cut test is the
coefficient-sign convex-hull test refined by a centred sub-grid, the inside test ``phi > 0``,
the segment-crossing test a sampled ``phi < 0``).

Design (SPARTA ``cut2d.cpp`` / ``cut3d.cpp`` / ``grid_surf.cpp``):

* **csurfs** (:func:`overlaps`): the elements passing SPARTA's *inclusive* clip test against
  the closed cell box -- an element that merely touches a cell at a vertex or lies along a
  face is listed, exactly as SPARTA's ``surf2grid`` does.  A cell is ``CUT`` iff its list is
  non-empty, so touching cells are ``CUT`` with a flow fraction of 1 (touched from the flow
  side) or 0 (a grazing element whose normal points out of the cell: the cell is body).
* **candidates** (:func:`stencil`): the elements whose bounding box overlaps the cell dilated
  by ``radius`` cells -- a superset of ``csurfs`` for every ``radius >= 0``.
* **cell types** (:func:`classify`): the corners of every uncut cell are classified by ray
  parity along one fixed irrational direction against *all* elements (a corner of an
  uncut cell is never on the surface, because a cell containing a surface point is cut by
  the inclusive test).  The 2^D corners of an uncut cell must agree.
* **outside corner** (:func:`outside_corner`): a corner of a cut cell that is in the flow and
  off every element of the cell; the reference point of SPARTA's ``outside_surfs`` trick.
* **point test** (:func:`point_outside`): a point of a cut cell is in the flow iff the segment
  from it to the outside corner crosses the cell's ``csurfs`` an even number of times.  The
  segment stays inside the closed convex cell, so only the cell's own elements can cross it.
  A cut cell without a clean flow corner falls back to the global ray-parity test.
* **flow volume** (:func:`volume_fraction`): stratified Monte Carlo on a ``sub^D`` sub-grid
  with one uniformly jittered point per sub-cell.  The estimate is unbiased; its error is
  set by the sub-cells the surface passes through, ``n_s ~ 2 L (sub / h)^(D-1)`` for a
  surface of measure ``L`` inside a cell of size ``h``: a deterministic bound of
  ``n_s / sub^D ~ 2 L / (h sub)`` on the fraction (of order ``4 / sub`` when the surface
  spans the cell) and a statistical standard deviation of ``sqrt(n_s) / (2 sub^D)``, i.e.
  ``~ sqrt(2 sub) / (2 sub^2)`` in 2-D (4e-3 at ``sub = 32``) and ``~ sqrt(2) / (2 sub^2)``
  in 3-D (3e-3 at ``sub = 16``) per cut cell.  Totals over ``n_cut`` cells add in
  quadrature.
* **split cells** (:func:`split_check`): refused.  The outside sub-samples of a cut cell are
  joined into a graph (neighbouring samples whose joining segment crosses no element),
  small components are re-attached by straight uncrossed segments to the others, and a cell
  whose flow samples still form more than one component raises :class:`SplitCellError`
  -- refine the grid.  A plate thinner than a sub-cell is still detected because
  connectivity is decided by segment crossings, not by adjacency.  Flow pockets smaller than
  a sub-cell can go unnoticed.

Tie rules (all crossing tests): an element is crossed at parameter ``t`` strictly inside
``(0, t_max)``; in 2-D the segment endpoints are classified by the half-open rule
``(s_a > 0) != (s_b > 0)`` on their side of the ray, in 3-D the projected triangle is
oriented counter-clockwise and a rasteriser top-left rule decides points exactly on an
edge, so a shared edge or vertex is counted exactly once for a crossing and zero or twice
for a graze.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from jax_dsmc.grid import cell_bounds, cell_volume, without_splits
from jax_dsmc.mesh import levelset as levelset_mod
from jax_dsmc.mesh import surface as surface_mod
from jax_dsmc.mesh.types import CUT, INSIDE, OUTSIDE, AnalyticBody, LevelSet, Mesh, Surface
from jax_dsmc.state import GridSpec

SURFMAX = 512
"""Maximum elements per cell / per stencil (SPARTA ``surfmax``)."""

RAY_DIRECTION = np.array([1.0, 0.317837245195782, 0.754912676723997])
"""Fixed ray direction of the parity test; irrational components so that a ray from a grid
node never passes through a vertex or along an edge of a generic surface."""

DEFAULT_SUB = {2: 32, 3: 16}
"""Default sub-grid resolution of :func:`volume_fraction` per dimension."""

_SPLIT_WINDOW = {2: 2, 3: 1}
"""Neighbourhood radius (in sub-cells) of the connectivity graph of :func:`split_check`."""

_SPLIT_SUB = {2: 32, 3: 8}
"""Default sub-grid resolution of :func:`split_check` per dimension."""


class SplitCellError(ValueError):
    """A cut cell whose flow region is disconnected (SPARTA would split it)."""

    def __init__(self, cells):
        self.cells = [int(c) for c in cells]
        super().__init__(
            f"{len(self.cells)} cut cell(s) have a disconnected flow region (split cells "
            f"are refused): cells {self.cells[:16]}{'...' if len(self.cells) > 16 else ''}. "
            "Refine the grid so no cell is cut into separate flow pockets."
        )


# --- small helpers ---------------------------------------------------------------


def _is_analytic(body) -> bool:
    return isinstance(body, AnalyticBody)


def _is_levelset(body) -> bool:
    return isinstance(body, LevelSet)


_phi_jit = jax.jit(levelset_mod.value)
_phi_grad_jit = jax.jit(levelset_mod.value_and_grad)


def _phi(body: LevelSet, x: np.ndarray, chunk: int = 1 << 18) -> np.ndarray:
    """``phi`` at host points, ``f64[N]`` (chunked through the jitted evaluator)."""
    x = np.asarray(x, dtype=np.float64)
    out = [
        np.asarray(_phi_jit(body, jnp.asarray(x[i : i + chunk])))
        for i in range(0, x.shape[0], chunk)
    ]
    return np.concatenate(out) if out else np.zeros(0)


def _levelset_cells_mixed(body: LevelSet, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """``bool[C]``: do the coefficients whose support meets the box ``[lo, hi]`` change
    sign?  (Convex-hull property: a box whose coefficients share a sign holds no surface,
    so this test never misses a cut cell; it can flag an uncut one.)"""
    coef = np.asarray(body.coef)
    origin = np.asarray(body.lo, dtype=np.float64)
    D = lo.shape[1]
    j_lo = np.floor((lo - origin) / body.h).astype(np.int64) - 1
    j_hi = np.floor((hi - origin) / body.h).astype(np.int64) + 2
    j_lo = np.clip(j_lo, 0, np.asarray(coef.shape) - 1)
    j_hi = np.clip(j_hi, 0, np.asarray(coef.shape) - 1)
    mixed = np.zeros(lo.shape[0], dtype=bool)
    for c in range(lo.shape[0]):
        sl = tuple(slice(int(j_lo[c, a]), int(j_hi[c, a]) + 1) for a in range(D))
        block = coef[sl]
        mixed[c] = (block.min() <= 0.0) and (block.max() >= 0.0)
    return mixed


def _levelset_cut(body: LevelSet, grid: GridSpec) -> np.ndarray:
    """``bool[C]`` cut cells of a level set: coefficient signs mixed (above) *and* either
    the corners / a centred sub-grid straddle the surface or the smallest ``|phi|`` among
    them is below the sub-grid spacing times the largest ``|grad phi|`` (a sliver)."""
    lo, hi = cell_boxes(grid)
    C, D = lo.shape
    cand = np.nonzero(_levelset_cells_mixed(body, lo, hi))[0]
    cut = np.zeros(C, dtype=bool)
    if cand.size == 0:
        return cut
    sub = _LEVELSET_CUT_SUB
    x, _mi = _samples(lo[cand], hi[cand], sub, _CentreJitter())
    x = np.concatenate([x, corners(lo[cand], hi[cand])], axis=1)  # [n, sub^D + 2^D, D]
    n_pts = x.shape[1]
    phi, g = _phi_grad_jit(body, jnp.asarray(x.reshape(-1, D)))
    phi = np.asarray(phi).reshape(cand.size, n_pts)
    lip = np.linalg.norm(np.asarray(g), axis=1).reshape(cand.size, n_pts).max(axis=1)
    spacing = (hi[cand] - lo[cand]).max(axis=1) / sub
    straddle = (phi.min(axis=1) <= 0.0) & (phi.max(axis=1) >= 0.0)
    sliver = np.abs(phi).min(axis=1) <= spacing * lip
    cut[cand] = straddle | sliver
    return cut


_LEVELSET_CUT_SUB = 8
"""Centred sub-grid per axis of the level-set cut-cell test."""


def _verts(body: Surface) -> np.ndarray:
    """``f64[E, D, D]`` vertex coordinates of every element."""
    p = np.asarray(body.points, dtype=np.float64)
    return p[np.asarray(body.elements)]


def _analytic(body: AnalyticBody):
    return np.asarray(body.centre, dtype=np.float64), float(body.radius)


def cell_boxes(grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """``(lo f64[C, D], hi f64[C, D])`` of every cell, from ``grid.cell_bounds``."""
    lo, hi = cell_bounds(grid)
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def _corner_bits(D: int) -> np.ndarray:
    """``i64[2^D, D]``: bit ``d`` of corner ``k`` (x fastest: corner 1 is the +x corner)."""
    k = np.arange(2**D)
    return np.stack([(k >> d) & 1 for d in range(D)], axis=1)


def corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """``f64[C, 2^D, D]`` corner coordinates, index x fastest."""
    bits = _corner_bits(lo.shape[1]).astype(np.float64)
    return lo[:, None, :] + bits[None, :, :] * (hi - lo)[:, None, :]


def _padded_table(cell: np.ndarray, elem: np.ndarray, C: int, what: str) -> np.ndarray:
    """``i32[C, S]`` of the element ids per cell, sorted ascending, ``-1`` padded."""
    order = np.lexsort((elem, cell))
    cell, elem = cell[order], elem[order]
    counts = np.bincount(cell, minlength=C)
    S = int(counts.max()) if counts.size else 0
    if S > SURFMAX:
        raise ValueError(
            f"{what}: {S} elements in one cell exceeds SURFMAX = {SURFMAX} "
            "(SPARTA surfmax); coarsen the surface or refine the grid"
        )
    table = np.full((C, max(S, 1)), -1, dtype=np.int32)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.arange(cell.size) - starts[cell]
    table[cell, slot] = elem
    return table


def _aabb_overlap(alo, ahi, blo, bhi) -> np.ndarray:
    """``bool[A, B]`` inclusive overlap of two families of boxes."""
    return ((alo[:, None, :] <= bhi[None, :, :]) & (ahi[:, None, :] >= blo[None, :, :])).all(axis=2)


# --- exact element / cell overlap tests ---------------------------------------------------


def _cliptest_2d(p, q, lo, hi) -> np.ndarray:
    """Vectorised port of SPARTA ``Cut2d::cliptest`` (``cut2d.cpp:1086-1138``).

    Inclusive: a segment touching the closed box ``[lo, hi]`` at a single point passes.
    """
    inside_p = ((p >= lo) & (p <= hi)).all(axis=1)
    inside_q = ((q >= lo) & (q <= hi)).all(axis=1)
    result = inside_p | inside_q
    a = p.copy()
    b = q.copy()
    alive = ~result
    for axis in (0, 1):
        other = 1 - axis
        for side, bound in ((0, lo[:, axis]), (1, hi[:, axis])):
            if side == 0:
                a_out = a[:, axis] < bound
                b_out = b[:, axis] < bound
            else:
                a_out = a[:, axis] > bound
                b_out = b[:, axis] > bound
            alive &= ~(a_out & b_out)
            move = alive & (a_out | b_out)
            den = b[:, axis] - a[:, axis]
            frac = np.divide(bound - a[:, axis], den, out=np.zeros_like(den), where=den != 0)
            val = a[:, other] + frac * (b[:, other] - a[:, other])
            a_move = move & a_out
            b_move = move & ~a_out
            a[a_move, axis] = bound[a_move]
            a[a_move, other] = val[a_move]
            b[b_move, axis] = bound[b_move]
            b[b_move, other] = val[b_move]
    return result | alive


def _tri_box_overlap(v, lo, hi) -> np.ndarray:
    """Inclusive triangle / box separating-axis test, ``bool[M]``.

    ``v`` is ``f64[M, 3, 3]``; the boxes are ``lo``/``hi`` ``f64[M, 3]``.  A pair is rejected
    only when an axis separates them *strictly* (beyond a relative ``1e-12`` slack), so an
    element touching the closed box is kept, as in SPARTA's ``Cut3d::clip``.
    """
    c = 0.5 * (lo + hi)
    h = 0.5 * (hi - lo)
    rel = v - c[:, None, :]  # [M, 3, 3]
    slack = 1e-12 * np.max(hi - lo, axis=1)
    ok = ((rel.min(axis=1) <= h + slack[:, None]) & (rel.max(axis=1) >= -h - slack[:, None])).all(
        axis=1
    )
    edges = np.stack([rel[:, 1] - rel[:, 0], rel[:, 2] - rel[:, 1], rel[:, 0] - rel[:, 2]], axis=1)
    n = np.cross(edges[:, 0], edges[:, 1])
    d = np.einsum("ij,ij->i", n, rel[:, 0])
    r = np.einsum("ij,ij->i", h, np.abs(n)) + slack * np.linalg.norm(n, axis=1)
    ok &= np.abs(d) <= r
    for axis in range(3):
        e = np.zeros(3)
        e[axis] = 1.0
        for j in range(3):
            ax = np.cross(e, edges[:, j])  # [M, 3]
            proj = np.einsum("ik,ijk->ij", ax, rel)  # [M, 3]
            r = np.einsum("ij,ij->i", h, np.abs(ax)) + slack * np.linalg.norm(ax, axis=1)
            ok &= (proj.min(axis=1) <= r) & (proj.max(axis=1) >= -r)
    return ok


# --- crossing tests ----------------------------------------------------------------------


def _perp_basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to each row of ``d`` (``f64[M, 3]``)."""
    k = np.argmin(np.abs(d), axis=1)
    e = np.zeros_like(d)
    e[np.arange(d.shape[0]), k] = 1.0
    w1 = np.cross(d, e)
    w1 /= np.linalg.norm(w1, axis=1, keepdims=True)
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    w2 = np.cross(dn, w1)
    return w1, w2


def _cross_pairs(p, d, tmax, v) -> np.ndarray:
    """``bool[M]``: does ``p + t d``, ``0 < t < tmax``, cross element ``v`` (``[M, D, D]``)?"""
    D = p.shape[1]
    if D == 2:
        w = np.stack([-d[:, 1], d[:, 0]], axis=1)
        sa = np.einsum("ij,ij->i", v[:, 0] - p, w)
        sb = np.einsum("ij,ij->i", v[:, 1] - p, w)
        crossed = (sa > 0) != (sb > 0)
        den = sa - sb
        lam = np.divide(sa, den, out=np.zeros_like(sa), where=den != 0)
        q = v[:, 0] + lam[:, None] * (v[:, 1] - v[:, 0])
        t = np.einsum("ij,ij->i", q - p, d) / np.einsum("ij,ij->i", d, d)
        return crossed & (t > 0) & (t < tmax)
    w1, w2 = _perp_basis(d)
    rel = v - p[:, None, :]
    qx = np.einsum("ijk,ik->ij", rel, w1)  # [M, 3]
    qy = np.einsum("ijk,ik->ij", rel, w2)
    area = (qx[:, 1] - qx[:, 0]) * (qy[:, 2] - qy[:, 0]) - (qy[:, 1] - qy[:, 0]) * (
        qx[:, 2] - qx[:, 0]
    )
    flip = area < 0
    x1 = np.where(flip, qx[:, 2], qx[:, 1])
    x2 = np.where(flip, qx[:, 1], qx[:, 2])
    y1 = np.where(flip, qy[:, 2], qy[:, 1])
    y2 = np.where(flip, qy[:, 1], qy[:, 2])
    inside = area != 0
    for ix, iy, jx, jy in (
        (qx[:, 0], qy[:, 0], x1, y1),
        (x1, y1, x2, y2),
        (x2, y2, qx[:, 0], qy[:, 0]),
    ):
        ex = jx - ix
        ey = jy - iy
        edge = ey * ix - ex * iy
        top_left = (ey > 0) | ((ey == 0) & (ex < 0))
        inside &= (edge > 0) | ((edge == 0) & top_left)
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    den = np.einsum("ij,ij->i", n, d)
    t = np.divide(
        np.einsum("ij,ij->i", n, v[:, 0] - p), den, out=np.full_like(den, -1.0), where=den != 0
    )
    return inside & (t > 0) & (t < tmax)


def _ray_counts(verts: np.ndarray, p: np.ndarray, u: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """``i32[M]`` crossings of the ray ``p + t u`` (``t > 0``) with every element."""
    D = p.shape[1]
    if D == 2:
        basis = np.array([[-u[1], u[0]]])
    else:
        w1, w2 = _perp_basis(u[None, :])
        basis = np.concatenate([w1, w2], axis=0)  # [D-1, D]
    pv = np.einsum("ejk,bk->ejb", verts, basis)  # [E, D, D-1]
    vlo = pv.min(axis=1)
    vhi = pv.max(axis=1)
    vmax_t = np.einsum("ejk,k->ej", verts, u).max(axis=1)  # [E]
    out = np.zeros(p.shape[0], dtype=np.int32)
    d = np.broadcast_to(u, (1, D))
    for start in range(0, p.shape[0], chunk):
        pp = p[start : start + chunk]
        pb = pp @ basis.T  # [m, D-1]
        cand = ((pb[:, None, :] >= vlo[None]) & (pb[:, None, :] <= vhi[None])).all(axis=2)
        cand &= (pp @ u)[:, None] <= vmax_t[None, :]
        i, e = np.nonzero(cand)
        if i.size == 0:
            continue
        hit = _cross_pairs(pp[i], np.broadcast_to(d, (i.size, D)), np.inf, verts[e])
        out[start : start + chunk] = np.bincount(i, weights=hit, minlength=pp.shape[0]).astype(
            np.int32
        )
    return out


def _segment_counts_all(verts: np.ndarray, p: np.ndarray, q: np.ndarray, chunk: int = 256):
    """``i32[M]`` crossings of the segments ``p -> q`` with every element."""
    vlo = verts.min(axis=1)
    vhi = verts.max(axis=1)
    out = np.zeros(p.shape[0], dtype=np.int32)
    for start in range(0, p.shape[0], chunk):
        pp = p[start : start + chunk]
        qq = q[start : start + chunk]
        cand = _aabb_overlap(np.minimum(pp, qq), np.maximum(pp, qq), vlo, vhi)
        i, e = np.nonzero(cand)
        if i.size == 0:
            continue
        hit = _cross_pairs(pp[i], qq[i] - pp[i], 1.0, verts[e])
        out[start : start + chunk] = np.bincount(i, weights=hit, minlength=pp.shape[0]).astype(
            np.int32
        )
    return out


def _segment_counts_cell(body, p, q, cell, csurfs, chunk: int = 1 << 15) -> np.ndarray:
    """``i32[M]``: crossings of the segments ``p -> q`` with the elements of ``csurfs[cell]``.

    For an analytic body this is 1 if the segment enters the ball, else 0; for a level
    set, 1 if any of ``_SEGMENT_SUB`` points along the segment has ``phi < 0``.
    """
    if _is_levelset(body):
        d = q - p
        s = (np.arange(_SEGMENT_SUB) + 0.5) / _SEGMENT_SUB
        x = p[:, None, :] + s[None, :, None] * d[:, None, :]
        phi = _phi(body, x.reshape(-1, p.shape[1])).reshape(p.shape[0], _SEGMENT_SUB)
        return (phi.min(axis=1) < 0.0).astype(np.int32)
    if _is_analytic(body):
        c, r = _analytic(body)
        d = q - p
        t = np.einsum("ij,ij->i", c[None, :] - p, d) / np.maximum(
            np.einsum("ij,ij->i", d, d), 1e-300
        )
        closest = p + np.clip(t, 0.0, 1.0)[:, None] * d
        return (np.linalg.norm(closest - c[None, :], axis=1) < r).astype(np.int32)
    verts = _verts(body)
    vlo = verts.min(axis=1)
    vhi = verts.max(axis=1)
    out = np.zeros(p.shape[0], dtype=np.int32)
    for start in range(0, p.shape[0], chunk):
        rows = csurfs[cell[start : start + chunk]]  # [m, S]
        i, s = np.nonzero(rows >= 0)
        if i.size == 0:
            continue
        e = rows[i, s]
        pp = p[start : start + chunk][i]
        qq = q[start : start + chunk][i]
        keep = ((np.minimum(pp, qq) <= vhi[e]) & (np.maximum(pp, qq) >= vlo[e])).all(axis=1)
        hit = np.zeros(i.size, dtype=bool)
        if keep.any():
            hit[keep] = _cross_pairs(pp[keep], qq[keep] - pp[keep], 1.0, verts[e[keep]])
        out[start : start + chunk] = np.bincount(i, weights=hit, minlength=rows.shape[0]).astype(
            np.int32
        )
    return out


def crossings(surface: Surface, p, q) -> np.ndarray:
    """``i32[N]`` number of elements the segments ``p -> q`` cross (all elements)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return _segment_counts_all(_verts(surface), p, q)


_SEGMENT_SUB = 8
"""Points per segment of the level-set segment-crossing test."""


def _global_outside(body, x: np.ndarray) -> np.ndarray:
    """``bool[N]``: is ``x`` in the flow, by the global test (ray parity / radius / phi)?"""
    if _is_levelset(body):
        return _phi(body, x) > 0.0
    if _is_analytic(body):
        c, r = _analytic(body)
        return np.linalg.norm(x - c[None, :], axis=1) > r
    D = x.shape[1]
    u = RAY_DIRECTION[:D] / np.linalg.norm(RAY_DIRECTION[:D])
    return _ray_counts(_verts(body), x, u) % 2 == 0


# --- the per-cell tables ------------------------------------------------------------


def overlaps(body, grid: GridSpec) -> np.ndarray:
    """``i32[C, S_cell]`` elements overlapping each closed cell (SPARTA ``csurfs``).

    Bounding-box rejection, then the inclusive clip test (``Cut2d::cliptest`` port in 2-D,
    a separating-axis test in 3-D).  Rows sorted ascending, ``-1`` padded.  For an analytic
    body the single element ``0`` is listed in every cell whose closed box the circle /
    sphere touches.
    """
    lo, hi = cell_boxes(grid)
    C, D = lo.shape
    if _is_levelset(body):
        cell = np.nonzero(_levelset_cut(body, grid))[0]
        return _padded_table(cell, np.zeros(cell.size, np.int64), C, "overlaps")
    if _is_analytic(body):
        c, r = _analytic(body)
        nearest = np.clip(c[None, :], lo, hi)
        dmin = np.linalg.norm(nearest - c[None, :], axis=1)
        dmax = np.linalg.norm(corners(lo, hi) - c[None, None, :], axis=2).max(axis=1)
        cut = (dmin <= r) & (r <= dmax)
        cell = np.nonzero(cut)[0]
        return _padded_table(cell, np.zeros(cell.size, np.int64), C, "overlaps")
    verts = _verts(body)
    if verts.shape[2] != D:
        raise ValueError(f"surface is {verts.shape[2]}-D but the grid is {D}-D")
    cand = _aabb_overlap(lo, hi, verts.min(axis=1), verts.max(axis=1))
    cell, elem = np.nonzero(cand)
    if D == 2:
        ok = _cliptest_2d(verts[elem, 0], verts[elem, 1], lo[cell], hi[cell])
    else:
        ok = _tri_box_overlap(verts[elem], lo[cell], hi[cell])
    return _padded_table(cell[ok], elem[ok], C, "overlaps")


def stencil(body, grid: GridSpec, radius: int) -> np.ndarray:
    """``i32[C, S_max]`` elements whose bounding box meets the cell dilated by ``radius`` cells.

    A superset of :func:`overlaps` for any ``radius >= 0`` (at ``radius = 0`` it is the
    bounding-box stage of the clip test).  Rows sorted ascending, ``-1`` padded; raises if
    a row exceeds ``SURFMAX``.
    """
    if radius < 0:
        raise ValueError("stencil radius must be >= 0")
    lo, hi = cell_boxes(grid)
    C = lo.shape[0]
    dl = radius * (hi - lo)
    if _is_levelset(body):
        cell = np.nonzero(_levelset_cells_mixed(body, lo - dl, hi + dl))[0]
        return _padded_table(cell, np.zeros(cell.size, np.int64), C, "stencil")
    if _is_analytic(body):
        c, r = _analytic(body)
        blo = (c - r)[None, :]
        bhi = (c + r)[None, :]
    else:
        verts = _verts(body)
        blo = verts.min(axis=1)
        bhi = verts.max(axis=1)
    cell, elem = np.nonzero(_aabb_overlap(lo - dl, hi + dl, blo, bhi))
    return _padded_table(cell, elem, C, "stencil")


def _corner_outside(body, grid: GridSpec) -> np.ndarray:
    """``bool[C, 2^D]``: is each cell corner in the flow (global test, shared nodes once)?"""
    lo, hi = cell_boxes(grid)
    x = corners(lo, hi)
    C, K, D = x.shape
    nodes, inverse = np.unique(x.reshape(C * K, D), axis=0, return_inverse=True)
    return _global_outside(body, nodes)[inverse.ravel()].reshape(C, K)


def classify(body, grid: GridSpec, csurfs: np.ndarray) -> np.ndarray:
    """``i32[C]`` cell types: ``CUT`` iff ``csurfs`` non-empty, else by corner parity."""
    cut = csurfs[:, 0] >= 0
    outside = _corner_outside(body, grid)
    agree = (outside == outside[:, :1]).all(axis=1)
    bad = np.nonzero(~cut & ~agree)[0]
    if bad.size:
        raise ValueError(
            f"{bad.size} uncut cell(s) have corners on both sides of the surface (first: "
            f"{bad[:8].tolist()}); the surface is not closed or the clip test missed a cell"
        )
    cell_type = np.where(outside[:, 0], OUTSIDE, INSIDE).astype(np.int32)
    cell_type[cut] = CUT
    return cell_type


def _plane_distance(body, x: np.ndarray, csurfs_rows: np.ndarray) -> np.ndarray:
    """``f64[N]`` min over the row's elements of the distance of ``x`` to the element's
    line / plane (``inf`` for an empty row) -- a lower bound on the distance to the element.
    For a level set ``|phi|`` (distance-like by construction)."""
    if _is_levelset(body):
        return np.abs(_phi(body, x))
    if _is_analytic(body):
        c, r = _analytic(body)
        return np.abs(np.linalg.norm(x - c[None, :], axis=1) - r)
    verts = _verts(body)
    n = np.asarray(surface_mod.normals(body.points, body.elements))
    i, s = np.nonzero(csurfs_rows >= 0)
    e = csurfs_rows[i, s]
    dist = np.abs(np.einsum("ij,ij->i", x[i] - verts[e, 0], n[e]))
    out = np.full(x.shape[0], np.inf)
    np.minimum.at(out, i, dist)
    return out


def outside_corner(body, grid: GridSpec, cell_type: np.ndarray, csurfs: np.ndarray) -> np.ndarray:
    """``i32[C]``: for ``CUT`` cells the index (x fastest: bit 0 = x hi, bit 1 = y hi, bit 2
    = z hi) of a corner in the flow (ray parity) whose distance to every element of the cell
    is strictly positive (``> 1e-10 h``; the farthest such corner); ``0`` for ``OUTSIDE``
    cells; ``-1`` for ``INSIDE`` cells and for cut cells with no flow at all (a cell inside
    the body that a grazing element only touches).  For a polygonal surface a cut cell that
    has flow but no clean flow corner (a flow pocket reaching no corner, or every flow
    corner on the surface) raises ``ValueError``: the tracker's point test needs the
    corner, so refine the grid.  A level set or an analytic body keeps ``-1`` there: its
    inside test (``phi < 0`` / ``|x - c| < r``) is exact and never reads the corner.
    """
    lo, hi = cell_boxes(grid)
    _C, D = lo.shape
    K = 2**D
    out = np.where(cell_type == INSIDE, -1, 0).astype(np.int32)
    cut = np.nonzero(cell_type == CUT)[0]
    if cut.size == 0:
        return out
    x = corners(lo[cut], hi[cut]).reshape(cut.size * K, D)
    flow = _global_outside(body, x)
    rows = np.repeat(csurfs[cut], K, axis=0)
    dist = _plane_distance(body, x, rows)
    eps = 1e-10 * np.repeat(np.max(hi[cut] - lo[cut], axis=1), K)
    score = np.where(flow & (dist > eps), dist, -1.0).reshape(cut.size, K)
    best = np.argmax(score, axis=1)
    found = score[np.arange(cut.size), best] > 0
    out[cut] = np.where(found, best, -1)
    missing = cut[~found]
    if missing.size and (_is_levelset(body) or _is_analytic(body)):
        # the corner is the reference point of the POLYGON inside test only: a level set
        # (phi < 0) and an analytic body (|x - c| < r) are tested exactly and never read
        # it, and a flow pocket entering a cell through a face with every corner in the
        # body is a valid shape (optc Adam S1 iterate 56 / S8 iterate 9, 2026-09-24)
        return out
    if missing.size:
        # no clean flow corner: acceptable only if the cell holds no flow (probe the
        # centres of a coarse sub-grid with the global test)
        probe, _mi = _samples(lo[missing], hi[missing], _PROBE_SUB, _CentreJitter())
        has_flow = _global_outside(body, probe.reshape(-1, D)).reshape(missing.size, -1).any(1)
        if has_flow.any():
            bad = missing[has_flow]
            raise ValueError(
                f"{bad.size} cut cell(s) contain flow but no corner that is in the flow and off "
                f"every element of the cell (first: {bad[:8].tolist()}); the point-in-flow "
                "test needs such a corner -- refine the grid or move the surface"
            )
    return out


_PROBE_SUB = 4
"""Sub-grid of the flow probe in :func:`outside_corner` for cells without a clean corner."""


class _CentreJitter:
    """Stand-in for a ``Generator`` that puts every sample at its sub-cell centre."""

    @staticmethod
    def random(shape):
        return np.full(shape, 0.5)


def point_outside(body, grid: GridSpec, cell, x, csurfs, outside_corner) -> np.ndarray:
    """``bool[N]``: is point ``x[i]`` of cut cell ``cell[i]`` in the flow?

    SPARTA's ``outside_surfs``: the segment from ``x`` to the cell's outside corner crosses
    the cell's elements an even number of times.  Points in cells without an outside corner
    (``-1``) use the global test; for an analytic body the test is ``|x - c| > r``.
    """
    x = np.asarray(x, dtype=np.float64)
    cell = np.asarray(cell)
    if _is_analytic(body) or _is_levelset(body):
        return _global_outside(body, x)
    lo, hi = cell_boxes(grid)
    oc = np.asarray(outside_corner)[cell]
    out = np.zeros(x.shape[0], dtype=bool)
    have = oc >= 0
    if have.any():
        bits = _corner_bits(x.shape[1]).astype(np.float64)[oc[have]]
        target = lo[cell[have]] + bits * (hi[cell[have]] - lo[cell[have]])
        n = _segment_counts_cell(body, x[have], target, cell[have], np.asarray(csurfs))
        out[have] = n % 2 == 0
    if (~have).any():
        out[~have] = _global_outside(body, x[~have])
    return out


def _samples(lo: np.ndarray, hi: np.ndarray, sub: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Stratified samples of the boxes: ``(x f64[n, sub^D, D], multi-index i64[sub^D, D])``.

    One point per sub-cell, uniformly jittered; the sub-grid is flattened with the last axis
    fastest so that ``x.reshape(n, *(sub,) * D, D)`` is the sub-grid.
    """
    n, D = lo.shape
    mi = np.stack(np.indices((sub,) * D), axis=-1).reshape(-1, D)
    jitter = rng.random((n, mi.shape[0], D))
    x = lo[:, None, :] + (mi[None, :, :] + jitter) / sub * (hi - lo)[:, None, :]
    return x, mi


def _cut_samples_outside(body, grid, cell_type, csurfs, outside_corner, sub, seed):
    """Samples of every cut cell and their flow flags: ``(cut ids, x [n, S, D], mi, flags)``."""
    lo, hi = cell_boxes(grid)
    D = lo.shape[1]
    sub = DEFAULT_SUB[D] if sub is None else int(sub)
    cut = np.nonzero(np.asarray(cell_type) == CUT)[0]
    rng = np.random.default_rng(seed)
    x, mi = _samples(lo[cut], hi[cut], sub, rng)
    n_s = mi.shape[0]
    cell = np.repeat(cut, n_s)
    flow = point_outside(body, grid, cell, x.reshape(-1, D), csurfs, outside_corner)
    return cut, x, mi, flow.reshape(cut.size, n_s)


def volume_fraction(
    body, grid: GridSpec, cell_type, csurfs, outside_corner, sub: int | None = None, seed: int = 0
) -> np.ndarray:
    """``f64[C]`` flow fraction of every cell in ``[0, 1]``: ``OUTSIDE`` 1, ``INSIDE`` 0,
    ``CUT`` a stratified Monte Carlo estimate on a ``sub^D`` sub-grid (default
    ``DEFAULT_SUB``; error bounds in the module docstring)."""
    cell_type = np.asarray(cell_type)
    frac = (cell_type != INSIDE).astype(np.float64)
    cut, _x, _mi, flow = _cut_samples_outside(
        body, grid, cell_type, csurfs, outside_corner, sub, seed
    )
    frac[cut] = flow.mean(axis=1)
    return frac


def _local_edges(mi: np.ndarray, sub: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs ``(a, b)`` of sub-cells within ``window`` of each other (each pair once)."""
    D = mi.shape[1]
    offsets = np.stack(np.indices((2 * window + 1,) * D), axis=-1).reshape(-1, D) - window
    # keep one of every +/- pair: the first non-zero component positive
    first = np.argmax(offsets != 0, axis=1)
    offsets = offsets[(offsets != 0).any(axis=1) & (offsets[np.arange(len(offsets)), first] > 0)]
    strides = sub ** np.arange(D - 1, -1, -1)
    flat = mi @ strides
    a_all, b_all = [], []
    for off in offsets:
        nb = mi + off
        ok = ((nb >= 0) & (nb < sub)).all(axis=1)
        a_all.append(flat[ok])
        b_all.append(nb[ok] @ strides)
    return np.concatenate(a_all), np.concatenate(b_all)


def _components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    graph = coo_matrix((np.ones(a.size), (a, b)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def _n_resolved(labels: np.ndarray, flow: np.ndarray, min_samples: int) -> int:
    """Number of flow components with at least ``min_samples`` samples."""
    _ids, counts = np.unique(labels[flow], return_counts=True)
    return int((counts >= min_samples).sum())


def _cell_components(body, cell, x, flow, a, b, csurfs, min_samples, k_near=32) -> int:
    """Number of resolved connected flow components among the samples ``x`` of one cell
    (:func:`_cell_labels`)."""
    labels = _cell_labels(body, cell, x, flow, a, b, csurfs, k_near)
    return _n_resolved(labels, flow, min_samples)


def _cell_labels(body, cell, x, flow, a, b, csurfs, k_near=32) -> np.ndarray:
    """Connected-component label of every sample ``x`` of one cell (``i64[S]``).

    Local edges ``(a, b)`` survive when both ends are in the flow and the segment crosses no
    element of the cell; then, up to four times, every sample outside the largest component
    is linked to its ``k_near`` nearest samples of the largest component when that segment
    is uncrossed.  Components smaller than ``min_samples`` are unresolved and ignored.
    """
    ok = flow[a] & flow[b]
    a, b = a[ok], b[ok]
    keep = _segment_counts_cell(body, x[a], x[b], np.full(a.size, cell), csurfs) == 0
    a, b = a[keep], b[keep]
    n = x.shape[0]
    labels = _components(n, a, b)
    for _ in range(4):
        ids, counts = np.unique(labels[flow], return_counts=True)
        if ids.size <= 1:
            break
        main = ids[np.argmax(counts)]
        small = np.nonzero(flow & (labels != main))[0]
        others = np.nonzero(flow & (labels == main))[0]
        dist = np.linalg.norm(x[small][:, None, :] - x[others][None, :, :], axis=2)
        k = min(k_near, others.size)
        near = others[np.argpartition(dist, k - 1, axis=1)[:, :k]]  # [n_small, k]
        aa = np.repeat(small, k)
        bb = near.ravel()
        new = _segment_counts_cell(body, x[aa], x[bb], np.full(aa.size, cell), csurfs) == 0
        if not new.any():
            break
        a = np.concatenate([a, aa[new]])
        b = np.concatenate([b, bb[new]])
        labels = _components(n, a, b)
    return labels


def split_check(
    body,
    grid: GridSpec,
    cell_type,
    csurfs,
    outside_corner,
    sub: int | None = None,
    seed: int = 0,
    min_samples: int | None = None,
    raise_on_split: bool = True,
) -> list[int]:
    """Find cut cells whose flow region is disconnected; raise :class:`SplitCellError`.

    Flow samples (``sub^D`` stratified, default ``_SPLIT_SUB``) are joined when the segment
    between two samples within ``_SPLIT_WINDOW`` sub-cells crosses no element of the cell;
    every component other than the largest is then re-attached through the nearest
    flow samples of the largest component when an uncrossed straight segment exists.
    Components with fewer than ``min_samples`` samples (default ``max(4, sub^D / 64)``,
    i.e. flow pockets below ~1.6 % of the cell) are unresolved and ignored -- straight
    segments between samples hugging a curved surface can dip into the body, so a thin
    sliver of flow along the surface would otherwise be a false split.  Cells with more
    than one resolved component are returned (and raised).
    """
    lo, _hi = cell_boxes(grid)
    D = lo.shape[1]
    sub = _SPLIT_SUB[D] if sub is None else int(sub)
    if min_samples is None:
        min_samples = max(4, sub**D // 64)
    csurfs = np.asarray(csurfs)
    cut, x, mi, flow = _cut_samples_outside(
        body, grid, cell_type, csurfs, outside_corner, sub, seed
    )
    a, b = _local_edges(mi, sub, _SPLIT_WINDOW[D])
    split = [
        int(cut[i])
        for i in range(cut.size)
        if _cell_components(body, cut[i], x[i], flow[i], a, b, csurfs, min_samples) > 1
    ]
    if split and raise_on_split:
        raise SplitCellError(split)
    return split


def split_regions(
    body, grid, cell_type, csurfs, outside_corner, sub=None, seed=0, min_samples=None
):
    """The flow regions of every split cell (SPARTA ``Cut2d::split`` / ``Cut3d::split``):
    a list of ``(cell, region i32[sub^D], share f64[R])`` -- for each stratum of the
    cell's ``sub^D`` sub-grid (last axis fastest) the region its sample belongs to,
    regions numbered by decreasing size, and each region's share of the cell's flow
    samples.  Strata in the body or in unresolved slivers (fewer than ``min_samples``
    samples, as :func:`split_check`) take the region of the nearest resolved flow sample,
    so every point of the cell maps to a region.  The components are those of
    :func:`split_check` (the same samples, edges and re-attachment)."""
    from scipy.spatial import cKDTree  # noqa: PLC0415 -- host-only dependency

    lo, _hi = cell_boxes(grid)
    D = lo.shape[1]
    sub = _SPLIT_SUB[D] if sub is None else int(sub)
    if min_samples is None:
        min_samples = max(4, sub**D // 64)
    csurfs = np.asarray(csurfs)
    cut, x, mi, flow = _cut_samples_outside(
        body, grid, cell_type, csurfs, outside_corner, sub, seed
    )
    a, b = _local_edges(mi, sub, _SPLIT_WINDOW[D])
    out = []
    for i in range(cut.size):
        labels = _cell_labels(body, cut[i], x[i], flow[i], a, b, csurfs)
        ids, counts = np.unique(labels[flow[i]], return_counts=True)
        keep = counts >= min_samples
        if keep.sum() <= 1:
            continue
        ids, counts = ids[keep], counts[keep]
        order = np.argsort(-counts, kind="stable")
        region_of = {int(ids[j]): r for r, j in enumerate(order)}
        region = np.array([region_of.get(int(lab), -1) for lab in labels])
        region = np.where(flow[i], region, -1)
        known = region >= 0
        # body and sliver strata: the region of the nearest resolved flow sample
        _d, near = cKDTree(x[i][known]).query(x[i][~known])
        region[~known] = region[known][near]
        share = counts[order] / counts.sum()
        out.append((int(cut[i]), region.astype(np.int32), share))
    return out


_SUB_CELL_BLOCK = 4
"""Sub-cell slots are allocated in blocks of this many (unused ones have zero volume), so
an optimiser whose shape gains or loses a split recompiles rarely."""


def split_tables(regions, volume: np.ndarray, n_cart: int):
    """``(n_sub, split_of, split_table, sub_parent, volume)`` of the sub-cells of
    ``regions`` (:func:`split_regions`) over ``n_cart`` Cartesian cells: the largest region
    keeps its cell's index, every other region gets the next sub-cell ``n_cart + j``;
    ``volume`` (``f64[n_cart]``, the cut volumes) is split by the regions' shares and
    extended with the sub-cells' (zero for unused slots)."""
    extra = sum(len(share) - 1 for _cell, _region, share in regions)
    n_sub = -(-extra // _SUB_CELL_BLOCK) * _SUB_CELL_BLOCK
    split_of = np.full(n_cart, -1, np.int32)
    n_strata = regions[0][1].size if regions else 1
    table = np.zeros((len(regions), n_strata), np.int32)
    sub_parent = np.zeros(n_sub, np.int32)
    vol = np.concatenate((np.asarray(volume, np.float64), np.zeros(n_sub)))
    j = 0
    for row, (cell, region, share) in enumerate(regions):
        split_of[cell] = row
        cells = [cell]
        for _r in range(1, len(share)):
            cells.append(n_cart + j)
            sub_parent[j] = cell
            j += 1
        cells = np.asarray(cells, np.int32)
        table[row] = cells[region]
        full = vol[cell]
        vol[cells] = full * share
    return n_sub, split_of, table, sub_parent, vol


def flow_mask(mesh: Mesh, grid: GridSpec, sub: int = 8) -> np.ndarray:
    """``bool[C, sub^D]``: which sub-cells (centres) of every cell lie in the flow.

    OUTSIDE cells are all True, INSIDE cells all False, CUT cells by the same
    :func:`point_outside` test the tracker's inside-body check uses, at the sub-cell
    centres (no jitter).  This is the **visualisation** of the cut: cells are still boxes
    and fields are per cell, but drawing a cut cell's value only on its flow side shows
    where the body actually is.  Sub-cell order is the last axis fastest, as
    :func:`_samples`.
    """
    lo, hi = cell_boxes(grid)
    _C, D = lo.shape
    cell_type = np.asarray(mesh.cell_type)
    n_s = sub**D
    mask = np.repeat((cell_type != INSIDE)[:, None], n_s, axis=1)
    cut = np.nonzero(cell_type == CUT)[0]
    if cut.size:
        x, _mi = _samples(lo[cut], hi[cut], sub, _CentreJitter)
        cell = np.repeat(cut, n_s)
        flow = point_outside(
            mesh.body,
            grid,
            cell,
            x.reshape(-1, D),
            np.asarray(mesh.csurfs),
            np.asarray(mesh.outside_corner),
        )
        mask[cut] = flow.reshape(cut.size, n_s)
    return mask


def with_volume(grid: GridSpec, mesh: Mesh) -> GridSpec:
    """The grid carrying the mesh's per-cell flow volumes (``grid.cell_volume`` reads them)
    and its split cells' sub-cells (``grid.collision_cell`` reads them)."""
    return dataclasses.replace(
        grid,
        volume=mesh.volume,
        n_sub=mesh.n_sub,
        split_of=mesh.split_of,
        split_table=mesh.split_table,
        sub_parent=mesh.sub_parent,
    )


def exact_flow_volume(body, grid: GridSpec) -> float:
    """Box volume (times ``grid.transverse``) minus the exact body volume (for a level set:
    minus the sharp sub-sampled body volume at ``_LEVELSET_VOLUME_SUB`` points per knot
    spacing -- there is no closed form)."""
    lo = np.asarray(grid.lo, dtype=np.float64)
    hi = np.asarray(grid.hi, dtype=np.float64)
    D = lo.size
    if _is_levelset(body):
        sub = _LEVELSET_VOLUME_SUB[D]
        counts = [int(np.ceil((hi[a] - lo[a]) / body.h)) * sub for a in range(D)]
        axes = [
            lo[a] + (hi[a] - lo[a]) * (np.arange(counts[a]) + 0.5) / counts[a] for a in range(D)
        ]
        x = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, D)
        inside = _phi(body, x) < 0.0
        vb = inside.mean() * np.prod(hi - lo)
    elif _is_analytic(body):
        _c, r = _analytic(body)
        vb = np.pi * r**2 if D == 2 else 4.0 / 3.0 * np.pi * r**3
    else:
        vb = surface_mod.body_volume(body)
    return float((np.prod(hi - lo) - vb) * float(grid.transverse))


_LEVELSET_VOLUME_SUB = {2: 16, 3: 4}
"""Points per knot spacing and axis of the level-set flow-volume reference (2-D: 1280² on
the SPARTA circle box; 3-D: 160³ on the sphere box -- 16 per axis there would be 640³
points, 6 GB of coordinates)."""


def build_mesh(
    body, grid: GridSpec, radius: int = 1, sub: int | None = None, seed: int = 0
) -> Mesh:
    """Assemble the :class:`Mesh` of one body on one grid (all tables computed here).

    A surface is checked to be watertight and inside the box first.  A cut cell whose flow
    is disconnected becomes a split cell: its regions get sub-cells
    (:func:`split_regions`, :func:`split_tables`), SPARTA's ``nsplit > 1``.
    """
    grid = without_splits(grid)
    lo = np.asarray(grid.lo, dtype=np.float64)
    D = lo.size
    if D not in (2, 3):
        raise NotImplementedError(f"build_mesh: D = {D} not supported (2 or 3)")
    if _is_levelset(body):
        if body.dim != D:
            raise ValueError(f"level set is {body.dim}-D but the grid is {D}-D")
        levelset_mod.check_covers(body, grid.lo, grid.hi)
        kind = "levelset"
        normals = np.zeros((1, D))
        areas = np.zeros(1)
    elif _is_analytic(body):
        _c, r = _analytic(body)
        kind = "circle" if D == 2 else "sphere"
        normals = np.zeros((1, D))
        areas = np.array([2.0 * np.pi * r if D == 2 else 4.0 * np.pi * r**2])
    else:
        surface_mod.check_watertight(body)
        surface_mod.check_inside_box(body, grid.lo, grid.hi)
        kind = "surface"
        normals = surface_mod.normals(body.points, body.elements)
        areas = surface_mod.areas(body.points, body.elements)
    csurfs = overlaps(body, grid)
    candidates = stencil(body, grid, radius)
    cell_type = classify(body, grid, csurfs)
    oc = outside_corner(body, grid, cell_type, csurfs)
    frac = volume_fraction(body, grid, cell_type, csurfs, oc, sub=sub, seed=seed)
    # a rebuild must not see old cut volumes
    uniform = dataclasses.replace(grid, volume=jnp.zeros(0, jnp.float64))
    volume = frac * np.asarray(cell_volume(uniform), dtype=np.float64)
    regions = split_regions(body, grid, cell_type, csurfs, oc, seed=seed)
    n_sub, split_of, table, sub_parent, volume = split_tables(regions, volume, volume.size)
    if not regions:
        split_of = np.zeros(0, np.int32)
        table = np.zeros((0, 1), np.int32)
    return Mesh(
        body=body,
        cell_type=jnp.asarray(cell_type),
        volume=jnp.asarray(volume),
        candidates=jnp.asarray(candidates),
        csurfs=jnp.asarray(csurfs),
        outside_corner=jnp.asarray(oc),
        normals=jnp.asarray(normals),
        areas=jnp.asarray(areas),
        radius=int(radius),
        kind=kind,
        n_sub=int(n_sub),
        split_of=jnp.asarray(split_of),
        split_table=jnp.asarray(table),
        sub_parent=jnp.asarray(sub_parent),
    )
