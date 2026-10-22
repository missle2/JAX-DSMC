"""Surface-aware ``first_hit``: box faces, polygonal elements and analytic bodies (JAX).

This is the event query of the stencil tracker.  A particle's candidate elements are
the padded per-cell list ``mesh.candidates[cell_of(x)]`` (elements whose bounding box
meets the cell dilated by ``mesh.radius`` cells); :func:`first_hit` tests all of them
at once, together with the ``2D`` box faces of :func:`jax_dsmc.geometry.first_hit`, and
returns the earliest.  Every public function is pure and jit-able with static shapes.

Reference: SPARTA ``geometry.cpp`` (``line_line_intersect``, ``line_tri_intersect``),
``update.cpp:1049-1200`` (surface check loop, ``exclude``), ``grid_surf.cpp``
(``outside_surfs``).  Two deliberate departures from SPARTA:

* **Ray, not segment.**  SPARTA clips the move to the cell and tests the segment
  ``x -> xnew``; the stencil tracker asks for the hit *time* along the ray and lets the
  bounce loop compare it with the time budget (``wall_adjoint._bounce``).
* **Lateral test by edge functions, not by the hit point.**  SPARTA locates the plane
  hit point and then tests it against the element's edges with an absolute
  ``EPSSQ = 1e-16`` slack so that a hit at a shared vertex/edge is caught by at least
  one element.  Here the lateral test is the sign of ``v x (p - x)`` (2-D) or
  ``v . ((p_i - x) x (p_j - x))`` (3-D) at the element's vertices, *inclusive* of zero.
  Two elements sharing a vertex (2-D) or an edge (3-D) evaluate the very same
  floating-point expression (up to an exact sign flip), so a watertight surface is
  watertight in floating point without any tolerance: the ray through the shared
  feature registers on at least one of the two, and both when the expression is
  exactly zero.

Tie rules (documented, tested):

* box face vs element at equal time: the **box face** wins (``t_el < t_box`` is the
  element condition).  SPARTA only reaches the surface check for the part of the move
  inside the cell, so a face and an element at the same time is a corner case either
  way; the box face is the safe choice because the face rules of ``boundary.py`` never
  push a particle into a body.
* element vs element at equal time: the **lowest element id** wins, whatever the order
  of the candidate table.
* ``exclude``: the element id (``-1`` = none) whose slot is forced to ``inf`` -- SPARTA's
  ``exclude``, the element the particle has just left, which a re-emitted particle with
  ``v . n > 0`` would not hit anyway but which a particle pushed *into* the surface
  (``v . n < 0``, ``t = 0``) would.  For an analytic body the id is ``0``.

Surface ids follow :mod:`jax_dsmc.mesh.types`: ``0 .. 2D-1`` box faces
(``2 * axis + side``), ``2D + e`` for element ``e``, ``2D`` for an analytic body.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from jax_dsmc import geometry as box_geometry
from jax_dsmc import step_axis
from jax_dsmc.grid import cell_bounds, cell_of
from jax_dsmc.mesh import analytic
from jax_dsmc.mesh import levelset as levelset_mod
from jax_dsmc.mesh.types import CUT, INSIDE, OUTSIDE, Mesh
from jax_dsmc.state import GridSpec

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Element tests.  Shapes: x, v f64[N, D]; p*, n f64[N, S, D]; results [N, S].
# ---------------------------------------------------------------------------


def _cross2(a: jax.Array, b: jax.Array) -> jax.Array:
    """Scalar 2-D cross product ``a_x b_y - a_y b_x`` over the last axis."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _lateral_lines(x: jax.Array, v: jax.Array, p1: jax.Array, p2: jax.Array) -> jax.Array:
    """Does the *line* ``x + t v`` meet the segment ``p1 p2``?  ``bool[N, S]``, inclusive.

    ``c(p) = v x (p - x)`` is the signed side of ``p`` with respect to the line; the
    segment is met when its endpoints are on opposite sides or on the line.  The
    expression at a vertex does not depend on which element it belongs to, which is
    what makes a closed polygon watertight (module docstring).
    """
    c1 = _cross2(v[:, None, :], p1 - x[:, None, :])
    c2 = _cross2(v[:, None, :], p2 - x[:, None, :])
    return ((c1 <= 0.0) & (c2 >= 0.0)) | ((c1 >= 0.0) & (c2 <= 0.0))


def _lateral_tris(
    x: jax.Array, v: jax.Array, p1: jax.Array, p2: jax.Array, p3: jax.Array
) -> jax.Array:
    """Does the *line* ``x + t v`` meet the triangle?  ``bool[N, S]``, inclusive.

    ``w_ij = v . ((p_i - x) x (p_j - x))`` is the edge function of edge ``ij`` seen from
    ``x``; the line passes through the triangle when all three have the same sign (or
    vanish).  A shared edge is traversed in opposite directions by its two triangles,
    and ``(a x b) = -(b x a)`` exactly in IEEE arithmetic, so the two edge functions are
    exact negatives: watertight without a tolerance.
    """
    q1 = p1 - x[:, None, :]
    q2 = p2 - x[:, None, :]
    q3 = p3 - x[:, None, :]
    vv = v[:, None, :]
    w1 = jnp.sum(vv * jnp.cross(q1, q2), axis=-1)
    w2 = jnp.sum(vv * jnp.cross(q2, q3), axis=-1)
    w3 = jnp.sum(vv * jnp.cross(q3, q1), axis=-1)
    non_neg = (w1 >= 0.0) & (w2 >= 0.0) & (w3 >= 0.0)
    non_pos = (w1 <= 0.0) & (w2 <= 0.0) & (w3 <= 0.0)
    return non_neg | non_pos


def _plane_time(x: jax.Array, v: jax.Array, p1: jax.Array, n: jax.Array) -> tuple:
    """Time to the element's plane from the flow side: ``(time, approaching & t >= 0)``.

    ``dist = (x - p1) . n`` is SPARTA's ``dotstart`` (positive on the flow side) and
    ``v . n < 0`` is the approach condition; ``t = dist / (-v . n)``.  ``t >= 0`` is
    inclusive: a particle exactly on the element moving into it hits at ``t = 0`` (the
    bounce loop's ``exclude`` removes the element it just left).  Padded candidates
    with NaN coordinates fail every comparison and come out as ``inf`` downstream.
    """
    dist = jnp.sum((x[:, None, :] - p1) * n, axis=-1)
    speed = jnp.sum(v[:, None, :] * n, axis=-1)
    approaching = speed < 0.0
    t = dist / jnp.where(approaching, -speed, 1.0)
    return t, approaching & (dist >= 0.0)


def hit_lines(x: jax.Array, v: jax.Array, p1: jax.Array, p2: jax.Array, n: jax.Array) -> jax.Array:
    """Ray-segment hit times, ``f64[N, S]`` (``inf`` = no hit).

    Args:
        x, v: ``f64[N, 2]`` ray origins and spatial velocities.
        p1, p2: ``f64[N, S, 2]`` segment endpoints per candidate (NaN = padding).
        n: ``f64[N, S, 2]`` unit normals into the flow (``z x (p2 - p1)``).

    A hit requires ``v . n < 0``, ``t >= 0`` and the inclusive lateral test of
    :func:`_lateral_lines` (both endpoints included).
    """
    t, ok = _plane_time(x, v, p1, n)
    return jnp.where(ok & _lateral_lines(x, v, p1, p2), t, jnp.inf)


def hit_tris(
    x: jax.Array,
    v: jax.Array,
    p1: jax.Array,
    p2: jax.Array,
    p3: jax.Array,
    n: jax.Array,
) -> jax.Array:
    """Ray-triangle hit times, ``f64[N, S]`` (``inf`` = no hit); see :func:`hit_lines`.

    ``n`` is the unit normal ``(p2 - p1) x (p3 - p1)`` into the flow; the lateral test
    is :func:`_lateral_tris` (edges included).
    """
    t, ok = _plane_time(x, v, p1, n)
    return jnp.where(ok & _lateral_tris(x, v, p1, p2, p3), t, jnp.inf)


def _segment_crosses_lines(a, b, p1, p2, n):
    """Does the segment ``a -> b`` cross segment ``p1 p2`` (either side)?  ``bool[N, S]``.

    SPARTA ``line_line_intersect`` conditions on the plane: the endpoints must not lie
    strictly on the same side, nor both on the line; the lateral test is the inclusive
    one of :func:`_lateral_lines` with the segment direction as the ray.
    """
    da = jnp.sum((a[:, None, :] - p1) * n, axis=-1)
    db = jnp.sum((b[:, None, :] - p1) * n, axis=-1)
    plane = ~((da > 0.0) & (db > 0.0)) & ~((da < 0.0) & (db < 0.0)) & ~((da == 0.0) & (db == 0.0))
    return plane & _lateral_lines(a, b - a, p1, p2)


def _segment_crosses_tris(a, b, p1, p2, p3, n):
    """Triangle version of :func:`_segment_crosses_lines` (SPARTA ``line_tri_intersect``)."""
    da = jnp.sum((a[:, None, :] - p1) * n, axis=-1)
    db = jnp.sum((b[:, None, :] - p1) * n, axis=-1)
    plane = ~((da > 0.0) & (db > 0.0)) & ~((da < 0.0) & (db < 0.0)) & ~((da == 0.0) & (db == 0.0))
    return plane & _lateral_tris(a, b - a, p1, p2, p3)


# ---------------------------------------------------------------------------
# Per-particle candidate gathers.
# ---------------------------------------------------------------------------


def _gather_elements(mesh: Mesh, ids: jax.Array) -> tuple:
    """Vertices and normals of the elements ``ids`` (``i32[N, S]``, ``-1`` = pad).

    Returns ``(valid bool[N, S], verts f64[N, S, D, D], normals f64[N, S, D])``; padded
    slots gather element 0 and are masked by ``valid`` (never by the coordinates).
    """
    valid = ids >= 0
    safe = jnp.where(valid, ids, 0)
    verts = mesh.body.points[mesh.body.elements[safe]]
    return valid, verts, mesh.normals[safe]


def _element_times(x: jax.Array, vd: jax.Array, mesh: Mesh, ids: jax.Array) -> jax.Array:
    """Hit times against the elements ``ids`` (``-1`` = pad -> ``inf``), ``f64[N, S]``."""
    valid, verts, n = _gather_elements(mesh, ids)
    if x.shape[1] == 2:
        t = hit_lines(x, vd, verts[:, :, 0], verts[:, :, 1], n)
    elif x.shape[1] == 3:
        t = hit_tris(x, vd, verts[:, :, 0], verts[:, :, 1], verts[:, :, 2], n)
    else:
        raise ValueError(f"embedded surfaces exist for D in (2, 3), got D = {x.shape[1]}")
    return jnp.where(valid, t, jnp.inf)


def _segment_crossings(a: jax.Array, b: jax.Array, mesh: Mesh, ids: jax.Array) -> jax.Array:
    """Number of elements in ``ids`` crossed by the segment ``a -> b``, ``i32[N]``."""
    valid, verts, n = _gather_elements(mesh, ids)
    if a.shape[1] == 2:
        hit = _segment_crosses_lines(a, b, verts[:, :, 0], verts[:, :, 1], n)
    elif a.shape[1] == 3:
        hit = _segment_crosses_tris(a, b, verts[:, :, 0], verts[:, :, 1], verts[:, :, 2], n)
    else:
        raise ValueError(f"embedded surfaces exist for D in (2, 3), got D = {a.shape[1]}")
    return jnp.sum(hit & valid, axis=1).astype(jnp.int32)


def cell_clearance(mesh: Mesh, grid: GridSpec) -> jax.Array:
    """``f64[C]``: a lower bound on the distance from any point of each Cartesian cell to any
    of its candidate elements (``mesh.candidates``) -- the gap between the cell's box and
    the element's bounding box, ``inf`` for a cell without candidates."""
    c0 = mesh.candidates.shape[0]  # the Cartesian cells (split sub-cells follow them)
    lower, upper = cell_bounds(grid)
    lower, upper = lower[:c0], upper[:c0]
    valid, verts, _n = _gather_elements(mesh, mesh.candidates)  # (C, S, D, D)
    e_lo = jnp.min(verts, axis=2)
    e_hi = jnp.max(verts, axis=2)
    gap = jnp.maximum(
        jnp.maximum(e_lo - upper[:, None, :], lower[:, None, :] - e_hi), 0.0
    )  # (C, S, D)
    dist = jnp.sqrt(jnp.sum(gap * gap, axis=-1))
    return jnp.min(jnp.where(valid, dist, jnp.inf), axis=1)


POLYGON_MARGIN = 1e-9
"""Relative slack of the polygon cull (:func:`first_hit`) over the round-off of a hit
time."""


def _earliest(a, b):
    """Variadic-reduce comparator of :func:`_element_hit`: the smaller time wins, ties go
    to the lower element id (whatever the order of the candidate table)."""
    ta, ia = a
    tb, ib = b
    take_a = (ta < tb) | ((ta == tb) & (ia <= ib))
    return jnp.where(take_a, ta, tb), jnp.where(take_a, ia, ib)


def _element_hit(x, vd, mesh, cand, exclude):
    """Earliest candidate element and its time, ``(t f64[N], e i32[N])`` -- the dense
    polygon search of :func:`first_hit`.

    The ``(N, S)`` hit times are reduced by ONE variadic ``lax.reduce`` over ``(t, id)``:
    a reduction is the root of its fusion on XLA:CPU, so the element tests are computed
    once (a ``min`` followed by a tie-break ``min`` recomputed them in both fusions).  The
    reduce runs on frozen inputs; the winner's time is then re-formed from its own plane
    (:func:`_plane_time`) for the tangent -- value-preserving, the primal is the reduced
    time bit for bit -- so ``t`` stays differentiable in ``x``, ``v`` and the vertices."""
    t = _element_times(*jax.lax.stop_gradient((x, vd)), jax.lax.stop_gradient(mesh), cand)
    t = jnp.where(cand == exclude[:, None], jnp.inf, t)
    big = jnp.iinfo(jnp.int32).max
    t_el, e = jax.lax.reduce(
        (t, cand), (jnp.asarray(jnp.inf, t.dtype), jnp.int32(big)), _earliest, (1,)
    )
    # the tangent of the winner's time: d (dist / -(v . n)) through x, v and its vertices
    _valid, verts, n = _gather_elements(mesh, jnp.maximum(e, 0)[:, None])
    t_diff, _ok = _plane_time(x, vd, verts[:, :, 0], n)
    t_diff = t_diff[:, 0]
    tangent = t_diff - jax.lax.stop_gradient(t_diff)
    return t_el + jnp.where(jnp.isfinite(t_el), tangent, 0.0), e


def _elements_none(args):
    x = args[0]
    return jnp.full(x.shape[0], jnp.inf), jnp.full(x.shape[0], -1, jnp.int32)


def _elements_dense(args, mesh):
    x, vd, cell, exclude, _slot = args
    return _element_hit(x, vd, mesh, mesh.candidates[cell], exclude)


def _elements_near(args, mesh, size):
    """:func:`_element_hit` of the first ``size`` compacted rows ``slot``, scattered back;
    the other rows get no element (``inf``)."""
    x, vd, cell, exclude, slot = args
    slot = slot[:size]
    n = x.shape[0]
    take = jnp.maximum(slot, 0)
    t, e = _element_hit(x[take], vd[take], mesh, mesh.candidates[cell[take]], exclude[take])
    put = jnp.where(slot >= 0, slot, n)
    return (
        jnp.full(n, jnp.inf).at[put].set(t, mode="drop"),
        jnp.full(n, -1, jnp.int32).at[put].set(e, mode="drop"),
    )


POLYGON_COMPACT = (8, 4, 2)
"""The compacted tiers of the polygon search (:func:`_polygon_hit`): ``N // 8`` rows, else
``N // 4``, else ``N // 2``, else every row.  Measured 9 % of the slots near a candidate
in pass 0 on the SPARTA circle (scratch/trk/stats.py); on the near lane of
``tracker="lane"`` (a chunk of near slots, part of it unused) about half."""


def _polygon_hit(x, vd, grid, mesh, exclude, remaining, tables=None):
    """The polygon search of :func:`first_hit`.  With ``remaining`` (a move of known
    length) only the rays that can reach a candidate within it are searched: a ray
    shorter than its cell's :func:`cell_clearance` (with ``tables``, the finer
    :func:`near_surface`) hits nothing before the move ends, and it gets ``inf`` -- the caller only acts on a surface within ``remaining``, which such
    a ray provably does not have.  Skipped / compacted (tiers :data:`POLYGON_COMPACT`) /
    dense branches as the level set's search (:func:`jax_dsmc.mesh.levelset.hit_body`);
    the candidate lists are gathered for the searched rows only."""
    cell = cell_of(x, grid)
    n = x.shape[0]
    if remaining is None or n < levelset_mod.NEAR_MIN:
        return _element_hit(x, vd, mesh, mesh.candidates[cell], exclude)
    if tables is None:
        reach = jnp.linalg.norm(vd, axis=1) * remaining
        clearance = cell_clearance(mesh, grid)[cell]
        near = jax.lax.stop_gradient(reach * (1.0 + POLYGON_MARGIN) >= clearance)
    else:
        near = near_surface(x, vd, remaining, grid, mesh, tables)
    sizes = [n // k for k in POLYGON_COMPACT]
    slot, count = levelset_mod._near_slots(near, sizes[-1])
    branch = jnp.int32(len(sizes) + 1)
    for i in reversed(range(len(sizes))):
        branch = jnp.where(count <= sizes[i], i + 1, branch)
    branch = jnp.where(count == 0, 0, branch)
    near_branches = tuple(functools.partial(_elements_near, mesh=mesh, size=m) for m in sizes)
    return jax.lax.switch(
        step_axis.tier(branch),
        (_elements_none, *near_branches, functools.partial(_elements_dense, mesh=mesh)),
        (x, vd, cell, exclude, slot),
    )


SUB_CELLS = {2: 8, 3: 2}
"""Sub-cells per axis of :func:`dense_threshold` (by dimension): the clearance is resolved
below the cell, so that a particle in a CUT cell far from its elements still takes the
dense pass (measured on the SPARTA circle: 2/3 of the near slots were in CUT cells)."""


def dense_threshold(mesh: Mesh, grid: GridSpec) -> jax.Array:
    """``f64[C, sub^D]``: per sub-cell of every Cartesian cell (:data:`SUB_CELLS`, row
    major), a path length below which no move starting there can meet the polygon or end
    in the body -- the gap from the sub-cell to its cell's candidate elements' bounding
    boxes, at most ``radius h_min`` (an element not among the candidates lies beyond the
    dilation), less a ``POLYGON_MARGIN h_min`` slack for the rounding of the sub-cell
    index; ``-1`` in an INSIDE cell (its particles were mis-tracked: they take the full
    tracker, whose inside check catches them).  The cut-free ring of SPARTA's optimised
    move, as a distance."""
    c0 = mesh.candidates.shape[0]
    d = grid.lo.shape[0]
    sub = SUB_CELLS[d]
    lower, upper = cell_bounds(grid)
    lower, upper = lower[:c0], upper[:c0]
    h = upper - lower
    h_min = jnp.min(h, axis=1)
    offsets = jnp.stack(
        jnp.meshgrid(*([jnp.arange(sub, dtype=jnp.float64)] * d), indexing="ij"), axis=-1
    ).reshape(-1, d)  # (sub^D, D), row-major
    s_lo = lower[:, None, :] + offsets[None] * (h / sub)[:, None, :]
    s_hi = s_lo + (h / sub)[:, None, :]  # (C, sub^D, D)
    valid, verts, _n = _gather_elements(mesh, mesh.candidates)  # (C, S, D, D)
    e_lo = jnp.min(verts, axis=2)[:, None]
    e_hi = jnp.max(verts, axis=2)[:, None]  # (C, 1, S, D)
    gap = jnp.maximum(
        jnp.maximum(e_lo - s_hi[:, :, None, :], s_lo[:, :, None, :] - e_hi), 0.0
    )  # (C, sub^D, S, D)
    dist = jnp.sqrt(jnp.sum(gap * gap, axis=-1))
    clearance = jnp.min(jnp.where(valid[:, None, :], dist, jnp.inf), axis=2)
    free = jnp.minimum(clearance, (mesh.radius * h_min)[:, None])
    free = free - POLYGON_MARGIN * h_min[:, None]
    return jnp.where((mesh.cell_type[:c0] == INSIDE)[:, None], -1.0, free)


def _fine_grid(table: jax.Array, n: tuple, sub: int) -> jax.Array:
    """:func:`dense_threshold` (``C, sub^D``) as the flat row-major table of the fine
    grid of ``n_a sub`` cells per axis."""
    d = len(n)
    t = table.reshape(*n, *([sub] * d))
    order = [ax for a in range(d) for ax in (a, d + a)]
    return jnp.transpose(t, order).reshape(-1)


def near_tables(mesh: Mesh, grid: GridSpec):
    """The free-path table :func:`near_surface` reads, frozen: ``(free, origin, inverse
    spacing, shape)`` of a uniform grid -- for each of its cells a path length below which
    no move starting there can meet the surface or end in the body (``-1``: always near).
    Polygon: :func:`dense_threshold` on the ``sub``-refined Cartesian grid.  Level set:
    :func:`jax_dsmc.mesh.levelset.free_path` on its lattice.  ``None`` for an analytic
    body.  Formed once per move and passed in: a table computed in the same computation
    as its gather is fused into it by XLA:CPU and recomputed per particle."""
    mesh = jax.lax.stop_gradient(mesh)
    if mesh.kind == "surface":
        d = grid.lo.shape[0]
        sub = SUB_CELLS[d]
        shape = tuple(int(m) * sub for m in grid.n)
        free = _fine_grid(dense_threshold(mesh, grid), tuple(grid.n), sub)
        spacing = (grid.hi - grid.lo) / jnp.asarray(shape, jnp.float64)
        return free, grid.lo, 1.0 / spacing, shape
    if mesh.kind == "levelset":
        ls = mesh.body
        free = levelset_mod.free_path(ls).reshape(-1)
        return free, ls.lo, jnp.full(ls.dim, 1.0 / ls.h), tuple(ls.coef.shape)
    return None


def near_surface(
    x: jax.Array,
    v: jax.Array,
    length: jax.Array,
    grid: GridSpec,
    mesh: Mesh,
    tables=None,
) -> jax.Array:
    """``bool[N]``: may a move of duration ``length`` from ``x`` meet the embedded surface
    (or start or end in the body)?  ``False`` only where it provably cannot, for ANY path
    of length ``|v| length`` from ``x`` -- so also for one folded at a specular box face;
    the dense pass of ``tracker="lane"`` moves those particles without a surface search.

    Polygon and level set: the path is shorter than the free path of the start's cell of
    :func:`near_tables` (with the :data:`POLYGON_MARGIN` slack of the search's own cull),
    one gather and no division.  Analytic body: the start's distance to the sphere
    exceeds the path.  Not differentiated (a routing decision).  ``tables``:
    :func:`near_tables`, if already formed."""
    x, v, length = jax.lax.stop_gradient((x, v, length))
    tables = near_tables(mesh, grid) if tables is None else tables
    d = x.shape[1]
    path2 = jnp.sum(v[:, :d] * v[:, :d], axis=1) * (length * length)
    if tables is not None:
        free, origin, inverse, shape = tables
        idx = jnp.floor((x - origin) * inverse).astype(jnp.int32)
        flat = jnp.zeros(x.shape[0], jnp.int32)
        for a in range(d):
            flat = flat * shape[a] + jnp.clip(idx[:, a], 0, shape[a] - 1)
        limit = free[flat]
        return (limit < 0.0) | (path2 * (1.0 + POLYGON_MARGIN) ** 2 >= limit * limit)
    body = jax.lax.stop_gradient(mesh.body)
    gap = jnp.sqrt(jnp.sum((x - body.centre) ** 2, axis=1)) - body.radius
    return gap <= jnp.sqrt(path2) * (1.0 + POLYGON_MARGIN) + POLYGON_MARGIN * body.radius


# ---------------------------------------------------------------------------
# Public queries.
# ---------------------------------------------------------------------------


def first_hit(
    x: jax.Array,
    v: jax.Array,
    grid: GridSpec,
    mesh: Mesh,
    exclude: jax.Array,
    remaining: jax.Array | None = None,
    tables=None,
) -> tuple[jax.Array, jax.Array]:
    """Earliest surface along each ray: ``(time f64[N], surface i32[N])``.

    Args:
        x: ``f64[N, D]`` positions (inside the box).
        v: ``f64[N, 3]`` velocities; the spatial part is ``v[:, :D]``.
        grid: the Cartesian grid (box faces, ``cell_of`` for the candidate list).
        mesh: cut-cell tables; ``mesh.kind`` selects polygonal / analytic / level set
            (STATIC).
        exclude: ``i32[N]`` element id to skip (``-1`` = none; ``0`` = the analytic body;
            ignored by a level set, whose re-hit rule is the sign pattern of ``phi``).
        remaining: ``f64[N]`` remaining move time; bounds the level set's root search and
            culls the polygon search to the rays that can reach a candidate within it
            (:func:`_polygon_hit`: an element beyond the move may then be reported as no
            element); analytic bodies ignore it.  ``None``: the whole ray, every row.
        tables: :func:`near_tables` of the mesh, if already formed (the polygon cull).

    Returns the box result of :func:`jax_dsmc.geometry.first_hit` unless an element or
    the analytic body is hit *strictly* earlier; ids and tie rules per the module
    docstring.  Candidates come from ``mesh.candidates[cell_of(x)]``; a particle whose
    move exceeds the stencil (see :func:`stencil_sufficient`) can miss elements outside
    it -- that is the diagnostic the tracker counts, not something this function hides.
    """
    d = x.shape[1]
    vd = v[:, :d]
    t_box, f_box = box_geometry.first_hit(x, v, grid)
    f_box = f_box.astype(jnp.int32)
    if mesh.kind == "surface":
        t_el, e = _polygon_hit(x, vd, grid, mesh, exclude, remaining, tables)
    elif mesh.kind in ("circle", "sphere"):
        t_el = analytic.hit_body(x, vd, mesh.body, exclude == 0)
        e = jnp.zeros_like(exclude)
    elif mesh.kind == "levelset":
        t_el = levelset_mod.hit_body(x, vd, mesh.body, remaining)
        e = jnp.zeros_like(exclude)
    else:
        raise ValueError(f"unknown mesh kind {mesh.kind!r}")
    element_hit = t_el < t_box
    time = jnp.where(element_hit, t_el, t_box)
    surface = jnp.where(element_hit, 2 * d + e, f_box).astype(jnp.int32)
    return time, surface


def surface_normal(mesh: Mesh, surface: jax.Array, x_hit: jax.Array, grid: GridSpec) -> jax.Array:
    """Unit normal into the flow of surface ``surface`` at ``x_hit``, ``f64[N, D]``.

    Box faces: ``+e_axis`` on side 0, ``-e_axis`` on side 1 (into the domain).  Elements:
    ``mesh.normals[e]``.  Analytic body: ``(x_hit - c) / r``.  Level set:
    ``grad phi / |grad phi|``.  Vectorised by ``jnp.where`` on the id ranges, so one call
    serves a mixed batch.
    """
    d = x_hit.shape[1]
    axis = jnp.clip(surface // 2, 0, d - 1)
    sign = jnp.where(surface % 2 == 0, 1.0, -1.0)
    n_box = sign[:, None] * (jnp.arange(d)[None, :] == axis[:, None]).astype(jnp.float64)
    e = jnp.clip(surface - 2 * d, 0, mesh.n_elements - 1)
    if mesh.kind == "surface":
        n_el = mesh.normals[e]
    elif mesh.kind == "levelset":
        n_el = levelset_mod.normal(mesh.body, x_hit)
    else:
        n_el = analytic.normal(mesh.body, x_hit)
    return jnp.where((surface >= 2 * d)[:, None], n_el, n_box)


def _cell_corner(grid: GridSpec, cell: jax.Array, corner: jax.Array) -> jax.Array:
    """Coordinates of corner ``corner`` (``0 .. 2^D - 1``, x fastest) of each cell."""
    lower, upper = cell_bounds(grid)
    lo = lower[cell]
    hi = upper[cell]
    d = lo.shape[1]
    bits = (corner[:, None] >> jnp.arange(d)[None, :]) & 1
    return jnp.where(bits == 1, hi, lo)


def _odd_crossings(args, mesh, grid):
    """Odd number of crossings from ``x`` to its cell's outside corner, ``bool[N]`` -- the
    test of :func:`inside_polygon` in a CUT cell, over every row."""
    x, cell, _slot = args
    corner = mesh.outside_corner[cell]
    target = _cell_corner(grid, cell, jnp.maximum(corner, 0))
    return _segment_crossings(x, target, mesh, mesh.csurfs[cell]) % 2 == 1


def _odd_none(args):
    return jnp.zeros(args[0].shape[0], bool)


def _odd_near(args, mesh, grid):
    """:func:`_odd_crossings` of the compacted rows ``slot``, scattered back (False
    elsewhere)."""
    x, cell, slot = args
    n = x.shape[0]
    take = jnp.maximum(slot, 0)
    odd = _odd_crossings((x[take], cell[take], slot), mesh, grid)
    return jnp.zeros(n, bool).at[jnp.where(slot >= 0, slot, n)].set(odd, mode="drop")


INSIDE_COMPACT = 8
"""The crossing test of :func:`inside_polygon` runs on the CUT-cell rows compacted into
``N // INSIDE_COMPACT`` entries (every row when more): measured 6 % of the slots on the
SPARTA circle (scratch/trk/stats.py)."""


def inside_polygon(
    x: jax.Array, cell: jax.Array, mesh: Mesh, grid: GridSpec, active: jax.Array | None = None
) -> jax.Array:
    """Is ``x`` inside the body?  ``bool[N]``.  SPARTA ``outside_surfs`` in JAX.

    For a polygonal body the segment from ``x`` to the cell's known-outside corner
    (``mesh.outside_corner[cell]``) is tested against the cell's own elements
    (``mesh.csurfs[cell]``); an odd number of crossings means inside.  INSIDE cells
    (``outside_corner == -1``) return True and OUTSIDE cells False without a test.  For
    an analytic body the answer is ``|x - c| < r``, for a level set ``phi(x) < 0``.
    ``active`` (``bool[N]``, optional): the rows that matter; the others come out False.

    As in SPARTA, a segment through a vertex (2-D) or edge (3-D) shared by two of the
    cell's elements counts twice; the corner choice of the cut routine is what keeps
    that measure-zero case away from the particles.

    Only the rows in CUT cells are tested: compacted (:func:`levelset._near_slots`) into
    ``N // INSIDE_COMPACT`` entries from ``levelset.NEAR_MIN`` rows, every row when more
    (a runtime ``switch``); row for row the dense test's arithmetic.
    """
    if mesh.kind == "levelset":
        inside = levelset_mod.inside(mesh.body, x)
    elif mesh.kind != "surface":
        inside = analytic.inside(mesh.body, x)
    else:
        ctype = mesh.cell_type[cell]
        n = x.shape[0]
        args = (x, cell, jnp.zeros(0, jnp.int32))
        if n < levelset_mod.NEAR_MIN:
            odd = _odd_crossings(args, mesh, grid)
        else:
            cut = ctype == CUT if active is None else (ctype == CUT) & active
            size = n // INSIDE_COMPACT
            slot, count = levelset_mod._near_slots(cut, size)
            branch = jnp.where(count == 0, 0, jnp.where(count <= size, 1, 2))
            odd = jax.lax.switch(
                step_axis.tier(branch),
                (
                    _odd_none,
                    functools.partial(_odd_near, mesh=mesh, grid=grid),
                    functools.partial(_odd_crossings, mesh=mesh, grid=grid),
                ),
                (x, cell, slot),
            )
        inside = jnp.where(ctype == INSIDE, True, jnp.where(ctype == OUTSIDE, False, odd))
    return inside if active is None else active & inside


def stencil_sufficient(
    v: jax.Array, remaining: jax.Array, grid: GridSpec, mesh: Mesh, x: jax.Array | None = None
) -> jax.Array:
    """Can the particle's remaining move be tracked within its stencil?  ``bool[N]``.

    ``|v_spatial| * remaining <= radius * h_min``: a move shorter than the dilation
    cannot leave the candidate list's coverage.  ``h_min`` is the smallest cell edge of
    the particle's cell when ``x`` is given, else of the whole grid (identical on a
    uniform grid); both come from ``grid.cell_bounds`` so a hierarchical grid changes
    only that call.  This is the diagnostic the stencil tracker counts.  For a level set
    the bound is the root search's ``reach`` (``n_sample * h_phi / 2``).
    """
    lower, upper = cell_bounds(grid)
    d = lower.shape[1]
    speed = jnp.sqrt(jnp.sum(v[:, :d] * v[:, :d], axis=1))
    if mesh.kind == "levelset":
        return speed * remaining <= mesh.body.reach
    h = upper - lower
    if x is None:
        h_min = jnp.min(h)
    else:
        h_min = jnp.min(h[cell_of(x, grid)], axis=1)
    return speed * remaining <= mesh.radius * h_min


def _segment_distance(x, p1, p2):
    """Distance from ``x`` (``N, D``) to segments ``p1 -> p2`` (``N, S, D``): ``f64[N, S]``."""
    d = p2 - p1
    t = jnp.sum((x[:, None, :] - p1) * d, axis=-1) / jnp.maximum(jnp.sum(d * d, axis=-1), 1e-300)
    t = jnp.clip(t, 0.0, 1.0)
    return jnp.linalg.norm(x[:, None, :] - (p1 + t[..., None] * d), axis=-1)


def _triangle_distance(x, p1, p2, p3):
    """Distance from ``x`` (``N, 3``) to triangles (``N, S, 3`` each): the plane distance
    where the foot lies inside, else the nearest edge (``f64[N, S]``)."""
    e1, e2 = p2 - p1, p3 - p1
    nrm = jnp.cross(e1, e2)
    nn = jnp.maximum(jnp.sum(nrm * nrm, axis=-1), 1e-300)
    rel = x[:, None, :] - p1
    dist_plane = jnp.sum(rel * nrm, axis=-1) / jnp.sqrt(nn)
    foot = rel - dist_plane[..., None] * nrm / jnp.sqrt(nn)[..., None]
    # barycentric test of the foot
    d11, d12, d22 = jnp.sum(e1 * e1, -1), jnp.sum(e1 * e2, -1), jnp.sum(e2 * e2, -1)
    f1, f2 = jnp.sum(foot * e1, -1), jnp.sum(foot * e2, -1)
    den = jnp.maximum(d11 * d22 - d12 * d12, 1e-300)
    u = (d22 * f1 - d12 * f2) / den
    w = (d11 * f2 - d12 * f1) / den
    inside = (u >= 0.0) & (w >= 0.0) & (u + w <= 1.0)
    edges = jnp.minimum(
        jnp.minimum(_segment_distance(x, p1, p2), _segment_distance(x, p2, p3)),
        _segment_distance(x, p3, p1),
    )
    return jnp.where(inside, jnp.abs(dist_plane), edges)


def nearest_normal(
    x: jax.Array, grid: GridSpec, mesh: Mesh | None, diffuse_faces=(), within: float = 1.0
) -> jax.Array:
    """Unit normal (into the flow, zero-padded to 3) of the surface nearest to each point
    among the elements of its cell's candidate list and the DIFFUSE box faces
    (``diffuse_faces``: face ids ``2*axis + side``), if it lies within ``within`` cell
    widths; ``f64[N, 3]``, zero rows otherwise.  The one-sided fit of the ensemble
    adjoint splits a cell's nodes by the sign of ``v . n`` with this ``n``: next to a
    surface the values of the particles about to hit and of those that just left differ
    by the hit's reward (measured: `scratch/mesh/probe_fit_classes.py`).
    """
    n, d = x.shape
    best = jnp.full(n, jnp.inf)
    reach = within * jnp.min((grid.hi - grid.lo) / jnp.asarray(grid.n))
    normal = jnp.zeros((n, d))
    if mesh is not None:
        if mesh.kind == "surface":
            ids = mesh.candidates[cell_of(x, grid)]
            valid, verts, normals = _gather_elements(mesh, ids)
            if d == 2:
                dist = _segment_distance(x, verts[:, :, 0], verts[:, :, 1])
            else:
                dist = _triangle_distance(x, verts[:, :, 0], verts[:, :, 1], verts[:, :, 2])
            dist = jnp.where(valid, dist, jnp.inf)
            k = jnp.argmin(dist, axis=1)
            best = jnp.min(dist, axis=1)
            normal = jnp.take_along_axis(normals, k[:, None, None], axis=1)[:, 0]
        elif mesh.kind == "levelset":
            phi, g = levelset_mod.value_and_grad(mesh.body, x)
            best = jnp.abs(phi)
            normal = g / jnp.maximum(jnp.linalg.norm(g, axis=1, keepdims=True), 1e-300)
        else:
            rel = x - mesh.body.centre
            dist = jnp.linalg.norm(rel, axis=1)
            best = jnp.abs(dist - mesh.body.radius)
            normal = rel / jnp.maximum(dist, 1e-300)[:, None]
    for f in diffuse_faces:
        axis, side = f // 2, f % 2
        dist = jnp.abs(x[:, axis] - (grid.hi[axis] if side else grid.lo[axis]))
        nf = jnp.zeros(d).at[axis].set(-1.0 if side else 1.0)
        closer = dist < best
        best = jnp.where(closer, dist, best)
        normal = jnp.where(closer[:, None], nf, normal)
    return jnp.zeros((n, 3)).at[:, :d].set(jnp.where((best <= reach)[:, None], normal, 0.0))
