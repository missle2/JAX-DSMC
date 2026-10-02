"""Polygonal surfaces: SPARTA file reader, generators, normals, checks, volumes.

Everything here is host-side set-up.  Inputs are numpy (or anything ``np.asarray``
accepts); the returned :class:`~jax_dsmc.mesh.types.Surface` carries ``jnp`` arrays, and
:func:`normals` / :func:`areas` / :func:`centroids` / :func:`transform` are ``jnp`` so that
a shape deformation of ``points`` is differentiable through them.

Orientation (fixed in :mod:`jax_dsmc.mesh.types`; SPARTA ``surf.cpp:875-926``): an element's
normal points **into the flow**, i.e. *out of* the body.

* 2-D: ``n = z x (p2 - p1) = (-dy, dx) / |d|``.  A loop must therefore be traversed
  **clockwise** for a body whose interior is solid; SPARTA's ``data.circle`` is (points
  listed counter-clockwise, line ``k`` running from point ``k + 1`` to point ``k``) and
  :func:`circle` reproduces exactly that construction.
* 3-D: ``n = (p2 - p1) x (p3 - p1)`` normalised, i.e. counter-clockwise seen from the flow.

:func:`body_volume` uses the divergence theorem ``V = (1/D) * sum_e (p1_e . n_e) A_e``; it is
positive for a body oriented as above, and *negative* for an inside-out surface, which
makes it a cheap orientation check.  In 2-D this is ``sum (x2 y1 - x1 y2) / 2``, i.e. the
shoelace sum with the sign that makes SPARTA's clockwise loops positive.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc.mesh.types import Surface

jax.config.update("jax_enable_x64", True)


# --- construction ---------------------------------------------------------------


def make_surface(points, elements, model=None, group=None) -> Surface:
    """Wrap host arrays as a :class:`Surface` (``model`` / ``group`` default to zeros)."""
    points = np.asarray(points, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int32)
    D = points.shape[1]
    if D not in (2, 3):
        raise NotImplementedError(f"surfaces need D = 2 or 3, got D = {D}")
    if elements.ndim != 2 or elements.shape[1] != D:
        raise ValueError(f"elements must be (E, {D}) for D = {D}, got {elements.shape}")
    if elements.size and (elements.min() < 0 or elements.max() >= points.shape[0]):
        raise ValueError("element vertex index out of range")
    E = elements.shape[0]
    model = np.zeros(E, np.int32) if model is None else np.asarray(model, np.int32)
    group = np.zeros(E, np.int32) if group is None else np.asarray(group, np.int32)
    return Surface(
        points=jnp.asarray(points),
        elements=jnp.asarray(elements),
        model=jnp.asarray(model),
        group=jnp.asarray(group),
    )


def _strip(line: str) -> str:
    """SPARTA's rule: everything from ``#`` onward is a comment."""
    return line.split("#", 1)[0].strip()


def _read_section(lines, start, count, n_expected, name):
    """Parse ``count`` rows of ``id [type] v1 .. vn`` starting at ``lines[start]``.

    A row with one integer more than ``1 + n_expected`` words carries SPARTA's optional
    ``type`` column between the id and the values.  Returns ``(ids, types, values)`` and the
    index of the line after the section.
    """
    ids = np.empty(count, np.int64)
    types = np.zeros(count, np.int64)
    values = []
    i = start
    got = 0
    while got < count:
        if i >= len(lines):
            raise ValueError(f"surf file: unexpected end of file in {name} section")
        row = _strip(lines[i])
        i += 1
        if not row:
            continue
        words = row.split()
        if len(words) == 1 + n_expected:
            ids[got] = int(words[0])
            values.append(words[1:])
        elif len(words) == 2 + n_expected:
            ids[got] = int(words[0])
            types[got] = int(words[1])
            values.append(words[2:])
        else:
            raise ValueError(
                f"surf file: {name} row {got + 1} has {len(words)} words, expected "
                f"{1 + n_expected} or {2 + n_expected}: {row!r}"
            )
        got += 1
    return ids, types, values, i


def read_sparta_surf(path: str, dim: int) -> Surface:  # noqa: PLR0912, PLR0915 -- one file parser
    """Read a SPARTA ``read_surf`` data file (``read_surf.cpp:752-1200``).

    Format: the first line is a comment and skipped; header lines ``N points`` and
    ``N lines`` (2-D) / ``N triangles`` (3-D); then the sections ``Points`` and
    ``Lines`` / ``Triangles``, each followed by a blank line and ``N`` rows
    ``id x y [z]`` / ``id [type] p1 p2 [p3]`` with 1-based ids.  Rows need not be in id
    order; the optional ``type`` column is stored as ``group`` (0-based: ``type - 1``, as
    SPARTA's default type is 1).  ``model`` is zeros.
    """
    if dim not in (2, 3):
        raise NotImplementedError(f"read_sparta_surf: dim must be 2 or 3, got {dim}")
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    if not lines:
        raise ValueError(f"surf file {path}: empty")
    n_points = n_elems = None
    elem_word = "lines" if dim == 2 else "triangles"
    i = 1  # the first line is always a comment
    while i < len(lines):
        row = _strip(lines[i])
        i += 1
        if not row:
            continue
        low = row.lower()
        words = low.split()
        if not words[0].lstrip("-").isdigit():  # a section keyword ends the header
            i -= 1
            break
        if "points" in low:
            n_points = int(words[0])
        elif elem_word in low:
            n_elems = int(words[0])
        elif "lines" in low or "triangles" in low:
            raise ValueError(f"surf file {path}: header {row!r} is not for dim = {dim}")
        else:
            raise ValueError(f"surf file {path}: unknown header line {row!r}")
    if not n_points or not n_elems:
        raise ValueError(f"surf file {path}: header lacks 'N points' / 'N {elem_word}'")

    points = np.full((n_points, dim), np.nan)
    elements = np.full((n_elems, dim), -1, np.int64)
    types = np.zeros(n_elems, np.int64)
    seen = set()
    while i < len(lines):
        row = _strip(lines[i])
        i += 1
        if not row:
            continue
        key = row.lower()
        if key == "points":
            ids, _, vals, i = _read_section(lines, i, n_points, dim, "Points")
            if ids.min() < 1 or ids.max() > n_points or len(set(ids)) != n_points:
                raise ValueError(f"surf file {path}: Points ids are not a permutation of 1..N")
            points[ids - 1] = np.asarray(vals, dtype=np.float64)
        elif key in ("lines", "triangles"):
            if key != elem_word:
                raise ValueError(f"surf file {path}: {row} section for dim = {dim}")
            ids, types_, vals, i = _read_section(lines, i, n_elems, dim, row)
            if ids.min() < 1 or ids.max() > n_elems or len(set(ids)) != n_elems:
                raise ValueError(f"surf file {path}: {row} ids are not a permutation of 1..N")
            elements[ids - 1] = np.asarray(vals, dtype=np.int64) - 1
            types[ids - 1] = types_
        else:
            raise ValueError(f"surf file {path}: unknown section {row!r}")
        seen.add(key)
    if seen != {"points", elem_word}:
        raise ValueError(f"surf file {path}: missing section(s), found {sorted(seen)}")
    if np.isnan(points).any() or (elements < 0).any():
        raise ValueError(f"surf file {path}: incomplete Points / {elem_word} section")
    if (elements >= n_points).any():
        raise ValueError(f"surf file {path}: vertex index exceeds the number of points")
    group = np.where(types > 0, types - 1, 0)
    return make_surface(points, elements, group=group)


def circle(radius: float, n: int, centre=(0.0, 0.0)) -> Surface:
    """A regular ``n``-gon inscribed in the circle, oriented as SPARTA's ``data.circle``.

    Point ``k`` sits at angle ``2 pi k / n`` (the first at angle 0, counter-clockwise) and
    line ``k`` runs from point ``k + 1`` to point ``k``, so that ``z x (p2 - p1)`` points
    outward, into the flow.
    """
    if n < 3:
        raise ValueError("circle: need at least 3 points")
    c = np.asarray(centre, dtype=np.float64)
    theta = 2.0 * np.pi * np.arange(n) / n
    points = c[None, :] + radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    k = np.arange(n)
    elements = np.stack([(k + 1) % n, k], axis=1)
    return make_surface(points, elements)


def _midpoint_index(cache: dict, points: list, i: int, j: int) -> int:
    """Index of the unit-sphere midpoint of vertices ``i``/``j``, created on first use."""
    key = (min(i, j), max(i, j))
    if key not in cache:
        m = 0.5 * (points[i] + points[j])
        points.append(m / np.linalg.norm(m))
        cache[key] = len(points) - 1
    return cache[key]


def sphere(radius: float, level: int, centre=(0.0, 0.0, 0.0)) -> Surface:
    """A geodesic sphere: an icosahedron subdivided ``level`` times (``20 * 4**level``
    triangles), vertices projected onto the sphere, triangles oriented outward."""
    if level < 0:
        raise ValueError("sphere: level must be >= 0")
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    raw = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    points = list(raw / np.linalg.norm(raw, axis=1, keepdims=True))
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    for _ in range(level):
        cache: dict = {}
        new_faces = []
        for a, b, c in faces:
            ab = _midpoint_index(cache, points, a, b)
            bc = _midpoint_index(cache, points, b, c)
            ca = _midpoint_index(cache, points, c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    pts = np.asarray(points)
    tri = np.asarray(faces, dtype=np.int64)
    # orient outward: flip a triangle whose (p2-p1) x (p3-p1) points inward
    n = np.cross(pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]])
    inward = np.einsum("ij,ij->i", n, pts[tri].mean(axis=1)) < 0
    tri[inward] = tri[inward][:, [0, 2, 1]]
    c = np.asarray(centre, dtype=np.float64)
    return make_surface(c[None, :] + radius * pts, tri)


# --- element geometry (jnp, differentiable in ``points``) --------------------------------


def _edge_cross(points, elements):
    """Un-normalised outward normal: 2-D ``z x (p2 - p1)``, 3-D ``(p2 - p1) x (p3 - p1)``."""
    v = points[elements]  # [E, D, D]
    if v.shape[-1] == 2:
        d = v[:, 1] - v[:, 0]
        return jnp.stack([-d[:, 1], d[:, 0]], axis=1)
    return jnp.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])


def normals(points, elements) -> jnp.ndarray:
    """Unit normals ``f64[E, D]`` pointing into the flow (see the module docstring)."""
    n = _edge_cross(jnp.asarray(points), jnp.asarray(elements))
    return n / jnp.linalg.norm(n, axis=1, keepdims=True)


def areas(points, elements) -> jnp.ndarray:
    """``f64[E]``: segment length in 2-D, triangle area in 3-D."""
    n = _edge_cross(jnp.asarray(points), jnp.asarray(elements))
    length = jnp.linalg.norm(n, axis=1)
    return length if n.shape[1] == 2 else 0.5 * length


def centroids(points, elements) -> jnp.ndarray:
    """``f64[E, D]`` element centroids (vertex mean)."""
    return jnp.mean(jnp.asarray(points)[jnp.asarray(elements)], axis=1)


def transform(points, scale=1.0, translate=0.0, origin=None) -> jnp.ndarray:
    """Scale ``points`` about ``origin`` (default: their centroid), then translate.

    ``scale`` may be a scalar or a per-axis ``f64[D]``.  Differentiable in everything.
    """
    points = jnp.asarray(points, dtype=jnp.float64)
    if origin is None:
        origin = jnp.mean(points, axis=0)
    origin = jnp.asarray(origin, dtype=jnp.float64)
    return origin + jnp.asarray(scale) * (points - origin) + jnp.asarray(translate)


def body_volume(surface: Surface) -> float:
    """Exact volume of the body enclosed by the surface (area in 2-D).

    Divergence theorem ``V = (1/D) sum_e (p1_e . n_e) A_e``: 2-D ``sum (x2 y1 - x1 y2)/2``,
    3-D ``sum p1 . ((p2 - p1) x (p3 - p1)) / 6``.  Positive for a closed surface whose
    normals point into the flow; a negative result means the surface is inside out.
    """
    p = np.asarray(surface.points, dtype=np.float64)
    e = np.asarray(surface.elements)
    v = p[e]
    if p.shape[1] == 2:
        return float(0.5 * np.sum(v[:, 1, 0] * v[:, 0, 1] - v[:, 0, 0] * v[:, 1, 1]))
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    return float(np.sum(np.einsum("ij,ij->i", v[:, 0], n)) / 6.0)


# --- checks -------------------------------------------------------------------------


def check_watertight(surface: Surface) -> None:
    """Raise ``ValueError`` unless the surface is closed and consistently oriented.

    2-D: every point is ``p1`` of exactly one line and ``p2`` of exactly one line.
    3-D: every directed edge occurs exactly once and so does its reverse (each undirected
    edge is shared by exactly two triangles that traverse it in opposite directions).
    """
    p = np.asarray(surface.points)
    e = np.asarray(surface.elements).astype(np.int64)
    P, D = p.shape
    if e.shape[0] == 0:
        raise ValueError("surface has no elements")
    if (e[:, :, None] == e[:, None, :]).sum() > e.shape[0] * D:
        bad = np.nonzero((e[:, :, None] == e[:, None, :]).sum(axis=(1, 2)) > D)[0]
        raise ValueError(f"degenerate elements (repeated vertex): {bad[:10].tolist()}")
    if D == 2:
        c1 = np.bincount(e[:, 0], minlength=P)
        c2 = np.bincount(e[:, 1], minlength=P)
        bad = np.nonzero((c1 != 1) | (c2 != 1))[0]
        if bad.size:
            detail = ", ".join(f"point {i}: p1 of {c1[i]}, p2 of {c2[i]} lines" for i in bad[:8])
            raise ValueError(f"2-D surface not watertight ({bad.size} bad points): {detail}")
        return
    edges = np.concatenate([e[:, [0, 1]], e[:, [1, 2]], e[:, [2, 0]]], axis=0)
    key = edges[:, 0] * P + edges[:, 1]
    rev = edges[:, 1] * P + edges[:, 0]
    uniq, counts = np.unique(key, return_counts=True)
    dup = uniq[counts > 1]
    if dup.size:
        detail = [(int(k // P), int(k % P)) for k in dup[:8]]
        raise ValueError(f"3-D surface: {dup.size} directed edges occur more than once: {detail}")
    missing = np.setdiff1d(rev, key)
    if missing.size:
        detail = [(int(k // P), int(k % P)) for k in missing[:8]]
        raise ValueError(
            f"3-D surface not watertight / inconsistently oriented: {missing.size} edges "
            f"lack their reverse (edge = (from, to) as it appears in one triangle): {detail}"
        )
    unused = np.setdiff1d(np.arange(P), e.ravel())
    if unused.size:
        raise ValueError(f"3-D surface: {unused.size} points unused: {unused[:8].tolist()}")


def check_inside_box(surface: Surface, lo, hi) -> None:
    """Raise ``ValueError`` if any point lies outside the closed box ``[lo, hi]``."""
    p = np.asarray(surface.points)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    bad = np.nonzero(((p < lo) | (p > hi)).any(axis=1))[0]
    if bad.size:
        raise ValueError(
            f"{bad.size} surface points outside the box {lo.tolist()}..{hi.tolist()}: "
            f"first is point {bad[0]} at {p[bad[0]].tolist()}"
        )
