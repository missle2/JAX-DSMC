"""``jax_dsmc.mesh.geometry`` / ``analytic`` / ``sampling.diffuse_velocity_general``.

The polygonal hit tests are checked against a brute-force numpy reference written with
the same rules (approach from the flow side, ``t >= 0``, inclusive edge-function lateral
test), and the polygon as a whole against the analytic circle it is inscribed in: the
two must agree to within the sagitta ``r (1 - cos(pi / E))`` of a chord.  The cut-cell
tables are built here with a ten-line AABB routine so that the test does not depend on
``jax_dsmc.mesh.cut``.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

jax.config.update("jax_enable_x64", True)

from jax_dsmc import geometry as box_geometry  # noqa: E402
from jax_dsmc import sampling  # noqa: E402
from jax_dsmc.constants import K_BOLTZ  # noqa: E402
from jax_dsmc.grid import cell_bounds, cell_of  # noqa: E402
from jax_dsmc.mesh import analytic  # noqa: E402
from jax_dsmc.mesh import geometry as mesh_geometry  # noqa: E402
from jax_dsmc.mesh.types import CUT, INSIDE, OUTSIDE, AnalyticBody, Mesh, Surface  # noqa: E402
from jax_dsmc.state import GridSpec  # noqa: E402

SPARTA_CIRCLE = Path("/home/aakash/sparta/sparta/examples/circle/data.circle")
CENTRE = np.array([5.0, 5.0])
RADIUS = 3.0
N_EDGES = 50
SAGITTA = RADIUS * (1.0 - math.cos(math.pi / N_EDGES))  # 5.92e-3 for r = 3, 50-gon

pytestmark = pytest.mark.skipif(not SPARTA_CIRCLE.exists(), reason="SPARTA data.circle missing")


# ---------------------------------------------------------------------------
# Host-side construction: surface, normals, AABB candidate tables, minimal Mesh.
# ---------------------------------------------------------------------------


def read_circle():
    """Parse SPARTA ``data.circle``: 50 points, 50 lines (``p1 p2`` per line)."""
    lines = SPARTA_CIRCLE.read_text().splitlines()
    ip = lines.index("Points") + 2
    points = np.array([[float(t) for t in lines[ip + i].split()[1:3]] for i in range(N_EDGES)])
    il = lines.index("Lines") + 2
    elements = np.array([[int(t) - 1 for t in lines[il + i].split()[1:3]] for i in range(N_EDGES)])
    return points, elements


def element_normals(points, elements):
    """Unit normals into the flow: ``z x (p2 - p1)`` in 2-D, ``(p2-p1) x (p3-p1)`` in 3-D."""
    p = points[elements]
    if points.shape[1] == 2:
        d = p[:, 1] - p[:, 0]
        n = np.stack([-d[:, 1], d[:, 0]], axis=1)
        areas = np.linalg.norm(d, axis=1)
    else:
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        areas = 0.5 * np.linalg.norm(n, axis=1)
    return n / np.linalg.norm(n, axis=1)[:, None], areas


def aabb_tables(points, elements, lo, hi, n, radius):
    """Padded ``i32[C, S]`` list of elements whose AABB meets each cell dilated by ``radius``."""
    lo, hi, n = np.asarray(lo), np.asarray(hi), np.asarray(n)
    dx = (hi - lo) / n
    p = points[elements]
    emin, emax = p.min(axis=1), p.max(axis=1)
    rows = []
    for idx in np.ndindex(*n):
        clo = lo + (np.array(idx) - radius) * dx
        chi = lo + (np.array(idx) + 1 + radius) * dx
        hit = np.all(emin <= chi, axis=1) & np.all(emax >= clo, axis=1)
        rows.append(np.nonzero(hit)[0])
    width = max(1, *(len(r) for r in rows))
    table = -np.ones((len(rows), width), dtype=np.int32)
    for i, r in enumerate(rows):
        table[i, : len(r)] = r
    return table


def classify_cells(csurfs, lo, hi, n, inside_fn):
    """``(cell_type, outside_corner)``: CUT where the cell has elements, else by centre."""
    lo, hi, n = np.asarray(lo), np.asarray(hi), np.asarray(n)
    dx = (hi - lo) / n
    d = len(n)
    types = np.empty(len(csurfs), dtype=np.int32)
    corners = -np.ones(len(csurfs), dtype=np.int32)
    for c, idx in enumerate(np.ndindex(*n)):
        clo = lo + np.array(idx) * dx
        if csurfs[c, 0] < 0:
            types[c] = INSIDE if inside_fn(clo + 0.5 * dx) else OUTSIDE
            corners[c] = -1 if types[c] == INSIDE else 0
            continue
        types[c] = CUT
        for k in range(2**d):
            bits = np.array([(k >> a) & 1 for a in range(d)])
            if not inside_fn(clo + bits * dx):
                corners[c] = k
                break
        if corners[c] < 0:  # AABB overlap without a corner in the flow: the proxy's INSIDE
            types[c] = INSIDE
    return types, corners


def build_mesh(points, elements, grid, radius, inside_fn):
    normals, areas = element_normals(points, elements)
    candidates = aabb_tables(points, elements, grid.lo, grid.hi, grid.n, radius)
    csurfs = aabb_tables(points, elements, grid.lo, grid.hi, grid.n, 0)
    cell_type, outside_corner = classify_cells(csurfs, grid.lo, grid.hi, grid.n, inside_fn)
    E = len(elements)
    body = Surface(
        points=jnp.asarray(points),
        elements=jnp.asarray(elements, dtype=jnp.int32),
        model=jnp.zeros(E, dtype=jnp.int32),
        group=jnp.zeros(E, dtype=jnp.int32),
    )
    return Mesh(
        body=body,
        cell_type=jnp.asarray(cell_type),
        volume=jnp.ones(len(cell_type)),
        candidates=jnp.asarray(candidates),
        csurfs=jnp.asarray(csurfs),
        outside_corner=jnp.asarray(outside_corner),
        normals=jnp.asarray(normals),
        areas=jnp.asarray(areas),
        radius=radius,
        kind="surface",
    )


def circle_inside(p):
    """Inside-or-on the circle: a corner *on* the surface must never be the reference
    point of the crossing count (vertex 0 of ``data.circle`` is a cell corner, (8, 5))."""
    return np.linalg.norm(p - CENTRE) <= RADIUS + 1e-9


def circle_grid():
    return GridSpec(lo=jnp.array([0.0, 0.0]), hi=jnp.array([10.0, 10.0]), n=(20, 20))


def circle_mesh(radius):
    points, elements = read_circle()
    return build_mesh(points, elements, circle_grid(), radius, circle_inside)


def analytic_mesh(template: Mesh, dim):
    body = AnalyticBody(centre=jnp.asarray(CENTRE[:dim]), radius=jnp.asarray(RADIUS))
    C = template.cell_type.shape[0]
    return Mesh(
        body=body,
        cell_type=template.cell_type,
        volume=template.volume,
        candidates=jnp.zeros((C, 1), dtype=jnp.int32),
        csurfs=jnp.zeros((C, 1), dtype=jnp.int32),
        outside_corner=template.outside_corner,
        normals=jnp.zeros((1, dim)),
        areas=jnp.zeros((1,)),
        radius=1,
        kind="circle" if dim == 2 else "sphere",
    )


def cube_surface():
    """A unit cube ``[0, 1]^3`` as 12 outward-oriented triangles."""
    pts = np.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)], dtype=float)
    quads = [  # outward (right-hand rule) quads, vertex ids i + 2 j + 4 k
        (0, 2, 3, 1),  # z = 0, normal -z
        (4, 5, 7, 6),  # z = 1, normal +z
        (0, 1, 5, 4),  # y = 0, normal -y
        (2, 6, 7, 3),  # y = 1, normal +y
        (0, 4, 6, 2),  # x = 0, normal -x
        (1, 3, 7, 5),  # x = 1, normal +x
    ]
    tris = []
    for a, b, c, d in quads:
        tris += [(a, b, c), (a, c, d)]
    return pts, np.array(tris)


# ---------------------------------------------------------------------------
# Brute-force numpy references (same rules as the module, written independently).
# ---------------------------------------------------------------------------


def ref_hit_lines(x, v, p1, p2, n):
    """``f64[N, E]`` ray-segment times, loops over rays and elements."""
    N, E = len(x), len(p1)
    out = np.full((N, E), np.inf)
    for i in range(N):
        for e in range(E):
            speed = v[i] @ n[e]
            dist = (x[i] - p1[e]) @ n[e]
            if not (speed < 0.0 and dist >= 0.0):
                continue
            c1 = v[i, 0] * (p1[e, 1] - x[i, 1]) - v[i, 1] * (p1[e, 0] - x[i, 0])
            c2 = v[i, 0] * (p2[e, 1] - x[i, 1]) - v[i, 1] * (p2[e, 0] - x[i, 0])
            if (c1 <= 0.0 and c2 >= 0.0) or (c1 >= 0.0 and c2 <= 0.0):
                out[i, e] = dist / -speed
    return out


def ref_hit_tris(x, v, p1, p2, p3, n):
    N, E = len(x), len(p1)
    out = np.full((N, E), np.inf)
    for i in range(N):
        for e in range(E):
            speed = v[i] @ n[e]
            dist = (x[i] - p1[e]) @ n[e]
            if not (speed < 0.0 and dist >= 0.0):
                continue
            q = [p1[e] - x[i], p2[e] - x[i], p3[e] - x[i]]
            w = [v[i] @ np.cross(q[a], q[(a + 1) % 3]) for a in range(3)]
            if all(wi >= 0.0 for wi in w) or all(wi <= 0.0 for wi in w):
                out[i, e] = dist / -speed
    return out


def broadcast(arr, N):
    return jnp.broadcast_to(jnp.asarray(arr)[None], (N, *arr.shape))


# ---------------------------------------------------------------------------
# 1. hit_lines / hit_tris vs the brute-force reference
# ---------------------------------------------------------------------------


def test_hit_lines_matches_reference():
    rng = np.random.default_rng(1)
    N = 2000
    points, elements = read_circle()
    normals, _ = element_normals(points, elements)
    p1, p2 = points[elements[:, 0]], points[elements[:, 1]]
    x = rng.uniform(0.0, 10.0, (N, 2))
    v = rng.normal(size=(N, 2)) * rng.uniform(0.1, 3.0, (N, 1))
    ref = ref_hit_lines(x, v, p1, p2, normals)
    got = np.asarray(
        jax.jit(mesh_geometry.hit_lines)(
            jnp.asarray(x),
            jnp.asarray(v),
            broadcast(p1, N),
            broadcast(p2, N),
            broadcast(normals, N),
        )
    )
    finite = np.isfinite(ref)
    assert np.array_equal(finite, np.isfinite(got))
    assert np.allclose(got[finite], ref[finite], rtol=0.0, atol=1e-12)
    assert np.array_equal(np.argmin(ref, axis=1), np.argmin(got, axis=1))
    assert finite.any(axis=1).sum() > 200  # the rays do exercise the polygon


def test_hit_tris_matches_reference():
    rng = np.random.default_rng(2)
    N, E = 2000, 30
    tri = rng.uniform(0.0, 1.0, (E, 3, 3))
    elements = np.arange(3 * E).reshape(E, 3)
    normals, _ = element_normals(tri.reshape(-1, 3), elements)
    x = rng.uniform(-0.2, 1.2, (N, 3))
    v = rng.normal(size=(N, 3))
    ref = ref_hit_tris(x, v, tri[:, 0], tri[:, 1], tri[:, 2], normals)
    got = np.asarray(
        jax.jit(mesh_geometry.hit_tris)(
            jnp.asarray(x),
            jnp.asarray(v),
            broadcast(tri[:, 0], N),
            broadcast(tri[:, 1], N),
            broadcast(tri[:, 2], N),
            broadcast(normals, N),
        )
    )
    finite = np.isfinite(ref)
    assert np.array_equal(finite, np.isfinite(got))
    assert np.allclose(got[finite], ref[finite], rtol=0.0, atol=1e-12)
    assert np.array_equal(np.argmin(ref, axis=1), np.argmin(got, axis=1))
    assert finite.any(axis=1).sum() > 200


def test_padded_candidates_are_inf():
    x = jnp.zeros((2, 2))
    v = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    p1 = jnp.array([[[1.0, -1.0], [jnp.nan, jnp.nan]], [[1.0, -1.0], [jnp.nan, jnp.nan]]])
    p2 = jnp.array([[[1.0, 1.0], [jnp.nan, jnp.nan]], [[1.0, 1.0], [jnp.nan, jnp.nan]]])
    n = jnp.array([[[-1.0, 0.0], [jnp.nan, jnp.nan]], [[-1.0, 0.0], [jnp.nan, jnp.nan]]])
    t = np.asarray(mesh_geometry.hit_lines(x, v, p1, p2, n))
    assert np.allclose(t[:, 0], 1.0) and np.all(np.isinf(t[:, 1]))


# ---------------------------------------------------------------------------
# 2. analytic circle vs the inscribed polygon
# ---------------------------------------------------------------------------


def test_analytic_circle_vs_polygon_within_sagitta():
    rng = np.random.default_rng(3)
    N = 500
    grid = circle_grid()
    poly = circle_mesh(radius=20)  # every element is a candidate of every cell
    circ = analytic_mesh(poly, 2)
    # rays from outside, aimed at the circle with impact parameter b < 0.9 r cos(pi/E):
    # both the circle and the inscribed polygon are hit
    ang = rng.uniform(0.0, 2 * np.pi, N)
    x = CENTRE + 4.5 * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    b = rng.uniform(-0.9, 0.9, N) * RADIUS * math.cos(math.pi / N_EDGES)
    to_centre = (CENTRE - x) / 4.5
    perp = np.stack([-to_centre[:, 1], to_centre[:, 0]], axis=1)
    speed = rng.uniform(0.5, 2.0, (N, 1))
    v = (to_centre * np.sqrt(4.5**2 - b[:, None] ** 2) + perp * b[:, None]) / 4.5 * speed
    v3 = np.concatenate([v, np.zeros((N, 1))], axis=1)
    exclude = jnp.full((N,), -1, dtype=jnp.int32)
    f = jax.jit(mesh_geometry.first_hit)
    t_poly, s_poly = f(jnp.asarray(x), jnp.asarray(v3), grid, poly, exclude)
    t_circ, s_circ = f(jnp.asarray(x), jnp.asarray(v3), grid, circ, exclude)
    t_poly, s_poly, t_circ, s_circ = (np.asarray(a) for a in (t_poly, s_poly, t_circ, s_circ))
    assert np.all(s_circ == 4) and np.all(s_poly >= 4)
    # exact along-ray bound through the annulus between the polygon's inscribed circle and r
    inner = RADIUS * math.cos(math.pi / N_EDGES)
    bound = (np.sqrt(RADIUS**2 - b**2) - np.sqrt(inner**2 - b**2)) / speed[:, 0]
    assert np.all(t_poly >= t_circ - 1e-12)
    assert np.all(t_poly - t_circ <= bound + 1e-12)
    # near-normal incidence (|b| < 0.3 r): the gap is the sagitta itself, within 5 %
    near = np.abs(b) < 0.3 * RADIUS
    assert near.sum() > 50
    assert np.all((t_poly - t_circ)[near] <= 1.05 * SAGITTA / speed[near, 0])
    assert np.max((t_poly - t_circ)[near] * speed[near, 0]) > 0.5 * SAGITTA  # not vacuous
    # normals: the chord normal is within pi/E of the radial direction at its hit point
    x_hit_p = x + t_poly[:, None] * v
    x_hit_c = x + t_circ[:, None] * v
    n_p = np.asarray(
        mesh_geometry.surface_normal(poly, jnp.asarray(s_poly), jnp.asarray(x_hit_p), grid)
    )
    n_c = np.asarray(
        mesh_geometry.surface_normal(circ, jnp.asarray(s_circ), jnp.asarray(x_hit_c), grid)
    )
    assert np.allclose(np.linalg.norm(n_c, axis=1), 1.0, atol=1e-12)
    cosang = np.clip(np.sum(n_p * n_c, axis=1), -1.0, 1.0)
    assert np.all(np.arccos(cosang) <= math.pi / N_EDGES + 1e-9)


def test_analytic_body_helpers():
    body = AnalyticBody(centre=jnp.array([5.0, 5.0]), radius=jnp.array(3.0))
    assert float(analytic.volume(body, 2)) == pytest.approx(math.pi * 9.0)
    assert float(analytic.area(body, 2)) == pytest.approx(2 * math.pi * 3.0)
    assert float(analytic.volume(body, 3)) == pytest.approx(4 * math.pi * 27.0 / 3)
    assert float(analytic.area(body, 3)) == pytest.approx(4 * math.pi * 9.0)
    x = jnp.array([[5.0, 5.0], [8.5, 5.0], [8.0, 5.0], [1.0, 5.0]])
    assert np.array_equal(np.asarray(analytic.inside(body, x)), [True, False, False, False])
    v = jnp.array([[1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    t = np.asarray(analytic.hit_body(x, v, body, jnp.zeros(4, dtype=bool)))
    # inside -> inf; outside approaching -> 0.5; on the surface moving in -> 0; far -> 1
    assert (
        np.isinf(t[0]) and t[1] == pytest.approx(0.5) and t[2] == 0.0 and t[3] == pytest.approx(1.0)
    )
    t = np.asarray(analytic.hit_body(x, v, body, jnp.array([False, False, True, False])))
    assert np.isinf(t[2])
    # leaving the surface: never re-hit
    t = np.asarray(analytic.hit_body(x[2:3], -v[2:3], body, jnp.zeros(1, dtype=bool)))
    assert np.isinf(t[0])


# ---------------------------------------------------------------------------
# 3. inclusive lateral test at a vertex, and the lowest-id tie rule
# ---------------------------------------------------------------------------


def test_vertex_hit_is_inclusive():
    grid = circle_grid()
    mesh = circle_mesh(radius=20)
    # vertex 0 = (8, 5) is shared by element 0 (points 2 -> 1) and element 49 (1 -> 50)
    x = jnp.array([[9.0, 5.0]])
    v = jnp.array([[-1.0, 0.0, 0.0]])
    t, s = mesh_geometry.first_hit(x, v, grid, mesh, jnp.array([-1], dtype=jnp.int32))
    assert float(t[0]) == pytest.approx(1.0, abs=1e-12)
    assert int(s[0]) in (4 + 0, 4 + 49)


def test_polygon_is_watertight_near_vertices():
    """Rays through (or within 1e-13 of) every vertex: hit iff the line crosses the polygon.

    The edge-function lateral test evaluates the same expression at a shared vertex for
    both elements, so a ray through the vertex never falls between them; a perturbed ray
    that grazes a silhouette vertex from outside is a genuine miss and must stay one.
    """
    rng = np.random.default_rng(14)
    points, elements = read_circle()
    normals, _ = element_normals(points, elements)
    N = len(points)
    ang = rng.uniform(0.0, 2 * np.pi, N)
    xs = CENTRE + 4.5 * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    exact = points - xs  # v = p_k - x: cross(v, p_k - x) is exactly zero
    perturbed = exact * (1 + 1e-13 * rng.normal(size=(N, 1))) + 1e-13 * rng.normal(size=(N, 2))
    for vs in (exact, perturbed):
        t = np.asarray(
            mesh_geometry.hit_lines(
                jnp.asarray(xs),
                jnp.asarray(vs),
                broadcast(points[elements[:, 0]], N),
                broadcast(points[elements[:, 1]], N),
                broadcast(normals, N),
            )
        )
        hit = np.isfinite(t).any(axis=1)
        c = vs[:, :1] * (points[None, :, 1] - xs[:, 1:]) - vs[:, 1:] * (
            points[None, :, 0] - xs[:, :1]
        )
        crosses = ~((c >= 0).all(axis=1) | (c <= 0).all(axis=1))
        assert np.array_equal(hit, crosses | (vs is exact))
    assert np.isfinite(t).sum(axis=1).max() <= 2


def square_mesh():
    """Unit square, clockwise so that ``z x (p2 - p1)`` points outward; elements 0..3
    are the left, top, right and bottom edges; every element is a candidate everywhere."""
    points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    elements = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    grid = GridSpec(lo=jnp.array([-1.0, -1.0]), hi=jnp.array([3.0, 3.0]), n=(4, 4))
    return grid, build_mesh(points, elements, grid, 4, lambda p: bool(np.all((p > 0) & (p < 1))))


def test_exact_tie_picks_lowest_element_id():
    grid, mesh = square_mesh()
    # from (2, 2) along (-1, -1): elements 1 (top) and 2 (right) are both hit at t = 1 exactly
    x = jnp.array([[2.0, 2.0]])
    v = jnp.array([[-1.0, -1.0, 0.0]])
    t, s = mesh_geometry.first_hit(x, v, grid, mesh, jnp.array([-1], dtype=jnp.int32))
    assert float(t[0]) == 1.0 and int(s[0]) == 4 + 1
    # reversing the candidate order does not change the winner
    flipped = Mesh(
        body=mesh.body,
        cell_type=mesh.cell_type,
        volume=mesh.volume,
        candidates=mesh.candidates[:, ::-1],
        csurfs=mesh.csurfs,
        outside_corner=mesh.outside_corner,
        normals=mesh.normals,
        areas=mesh.areas,
        radius=mesh.radius,
        kind=mesh.kind,
    )
    t, s = mesh_geometry.first_hit(x, v, grid, flipped, jnp.array([-1], dtype=jnp.int32))
    assert float(t[0]) == 1.0 and int(s[0]) == 4 + 1


# ---------------------------------------------------------------------------
# 4. exclude
# ---------------------------------------------------------------------------


def test_exclude_rule():
    grid, mesh = square_mesh()
    # element 2 is the right edge (1, 1) -> (1, 0), normal +x; (1, 0.5) is exactly on it
    x = jnp.array([[1.0, 0.5], [1.0, 0.5], [1.0, 0.5]])
    v = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    exclude = jnp.array([-1, -1, 2], dtype=jnp.int32)
    t, s = mesh_geometry.first_hit(x, v, grid, mesh, exclude)
    t, s = np.asarray(t), np.asarray(s)
    # leaving along +n: never re-hit, ends on the x-high box face
    assert s[0] == 1 and t[0] == 2.0
    # pushed into the element: hit at t = 0 ...
    assert s[1] == 4 + 2 and t[1] == 0.0
    # ... unless excluded; from inside every other element has v . n > 0 -> x-low face
    assert s[2] == 0 and t[2] == 2.0


# ---------------------------------------------------------------------------
# 5. box faces
# ---------------------------------------------------------------------------


def test_box_faces_and_empty_cells_match_box_first_hit():
    rng = np.random.default_rng(5)
    grid = circle_grid()
    mesh = circle_mesh(radius=1)
    # a ray that reaches the box before any element: from (9, 5) heading +x
    x = jnp.array([[9.0, 5.0], [5.0, 0.3]])
    v = jnp.array([[1.0, 0.1, 0.0], [0.2, -1.0, 0.0]])
    exclude = jnp.array([-1, -1], dtype=jnp.int32)
    t, s = mesh_geometry.first_hit(x, v, grid, mesh, exclude)
    t_box, s_box = box_geometry.first_hit(x, v, grid)
    assert np.array_equal(np.asarray(t), np.asarray(t_box))
    assert np.array_equal(np.asarray(s), np.asarray(s_box))
    assert int(s[0]) == 1 and int(s[1]) == 2
    # random particles in cells with no candidates at all: exactly the box result
    N = 400
    x = rng.uniform(0.0, 10.0, (N, 2))
    v3 = rng.normal(size=(N, 3))
    cells = np.asarray(cell_of(jnp.asarray(x), grid))
    empty = np.asarray(mesh.candidates)[cells, 0] < 0
    assert empty.sum() > 50
    xe, ve = jnp.asarray(x[empty]), jnp.asarray(v3[empty])
    t, s = mesh_geometry.first_hit(xe, ve, grid, mesh, jnp.full((int(empty.sum()),), -1, jnp.int32))
    t_box, s_box = box_geometry.first_hit(xe, ve, grid)
    assert np.array_equal(np.asarray(t), np.asarray(t_box))
    assert np.array_equal(np.asarray(s), np.asarray(s_box))
    # box-face normals point into the domain
    n = np.asarray(
        mesh_geometry.surface_normal(mesh, jnp.arange(4, dtype=jnp.int32), jnp.zeros((4, 2)), grid)
    )
    assert np.array_equal(n, [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


# ---------------------------------------------------------------------------
# 6. inside_polygon
# ---------------------------------------------------------------------------


def test_inside_polygon_matches_circle_outside_sagitta_band():
    rng = np.random.default_rng(6)
    grid = circle_grid()
    mesh = circle_mesh(radius=1)
    N = 5000
    x = rng.uniform(0.0, 10.0, (N, 2))
    dist = np.linalg.norm(x - CENTRE, axis=1)
    keep = np.abs(dist - RADIUS) > SAGITTA
    cell = cell_of(jnp.asarray(x), grid)
    got = np.asarray(jax.jit(mesh_geometry.inside_polygon)(jnp.asarray(x), cell, mesh, grid))
    assert np.array_equal(got[keep], dist[keep] < RADIUS)
    assert (dist < RADIUS).sum() > 1000 and (dist > RADIUS).sum() > 1000
    # the analytic body is exact everywhere
    circ = analytic_mesh(mesh, 2)
    got_c = np.asarray(mesh_geometry.inside_polygon(jnp.asarray(x), cell, circ, grid))
    assert np.array_equal(got_c, dist < RADIUS)


def test_inside_polygon_short_circuits_on_cell_type():
    grid = circle_grid()
    mesh = circle_mesh(radius=1)
    # an OUTSIDE cell relabelled INSIDE (and vice versa) flips the answer without a test
    cell_type = np.asarray(mesh.cell_type).copy()
    outside_cell = int(np.nonzero(cell_type == OUTSIDE)[0][0])
    inside_cell = int(np.nonzero(cell_type == INSIDE)[0][0])
    cell_type[outside_cell], cell_type[inside_cell] = INSIDE, OUTSIDE
    forged = Mesh(
        body=mesh.body,
        cell_type=jnp.asarray(cell_type),
        volume=mesh.volume,
        candidates=mesh.candidates,
        csurfs=mesh.csurfs,
        outside_corner=mesh.outside_corner,
        normals=mesh.normals,
        areas=mesh.areas,
        radius=mesh.radius,
        kind=mesh.kind,
    )
    lower, upper = cell_bounds(grid)
    x = 0.5 * (lower + upper)[jnp.array([outside_cell, inside_cell])]
    cell = jnp.array([outside_cell, inside_cell], dtype=jnp.int32)
    assert np.array_equal(
        np.asarray(mesh_geometry.inside_polygon(x, cell, mesh, grid)), [False, True]
    )
    assert np.array_equal(
        np.asarray(mesh_geometry.inside_polygon(x, cell, forged, grid)), [True, False]
    )


def test_inside_polygon_3d_cube():
    rng = np.random.default_rng(7)
    points, elements = cube_surface()
    # grid faces at -0.1 + 0.5 k: no cell corner lies on a cube face
    grid = GridSpec(lo=jnp.array([-1.1, -1.1, -1.1]), hi=jnp.array([1.9, 1.9, 1.9]), n=(6, 6, 6))
    mesh = build_mesh(points, elements, grid, 1, lambda p: bool(np.all((p > 0) & (p < 1))))
    normals = np.asarray(mesh.normals)
    centres = np.asarray(mesh.body.points)[np.asarray(mesh.body.elements)].mean(axis=1)
    assert np.all(np.sum(normals * (centres - 0.5), axis=1) > 0)  # outward
    N = 3000
    x = rng.uniform(-1.1, 1.9, (N, 3))
    cell = cell_of(jnp.asarray(x), grid)
    got = np.asarray(jax.jit(mesh_geometry.inside_polygon)(jnp.asarray(x), cell, mesh, grid))
    assert np.array_equal(got, np.all((x > 0) & (x < 1), axis=1))
    # and first_hit on the cube: a ray at the +x face from outside
    xr = jnp.array([[1.5, 0.3, 0.6], [0.5, 0.5, 0.5]])
    vr = jnp.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    t, s = mesh_geometry.first_hit(xr, vr, grid, mesh, jnp.array([-1, -1], dtype=jnp.int32))
    assert float(t[0]) == pytest.approx(0.5) and int(s[0]) in (6 + 10, 6 + 11)
    assert int(s[1]) < 6  # from inside the cube, no element is hit from the flow side
    n = np.asarray(mesh_geometry.surface_normal(mesh, s, xr + t[:, None] * vr, grid))
    assert np.allclose(n[0], [1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# 7. stencil_sufficient
# ---------------------------------------------------------------------------


def test_stencil_sufficient():
    grid = circle_grid()  # h = 0.5
    mesh = circle_mesh(radius=1)
    v = jnp.array([[1.0, 0.0, 5.0], [3.0, 4.0, 0.0], [3.0, 4.0, 0.0]])
    remaining = jnp.array([0.4, 0.1, 0.11])
    ok = np.asarray(mesh_geometry.stencil_sufficient(v, remaining, grid, mesh))
    assert np.array_equal(ok, [True, True, False])
    x = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    ok = np.asarray(mesh_geometry.stencil_sufficient(v, remaining, grid, mesh, x))
    assert np.array_equal(ok, [True, True, False])


# ---------------------------------------------------------------------------
# 8. diffuse_velocity_general
# ---------------------------------------------------------------------------

MASS = 6.63e-26
T_WALL = 300.0


def test_diffuse_general_matches_axis_kernel_law():
    N = 20000
    z, r = sampling.standard_diffuse_draws(jax.random.PRNGKey(8), N)
    sigma = math.sqrt(K_BOLTZ * T_WALL / MASS)
    vwall = jnp.array([0.0, 50.0, 0.0])
    ref = np.asarray(sampling.diffuse_velocity_from_draws(z, r, T_WALL, MASS, 0, 1.0, vwall))
    normal = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), (N, 3))
    v_in = jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0]), (N, 3))
    got = np.asarray(
        jax.jit(sampling.diffuse_velocity_general)(z, r, normal, T_WALL, MASS, vwall, v_in)
    )
    # same law, component by component: means and variances within 3 sigma of each other
    for k in range(3):
        a, b = ref[:, k], got[:, k]
        se_mean = math.sqrt(a.var() / N + b.var() / N)
        assert abs(a.mean() - b.mean()) < 3 * se_mean
        se_var = math.sqrt((np.var((a - a.mean()) ** 2) + np.var((b - b.mean()) ** 2)) / N)
        assert abs(a.var() - b.var()) < 3 * se_var
    # normal component: Rayleigh with scale sigma; tangentials: N(vwall, sigma^2)
    assert stats.kstest(got[:, 0], stats.rayleigh(scale=sigma).cdf).pvalue > 1e-3
    assert abs(got[:, 0].mean() - sigma * math.sqrt(math.pi / 2)) < 3 * sigma * math.sqrt(
        (2 - math.pi / 2) / N
    )
    assert abs(got[:, 1].mean() - 50.0) < 3 * sigma / math.sqrt(N)
    assert abs(got[:, 2].var() - sigma**2) < 3 * sigma**2 * math.sqrt(2.0 / N)
    # energy: mean |v - vwall|^2 = 4 sigma^2 (2 kT per emitted particle)
    e = np.sum((got - np.asarray(vwall)) ** 2, axis=1)
    assert abs(e.mean() - 4 * sigma**2) < 3 * e.std() / math.sqrt(N)
    # cosine law: E[cos theta] = 2/3 for a diffuse emitter
    cos = got[:, 0] / np.linalg.norm(got - np.asarray(vwall), axis=1)
    assert abs(cos.mean() - 2.0 / 3.0) < 3 * cos.std() / math.sqrt(N)


def test_diffuse_general_tilted_normal():
    N = 20000
    z, r = sampling.standard_diffuse_draws(jax.random.PRNGKey(9), N)
    sigma = math.sqrt(K_BOLTZ * T_WALL / MASS)
    n = np.array([1.0, 2.0, -0.5])
    n /= np.linalg.norm(n)
    normal = jnp.broadcast_to(jnp.asarray(n), (N, 3))
    vwall = jnp.asarray(np.cross(n, [0.0, 0.0, 1.0]) * 30.0)  # tangential
    v_in = jax.random.normal(jax.random.PRNGKey(10), (N, 3)) * sigma
    v_in = v_in - jnp.abs(v_in @ jnp.asarray(n))[:, None] * jnp.asarray(n)  # incident: v . n < 0
    got = np.asarray(sampling.diffuse_velocity_general(z, r, normal, T_WALL, MASS, vwall, v_in))
    assert np.all(got @ n > 0.0)
    # same draws in an orthonormal frame: the speed relative to the wall is frame-independent
    speed_axis = np.sqrt(
        sigma**2 * (np.asarray(r) ** 2 + np.asarray(z[:, 0]) ** 2 + np.asarray(z[:, 1]) ** 2)
    )
    assert np.allclose(np.linalg.norm(got - np.asarray(vwall), axis=1), speed_axis, rtol=1e-10)
    assert stats.kstest(got @ n, stats.rayleigh(scale=sigma).cdf).pvalue > 1e-3
    # normal incidence (tangential part of v_in vanishes) is handled without NaN
    v_norm = -jnp.broadcast_to(jnp.asarray(n), (N, 3)) * 100.0
    got = np.asarray(sampling.diffuse_velocity_general(z, r, normal, T_WALL, MASS, vwall, v_norm))
    assert np.all(np.isfinite(got)) and np.all(got @ n > 0.0)
    got0 = np.asarray(
        sampling.diffuse_velocity_general(z, r, normal, T_WALL, MASS, vwall, jnp.zeros((N, 3)))
    )
    assert np.all(np.isfinite(got0)) and np.all(got0 @ n > 0.0)
    assert abs(np.var(got0 @ n) - sigma**2 * (2 - math.pi / 2)) < 3 * sigma**2 * math.sqrt(2.0 / N)


def test_axis_kernel_unchanged():
    """``diffuse_velocity_from_draws`` is the reference law; pin its closed form."""
    z, r = sampling.standard_diffuse_draws(jax.random.PRNGKey(11), 16)
    sigma = math.sqrt(K_BOLTZ * T_WALL / MASS)
    got = np.asarray(
        sampling.diffuse_velocity_from_draws(z, r, T_WALL, MASS, 1, -1.0, jnp.zeros(3))
    )
    expect = sigma * np.asarray(z)
    expect[:, 1] = -sigma * np.asarray(r)
    assert np.array_equal(got, expect)


# ---------------------------------------------------------------------------
# 9. jit: one compile, no recompilation on the second call
# ---------------------------------------------------------------------------


def test_everything_jits_once():
    rng = np.random.default_rng(12)
    grid = circle_grid()
    mesh = circle_mesh(radius=1)
    N = 256
    x = jnp.asarray(rng.uniform(0.0, 10.0, (N, 2)))
    v = jnp.asarray(rng.normal(size=(N, 3)))
    exclude = jnp.full((N,), -1, dtype=jnp.int32)
    cell = cell_of(x, grid)
    remaining = jnp.full((N,), 0.1)
    z, r = sampling.standard_diffuse_draws(jax.random.PRNGKey(13), N)
    normal = jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0]), (N, 3))
    cases = [
        (mesh_geometry.first_hit, (x, v, grid, mesh, exclude)),
        (mesh_geometry.surface_normal, (mesh, exclude + 5, x, grid)),
        (mesh_geometry.inside_polygon, (x, cell, mesh, grid)),
        (mesh_geometry.stencil_sufficient, (v, remaining, grid, mesh)),
        (
            analytic.hit_body,
            (x, v, AnalyticBody(jnp.asarray(CENTRE), jnp.asarray(RADIUS)), exclude == 0),
        ),
        (sampling.diffuse_velocity_general, (z, r, normal, T_WALL, MASS, jnp.zeros(3), v)),
    ]
    t0 = time.perf_counter()
    for fn, args in cases:
        jitted = jax.jit(fn)
        first = jax.tree_util.tree_map(np.asarray, jitted(*args))
        compiled = jitted._cache_size()  # shared with earlier jax.jit(fn) wrappers
        second = jax.tree_util.tree_map(np.asarray, jitted(*args))
        assert jitted._cache_size() == compiled  # the second call did not recompile
        for a, b in zip(
            jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second), strict=True
        ):
            assert np.array_equal(a, b)
    assert time.perf_counter() - t0 < 60.0
