"""Host-side cut-cell tables (`jax_dsmc.mesh.cut`, `jax_dsmc.mesh.surface`).

References: SPARTA's shipped `circle` / `sphere` examples (`examples/mesh/sparta/`) and
the per-cell flow volumes dumped by the rebuilt argon runs
(`examples/mesh/results/sparta/grid_vol.*`).  Everything is numpy; JAX only for `cell_of`.
"""

from __future__ import annotations

import math
import os
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from jax_dsmc.grid import cell_of, cell_volume
from jax_dsmc.mesh import cut
from jax_dsmc.mesh import surface as S
from jax_dsmc.mesh.types import CUT, INSIDE, OUTSIDE, AnalyticBody
from jax_dsmc.state import GridSpec

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPARTA = os.path.join(ROOT, "examples", "mesh", "sparta")
RESULTS = os.path.join(ROOT, "examples", "mesh", "results", "sparta")
sys.path.insert(0, os.path.join(ROOT, "examples", "mesh"))

from sparta_io import read_grid_dump  # noqa: E402


def _grid(lo, hi, n):
    return GridSpec(lo=jnp.array(lo, dtype=jnp.float64), hi=jnp.array(hi, dtype=jnp.float64), n=n)


GRID2 = _grid([0.0, 0.0], [10.0, 10.0], (20, 20))
GRID3 = _grid([-2.0] * 3, [2.0] * 3, (20, 20, 20))


def _counts(cell_type):
    return tuple(int((np.asarray(cell_type) == k).sum()) for k in (OUTSIDE, INSIDE, CUT))


@pytest.fixture(scope="module")
def circle_surface():
    return S.read_sparta_surf(os.path.join(SPARTA, "data.circle"), 2)


@pytest.fixture(scope="module")
def sphere_surface():
    return S.read_sparta_surf(os.path.join(SPARTA, "data.sphere"), 3)


@pytest.fixture(scope="module")
def circle_mesh(circle_surface):
    return cut.build_mesh(circle_surface, GRID2)


def _sparta_volumes(path, grid):
    """Per-cell flow volume from a `dump grid ... vol` file, indexed by our flat cell id."""
    d = read_grid_dump(path, snapshot=0)
    D = len(grid.n)
    cols = [d["xc"], d["yc"]] + ([d["zc"]] if D == 3 else [])
    cid = np.asarray(cell_of(jnp.asarray(np.stack(cols, axis=1)), grid))
    ref = np.zeros(int(np.prod(grid.n)))
    np.add.at(ref, cid, d["vol"])  # split cells would repeat an id; SPARTA has none here
    return ref


# --- surface reader / generators / checks -------------------------------------------


def test_read_circle_orientation_and_volume(circle_surface):
    c = circle_surface
    assert c.points.shape == (50, 2) and c.elements.shape == (50, 2)
    S.check_watertight(c)
    S.check_inside_box(c, GRID2.lo, GRID2.hi)
    n = np.asarray(S.normals(c.points, c.elements))
    cen = np.asarray(S.centroids(c.points, c.elements)) - 5.0
    # normals point into the flow (= radially out of the body)
    assert (np.einsum("ed,ed->e", n, cen) > 0).all()
    # 50-gon inscribed in r = 3: area = 50/2 r^2 sin(2 pi / 50)
    assert S.body_volume(c) == pytest.approx(25.0 * 9.0 * math.sin(2 * math.pi / 50), rel=1e-12)


def test_generated_circle_matches_sparta_file(circle_surface):
    g = S.circle(3.0, 50, (5.0, 5.0))
    np.testing.assert_allclose(np.asarray(g.points), np.asarray(circle_surface.points), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(g.elements), np.asarray(circle_surface.elements))


def test_sphere_file_and_generator(sphere_surface):
    s = sphere_surface
    assert s.points.shape == (602, 3) and s.elements.shape == (1200, 3)
    S.check_watertight(s)
    n = np.asarray(S.normals(s.points, s.elements))
    cen = np.asarray(S.centroids(s.points, s.elements))
    assert (np.einsum("ed,ed->e", n, cen) > 0).all()
    assert 0.9 * 4 / 3 * math.pi < S.body_volume(s) < 4 / 3 * math.pi
    g = S.sphere(1.0, 2)
    S.check_watertight(g)
    assert g.elements.shape == (320, 3)
    assert 0.95 * 4 / 3 * math.pi < S.body_volume(g) < 4 / 3 * math.pi
    assert np.asarray(S.areas(g.points, g.elements)).sum() < 4 * math.pi


def test_watertight_rejects_open_and_inconsistent(circle_surface):
    c = circle_surface
    e = np.asarray(c.elements)
    with pytest.raises(ValueError, match="watertight"):
        S.check_watertight(S.make_surface(np.asarray(c.points), e[:-1]))
    flipped = e.copy()
    flipped[3] = flipped[3, ::-1]
    with pytest.raises(ValueError, match="watertight"):
        S.check_watertight(S.make_surface(np.asarray(c.points), flipped))
    # a reversed loop is watertight but inside-out: negative volume
    assert S.body_volume(S.make_surface(np.asarray(c.points), e[:, ::-1])) < 0
    with pytest.raises(ValueError, match="outside the box"):
        S.check_inside_box(c, [0.0, 0.0], [7.0, 10.0])


def test_transform_scales_volume_and_moves_candidates(circle_surface):
    c = circle_surface
    p2 = S.transform(c.points, scale=2.0, origin=jnp.array([5.0, 5.0]))
    c2 = S.make_surface(np.asarray(p2), np.asarray(c.elements))
    assert S.body_volume(c2) == pytest.approx(4.0 * S.body_volume(c), rel=1e-12)
    p3 = S.transform(c.points, translate=jnp.array([1.0, 0.0]))
    c3 = S.make_surface(np.asarray(p3), np.asarray(c.elements))
    st = cut.stencil(c, GRID2, 1)
    st3 = cut.stencil(c3, GRID2, 1)
    # a shift by two cells in x moves every stencil row by 2 * n_y cell ids (cell ids are
    # row-major, last axis fastest, as `grid.cell_of`)
    rows = np.nonzero(st[:, 0] >= 0)[0]
    np.testing.assert_array_equal(st3[rows + 2 * GRID2.n[1]], st[rows])


# --- circle tables vs SPARTA ------------------------------------------------------


def test_circle_counts_and_cut_set(circle_mesh):
    m = circle_mesh
    assert _counts(m.cell_type) == (264, 88, 48)  # SPARTA log: outside / inside / overlap
    ref = _sparta_volumes(os.path.join(RESULTS, "grid_vol.circle_ar.12345.txt"), GRID2)
    full = float(cell_volume(GRID2)[0])
    partial = (ref > 1e-12) & (ref < full - 1e-12)
    ours = np.asarray(m.volume)
    ct = np.asarray(m.cell_type)
    # every partially-filled SPARTA cell is CUT here; the 4 CUT cells with full volume are
    # those merely touched at the grid corners (2, 5) and (8, 5)
    assert partial[ct == CUT].sum() == 44 and not partial[ct != CUT].any()
    touched = (ct == CUT) & ~partial
    assert touched.sum() == 4
    assert np.array_equal(ours[touched], np.full(4, full)) and np.allclose(ref[touched], full)
    assert (ours[ct == OUTSIDE] == full).all() and (ours[ct == INSIDE] == 0.0).all()


def test_circle_volumes_vs_sparta(circle_mesh):
    ref = _sparta_volumes(os.path.join(RESULTS, "grid_vol.circle_ar.12345.txt"), GRID2)
    ours = np.asarray(circle_mesh.volume)
    exact = cut.exact_flow_volume(circle_mesh.body, GRID2)
    assert ref.sum() == pytest.approx(exact, abs=1e-6)  # SPARTA's cut volumes are exact
    assert ref.sum() == pytest.approx(71.8, abs=5e-4)  # printed in the SPARTA log
    # stratified MC at sub = 32: per-cell sd ~4e-3 x cell volume (module docstring)
    err = ours - ref
    assert np.abs(err).max() < 4 * 4e-3 * 0.25
    assert abs(err.sum()) < 4 * 4e-3 * 0.25 * math.sqrt(44)


def test_outside_corner_is_off_every_element(circle_mesh):
    m = circle_mesh
    ct = np.asarray(m.cell_type)
    oc = np.asarray(m.outside_corner)
    assert (oc[ct == CUT] >= 0).all() and (oc[ct == INSIDE] == -1).all()
    lo, hi = cut.cell_boxes(GRID2)
    corner = cut.corners(lo, hi)[np.arange(lo.shape[0]), np.maximum(oc, 0)]
    cutc = ct == CUT
    assert cut._global_outside(m.body, corner[cutc]).all()
    d = cut._plane_distance(m.body, corner[cutc], np.asarray(m.csurfs)[cutc])
    assert (d > 1e-12).all()


def test_point_outside_matches_global_test(circle_mesh):
    m = circle_mesh
    rng = np.random.default_rng(3)
    ct = np.asarray(m.cell_type)
    cells = np.nonzero(ct == CUT)[0]
    lo, hi = cut.cell_boxes(GRID2)
    cell = rng.choice(cells, 4000)
    x = lo[cell] + rng.random((4000, 2)) * (hi[cell] - lo[cell])
    local = cut.point_outside(
        m.body, GRID2, cell, x, np.asarray(m.csurfs), np.asarray(m.outside_corner)
    )
    np.testing.assert_array_equal(local, cut._global_outside(m.body, x))


def test_segments_cross_even_times_outside_to_outside(circle_surface):
    rng = np.random.default_rng(5)
    p = rng.random((2000, 2)) * 10.0
    q = rng.random((2000, 2)) * 10.0
    out = cut._global_outside(circle_surface, p) & cut._global_outside(circle_surface, q)
    n = cut.crossings(circle_surface, p[out], q[out])
    assert (n % 2 == 0).all() and n.max() == 2


def test_stencil_is_superset_and_exact_at_radius_zero(circle_surface):
    cs = cut.overlaps(circle_surface, GRID2)
    for radius in (0, 1, 2):
        st = cut.stencil(circle_surface, GRID2, radius)
        for c in range(cs.shape[0]):
            a = set(cs[c][cs[c] >= 0].tolist())
            b = set(st[c][st[c] >= 0].tolist())
            assert a <= b
    # radius 0 = the bounding-box stage of the clip test: rows differ only where an
    # element's AABB overlaps a cell its segment misses
    st0 = cut.stencil(circle_surface, GRID2, 0)
    assert (st0[:, 0] >= 0).sum() >= (cs[:, 0] >= 0).sum()


def test_candidates_vs_brute_force_clip(circle_surface):
    """`overlaps` == elements whose segment meets the closed cell (brute force)."""
    cs = cut.overlaps(circle_surface, GRID2)
    lo, hi = cut.cell_boxes(GRID2)
    p = np.asarray(circle_surface.points)[np.asarray(circle_surface.elements)]
    for c in range(lo.shape[0]):
        listed = set(cs[c][cs[c] >= 0].tolist())
        brute = set()
        for e in range(p.shape[0]):
            if _segment_meets_box(p[e, 0], p[e, 1], lo[c], hi[c]):
                brute.add(e)
        assert listed == brute, c


def _segment_meets_box(a, b, lo, hi):
    """Liang-Barsky clip of the closed segment against the closed box."""
    t0, t1 = 0.0, 1.0
    d = b - a
    for k in range(a.size):
        for sign, bound in ((-1.0, lo[k]), (1.0, hi[k])):
            p = sign * d[k]
            q = sign * (bound - a[k])
            if p == 0.0:
                if q < 0.0:
                    return False
            else:
                t = q / p
                if p < 0.0:
                    t0 = max(t0, t)
                else:
                    t1 = min(t1, t)
    return t0 <= t1


def _thin_plate():
    # a closed thin plate (thickness 0.1 < h) inside the cell row y in [4.5, 5]: every cell
    # it spans has flow below and above -> split (12 cells, x in [2, 8]); the cells whose
    # face carries the plate tips (x = 2 and 8) are touched but connected
    pts = np.array([[2.0, 4.7], [8.0, 4.7], [8.0, 4.8], [2.0, 4.8]])  # counter-clockwise
    return S.make_surface(pts, np.array([[1, 0], [2, 1], [3, 2], [0, 3]]))


def test_split_check_finds_the_split_cells():
    plate = _thin_plate()
    assert S.body_volume(plate) == pytest.approx(0.6)
    m = cut.build_mesh(plate, GRID2)  # builds: split cells get sub-cells
    with pytest.raises(cut.SplitCellError) as info:
        cut.split_check(plate, GRID2, m.cell_type, m.csurfs, m.outside_corner)
    assert len(info.value.cells) == 12


def test_split_cells_get_sub_cells():
    """SPARTA's split cells: each of the 12 cells cut by the plate keeps its larger flow
    region (here the regions are equal, 0.1 each) and gives the other a sub-cell; the
    volumes add up, and positions below / above the plate in one cell fall in different
    collision cells."""
    from jax_dsmc.grid import collision_cell, num_cells  # noqa: PLC0415

    m = cut.build_mesh(_thin_plate(), GRID2)
    assert m.n_sub == 12
    grid = cut.with_volume(GRID2, m)
    assert num_cells(grid) == 400 + 12
    vol = np.asarray(m.volume)
    assert vol.size == 412
    assert vol.sum() == pytest.approx(100.0 - 0.6, abs=0.01)
    split = np.nonzero(np.asarray(m.split_of) >= 0)[0]
    assert split.size == 12
    parent = np.asarray(m.sub_parent)
    np.testing.assert_array_equal(np.sort(parent), np.sort(split))
    # each split cell: two regions of ~0.1 (0.2 x 0.5 below, above the plate)
    for j, cell in enumerate(parent):
        np.testing.assert_allclose([vol[cell], vol[400 + j]], [0.1, 0.1], atol=0.01)
    # the cells 4.5 <= y < 5, x in [3, 3.5]: points below and above the plate
    below = jnp.array([[3.2, 4.6], [3.4, 4.55], [3.1, 4.69]])
    above = jnp.array([[3.2, 4.9], [3.4, 4.95], [3.1, 4.81]])
    cb, ca = np.asarray(collision_cell(below, grid)), np.asarray(collision_cell(above, grid))
    assert len(set(cb)) == 1 and len(set(ca)) == 1 and cb[0] != ca[0]
    assert {int(cb[0]), int(ca[0])} & set(range(400, 412))
    # outside the split cells the collision cell is the Cartesian one
    far = jnp.array([[1.2, 1.3], [9.7, 9.9], [5.0, 2.0]])
    np.testing.assert_array_equal(
        np.asarray(collision_cell(far, grid)), np.asarray(collision_cell(far, GRID2))
    )


def test_axis_aligned_square_on_grid_lines():
    """Edges along cell faces and through corners: no error, exact volume, touched cells."""
    pts = np.array([[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]])  # counter-clockwise
    sq = S.make_surface(pts, np.array([[1, 0], [2, 1], [3, 2], [0, 3]]))
    assert S.body_volume(sq) == pytest.approx(4.0)
    m = cut.build_mesh(sq, GRID2)
    ours = np.asarray(m.volume)
    assert ours.sum() == pytest.approx(96.0, abs=1e-12)
    ct = np.asarray(m.cell_type)
    lo, hi = cut.cell_boxes(GRID2)
    mid = 0.5 * (lo + hi)
    interior = (mid > 2.0).all(axis=1) & (mid < 4.0).all(axis=1)
    assert (ours[interior] == 0.0).all() and (ours[~interior] == 0.25).all()
    # every cell whose closed box touches the square is CUT (SPARTA's inclusive rule)
    overlap = (hi >= 2.0).all(axis=1) & (lo <= 4.0).all(axis=1)
    strictly_inside = (lo > 2.0).all(axis=1) & (hi < 4.0).all(axis=1)
    assert np.array_equal(ct == CUT, overlap & ~strictly_inside)


def test_analytic_circle_tables():
    body = AnalyticBody(centre=jnp.array([5.0, 5.0]), radius=jnp.float64(3.0))
    m = cut.build_mesh(body, GRID2)
    assert m.kind == "circle" and m.n_elements == 1
    assert _counts(m.cell_type) == (260, 88, 52)
    assert float(m.volume.sum()) == pytest.approx(100.0 - 9.0 * math.pi, abs=0.02)
    ct = np.asarray(m.cell_type)
    lo, hi = cut.cell_boxes(GRID2)
    c = np.array([5.0, 5.0])
    dmin = np.linalg.norm(np.clip(c, lo, hi) - c, axis=1)
    dmax = np.linalg.norm(cut.corners(lo, hi) - c, axis=2).max(axis=1)
    np.testing.assert_array_equal(ct == CUT, (dmin <= 3.0) & (dmax >= 3.0))


def test_volume_fraction_error_shrinks_with_sub(circle_surface):
    cs = cut.overlaps(circle_surface, GRID2)
    ct = cut.classify(circle_surface, GRID2, cs)
    oc = cut.outside_corner(circle_surface, GRID2, ct, cs)
    ref = _sparta_volumes(os.path.join(RESULTS, "grid_vol.circle_ar.12345.txt"), GRID2)
    errs = []
    for sub in (4, 16, 64):
        fr = cut.volume_fraction(circle_surface, GRID2, ct, cs, oc, sub=sub)
        errs.append(np.abs(fr * 0.25 - ref).max())
    assert errs[2] < errs[0] / 4  # ~1/sub^1.5 per cell in 2-D; demand at least 1/sub^0.5 x 4


# --- sphere (slower: ~6 s) ------------------------------------------------------


def test_sphere_counts_and_volumes_vs_sparta(sphere_surface):
    m = cut.build_mesh(sphere_surface, GRID3, sub=8)
    assert _counts(m.cell_type) == (7256, 304, 440)
    ref = _sparta_volumes(os.path.join(RESULTS, "grid_vol.sphere_ar.12345.txt"), GRID3)
    assert ref.sum() == pytest.approx(cut.exact_flow_volume(sphere_surface, GRID3), abs=1e-5)
    assert ref.sum() == pytest.approx(59.8617, abs=5e-5)
    ours = np.asarray(m.volume)
    ct = np.asarray(m.cell_type)
    full = float(cell_volume(GRID3)[0])
    partial = (ref > 1e-12) & (ref < full - 1e-12)
    assert not partial[ct != CUT].any()
    assert (np.abs(ours - ref)[ct != CUT] < 1e-15).all()
    # sub = 8 in 3-D: per-cell sd ~ sqrt(2)/(2 sub^2) = 1.1e-2 x cell volume
    assert np.abs(ours - ref).max() < 4 * 1.1e-2 * full
