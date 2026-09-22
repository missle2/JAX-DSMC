"""LS0 -- the level-set body (`jax_dsmc.mesh.levelset`) against the analytic circle / sphere.

Interpolation, values and normals near the surface, hits and hit derivatives (the Newton
step), tracking through `wall_adjoint.advance` on both trackers, the cut-cell tables, the
grazing (tunnelling) family, the box-only invariance and reinitialisation.  CPU, seconds.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_dsmc import wall_adjoint
from jax_dsmc.boundary import BC, Faces, WallParams
from jax_dsmc.mesh import analytic, cut
from jax_dsmc.mesh import levelset as L
from jax_dsmc.mesh.types import CUT, AnalyticBody, LevelSet
from jax_dsmc.state import GridSpec

MASS = 6.63e-26
T_WALL = 300.0
H_PHI = {2: 0.125, 3: 0.1}


@pytest.fixture(autouse=True)
def _release_compiled_executables():
    """The level-set hit search compiles once per (N, D, n_sample) and the full suite runs
    in one process: drop the executables after every test so the suite's resident memory
    does not grow by several GB here (a later test's XLA cache read segfaulted at ~11 GB)."""
    yield
    jax.clear_caches()


def _grid(dim):
    if dim == 2:
        return GridSpec(lo=jnp.array([0.0, 0.0]), hi=jnp.array([10.0, 10.0]), n=(20, 20))
    return GridSpec(lo=jnp.array([-2.0] * 3), hi=jnp.array([2.0] * 3), n=(10, 10, 10))


def _body(dim):
    if dim == 2:
        return AnalyticBody(centre=jnp.array([5.0, 5.0]), radius=jnp.float64(3.0))
    return AnalyticBody(centre=jnp.zeros(3), radius=jnp.float64(1.0))


def _levelset(dim, n_sample=8, margin=3):
    grid = _grid(dim)
    body = _body(dim)
    lo, m = L.grid_for_box(grid.lo, grid.hi, H_PHI[dim], margin=margin)
    return L.from_analytic(body.centre, float(body.radius), lo, H_PHI[dim], m, n_sample=n_sample)


def _outside(key, n, dim, grid, body, speed=337.0, margin=0.05):
    kx, kv = jax.random.split(key)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    x = lo + jax.random.uniform(kx, (n, dim), dtype=jnp.float64) * (hi - lo)
    v = jax.random.normal(kv, (n, 3), dtype=jnp.float64) * speed
    if dim == 2:
        v = v.at[:, 2].set(0.0)
    keep = jnp.linalg.norm(x - body.centre, axis=1) > float(body.radius) + margin
    return x[keep], v[keep]


def _advance(x, v, alive, grid, faces, wall, mesh, tracker, dt, **kw):
    return wall_adjoint.advance(
        x,
        v,
        alive,
        grid,
        faces,
        wall,
        MASS,
        dt,
        jax.random.PRNGKey(0),
        mesh=mesh,
        tracker=tracker,
        max_bounces=kw.pop("max_bounces", 16),
        max_walk=kw.pop("max_walk", 64),
        **kw,
    )


# --- interpolation, values, normals -----------------------------------------------------


@pytest.mark.parametrize("dim", [2, 3])
def test_interpolation_reproduces_interior_knot_values(dim):
    rng = np.random.default_rng(0)
    shape = (9, 11, 8)[:dim]
    values = rng.normal(size=shape)
    ls = L.from_values(values, np.zeros(dim), 0.5)
    phi = np.asarray(L.value(ls, L.knots(ls))).reshape(shape)
    interior = tuple(slice(1, -1) for _ in range(dim))
    np.testing.assert_allclose(phi[interior], values[interior], atol=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
def test_circle_values_and_normals_near_the_surface(dim):
    ls = _levelset(dim)
    body = _body(dim)
    grid = _grid(dim)
    rng = np.random.default_rng(1)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    x = lo + rng.uniform(size=(20000, dim)) * (hi - lo)
    sd = np.asarray(L.analytic_distance(body.centre, float(body.radius), jnp.asarray(x)))
    band = np.abs(sd) < 2 * H_PHI[dim]
    phi, g = L.value_and_grad(ls, jnp.asarray(x))
    phi, g = np.asarray(phi), np.asarray(g)
    assert np.abs(phi - sd)[band].max() < 1e-5 * float(body.radius)
    n_exact = (x - np.asarray(body.centre)) / np.linalg.norm(x - np.asarray(body.centre), axis=1)[
        :, None
    ]
    n_ls = g / np.linalg.norm(g, axis=1)[:, None]
    assert np.abs(n_ls - n_exact)[band].max() < 1e-4
    assert np.abs(np.linalg.norm(g, axis=1) - 1.0)[band].max() < 1e-3
    # the sign is right everywhere in the box
    assert np.all((phi < 0) == (sd < 0))


# --- hits and their derivative ----------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 3])
def test_hits_match_the_analytic_body(dim):
    ls = _levelset(dim)
    body = _body(dim)
    grid = _grid(dim)
    x, v = _outside(jax.random.PRNGKey(2), 20000, dim, grid, body, speed=600.0)
    t_an = np.asarray(analytic.hit_body(x, v, body, jnp.zeros(x.shape[0], bool)))
    t_ls = np.asarray(jax.jit(L.hit_body)(x, v, ls))
    speed = np.linalg.norm(np.asarray(v)[:, :dim], axis=1)
    within = np.isfinite(t_an) & (t_an * speed <= ls.reach)
    # grazing incidence: |v . n| below 1e-3 |v| may go either way (the polygon test's caveat)
    x_hit = np.asarray(x) + t_an[:, None] * np.asarray(v)[:, :dim]
    n = (x_hit - np.asarray(body.centre)) / float(body.radius)
    grazing = np.abs(np.sum(n * np.asarray(v)[:, :dim], axis=1)) < 1e-3 * speed
    assert within.sum() > 250
    missed = within & ~np.isfinite(t_ls) & ~grazing
    spurious = ~within & np.isfinite(t_ls) & (t_ls * speed <= ls.reach) & ~grazing
    assert missed.sum() == 0 and spurious.sum() == 0
    both = within & np.isfinite(t_ls)
    err = np.abs(t_ls[both] - t_an[both]) * speed[both]
    # the level-set surface itself is off the circle by its interpolation error
    x_ls = np.asarray(x) + t_ls[:, None] * np.asarray(v)[:, :dim]
    sd = np.asarray(L.analytic_distance(body.centre, float(body.radius), jnp.asarray(x_ls[both])))
    assert err.max() < 1e-4 * float(body.radius)
    assert np.abs(np.asarray(L.value(ls, jnp.asarray(x_ls[both])))).max() < 1e-12
    assert np.abs(sd).max() < 3e-5 * float(body.radius)
    # a particle on the (level-set) surface moving inward hits at t = 0; moving outward never
    on = jnp.asarray((np.asarray(x) + t_ls[:, None] * np.asarray(v)[:, :dim])[both][:50])
    vin = jnp.asarray(np.asarray(v)[both][:50])
    t0 = np.asarray(L.hit_body(on, vin, ls))
    assert np.all(t0 * 600.0 < 1e-12)  # exactly 0, or a roundoff-positive start
    vout = vin - 2.0 * jnp.sum(vin[:, :dim] * jnp.asarray(n[both][:50]), axis=1, keepdims=True) * (
        jnp.zeros((50, 3)).at[:, :dim].set(jnp.asarray(n[both][:50]))
    )
    assert np.all(np.isinf(np.asarray(L.hit_body(on, vout, ls))))
    # deeper inside: no hit (the tracker's inside check is the catch-all)
    deep = jnp.asarray(
        np.asarray(body.centre)[None, :] + 0.5 * (x_hit[both][:50] - np.asarray(body.centre))
    )
    assert np.all(np.isinf(np.asarray(L.hit_body(deep, vin, ls))))


def _t_first(coef, ls, x, v):
    return L.hit_body(x, v, LevelSet(coef, ls.lo, ls.h, ls.n_sample, ls.n_bisect))[0]


def _t_xv(x, v, ls):
    return L.hit_body(x, v, ls)[0]


@pytest.mark.parametrize("dim", [2, 3])
def test_hit_time_derivative_is_the_implicit_function_derivative(dim):
    """`d t / d coef` contracted with `d coef / d r = -1` equals the analytic `d t / d r`;
    `d t / d x`, `d t / d v` match central differences of the level-set hit itself."""
    ls = _levelset(dim)
    body = _body(dim)
    grid = _grid(dim)
    x, v = _outside(jax.random.PRNGKey(3), 4000, dim, grid, body, speed=600.0)
    t_an = np.asarray(analytic.hit_body(x, v, body, jnp.zeros(x.shape[0], bool)))
    speed = np.linalg.norm(np.asarray(v)[:, :dim], axis=1)
    ok = np.nonzero(np.isfinite(t_an) & (t_an * speed <= 0.8 * ls.reach))[0][:6]
    assert ok.size == 6
    h = 1e-6
    for i in ok:
        xi, vi = x[i : i + 1], v[i : i + 1]
        g_coef = jax.grad(_t_first)(ls.coef, ls, xi, vi)
        dt_dr = -float(jnp.sum(g_coef))
        b_plus = AnalyticBody(body.centre, body.radius + h)
        b_minus = AnalyticBody(body.centre, body.radius - h)
        z = jnp.zeros(1, bool)
        fd = (
            float(analytic.hit_body(xi, vi, b_plus, z)[0])
            - float(analytic.hit_body(xi, vi, b_minus, z)[0])
        ) / (2 * h)
        np.testing.assert_allclose(dt_dr, fd, rtol=1e-4)
        g_x, g_v = jax.grad(_t_xv, argnums=(0, 1))(xi, vi, ls)
        for axis in range(dim):
            d = jnp.zeros_like(xi).at[0, axis].set(h)
            fd_x = (float(_t_xv(xi + d, vi, ls)) - float(_t_xv(xi - d, vi, ls))) / (2 * h)
            np.testing.assert_allclose(float(g_x[0, axis]), fd_x, rtol=1e-5, atol=1e-12)
            dv = jnp.zeros_like(vi).at[0, axis].set(h * 1e3)
            fd_v = (float(_t_xv(xi, vi + dv, ls)) - float(_t_xv(xi, vi - dv, ls))) / (2 * h * 1e3)
            np.testing.assert_allclose(float(g_v[0, axis]), fd_v, rtol=1e-5, atol=1e-15)


# --- tracking -----------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("tracker", ["stencil", "walk"])
def test_tracking_matches_the_analytic_body(dim, tracker):
    grid = _grid(dim)
    body = _body(dim)
    ls = _levelset(dim, n_sample=16)
    mesh_a = cut.build_mesh(body, grid, radius=2, sub=4)
    mesh_l = cut.build_mesh(ls, grid, radius=2, sub=4)
    assert mesh_l.kind == "levelset"
    x, v = _outside(jax.random.PRNGKey(4), 12000 if dim == 2 else 5000, dim, grid, body)
    alive = jnp.ones(x.shape[0], bool)
    faces = Faces((BC.OUTFLOW, BC.OUTFLOW) + (BC.SPECULAR,) * (2 * dim - 2))
    wall = WallParams.uniform(dim, T_WALL, acc=0.0, n_models=1)
    # reach = 16 h / 2 = 1.0 (2-D) / 0.8 (3-D) covers the 4-sigma move: no overrun
    dt = 5.0e-4 if dim == 2 else 4.0e-4
    out_a = _advance(x, v, alive, grid, faces, wall, mesh_a, tracker, dt)
    out_l = _advance(x, v, alive, grid, faces, wall, mesh_l, tracker, dt)
    assert out_l[3].inside_body == 0 and out_l[3].stencil_overrun == 0 and out_l[3].exhausted == 0
    hit_a = (np.asarray(out_a[1]) != np.asarray(v)).any(axis=1)
    hit_l = (np.asarray(out_l[1]) != np.asarray(v)).any(axis=1)
    assert hit_a.sum() > 200
    # the hit sets agree except for grazing rays (surface error ~1e-7 in position)
    assert (hit_a != hit_l).sum() <= 2
    both = hit_a & hit_l
    # positions differ by the level set's own surface error (1e-7 at h = 0.125 on r = 3,
    # 1e-5 at h = 0.1 on r = 1) propagated through the reflection
    np.testing.assert_allclose(np.asarray(out_l[0])[both], np.asarray(out_a[0])[both], atol=5e-5)
    va, vl = np.asarray(out_a[1])[both], np.asarray(out_l[1])[both]
    # the reflected velocity carries the normal's error (< 1e-4 in direction)
    assert np.abs(vl - va).max() < 2e-4 * np.abs(va).max()
    assert int(out_l[3].surface_hits) == pytest.approx(int(out_a[3].surface_hits), abs=2)
    # energy exact through the reflections
    np.testing.assert_allclose(
        np.sum(np.asarray(out_l[1]) ** 2, axis=1)[out_l[2]],
        np.sum(np.asarray(v) ** 2, axis=1)[out_l[2]],
        rtol=1e-13,
    )


def test_grazing_family_tunnelling_is_bounded():
    """Rays tangent-ish to the circle with impact parameter `r - delta`: every ray whose
    chord inside the body is longer than the sample spacing `h/2` is found; the shorter
    ones may tunnel -- the rate is reported, not hidden."""
    ls = _levelset(2)
    body = _body(2)
    c, r = np.asarray(body.centre), float(body.radius)
    deltas = np.geomspace(1e-6, 1e-2, 400)
    # ray along +x at height c_y + (r - delta), starting left of the body
    x = np.stack([np.full(deltas.size, c[0] - 0.3), c[1] + r - deltas], axis=1)
    v = np.zeros((deltas.size, 3))
    v[:, 0] = 500.0
    t = np.asarray(L.hit_body(jnp.asarray(x), jnp.asarray(v), ls))
    chord = 2.0 * np.sqrt(2.0 * r * deltas)
    spacing = ls.h / 2.0
    resolved = chord > spacing
    assert resolved.sum() > 100
    assert np.isfinite(t[resolved]).all()
    # the analytic body hits all of them; count the level set's misses below the spacing
    rate = 1.0 - np.isfinite(t[~resolved]).mean()
    assert rate <= 1.0  # documented, not gated: tunnelling lives below one sample spacing


@pytest.mark.parametrize("tracker", ["stencil", "walk"])
def test_box_only_invariance(tracker):
    """A level-set body no particle can reach leaves the box tracker's result unchanged."""
    dim = 2
    grid = _grid(dim)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    h = (hi - lo) / np.asarray(grid.n)
    centre = lo + 0.5 * h
    ls_lo, m = L.grid_for_box(grid.lo, grid.hi, 0.05)
    ls = L.from_analytic(jnp.asarray(centre), 0.25 * float(h.min()), ls_lo, 0.05, m)
    mesh = cut.build_mesh(ls, grid, radius=1, sub=4)
    faces = Faces((BC.DIFFUSE, BC.SPECULAR, BC.PERIODIC, BC.OUTFLOW))
    wall = WallParams.uniform(dim, T_WALL, acc=1.0, n_models=1)
    kx, kv = jax.random.split(jax.random.PRNGKey(5))
    n = 2000
    x = lo + h + jax.random.uniform(kx, (n, dim), dtype=jnp.float64) * (hi - lo - 2 * h)
    v = jax.random.normal(kv, (n, 3), dtype=jnp.float64) * 300.0
    alive = jnp.ones(n, bool)
    dt = 1e-4
    ref = _advance(x, v, alive, grid, faces, wall, None, "stencil", dt, epsilon=0.0)
    out = _advance(x, v, alive, grid, faces, wall, mesh, tracker, dt, epsilon=0.0)
    np.testing.assert_allclose(np.asarray(out[0]), np.asarray(ref[0]), rtol=0, atol=1e-13)
    assert bool((ref[1] == out[1]).all()) and bool((ref[2] == out[2]).all())
    assert out[3].surface_hits == 0 and out[3].inside_body == 0


# --- host tables --------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 3])
def test_cut_tables_agree_with_the_analytic_body(dim):
    grid = _grid(dim)
    body = _body(dim)
    ls = _levelset(dim)
    mesh_a = cut.build_mesh(body, grid, radius=1, sub=16)
    mesh_l = cut.build_mesh(ls, grid, radius=1, sub=16)
    ct_a, ct_l = np.asarray(mesh_a.cell_type), np.asarray(mesh_l.cell_type)
    # the coefficient test is conservative: every analytic cut cell is a level-set cut cell
    assert np.all(ct_l[ct_a == CUT] == CUT)
    # and it does not over-flag much: the extra cut cells are full or empty
    extra = (ct_l == CUT) & (ct_a != CUT)
    assert extra.sum() <= 0.5 * (ct_a == CUT).sum()
    # the cut-cell sets differ, so the stratified samples differ: per-cell noise of a few
    # per cent of a cell, totals to 1e-3
    v_a, v_l = np.asarray(mesh_a.volume), np.asarray(mesh_l.volume)
    np.testing.assert_allclose(v_l, v_a, rtol=0, atol=3e-2 * float(v_a.max()))
    np.testing.assert_allclose(v_l.sum(), v_a.sum(), rtol=1e-3)
    vol_ref = cut.exact_flow_volume(body, grid)
    vol_ls = cut.exact_flow_volume(ls, grid)
    assert abs(vol_ls - vol_ref) < 2e-3 * vol_ref
    assert mesh_l.n_elements == 1 and mesh_l.normals.shape == (1, dim)


def test_build_mesh_refuses_an_uncovered_box():
    grid = _grid(2)
    ls = L.from_analytic(jnp.array([5.0, 5.0]), 3.0, np.array([2.0, 2.0]), 0.5, (12, 12))
    with pytest.raises(ValueError, match="cover"):
        cut.build_mesh(ls, grid, radius=1, sub=4)


# --- reinitialisation and constraints --------------------------------------------------


def test_reinitialise_keeps_the_zero_set_and_restores_the_distance():
    ls = _levelset(2)
    # a smooth non-distance perturbation: phi -> phi (1 + 0.3 sin) + 0.2 phi^2 keeps the
    # zero set but ruins |grad phi| = 1
    x = L.knots(ls)
    phi = L.value(ls, x)
    bumped = phi * (1.0 + 0.3 * jnp.sin(x[:, 0])) + 0.2 * phi * phi
    ls_b = L.from_values(bumped.reshape(ls.coef.shape), ls.lo, ls.h)
    ls_r = L.reinitialise(ls_b, n_iter=10)
    ang = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    on = jnp.asarray(np.array([5.0, 5.0]) + 3.0 * np.stack([np.cos(ang), np.sin(ang)], axis=1))
    assert np.abs(np.asarray(L.value(ls_r, on))).max() < 1e-3
    near = jnp.asarray(np.array([5.0, 5.0]) + 3.2 * np.stack([np.cos(ang), np.sin(ang)], axis=1))
    _, g = L.value_and_grad(ls_r, near)
    assert np.abs(np.linalg.norm(np.asarray(g), axis=1) - 1.0).max() < 2e-2
    np.testing.assert_allclose(np.asarray(L.value(ls_r, near)), 0.2, atol=2e-3)


def _volume_of_coef(coef, ls):
    return L.body_volume(LevelSet(coef, ls.lo, ls.h), [0.0, 0.0], [10.0, 10.0], 4)


def test_body_volume_and_band():
    ls = _levelset(2)
    vol = float(L.body_volume(ls, [0.0, 0.0], [10.0, 10.0], sub=4))
    assert abs(vol - math.pi * 9.0) < 2e-2 * math.pi * 9.0
    g = jax.grad(_volume_of_coef)(ls.coef, ls)
    # d(area)/d(uniform shift of phi by -1) = perimeter 2 pi r
    assert abs(-float(jnp.sum(g)) - 2 * math.pi * 3.0) < 0.05 * 2 * math.pi * 3.0
    # the smoothed Heaviside (half-width h / 2) has tanh tails: nothing beyond ~5 h
    band = np.asarray(L.coefficient_band(ls, 5 * ls.h))
    assert 0 < band.sum() < band.size
    outside_band = np.abs(np.asarray(g))[~band]
    assert outside_band.max() < 1e-4 * np.abs(np.asarray(g)).max()


def test_from_surface_matches_the_polygon_within_its_sagitta():
    """The level set of the SPARTA 50-gon: its signed distance differs from the circle's
    by at most the polygon's own chord error (plus the interpolation error)."""
    import os  # noqa: PLC0415

    from jax_dsmc.mesh import surface as S  # noqa: PLC0415

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    poly = S.read_sparta_surf(os.path.join(root, "examples", "mesh", "sparta", "data.circle"), 2)
    lo, m = L.grid_for_box([0.0, 0.0], [10.0, 10.0], 0.125)
    ls = L.from_surface(poly, lo, 0.125, m)
    rng = np.random.default_rng(3)
    x = rng.uniform(0.5, 9.5, (20000, 2))
    sd = np.linalg.norm(x - 5.0, axis=1) - 3.0
    band = np.abs(sd) < 0.3
    phi = np.asarray(L.value(ls, jnp.asarray(x)))
    sagitta = 3.0 * (1.0 - math.cos(math.pi / 50))
    assert np.abs(phi - sd)[band].max() < 1.5 * sagitta
    assert np.mean((phi < 0) == (sd < 0)) > 0.995
    # and it tracks: the 50-gon's level set is a valid body on the SPARTA grid
    mesh = cut.build_mesh(ls, _grid(2), radius=1, sub=4)
    assert (np.asarray(mesh.cell_type) == CUT).sum() >= 48


def _t_first_grazing(coef, ls, x, v, eps):
    body = LevelSet(coef, ls.lo, ls.h, ls.n_sample, ls.n_bisect, grazing=eps)
    return L.hit_body(x, v, body)[0]


def _t_analytic(r, body, x, v, eps):
    b = AnalyticBody(body.centre, r, grazing=eps)
    return analytic.hit_body(x, v, b, jnp.zeros(1, bool))[0]


def test_grazing_regularisation_keeps_the_primal_and_bounds_the_derivative():
    """Both bodies: with the cut-off the hit time is bitwise unchanged, the derivative of a
    well-incident ray is unchanged, and a grazing ray's derivative is bounded by the
    cut-off instead of ``1 / cos``."""
    ls = _levelset(2)
    body = _body(2)
    c, r = np.asarray(body.centre), float(body.radius)
    # a normal-incidence ray and a grazing one (impact parameter r - 1e-4 -> cos ~ 8e-3)
    rays = {
        "normal": (jnp.array([[c[0] - r - 0.3, c[1]]]), jnp.array([[500.0, 0.0, 0.0]])),
        "grazing": (jnp.array([[c[0] - 0.5, c[1] + r - 1e-4]]), jnp.array([[500.0, 0.0, 0.0]])),
    }
    for name, (x, v) in rays.items():
        t0 = float(L.hit_body(x, v, ls)[0])
        t_reg = float(L.hit_body(x, v, dataclasses.replace(ls, grazing=0.1))[0])
        assert t_reg == t0, name
        g0 = -float(jnp.sum(jax.grad(_t_first_grazing)(ls.coef, ls, x, v, 0.0)))
        g1 = -float(jnp.sum(jax.grad(_t_first_grazing)(ls.coef, ls, x, v, 0.1)))
        a0 = float(jax.grad(_t_analytic)(jnp.float64(r), body, x, v, 0.0))
        a1 = float(jax.grad(_t_analytic)(jnp.float64(r), body, x, v, 0.1))
        assert float(
            analytic.hit_body(
                x, v, AnalyticBody(body.centre, body.radius, 0.1), jnp.zeros(1, bool)
            )[0]
        ) == float(analytic.hit_body(x, v, body, jnp.zeros(1, bool))[0])
        np.testing.assert_allclose(g0, a0, rtol=1e-3)
        if name == "normal":
            np.testing.assert_allclose(g1, g0, rtol=1e-12)
            np.testing.assert_allclose(a1, a0, rtol=1e-12)
        else:
            # exact: dt/dr = -1 / (|v| cos) ~ -1 / (500 * 8e-3); clipped at cos = 0.1
            assert abs(g0) > 5.0 * abs(g1) and abs(a0) > 5.0 * abs(a1)
            np.testing.assert_allclose(g1, -1.0 / (500.0 * 0.1), rtol=2e-2)
            np.testing.assert_allclose(a1, -1.0 / (500.0 * 0.1), rtol=2e-2)


def test_closest_approach_of_rays_past_the_circle():
    """Rays along +x at impact parameter ``3 + b``: the minimum of ``phi`` along the flight
    is ``|b|`` for a miss and ``-|b|`` (the chord's depth) for a hit, at the abscissa of
    the centre."""
    ls = _levelset(2, n_sample=16)
    b = np.array([0.05, 0.2, -0.1, -0.5, 1e-4, -1.0])
    x = jnp.asarray(np.stack([np.full(b.size, 5.0 - 0.6), 5.0 + 3.0 + b], axis=1))
    v = jnp.asarray(np.tile([500.0, 0.0, 0.0], (b.size, 1)))
    phi_min, s_min = L.closest_approach(x, v, ls, jnp.full(b.size, 0.002))
    np.testing.assert_allclose(np.asarray(phi_min), b, atol=2e-4)
    np.testing.assert_allclose(np.asarray(s_min), 0.6, atol=1e-4)
    # beyond the search reach the minimum is not found: the segment's end is returned
    far = jnp.asarray(np.stack([np.full(2, 5.0 - 3.0), 5.0 + 3.0 + b[:2]], axis=1))
    phi_far, s_far = L.closest_approach(far, v[:2], ls, None)
    assert np.all(np.asarray(s_far) >= ls.reach * (1 - 1e-6)) and np.all(
        np.asarray(phi_far) > b[:2]
    )
