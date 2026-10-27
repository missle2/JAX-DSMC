"""LS0 -- the level-set body (`jax_dsmc.mesh.levelset`) against the analytic circle / sphere.

Interpolation, values and normals near the surface, hits and hit derivatives (the Newton
step), tracking through `wall_adjoint.advance` on both trackers, the cut-cell tables, the
grazing (tunnelling) family, the box-only invariance and reinitialisation.  CPU, seconds.
"""

from __future__ import annotations

import dataclasses
import functools
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


def test_normal_derivative_is_finite_on_a_flat_patch():
    """A flat patch of coefficients (the far-field clamp, unreinitialised) has ``grad phi = 0``
    exactly: the normal's derivative there must be finite, not ``0 / 0`` (2026-10-27: one
    such particle made a whole shape gradient nan)."""
    ls = _levelset(2)
    coef = ls.coef.at[-6:, :6].set(ls.coef.max())
    flat = dataclasses.replace(ls, coef=coef)
    x = jnp.asarray(flat.lo)[None, :] + flat.h * jnp.array([[coef.shape[0] - 4.0, 2.0]])
    assert float(L.safe_norm(L.value_and_grad(flat, x)[1])[0]) == 0.0

    def normal_sum(c):
        return jnp.sum(L.normal(dataclasses.replace(flat, coef=c), x))

    assert np.all(np.isfinite(np.asarray(jax.grad(normal_sum)(coef))))


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


def test_repeated_reinitialise_is_stable_when_the_band_exceeds_the_old_eight_steps():
    # n_sample = 16: band 3 h + 8 h = 11 h, beyond the 8 clipped steps the projection used
    # to take; the knots that stopped short entered the surface cloud and two reinitialisations
    # put spurious zero crossings ~0.5-0.7 off the circle (the optc GD runs, 2026-09-23)
    ls = _levelset(2, n_sample=16)
    xs = np.linspace(1.0, 9.0, 161)
    pts = np.stack(np.meshgrid(xs, xs, indexing="ij"), axis=-1).reshape(-1, 2)
    exact = np.linalg.norm(pts - 5.0, axis=1) - 3.0
    near = np.abs(exact) < 1.5
    for _ in range(3):
        ls = L.reinitialise(ls)
        phi = np.asarray(L.value(ls, jnp.asarray(pts)))
        assert np.abs(phi - exact)[near].max() < 1e-3
        assert not np.any((np.sign(phi) != np.sign(exact)) & (np.abs(exact) > 0.05))


@pytest.mark.parametrize("offset", [-1e-13, 0.0, 1e-13])
def test_grazing_reentry_on_a_concave_wall_is_found_from_the_surface(offset):
    """A particle leaving a concave wall at grazing incidence re-enters after a chord
    shorter than the search's sample spacing.  Starting on the surface (the state right
    after a reflection, ``phi`` = round-off of either sign) the re-entry must be found:
    with ``phi(start) <= 0`` read as "not outside" the first sample interval was never a
    ``+ -> -`` bracket and the particle flew into the body (optc S4 gd iterate 6,
    2026-09-23: a 0.4 degree grazing particle skimming a concave notch)."""
    h, radius, centre = 0.125, 3.0, np.array([5.0, 5.0])
    lo, m = L.grid_for_box(jnp.array([0.0, 0.0]), jnp.array([10.0, 10.0]), h, margin=3)
    knots = np.asarray(lo) + h * np.stack(
        np.meshgrid(*[np.arange(n) for n in m], indexing="ij"), -1
    )
    # flow inside the circle, body outside: a concave wall of radius 3
    values = radius - np.linalg.norm(knots - centre, axis=-1)
    ls = L.from_values(values, lo, h, n_sample=8)
    theta = 0.005  # chord 2 R sin(theta) = 0.03 < sample spacing reach / n_sample = 0.0625
    alpha = 0.3
    # on the spline's own zero set (it differs from the circle by the interpolation error),
    # then pushed by `offset` along the normal: phi(start) = round-off of either sign
    guess = centre + radius * np.array([np.cos(alpha), np.sin(alpha)])
    on = np.asarray(L.project(ls, jnp.asarray(guess[None]), 20))[0]
    x0 = on - offset * np.asarray(L.normal(ls, jnp.asarray(on[None])))[0]
    tangent = np.array([-np.sin(alpha), np.cos(alpha)])
    inward = -np.array([np.cos(alpha), np.sin(alpha)])
    speed = 500.0
    v = speed * (np.cos(theta) * tangent + np.sin(theta) * inward)
    t = float(L.hit_body(jnp.asarray(x0[None]), jnp.asarray(np.r_[v, 0.0][None]), ls)[0])
    chord = 2.0 * radius * np.sin(theta)
    assert np.isfinite(t)
    np.testing.assert_allclose(t * speed, chord, rtol=1e-3, atol=1e-6)


def test_flow_pocket_through_a_face_with_every_corner_in_the_body():
    """A cut cell whose flow is a pocket entering through the middle of a face while all
    its corners lie in the body (optc Adam S1 iterate 56 / S8 iterate 9, 2026-09-24):
    valid for a level set (its inside test is phi < 0, exact) -- the mesh builds and the
    cell's flow volume is the pocket's.  Only the polygon inside test needs a flow corner.
    """
    h = 0.0625
    grid = GridSpec(lo=jnp.array([0.0, 0.0]), hi=jnp.array([10.0, 10.0]), n=(20, 20))
    lo, m = L.grid_for_box(grid.lo, grid.hi, h, margin=3)
    knots = np.asarray(lo) + h * np.stack(
        np.meshgrid(*[np.arange(n) for n in m], indexing="ij"), -1
    )
    # body = the box [3, 7.1]^2 (flow outside), plus a flow disk of radius 0.25 at
    # (5.25, 7.1) cutting into the top of the box: in cell [5, 5.5] x [6.5, 7] the flow
    # reaches down to y = 6.85 from its top face while its corners (distance 0.27 from
    # the disk centre, 0.1 below the box top) are all in the body
    q = np.abs(knots - 5.05) - 2.05
    sdf_box = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)
    disk = 0.25 - np.linalg.norm(knots - np.array([5.25, 7.1]), axis=-1)
    ls = L.from_values(np.maximum(sdf_box, disk), lo, h, n_sample=16)
    cell = 10 * 20 + 13  # x index 10 (5.0-5.5), y index 13 (6.5-7.0)
    lo_c, hi_c = cut.cell_boxes(grid)
    corners = cut.corners(lo_c[cell : cell + 1], hi_c[cell : cell + 1])[0]
    assert (np.asarray(L.value(ls, jnp.asarray(corners))) < 0.0).all()
    mesh = cut.build_mesh(ls, grid, radius=1)
    assert int(mesh.cell_type[cell]) == CUT and int(mesh.outside_corner[cell]) == -1
    # the pocket: the disk segment below y = 7 inside the cell (~0.042; the spline's
    # rounding of the box corner-free edge is small)
    theta = 2.0 * np.arccos(0.1 / 0.25)
    segment = 0.5 * 0.25**2 * (theta - np.sin(theta))
    np.testing.assert_allclose(float(mesh.volume[cell]), segment, rtol=0.1)


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


def _perimeter_of_coef(coef, ls):
    return L.body_perimeter(dataclasses.replace(ls, coef=coef), [0.0, 0.0], [10.0, 10.0], 4)


def test_body_perimeter_and_its_curvature_gradient():
    """The circle's perimeter 2 pi r, and its shape derivative: a uniform outward move of
    the boundary (phi down by 1) adds d P / d r = 2 pi."""
    ls = _levelset(2)
    per = float(L.body_perimeter(ls, [0.0, 0.0], [10.0, 10.0], sub=4))
    assert abs(per - 2 * math.pi * 3.0) < 1e-2 * 2 * math.pi * 3.0
    g = jax.grad(_perimeter_of_coef)(ls.coef, ls)
    assert abs(-float(jnp.sum(g)) - 2 * math.pi) < 0.05 * 2 * math.pi


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


def _dense(fn, fill, ls, rows, candidate):
    """``_culled`` without the cull: ``fn`` on every row."""
    del fill, candidate
    return fn(ls, *rows)


def _jit_hit(x, v, ls, remaining):
    """``hit_body`` compiled afresh (a new function object: no cache shared across a
    monkeypatch)."""
    return jax.jit(functools.partial(L.hit_body, remaining=remaining))(x, v, ls)


def _t_sum(coef, ls, x, v, remaining=None):
    t = L.hit_body(x, v, dataclasses.replace(ls, coef=coef), remaining)
    return jnp.sum(jnp.where(jnp.isfinite(t), t, 0.0))


def _perturbed(dim):
    """The level-set circle / sphere with a smooth bump: a non-distance ``phi`` whose
    gradient norm varies."""
    ls = _levelset(dim)
    knots = L.knots(ls)
    bump = 0.4 * jnp.sin(1.3 * knots[:, 0]) * jnp.cos(0.7 * knots[:, 1])
    return dataclasses.replace(ls, coef=ls.coef + bump.reshape(ls.coef.shape))


@pytest.mark.parametrize("radius", [0, 1, 3])
def test_block_min_table_bounds_phi_on_the_cells_within_radius(radius):
    """``block_min_table`` at a point's lattice cell bounds ``phi`` from below on every
    segment of length ``<= radius h`` from it (at the point itself for radius 0), inside
    and beyond the lattice, and is attained closely enough to cull (positive far from the
    surface)."""
    ls = _perturbed(2)
    key = jax.random.PRNGKey(5 + radius)
    kx, kd, ks = jax.random.split(key, 3)
    n = 20000
    x = jax.random.uniform(kx, (n, 2), minval=-3.0, maxval=13.0)
    ang = jax.random.uniform(kd, (n,), maxval=2 * math.pi)
    vhat = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=1)
    lower = L.block_min_table(ls, radius)[L._table_index(ls, x, radius)]
    worst = jnp.full(n, jnp.inf)
    for frac in np.linspace(0.0, 1.0, 11):
        s = frac * radius * ls.h * jax.random.uniform(jax.random.fold_in(ks, int(frac * 10)), (n,))
        worst = jnp.minimum(worst, L.value(ls, x + s[:, None] * vhat))
    # the bound up to the round-off of the weights' sum; the sign exactly (what the cull uses)
    assert bool(jnp.all(worst >= lower - 1e-14 * jnp.abs(lower)))
    assert bool(jnp.all((worst > 0.0) | (lower <= 0.0)))
    assert float(jnp.mean(lower > 0.0)) > 0.3


@pytest.mark.parametrize(
    ("n", "far"),
    [(4096, False), (4096, True), (200, False), (200, True), (2600, "most_near")],
)
def test_culled_hit_search_is_the_dense_search_bitwise(monkeypatch, n, far):
    """The candidate cull of ``hit_body`` and ``inside`` (skip / compacted / dense branches,
    and the small-N ``cond``) returns the search over every ray, ray for ray, bitwise, and
    the same hit-time derivatives to round-off; ``inside`` is ``phi < 0`` bitwise."""
    ls = _perturbed(2)
    grid = _grid(2)
    key = jax.random.PRNGKey(11)
    kx, kv, kr = jax.random.split(key, 3)
    if far is True:
        # every ray far from the body: the search is skipped
        x = jnp.stack([jax.random.uniform(kx, (n,), minval=0.1, maxval=0.9)] * 2, axis=1)
    elif far == "most_near":
        # rays on a ring just outside the surface: more candidates than the compaction holds
        ang = jax.random.uniform(kx, (n,), maxval=2 * math.pi)
        x = 5.0 + 3.05 * jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=1)
    else:
        x, _v = _outside(kx, n, 2, grid, _body(2))
    n = x.shape[0]
    v = 700.0 * jax.random.normal(kv, (n, 3))
    remaining = jax.random.uniform(kr, (n,), minval=1e-5, maxval=1e-4)
    # both compiled (eager dispatch rounds differently from a fused program)
    culled = _jit_hit(x, v, ls, remaining)
    culled_reach = _jit_hit(x, v, ls, None)
    inside_culled = jax.jit(L.inside)(ls, x)
    with monkeypatch.context() as m:
        m.setattr(L, "_culled", _dense)
        dense = _jit_hit(x, v, ls, remaining)
        dense_reach = _jit_hit(x, v, ls, None)
        g_dense = jax.jit(jax.grad(_t_sum))(ls.coef, ls, x, v, remaining)
    np.testing.assert_array_equal(np.asarray(culled), np.asarray(dense))
    np.testing.assert_array_equal(np.asarray(culled_reach), np.asarray(dense_reach))
    want = jax.jit(L.value)(ls, x) < 0.0
    np.testing.assert_array_equal(np.asarray(inside_culled), np.asarray(want))
    g_culled = jax.jit(jax.grad(_t_sum))(ls.coef, ls, x, v, remaining)
    # the transpose's scatter-add into coef sums in the order of its program: round-off
    scale = max(float(np.abs(np.asarray(g_dense)).max()), 1e-300)
    np.testing.assert_allclose(
        np.asarray(g_culled), np.asarray(g_dense), rtol=0, atol=1e-13 * scale
    )
    hits = int(np.isfinite(np.asarray(dense_reach)).sum())
    if far is True:
        assert hits == 0
    else:
        assert hits > 0


def test_inside_cull_is_exact_for_points_inside_the_body():
    """``inside`` on points spread over the body, its surface band and beyond the lattice
    equals ``phi < 0`` bitwise, on every branch of the cull."""
    ls = _perturbed(2)
    key = jax.random.PRNGKey(4)
    for n, lo, hi in [(5000, 1.0, 9.0), (5000, -3.0, 13.0), (100, 2.0, 8.0)]:
        x = jax.random.uniform(jax.random.fold_in(key, n + int(lo)), (n, 2), minval=lo, maxval=hi)
        want = np.asarray(L.value(ls, x) < 0.0)
        np.testing.assert_array_equal(np.asarray(L.inside(ls, x)), want)
        assert want.any()


# --- the refinement against the bisection it replaced -------------------------------------


def _bisection_hit(x, v, ls, remaining):
    """The hit search before 2026-09-30: sampled bracket (a loop), ``n_bisect`` bisections,
    one Newton polish -- the reference of the safeguarded Newton refinement."""
    d = x.shape[1]
    vd = v[:, :d]
    speed = jnp.linalg.norm(vd, axis=1)
    moving = speed > 0.0
    vhat = vd / jnp.where(moving, speed, 1.0)[:, None]
    length = jnp.minimum(ls.reach, jnp.where(moving, speed, 1.0) * remaining)
    ds = length / ls.n_sample
    phi0, g0 = L.value_and_grad(ls, x)
    tol = L.SURFACE_TOL * ls.h
    inward = jnp.sum(vhat * g0, axis=1) < 0.0
    hit_now = (phi0 <= 0.0) & (phi0 >= -tol) & inward & moving
    deep = phi0 < -tol
    leaving = (jnp.abs(phi0) <= tol) & ~inward
    phi_prev = jnp.where(leaving, jnp.maximum(phi0, jnp.finfo(phi0.dtype).tiny), phi0)
    fr, xf, vf = jax.lax.stop_gradient((ls, x, vhat))
    found = jnp.zeros(x.shape[0], bool)
    s_lo = s_hi = jnp.zeros(x.shape[0])
    for i in range(1, ls.n_sample + 1):
        s = i * ds
        phi = L.value(fr, xf + s[:, None] * vf)
        cross = ~found & (phi_prev > 0.0) & (phi <= 0.0)
        s_lo = jnp.where(cross, s - ds, s_lo)
        s_hi = jnp.where(cross, s, s_hi)
        found, phi_prev = found | cross, phi
    for _ in range(ls.n_bisect):
        mid = 0.5 * (s_lo + s_hi)
        positive = L.value(fr, xf + mid[:, None] * vf) > 0.0
        s_lo, s_hi = jnp.where(positive, mid, s_lo), jnp.where(positive, s_hi, mid)
    phi_b, g_b = L.value_and_grad(ls, x + s_hi[:, None] * vhat)
    den = jnp.sum(vhat * g_b, axis=1)
    floor = 1e-12 * jnp.linalg.norm(g_b, axis=1)
    den = jnp.where(jnp.abs(den) < floor, -floor, den)
    den = jnp.where(den == 0.0, -1.0, den)
    s = s_hi - L.regularised_quotient(phi_b, den, ls.grazing * jnp.linalg.norm(g_b, axis=1))
    t = jnp.maximum(s, 0.0) / jnp.where(moving, speed, 1.0)
    ok = found & moving & ~deep & ~hit_now
    return jnp.where(hit_now, 0.0, jnp.where(ok, t, jnp.inf))


def _t_sum_bisection(coef, ls, x, v, remaining):
    t = _bisection_hit(x, v, dataclasses.replace(ls, coef=coef), remaining)
    return jnp.sum(jnp.where(jnp.isfinite(t), t, 0.0))


@pytest.mark.parametrize("dim", [2, 3])
def test_refinement_matches_the_bisection_reference(dim):
    """Safeguarded Newton vs ``n_bisect`` bisections, both then Newton-polished: the same
    rays hit (the sampled bracket decides), the hit points agree to the round-off of the
    root (``|phi|`` noise over the slope) and the ``coef`` derivatives to round-off -- on
    rays from outside the bumped surface, many of them grazing it.  (Starts on the surface
    are left out: their ``t = 0`` / ``leaving`` rules turn on the sign of a round-off
    ``phi``, which any two compiled programs may evaluate differently.)"""
    ls = dataclasses.replace(_perturbed(dim), n_sample=16)
    body = _body(dim)
    key = jax.random.PRNGKey(21)
    k1, k2, k3 = jax.random.split(key, 3)
    n = 6000
    u = jax.random.normal(k1, (n, dim))
    u = u / jnp.linalg.norm(u, axis=1, keepdims=True)
    rad = float(body.radius) * jax.random.uniform(k2, (n,), minval=0.6, maxval=1.4)
    x = body.centre + rad[:, None] * u
    x = x[np.abs(np.asarray(L.value(ls, x))) > 1e-3]
    v = 500.0 * jax.random.normal(k3, (x.shape[0], 3))
    if dim == 2:
        v = v.at[:, 2].set(0.0)
    remaining = jnp.full(x.shape[0], 1e-3)  # the whole reach
    new = np.asarray(jax.jit(L.hit_body)(x, v, ls, remaining))
    ref = np.asarray(jax.jit(_bisection_hit)(x, v, ls, remaining))
    np.testing.assert_array_equal(np.isfinite(new), np.isfinite(ref))
    hit = np.isfinite(ref)
    assert hit.sum() > 300
    speed = np.linalg.norm(np.asarray(v)[:, :dim], axis=1)
    np.testing.assert_allclose(new[hit] * speed[hit], ref[hit] * speed[hit], rtol=0, atol=1e-12)
    g_new = jax.jit(jax.grad(_t_sum))(ls.coef, ls, x, v, remaining)
    g_ref = jax.jit(jax.grad(_t_sum_bisection))(ls.coef, ls, x, v, remaining)
    scale = float(np.abs(np.asarray(g_ref)).max())
    np.testing.assert_allclose(np.asarray(g_new), np.asarray(g_ref), rtol=0, atol=1e-10 * scale)
