"""Embedded-surface tracking: the stencil and cell-walk trackers of `wall_adjoint.advance`.

Billiards against exact reflection integrators (analytic circle / sphere and the SPARTA
50-gon), tracker equivalence, the diffuse law on a curved body, the diagnostics, tallies
and events, and the box-only invariance.  Everything runs on the CPU in seconds.
"""

from __future__ import annotations

import itertools
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_dsmc import wall_adjoint
from jax_dsmc.boundary import BC, Faces, WallParams
from jax_dsmc.mesh import cut
from jax_dsmc.mesh import surface as S
from jax_dsmc.mesh.types import AnalyticBody
from jax_dsmc.state import GridSpec

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KB = 1.380649e-23
MASS = 6.63e-26  # argon
T_WALL = 300.0
SIGMA = math.sqrt(KB * T_WALL / MASS)


def _grid(dim):
    if dim == 2:
        return GridSpec(lo=jnp.array([0.0, 0.0]), hi=jnp.array([10.0, 10.0]), n=(20, 20))
    return GridSpec(lo=jnp.array([-2.0] * 3), hi=jnp.array([2.0] * 3), n=(10, 10, 10))


def _closed(dim):
    return Faces((BC.SPECULAR,) * (2 * dim))


def _open(dim):
    return Faces((BC.OUTFLOW, BC.OUTFLOW) + (BC.SPECULAR,) * (2 * dim - 2))


def _wall(dim, acc=0.0):
    return WallParams.uniform(dim, T_WALL, acc=acc, n_models=1)


def _body(dim):
    if dim == 2:
        return AnalyticBody(centre=jnp.array([5.0, 5.0]), radius=jnp.float64(3.0))
    return AnalyticBody(centre=jnp.zeros(3), radius=jnp.float64(1.0))


def _polygon():
    return S.read_sparta_surf(os.path.join(ROOT, "examples", "mesh", "sparta", "data.circle"), 2)


def _outside_state(key, n, dim, grid, body, speed=337.0):
    kx, kv = jax.random.split(key)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    x = lo + jax.random.uniform(kx, (n, dim), dtype=jnp.float64) * (hi - lo)
    v = jax.random.normal(kv, (n, 3), dtype=jnp.float64) * speed
    if dim == 2:
        v = v.at[:, 2].set(0.0)
    r = float(body.radius)
    alive = jnp.sum((x - body.centre) ** 2, axis=1) > (r + 0.05) ** 2
    return x, v, alive


# --- exact reference integrators (numpy, one particle) ---------------------------------


def _reflect_box(x, v, lo, hi, t_rem):
    """Time to the box (all specular) and the reflected velocity there."""
    d = x.size
    best, axis = np.inf, -1
    for a in range(d):
        if v[a] > 0:
            t = (hi[a] - x[a]) / v[a]
        elif v[a] < 0:
            t = (lo[a] - x[a]) / v[a]
        else:
            continue
        if t < best:
            best, axis = t, a
    return best, axis


def _circle_hit(x, v, c, r):
    rel = x - c
    a = v @ v
    b = rel @ v
    cc = rel @ rel - r * r
    disc = b * b - a * cc
    if disc < 0 or b >= 0 or cc < 0:
        return np.inf
    t = (-b - math.sqrt(disc)) / a
    return t if t >= 0 else np.inf


def _polygon_hit(x, v, p1, p2, n, exclude):
    """Earliest from-outside hit over all segments (2-D), inclusive edges, lowest id on ties."""
    best, which = np.inf, -1
    for e in range(p1.shape[0]):
        if e == exclude:
            continue
        vn = v @ n[e]
        if vn >= 0:
            continue
        t = ((p1[e] - x) @ n[e]) / vn
        if t < 0:
            continue
        h = x + t * v
        d = p2[e] - p1[e]
        s = ((h - p1[e]) @ d) / (d @ d)
        if s < -1e-12 or s > 1 + 1e-12:
            continue
        if t < best - 1e-15 or (abs(t - best) <= 1e-15 and e < which):
            best, which = t, e
    return best, which


def _integrate(x, v, lo, hi, dt, body_hit, body_normal):
    """Exact specular billiards in a closed specular box with one body; returns the state
    and the sequence of surfaces hit (box faces ``2*axis+side``, the body ``2*D``)."""
    d = x.size
    t_rem = dt
    hits = []
    exclude = -1
    for _ in range(64):
        tb, axis = _reflect_box(x, v[:d], lo, hi, t_rem)
        te, which = body_hit(x, v[:d], exclude)
        if te < tb and te <= t_rem:
            x = x + te * v[:d]
            n = body_normal(x, which)
            v = v - 2.0 * (v[:d] @ n) * np.concatenate([n, np.zeros(3 - d)])
            t_rem -= te
            hits.append(2 * d + (which if which >= 0 else 0))
            exclude = which
        elif tb <= t_rem:
            x = x + tb * v[:d]
            x[axis] = hi[axis] if v[axis] > 0 else lo[axis]
            v[axis] = -v[axis]
            t_rem -= tb
            hits.append(2 * axis + (1 if v[axis] < 0 else 0))
            exclude = -1
        else:
            return x + t_rem * v[:d], v, hits
    raise AssertionError("reference integrator did not finish")


def _advance(x, v, alive, grid, faces, wall, mesh, tracker, dt, key=0, **kw):
    return wall_adjoint.advance(
        x,
        v,
        alive,
        grid,
        faces,
        wall,
        MASS,
        dt,
        jax.random.PRNGKey(key),
        mesh=mesh,
        tracker=tracker,
        max_bounces=kw.pop("max_bounces", 32),
        max_walk=kw.pop("max_walk", 96),
        **kw,
    )


# --- billiards -----------------------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("tracker", ["stencil", "walk"])
def test_billiards_analytic_body(dim, tracker):
    grid = _grid(dim)
    body = _body(dim)
    # a stencil of the whole box: the candidate search is then the brute force one
    mesh = cut.build_mesh(body, grid, radius=4 * max(grid.n), sub=4)
    x, v, alive = _outside_state(jax.random.PRNGKey(1), 200, dim, grid, body, speed=1500.0)
    # a long move (|v| dt ~ 5 units): several box and body reflections per particle
    dt = 3.0e-3 if dim == 2 else 1.5e-3
    xo, vo, ao, counts, _ = _advance(x, v, alive, grid, _closed(dim), _wall(dim), mesh, tracker, dt)
    assert counts.exhausted == 0 and counts.inside_body == 0 and ao.sum() == alive.sum()
    if tracker == "stencil":
        assert counts.stencil_overrun == 0  # the dilation is the whole box
    c = np.asarray(body.centre)
    r = float(body.radius)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    n_body = 0
    for i in np.nonzero(np.asarray(alive))[0]:
        xr, vr, hits = _integrate(
            np.asarray(x[i]).copy(),
            np.asarray(v[i]).copy(),
            lo,
            hi,
            dt,
            lambda p, u, ex: (_circle_hit(p, u, c, r), -1),
            lambda p, w: (p - c) / r,
        )
        n_body += sum(h == 2 * dim for h in hits)
        np.testing.assert_allclose(np.asarray(xo[i]), xr, atol=1e-11)
        np.testing.assert_allclose(np.asarray(vo[i]), vr, rtol=1e-12, atol=1e-9)
    assert n_body > 20 and int(counts.surface_hits) == n_body
    # energy exact
    np.testing.assert_allclose(
        np.sum(np.asarray(vo[ao]) ** 2, axis=1), np.sum(np.asarray(v[ao]) ** 2, axis=1), rtol=1e-13
    )


@pytest.mark.parametrize("tracker", ["stencil", "walk"])
def test_billiards_polygon(tracker):
    grid = _grid(2)
    poly = _polygon()
    mesh = cut.build_mesh(poly, grid, radius=4 * max(grid.n))
    body = _body(2)
    x, v, alive = _outside_state(jax.random.PRNGKey(2), 200, 2, grid, body, speed=1500.0)
    dt = 3.0e-3
    xo, vo, ao, counts, _, events = _advance(
        x, v, alive, grid, _closed(2), _wall(2), mesh, tracker, dt, return_events=True
    )
    assert counts.exhausted == 0 and counts.inside_body == 0 and counts.stencil_overrun == 0
    pts = np.asarray(poly.points)
    el = np.asarray(poly.elements)
    p1, p2 = pts[el[:, 0]], pts[el[:, 1]]
    n = np.asarray(mesh.normals)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    face = np.asarray(events.face)
    n_body = 0
    for i in np.nonzero(np.asarray(alive))[0]:
        xr, vr, hits = _integrate(
            np.asarray(x[i]).copy(),
            np.asarray(v[i]).copy(),
            lo,
            hi,
            dt,
            lambda p, u, ex: _polygon_hit(p, u, p1, p2, n, ex),
            lambda p, w: n[w],
        )
        n_body += sum(h >= 4 for h in hits)
        np.testing.assert_allclose(np.asarray(xo[i]), xr, atol=1e-11)
        np.testing.assert_allclose(np.asarray(vo[i]), vr, rtol=1e-12, atol=1e-9)
        # the recorded element sequence is the reference's (box faces are not recorded)
        recorded = [int(f) for f in face[:, i] if f >= 0]
        assert recorded == [h for h in hits if h >= 4]
    assert n_body > 20 and int(counts.surface_hits) == n_body


def test_polygon_vs_analytic_within_sagitta():
    grid = _grid(2)
    body = _body(2)
    mesh_a = cut.build_mesh(body, grid, radius=2, sub=4)
    mesh_p = cut.build_mesh(_polygon(), grid, radius=2)
    x, v, alive = _outside_state(jax.random.PRNGKey(3), 20000, 2, grid, body)
    dt = 5.0e-4
    out_a = _advance(x, v, alive, grid, _open(2), _wall(2), mesh_a, "stencil", dt)
    out_p = _advance(x, v, alive, grid, _open(2), _wall(2), mesh_p, "stencil", dt)
    hit_a = np.asarray(out_a[4] != 0) | (np.asarray(out_a[1]) != np.asarray(v)).any(axis=1)
    hit_p = (np.asarray(out_p[1]) != np.asarray(v)).any(axis=1)
    # the hit sets differ only for near-grazing rays: a few percent
    assert (hit_a != hit_p).mean() < 0.05 and hit_a.sum() > 100
    both = hit_a & hit_p
    sagitta = 3.0 * (1.0 - math.cos(math.pi / 50))
    # the exit velocity direction differs by at most the normal's error (one chord's angle)
    va, vp = np.asarray(out_a[1])[both], np.asarray(out_p[1])[both]
    cosang = np.sum(va * vp, axis=1) / np.sqrt(np.sum(va * va, axis=1) * np.sum(vp * vp, axis=1))
    assert np.arccos(np.clip(cosang, -1, 1)).max() < 2 * (2 * math.pi / 50) + 1e-9
    # and the positions by at most the sagitta times the residual flight (order 1e-2 here)
    assert np.abs(np.asarray(out_a[0])[both] - np.asarray(out_p[0])[both]).max() < 0.2
    assert sagitta < 6e-3


# --- tracker equivalence --------------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 3])
def test_trackers_agree(dim):
    grid = _grid(dim)
    body = _polygon() if dim == 2 else S.sphere(1.0, 2)
    mesh = cut.build_mesh(body, grid, radius=4 * max(grid.n), sub=4)
    x, v, alive = _outside_state(jax.random.PRNGKey(4), 3000, dim, grid, _body(dim))
    dt = 1.5e-3
    outs = {}
    for tracker in ("stencil", "walk"):
        outs[tracker] = _advance(
            x, v, alive, grid, _open(dim), _wall(dim), mesh, tracker, dt, return_events=True
        )
    xs, vs, als, cs, _, es = outs["stencil"]
    xw, vw, alw, cw, _, ew = outs["walk"]
    assert cs.stencil_overrun == 0 and cw.exhausted == 0 and cs.exhausted == 0
    assert int(cs.surface_hits) == int(cw.surface_hits) > 50
    assert bool((als == alw).all())
    # the walk moves the particle onto each cell face first: agreement to roundoff, not
    # bitwise; the recorded element sequences are identical
    np.testing.assert_allclose(np.asarray(xw), np.asarray(xs), atol=1e-12)
    np.testing.assert_allclose(np.asarray(vw), np.asarray(vs), rtol=1e-12, atol=1e-9)
    fs, fw = np.asarray(es.face), np.asarray(ew.face)
    for i in range(x.shape[0]):
        assert [f for f in fs[:, i] if f >= 0] == [f for f in fw[:, i] if f >= 0]


def _lane_case(kind):
    """(dim, grid, body for build_mesh, body for the outside test) of a lane-gate case."""
    from jax_dsmc.mesh import levelset as L  # noqa: PLC0415 -- only this case needs it

    if kind == "sphere":
        return 3, _grid(3), S.sphere(1.0, 2), _body(3)
    grid = _grid(2)
    if kind == "polygon":
        return 2, grid, _polygon(), _body(2)
    if kind == "levelset":
        lo, m = L.grid_for_box(grid.lo, grid.hi, 0.125, margin=3)
        body = L.from_analytic(jnp.array([5.0, 5.0]), 3.0, lo, 0.125, m, n_sample=16)
        return 2, grid, body, _body(2)
    return 2, grid, _body(2), _body(2)


@pytest.mark.parametrize("kind", ["analytic", "polygon", "levelset", "sphere"])
@pytest.mark.parametrize("acc", [0.0, 1.0])
def test_lane_tracker_equals_the_stencil(kind, acc):
    """The lane tracker (pass 0 over all, later passes on the compacted lane, the dense
    pass's draws gathered per slot) reproduces the dense stencil tracker: the same
    discrete outcome (alive, event faces, diffuse branches, hit counts) exactly and the
    floats to the last bit or two (XLA compiles the lane-sized batch with a different
    fusion, 1 ulp on ~0.03 % of diffuse re-emissions); the split events expand to the
    dense ones.  A diffuse box face and a closed box give
    multi-event steps; dt is large so particles meet several surfaces per step."""
    dim, grid, body, outside = _lane_case(kind)
    mesh = cut.build_mesh(body, grid, radius=4 * max(grid.n), sub=4)
    x, v, alive = _outside_state(jax.random.PRNGKey(11), 2000, dim, grid, outside)
    faces = Faces((BC.DIFFUSE,) + (BC.SPECULAR,) * (2 * dim - 1))
    wall = _wall(dim, acc)
    dt = 1.5e-3
    common = {"return_events": True, "return_tallies": True, "max_bounces": 12}
    ref = _advance(x, v, alive, grid, faces, wall, mesh, "stencil", dt, key=3, **common)
    got = _advance(
        x, v, alive, grid, faces, wall, mesh, "lane", dt, key=3, lane_size=x.shape[0], **common
    )
    xs, vs, als, cs, ls, es, ts = ref
    xl, vl, all_, cl, ll, el, tl = got
    assert int(cs.surface_hits) > 10 and int(cs.exhausted) == 0
    assert int(cl.lane_overflow) == 0
    np.testing.assert_array_equal(np.asarray(als), np.asarray(all_))
    for a, b in ((xs, xl), (vs, vl), (ls, ll)):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-14, atol=1e-12)
    assert int(cs.surface_hits) == int(cl.surface_hits)
    assert int(cs.stencil_overrun) == int(cl.stencil_overrun)
    # tallies: the same events summed in another order
    for a, b in zip(ts, tl, strict=True):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-12, atol=1e-30)
    dense = wall_adjoint.to_dense(el)
    for name in ("face", "diffuse"):
        np.testing.assert_array_equal(
            np.asarray(getattr(dense, name)), np.asarray(getattr(es, name))
        )
    np.testing.assert_allclose(np.asarray(dense.v_out), np.asarray(es.v_out), rtol=1e-14)
    # some slots really needed the later passes
    assert int(np.sum(np.asarray(es.face)[1:] >= 0)) > 0


def test_lane_overflow_is_counted_and_deleted():
    dim, grid, body, outside = _lane_case("analytic")
    mesh = cut.build_mesh(body, grid, radius=4 * max(grid.n), sub=4)
    x, v, alive = _outside_state(jax.random.PRNGKey(11), 2000, dim, grid, outside)
    out = _advance(x, v, alive, grid, _closed(2), _wall(2), mesh, "lane", 1.5e-3, lane_size=8)
    n_event = int(out[3].lane_overflow)
    assert n_event > 0
    assert int(np.sum(np.asarray(out[2]))) <= int(np.sum(np.asarray(alive))) - n_event


def test_lane_resolves_many_bounces():
    # the trapped particle of test_stuck_particle_is_deleted_and_counted: ~10 reflections
    grid = _grid(2)
    mesh = cut.build_mesh(_body(2), grid, radius=1)
    x = jnp.array([[5.0, 9.0], [1.0, 1.0]])
    v = jnp.array([[0.0, 1.0e5, 0.0], [10.0, 0.0, 0.0]])
    ref = _advance(x, v, jnp.ones(2, bool), grid, _closed(2), _wall(2), mesh, "stencil", 1e-4)
    out = _advance(x, v, jnp.ones(2, bool), grid, _closed(2), _wall(2), mesh, "lane", 1e-4)
    assert int(out[3].exhausted) == 0 and bool(out[2][0])
    np.testing.assert_array_equal(np.asarray(out[0]), np.asarray(ref[0]))


# --- box-only invariance ------------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("tracker", ["stencil", "walk"])
def test_box_only_invariance(dim, tracker):
    """A body no particle can reach leaves the box tracker's result bitwise unchanged."""
    grid = _grid(dim)
    lo, hi = np.asarray(grid.lo), np.asarray(grid.hi)
    h = (hi - lo) / np.asarray(grid.n)
    centre = jnp.asarray(lo + 0.5 * h)
    body = AnalyticBody(centre=centre, radius=jnp.float64(0.25 * float(h.min())))
    mesh = cut.build_mesh(body, grid, radius=1, sub=4)
    faces = Faces(
        (BC.DIFFUSE, BC.SPECULAR, BC.PERIODIC, BC.OUTFLOW) + (BC.SPECULAR,) * (2 * dim - 4)
    )
    wall = _wall(dim, acc=1.0)
    kx, kv = jax.random.split(jax.random.PRNGKey(5))
    n = 2000
    x = lo + h + jax.random.uniform(kx, (n, dim), dtype=jnp.float64) * (hi - lo - 2 * h)
    v = jax.random.normal(kv, (n, 3), dtype=jnp.float64) * 300.0
    alive = jnp.ones(n, bool)
    dt = 1e-4  # moves of < 0.1 cell: nothing reaches the corner cell
    ref = _advance(x, v, alive, grid, faces, wall, None, "stencil", dt, epsilon=0.0)
    out = _advance(x, v, alive, grid, faces, wall, mesh, tracker, dt, epsilon=0.0)
    if tracker == "stencil":
        assert bool((ref[0] == out[0]).all())
    else:
        # the walk advances to every crossed cell face first: positions to roundoff
        np.testing.assert_allclose(np.asarray(out[0]), np.asarray(ref[0]), rtol=0, atol=1e-13)
    assert bool((ref[1] == out[1]).all()) and bool((ref[2] == out[2]).all())
    assert bool((ref[4] == out[4]).all())
    assert out[3].surface_hits == 0 and out[3].inside_body == 0 and out[3].stencil_overrun == 0


# --- diffuse law on a curved body ---------------------------------------------------


@pytest.mark.parametrize("tracker", ["stencil", "walk"])
def test_diffuse_body_reemission_law(tracker):
    grid = _grid(2)
    body = _body(2)
    mesh = cut.build_mesh(body, grid, radius=2, sub=4)
    wall = _wall(2, acc=1.0)
    # every particle aimed at the centre from a ring just outside the body
    n = 20000
    k = jax.random.PRNGKey(6)
    ang = jax.random.uniform(k, (n,), dtype=jnp.float64) * 2 * math.pi
    ring = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=1)
    x = body.centre + 3.2 * ring
    v = jnp.concatenate([-400.0 * ring, jnp.zeros((n, 1))], axis=1)
    dt = 0.2 / 400.0 + 1e-9  # reaches the surface just before the end of the step
    xo, vo, ao, counts, _, events = _advance(
        x, v, jnp.ones(n, bool), grid, _open(2), wall, mesh, tracker, dt, return_events=True
    )
    assert int(counts.surface_hits) == n and counts.inside_body == 0
    n_hat = (np.asarray(xo) - np.asarray(body.centre)) / 3.0
    vo = np.asarray(vo)
    vn = np.sum(vo[:, :2] * n_hat, axis=1)
    vt = vo[:, 0] * (-n_hat[:, 1]) + vo[:, 1] * n_hat[:, 0]
    vz = vo[:, 2]
    # Rayleigh normal: mean sigma sqrt(pi/2); Gaussian tangential (both in-plane and z)
    se = SIGMA / math.sqrt(n)
    assert (vn > 0).all()
    assert abs(vn.mean() - SIGMA * math.sqrt(math.pi / 2)) < 4 * 0.66 * se
    assert abs(vt.mean()) < 4 * se and abs(vz.mean()) < 4 * se
    assert abs(vt.std() / SIGMA - 1) < 0.03 and abs(vz.std() / SIGMA - 1) < 0.03
    assert abs(np.mean(vn**2) / (2 * SIGMA**2) - 1) < 0.03
    assert bool(np.asarray(events.diffuse).any(axis=0).all())


# --- diagnostics, tallies, events, jit ------------------------------------------------


def test_stencil_overrun_and_walk_finish():
    grid = _grid(2)
    mesh = cut.build_mesh(_polygon(), grid, radius=1)
    # one fast particle crossing three cells straight at the body from a cell whose
    # radius-1 stencil does not contain the elements it hits
    x = jnp.array([[0.25, 5.1]])
    v = jnp.array([[2.0e4, 0.0, 0.0]])
    dt = 1e-4  # travels 2.0 = 4 cells; the body starts at x = 2
    alive = jnp.ones(1, bool)
    out_s = _advance(x, v, alive, grid, _open(2), _wall(2), mesh, "stencil", dt)
    assert out_s[3].stencil_overrun == 1
    # the missed hit leaves the particle inside the body: caught and killed
    assert out_s[3].inside_body == 1 and not bool(out_s[2][0])
    out_w = _advance(x, v, alive, grid, _open(2), _wall(2), mesh, "walk", dt)
    assert out_w[3].exhausted == 0 and out_w[3].surface_hits == 1 and bool(out_w[2][0])
    assert float(out_w[1][0, 0]) < 0.0
    # too few walk iterations: reported and the particle discarded (SPARTA MAXSTUCK)
    out_k = _advance(x, v, alive, grid, _open(2), _wall(2), mesh, "walk", dt, max_walk=2)
    assert out_k[3].exhausted == 1 and not bool(out_k[2][0])


def test_stuck_particle_is_deleted_and_counted():
    # a particle bouncing between the body's top (y = 8) and the specular box face y = 10:
    # ~10 reflections in one step.  With the budget it finishes; with 4 passes the stencil
    # tracker deletes it (SPARTA MAXSTUCK) and counts it instead of leaving an unresolved hit
    grid = _grid(2)
    mesh = cut.build_mesh(_body(2), grid, radius=1)
    x = jnp.array([[5.0, 9.0], [1.0, 1.0]])
    v = jnp.array([[0.0, 1.0e5, 0.0], [10.0, 0.0, 0.0]])
    dt = 1e-4  # 10.0 of travel: ~5 round trips across the 2.0 gap
    alive = jnp.ones(2, bool)
    full = _advance(x, v, alive, grid, _closed(2), _wall(2), mesh, "stencil", dt)
    assert full[3].exhausted == 0 and bool(full[2][0])
    short = _advance(x, v, alive, grid, _closed(2), _wall(2), mesh, "stencil", dt, max_bounces=4)
    assert short[3].exhausted == 1 and short[3].inside_body == 0
    assert not bool(short[2][0]) and bool(short[2][1])


def test_inside_body_is_killed_and_counted():
    grid = _grid(2)
    mesh = cut.build_mesh(_polygon(), grid, radius=1)
    x = jnp.array([[5.0, 5.0], [1.0, 1.0]])
    v = jnp.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    for tracker in ("stencil", "walk"):
        out = _advance(x, v, jnp.ones(2, bool), grid, _open(2), _wall(2), mesh, tracker, 1e-4)
        assert out[3].inside_body == 1
        assert not bool(out[2][0]) and bool(out[2][1])


def test_tallies_and_events_on_elements():
    grid = _grid(2)
    poly = _polygon()
    mesh = cut.build_mesh(poly, grid, radius=2)
    x, v, alive = _outside_state(jax.random.PRNGKey(7), 3000, 2, grid, _body(2))
    dt = 1e-3
    xo, vo, ao, counts, _, events, tally = _advance(
        x,
        v,
        alive,
        grid,
        _open(2),
        _wall(2),
        mesh,
        "stencil",
        dt,
        return_events=True,
        return_tallies=True,
    )
    E = poly.n_elements
    assert tally.hits.shape == (4 + E,)
    assert int(tally.hits[4:].sum()) == int(counts.surface_hits) > 50
    # every recorded element hit is in the tally, and the momentum tally is the impulse
    face = np.asarray(events.face)
    rec = face[face >= 0]
    assert (rec >= 4).all()
    counts_rec = np.bincount(rec, minlength=4 + E)
    np.testing.assert_array_equal(counts_rec[4:], np.asarray(tally.hits)[4:])
    assert int(np.asarray(tally.hits)[:4].sum()) > 0  # box faces are tallied, not recorded
    # specular: momentum transfer is along the element normal, energy transfer zero
    mom = np.asarray(tally.momentum)[4:, :2]
    n = np.asarray(mesh.normals)
    tang = mom[:, 0] * (-n[:, 1]) + mom[:, 1] * n[:, 0]
    assert np.abs(tang).max() < 1e-30 + 1e-12 * np.abs(mom).max()
    assert (np.sum(mom * n, axis=1) <= 0).all()  # the flow pushes the surface inward (-n)
    assert (
        np.abs(np.asarray(tally.energy)[4:]).max() < 1e-12 * MASS * 337.0**2
    )  # outflow rows carry the lost energy
    # exclude: no element is recorded twice in a row for the same particle
    for i in range(face.shape[1]):
        seq = [f for f in face[:, i] if f >= 0]
        assert all(a != b for a, b in itertools.pairwise(seq))


def test_periodic_and_outflow_with_walk():
    grid = _grid(2)
    mesh = cut.build_mesh(_body(2), grid, radius=1, sub=4)
    faces = Faces((BC.OUTFLOW, BC.OUTFLOW, BC.PERIODIC, BC.PERIODIC))
    x = jnp.array([[9.9, 9.9], [9.9, 5.0], [0.3, 9.95]])
    v = jnp.array([[0.0, 1.0e3, 0.0], [1.0e3, 0.0, 0.0], [-1.0e3, 1.0e3, 0.0]])
    out = _advance(x, v, jnp.ones(3, bool), grid, faces, _wall(2), mesh, "walk", 1e-3)
    xo, alive = np.asarray(out[0]), np.asarray(out[2])
    assert out[3].exhausted == 0
    assert alive[0] and abs(xo[0, 1] - 0.9) < 1e-12  # wrapped through the periodic face
    assert not alive[1]  # left through the outflow face
    assert not alive[2]  # wrapped in y, then left through x = 0


def test_advance_jits_with_mesh():
    grid = _grid(2)
    mesh = cut.build_mesh(_body(2), grid, radius=1, sub=4)
    x, v, alive = _outside_state(jax.random.PRNGKey(8), 500, 2, grid, _body(2))
    f = jax.jit(
        wall_adjoint._advance_impl,
        static_argnames=(
            "faces",
            "max_bounces",
            "return_tallies",
            "return_events",
            "tracker",
            "max_walk",
        ),
    )
    for tracker in ("stencil", "walk"):
        out = f(
            x,
            v,
            alive,
            grid,
            _open(2),
            _wall(2),
            MASS,
            1e-4,
            jax.random.PRNGKey(0),
            mesh=mesh,
            tracker=tracker,
        )
        ref = _advance(x, v, alive, grid, _open(2), _wall(2), mesh, tracker, 1e-4)
        assert bool((out[0] == ref[0]).all()) and bool((out[1] == ref[1]).all())


@pytest.mark.parametrize("kind", ["analytic", "polygon", "levelset", "sphere"])
def test_compact_record_keeps_every_later_pass_event(kind):
    """The stored form of a lane step (``RecordEvents``: pass 0 dense, the later passes as a
    list) expands on the host to exactly the dense events; a list too short counts the
    events it could not keep."""
    from jax_dsmc.ensemble_backward import _step_walls  # noqa: PLC0415

    dim, grid, body, outside = _lane_case(kind)
    mesh = cut.build_mesh(body, grid, radius=4 * max(grid.n), sub=4)
    x, v, alive = _outside_state(jax.random.PRNGKey(11), 2000, dim, grid, outside)
    faces = Faces((BC.DIFFUSE,) * 2 + (BC.SPECULAR,) * (2 * dim - 2))
    out = _advance(
        x,
        v,
        alive,
        grid,
        faces,
        _wall(dim, 1.0),
        mesh,
        "lane",
        1.5e-3,
        key=3,
        return_events=True,
        max_bounces=12,
        lane_size=x.shape[0],
    )
    split = out[5]
    dense = wall_adjoint.to_dense(split)
    n_later = int(np.sum(np.asarray(dense.face)[1:] >= 0))
    assert n_later >= 1
    record, lost = wall_adjoint.compact_record(split, n_later + 3)
    assert int(lost) == 0
    stacked = jax.tree.map(lambda a: a[None], record)  # one captured step
    face, v_out, diffuse = _step_walls(stacked, 0)
    rows = face.shape[0]
    np.testing.assert_array_equal(face, np.asarray(dense.face)[:rows])
    assert (np.asarray(dense.face)[rows:] < 0).all()
    np.testing.assert_array_equal(v_out, np.asarray(dense.v_out)[:rows])
    np.testing.assert_array_equal(diffuse, np.asarray(dense.diffuse)[:rows])
    _short, lost = wall_adjoint.compact_record(split, n_later - 1)
    assert int(lost) == 1
