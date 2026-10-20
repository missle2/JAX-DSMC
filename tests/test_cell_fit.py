"""The per-cell field fit of the virtual pull-back (jax_dsmc.cell_fit) and its plumbing
through ensemble_device.backward_virtual: exact on polynomials, the least-squares solution,
the order guard, finite empty sides, exact sigma points, and the complex-cell fallback equal
to the per-query one-sided fit."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples import ensemble_relaxation as R
from jax_dsmc import cell_fit, ensemble_device, simulate
from jax_dsmc.collide.vss import scatter
from jax_dsmc.state import CollideState, GridSpec, ParticleState, SimParams, SimStatic

D, C, P = 2, 6, 64
COUNTS = np.array([60, 40, 25, 14, 3, 0])
SCALE = jnp.asarray([0.1, 0.1, 2.0, 2.0, 2.0])


def _tables(seed=0):
    rng = np.random.default_rng(seed)
    valid = np.arange(P)[None, :] < COUNTS[:, None]
    x = rng.uniform(0, 0.1, (C, P, D)) + np.arange(C)[:, None, None] * 0.1
    v = rng.normal(0, 2.0, (C, P, 3)) + 1.5
    return x, v, valid, rng


def _poly(x, v, poly, seed=1):
    o = np.concatenate((x / 0.1, v / 2.0), -1)
    terms = cell_fit.monomials(poly, D)
    coef = np.random.default_rng(seed).normal(size=len(terms))
    return sum(
        c * np.prod([o[..., a] for a in t], axis=0) for c, t in zip(coef, terms, strict=True)
    )


@pytest.mark.parametrize("poly", cell_fit.POLYS)
def test_reproduces_its_own_polynomials(poly):
    x, v, valid, _ = _tables()
    lam = _poly(x, v, poly)
    basis = f"cell:{poly}:ridge0"
    f = cell_fit.fit(jnp.asarray(x), jnp.asarray(v), jnp.asarray(valid), jnp.asarray(lam), SCALE,
                     basis)  # fmt: skip
    c = jnp.asarray(np.repeat(np.arange(C), P))
    got = cell_fit.evaluate(f, c, jnp.asarray(x.reshape(-1, D)), jnp.asarray(v.reshape(-1, 3)),
                            SCALE, basis)  # fmt: skip
    ok = valid & (COUNTS > len(cell_fit.monomials(poly, D)))[:, None]
    err = np.abs(np.asarray(got).reshape(C, P) - lam)[ok].max() / np.abs(lam).max()
    assert err < 1e-8


def test_is_the_least_squares_fit_and_its_slopes_are_the_polynomial_derivative():
    x, v, valid, rng = _tables()
    lam = rng.normal(size=(C, P))
    f = cell_fit.fit(jnp.asarray(x), jnp.asarray(v), jnp.asarray(valid), jnp.asarray(lam), SCALE,
                     "cell:quad:ridge0")  # fmt: skip
    n = COUNTS[0]
    z = np.concatenate((x[0, :n], v[0, :n]), -1) / np.asarray(SCALE)
    design = np.asarray(cell_fit.features(jnp.asarray(z - z.mean(0)), "quad"))
    beta = np.linalg.lstsq(design, lam[0, :n], rcond=None)[0]
    np.testing.assert_allclose(np.asarray(f.beta[0]), beta, rtol=1e-8, atol=1e-10)
    lam = _poly(x, v, "quad")
    f = cell_fit.fit(jnp.asarray(x), jnp.asarray(v), jnp.asarray(valid), jnp.asarray(lam), SCALE,
                     "cell:quad:ridge0")  # fmt: skip
    qx, qv = x[0, :4], v[0, :4]
    _, gx, gv = cell_fit.evaluate(f, jnp.zeros(4, jnp.int32), jnp.asarray(qx), jnp.asarray(qv),
                                  SCALE, "cell:quad:ridge0", gradients=True)  # fmt: skip
    eps, num = 1e-6, []
    for a in range(D + 3):
        dz = np.zeros(D + 3)
        dz[a] = eps
        up = _poly(qx + dz[:D], qv + dz[D:], "quad")
        dn = _poly(qx - dz[:D], qv - dz[D:], "quad")
        num.append((up - dn) / (2 * eps))
    got = np.concatenate((np.asarray(gx), np.asarray(gv)), 1)
    np.testing.assert_allclose(got, np.stack(num, 1), rtol=1e-6, atol=1e-8)


def test_order_guard_caps_the_columns_by_the_node_count():
    x, v, valid, rng = _tables()
    lam = rng.normal(size=(C, P))
    f = cell_fit.fit(jnp.asarray(x), jnp.asarray(v), jnp.asarray(valid), jnp.asarray(lam), SCALE,
                     "cell:quad:adapt1.5")  # fmt: skip
    order = np.asarray(cell_fit.orders("quad", D))
    top = [order[np.abs(np.asarray(f.beta[c])) > 0].max(initial=-1) for c in range(C)]
    # columns 1 / 6 / 14 by order: 60, 40, 25 nodes -> quadratic; 14 -> plane; 3 -> mean
    assert top == [2, 2, 2, 1, 0, -1]


def _leverage(z_nodes, z_query, poly):
    """The plain least-squares leverage of queries (unscaled numpy reference)."""
    centre = z_nodes.mean(0)
    f = np.asarray(cell_fit.features(jnp.asarray(z_nodes - centre), poly))
    fq = np.asarray(cell_fit.features(jnp.asarray(z_query - centre), poly))
    return np.einsum("ni,ij,nj->n", fq, np.linalg.inv(f.T @ f), fq)


def test_leverage_guard_takes_the_highest_order_the_nodes_determine_at_the_query():
    x, v, valid, rng = _tables()
    lam = rng.normal(size=(C, P))
    args = (jnp.asarray(x), jnp.asarray(v), jnp.asarray(valid), jnp.asarray(lam), SCALE)
    # queries in cell 0: its own nodes (interpolation) and far outside its velocity cloud
    qx = np.repeat(x[0, :1], 4, axis=0)
    qv = np.array([v[0, 0], v[0, 1], [12.0, 9.0, -9.0], [30.0, 0.0, 0.0]])
    c = jnp.zeros(4, jnp.int32)

    def at(basis, f):
        return np.asarray(cell_fit.evaluate(f, c, jnp.asarray(qx), jnp.asarray(qv), SCALE, basis))

    plain = at("cell:cubicv:ridge0", cell_fit.fit(*args, "cell:cubicv:ridge0"))
    # a guard nothing reaches is the plain fit
    loose = "cell:cubicv:ridge0:guard1e30"
    np.testing.assert_allclose(at(loose, cell_fit.fit(*args, loose)), plain, rtol=1e-9)
    # a guard below every order's leverage is the cell mean
    tight = "cell:cubicv:ridge0:guard1e-12"
    np.testing.assert_allclose(at(tight, cell_fit.fit(*args, tight)), lam[0, :60].mean(), rtol=1e-9)
    # guard 1: each query the highest order whose own leverage is <= 1 (numpy reference)
    z = np.concatenate((x[0, :60], v[0, :60]), -1) / np.asarray(SCALE)
    zq = np.concatenate((qx, qv), -1) / np.asarray(SCALE)
    want = []
    for i in range(4):
        level = 0
        for g, poly in ((1, "lin"), (2, "quad"), (3, "cubicv")):
            if _leverage(z, zq[i : i + 1], poly)[0] <= 1.0:
                level = g
        basis = f"cell:{('mean', 'lin', 'quad', 'cubicv')[level]}:ridge0"
        want.append(at(basis, cell_fit.fit(*args, basis))[i])
    guarded = "cell:cubicv:ridge0:guard1"
    np.testing.assert_allclose(at(guarded, cell_fit.fit(*args, guarded)), want, rtol=1e-8)
    assert want[0] == pytest.approx(plain[0]) and want[3] != pytest.approx(plain[3])


def test_leverage_guard_trace_counts_the_live_queries():
    x, v, valid, rng = _tables()
    lam = rng.normal(size=(C, P))
    basis = "cell:cubicv:adapt1.5:guard4"
    f = cell_fit.fit(jnp.asarray(x), jnp.asarray(v), jnp.asarray(valid), jnp.asarray(lam), SCALE,
                     basis)  # fmt: skip
    c = jnp.asarray(np.repeat(np.arange(C), P))
    live = jnp.asarray(valid.reshape(-1))
    cell_fit.TRACE = {}
    try:
        jax.block_until_ready(cell_fit.evaluate(f, c, jnp.asarray(x.reshape(-1, D)),
                                                jnp.asarray(v.reshape(-1, 3)), SCALE, basis,
                                                q_valid=live))  # fmt: skip
        counts = cell_fit.TRACE[C]
    finally:
        cell_fit.TRACE = None
    np.testing.assert_array_equal(counts.sum(1), COUNTS)
    # adapt still caps the order: 14 nodes -> at most a plane, 3 -> the mean
    assert counts[3, 2:].sum() == 0 and counts[4, 1:].sum() == 0


def test_split_fit_is_finite_with_an_empty_side_and_defers_to_the_unsplit_fit():
    x, v, valid, rng = _tables()
    lam = rng.normal(size=(C, P))
    v = np.abs(v)  # every node moves along +x: the "behind" side of n = +x is empty
    node_normal = np.zeros((C, P, 3))
    node_normal[..., 0] = 1.0
    normal, complex_ = cell_fit.cell_normals(jnp.asarray(node_normal), jnp.asarray(valid), 0.9)
    basis = "cell:quad:adapt1.5"
    f = cell_fit.fit_split(jnp.asarray(x), jnp.asarray(v), jnp.asarray(valid), jnp.asarray(lam),
                           jnp.asarray(node_normal), normal, complex_, SCALE, basis)  # fmt: skip
    assert not bool(jnp.any(complex_))
    qv = jnp.asarray([[-1.0, 0.0, 0.0]] * 3)  # queries on the empty side
    qx = jnp.asarray(x[:3, 0])
    qn = jnp.asarray([[1.0, 0.0, 0.0]] * 3)
    got = cell_fit.evaluate(f, jnp.arange(3), qx, qv, SCALE, basis, q_normal=qn)
    want = cell_fit.evaluate(f.both, jnp.arange(3), qx, qv, SCALE, basis)
    assert np.all(np.isfinite(np.asarray(got)))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize("alpha", [1.0, 1.4])
def test_sigma_points_are_the_scattering_expectation_of_quadratics(alpha):
    rng = np.random.default_rng(0)
    vi, vj = jnp.asarray(rng.normal(size=(3, 3))), jnp.asarray(rng.normal(size=(3, 3)))
    sp = type("SP", (), {"alpha": jnp.full((1, 1), alpha), "mass": jnp.ones(1)})()
    a = rng.normal(size=(3, 3))
    a = jnp.asarray(a + a.T)
    pi, _ = ensemble_device._scatter_sigma(vi, vj, sp)
    sigma = sum(jnp.einsum("na,ab,nb->n", p, a, p) for p in pi) / 6
    m = 200_000
    u = jnp.asarray(rng.uniform(size=(2, m, 1)))
    vin, _ = scatter(vi[None], vj[None], 0, 0, sp, u[0], u[1])
    mc = jnp.einsum("mna,ab,mnb->mn", vin, a, vin)
    se = jnp.std(mc, 0) / np.sqrt(m)
    assert float(jnp.max(jnp.abs(sigma - jnp.mean(mc, 0)) / se)) < 4.5


def test_complex_cell_fallback_is_the_per_query_one_sided_backward():
    """Every slot with a normal and every cell complex (a resultant cannot exceed 1): the
    hybrid answers every query with the per-query one-sided fit -- the same backward."""
    dim, cells, per = 2, 9, 30
    n = cells * per
    key = jax.random.key(4)
    grid = GridSpec(jnp.zeros(dim), jnp.ones(dim), (3, 3))
    static = SimStatic(n, dim, 1, 1, 256, 1, 3, majorant_mode="certified")
    params = SimParams(jnp.float64(0.2), jnp.full(cells, 1.0 / n), jnp.float64(0.0))
    kx, kv = jax.random.split(key)
    state = ParticleState(
        jax.random.uniform(kx, (n, dim), dtype=jnp.float64),
        jax.random.normal(kv, (n, 3), dtype=jnp.float64), jnp.zeros(n, jnp.int32),
        jnp.ones(n, bool), jnp.zeros(n),
    )  # fmt: skip
    (out, _), aux = simulate.run(
        state, CollideState(jnp.zeros((cells, 1, 1))), grid, R.species(R.CONTROLS), params,
        static, key, collide_only=True, emit_moments="none", capture=True,
    )  # fmt: skip
    rec = aux[2]
    normals = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0]), (*rec.x.shape[:2], 3))
    scale = jnp.asarray([1 / 3, 1 / 3, 1.0, 1.0, 1.0])
    basis = "cell:quad:align1.01"
    cap = ensemble_device.fallback_capacity(rec, normals, grid, 64, basis)
    assert cap >= n
    run = [
        ensemble_device.backward_virtual(
            rec, state.alive, out, out.v[:, 0] ** 2 / n, grid, scale, R.species(R.CONTROLS),
            jax.random.key(9), 64, 32, b, 512, 2, 4, "staged", None, normals=normals,
            fallback_capacity=c,
        ).lambda0
        for b, c in (("quadratic_full", 0), (basis, cap))
    ]  # fmt: skip
    np.testing.assert_allclose(np.asarray(run[1]), np.asarray(run[0]), rtol=1e-12, atol=1e-15)


def test_leverage_guard_through_the_virtual_backward():
    """A guard nothing reaches is the unguarded backward; a finite guard runs, stays finite and
    counts every live query (the collisional pull-back, no normals)."""
    dim, cells, per = 2, 4, 60
    n = cells * per
    key = jax.random.key(5)
    grid = GridSpec(jnp.zeros(dim), jnp.ones(dim), (2, 2))
    static = SimStatic(n, dim, 1, 1, 256, 1, 3, majorant_mode="certified")
    params = SimParams(jnp.float64(0.2), jnp.full(cells, 1.0 / n), jnp.float64(0.0))
    kx, kv = jax.random.split(key)
    v = jax.random.normal(kv, (n, 3), dtype=jnp.float64)
    v = v.at[: n // 2, 0].add(8.0)  # a beam and a cold gas in every cell
    state = ParticleState(
        jax.random.uniform(kx, (n, dim), dtype=jnp.float64), v, jnp.zeros(n, jnp.int32),
        jnp.ones(n, bool), jnp.zeros(n),
    )  # fmt: skip
    (out, _), aux = simulate.run(
        state, CollideState(jnp.zeros((cells, 1, 1))), grid, R.species(R.CONTROLS), params,
        static, key, collide_only=True, emit_moments="none", capture=True,
    )  # fmt: skip
    scale = jnp.asarray([1 / 2, 1 / 2, 1.0, 1.0, 1.0])

    def lam0(basis):
        return ensemble_device.backward_virtual(
            aux[2], state.alive, out, out.v[:, 0] ** 3 / n, grid, scale, R.species(R.CONTROLS),
            jax.random.key(9), 128, 32, basis, 512, 2, 4, "staged", None,
        ).lambda0  # fmt: skip

    plain = lam0("cell:cubicv:adapt1.5")
    loose = lam0("cell:cubicv:adapt1.5:guard1e30")
    np.testing.assert_allclose(np.asarray(loose), np.asarray(plain), rtol=1e-10, atol=1e-14)
    cell_fit.TRACE = {}
    try:
        guarded = lam0("cell:cubicv:adapt1.5:guard4")
        counts = cell_fit.take_trace()
    finally:
        cell_fit.TRACE = None
    assert np.all(np.isfinite(np.asarray(guarded)))
    assert counts is not None and counts.shape == (cells, 4) and counts.sum() > 0
    # the backward's slot-aligned leverages through the cell tables == the per-query gather
    real = cell_fit._guard_level
    try:
        cell_fit._guard_level = functools.partial(_without_layout, real)
        jax.clear_caches()
        gathered = lam0("cell:cubicv:adapt1.5:guard4")
    finally:
        cell_fit._guard_level = real
        jax.clear_caches()
    np.testing.assert_allclose(np.asarray(guarded), np.asarray(gathered), rtol=1e-12, atol=1e-15)


def _without_layout(real, field, c, f, guard, q_valid, sizes, layout=None):
    del layout
    return real(field, c, f, guard, q_valid, sizes)


def test_leverage_guard_through_the_cell_tables_is_the_gathered_one():
    """Slot-aligned queries (the backward's): the per-cell table leverages choose the same
    orders, so the values match the per-query gather exactly."""
    n, npc = 600, 96
    rng = np.random.default_rng(3)
    grid = GridSpec(jnp.zeros(2), jnp.ones(2), (3, 3))
    x = jnp.asarray(rng.uniform(0, 1, (n, 2)))
    v = rng.normal(0, 1, (n, 3))
    v[: n // 2, 0] += 8.0  # beam + cold gas: some queries extrapolate
    v = jnp.asarray(v)
    alive = jnp.asarray(rng.uniform(size=n) < 0.95)
    ids, valid, cell, rank, _count = ensemble_device.cell_nodes(x, alive, grid, npc)
    lam = jnp.asarray(rng.normal(size=n))
    scale = jnp.asarray([1 / 3, 1 / 3, 1.0, 1.0, 1.0])
    basis = "cell:cubicv:adapt1.5:guard32"
    f = cell_fit.fit(x[ids], v[ids], valid, lam[ids], scale, basis)
    c = jnp.minimum(cell, ids.shape[0] - 1)
    qv = v + jnp.asarray(rng.normal(0, 4.0, (n, 3)))  # post-collision-like velocities
    gathered = cell_fit.evaluate(f, c, x, qv, scale, basis)
    tabled = cell_fit.evaluate(f, c, x, qv, scale, basis, layout=(ids, rank))
    live = np.asarray(alive)
    np.testing.assert_allclose(np.asarray(tabled)[live], np.asarray(gathered)[live], rtol=1e-12)


@pytest.mark.parametrize("cap", [256, 2])
def test_pooled_query_fits_widen_only_the_queried_cells(cap, monkeypatch):
    """The shape adjoint's pooled slope fits at its hit queries: widening only the queried
    cells (or, past the cap, every cell) gives the full call's fits at those slots."""
    n, npc = 900, 96
    rng = np.random.default_rng(5)
    grid = GridSpec(jnp.zeros(2), jnp.asarray([6.0, 6.0]), (6, 6))
    x = jnp.asarray(rng.uniform(0, 6, (n, 2)))
    v = jnp.asarray(rng.normal(0, 1, (n, 3)))
    alive = jnp.asarray(rng.uniform(size=n) < 0.95)
    lam = jnp.asarray(rng.normal(size=n))
    scale = jnp.asarray([1.0, 1.0, 1.0, 1.0, 1.0])
    direction = jnp.asarray(np.tile([1.0, 0.0, 0.0], (n, 1)))
    near = np.where(np.linalg.norm(np.asarray(x) - 3.0, axis=1) < 1.2)[0][:40]
    queries = jnp.asarray(np.concatenate([near, -np.ones(24, int)]))
    args = (x, v, alive, lam, grid, npc, scale, 32, "quadratic_full", 4096, direction, 1)
    full = ensemble_device.node_gradients(*args, halfspace=0.25)
    monkeypatch.setattr(ensemble_device, "POOL_QUERY_CELLS", cap)
    gx, gv = ensemble_device.node_gradients(*args, queries=queries, halfspace=0.25)
    for got, want in ((gx, full[0]), (gv, full[1])):
        np.testing.assert_allclose(np.asarray(got)[near], np.asarray(want)[near], rtol=1e-9,
                                   atol=1e-12)  # fmt: skip
