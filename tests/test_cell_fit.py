"""The per-cell field fit of the virtual pull-back (jax_dsmc.cell_fit) and its plumbing
through ensemble_device.backward_virtual: exact on polynomials, the least-squares solution,
the order guard, finite empty sides, exact sigma points, and the complex-cell fallback equal
to the per-query one-sided fit."""

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
    return sum(c * np.prod([o[..., a] for a in t], axis=0) for c, t in zip(coef, terms, strict=True))


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
