"""Round-1 2-D windows (:mod:`examples.learned_collision.windows2d`): pool thinning, the
shock-frame target, and reverse-mode window gradients (both estimators) against finite
differences on a shear-wave and a shock slab."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from examples.learned_collision import flows, operators, refdata
from examples.learned_collision import windows2d as w2

DX_F, PPC_F, P_POOL = 0.5, 10, 0.5


def _meta(c, height, kind):
    return {
        "dx_fine": DX_F,
        "ppc": PPC_F,
        "p_pool": P_POOL,
        "height": height,
        "x_shock": flows.SHOCK_X,
        "vth_ms": c["vth"],
        "ppc_up": c["ppc_up"],
        "kind": kind,
    }


def _pools(state, c, key):
    cap = int(state.x.shape[0] * P_POOL + 6 * np.sqrt(state.x.shape[0]) + 32)
    x, v, a, over = refdata.pool(state.x, state.v, state.alive, key, P_POOL, cap)
    assert int(over) == 0
    return np.asarray(x / c["lambda"]), np.asarray(v / c["vth"]), np.asarray(a)


def _shock():
    state, *_, c, _ = flows.shock_setup(3.0, DX_F, PPC_F, 0, height_mfp=10.0)
    meta = _meta(c, 10.0, "shock")
    case = w2.make_case("K3", meta, ppc_c=5)
    px, pv, pa = _pools(state, c, jax.random.PRNGKey(3))
    sums = np.asarray(refdata._fine_sums(state, c["length"], c["cells"]))
    target = w2.shock_target(sums, flows.SHOCK_X, case, meta, 2)
    return case, px, pv, pa, target


def _shear():
    state, *_, c, _ = flows.shear_setup(0.5, 32.0, DX_F, PPC_F, 0, height_mfp=10.0)
    case = w2.make_case("S3", _meta(c, 10.0, "shear"), ppc_c=5)
    px, pv, pa = _pools(state, c, jax.random.PRNGKey(4))
    return case, px, pv, pa, np.zeros((2, case.n_coarse, 13))


def test_shock_target_has_the_rankine_hugoniot_plateaus():
    case, *_, target = _shock()
    st = flows.shock_states(3.0)
    n, ux = target[0, :, 0], target[0, :, 1]
    np.testing.assert_allclose(n[:20].mean(), 1.0, rtol=0.03)
    np.testing.assert_allclose(n[-20:].mean(), st["ratio"], rtol=0.03)
    np.testing.assert_allclose(ux[:20].mean(), st["u1"] / case.vth, rtol=0.02)


def test_pool_state_crops_and_thins_to_the_coarse_count():
    case, px, pv, pa, _ = _shock()
    counts = []
    for k in range(4):
        state, _, over = w2.pool_state(px[0], pv[0], pa[0], case, jax.random.PRNGKey(k))
        assert int(over) == 0
        counts.append(int(state.alive.sum()))
        assert float(jnp.max(jnp.where(state.alive, state.x[:, 1], 0.0))) < case.height * case.lam
    st = flows.shock_states(3.0)
    up, dn = flows.SHOCK_X, flows.SHOCK_LENGTH - flows.SHOCK_X
    expected = 5 * case.height * (up + st["ratio"] * dn) / case.dx_c**2
    np.testing.assert_allclose(np.mean(counts), expected, rtol=0.05)


def _objective(theta, states, cstates, keys, target, weights, bias, case, fn, estimator):
    return w2.window_objective(theta, states, cstates, keys, target, weights, bias, case, fn,
                               2, 0, estimator)[0]  # fmt: skip


def _replicas(case, px, pv, pa):
    pairs = [w2.pool_state(px[r], pv[r], pa[r], case, jax.random.PRNGKey(20 + r))
             for r in range(2)]  # fmt: skip
    states = jax.tree.map(flows.stack, *[p[0] for p in pairs])
    cstates = jax.tree.map(flows.stack, *[p[1] for p in pairs])
    return states, cstates, jnp.stack([jax.random.PRNGKey(31), jax.random.PRNGKey(32)])


def _check(case, px, pv, pa, target, model, leaf, estimator):
    fn, theta, _ = operators.operator(model, case.kind)
    if model == "linear":
        theta = dict(theta, A=jnp.full_like(theta["A"], 0.03))
    states, cstates, keys = _replicas(case, px, pv, pa)
    weights = jnp.ones(len(w2.LOSS_FIELDS[case.kind]))
    k = 2 if case.kind == "shear" else case.n_coarse
    bias = jax.random.normal(jax.random.PRNGKey(5), (2, k, len(weights)))
    f = jax.jit(functools.partial(_objective, states=states, cstates=cstates, keys=keys,
                                  target=jnp.asarray(target), weights=weights, bias=bias,
                                  case=case, fn=fn, estimator=estimator))  # fmt: skip
    grad = jax.jit(jax.grad(f))(theta)
    g = np.asarray(grad[leaf]).ravel()[0]
    # a small step: a shock slab deletes particles at its faces every step, and a central
    # difference straddling one such flip is garbage (h = 1e-5 did); forward and backward
    # differences agreeing shows this step is flip-free
    h = 1e-7
    e = (jnp.arange(theta[leaf].size) == 0).reshape(theta[leaf].shape)
    f0 = float(f(theta))
    fu = float(f(dict(theta, **{leaf: theta[leaf] + h * e})))
    fdn = float(f(dict(theta, **{leaf: theta[leaf] - h * e})))
    np.testing.assert_allclose((fu - f0) / h, (f0 - fdn) / h, rtol=1e-3)
    assert np.isfinite(g) and abs(g) > 0.0
    np.testing.assert_allclose(g, (fu - fdn) / (2 * h), rtol=1e-3)  # h = 1e-7: round-off


def test_shock_window_gradients_match_finite_differences():
    case, px, pv, pa, target = _shock()
    _check(case, px, pv, pa, target, "scalar", "log_tau", "pair")
    _check(case, px, pv, pa, target, "linear", "A", "bias")


def test_shear_window_gradients_match_finite_differences():
    case, px, pv, pa, target = _shear()
    _check(case, px, pv, pa, target, "scalar", "log_tau", "pair")
    _check(case, px, pv, pa, target, "linear", "A", "bias")
