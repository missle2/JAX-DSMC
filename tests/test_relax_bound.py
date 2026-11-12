"""ES-FP's positive-definiteness bound against the paper's limiter, the rate basis of the
learned correction, and the ``bound`` feature (jax_dsmc/collide/relax.py, 2026-11-12)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_dsmc import sort
from jax_dsmc.collide import relax


def test_bound_is_the_papers_clamped_nu_integrated_exactly():
    """Mathiaud & Mieussens (2016) eq. 35 clamps ``nu`` at ``-theta / pd_max``; the exact
    frozen-coefficient step's stress factor is then ``a^2 + (1 - a^2) nu`` -- the bound, up
    to SIGMA_FLOOR.  Below the onset the bound leaves ``e_s`` alone."""
    theta = jnp.array([1.0, 1.0, 2.0])
    pd_max = jnp.array([1.5, 1.9, 3.0])  # past 0.8 theta in every cell
    a2 = jnp.array([0.9, 0.99, 0.95])
    e_s = a2 - 1.25 * (1.0 - a2)  # a^2 + (1 - a^2) nu at nu = -5/4
    nu = -theta / pd_max
    paper = a2 + (1.0 - a2) * nu * (1.0 - relax.SIGMA_FLOOR)
    np.testing.assert_allclose(relax.stress_factor_bound(e_s, a2, theta, pd_max), paper,
                               rtol=1e-12)  # fmt: skip
    low = 0.7 * theta
    np.testing.assert_array_equal(relax.stress_factor_bound(e_s, a2, theta, low), e_s)
    assert relax.BOUND_ONSET == 0.8


def _shear_cells():
    from examples.learned_collision import shear_wave  # noqa: PLC0415

    state, _, grid, sp, params, _, _, _ = shear_wave.setup(2.0, 400, 0, 0)
    table = sort.build_table(state, grid)
    _, _, n, u, _, pi, theta_c = relax.local_state(state, table, grid, (True,))
    return n, u, pi, theta_c, grid, sp, params


def test_rate_basis_corrects_each_rate_alone():
    n, _, pi, theta_c, grid, sp, params = _shear_cells()
    _, n_s, _, theta_s = relax.safe_cells(n, pi, theta_c)
    base = {k: jnp.float64(v) for k, v in relax.BASELINE.items()}
    r0 = relax.rates(n_s, theta_s, grid, sp, params, base, False)
    d = jnp.full(n.shape, 0.3)
    zero = jnp.zeros(n.shape)
    rs, rq = relax.rates(n_s, theta_s, grid, sp, params, base, False, (d, zero))
    np.testing.assert_allclose(rs, r0[0] * np.exp(0.3), rtol=1e-12)
    np.testing.assert_allclose(rq, r0[1], rtol=1e-12)
    rs, rq = relax.rates(n_s, theta_s, grid, sp, params, base, False, (zero, d))
    np.testing.assert_allclose(rs, r0[0], rtol=1e-12)
    np.testing.assert_allclose(rq, r0[1] * np.exp(0.3), rtol=1e-12)
    np.testing.assert_allclose(r0[1] / r0[0], relax.PRANDTL, rtol=1e-12)


def test_bound_feature_and_activity_mark_strongly_anisotropic_cells():
    """An isotropic cell gives ``bound = 0`` and no slowdown; a cell stretched past
    ``pd_max = 0.8 theta`` gives the excess and a positive slowdown."""
    n, u, pi, theta_c, grid, sp, params = _shear_cells()
    theta = jnp.mean(theta_c)
    iso = jnp.broadcast_to(theta * jnp.eye(3), pi.shape)
    stretch = jnp.diag(jnp.array([2.2, 0.4, 0.4])) * theta  # Pi_dev eigen 1.2 theta
    pis = jnp.where((jnp.arange(n.shape[0]) % 2 == 0)[:, None, None], iso, stretch)
    tc = jnp.full(n.shape, theta)
    nn = jnp.full(n.shape, 50.0)
    phi = np.asarray(relax.cell_features(nn, nn, u, pis, tc, grid, sp, params, (True,)))
    col = phi[:, relax.FEATURES.index("bound")] * relax.FEATURE_SCALE[-1]
    np.testing.assert_allclose(col[0::2], 0.0)
    np.testing.assert_allclose(col[1::2], 1.2 - 0.8, rtol=1e-10)
    base = {k: jnp.float64(v) for k, v in relax.BASELINE.items()}
    ratio, slow = relax.bound_activity(nn, pis, tc, grid, sp, params, base)
    np.testing.assert_allclose(ratio[1::2], 1.2, rtol=1e-10)
    assert np.all(np.asarray(slow[0::2]) == 0.0)
    assert np.all(np.asarray(slow[1::2]) > 0.0)
