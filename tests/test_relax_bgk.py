"""The particle BGK / ES-BGK baselines and the learned corrections of
:mod:`jax_dsmc.collide.relax`: conservation, equilibrium, the stress and heat-flux rates
(Pr = 1 and 2/3), zero corrections reproducing the base operator bitwise, and gradients in
the correction coefficients."""

from __future__ import annotations

import dataclasses
import functools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import test_relax

from jax_dsmc import simulate, sort
from jax_dsmc.collide import relax

THETA = test_relax.THETA
BGK = functools.partial(relax.bgk_collide, es=False)
ESBGK = functools.partial(relax.bgk_collide, es=True)
RECON = functools.partial(relax.collide, periodic=(True,))
LINEAR = functools.partial(relax.collide, periodic=(True,), corrected="linear")
MLP = functools.partial(relax.collide, periodic=(True,), corrected="mlp")


def _run(op, state, cs, grid, sp, params, static, key, steps, dt, theta=THETA, moments=True):
    st = dataclasses.replace(static, n_steps=steps)
    pr = dataclasses.replace(params, dt=jnp.float64(dt))
    return simulate.run(
        state, cs, grid, sp, pr, st, key, collide_only=True,
        emit_moments=True if moments else "none", collide_fn=op, collision_params=theta,
    )  # fmt: skip


@pytest.mark.parametrize("op", [BGK, ESBGK])
def test_bgk_conserves_in_every_cell_and_leaves_sparse_cells_alone(op):
    state, cs, grid, sp, params, static, key = test_relax._box(n_part=600, cells=(6,))
    cell = np.asarray(sort.build_table(state, grid).cell)
    alive = np.ones(600, bool)
    for c, keep in ((0, 0), (1, 1), (2, 2)):
        alive[np.nonzero(cell == c)[0][keep:]] = False
    state = dataclasses.replace(state, alive=jnp.asarray(alive))
    tau = test_relax._tau_s(state, grid, sp, params)
    (out, _), _ = _run(op, state, cs, grid, sp, params, static, key, 1, 0.5 * tau, moments=False)
    m0, e0 = test_relax._cell_momentum_energy(state, cell, 6)
    m1, e1 = test_relax._cell_momentum_energy(out, cell, 6)
    np.testing.assert_allclose(m1, m0, rtol=0, atol=1e-9 * np.abs(m0).max())
    np.testing.assert_allclose(e1, e0, rtol=1e-12)
    single = np.nonzero(alive & (cell == 1))[0]
    np.testing.assert_array_equal(np.asarray(out.v)[single], np.asarray(state.v)[single])


@pytest.mark.parametrize("op", [BGK, ESBGK])
def test_bgk_keeps_a_maxwellian(op):
    state, cs, grid, sp, params, static, key = test_relax._box()
    tau = test_relax._tau_s(state, grid, sp, params)
    (out, _), _ = _run(op, state, cs, grid, sp, params, static, key, 100, 0.2 * tau,
                       moments=False)  # fmt: skip
    c = np.asarray(out.v) - np.asarray(out.v).mean(axis=0)
    kurt = np.mean(c**4, axis=0) / np.mean(c**2, axis=0) ** 2
    np.testing.assert_allclose(kurt, 3.0, atol=0.15)


@pytest.mark.parametrize(("op", "prandtl"), [(BGK, 1.0), (ESBGK, 2.0 / 3.0)])
def test_bgk_stress_relaxes_at_p_over_mu_and_heat_flux_at_pr_times_it(op, prandtl):
    state, cs, grid, sp, params, static, key = test_relax._box(n_part=40000)
    v = np.asarray(state.v) * np.array([1.25, 0.85, 0.88])
    v[:, 0] = v[:, 0] + 0.35 * (v[:, 0] ** 2 - np.mean(v[:, 0] ** 2)) / np.std(v[:, 0])
    v = v - v.mean(axis=0)
    v = v * np.sqrt(np.var(np.asarray(state.v), axis=0).sum() / np.var(v, axis=0).sum())
    state = dataclasses.replace(state, v=jnp.asarray(v))
    tau = test_relax._tau_s(state, grid, sp, params)
    dt = 0.05 * tau
    _, (m, _) = _run(op, state, cs, grid, sp, params, static, key, 80, dt)
    p = np.asarray(m.P)[:, 0]
    stress = p[:, 0, 0] - np.trace(p, axis1=1, axis2=2) / 3.0
    q = np.asarray(m.q)[:, 0, 0]
    rate_stress = test_relax._decay_rate(stress, dt) * tau
    rate_q = test_relax._decay_rate(q, dt) * tau
    np.testing.assert_allclose(rate_stress, 1.0, rtol=0.08)
    np.testing.assert_allclose(rate_q / rate_stress, prandtl, rtol=0.1)


@pytest.mark.parametrize(("op", "kind"), [(LINEAR, "linear"), (MLP, "mlp")])
def test_a_zero_correction_is_the_base_operator_bitwise(op, kind):
    state, cs, grid, sp, params, static, key = test_relax._shear_box(cells=8, npc=200)
    tau = test_relax._tau_s(state, grid, sp, params)
    (a, _), _ = _run(RECON, state, cs, grid, sp, params, static, key, 3, 0.3 * tau,
                     moments=False)  # fmt: skip
    theta = relax.init_correction(kind)
    (b, _), _ = _run(op, state, cs, grid, sp, params, static, key, 3, 0.3 * tau, theta,
                     moments=False)  # fmt: skip
    np.testing.assert_array_equal(np.asarray(b.v), np.asarray(a.v))


def test_features_measure_the_cell_size_in_local_mean_free_paths():
    """In the shear-wave box at ``dx = 2 lambda`` the scaled ``r^2`` feature is ``(dx /
    lambda)^2 / 4 = 1`` up to the cells' density and temperature noise; all are finite."""
    from examples.learned_collision import shear_wave  # noqa: PLC0415

    state, _, grid, sp, params, _, _, _ = shear_wave.setup(2.0, 400, 0, 0)
    table = sort.build_table(state, grid)
    _, _, n, u, _, pi, theta_c = relax.local_state(state, table, grid, (True,))
    _, n_s, pi_s, theta_s = relax.safe_cells(n, pi, theta_c)
    phi = np.asarray(relax.cell_features(n, n_s, u, pi_s, theta_s, grid, sp, params, (True,)))
    assert phi.shape == (32, len(relax.FEATURES))
    assert np.isfinite(phi).all()
    assert (phi >= 0.0).all()
    np.testing.assert_allclose(np.median(phi[:, relax.FEATURES.index("dx2")]), 1.0, rtol=0.1)


def test_gradient_in_the_linear_coefficients_matches_finite_differences():
    state, cs, grid, sp, params, static, key = test_relax._shear_box(cells=8, npc=250)
    tau = test_relax._tau_s(state, grid, sp, params)
    base = relax.init_correction("linear")

    def stress_after(a):
        theta = dict(base, A=a)
        (out, _), _ = _run(LINEAR, state, cs, grid, sp, params, static, key, 4, 0.2 * tau,
                           theta, moments=False)  # fmt: skip
        x = out.x[:, 0]
        k = 2.0 * jnp.pi / (grid.hi[0] - grid.lo[0])
        return jnp.mean(out.v[:, 0] * out.v[:, 1] * jnp.cos(k * x))

    a0 = jnp.full((2, len(relax.FEATURES)), 0.05)
    grad = np.asarray(jax.grad(stress_after)(a0))
    assert np.isfinite(grad).all()
    h = 1e-5
    for i, j in ((0, 0), (0, 6), (1, 4)):
        fd = float(stress_after(a0.at[i, j].add(h))) - float(stress_after(a0.at[i, j].add(-h)))
        np.testing.assert_allclose(grad[i, j], fd / (2 * h), rtol=1e-4, atol=1e-9)
