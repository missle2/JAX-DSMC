"""The learned cubic Fokker-Planck drift (relax.cubic_drift, ``--cubic``, 2026-11-13): a zero
coefficient reproduces the two-rate operator bitwise, the step still conserves, a positive
coefficient moves the heat flux and leaves the stress nearly alone, and its gradient matches
finite differences."""

from __future__ import annotations

import dataclasses
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import test_relax
import test_relax_bgk

from jax_dsmc import sort
from jax_dsmc.collide import relax

LINEAR = test_relax_bgk.LINEAR


def _heat_box(cells=4, npc=4000, eps=0.05):
    """A periodic 1-D box whose gas carries a heat flux along ``x``: each velocity
    ``xi + eps s (|xi|^2 / s^2 - 3) e_x`` (``xi`` the box's Maxwellian, ``s^2`` its variance;
    an even map, so the mean is unchanged and ``q_x ~ 5 eps s^3 > 0``)."""
    state, cs, grid, sp, params, static, key = test_relax._box(n_part=cells * npc,
                                                               cells=(cells,))  # fmt: skip
    v = np.asarray(state.v).copy()
    s2 = np.mean(np.var(v, axis=0))
    v[:, 0] += eps * np.sqrt(s2) * (np.sum(v**2, axis=1) / s2 - 3.0)
    return dataclasses.replace(state, v=jnp.asarray(v)), cs, grid, sp, params, static, key


INV_N = relax.FEATURES.index("inv_n")


def _theta(beta, npc=4000):
    """A cubic model whose coefficient is ``~beta`` in every cell of a box of ``npc`` particles
    per cell: through the ``1 / N`` feature alone (``(1 / npc) / 0.1`` scaled; the box's other
    features -- its ``dt / tau`` and gradients -- are far outside the training range and
    would saturate the bound)."""
    theta = relax.init_correction("linear", cubic=True)
    theta["A"] = theta["A"].at[2, INV_N].set(beta * 0.1 * npc)
    return theta


def _moments(state):
    v = np.asarray(state.v)
    c = v - v.mean(axis=0)
    q = 0.5 * np.mean(np.sum(c**2, axis=1) * c[:, 0])
    pxx = np.mean(c[:, 0] ** 2) - np.mean(np.sum(c**2, axis=1)) / 3.0
    return q, pxx


def _step(theta, box, dt_frac=0.2, steps=1):
    state, cs, grid, sp, params, static, key = box
    tau = test_relax._tau_s(state, grid, sp, params)
    (out, _), _ = test_relax_bgk._run(LINEAR, state, cs, grid, sp, params, static, key, steps,
                                      dt_frac * tau, theta, moments=False)  # fmt: skip
    return out


def test_zero_cubic_coefficient_is_the_two_rate_operator_bitwise():
    box = _heat_box()
    two = _step(relax.init_correction("linear"), box)
    three = _step(_theta(0.0), box)
    np.testing.assert_array_equal(np.asarray(three.v), np.asarray(two.v))


def test_cubic_step_conserves_every_cell():
    box = _heat_box()
    state, _, grid = box[0], box[1], box[2]
    cell = np.asarray(sort.build_table(state, grid).cell)
    out = _step(_theta(0.1), box)
    m0, e0 = test_relax._cell_momentum_energy(state, cell, 4)
    m1, e1 = test_relax._cell_momentum_energy(out, cell, 4)
    np.testing.assert_allclose(m1, m0, rtol=0, atol=1e-9 * np.abs(m0).max())
    np.testing.assert_allclose(e1, e0, rtol=1e-12)


def test_positive_cubic_coefficient_raises_the_heat_flux_not_the_stress():
    """Same noise, coefficient 0 against 0.1: the heat flux after the step grows by ~27 s beta
    (Grad's 13 moments; s = 1 - exp(-r_q dt) = 0.125 at dt = 0.2 tau, so ~34 %) and the
    normal stress, in units of theta against q in units of theta^(3/2), moves far less."""
    box = _heat_box()
    q0, p0 = _moments(_step(_theta(0.0), box))
    q1, p1 = _moments(_step(_theta(0.1), box))
    qm, _ = _moments(_step(_theta(-0.1), box))
    assert qm < q0 < q1
    np.testing.assert_allclose((q1 - q0) / q0, 27 * (1 - np.exp(-0.2 * 2 / 3)) * 0.1, rtol=0.2)
    theta = np.mean(np.var(np.asarray(box[0].v), axis=0))
    assert abs(p1 - p0) / theta < 0.2 * abs(q1 - q0) / theta**1.5


def test_gradient_in_the_cubic_coefficients_matches_finite_differences():
    box = _heat_box(cells=4, npc=500)
    base = _theta(0.0, npc=500)

    def heat_after(a):
        out = _step(dict(base, A=a), box, steps=3)
        c = out.v - jnp.mean(out.v, axis=0)
        return 0.5 * jnp.mean(jnp.sum(c**2, axis=1) * c[:, 0])

    a0 = base["A"].at[2, INV_N].set(0.05 * 0.1 * 500)
    grad = np.asarray(jax.grad(heat_after)(a0))
    assert np.isfinite(grad).all() and abs(grad[2, INV_N]) > 0.0
    h = 1e-6
    for i, j in ((2, INV_N), (1, INV_N), (0, INV_N)):
        fd = float(heat_after(a0.at[i, j].add(h))) - float(heat_after(a0.at[i, j].add(-h)))
        np.testing.assert_allclose(grad[i, j], fd / (2 * h), rtol=1e-4, atol=1e-12)
