"""The rich learned correction (relax.RICH_OUTPUTS, ``--outputs rich``, 2026-11-14).

- zero rich outputs give the rate-only operator bitwise, and the rate rows act alike;
- every cell's momentum and energy are conserved with every output at its extremes;
- a long run with every output saturated (both signs) stays finite and positive, its heat
  flux and stress bounded -- the failure the cubic drift had;
- the gradients in every output row are finite at zero (the start of training) and match
  finite differences away from it;
- the pieces: the noise shape keeps the covariance and moves the fourth moment as derived;
  the Hermite drift moves the heat flux by ``5 theta d`` and not the momentum or stress; the
  congruence keeps the tensor positive semi-definite; the Burnett bases are symmetric,
  traceless and vanish without gradients;
- the reconstruction gains act in the intended direction (more in-cell shear with a larger
  velocity gain, more in-cell temperature slope with a temperature gain).
"""

from __future__ import annotations

import dataclasses
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import test_relax
import test_relax_bgk

from jax_dsmc import sort
from jax_dsmc.collide import relax

LINEAR = test_relax_bgk.LINEAR
INV_N = relax.FEATURES.index("inv_n")
ROWS = len(relax.RICH_OUTPUTS)


def _gradient_box(cells=8, npc=1000, seed=0, shear=60.0, sound=40.0, heat=0.3):
    """A periodic 1-D box with every gradient the bases read: a transverse wave ``u_y = A
    sin(k x)`` (strain, spin), a longitudinal one ``u_x = B sin(k x)`` (divergence) and a
    temperature wave ``T = T0 (1 + h cos(k x))`` (the thermal velocities scaled about their
    mean)."""
    state, cs, grid, sp, params, static, key = test_relax._box(n_part=cells * npc,
                                                               cells=(cells,), seed=seed)  # fmt: skip
    x = np.asarray(state.x)[:, 0]
    k = 2 * np.pi / float(grid.hi[0] - grid.lo[0])
    v = np.asarray(state.v).copy()
    v = v.mean(axis=0) + (v - v.mean(axis=0)) * np.sqrt(1.0 + heat * np.cos(k * x))[:, None]
    v[:, 0] += sound * np.sin(k * x)
    v[:, 1] += shear * np.sin(k * x)
    return dataclasses.replace(state, v=jnp.asarray(v)), cs, grid, sp, params, static, key


def _theta(raw, npc, rich=True):
    """A linear correction whose rows' raw outputs are ``~raw`` in every cell of ``npc``
    particles (through the ``1 / N`` feature, ``(1 / npc) / 0.1``)."""
    theta = relax.init_correction("linear", rich=rich)
    a = np.zeros(theta["A"].shape)
    a[: len(raw), INV_N] = np.asarray(raw) * 0.1 * npc
    return dict(theta, A=jnp.asarray(a))


def _step(theta, box, steps=1, dt_frac=0.2):
    state, cs, grid, sp, params, static, key = box
    tau = test_relax._tau_s(state, grid, sp, params)
    (out, _), _ = test_relax_bgk._run(LINEAR, state, cs, grid, sp, params, static, key, steps,
                                      dt_frac * tau, theta, moments=False)  # fmt: skip
    return out


def _cells(state, grid):
    return np.asarray(sort.build_table(state, grid).cell)


def _cell_moments(state, cell, cells):
    """Per cell: ``theta``, ``q / theta^(3/2)`` (3-vector) and ``|Pi_dev| / theta``."""
    v = np.asarray(state.v)
    out = []
    for c in range(cells):
        w = v[cell == c]
        cc = w - w.mean(axis=0)
        th = np.mean(np.sum(cc**2, axis=1)) / 3.0
        q = 0.5 * np.mean(np.sum(cc**2, axis=1)[:, None] * cc, axis=0)
        pi = cc.T @ cc / len(cc)
        out.append((th, q / th**1.5, np.linalg.norm(pi - th * np.eye(3)) / th))
    return out


def test_zero_rich_outputs_are_the_rate_operator_bitwise():
    box = _gradient_box(npc=500)
    rates = [0.3, -0.2]
    base = _step(_theta(rates, 500, rich=False), box, steps=2)
    rich = _step(_theta(rates, 500, rich=True), box, steps=2)
    np.testing.assert_array_equal(np.asarray(rich.v), np.asarray(base.v))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_rich_step_conserves_every_cell(sign):
    box = _gradient_box(npc=500, seed=1)
    raw = sign * np.array([0.5, -0.5, 1, -1, 1, -1, 1, -1, 1, 0.4, 1.0, 0.1]) * 3.0
    out = _step(_theta(raw, 500), box)
    state, grid = box[0], box[2]
    cell = _cells(state, grid)
    m0, e0 = test_relax._cell_momentum_energy(state, cell, 8)
    m1, e1 = test_relax._cell_momentum_energy(out, cell, 8)
    np.testing.assert_allclose(m1, m0, rtol=0, atol=1e-9 * np.abs(m0).max())
    np.testing.assert_allclose(e1, e0, rtol=1e-12)
    assert not np.allclose(np.asarray(out.v), np.asarray(_step(_theta([], 500), box).v))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_saturated_outputs_stay_finite_positive_and_bounded(sign):
    """Every output pinned at its bound (``raw`` 10x it), 150 steps at ``dt = 0.5 tau``: the
    cubic drift's failure (``beta > 1/27`` grew ``q`` every step) would show here as a heat
    flux running away.  The targets are capped at 0.5 in units of ``theta``."""
    box = _gradient_box(npc=400, seed=2)
    raw = sign * 10.0 * relax.RICH_BOUND * np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1])
    out = _step(_theta(raw, 400), box, steps=150, dt_frac=0.5)
    v = np.asarray(out.v)
    assert np.isfinite(v).all()
    for th, q, pd in _cell_moments(out, _cells(out, box[2]), 8):
        assert th > 0.0
        assert np.linalg.norm(q) < 1.0  # the target's cap is 0.5; Maxwellian noise ~0.05
        assert pd < 1.0
    e0 = np.sum(np.asarray(box[0].v) ** 2)
    np.testing.assert_allclose(np.sum(v**2), e0, rtol=1e-10)


def _scalar_after(a, base, box, steps=3):
    """A heat-flux + normal-stress + kurtosis functional of the box after ``steps`` -- every
    output row reaches it."""
    out = _step(dict(base, A=a), box, steps=steps)
    c = out.v - jnp.mean(out.v, axis=0)
    x2 = jnp.sum(c**2, axis=1)
    th = jnp.mean(x2) / 3.0
    return (0.5 * jnp.mean(x2 * c[:, 0]) / th**1.5 + jnp.mean(c[:, 0] ** 2) / th
            + 0.1 * jnp.mean(x2**2) / th**2)  # fmt: skip


def test_gradients_are_finite_at_zero_outputs():
    box = _gradient_box(cells=4, npc=300, seed=3)
    base = _theta([], 300)
    grad = np.asarray(jax.grad(_scalar_after)(base["A"], base, box))
    assert np.isfinite(grad).all()
    assert (np.abs(grad[:, INV_N]) > 0.0).sum() >= ROWS - 1  # (the kurtosis row: odd at 0)


def test_gradient_in_every_rich_output_matches_finite_differences():
    box = _gradient_box(cells=4, npc=300, seed=4)
    raw = np.array([0.1, -0.1, 0.3, -0.2, 0.2, 0.1, 0.3, -0.2, 0.2, 0.1, 0.4, 0.03])
    base = _theta(raw, 300)
    a0 = base["A"]
    grad = np.asarray(jax.grad(_scalar_after)(a0, base, box))
    h = 1e-6
    for row in range(ROWS):
        fd = (float(_scalar_after(a0.at[row, INV_N].add(h), base, box))
              - float(_scalar_after(a0.at[row, INV_N].add(-h), base, box))) / (2 * h)  # fmt: skip
        np.testing.assert_allclose(grad[row, INV_N], fd, rtol=1e-4, atol=1e-9,
                                   err_msg=relax.RICH_OUTPUTS[row])  # fmt: skip


@pytest.mark.parametrize("t", [-0.125, -0.05, 0.05, 0.125])
def test_shaped_noise_keeps_the_covariance_and_moves_the_fourth_moment(t):
    xi = jax.random.normal(jax.random.PRNGKey(0), (2_000_000, 3), dtype=jnp.float64)
    out = np.asarray(relax.shaped_noise(xi, jnp.full(xi.shape[0], t)))
    np.testing.assert_allclose(out.T @ out / len(out), np.eye(3), atol=0.01)
    np.testing.assert_allclose(np.mean(out, axis=0), 0.0, atol=0.003)
    np.testing.assert_allclose(np.mean(np.sum(out**2, axis=1) * out[:, 0]), 0.0, atol=0.05)
    # the exact moments of the capped map, over the chi(3) density of |xi|
    r = np.linspace(0.0, 30.0, 300_001)
    pdf = np.sqrt(2.0 / np.pi) * r**2 * np.exp(-(r**2) / 2.0)
    w = (1 - 2 * t) ** 2.5 * np.exp(t * np.minimum(r**2, relax.KURTOSIS_CAP))
    np.testing.assert_allclose(np.trapezoid(w * r**2 * pdf, r), 3.0, rtol=1e-3)
    expected = np.trapezoid(w**2 * r**4 * pdf, r)  # 11.07 at -1/8, 38.24 at 1/8 (uncapped 40.3)
    np.testing.assert_allclose(np.mean(np.sum(out**2, axis=1) ** 2), expected, rtol=0.02)


def test_hermite_drift_moves_the_heat_flux_alone():
    c = jax.random.normal(jax.random.PRNGKey(1), (2_000_000, 3), dtype=jnp.float64)
    d = jnp.array([0.01, -0.005, 0.0])
    out = np.asarray(c + relax.heat_drift(c, jnp.broadcast_to(d, c.shape), jnp.ones(len(c))))
    cn = np.asarray(c)

    def q(w):
        return 0.5 * np.mean(np.sum(w**2, axis=1)[:, None] * w, axis=0)

    np.testing.assert_allclose(relax.HERMITE_MEAN, 2.936, atol=1e-3)
    np.testing.assert_allclose(relax.HERMITE_GAIN, 4.455, atol=1e-3)
    np.testing.assert_allclose(q(out) - q(cn), relax.HERMITE_GAIN * np.asarray(d), atol=1.5e-3)
    np.testing.assert_allclose(out.mean(axis=0) - cn.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(out.T @ out / len(out) - cn.T @ cn / len(cn), 0.0, atol=2e-3)


def test_congruence_is_positive_semidefinite_and_moves_the_tensor_by_the_target():
    rng = np.random.default_rng(0)
    c = jnp.asarray(rng.standard_normal((200_000, 3)))
    for scale in (0.05, 0.9):
        m = rng.standard_normal((3, 3))
        k = scale * (m + m.T) / np.linalg.norm(m + m.T)
        out = np.asarray(relax.stress_stretch(c, jnp.broadcast_to(jnp.asarray(k), (len(c), 3, 3))))
        pi0, pi1 = np.asarray(c).T @ np.asarray(c) / len(c), out.T @ out / len(out)
        assert np.linalg.eigvalsh(pi1).min() >= 0.0
        if scale < 0.1:  # first order: (I + K) Pi (I + K) - Pi ~ 2 K for Pi ~ I
            np.testing.assert_allclose(pi1 - pi0, 2 * k, atol=0.02)


def _bases(u, theta_c, box):
    state, _, grid, sp, params, _, _ = box
    n = jnp.full(theta_c.shape[0], 1000.0)
    return relax.burnett_bases(n, n, jnp.asarray(u), jnp.asarray(theta_c), grid, sp, params,
                               (True,))  # fmt: skip


def test_burnett_bases_are_symmetric_traceless_and_need_gradients():
    box = _gradient_box(npc=50)
    cells = 8
    th0 = float(np.mean(np.var(np.asarray(box[0].v), axis=0)))
    uniform = _bases(np.tile([10.0, 5.0, 0.0], (cells, 1)), np.full(cells, th0), box)
    for part in uniform:
        assert np.abs(np.asarray(part)).max() == 0.0
    xc = (np.arange(cells) + 0.5) / cells
    shear = np.zeros((cells, 3))
    shear[:, 1] = 30.0 * np.sin(2 * np.pi * xc)
    t_shear, b_shear, _ = (np.asarray(p) for p in _bases(shear, np.full(cells, th0), box))
    np.testing.assert_allclose(t_shear, np.swapaxes(t_shear, 2, 3), atol=1e-12)
    np.testing.assert_allclose(np.trace(t_shear, axis1=2, axis2=3), 0.0, atol=1e-12)
    bulk, ss, tt, sw = (relax.STRESS_BASES.index(b) for b in relax.STRESS_BASES)
    assert np.abs(t_shear[:, bulk]).max() == 0.0  # no divergence in a shear
    assert np.abs(t_shear[:, tt]).max() == 0.0  # no temperature gradient
    assert np.abs(t_shear[:, ss]).max() > 0.0 and np.abs(t_shear[:, sw]).max() > 0.0
    assert np.abs(b_shear).max() == 0.0  # every flux basis needs grad T or grad p
    hot = th0 * (1.0 + 0.3 * np.cos(2 * np.pi * xc))
    sound = np.zeros((cells, 3))
    sound[:, 0] = 30.0 * np.sin(2 * np.pi * xc)
    t_all, b_all, g_t = (np.asarray(p) for p in _bases(sound, hot, box))
    assert np.isfinite(t_all).all() and np.isfinite(b_all).all()
    assert np.abs(t_all[:, bulk]).max() > 0.0 and np.abs(t_all[:, tt]).max() > 0.0
    assert np.abs(b_all).max() > 0.0
    # grad ln T is -0.3 k sin / (1 + 0.3 cos): negative in the first half of the box
    assert np.all(g_t[1:3, 0] < 0.0) and np.all(g_t[5:7, 0] > 0.0)


def _in_cell_slope(state, grid, cells, field):
    x = np.asarray(state.x)[:, 0]
    dx = float(grid.hi[0] - grid.lo[0]) / cells
    cell = np.minimum((x / dx).astype(int), cells - 1)
    v = np.asarray(state.v)
    out = []
    for c in range(cells):
        sel = cell == c
        w = v[sel] - v[sel].mean(axis=0)
        y = w[:, 1] if field == "shear" else np.sum(w**2, axis=1)
        out.append(np.polyfit(x[sel] - x[sel].mean(), y, 1)[0])
    return np.array(out)


def test_velocity_gain_keeps_more_in_cell_shear():
    box = _gradient_box(cells=4, npc=4000, seed=5, sound=0.0, heat=0.0)
    row = relax.ROW["recon_u"]
    slopes = []
    for gain in (0.0, 0.4):  # raw: gain ~0.5 tanh(raw / 0.5)
        raw = np.zeros(row + 1)
        raw[row] = gain
        out = _step(_theta(raw, 4000), box, steps=4, dt_frac=1.0)
        slopes.append(np.abs(_in_cell_slope(out, box[2], 4, "shear")).mean())
    assert slopes[1] > slopes[0]


def test_temperature_gain_keeps_more_in_cell_temperature_slope():
    """Cells with a strong in-cell temperature slope (a short wave: 2 cells per wavelength
    would cancel, so 4 cells over the box) relax at dt = tau: with a temperature gain the
    noise is hotter on the hot side, so the in-cell slope of |c|^2 decays less."""
    box = _gradient_box(cells=4, npc=4000, seed=6, shear=0.0, sound=0.0, heat=0.6)
    row = relax.ROW["recon_t"]
    slopes = []
    for gain in (0.0, 1.0):
        raw = np.zeros(row + 1)
        raw[row] = gain
        out = _step(_theta(raw, 4000), box, steps=4, dt_frac=1.0)
        slopes.append(np.abs(_in_cell_slope(out, box[2], 4, "temp")).mean())
    assert slopes[1] > 1.1 * slopes[0]
