"""The ES-Fokker-Planck collision operator (:mod:`jax_dsmc.collide.relax`) in a homogeneous
box: conservation, equilibrium, the stress and heat-flux relaxation rates (Pr = 2/3),
Galilean invariance, large time steps, entropy, and gradients in its coefficients."""

from __future__ import annotations

import dataclasses
import functools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import test_ntc

from jax_dsmc import simulate
from jax_dsmc.collide import relax
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.grid import cell_volume

THETA = {k: jnp.float64(v) for k, v in relax.BASELINE.items()}
OPERATOR = functools.partial(relax.collide)


def _box(n_part=20000, cells=(1,), seed=0):
    state, cs, grid, sp, params, static, _t, key = test_ntc.build(
        n=cells, N=n_part, A_max=8, dt=1e-9, seed=seed
    )
    return state, cs, grid, sp, params, static, key


def _tau_s(state, grid, sp, params):
    """The stress relaxation time ``mu / p`` of the box (its mean density)."""
    v = np.asarray(state.v)
    temp = float(sp.mass[0]) * np.mean(np.var(v, axis=0)) / K_BOLTZ
    alive = int(np.asarray(state.alive).sum())
    nrho = alive * float(params.fnum[0]) / float(np.sum(np.asarray(cell_volume(grid))))
    return float(relax.vss_viscosity(temp, sp)) / (nrho * K_BOLTZ * temp)


def _run(state, cs, grid, sp, params, static, key, steps, dt, theta=THETA, moments=True):
    st = dataclasses.replace(static, n_steps=steps)
    pr = dataclasses.replace(params, dt=jnp.float64(dt))
    return simulate.run(
        state, cs, grid, sp, pr, st, key, collide_only=True,
        emit_moments=True if moments else "none", collide_fn=OPERATOR, collision_params=theta,
    )  # fmt: skip


def _cell_momentum_energy(state, cell, n_cells):
    w = np.asarray(state.alive)
    v = np.asarray(state.v)
    mom = np.zeros((n_cells, 3))
    en = np.zeros(n_cells)
    np.add.at(mom, cell[w], v[w])
    np.add.at(en, cell[w], np.sum(v[w] ** 2, axis=1))
    return mom, en


def test_momentum_and_energy_are_conserved_in_every_cell_and_sparse_cells_are_left_alone():
    state, cs, grid, sp, params, static, key = _box(n_part=600, cells=(6,))
    from jax_dsmc import sort  # noqa: PLC0415

    cell = np.asarray(sort.build_table(state, grid).cell)
    # cell 0 empty, cell 1 one particle, cell 2 two particles, the rest crowded
    alive = np.ones(600, bool)
    for c, keep in ((0, 0), (1, 1), (2, 2)):
        idx = np.nonzero(cell == c)[0]
        alive[idx[keep:]] = False
    state = dataclasses.replace(state, alive=jnp.asarray(alive))
    tau = _tau_s(state, grid, sp, params)
    (out, _), _ = _run(state, cs, grid, sp, params, static, key, 1, 0.3 * tau, moments=False)
    m0, e0 = _cell_momentum_energy(state, cell, 6)
    m1, e1 = _cell_momentum_energy(out, cell, 6)
    np.testing.assert_allclose(m1, m0, rtol=0, atol=1e-9 * np.abs(m0).max())
    np.testing.assert_allclose(e1, e0, rtol=1e-12)
    single = np.nonzero(alive & (cell == 1))[0]
    np.testing.assert_array_equal(np.asarray(out.v)[single], np.asarray(state.v)[single])
    assert not np.allclose(np.asarray(out.v)[alive & (cell == 3)],
                           np.asarray(state.v)[alive & (cell == 3)])  # fmt: skip


def test_a_maxwellian_stays_maxwellian():
    state, cs, grid, sp, params, static, key = _box()
    tau = _tau_s(state, grid, sp, params)
    (out, _), _ = _run(state, cs, grid, sp, params, static, key, 100, 0.2 * tau, moments=False)
    c = np.asarray(out.v) - np.asarray(out.v).mean(axis=0)
    kurt = np.mean(c**4, axis=0) / np.mean(c**2, axis=0) ** 2
    np.testing.assert_allclose(kurt, 3.0, atol=0.15)  # 20 000 samples: s.e. ~ 0.035
    np.testing.assert_allclose(np.var(c, axis=0), np.var(np.asarray(state.v), axis=0).mean(),
                               rtol=0.05)  # fmt: skip


def _decay_rate(series, dt):
    """Least-squares slope of ``log |series|`` against time (first two thirds of the decay)."""
    s = np.abs(np.asarray(series))
    keep = s > 0.15 * s[0]
    t = dt * np.arange(1, s.size + 1)
    return -np.polyfit(t[keep], np.log(s[keep]), 1)[0]


def test_stress_relaxes_at_p_over_mu_and_heat_flux_at_two_thirds_of_it():
    """``Pi_dev`` decays at ``2 (1 - nu) / tau = 1 / tau_s``; the heat flux at ``3 / tau =
    (2/3) / tau_s`` -- the monatomic Prandtl number from the rate ratio."""
    state, cs, grid, sp, params, static, key = _box(n_part=40000)
    v = np.asarray(state.v)
    # anisotropic (stress) and skewed (heat flux) start, same temperature
    v = v * np.array([1.25, 0.85, 0.88])
    v[:, 0] = v[:, 0] + 0.35 * (v[:, 0] ** 2 - np.mean(v[:, 0] ** 2)) / np.std(v[:, 0])
    v = v - v.mean(axis=0)
    v = v * np.sqrt(np.var(np.asarray(state.v), axis=0).sum() / np.var(v, axis=0).sum())
    state = dataclasses.replace(state, v=jnp.asarray(v))
    tau = _tau_s(state, grid, sp, params)
    dt = 0.05 * tau
    _, (m, _) = _run(state, cs, grid, sp, params, static, key, 80, dt)
    p = np.asarray(m.P)[:, 0]
    stress = p[:, 0, 0] - np.trace(p, axis1=1, axis2=2) / 3.0
    q = np.asarray(m.q)[:, 0, 0]
    rate_stress = _decay_rate(stress, dt) * tau
    rate_q = _decay_rate(q, dt) * tau
    np.testing.assert_allclose(rate_stress, 1.0, rtol=0.06)
    np.testing.assert_allclose(rate_q / rate_stress, 2.0 / 3.0, rtol=0.08)


def test_a_galilean_boost_passes_through():
    state, cs, grid, sp, params, static, key = _box(n_part=3000)
    tau = _tau_s(state, grid, sp, params)
    boost = jnp.array([400.0, -120.0, 35.0])
    (a, _), _ = _run(state, cs, grid, sp, params, static, key, 3, 0.3 * tau, moments=False)
    moved = dataclasses.replace(state, v=state.v + boost)
    (b, _), _ = _run(moved, cs, grid, sp, params, static, key, 3, 0.3 * tau, moments=False)
    np.testing.assert_allclose(np.asarray(b.v), np.asarray(a.v + boost), rtol=0, atol=1e-8)


def test_a_huge_time_step_is_finite_and_thermalises():
    state, cs, grid, sp, params, static, key = _box()
    v = np.asarray(state.v) * np.array([1.4, 0.8, 0.7])
    state = dataclasses.replace(state, v=jnp.asarray(v))
    tau = _tau_s(state, grid, sp, params)
    (out, _), _ = _run(state, cs, grid, sp, params, static, key, 1, 10.0 * 4.5 * tau,
                       moments=False)  # fmt: skip
    w = np.asarray(out.v)
    assert np.isfinite(w).all()
    var = np.var(w, axis=0)
    np.testing.assert_allclose(var / var.mean(), 1.0, atol=0.03)


def test_gaussian_entropy_never_decreases():
    """For a Gaussian start the state stays near-Gaussian; its entropy ``log det Pi`` must
    not fall (H-theorem), up to sampling noise."""
    state, cs, grid, sp, params, static, key = _box(n_part=40000)
    v = np.asarray(state.v) * np.array([1.3, 0.9, 0.8])
    state = dataclasses.replace(state, v=jnp.asarray(v))
    tau = _tau_s(state, grid, sp, params)
    _, (m, _) = _run(state, cs, grid, sp, params, static, key, 40, 0.1 * tau)
    logdet = np.linalg.slogdet(np.asarray(m.P)[:, 0])[1]
    assert np.all(np.diff(logdet) > -2e-3)
    assert logdet[-1] > logdet[0]


def test_gradient_in_the_coefficients_matches_finite_differences():
    state, cs, grid, sp, params, static, key = _box(n_part=2000)
    v = np.asarray(state.v) * np.array([1.3, 0.9, 0.8])
    state = dataclasses.replace(state, v=jnp.asarray(v))
    tau = _tau_s(state, grid, sp, params)

    def stress_after(theta):
        (out, _), _ = _run(state, cs, grid, sp, params, static, key, 4, 0.2 * tau, theta,
                           moments=False)  # fmt: skip
        c = out.v - jnp.mean(out.v, axis=0)
        return jnp.mean(c[:, 0] ** 2) - jnp.mean(c**2) / 3.0

    grad = jax.grad(stress_after)(THETA)
    for name in ("log_pr", "log_tau"):
        assert np.isfinite(float(grad[name]))
        h = 1e-5
        up = dict(THETA, **{name: THETA[name] + h})
        dn = dict(THETA, **{name: THETA[name] - h})
        fd = (float(stress_after(up)) - float(stress_after(dn))) / (2 * h)
        np.testing.assert_allclose(float(grad[name]), fd, rtol=1e-5)


def _shear_box(cells=8, npc=4000, amplitude=60.0):
    """A periodic 1-D box with the transverse wave ``u_y = A sin(2 pi x / L)``."""
    state, cs, grid, sp, params, static, key = _box(n_part=cells * npc, cells=(cells,))
    x = np.asarray(state.x)[:, 0]
    length = float(grid.hi[0] - grid.lo[0])
    v = np.asarray(state.v).copy()
    v[:, 1] += amplitude * np.sin(2 * np.pi * x / length)
    return dataclasses.replace(state, v=jnp.asarray(v)), cs, grid, sp, params, static, key


def _in_cell_shear(state, grid, cells):
    """Per cell, the slope of ``v_y`` against ``x`` inside the cell (least squares)."""
    x = np.asarray(state.x)[:, 0]
    vy = np.asarray(state.v)[:, 1]
    dx = float(grid.hi[0] - grid.lo[0]) / cells
    cell = np.minimum((x / dx).astype(int), cells - 1)
    return np.array([np.polyfit(x[cell == c], vy[cell == c], 1)[0] for c in range(cells)])


def test_position_aware_target_conserves_and_keeps_the_in_cell_shear():
    """With the reconstructed target a coarse cell keeps its internal velocity profile
    through a collision step; the cell-mean target relaxes it away (the coarse-cell
    excess viscosity).  Momentum and energy stay exact in every cell either way."""
    cells = 8
    state, cs, grid, sp, params, static, key = _shear_box(cells)
    tau = _tau_s(state, grid, sp, params)
    recon = functools.partial(relax.collide, periodic=(True,))
    slope0 = _in_cell_shear(state, grid, cells)
    kept = {}
    for name, op in (("mean", OPERATOR), ("recon", recon)):
        st = dataclasses.replace(static, n_steps=1)
        pr = dataclasses.replace(params, dt=jnp.float64(4.0 * tau))
        (out, _), _ = simulate.run(state, cs, grid, sp, pr, st, key, collide_only=True,
                                   emit_moments="none", collide_fn=op,
                                   collision_params=THETA)  # fmt: skip
        from jax_dsmc import sort  # noqa: PLC0415

        cell = np.asarray(sort.build_table(state, grid).cell)
        m0, e0 = _cell_momentum_energy(state, cell, cells)
        m1, e1 = _cell_momentum_energy(out, cell, cells)
        np.testing.assert_allclose(m1, m0, rtol=0, atol=1e-8 * np.abs(m0).max())
        np.testing.assert_allclose(e1, e0, rtol=1e-11)
        kept[name] = float(np.mean(_in_cell_shear(out, grid, cells) / slope0))
    # four stress times: a = exp(-(2/3) 4 / 3) = 0.41 -- the cell-mean target damps it so
    assert kept["mean"] < 0.5
    assert kept["recon"] > 0.8  # the reconstruction keeps it (gradient from noisy means)
