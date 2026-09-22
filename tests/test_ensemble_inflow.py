"""P3 ensemble-adjoint identities independent of particle trajectory AD."""

import numpy as np
import pytest
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss

from jax_dsmc import ensemble_inflow as ei


def test_affine_reconstruction_reproduces_affine_field():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(128, 4))
    coeff = np.array([0.4, -0.7, 0.2, 1.1])
    field = ei.LocalAffine(points, 2.0 + points @ coeff, neighbors=32)
    query = rng.normal(size=(20, 4))
    np.testing.assert_allclose(field(query), 2.0 + query @ coeff, atol=1e-7)


def test_flux_scores_have_zero_mean_and_correct_energy_derivatives():
    # Rayleigh normal energy mean=2T; each tangential mean=T.
    rng = np.random.default_rng(4)
    temp = np.array([[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]])
    for side in (0, 1):
        v = ei.sample_flux(rng, 200000, temp[side], side)
        score = ei.temperature_score(v, temp[side])
        np.testing.assert_allclose(score.mean(axis=0), 0.0, atol=0.025)
        energy = (v * v).sum(axis=1)
        grad = ((energy - energy.mean())[:, None] * score).mean(axis=0)
        np.testing.assert_allclose(grad, [2.0, 1.0, 1.0], atol=0.07)


def test_linearized_collision_has_both_partner_terms_and_conserves_invariants():
    rng = np.random.default_rng(2)
    v, w = rng.normal(size=(2, 20, 3))
    vp, wp = ei.scatter_maxwell(v, w, rng)
    for fn in (lambda u: np.ones(len(u)), lambda u: u[:, 0], lambda u: np.sum(u * u, axis=1)):
        np.testing.assert_allclose(ei.four_term(fn(vp), fn(wp), fn(v), fn(w)), 0.0, atol=1e-14)


@pytest.mark.parametrize("reconstruction", ["affine", "ballistic"])
def test_reservoir_uses_full_counts_and_fixed_source_normalization(reconstruction):
    cfg = ei.Config(
        n_initial=200,
        cells=4,
        dt=0.025,
        steps=2,
        retained=8,
        source_samples=16,
        partners=2,
        reconstruction=reconstruction,
    )
    trace = ei.forward(cfg, np.full((2, 3), 4.0), seed=1)
    assert trace.normalization == cfg.n_initial + cfg.steps * sum(trace.injection_counts)
    assert trace.frames[0].counts.sum() == cfg.n_initial
    assert trace.frames[0].valid.sum() <= cfg.cells * cfg.retained
    assert trace.frames[0].counts.sum() > trace.frames[0].valid.sum()
    assert np.isfinite(ei.backward(trace, cfg, seed=5)).all()


def test_collisionless_source_gradient_against_independent_quadrature():
    # Integrate Rayleigh normal energy by Gauss-Laguerre; tangential energies
    # integrate analytically. This checks source time, normalization and all scores.
    cfg = ei.Config(
        n_initial=100, cells=4, dt=0.025, steps=3, q0=0.0, retained=4, source_samples=40000
    )
    temp = np.array([[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]])
    counts = (40000, 40000)
    trace = ei.forward(cfg, temp, seed=8, injection_counts=counts)
    got = ei.backward(trace)
    u, wu = laggauss(64)
    t, wt = leggauss(32)

    def expected(temperature):
        total = 0.0
        for side in (0, 1):
            v = (1 if side == 0 else -1) * np.sqrt(2 * temperature[side, 0] * u)
            energy = v * v + temperature[side, 1:].sum()
            for k in range(cfg.steps):
                age = cfg.dt * ((t + 1) / 2 + cfg.steps - 1 - k)
                x = side + v[:, None] * age[None, :]
                integrand = energy[:, None] * np.exp(-36 * (x - 0.5) ** 2) * ((x >= 0) & (x <= 1))
                total += counts[side] * np.sum(wu[:, None] * wt[None, :] * integrand) / 2
        return total / trace.normalization

    expected_grad = np.zeros((2, 3))
    for i in np.ndindex(2, 3):
        plus, minus = temp.copy(), temp.copy()
        plus[i] += 1e-5
        minus[i] -= 1e-5
        expected_grad[i] = (expected(plus) - expected(minus)) / 2e-5
    np.testing.assert_allclose(got, expected_grad, rtol=0.12, atol=5e-5)


def test_escaped_newborns_still_belong_to_source_sample_and_normalization():
    cfg = ei.Config(n_initial=20, cells=1, dt=2.0, steps=1, q0=0.0, retained=4, source_samples=2000)
    trace = ei.forward(cfg, np.ones((2, 3)), seed=3, injection_counts=(2000, 2000))
    assert trace.normalization == 4020
    assert any(np.any((s.points[:, 0] < 0) | (s.points[:, 0] > 1)) for s in trace.sources[0])
    for s in trace.sources[0]:
        assert len(s.points) == 2000
        assert s.weight == 1.0
    assert trace.frames[-1].counts.sum() < trace.normalization
    assert np.isfinite(ei.backward(trace)).all()


@pytest.mark.parametrize("reconstruction", ["affine", "ballistic"])
def test_custom_terminal_reward_is_carried_to_backward(reconstruction):
    cfg = ei.Config(
        n_initial=100,
        cells=2,
        steps=2,
        retained=16,
        source_samples=16,
        reconstruction=reconstruction,
    )
    ordinary = ei.forward(cfg, np.ones((2, 3)), seed=9, injection_counts=(20, 20))
    scaled = ei.forward(
        cfg,
        np.ones((2, 3)),
        seed=9,
        injection_counts=(20, 20),
        terminal_reward=lambda p: 3 * ei.reward(p),
    )
    np.testing.assert_allclose(scaled.objective, 3 * ordinary.objective, rtol=1e-13)
    np.testing.assert_allclose(
        ei.backward(scaled, seed=5), 3 * ei.backward(ordinary, seed=5), rtol=1e-10, atol=1e-12
    )


def test_backward_never_pairs_a_retained_state_with_itself():
    """A self-pair is an identically zero four-term sample and biases the correction."""
    cfg = ei.Config(
        n_initial=64, cells=2, dt=0.025, steps=1, retained=4, source_samples=4, partners=64
    )
    trace = ei.forward(cfg, np.full((2, 3), 4.0), seed=3)
    seen = {}
    original = ei.four_term

    def spy(target_gain, partner_gain, target_loss, partner_loss):
        seen["loss"] = (np.asarray(target_loss), np.asarray(partner_loss))
        return original(target_gain, partner_gain, target_loss, partner_loss)

    ei.four_term = spy
    try:
        ei.backward(trace, cfg, seed=1)
    finally:
        ei.four_term = original
    target_loss, partner_loss = seen["loss"]
    # with 64 partners per target and <= 4 retained states, sampling WITH self would
    # make target and partner losses coincide for ~1/4 of the samples
    same = np.mean(np.isclose(target_loss, partner_loss))
    assert same < 0.02
