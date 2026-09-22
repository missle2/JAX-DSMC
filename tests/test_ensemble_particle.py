"""Route (b), the particle-carried ensemble adjoint, against route (a) and exact oracles."""

import numpy as np
import pytest
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss

from jax_dsmc import ensemble_inflow as ei
from jax_dsmc import ensemble_particle as ep


def _quadrature_gradient(cfg, temp, counts, normalization):
    """Collisionless expected objective by Gauss-Laguerre x Gauss-Legendre, differenced."""
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
        return total / normalization

    grad = np.zeros((2, 3))
    for i in np.ndindex(2, 3):
        plus, minus = temp.copy(), temp.copy()
        plus[i] += 1e-5
        minus[i] -= 1e-5
        grad[i] = (expected(plus) - expected(minus)) / 2e-5
    return grad


def test_forward_is_bit_identical_to_the_reference():
    cfg = ei.Config(
        n_initial=300, cells=8, dt=0.025, steps=4, retained=8, source_samples=16, partners=2
    )
    temp = np.array([[4.0, 3.0, 5.0], [2.0, 4.0, 4.0]])
    counts = (50, 70)
    a = ei.forward(cfg, temp, seed=3, injection_counts=counts)
    b = ep.forward(cfg, temp, seed=3, injection_counts=counts)
    assert a.objective == b.objective
    assert a.normalization == b.normalization
    assert a.frames[-1].counts.sum() == len(b.final)


def test_collisionless_gradient_matches_quadrature_without_any_interpolation():
    """With q0 = 0 route (b) never interpolates: every birth carries its own adjoint."""
    cfg = ei.Config(n_initial=100, cells=4, dt=0.025, steps=3, q0=0.0, retained=4, source_samples=4)
    temp = np.array([[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]])
    counts = (40000, 40000)
    trace = ep.forward(cfg, temp, seed=8, injection_counts=counts)
    got = ep.backward(trace)
    expected = _quadrature_gradient(cfg, temp, counts, trace.normalization)
    # normal components agree to <1 %; the tiny tangential ones carry the 40000-sample
    # Monte Carlo noise of the source itself (same tolerance as the route-(a) oracle)
    np.testing.assert_allclose(got[:, 0], expected[:, 0], rtol=0.02)
    np.testing.assert_allclose(got, expected, rtol=0.13, atol=5e-5)


def test_recorded_pairs_are_the_collisions_that_happened():
    """Energy is conserved pair by pair between the recorded pre state and the next frame."""
    cfg = ei.Config(n_initial=400, cells=4, dt=0.05, steps=2, retained=4, source_samples=4)
    trace = ep.forward(cfg, np.full((2, 3), 4.0), seed=1, injection_counts=(0, 0))
    for k, st in enumerate(trace.steps):
        if len(st.pairs) == 0:
            continue
        a, b = st.pairs[:, 0], st.pairs[:, 1]
        nxt = trace.steps[k + 1].pre if k + 1 < len(trace.steps) else trace.final
        sa, sb = st.next_slot[a], st.next_slot[b]
        ok = (sa >= 0) & (sb >= 0)
        e_pre = np.sum(st.pre[a[ok], 1:] ** 2, axis=1) + np.sum(st.pre[b[ok], 1:] ** 2, axis=1)
        e_post = np.sum(nxt[sa[ok], 1:] ** 2, axis=1) + np.sum(nxt[sb[ok], 1:] ** 2, axis=1)
        np.testing.assert_allclose(e_post, e_pre, rtol=1e-12)
        # and the transported ghost of an uncollided survivor IS its next state
        free = np.setdiff1d(np.flatnonzero(st.survived), np.concatenate((a, b)))[:50]
        ghost = st.pre[free].copy()
        ghost[:, 0] += cfg.dt * ghost[:, 1]
        np.testing.assert_allclose(nxt[st.next_slot[free]], ghost, rtol=1e-12)


@pytest.mark.parametrize("seed", [0, 1])
def test_routes_a_and_b_agree_on_the_same_trajectory(seed):
    """Paired on one forward realization, both routes estimate the same adjoint
    gradient; the difference is interpolation noise/bias, bounded here at the
    level the retained-frame route resolves at this budget."""
    cfg = ei.Config(
        n_initial=800,
        cells=20,
        dt=0.025,
        steps=8,
        retained=256,
        source_samples=400,
        partners=8,
        neighbors=32,
        reconstruction="ballistic",
    )
    temp = np.full((2, 3), 4.0)
    counts = ei.nominal_counts(cfg, temp)
    ga = ei.backward(ei.forward(cfg, temp, seed=seed, injection_counts=counts), seed=seed)
    gb = ep.backward(ep.forward(cfg, temp, seed=seed, injection_counts=counts))
    assert np.isfinite(gb).all()
    np.testing.assert_allclose(gb[:, 0], ga[:, 0], rtol=0.25)


def test_one_step_routes_differ_only_by_collision_quadrature():
    """For a single step the adjoint field is the terminal reward exactly, so route (a)
    averaged over its partner draws and route (b) on the realized pairs estimate the
    same conditional expectation; (b) is deterministic given the forward."""
    cfg = ei.Config(
        n_initial=20000,
        cells=20,
        dt=0.05,
        steps=1,
        retained=1000,
        source_samples=100000,
        partners=8,
        neighbors=32,
    )
    temp = np.full((2, 3), 4.0)
    counts = (5000, 5000)
    ta = ei.forward(cfg, temp, seed=4, injection_counts=counts)
    tb = ep.forward(cfg, temp, seed=4, injection_counts=counts)
    assert ta.objective == tb.objective
    gb = ep.backward(tb, exact_last=True)[:, 0]
    ga = np.array([ei.backward(ta, seed=s)[:, 0] for s in range(8)])
    se = ga.std(0, ddof=1) / np.sqrt(len(ga))
    # (a)'s retained subsample (1000 of 1000 -- all particles at N0/cells) and partner
    # draws leave a small residual; (b)'s pair noise is averaged by the 10^4 births
    assert np.all(np.abs(ga.mean(0) - gb) < 3 * se + 0.01 * np.abs(gb)), (ga.mean(0), se, gb)
