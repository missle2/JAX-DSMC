"""P3 ensemble-adjoint identities independent of particle trajectory AD."""
import numpy as np

from jax_dsmc import ensemble_inflow as ei


def test_affine_reconstruction_reproduces_affine_field():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(128, 4))
    coeff = np.array([.4, -.7, .2, 1.1])
    field = ei.LocalAffine(points, 2.+points@coeff, neighbors=32)
    query = rng.normal(size=(20, 4))
    np.testing.assert_allclose(field(query), 2.+query@coeff, atol=1e-7)


def test_flux_scores_have_zero_mean_and_correct_energy_derivatives():
    # Rayleigh normal energy mean=2T; each tangential mean=T.
    rng = np.random.default_rng(4)
    temp = np.array([[.6, .7, .8], [.9, 1., 1.1]])
    for side in (0, 1):
        v = ei.sample_flux(rng, 200000, temp[side], side)
        score = ei.temperature_score(v, temp[side])
        np.testing.assert_allclose(score.mean(axis=0), 0., atol=.025)
        energy = (v*v).sum(axis=1)
        grad = ((energy-energy.mean())[:, None]*score).mean(axis=0)
        np.testing.assert_allclose(grad, [2., 1., 1.], atol=.07)


def test_linearized_collision_has_both_partner_terms_and_conserves_invariants():
    rng = np.random.default_rng(2)
    v, w = rng.normal(size=(2, 20, 3))
    vp, wp = ei.scatter_maxwell(v, w, rng)
    for fn in (lambda u: np.ones(len(u)), lambda u: u[:, 0],
               lambda u: np.sum(u*u, axis=1)):
        np.testing.assert_allclose(ei.four_term(fn(vp), fn(wp), fn(v), fn(w)), 0., atol=1e-14)


def test_reservoir_uses_full_counts_and_fixed_source_normalization():
    cfg = ei.Config(n_initial=200, cells=4, dt=.025, steps=2, retained=8,
                    source_samples=16, partners=2)
    trace = ei.forward(cfg, np.full((2, 3), 4.), seed=1)
    assert trace.normalization == cfg.n_initial+cfg.steps*sum(trace.injection_counts)
    assert trace.frames[0].counts.sum() == cfg.n_initial
    assert trace.frames[0].valid.sum() <= cfg.cells*cfg.retained
    assert trace.frames[0].counts.sum() > trace.frames[0].valid.sum()
    assert np.isfinite(ei.backward(trace, cfg, seed=5)).all()
