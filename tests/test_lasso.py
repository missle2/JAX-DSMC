"""The Gauss-Newton Lasso's solver (examples.learned_collision.lasso): coordinate descent
against closed forms, the bounds and the one-standard-error rule."""

from __future__ import annotations

import numpy as np

from examples.learned_collision import lasso


def test_orthonormal_design_is_soft_thresholding():
    c = np.array([3.0, -0.5, 1.2, -2.0])
    a = lasso.lasso_cd(np.eye(4), c, 1.0)
    np.testing.assert_allclose(a, [2.0, 0.0, 0.2, -1.0], atol=1e-12)


def test_correlated_design_matches_a_brute_force_minimum():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3))
    x[:, 2] = x[:, 0] + 0.1 * rng.normal(size=50)  # near-collinear pair
    y = x @ np.array([1.0, -0.5, 0.0]) + 0.1 * rng.normal(size=50)
    g, c, alpha = x.T @ x / 50, x.T @ y / 50, 0.05
    a = lasso.lasso_cd(g, c, alpha, sweeps=5000)

    def f(b):
        return 0.5 * b @ g @ b - c @ b + alpha * np.abs(b).sum()

    for _ in range(200):  # no random perturbation does better
        assert f(a) <= f(a + 1e-3 * rng.normal(size=3)) + 1e-12


def test_bounds_hold_and_bind():
    box = np.array([[-np.inf, -np.inf, 0.0], [0.0, np.inf, np.inf]])
    a = lasso.lasso_cd(np.eye(3), np.array([2.0, 2.0, -2.0]), 0.5, bounds=box)
    np.testing.assert_allclose(a, [0.0, 1.5, 0.0])


def test_sign_bounds_hold_the_stress_rate_row_nonnegative():
    lower, upper = lasso._bounds(True).reshape(2, 2, -1)
    for name in lasso.POSITIVE_RATE:
        assert lower[0, lasso.relax.FEATURES.index(name)] == 0.0
    assert np.all(np.isinf(lower[1])) and np.all(np.isinf(upper))


def test_one_standard_error_rule_takes_the_sparsest_fit_in_range():
    alphas = np.array([1.0, 0.5, 0.1, 0.01])
    # fold means 5.0, 2.03, 2.0, 2.3; s.e. at the minimum 0.058 -> alpha 0.5 is in range
    errors = np.array([[5.0, 2.03, 2.0, 2.3], [5.0, 2.13, 2.1, 2.4], [5.0, 1.93, 1.9, 2.2]])
    chosen, best = lasso.one_se(errors, alphas)
    assert best == 0.1
    assert chosen == 0.5
