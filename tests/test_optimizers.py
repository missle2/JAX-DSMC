"""The optimiser's step rules (examples/mesh/optimizers.py) on small numpy problems: the trust
region's judgement, L-BFGS on a quadratic, and one MMA subproblem under the area constraint."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "mesh"))
import optimizers as OZ


def test_trust_region_grows_on_agreement_shrinks_and_rejects_on_a_rise():
    state = {"radius": 0.05, "base_objective": 1.0, "predicted": -0.1, "at_edge": True}
    ok, rho, radius = OZ.judge(state, 0.91, 1e-4, 0.05)
    assert ok and abs(rho - 0.9) < 1e-12 and radius == 0.05 * OZ.GROW
    ok, rho, radius = OZ.judge(state, 0.99, 1e-4, 0.05)
    assert ok and radius == 0.05 * OZ.SHRINK
    ok, rho, radius = OZ.judge(state, 1.02, 1e-4, 0.05)
    assert not ok and rho < 0 and radius == 0.05 * OZ.SHRINK
    # a prediction below the seed noise is inconclusive
    ok, rho, radius = OZ.judge(dict(state, predicted=-1e-5), 1.02, 1e-3, 0.05)
    assert ok and np.isnan(rho) and radius == 0.05
    # the radius stays within its bounds
    lo, hi = OZ.radius_bounds(0.05)
    assert OZ.judge(dict(state, radius=lo), 1.02, 1e-4, 0.05)[2] == lo
    assert OZ.judge(dict(state, radius=hi), 0.91, 1e-4, 0.05)[2] == hi


def test_lbfgs_solves_a_quadratic_with_the_identity_preconditioner():
    rng = np.random.default_rng(0)
    n = 12
    a = rng.normal(size=(n, n))
    hess = a @ a.T + n * np.eye(n)
    x_star = rng.normal(size=n)

    def grad(x):
        return hess @ (x - x_star)

    x = np.zeros(n)
    pairs = []
    g = grad(x)
    for _ in range(40):
        if np.linalg.norm(g) < 1e-12:
            break
        d, _ = OZ.lbfgs_direction(g, pairs, lambda v: v)
        # exact line search on the quadratic
        t = -float(g @ d) / float(d @ hess @ d)
        x_new = x + t * d
        g_new = grad(x_new)
        pairs, _ = OZ.add_pair(pairs, x_new - x, g_new - g)
        x, g = x_new, g_new
    assert np.linalg.norm(x - x_star) < 1e-6 * np.linalg.norm(x_star)
    # negative curvature is not kept
    assert not OZ.add_pair([], np.ones(3), -np.ones(3))[1]


def test_mma_subproblem_decreases_a_linear_objective_at_the_area_bound():
    """Shrinking pays (the objective falls with the area, like drag): the step falls in the
    objective, keeps the linearised area at its bound, every move within the move limit."""
    n = 40
    x = np.zeros(n)
    g_area = np.full(n, -1.0)  # area = A0 - sum x (phi up -> body shrinks)
    g0 = np.linspace(0.5, 1.5, n)  # every variable pays to shrink, the right ones more
    area0 = 100.0
    low, upp = OZ.mma_asymptotes({}, x, 0.05, 0)
    xs, lam = OZ.mma_step(x, g0, area0, g_area, area0, low, upp, 0.05)
    dx = xs - x
    assert float(g0 @ dx) < 0.0
    assert float(g_area @ dx) >= -1e-6 * area0  # the area does not fall (to first order)
    assert np.abs(dx).max() <= 0.05 + 1e-12
    assert lam[0] > 0.0  # the constraint is active
