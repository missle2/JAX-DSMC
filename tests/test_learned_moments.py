"""The coarse-cell moment estimator of :mod:`examples.learned_collision.moments`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from examples.learned_collision import moments


def test_central_moments_match_a_direct_computation():
    rng = np.random.default_rng(0)
    n, cells = 400, 5
    x = rng.uniform(0.0, 1.0, n)
    v = rng.standard_normal((n, 3)) * [1.0, 2.0, 0.5] + [0.3, -1.0, 2.0]
    v[:, 0] += 0.5 * v[:, 1] ** 2  # skewed, so the heat flux is nonzero
    alive = rng.uniform(size=n) > 0.1
    got = np.asarray(moments.central(*moments.raw_sums(
        jnp.asarray(x), jnp.asarray(v), jnp.asarray(alive), 0.0, 1.0, cells)))  # fmt: skip
    cell = np.minimum((x * cells).astype(int), cells - 1)
    for k in range(cells):
        vk = v[(cell == k) & alive]
        m = len(vk)
        c = vk - vk.mean(0)
        cov = np.cov(vk.T, ddof=1)
        q = 0.5 * m**2 / ((m - 1) * (m - 2)) * np.mean(c * np.sum(c**2, 1, keepdims=True), 0)
        want = [m, *vk.mean(0), cov[0, 0], cov[1, 1], cov[2, 2], cov[0, 1], cov[0, 2],
                cov[1, 2], *q]  # fmt: skip
        np.testing.assert_allclose(got[k], want, rtol=1e-10, atol=1e-12)


def test_small_sample_moments_are_unbiased():
    """Five particles per cell: the factors make the expected stress and heat flux equal to
    the population values (exactly, for any distribution)."""
    rng = np.random.default_rng(1)
    reps, m = 40000, 5
    g = rng.standard_normal((reps * m, 3))
    v = np.stack([g[:, 0] + 0.4 * (g[:, 1] ** 2 - 1.0), g[:, 1], 0.7 * g[:, 2]], axis=1)
    x = (np.repeat(np.arange(reps), m) + 0.5) / reps
    out = np.asarray(moments.central(*moments.raw_sums(
        jnp.asarray(x), jnp.asarray(v), jnp.ones(reps * m, bool), 0.0, 1.0, reps)))  # fmt: skip
    big = np.stack([rng.standard_normal(4_000_000) for _ in range(3)], axis=1)
    pop = np.stack([big[:, 0] + 0.4 * (big[:, 1] ** 2 - 1.0), big[:, 1], 0.7 * big[:, 2]], 1)
    c = pop - pop.mean(0)
    want_p = np.cov(pop.T)
    want_q = 0.5 * np.mean(c * np.sum(c**2, 1, keepdims=True), 0)
    se = out.std(0) / np.sqrt(reps)
    mean = out.mean(0)
    for col, want in ((4, want_p[0, 0]), (7, want_p[0, 1]), (10, want_q[0]), (11, want_q[1])):
        assert abs(mean[col] - want) < 4.0 * se[col] + 5e-3, (moments.FIELDS[col], mean[col], want)


def _total_pxy(v, x, alive):
    return jnp.sum(moments.central(*moments.raw_sums(x, v, alive, 0.0, 1.0, 4))[:, 7])


def test_moments_are_differentiable_in_v():
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.uniform(0.0, 1.0, 200))
    v = jnp.asarray(rng.standard_normal((200, 3)))
    g = jax.grad(_total_pxy)(v, x, jnp.ones(200, bool))
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).sum()) > 0.0
