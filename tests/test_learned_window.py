"""Coarse training windows (:mod:`examples.learned_collision.windows`): thinning and the
reverse-mode gradient of the two-replica window loss."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from examples.learned_collision import shear_wave, windows
from jax_dsmc.collide import relax

LENGTH = 16.0


def test_thinning_keeps_the_expected_fraction_in_order():
    x = jnp.arange(10000.0)
    v = jnp.stack([x, -x, 2 * x], axis=1)
    xs, vs, alive, overflow = windows.thin(x, v, 0.1, jax.random.PRNGKey(0), 1200)
    n = int(alive.sum())
    assert int(overflow) == 0
    assert abs(n - 1000) < 4 * np.sqrt(900)
    kept = np.asarray(xs)[:n]
    assert np.all(np.diff(kept) > 0)
    np.testing.assert_array_equal(np.asarray(vs)[:n, 2], 2 * kept)
    _, _, _, overflow = windows.thin(x, v, 0.1, jax.random.PRNGKey(0), 500)
    assert int(overflow) == n - 500


def _window():
    """Two replicas of a thinned 16-lambda shear wave on 2-lambda cells, ES-FP."""
    fine, *_, c = shear_wave.setup(0.25, 50, 0, 0, "shear", LENGTH)
    case = windows.coarse_case("shear", LENGTH, 2.0, 20, c["n_part"])
    snap_x = np.asarray(fine.x) / c["lambda"]
    snap_v = np.asarray(fine.v) / c["vth"]
    pairs = [windows.initial_state(snap_x, snap_v, case, jax.random.PRNGKey(k)) for k in (1, 2)]
    assert all(int(p[2]) == 0 for p in pairs)
    states = jax.tree.map(shear_wave._stack, *[p[0] for p in pairs])
    cstates = jax.tree.map(shear_wave._stack, *[p[1] for p in pairs])
    keys = jnp.stack([jax.random.PRNGKey(11), jax.random.PRNGKey(12)])
    ref = jnp.zeros((2, case.n_coarse, 13))
    return states, cstates, keys, ref, case


def _loss(log_tau, log_pr, states, cstates, keys, ref, case):
    theta = {"log_tau": log_tau, "log_pr": log_pr}
    return windows.window_loss(theta, states, cstates, keys, ref, jnp.ones(2), case,
                               relax.collide, blocks=2)  # fmt: skip


def test_window_loss_gradient_matches_finite_differences():
    states, cstates, keys, ref, case = _window()
    f = jax.jit(functools.partial(_loss, states=states, cstates=cstates, keys=keys, ref=ref,
                                  case=case))  # fmt: skip
    t0, p0 = jnp.float64(0.0), jnp.float64(np.log(2.0 / 3.0))
    g_tau, g_pr = jax.jit(jax.grad(f, argnums=(0, 1)))(t0, p0)
    h = 1e-5
    fd_tau = (f(t0 + h, p0) - f(t0 - h, p0)) / (2 * h)
    fd_pr = (f(t0, p0 + h) - f(t0, p0 - h)) / (2 * h)
    assert float(jnp.abs(g_tau)) > 0.0
    np.testing.assert_allclose(float(g_tau), float(fd_tau), rtol=1e-4)
    np.testing.assert_allclose(float(g_pr), float(fd_pr), rtol=1e-4, atol=1e-6 * abs(float(g_tau)))
