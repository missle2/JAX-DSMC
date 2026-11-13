"""The training step (examples/learned_collision/train.py, 2026-11-12): Adam's step divided by
each weight matrix's fan-in, the decoupled weight shrink, frozen leaves, tail averaging."""

from __future__ import annotations

import argparse
import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from examples.learned_collision import train
from jax_dsmc.collide import relax


def _args(**kw):
    base = {"lr": 3e-3, "weight_decay": 0.0, "l1": 0.0, "model": "mlp", "train_intercept": False}
    return argparse.Namespace(**{**base, **kw})


def _ones(x):
    return jnp.ones_like(x)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_first_step_is_lr_over_fan_in(kind):
    theta = relax.init_correction(kind, jax.random.PRNGKey(0), width=8, depth=2)
    args = _args(model=kind)
    opt = optax.adam(args.lr)
    grads = jax.tree.map(_ones, theta)
    new, _, _ = train._step(theta, grads, opt.init(theta), opt, args, train._frozen(args))
    fan = train._fan_in(theta)
    for k in theta:
        step = np.asarray(new[k] - theta[k])
        if k in ("log_tau", "log_pr"):
            assert np.all(step == 0.0)  # frozen intercepts
        else:  # Adam's first step is lr * sign(g)
            np.testing.assert_allclose(step, -args.lr / fan[k], rtol=1e-6)
    if kind == "linear":
        assert fan["A"] == len(relax.FEATURES)
    else:
        assert fan["W0"] == len(relax.FEATURES) and fan["W2"] == 8.0


def test_weight_decay_is_a_per_step_shrink():
    theta = relax.init_correction("mlp", jax.random.PRNGKey(1), width=8, depth=1)
    args = _args(weight_decay=1e-2)
    opt = optax.adam(args.lr)
    zero = jax.tree.map(jnp.zeros_like, theta)
    new, _, _ = train._step(theta, zero, opt.init(theta), opt, args, train._frozen(args))
    np.testing.assert_allclose(new["W0"], 0.99 * theta["W0"], rtol=1e-12)
    np.testing.assert_array_equal(new["b0"], theta["b0"])


def test_tail_average():
    thetas = [{"A": jnp.full((2, 3), float(i))} for i in range(4)]
    avg = jax.tree.map(jnp.zeros_like, thetas[0])
    for t in thetas[2:]:
        avg = jax.tree.map(functools.partial(train._accumulate, w=0.5), avg, t)
    np.testing.assert_allclose(avg["A"], 2.5)


def test_chain_windows_and_write_back_keep_replica_order():
    """Windows take consecutive chain pairs ``(2w, 2w + 1)``; their ``[m, 2]`` end states go
    back to the same chains (train.py ``--warm-pool``)."""
    chains = jnp.arange(8.0)[:, None] * jnp.ones((8, 3))
    win = np.array([3, 1])
    idx = jnp.asarray(np.stack([2 * win, 2 * win + 1], axis=1).ravel())
    w = train._windows_of(chains, idx)
    np.testing.assert_array_equal(w[:, :, 0], [[6.0, 7.0], [2.0, 3.0]])
    back = train._put(chains, train._pairs(w + 100.0), idx)
    np.testing.assert_array_equal(back[:, 0], [0, 1, 102, 103, 4, 5, 106, 107])


def test_shift_moves_a_deviation_between_cells_and_the_shock_frame():
    x = np.arange(1.0, 6.0)[None, :, None] * np.ones((2, 1, 2))  # [T, K = 5, F]
    np.testing.assert_array_equal(train._shift(x, 2)[0, :, 0], [0, 0, 1, 2, 3])
    np.testing.assert_array_equal(train._shift(x, -1)[0, :, 0], [2, 3, 4, 5, 0])
    np.testing.assert_array_equal(train._shift(train._shift(x, 1), -1)[0, 1:4, 0], [2, 3, 4])
    np.testing.assert_array_equal(train._shift(x, 0), x)
