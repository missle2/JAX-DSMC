"""The replacement-collision hook of :func:`jax_dsmc.simulate.run` (``collide_fn`` /
``collision_params``) and the padded per-cell layout :func:`jax_dsmc.sort.padded_cells`."""

from __future__ import annotations

import dataclasses
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import test_ntc

from jax_dsmc import simulate, sort
from jax_dsmc.collide import ntc


def _ntc_passthrough(state, table, cstate, grid, sp, params, static, temp, k, key, theta):
    """NTC through the hook: must reproduce the default path bit for bit."""
    del theta
    return ntc.collide(state, table, cstate, grid, sp, params, static, temp, k, key)[:3]


def _scale(state, table, cstate, grid, sp, params, static, temp, k, key, theta):
    """A toy operator, ``v <- theta v``: makes the parameter's gradient analytic."""
    del grid, sp, params, static, temp, k, key
    diag = ntc.replacement_diagnostics(state, table)
    return dataclasses.replace(state, v=state.v * theta), cstate, diag


def _case(n_steps):
    state, cs, grid, sp, params, static, _temp, key = test_ntc.build(
        n=(4,), N=400, A_max=64, dt=2e-9
    )
    return state, cs, grid, sp, params, dataclasses.replace(static, n_steps=n_steps), key


def test_passthrough_reproduces_the_default_collision_bitwise():
    state, cs, grid, sp, params, static, key = _case(5)
    (a, ca), _ = simulate.run(state, cs, grid, sp, params, static, key, emit_moments="none")
    (b, cb), _ = simulate.run(
        state, cs, grid, sp, params, static, key, emit_moments="none",
        collide_fn=_ntc_passthrough, collision_params=jnp.float64(0.0),
    )  # fmt: skip
    for x, y in zip(jax.tree.leaves((a, ca)), jax.tree.leaves((b, cb)), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_gradient_reaches_the_collision_parameters():
    """``v <- theta v`` each step, free flight in a periodic box: the kinetic energy after
    ``n`` steps is ``theta^(2n) sum v0^2``, so ``dJ/dtheta = 2n theta^(2n-1) sum v0^2``."""
    n = 4
    state, cs, grid, sp, params, static, key = _case(n)

    def objective(theta):
        (out, _), _ = simulate.run(
            state, cs, grid, sp, params, static, key, emit_moments="none",
            collide_fn=_scale, collision_params=theta,
        )  # fmt: skip
        return jnp.sum(jnp.where(out.alive[:, None], out.v**2, 0.0))

    theta = jnp.float64(0.97)
    v0 = float(jnp.sum(jnp.where(state.alive[:, None], state.v**2, 0.0)))
    got = float(jax.grad(objective)(theta))
    want = 2 * n * float(theta) ** (2 * n - 1) * v0
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_capture_is_refused_with_a_replacement_operator():
    state, cs, grid, sp, params, static, key = _case(1)
    with pytest.raises(ValueError, match="capture"):
        simulate.run(
            state, cs, grid, sp, params, static, key, capture=True,
            collide_fn=_scale, collision_params=jnp.float64(1.0),
        )  # fmt: skip


def test_padded_cells_lists_every_cell_member_and_counts_overflow():
    state, _cs, grid, *_ = _case(1)
    state = dataclasses.replace(state, alive=state.alive.at[::7].set(False))
    table = sort.build_table(state, grid)
    counts = np.asarray(table.counts)
    cell = np.asarray(table.cell)
    ids, valid, overflow = sort.padded_cells(table, int(counts.max()))
    ids, valid = np.asarray(ids), np.asarray(valid)
    assert int(overflow) == 0
    for c in range(counts.size):
        np.testing.assert_array_equal(np.sort(ids[c][valid[c]]), np.nonzero(cell == c)[0])
    cap = int(counts.max()) - 3
    _ids, valid_cap, overflow = sort.padded_cells(table, cap)
    assert int(overflow) == int(np.maximum(counts - cap, 0).sum()) > 0
    np.testing.assert_array_equal(np.asarray(valid_cap).sum(axis=1), np.minimum(counts, cap))
