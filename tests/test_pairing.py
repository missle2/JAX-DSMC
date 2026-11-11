"""NTC's 1-D nearest-neighbour partner selection (``SimStatic.pairing = "adjacent"``) and
the x-sorted cell order it indexes (:func:`jax_dsmc.sort.x_sorted_order`)."""

from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import test_ntc

from jax_dsmc import sort
from jax_dsmc.collide import ntc
from jax_dsmc.state import SimStatic


def _case(pairing, seed=0):
    state, cs, grid, sp, params, static, temp, key = test_ntc.build(
        n=(4,), N=400, A_max=128, dt=3e-4, seed=seed
    )
    return state, cs, grid, sp, params, dataclasses.replace(static, pairing=pairing), temp, key


def _collide(pairing, seed=0):
    state, cs, grid, sp, params, static, temp, key = _case(pairing, seed)
    table = sort.build_table(state, grid)
    out, _, diag, pairs, _ = ntc.collide(
        state, table, cs, grid, sp, params, static, temp, 0, key, capture=True
    )
    return state, table, out, diag, pairs


def test_x_sorted_order_keeps_the_cells_and_ranks_by_x():
    state, _, grid, *_ = _case("uniform")
    table = sort.build_table(state, grid)
    order = np.asarray(sort.x_sorted_order(state, table))
    cell, x = np.asarray(table.cell), np.asarray(state.x[:, 0])
    np.testing.assert_array_equal(cell[order], cell[np.asarray(table.order)])
    for c, (s, n) in enumerate(
        zip(np.asarray(table.starts), np.asarray(table.counts), strict=True)
    ):
        members = order[s : s + n]
        assert np.all(cell[members] == c)
        assert np.all(np.diff(x[members]) >= 0.0)


def test_adjacent_partners_are_x_neighbours_and_conserve():
    state, table, out, diag, pairs = _collide("adjacent")
    order = np.asarray(sort.x_sorted_order(state, table))
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size) - np.asarray(table.starts)[np.asarray(table.cell)[order]]
    counts = np.asarray(table.counts)
    acc = np.asarray(pairs.accept)
    pi, pj = np.asarray(pairs.pi)[acc], np.asarray(pairs.pj)[acc]
    cells = np.broadcast_to(np.arange(counts.size), acc.shape)[acc]
    assert pi.size > 50
    gap = (rank[pj] - rank[pi]) % counts[cells]
    np.testing.assert_array_equal(gap, 1)
    assert int(diag.attempts_clipped) == 0
    np.testing.assert_allclose(np.sum(out.v, 0), np.sum(state.v, 0), atol=1e-9)
    np.testing.assert_allclose(np.sum(out.v**2), np.sum(state.v**2), rtol=1e-12)


def test_adjacent_partners_are_close_and_collide_at_the_uniform_rate():
    """Equilibrium: positions and velocities are independent, so the acceptance count is
    the uniform scheme's in distribution; the partner separation drops from ~dx/3 to
    ~2 dx / N_c."""
    sep, accepted = {}, {}
    for pairing in ("uniform", "adjacent"):
        seps, n_acc = [], 0
        for seed in range(4):
            state, _, _, _, pairs = _collide(pairing, seed)
            acc = np.asarray(pairs.accept)
            x = np.asarray(state.x[:, 0])
            pi, pj = np.asarray(pairs.pi)[acc], np.asarray(pairs.pj)[acc]
            seps.append(np.abs(x[pi] - x[pj]))
            n_acc += int(acc.sum())
        sep[pairing], accepted[pairing] = np.mean(np.concatenate(seps)), n_acc
    dx = 1e-4 / 4
    assert 0.25 * dx < sep["uniform"] < 0.42 * dx
    assert sep["adjacent"] < 0.1 * dx
    assert abs(accepted["adjacent"] - accepted["uniform"]) < 5 * np.sqrt(accepted["uniform"])


def test_unknown_pairing_is_rejected():
    with pytest.raises(ValueError, match="pairing must be uniform or adjacent"):
        SimStatic(N=8, D=1, S=1, G=1, A_max=4, K_bounce=1, n_steps=1, pairing="nearest")
