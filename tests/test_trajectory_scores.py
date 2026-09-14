"""Production likelihood bookkeeping, distinct from P2 terminal-label oracles."""

import dataclasses

import jax
import numpy as np

from jax_dsmc.collide.ntc import attempt_count, collide
from jax_dsmc.grid import cell_volume
from jax_dsmc.rng import STAGE_NTC_ROUND, key_for
from jax_dsmc.sort import build_table
from tests.test_ntc import build


def test_full_trajectory_counts_each_pair_and_cell_once():
    state, cs, grid, sp, params, static, temp, key = build(N=16, A_max=32, dt=1e-14, n=(2,))
    table = build_table(state, grid)

    def scores(dt):
        p = dataclasses.replace(params, dt=dt)
        local, _, _ = collide(state, table, cs, grid, sp, p, static, temp, 0, key)
        full, _, _ = collide(
            state, table, cs, grid, sp, p, static, temp, 0, key, score_mode="trajectory"
        )
        # Tiny dt: this seed draws no attempts. Each cell's count event occurs once,
        # whereas the paper accumulator repeats it for every particle in that cell.
        _, lp = attempt_count(
            table.counts,
            cs.vremax[:, 0, 0],
            cell_volume(grid),
            dt,
            p.fnum,
            key_for(key, 0, STAGE_NTC_ROUND),
        )
        return full.logh_acc, lp.sum()

    got, expected = scores(params.dt)
    np.testing.assert_allclose(got, expected, atol=1e-14)
    dg, de = jax.jacfwd(scores)(params.dt)
    np.testing.assert_allclose(dg, de, rtol=1e-12)


def test_full_trajectory_counts_accepted_and_rejected_pairs_once():
    state, cs, grid, sp, params, static, temp, key = build(N=64, A_max=256, dt=1e-3)
    table = build_table(state, grid)
    local, _, diag = collide(state, table, cs, grid, sp, params, static, temp, 0, key)
    full, _, _ = collide(
        state, table, cs, grid, sp, params, static, temp, 0, key, score_mode="trajectory"
    )
    assert int(diag.max_attempts) > 0
    assert int(diag.attempts_clipped) == 0
    _, lp = attempt_count(
        table.counts,
        cs.vremax[:, 0, 0],
        cell_volume(grid),
        params.dt,
        params.fnum,
        key_for(key, 0, STAGE_NTC_ROUND),
    )
    expected = 0.5 * local.logh_acc.sum() + ((1 - 0.5 * table.counts) * lp).sum()
    np.testing.assert_allclose(full.logh_acc, expected, atol=1e-12)
    np.testing.assert_array_equal(full.v, local.v)
