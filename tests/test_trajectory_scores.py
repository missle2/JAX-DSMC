"""Production likelihood bookkeeping, distinct from P2 terminal-label oracles."""
import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc.collide.ntc import collide, attempt_count
from jax_dsmc.sort import build_table
from tests.test_ntc import build


def test_full_trajectory_counts_each_pair_and_cell_once():
    state, cs, grid, sp, params, static, temp, key = build(N=16, A_max=32, dt=1e-14, n=(2,))
    table = build_table(state, grid)
    def scores(dt):
        import dataclasses
        p = dataclasses.replace(params, dt=dt)
        local, _, _ = collide(state, table, cs, grid, sp, p, static, temp, 0, key)
        full, _, _ = collide(state, table, cs, grid, sp, p, static, temp, 0, key,
                             score_mode="trajectory")
        # Tiny dt: this seed draws no attempts. Each cell's count event occurs once,
        # whereas the paper accumulator repeats it for every particle in that cell.
        from jax_dsmc.rng import key_for, STAGE_NTC_ROUND
        from jax_dsmc.grid import cell_volume
        _, lp = attempt_count(table.counts, cs.vremax[:, 0, 0], cell_volume(grid), dt,
                              p.fnum, key_for(key, 0, STAGE_NTC_ROUND))
        return full.logh_acc, lp.sum()
    got, expected = scores(params.dt)
    np.testing.assert_allclose(got, expected, atol=1e-14)
    dg, de = jax.jacfwd(scores)(params.dt)
    np.testing.assert_allclose(dg, de, rtol=1e-12)
