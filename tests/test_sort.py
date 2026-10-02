"""The cell sort (``jax_dsmc.sort``): the carried partition sort (``SimStatic.sort =
"partition"``) against the full stable sort -- the same cells, counts and members exactly,
the within-cell order as documented, and a collisional run's statistics unchanged."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import test_ntc  # noqa: E402  (its collisional single-species case)

from jax_dsmc import simulate, sort  # noqa: E402
from jax_dsmc.grid import num_cells  # noqa: E402
from jax_dsmc.state import GridSpec, ParticleState, SortCarry  # noqa: E402


def _state(key, n, grid, alive_frac=0.8):
    kx, ka = jax.random.split(key)
    x = jax.random.uniform(kx, (n, 2)) * (grid.hi - grid.lo) + grid.lo
    alive = jax.random.uniform(ka, (n,)) < alive_frac
    return ParticleState(
        x=x,
        v=jnp.zeros((n, 3)),
        species=jnp.zeros(n, jnp.int32),
        alive=alive,
        logh_acc=jnp.zeros(n),
    )


def _step(key, state, grid, move_frac, flip_frac):
    """Move a fraction of the particles to random places, kill / revive a few."""
    km, kx, kf = jax.random.split(key, 3)
    n = state.x.shape[0]
    move = jax.random.uniform(km, (n,)) < move_frac
    x_new = jax.random.uniform(kx, (n, 2)) * (grid.hi - grid.lo) + grid.lo
    flip = jax.random.uniform(kf, (n,)) < flip_frac
    return dataclasses.replace(
        state, x=jnp.where(move[:, None], x_new, state.x), alive=state.alive ^ flip
    )


@pytest.mark.parametrize("move_frac", [0.0, 0.02, 0.1, 0.2, 0.6])
def test_partition_sort_has_the_full_sorts_cells_and_members(move_frac):
    """Chained steps (moves, deaths, births): starts, counts and per-slot cells equal the
    full sort's exactly, and each cell holds the same slots; within a cell the slots that
    kept it come first in their previous order, then the newcomers by slot.  The move
    fractions cover every mover buffer and the full-sort fallback (> N / 4)."""
    grid = GridSpec(lo=jnp.zeros(2), hi=jnp.ones(2), n=(7, 5))
    n = 3000
    C = num_cells(grid)
    key = jax.random.PRNGKey(3)
    state = _state(key, n, grid)
    carry = sort.initial_carry(state, grid)
    update = jax.jit(sort.update_table)
    for k in range(4):
        prev = carry
        state = _step(jax.random.fold_in(key, k), state, grid, move_frac, 0.01)
        table, carry = update(state, grid, prev)
        ref = sort.build_table(state, grid)
        for name in ("starts", "counts", "cell"):
            np.testing.assert_array_equal(getattr(table, name), getattr(ref, name), name)
        order, ref_order = np.asarray(table.order), np.asarray(ref.order)
        assert sorted(order) == list(range(n))
        np.testing.assert_array_equal(np.asarray(carry.key), np.asarray(ref.cell)[order])
        old_cell = np.empty(n, np.int64)
        old_cell[np.asarray(prev.order)] = np.asarray(prev.key)
        old_rank = np.empty(n, np.int64)
        old_rank[np.asarray(prev.order)] = np.arange(n)
        starts, counts = np.asarray(table.starts), np.asarray(table.counts)
        for c in range(C):
            got = order[starts[c] : starts[c] + counts[c]]
            assert sorted(got) == sorted(ref_order[starts[c] : starts[c] + counts[c]])
            stayed = old_cell[got] == c
            if move_frac <= 0.2:  # the partition (the fallback is the stable sort)
                k_stay = int(stayed.sum())
                assert stayed[:k_stay].all() and not stayed[k_stay:].any()
                assert (np.diff(old_rank[got[:k_stay]]) > 0).all()
                assert (np.diff(got[k_stay:]) > 0).all()


def test_partition_sort_carry_type():
    grid = GridSpec(lo=jnp.zeros(2), hi=jnp.ones(2), n=(3, 3))
    state = _state(jax.random.PRNGKey(0), 50, grid)
    carry = sort.initial_carry(state, grid)
    assert isinstance(carry, SortCarry)
    assert carry.order.dtype == jnp.int32 and carry.key.dtype == jnp.int32


def _relaxation(seed, sort_mode):
    """Anisotropy ``(var vx - var vy) / mean var`` after 40 collisional steps of a periodic
    16-cell box started with ``T_x = 2.56 T_y`` (about 5 % of the particles change cell per
    step, so the partition sort is the one exercised)."""
    state, cs, g, sp, params, static, _t, k = test_ntc.build(
        n=(4, 4), N=4000, dt=2e-9, seed=seed, A_max=200
    )
    state = dataclasses.replace(state, v=state.v.at[:, 0].multiply(1.6))
    params = dataclasses.replace(params, fnum=jnp.full_like(params.fnum, 1e11))
    static = dataclasses.replace(static, n_steps=40, sort=sort_mode)
    (st, _c), (_m, diag) = simulate.run(state, cs, g, sp, params, static, k, emit_moments="none")
    assert int(np.asarray(diag.attempts_clipped).sum()) == 0
    v = np.asarray(st.v)
    return (v[:, 0].var() - v[:, 1].var()) / v.var(axis=0).mean()


def test_partition_sort_leaves_collisional_relaxation_unchanged_in_distribution():
    """The partition sort reorders a cell's members, so NTC draws different pairs: the
    runs differ seed by seed, the relaxation does not (12 seeds each, 3 standard
    errors)."""
    seeds = range(12)
    full = np.array([_relaxation(s, "full") for s in seeds])
    part = np.array([_relaxation(s, "partition") for s in seeds])
    assert (full != part).any()  # the pair draws really differ
    se = np.sqrt(full.var(ddof=1) / len(full) + part.var(ddof=1) / len(part))
    assert abs(full.mean() - part.mean()) < 3.0 * se, (full.mean(), part.mean(), se)
    assert 0.3 < full.mean() < 1.2  # partly relaxed from 1.56: the test is sensitive
