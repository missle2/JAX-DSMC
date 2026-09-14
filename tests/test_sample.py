"""Tests for the per-cell moment reductions (spec section 10.1)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_dsmc import sample
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.grid import cell_of, cell_volume, num_cells
from jax_dsmc.sort import build_table
from jax_dsmc.state import CellTable, GridSpec, ParticleState, SimParams, SpeciesParams

jax.config.update("jax_enable_x64", True)

MASS = 6.63e-26  # kg, argon
TEMP = 273.0  # K
NRHO = 1.0e20  # 1/m^3
VSTREAM = np.array([120.0, -40.0, 15.0])


# --- fixtures built as module-level helpers (spec 0.6: no nested defs) --------


def _grid(n=(4,), lo=0.0, hi=1.0) -> GridSpec:
    d = len(n)
    return GridSpec(
        lo=jnp.full((d,), lo, dtype=jnp.float64),
        hi=jnp.full((d,), hi, dtype=jnp.float64),
        n=tuple(n),
    )


def _species(mass=MASS) -> SpeciesParams:
    one = jnp.ones((1, 1), dtype=jnp.float64)
    return SpeciesParams(
        mass=jnp.asarray([mass], dtype=jnp.float64),
        diam=4.17e-10 * one,
        omega=one,
        tref=273.0 * one,
        alpha=one,
        mr=jnp.asarray([[mass / 2.0]], dtype=jnp.float64),
        prefactor=one,
    )


def _params(grid: GridSpec, fnum: float) -> SimParams:
    return SimParams(
        dt=jnp.asarray(1e-9, dtype=jnp.float64),
        fnum=jnp.full((num_cells(grid),), fnum, dtype=jnp.float64),
        epsilon=jnp.asarray(0.0, dtype=jnp.float64),
    )


def _state(x, v, alive=None) -> ParticleState:
    n = x.shape[0]
    if alive is None:
        alive = jnp.ones((n,), dtype=bool)
    return ParticleState(
        x=jnp.asarray(x, dtype=jnp.float64),
        v=jnp.asarray(v, dtype=jnp.float64),
        species=jnp.zeros((n,), dtype=jnp.int32),
        alive=alive,
        logh_acc=jnp.zeros((n,), dtype=jnp.float64),
    )


def _maxwellian_state(key, n, grid, temp=TEMP, vstream=VSTREAM, mass=MASS) -> ParticleState:
    """A shifted Maxwellian filling the whole domain uniformly."""
    kx, kv = jax.random.split(key)
    d = len(grid.n)
    x = jax.random.uniform(kx, (n, d), dtype=jnp.float64, minval=0.0, maxval=1.0)
    vth = math.sqrt(K_BOLTZ * temp / mass)
    v = vth * jax.random.normal(kv, (n, 3), dtype=jnp.float64) + jnp.asarray(vstream)
    return _state(x, v)


# --- 1. a Maxwellian recovers (nrho, T, vstream) ------------------------------


def test_maxwellian_moments_recovered():
    """A known shifted Maxwellian is recovered within Monte Carlo error."""
    grid = _grid((2,))
    n = 120_000
    state = _maxwellian_state(jax.random.key(0), n, grid)
    table = build_table(state, grid)
    # fnum chosen so that nrho comes out at the target
    volume = float(cell_volume(grid)[0])
    fnum = NRHO * volume / (n / num_cells(grid))
    m = sample.moments(state, table, grid, _params(grid, fnum), _species())

    vth = math.sqrt(K_BOLTZ * TEMP / MASS)
    per_cell = n / num_cells(grid)
    se_u = vth / math.sqrt(per_cell)
    se_t = TEMP * math.sqrt(2.0 / (3.0 * per_cell))

    assert np.allclose(np.asarray(m.nrho), NRHO, rtol=0.02)
    assert np.all(np.abs(np.asarray(m.u) - VSTREAM) < 5.0 * se_u)
    assert np.all(np.abs(np.asarray(m.temp) - TEMP) < 5.0 * se_t)


def test_heat_flux_vanishes_at_equilibrium():
    """``q == 0`` within MC error for an equilibrium population."""
    grid = _grid((1,))
    n = 150_000
    state = _maxwellian_state(jax.random.key(3), n, grid)
    m = sample.moments(state, build_table(state, grid), grid, _params(grid, 1.0), _species())
    vth = math.sqrt(K_BOLTZ * TEMP / MASS)
    rho = float(m.nrho[0]) * MASS
    # sd of (1/2) rho <c_a |c|^2> over n samples; <c_a^2 |c|^4> = 35 vth^6
    se = 0.5 * rho * math.sqrt(35.0) * vth**3 / math.sqrt(n)
    assert np.all(np.abs(np.asarray(m.q[0])) < 5.0 * se)


# --- 2. the trace identity, as a refactor guard only --------------------------


def test_pressure_trace_identity():
    """``trace(P)/3 == nrho k T`` to machine precision.

    Spec 10.1: this is an **identity by construction**, not a physics check —
    ``temp`` is defined from the same ``<|c|^2>`` that forms ``trace(P)``.  It
    is kept purely as a typo/refactor regression guard and must never be cited
    as validation of ``P``.  The real check is against the Chapman-Enskog
    stress at a known shear rate, which lands with M4.
    """
    grid = _grid((3,))
    state = _maxwellian_state(jax.random.key(5), 40_000, grid)
    m = sample.moments(state, build_table(state, grid), grid, _params(grid, 3.7), _species())
    lhs = np.asarray(sample.pressure_scalar(m))
    rhs = np.asarray(m.nrho * K_BOLTZ * m.temp)
    assert np.allclose(lhs, rhs, rtol=1e-13)


# --- 3. exactness on a hand-computed population -------------------------------


def test_hand_computed_single_cell():
    """Four particles in one cell, moments computed by hand in numpy."""
    grid = _grid((1,))
    v = np.array([[100.0, 0.0, 0.0], [-40.0, 20.0, 5.0], [0.0, -60.0, 30.0], [10.0, 10.0, -10.0]])
    x = np.full((4, 1), 0.5)
    state = _state(x, v)
    fnum = 2.5
    m = sample.moments(state, build_table(state, grid), grid, _params(grid, fnum), _species())

    volume = float(cell_volume(grid)[0])
    u = v.mean(axis=0)
    cpec = v - u
    c2 = np.sum(cpec**2, axis=1)
    nrho = 4 * fnum / volume
    rho = MASS * nrho

    assert float(m.count[0]) == 4.0
    assert float(m.nrho[0]) == pytest.approx(nrho, rel=1e-14)
    assert np.allclose(np.asarray(m.u[0]), u, rtol=1e-14)
    assert float(m.temp[0]) == pytest.approx(MASS * c2.mean() / (3 * K_BOLTZ), rel=1e-14)
    assert np.allclose(np.asarray(m.P[0]), rho * (cpec.T @ cpec) / 4.0, rtol=1e-13)
    assert np.allclose(
        np.asarray(m.q[0]), 0.5 * rho * (cpec * c2[:, None]).mean(axis=0), rtol=1e-13
    )


def test_two_pass_beats_one_pass_near_equilibrium():
    """The two-pass form is why ``temp`` survives a large bulk velocity.

    Spec section 10: ``<|v|^2> - |<v>|^2`` cancels catastrophically when the
    bulk speed dwarfs the thermal speed.  Here the one-pass estimate, computed
    explicitly in f64, loses most of its digits while ``moments`` does not.
    """
    grid = _grid((1,))
    n = 20_000
    vth = math.sqrt(K_BOLTZ * TEMP / MASS)
    shift = 1e8 * vth  # absurd, and that is the point
    key = jax.random.key(17)
    v = vth * jax.random.normal(key, (n, 3), dtype=jnp.float64) + shift
    state = _state(np.full((n, 1), 0.5), v)
    m = sample.moments(state, build_table(state, grid), grid, _params(grid, 1.0), _species())

    v_np = np.asarray(v)
    cpec = v_np - v_np.mean(axis=0)
    ref = MASS * np.sum(cpec**2, axis=1).mean() / (3 * K_BOLTZ)

    one_pass_e = (v_np**2).sum(axis=1).mean() - (v_np.mean(axis=0) ** 2).sum()
    one_pass_t = MASS * one_pass_e / (3 * K_BOLTZ)

    assert float(m.temp[0]) == pytest.approx(ref, rel=1e-10)
    assert abs(one_pass_t - ref) / ref > 1e-3  # one pass has lost the answer


# --- 4. empty cells, dead particles, ordering ---------------------------------


def test_empty_cells_are_zero_not_nan():
    """Every field is exactly 0 in an empty cell, and gradients stay finite."""
    grid = _grid((6,))
    # all six particles in cell 0 => cells 1..5 empty, including trailing ones
    x = np.full((6, 1), 0.05)
    v = np.arange(18, dtype=float).reshape(6, 3)
    state = _state(x, v)
    m = sample.moments(state, build_table(state, grid), grid, _params(grid, 1.0), _species())
    for field in (m.nrho, m.temp, m.count):
        arr = np.asarray(field)
        assert np.all(np.isfinite(arr))
        assert np.all(arr[1:] == 0.0)
    for field in (m.u, m.q, m.P):
        arr = np.asarray(field)
        assert np.all(np.isfinite(arr))
        assert np.all(arr[1:] == 0.0)

    g = jax.grad(_scalar_objective)(state.v, state, grid, _params(grid, 1.0), _species())
    assert np.all(np.isfinite(np.asarray(g)))


def _scalar_objective(v, state, grid, params, species):
    """``sum(temp) + sum(q^2)`` — a smooth functional of the moments, for grad."""
    st = ParticleState(
        x=state.x, v=v, species=state.species, alive=state.alive, logh_acc=state.logh_acc
    )
    m = sample.moments(st, build_table(st, grid), grid, params, species)
    return jnp.sum(m.temp) + jnp.sum(m.q**2) + jnp.sum(m.u**2)


def test_dead_particles_are_excluded():
    """``alive == False`` contributes nothing, whatever its velocity."""
    grid = _grid((2,))
    x = np.array([[0.1], [0.2], [0.3], [0.8]])
    v = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1e9, 1e9, 1e9], [7.0, 8.0, 9.0]])
    alive = jnp.asarray([True, True, False, True])
    state = _state(x, v, alive)
    m = sample.moments(state, build_table(state, grid), grid, _params(grid, 1.0), _species())

    live = _state(x[[0, 1, 3]], v[[0, 1, 3]])
    m_live = sample.moments(live, build_table(live, grid), grid, _params(grid, 1.0), _species())
    assert np.allclose(np.asarray(m.u), np.asarray(m_live.u), rtol=1e-14)
    assert np.allclose(np.asarray(m.temp), np.asarray(m_live.temp), rtol=1e-14)
    assert float(m.count[0]) == 2.0


def test_permutation_invariance():
    """Moments do not depend on the order particles are stored in."""
    grid = _grid((3,))
    state = _maxwellian_state(jax.random.key(9), 5000, grid)
    m = sample.moments(state, build_table(state, grid), grid, _params(grid, 1.0), _species())

    perm = np.asarray(jax.random.permutation(jax.random.key(21), 5000))
    shuffled = _state(np.asarray(state.x)[perm], np.asarray(state.v)[perm])
    m2 = sample.moments(shuffled, build_table(shuffled, grid), grid, _params(grid, 1.0), _species())
    assert np.allclose(np.asarray(m.u), np.asarray(m2.u), rtol=1e-12, atol=1e-12)
    assert np.allclose(np.asarray(m.temp), np.asarray(m2.temp), rtol=1e-12)
    assert np.allclose(np.asarray(m.P), np.asarray(m2.P), rtol=1e-11, atol=1e-11)


# --- 5. fnum is per-cell, not a scalar ----------------------------------------


def test_fnum_is_per_cell():
    """``nrho`` scales cell-by-cell with ``fnum[c]`` (spec 3.3)."""
    grid = _grid((3,))
    x = np.array([[0.1], [0.5], [0.9]])
    v = np.zeros((3, 3))
    state = _state(x, v)
    table = build_table(state, grid)
    fnum = jnp.asarray([1.0, 10.0, 100.0], dtype=jnp.float64)
    params = SimParams(dt=jnp.asarray(1e-9), fnum=fnum, epsilon=jnp.asarray(0.0, dtype=jnp.float64))
    m = sample.moments(state, table, grid, params, _species())
    volume = np.asarray(cell_volume(grid))
    assert np.allclose(np.asarray(m.nrho), np.asarray(fnum) / volume, rtol=1e-14)


# --- 6. shapes, dtypes, static segment count, jit ------------------------------


@pytest.mark.parametrize("n_cells", [(4,), (2, 3), (2, 2, 2)])
def test_shapes_and_dtypes(n_cells):
    """``C = prod(n)`` leading dimension everywhere, f64 throughout, any ``D``."""
    grid = _grid(n_cells)
    C = num_cells(grid)
    d = len(n_cells)
    key = jax.random.key(1)
    x = jax.random.uniform(key, (500, d), dtype=jnp.float64)
    v = jax.random.normal(jax.random.key(2), (500, 3), dtype=jnp.float64)
    state = _state(x, v)
    m = sample.moments(state, build_table(state, grid), grid, _params(grid, 1.0), _species())
    assert m.nrho.shape == (C,)
    assert m.u.shape == (C, 3)
    assert m.temp.shape == (C,)
    assert m.P.shape == (C, 3, 3)
    assert m.q.shape == (C, 3)
    for field in (m.nrho, m.u, m.temp, m.P, m.q, m.count):
        assert field.dtype == jnp.float64


def test_moments_is_jittable():
    """``num_segments`` is static, so the whole thing must compile.

    This is a real requirement, not a nicety: ``moments`` is called from inside
    the jitted step whenever an objective functional is built on it (section 11),
    so a host-only ``moments`` would make the adjoint unreachable.
    """
    grid = _grid((4,))
    state = _maxwellian_state(jax.random.key(31), 2000, grid)
    table = build_table(state, grid)
    params = _params(grid, 1.0)
    species = _species()
    fn = jax.jit(sample.moments, static_argnums=())
    m1 = sample.moments(state, table, grid, params, species)
    m2 = fn(state, table, grid, params, species)
    assert np.allclose(np.asarray(m1.temp), np.asarray(m2.temp), rtol=1e-14)


def test_cell_index_agrees_with_the_table():
    """``moments`` and ``sort.build_table`` place live particles identically.

    ``sample`` recomputes ``cell_of`` rather than reading ``table.cell`` (the
    mesh interface is the only place cell arithmetic may live, section 4).
    This pins the two together so a future divergence is caught here.
    """
    grid = _grid((5,))
    state = _maxwellian_state(jax.random.key(41), 3000, grid)
    table: CellTable = build_table(state, grid)
    assert np.array_equal(np.asarray(cell_of(state.x, grid)), np.asarray(table.cell))


# --- 7. time averaging ---------------------------------------------------------


def test_accumulator_averages_sums_once():
    """``finalize(accumulate(...))`` is the arithmetic mean of the samples."""
    grid = _grid((2,))
    acc = sample.zero_accumulator(grid, _species().mass[0])
    states = []
    for seed in range(5):
        state = _maxwellian_state(jax.random.key(seed), 4000, grid)
        states.append(state)
        m = sample.moments(state, build_table(state, grid), grid, _params(grid, 1.0), _species())
        acc = sample.accumulate(acc, m)
    pooled = jax.tree.map(lambda *a: jnp.concatenate(a), *states)
    expected = sample.moments(pooled, build_table(pooled, grid), grid, _params(grid, 1.0), _species())
    avg = sample.finalize(acc)
    np.testing.assert_allclose(avg.temp, expected.temp, rtol=1e-12)
    np.testing.assert_allclose(avg.u, expected.u, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(avg.P, expected.P/5, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(avg.q, expected.q/5, rtol=1e-11, atol=1e-12)


def test_empty_accumulator_finalizes_to_zero():
    grid = _grid((2,))
    avg = sample.finalize(sample.zero_accumulator(grid, _species().mass[0]))
    assert np.all(np.asarray(avg.temp) == 0.0)
    assert np.all(np.isfinite(np.asarray(avg.P)))


# --- 8. differentiability ------------------------------------------------------


def test_gradient_wrt_velocity_matches_finite_difference():
    """``d(sum temp)/dv`` is exact: temp is a smooth quadratic in ``v``."""
    grid = _grid((2,))
    state = _maxwellian_state(jax.random.key(13), 400, grid)
    params = _params(grid, 1.0)
    species = _species()
    g = np.asarray(jax.grad(_scalar_objective)(state.v, state, grid, params, species))

    h = 1e-3 * math.sqrt(K_BOLTZ * TEMP / MASS)
    v = np.asarray(state.v)
    for i, comp in ((0, 0), (7, 2), (123, 1)):
        vp, vm = v.copy(), v.copy()
        vp[i, comp] += h
        vm[i, comp] -= h
        jp = float(_scalar_objective(jnp.asarray(vp), state, grid, params, species))
        jm = float(_scalar_objective(jnp.asarray(vm), state, grid, params, species))
        assert g[i, comp] == pytest.approx((jp - jm) / (2 * h), rel=1e-6)


def test_gradient_wrt_mesh_geometry_is_nonzero():
    """``nrho`` depends on ``cell_volume``, so ``lo``/``hi`` carry a gradient."""
    grid = _grid((2,))
    state = _maxwellian_state(jax.random.key(23), 2000, grid)
    table = build_table(state, grid)
    params = _params(grid, 1.0)
    species = _species()
    g = jax.grad(_nrho_objective, argnums=0)(
        grid.hi, grid.lo, grid.n, state, table, params, species
    )
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.any(np.asarray(g) != 0.0)


def _nrho_objective(hi, lo, n, state, table, params, species):
    """``sum(nrho^2)`` as a function of the mesh corners, for grad."""
    grid = GridSpec(lo=lo, hi=hi, n=n)
    m = sample.moments(state, table, grid, params, species)
    return jnp.sum(m.nrho**2)


def test_cell_temperature_agrees_with_full_moments():
    """The `moments` -> `cell_temperature` optimisation must not change the answer.

    `cell_temperature` skips the pressure tensor and heat flux and reuses
    `table.cell` instead of recomputing `cell_of` (2.9-3.5x and a further
    1.6-3.0x, section 8.6).  Both shortcuts are only legitimate while the
    temperature it returns is bit-comparable to the full path's.
    """
    for n in ((4,), (2, 2), (2, 2, 2)):
        grid = _grid(n)
        state = _maxwellian_state(jax.random.key(77), 4000, grid)
        table = build_table(state, grid)
        params = _params(grid, 1.0)
        species = _species()
        full = sample.moments(state, table, grid, params, species).temp
        fast = sample.cell_temperature(state, table, grid, species)
        assert np.allclose(np.asarray(full), np.asarray(fast), rtol=1e-12, atol=0.0)


def test_cell_temperature_is_jittable_and_differentiable():
    grid = _grid((4,))
    state = _maxwellian_state(jax.random.key(78), 2000, grid)
    table = build_table(state, grid)
    species = _species()
    t = jax.jit(sample.cell_temperature)(state, table, grid, species)
    assert t.shape == (4,)
    g = jax.grad(
        lambda v: jnp.sum(
            sample.cell_temperature(
                ParticleState(
                    x=state.x,
                    v=v,
                    species=state.species,
                    alive=state.alive,
                    logh_acc=state.logh_acc,
                ),
                table,
                grid,
                species,
            )
        )
    )(state.v)
    assert bool(jnp.all(jnp.isfinite(g)))
