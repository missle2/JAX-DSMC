"""Tests for ``jax_dsmc.state`` — spec section 3.

What is actually being pinned down here:

- every container round-trips through ``tree_flatten``/``tree_unflatten``;
- **traced fields are leaves and static fields are not** (section 0.4) — the
  recurring mistake is putting ``n`` in a traced field or ``dt`` in a static one;
- :class:`SimStatic` is hashable and works as ``static_argnums`` **without
  retracing** on repeated identical calls, and *does* retrace when a static
  value changes (that is the point of it being static);
- the shape/dtype contract of section 3.1, including the two things that are
  easy to get wrong: ``fnum`` is ``f64[C]`` per-cell, and ``CollideState`` has
  no ``remain`` field (section 8.1);
- ``CellTable`` carries ``cell`` (section 6).

No nested function definitions (section 0.6).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_dsmc.state import (
    CellTable,
    CollideState,
    Diagnostics,
    FlowParams,
    GridSpec,
    ParticleState,
    SimParams,
    SimStatic,
    SpeciesParams,
    check_fnum_uniform,
    num_cells_of,
    static_fields,
    traced_fields,
)

jax.config.update("jax_enable_x64", True)

N = 6
D = 2
S = 1
G = 1
GRID_N = (2, 3)
C = 6


# --- builders (module level, section 0.6) ---------------------------------


def _particle_state():
    """A ``ParticleState`` with exactly the declared shapes and dtypes."""
    return ParticleState(
        x=jnp.zeros((N, D), dtype=jnp.float64),
        v=jnp.ones((N, 3), dtype=jnp.float64),
        species=jnp.zeros(N, dtype=jnp.int32),
        alive=jnp.ones(N, dtype=bool),
        logh_acc=jnp.zeros(N, dtype=jnp.float64),
    )


def _grid():
    return GridSpec(
        lo=jnp.zeros(D, dtype=jnp.float64),
        hi=jnp.ones(D, dtype=jnp.float64),
        n=GRID_N,
    )


def _collide_state():
    return CollideState(vremax=jnp.full((C, G, G), 1e-16, dtype=jnp.float64))


def _cell_table():
    return CellTable(
        order=jnp.arange(N, dtype=jnp.int32),
        starts=jnp.zeros(C, dtype=jnp.int32),
        counts=jnp.zeros(C, dtype=jnp.int32),
        cell=jnp.zeros(N, dtype=jnp.int32),
    )


def _diagnostics():
    return Diagnostics(
        attempts_clipped=jnp.asarray(0, dtype=jnp.int32),
        max_attempts=jnp.asarray(0, dtype=jnp.int32),
        majorant_violations=jnp.asarray(0, dtype=jnp.int32),
        min_majorant_margin=jnp.asarray(np.inf, dtype=jnp.float64),
        decision_checksum=jnp.asarray(0.0, dtype=jnp.float64),
        wall_recrossed=jnp.asarray(0, dtype=jnp.int32),
        inflow_injected=jnp.asarray(0, dtype=jnp.int32),
        inflow_shortfall=jnp.asarray(0.0, dtype=jnp.float64),
    )


def _species():
    ones = jnp.ones((S, S), dtype=jnp.float64)
    return SpeciesParams(
        mass=jnp.full((S,), 4.65e-26, dtype=jnp.float64),
        diam=4.17e-10 * ones,
        omega=0.74 * ones,
        tref=273.0 * ones,
        alpha=1.0 * ones,
        mr=2.325e-26 * ones,
        prefactor=1e-15 * ones,
    )


def _flow():
    return FlowParams(
        nrho=jnp.asarray(1e20, dtype=jnp.float64),
        temp=jnp.asarray(300.0, dtype=jnp.float64),
        vstream=jnp.zeros(3, dtype=jnp.float64),
    )


def _params():
    return SimParams(
        dt=jnp.asarray(1e-6, dtype=jnp.float64),
        fnum=jnp.full((C,), 1e12, dtype=jnp.float64),
        epsilon=jnp.asarray(0.0, dtype=jnp.float64),
    )


def _static(**overrides):
    kwargs = {"N": N, "D": D, "S": S, "G": G, "A_max": 8, "K_bounce": 4, "n_steps": 3}
    kwargs.update(overrides)
    return SimStatic(**kwargs)


ALL_CONTAINERS = [
    _particle_state,
    _grid,
    _collide_state,
    _cell_table,
    _diagnostics,
    _species,
    _flow,
    _params,
]


# --- differentiated / jitted bodies (module level, section 0.6) -----------

_TRACES: list[str] = []


def _objective(params, grid, state):
    """A scalar touching one traced leaf of each of three containers."""
    return params.dt * jnp.sum(params.fnum) * jnp.sum(grid.hi - grid.lo) * jnp.sum(state.v)


def _counted_body(params, static):
    """Records one entry per *trace*, so retracing is observable."""
    _TRACES.append(f"{static.N}-{static.n_steps}")
    return params.dt * static.n_steps + jnp.sum(params.fnum)


def _grid_body(grid):
    """Uses the STATIC ``n`` in Python control flow — legal only because it is static."""
    return jnp.sum(grid.hi - grid.lo) * float(num_cells_of(grid.n))


def _identity(obj):
    """A jittable identity, to check a container survives a round trip through XLA."""
    return obj


def _double_floats(a):
    """Double a leaf unless it is boolean (``bool * 2`` is not meaningful here)."""
    return a if a.dtype == bool else a * 2


def _builder_id(build):
    """pytest id for a builder function."""
    return build.__name__


# --- pytree round trips ---------------------------------------------------


@pytest.mark.parametrize("build", ALL_CONTAINERS, ids=_builder_id)
def test_container_round_trips_through_flatten_unflatten(build):
    obj = build()
    leaves, treedef = jax.tree_util.tree_flatten(obj)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is type(obj)
    for name in traced_fields(type(obj)):
        assert np.array_equal(np.asarray(getattr(rebuilt, name)), np.asarray(getattr(obj, name)))
    for name in static_fields(type(obj)):
        assert getattr(rebuilt, name) == getattr(obj, name)


@pytest.mark.parametrize("build", ALL_CONTAINERS, ids=_builder_id)
def test_traced_fields_are_leaves_and_static_fields_are_not(build):
    obj = build()
    leaves = jax.tree_util.tree_leaves(obj)
    assert len(leaves) == len(traced_fields(type(obj)))
    static_values = [getattr(obj, name) for name in static_fields(type(obj))]
    for value in static_values:
        assert not any(value is leaf for leaf in leaves)


@pytest.mark.parametrize("build", ALL_CONTAINERS, ids=_builder_id)
def test_container_survives_tree_map_and_jit(build):
    obj = build()
    doubled = jax.tree_util.tree_map(_double_floats, obj)
    assert type(doubled) is type(obj)
    identity = jax.jit(_identity)(obj)
    assert type(identity) is type(obj)
    for name in traced_fields(type(obj)):
        assert getattr(identity, name).dtype == getattr(obj, name).dtype
        assert getattr(identity, name).shape == getattr(obj, name).shape


# --- the shape and dtype contract (section 3.1) ---------------------------


def test_particle_state_shapes_and_dtypes():
    s = _particle_state()
    assert s.x.shape == (N, D) and s.x.dtype == jnp.float64
    # velocities are ALWAYS 3-component, whatever D is (section 0.2)
    assert s.v.shape == (N, 3) and s.v.dtype == jnp.float64
    assert s.species.shape == (N,) and s.species.dtype == jnp.int32
    assert s.alive.shape == (N,) and s.alive.dtype == bool
    assert s.logh_acc.shape == (N,) and s.logh_acc.dtype == jnp.float64


def test_collide_state_has_no_remain_field():
    """Section 8.1: stochastic rounding replaces the deterministic carry.

    A ``remain`` field would mean a degenerate ``P(n | f, remain)``, no score
    term, and therefore an exactly zero gradient to ``dt``, ``fnum`` and the cell
    volume — the most serious defect in revision 2.
    """
    assert traced_fields(CollideState) == ("vremax", "remainder")
    assert not hasattr(_collide_state(), "remain")
    assert _collide_state().vremax.shape == (C, G, G)


def test_cell_table_carries_the_per_particle_cell_index():
    """Section 6: the count score of 8.1 is per-cell but applied per-particle."""
    assert traced_fields(CellTable) == ("order", "starts", "counts", "cell")
    t = _cell_table()
    assert t.cell.shape == (N,) and t.cell.dtype == jnp.int32
    assert t.order.shape == (N,)
    assert t.starts.shape == (C,) and t.counts.shape == (C,)


def test_sim_params_fnum_is_per_cell_not_scalar():
    """Section 3.3: ``fnum`` is ``f64[C]``.  A scalar here is the classic error."""
    p = _params()
    assert p.fnum.shape == (C,)
    assert p.fnum.ndim == 1
    assert p.fnum.dtype == jnp.float64
    assert p.dt.shape == () and p.epsilon.shape == ()


def test_grid_spec_splits_traced_geometry_from_static_resolution():
    g = _grid()
    assert traced_fields(GridSpec) == ("lo", "hi", "transverse", "volume")
    assert static_fields(GridSpec) == ("n",)
    leaves, treedef = jax.tree_util.tree_flatten(g)
    # lo, hi and transverse are traced geometry; `n` is static and stays in the
    # treedef.  `transverse` carries the extent of the 3 - D suppressed directions,
    # so `cell_volume` is a true volume at every D -- see GridSpec.
    assert len(leaves) == 4  # lo, hi, transverse, volume (empty unless cut cells)
    assert all(hasattr(x, "dtype") for x in leaves), "every traced leaf must be an array"
    assert GRID_N in treedef.node_data()[1]  # `n` lives in the treedef, not the leaves
    assert num_cells_of(g.n) == C


def test_species_and_flow_shapes():
    sp = _species()
    assert sp.mass.shape == (S,)
    for name in ("diam", "omega", "tref", "alpha", "mr", "prefactor"):
        assert getattr(sp, name).shape == (S, S)
        assert getattr(sp, name).dtype == jnp.float64
    fl = _flow()
    assert fl.nrho.shape == () and fl.temp.shape == ()
    assert fl.vstream.shape == (3,)  # 3 components, whatever D is


def test_diagnostics_counters_are_integers_and_the_margin_is_f64():
    d = _diagnostics()
    for name in ("attempts_clipped", "max_attempts", "majorant_violations"):
        assert getattr(d, name).dtype == jnp.int32
    assert d.min_majorant_margin.dtype == jnp.float64


# --- gradients flow to the traced leaves ----------------------------------


def test_gradient_structure_matches_the_traced_fields():
    g_params, g_grid, g_state = jax.grad(_objective, argnums=(0, 1, 2), allow_int=True)(
        _params(), _grid(), _particle_state()
    )
    assert type(g_params) is SimParams
    assert g_params.dt.shape == () and g_params.fnum.shape == (C,)
    assert float(g_params.dt) != 0.0
    assert np.all(np.asarray(g_params.fnum) != 0.0)  # per-cell sensitivity exists
    # the STATIC `n` is untouched by grad; the TRACED corners carry cotangents
    assert type(g_grid) is GridSpec
    assert g_grid.n == GRID_N
    assert np.any(np.asarray(g_grid.lo) != 0.0)
    # integer and boolean leaves carry no tangent (frozen decisions, section 0.5)
    assert g_state.v.shape == (N, 3)
    assert g_state.species.dtype == jax.dtypes.float0
    assert g_state.alive.dtype == jax.dtypes.float0


def test_f64_is_preserved_through_jit():
    out = jax.jit(_objective)(_params(), _grid(), _particle_state())
    assert out.dtype == jnp.float64


# --- SimStatic: hashable, static_argnums, no spurious retracing -----------


def test_sim_static_is_frozen_hashable_and_compared_by_value():
    a, b = _static(), _static()
    assert a == b
    assert hash(a) == hash(b)
    assert a != _static(n_steps=4)
    with pytest.raises(Exception):  # noqa: B017 -- frozen dataclass, any raise is correct
        a.N = 99


def test_sim_static_is_a_leaf_not_a_pytree_node():
    """It must never be flattened: every field is a shape or a trip count."""
    leaves = jax.tree_util.tree_leaves(_static())
    assert leaves == [_static()]


def test_jit_accepts_sim_static_as_static_argnums_without_retracing():
    jax.clear_caches()  # JAX caches traces by function identity, not by wrapper
    _TRACES.clear()
    counted = jax.jit(_counted_body, static_argnums=(1,))
    first = counted(_params(), _static())
    assert len(_TRACES) == 1
    for _ in range(4):
        again = counted(_params(), _static())  # equal by value, a fresh object each time
        assert float(again) == float(first)
    assert len(_TRACES) == 1, f"retraced on an identical static container: {_TRACES}"


def test_changing_a_static_field_forces_a_retrace():
    jax.clear_caches()
    _TRACES.clear()
    counted = jax.jit(_counted_body, static_argnums=(1,))
    counted(_params(), _static())
    counted(_params(), _static(n_steps=4))
    assert len(_TRACES) == 2, f"a changed static field did not recompile: {_TRACES}"


def test_static_n_may_be_used_in_python_control_flow_inside_jit():
    out = jax.jit(_grid_body)(_grid())
    assert float(out) == pytest.approx(2.0 * C)


def test_sim_static_rejects_shapes_that_would_be_silently_wrong():
    with pytest.raises(ValueError, match="D must be"):
        _static(D=4)
    with pytest.raises(ValueError, match="G <= S"):
        _static(S=1, G=2)
    with pytest.raises(ValueError, match="positive Python int"):
        _static(N=0)
    with pytest.raises(ValueError, match="positive Python int"):
        _static(A_max=8.0)


# --- the fnum loaded gun (section 3.3, gap U7) ----------------------------


def test_check_fnum_uniform_accepts_uniform_and_single_cell():
    check_fnum_uniform(jnp.full((C,), 1e12))
    check_fnum_uniform(jnp.asarray([3.0]))  # C == 1: non-uniformity is meaningless
    check_fnum_uniform(jnp.asarray([1.0, 2.0]), has_reweighting=True)


def test_check_fnum_uniform_rejects_a_non_uniform_multi_cell_field():
    with pytest.raises(ValueError, match="non-uniform fnum"):
        check_fnum_uniform(jnp.asarray([1e12, 2e12, 1e12]))


# --- the containers compose (the shapes the rest of M1 will see) ----------


def test_containers_nest_into_a_single_pytree():
    bundle = (_particle_state(), _collide_state(), _cell_table(), _params(), _grid())
    leaves, treedef = jax.tree_util.tree_flatten(bundle)
    assert len(leaves) == 5 + 2 + 4 + 4 + 3  # GridSpec carries lo, hi, transverse, volume
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt[4].n == GRID_N
    doubled = jax.jit(_identity)(bundle)
    assert type(doubled[0]) is ParticleState
