"""Initial condition assembly (spec sections 5 and 12)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

jax.config.update("jax_enable_x64", True)

from jax_dsmc.collide import make_prefactor  # noqa: E402
from jax_dsmc.constants import K_BOLTZ  # noqa: E402
from jax_dsmc.grid import cell_of, num_cells  # noqa: E402
from jax_dsmc.initialize import (  # noqa: E402
    initial_collide_state,
    initial_state,
    suggested_a_max,
    target_particle_count,
)
from jax_dsmc.state import FlowParams, GridSpec, SimStatic, SpeciesParams  # noqa: E402

AR_MASS, AR_DIAM, AR_TREF = 6.63e-26, 4.11e-10, 273.15


def species(omega=0.81):
    mass = jnp.array([AR_MASS])
    mr = jnp.array([[AR_MASS / 2]])
    diam = jnp.array([[AR_DIAM]])
    om = jnp.array([[omega]])
    tref = jnp.array([[AR_TREF]])
    al = jnp.array([[1.4]])
    return SpeciesParams(
        mass=mass,
        diam=diam,
        omega=om,
        tref=tref,
        alpha=al,
        mr=mr,
        prefactor=make_prefactor(diam, om, tref, mr),
    )


def setup(n=(4,), N=20000, T=300.0):
    g = GridSpec(lo=jnp.zeros(len(n)), hi=jnp.full(len(n), 1e-4), n=n)
    flow = FlowParams(nrho=jnp.float64(1e20), temp=jnp.float64(T), vstream=jnp.zeros(3))
    st = SimStatic(N=N, D=len(n), S=1, G=1, A_max=100, K_bounce=3, n_steps=1)
    return g, flow, st, species()


@pytest.mark.parametrize("n", [(5,), (3, 4), (2, 3, 4)])
def test_shapes_and_dtypes_across_dimensions(n):
    g, flow, st, sp = setup(n=n, N=1000)
    s = initial_state(jax.random.PRNGKey(0), flow, g, sp, st)
    assert s.x.shape == (st.N, len(n))
    assert s.v.shape == (st.N, 3), "velocities are always 3-component (section 0.2)"
    assert s.x.dtype == jnp.float64 and s.v.dtype == jnp.float64
    assert s.species.shape == (st.N,) and s.species.dtype == jnp.int32
    assert s.alive.shape == (st.N,) and s.alive.dtype == jnp.bool_
    assert s.logh_acc.shape == (st.N,)
    assert float(jnp.max(jnp.abs(s.logh_acc))) == 0.0


def test_positions_fill_the_domain_uniformly():
    g, flow, st, sp = setup(n=(8,), N=40000)
    s = initial_state(jax.random.PRNGKey(1), flow, g, sp, st)
    assert bool(jnp.all(s.x >= g.lo)) and bool(jnp.all(s.x <= g.hi))
    u = np.asarray((s.x[:, 0] - g.lo[0]) / (g.hi[0] - g.lo[0]))
    assert stats.kstest(u, "uniform").pvalue > 1e-3
    # and every cell gets a share
    counts = np.bincount(np.asarray(cell_of(s.x, g)), minlength=num_cells(g))
    assert stats.chisquare(counts).pvalue > 1e-3


def test_velocities_are_maxwellian_with_the_requested_moments():
    g, flow, st, sp = setup(N=200000, T=450.0)
    vstream = jnp.array([120.0, -30.0, 5.0])
    flow = FlowParams(nrho=flow.nrho, temp=flow.temp, vstream=vstream)
    s = initial_state(jax.random.PRNGKey(2), flow, g, sp, st)
    want_var = K_BOLTZ * 450.0 / AR_MASS
    for d in range(3):
        col = np.asarray(s.v[:, d])
        assert col.mean() == pytest.approx(float(vstream[d]), abs=5 * np.sqrt(want_var / col.size))
        assert col.var() == pytest.approx(want_var, rel=0.02)
        z = (col - float(vstream[d])) / np.sqrt(want_var)
        assert stats.kstest(z, "norm").pvalue > 1e-4


def test_dead_slots_are_inert_and_finite():
    """Dead slots must be valid data, never NaN: they are gathered before masking."""
    g, flow, st, sp = setup(n=(2,), N=500)
    s = initial_state(jax.random.PRNGKey(3), flow, g, sp, st, n_alive=200)
    assert int(jnp.sum(s.alive)) == 200
    assert bool(jnp.all(jnp.isfinite(s.x))) and bool(jnp.all(jnp.isfinite(s.v)))
    assert float(jnp.max(jnp.abs(s.v[~s.alive]))) == 0.0


def test_n_alive_is_validated():
    g, flow, st, sp = setup(n=(2,), N=100)
    with pytest.raises(ValueError, match="n_alive"):
        initial_state(jax.random.PRNGKey(0), flow, g, sp, st, n_alive=101)


def test_gradients_reach_temp_and_vstream_pathwise():
    """Reparameterised, so these are exact -- not score-based (section 5)."""
    g, flow, st, sp = setup(n=(2,), N=4000)

    def mean_energy(temp, vstream):
        f = FlowParams(nrho=flow.nrho, temp=temp, vstream=vstream)
        s = initial_state(jax.random.PRNGKey(4), f, g, sp, st)
        return jnp.mean(jnp.sum(s.v**2, axis=-1))

    gt, gv = jax.grad(mean_energy, argnums=(0, 1))(flow.temp, flow.vstream)
    assert bool(jnp.isfinite(gt)) and float(gt) != 0.0
    assert bool(jnp.all(jnp.isfinite(gv)))
    # d<|v|^2>/dT = 3k/m for a zero-mean Maxwellian
    assert float(gt) == pytest.approx(3 * K_BOLTZ / AR_MASS, rel=0.05)

    # common random numbers: the FD check is only meaningful with a shared stream
    h = 1e-3
    fd = (mean_energy(flow.temp + h, flow.vstream) - mean_energy(flow.temp - h, flow.vstream)) / (
        2 * h
    )
    assert float(gt) == pytest.approx(float(fd), rel=1e-6)


def test_initialisation_is_reproducible_and_key_dependent():
    g, flow, st, sp = setup(n=(2,), N=1000)
    a = initial_state(jax.random.PRNGKey(5), flow, g, sp, st)
    b = initial_state(jax.random.PRNGKey(5), flow, g, sp, st)
    c = initial_state(jax.random.PRNGKey(6), flow, g, sp, st)
    assert bool(jnp.array_equal(a.v, b.v))
    assert not bool(jnp.array_equal(a.v, c.v))


def test_positions_and_velocities_use_different_streams():
    """A shared stream would correlate position with velocity -- a real IC artefact."""
    g, flow, st, sp = setup(n=(1,), N=50000)
    s = initial_state(jax.random.PRNGKey(7), flow, g, sp, st)
    r = np.corrcoef(np.asarray(s.x[:, 0]), np.asarray(s.v[:, 0]))[0, 1]
    assert abs(r) < 5.0 / np.sqrt(st.N)


def test_collide_state_is_seeded_from_the_thermal_majorant():
    g, flow, st, sp = setup(n=(3,))
    cs = initial_collide_state(flow, g, sp, st)
    assert cs.vremax.shape == (num_cells(g), st.G, st.G)
    assert bool(jnp.all(cs.vremax > 0))
    assert bool(jnp.all(cs.vremax == cs.vremax[0, 0, 0]))  # uniform IC => uniform seed


def test_target_particle_count_and_a_max_are_sane():
    g, flow, st, sp = setup(n=(4,))
    fnum = jnp.full((num_cells(g),), 1e12)
    n_target = float(target_particle_count(flow, g, fnum))
    assert n_target == pytest.approx(1e20 * float(jnp.prod(g.hi - g.lo)) / 1e12, rel=1e-12)
    a = suggested_a_max(flow, g, sp, st, jnp.float64(1e-7), fnum)
    assert isinstance(a, int) and a >= 1


def test_initial_state_is_jittable():
    g, flow, st, sp = setup(n=(2, 2), N=800)
    f = jax.jit(initial_state, static_argnames=("static", "n_alive"))
    s = f(jax.random.PRNGKey(8), flow, g, sp, st)
    assert s.v.shape == (800, 3)
