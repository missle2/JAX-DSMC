import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc import geometry, inflow
from jax_dsmc.boundary import BC, Faces
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.state import GridSpec, ParticleState
from jax_dsmc.wall_adjoint import advance


def test_first_hit_prefers_earliest_surface_in_two_dimensions():
    grid = GridSpec(jnp.zeros(2), jnp.ones(2), (2, 2))
    time, surface = geometry.first_hit(jnp.array([[0.2, 0.8]]), jnp.array([[1.0, 2.0, 0.0]]), grid)
    np.testing.assert_allclose(time, [0.1])
    np.testing.assert_array_equal(surface, [3])


def test_two_dimensional_specular_then_absorption():
    grid = GridSpec(jnp.zeros(2), jnp.ones(2), (2, 2))
    faces = Faces((BC.SPECULAR, BC.OUTFLOW, BC.SPECULAR, BC.SPECULAR))
    out = advance(
        jnp.array([[0.2, 0.5]]),
        jnp.array([[-1.0, 0.1, 0.0]]),
        jnp.array([True]),
        grid,
        faces,
        None,
        K_BOLTZ,
        1.5,
        jax.random.key(1),
        max_bounces=4,
    )
    assert not bool(out[2][0])
    assert int(out[3]) == 0
    np.testing.assert_allclose(out[0][0], [1.0, 0.62])


def test_newborn_opposite_face_escape_is_not_storage_shortfall():
    grid = GridSpec(jnp.zeros(1), jnp.ones(1), (2,))
    n = 128
    state = ParticleState(
        jnp.zeros((n, 1)),
        jnp.zeros((n, 3)),
        jnp.zeros(n, jnp.int32),
        jnp.zeros(n, bool),
        jnp.zeros(n),
    )
    res = inflow.InflowParams.uniform(1, 0.1, 1.0, [10.0, 0.0, 0.0])
    faces = Faces((BC.INFLOW, BC.OUTFLOW))
    out, inserted, shortfall = inflow.inject(
        state, grid, faces, res, K_BOLTZ, 1.0, 10.0, jax.random.key(3), n, track=True
    )
    assert float(shortfall) == 0
    assert int(inserted) < 10
    assert bool(jnp.all((out.x[out.alive] >= 0) & (out.x[out.alive] <= 1)))


def test_planar_shape_reference_includes_hit_probability_and_area():
    v = jnp.array([[-0.2, 0.0, 0.0], [-0.4, 0.0, 0.0]])
    normal = jnp.array([1.0, 0.0, 0.0])

    def objective(gap, area):
        return geometry.averaged_planar_impulse(v, normal, gap, 0.5, 2.0, area)[:, 0].mean()

    gap, area = 1.3, 0.7
    # All particles lie in the unsaturated-hit branch: force=-2m E[vn²]/(gap area).
    expected = -4 * 0.1 / (gap * area)
    np.testing.assert_allclose(objective(gap, area), expected)
    dg, da = jax.grad(objective, (0, 1))(gap, area)
    np.testing.assert_allclose([dg, da], [-expected / gap, -expected / area])


def test_specular_surface_impulse_closes_momentum_balance():
    grid = GridSpec(jnp.zeros(1), jnp.ones(1), (1,))
    x, v = jnp.array([[0.2]]), jnp.array([[-2.0, 1.0, 0.0]])
    out = advance(
        x,
        v,
        jnp.array([True]),
        grid,
        Faces.uniform(BC.SPECULAR, 1),
        None,
        3.0,
        0.2,
        jax.random.key(0),
        return_tallies=True,
    )
    tally = out[-1]
    np.testing.assert_allclose(tally.momentum.sum(axis=0), 3 * (v - out[1]).sum(axis=0))
    np.testing.assert_allclose(tally.energy, 0.0)
    np.testing.assert_array_equal(tally.hits, [1, 0])
