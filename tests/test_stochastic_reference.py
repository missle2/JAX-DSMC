"""Independent expectation tests; no hand-coded copy of an adjoint recursion."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ndtr

from jax_dsmc import adjoint, boundary
from jax_dsmc.collide.vss import collision_rate, scatter, species_params
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.state import GridSpec


def test_same_realization_baseline_is_rejected():
    # For phi=B~Bernoulli(p), centering by phi itself deletes the entire gradient.
    with pytest.raises(ValueError, match="independent"):
        adjoint.surrogate(jnp.ones(1), jnp.zeros(1), jnp.ones(1, bool), baseline="mean")


def _history(theta, bits):
    """Four particles, two disjoint matchings, all 16 acceptance histories."""
    sp = species_params(jnp.array([K_BOLTZ]), jnp.array([[.1]]),
                        jnp.array([[.5]]), jnp.ones((1, 1)), jnp.array([[1.4]]))
    v = jnp.array([[1.2, .3, -.2], [-.8, .1, .7], [.2, 1., .4], [-.3, -.7, -.5]])
    v = v.at[0, 0].set(theta)
    local = jnp.zeros(4)
    total = jnp.float64(0.)
    for k, (i, j) in enumerate(((0, 1), (2, 3), (0, 2), (1, 3))):
        q = collision_rate(v[i], v[j], 0, 0, sp)
        p = q / .3
        lh = jnp.log(jnp.where(bits[k], p, 1-p))
        a, b = scatter(v[i], v[j], 0, 0, sp, .17 + .13*k, .21 + .11*k)
        v = v.at[i].set(jnp.where(bits[k], a, v[i]))
        v = v.at[j].set(jnp.where(bits[k], b, v[j]))
        local = local.at[i].add(lh).at[j].add(lh)
        total = total + lh
    phi = v[:, 1]**4
    return jnp.mean(phi), local, total, phi


def test_global_likelihood_matches_exact_enumeration_after_changing_partners():
    histories = jnp.array(list(itertools.product((False, True), repeat=4)))
    def exact(theta):
        value, _, lp, _ = jax.vmap(_history, in_axes=(None, 0))(theta, histories)
        return jnp.sum(jnp.exp(lp)*value)
    def estimated(theta):
        value, _, lp, _ = jax.vmap(_history, in_axes=(None, 0))(theta, histories)
        values = jax.vmap(adjoint.trajectory_surrogate)(value, lp)
        return jnp.sum(jax.lax.stop_gradient(jnp.exp(lp))*values)
    h = 1e-5
    fd = (exact(1.2+h)-exact(1.2-h))/(2*h)
    np.testing.assert_allclose(jax.grad(estimated)(1.2), fd, rtol=1e-7)
    _, _, lp, _ = jax.vmap(_history, in_axes=(None, 0))(1.2, histories)
    np.testing.assert_allclose(jnp.exp(lp).sum(), 1., atol=1e-14)


@pytest.mark.parametrize("observable", ["constant", "position"])
def test_wall_expected_derivative_against_one_particle_quadrature(observable):
    # Independent identical particles implement many draws of ONE particle, no collisions.
    n = 65536
    grid = GridSpec(lo=jnp.array([0.]), hi=jnp.array([1.]), n=(1,))
    faces = boundary.Faces.uniform(boundary.BC.DIFFUSE, 1)
    wall = boundary.WallParams.uniform(1, 1., acc=0.)
    def value(x):
        out = boundary.apply_faces(jnp.full((n, 1), x),
            jnp.tile(jnp.array([[1., 0., 0.]]), (n, 1)), jnp.ones(n, bool),
            grid, faces, wall, K_BOLTZ, .1, jax.random.PRNGKey(8), epsilon=.02)
        phi = jnp.ones(n) if observable == "constant" else out[0][:, 0]
        return adjoint.surrogate(phi, out[4], out[2])
    got = jax.jit(jax.grad(value))(.89)
    expected = 0. if observable == "constant" else 2*ndtr(.5)-1
    np.testing.assert_allclose(got, expected, atol=.06)


def test_relaxed_wall_score_is_zero():
    wall = boundary.WallParams.uniform(1, 1., acc=.6)
    _, lp = boundary._diffuse_velocity(jnp.array([[-1., 2., 0.]]), wall,
        K_BOLTZ, 0, 0, jax.random.PRNGKey(4), delta=.1)
    np.testing.assert_array_equal(lp, 0.)
