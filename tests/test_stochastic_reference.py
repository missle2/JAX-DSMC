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
from jax_dsmc.sampling import sample_diffuse_velocity
from jax_dsmc.state import GridSpec


def test_same_realization_baseline_is_rejected():
    # For phi=B~Bernoulli(p), centering by phi itself deletes the entire gradient.
    with pytest.raises(ValueError, match="independent"):
        adjoint.surrogate(jnp.ones(1), jnp.zeros(1), jnp.ones(1, bool), baseline="mean")


def _history(theta, bits):
    """Four particles, two disjoint matchings, all 16 acceptance histories."""
    sp = species_params(
        jnp.array([K_BOLTZ]),
        jnp.array([[0.1]]),
        jnp.array([[0.5]]),
        jnp.ones((1, 1)),
        jnp.array([[1.4]]),
    )
    v = jnp.array([[1.2, 0.3, -0.2], [-0.8, 0.1, 0.7], [0.2, 1.0, 0.4], [-0.3, -0.7, -0.5]])
    v = v.at[0, 0].set(theta)
    local = jnp.zeros(4)
    total = jnp.float64(0.0)
    for k, (i, j) in enumerate(((0, 1), (2, 3), (0, 2), (1, 3))):
        q = collision_rate(v[i], v[j], 0, 0, sp)
        p = q / 0.3
        lh = jnp.log(jnp.where(bits[k], p, 1 - p))
        a, b = scatter(v[i], v[j], 0, 0, sp, 0.17 + 0.13 * k, 0.21 + 0.11 * k)
        v = v.at[i].set(jnp.where(bits[k], a, v[i]))
        v = v.at[j].set(jnp.where(bits[k], b, v[j]))
        local = local.at[i].add(lh).at[j].add(lh)
        total = total + lh
    phi = v[:, 1] ** 4
    return jnp.mean(phi), local, total, phi


def test_global_likelihood_matches_exact_enumeration_after_changing_partners():
    histories = jnp.array(list(itertools.product((False, True), repeat=4)))

    def exact(theta):
        value, _, lp, _ = jax.vmap(_history, in_axes=(None, 0))(theta, histories)
        return jnp.sum(jnp.exp(lp) * value)

    def estimated(theta):
        value, _, lp, _ = jax.vmap(_history, in_axes=(None, 0))(theta, histories)
        values = jax.vmap(adjoint.trajectory_surrogate)(value, lp)
        return jnp.sum(jax.lax.stop_gradient(jnp.exp(lp)) * values)

    def terminal_labels(theta):
        _, local, lp, phi = jax.vmap(_history, in_axes=(None, 0))(theta, histories)
        values = jax.vmap(lambda p, l: adjoint.surrogate(p, l, jnp.ones(4, bool)))(phi, local)
        return jnp.sum(jax.lax.stop_gradient(jnp.exp(lp)) * values)

    h = 1e-5
    fd = (exact(1.2 + h) - exact(1.2 - h)) / (2 * h)
    # Prove this benchmark actually detects the old terminal-label omission.
    # This statement conditions on scattering and matching; it is not a theorem
    # about averaging every random variable in the paper's numerical protocol.
    assert abs(float(jax.grad(terminal_labels)(1.2) - fd)) > 1e-4
    np.testing.assert_allclose(jax.grad(estimated)(1.2), fd, rtol=1e-7)
    _, _, lp, _ = jax.vmap(_history, in_axes=(None, 0))(1.2, histories)
    np.testing.assert_allclose(jnp.exp(lp).sum(), 1.0, atol=1e-14)


@pytest.mark.parametrize("observable", ["constant", "position"])
def test_wall_expected_derivative_against_one_particle_quadrature(observable):
    # Independent identical particles implement many draws of ONE particle, no collisions.
    n = 65536
    grid = GridSpec(lo=jnp.array([0.0]), hi=jnp.array([1.0]), n=(1,))
    faces = boundary.Faces.uniform(boundary.BC.DIFFUSE, 1)
    wall = boundary.WallParams.uniform(1, 1.0, acc=0.0)

    def value(x):
        out = boundary.apply_faces(
            jnp.full((n, 1), x),
            jnp.tile(jnp.array([[1.0, 0.0, 0.0]]), (n, 1)),
            jnp.ones(n, bool),
            grid,
            faces,
            wall,
            K_BOLTZ,
            0.1,
            jax.random.PRNGKey(8),
            epsilon=0.02,
        )
        phi = jnp.ones(n) if observable == "constant" else out[0][:, 0]
        return adjoint.surrogate(phi, out[4], out[2])

    got = jax.jit(jax.grad(value))(0.89)
    expected = 0.0 if observable == "constant" else 2 * ndtr(0.5) - 1
    np.testing.assert_allclose(got, expected, atol=0.06)


def test_relaxed_wall_score_is_zero():
    wall = boundary.WallParams.uniform(1, 1.0, acc=0.6)
    _, lp = boundary._diffuse_velocity(
        jnp.array([[-1.0, 2.0, 0.0]]), wall, K_BOLTZ, 0, 0, jax.random.PRNGKey(4), delta=0.1
    )
    np.testing.assert_array_equal(lp, 0.0)


def test_mixed_wall_resolves_specular_event_before_thermal_event():
    grid = GridSpec(lo=jnp.array([0.0]), hi=jnp.array([1.0]), n=(1,))
    faces = boundary.Faces((boundary.BC.SPECULAR, boundary.BC.DIFFUSE))
    wall = boundary.WallParams.uniform(1, 1.0, acc=0.0)
    # Left specular hit at .1, right wall at 1.1, then travel left for .2.
    out = boundary.apply_faces(
        jnp.array([[0.1]]),
        jnp.array([[-1.0, 0.0, 0.0]]),
        jnp.ones(1, bool),
        grid,
        faces,
        wall,
        K_BOLTZ,
        1.3,
        jax.random.PRNGKey(1),
    )
    np.testing.assert_allclose(out[0], [[0.8]], atol=1e-14)
    np.testing.assert_allclose(out[1], [[-1.0, 0.0, 0.0]], atol=1e-14)
    assert int(out[3]) == 0


def test_wall_reports_exhausted_event_capacity():
    grid = GridSpec(lo=jnp.array([0.0]), hi=jnp.array([1.0]), n=(1,))
    faces = boundary.Faces.uniform(boundary.BC.DIFFUSE, 1)
    wall = boundary.WallParams.uniform(1, 1.0, acc=0.0)
    out = boundary.apply_faces(
        jnp.array([[0.1]]),
        jnp.array([[1.0, 0.0, 0.0]]),
        jnp.ones(1, bool),
        grid,
        faces,
        wall,
        K_BOLTZ,
        10.0,
        jax.random.PRNGKey(1),
    )
    assert int(out[3]) == 1


def test_p3_anisotropic_wall_temperature_components():
    def energy(temp):
        v = sample_diffuse_velocity(
            jax.random.PRNGKey(2), 100000, temp, K_BOLTZ, 0, 1.0, jnp.zeros(3)
        )
        return jnp.mean(jnp.sum(v * v, axis=1))

    np.testing.assert_allclose(
        jax.grad(energy)(jnp.array([0.6, 0.7, 0.8])), [2.0, 1.0, 1.0], rtol=0.02
    )


def test_eager_wall_steps_reuse_one_compilation_for_changing_particle_values():
    from jax_dsmc.wall_adjoint import _advance_jitted, advance  # noqa: PLC0415

    grid = GridSpec(lo=jnp.array([0.0]), hi=jnp.array([1.0]), n=(1,))
    faces = boundary.Faces.uniform(boundary.BC.DIFFUSE, 1)
    wall = boundary.WallParams.uniform(1, 1.0)
    x, v = jnp.full((32, 1), 0.5, dtype=jnp.float64), jnp.ones((32, 3), dtype=jnp.float64)
    alive = jnp.ones(32, bool)
    args = (alive, grid, faces, wall, K_BOLTZ, 0.1)
    x, v, _, _, _ = advance(x, v, *args, jax.random.PRNGKey(0), epsilon=0.01)
    count = _advance_jitted._cache_size()
    for k in range(1, 5):
        x, v, _, _, _ = advance(x, v, *args, jax.random.PRNGKey(k), epsilon=0.01)
    jax.block_until_ready(v)
    assert _advance_jitted._cache_size() == count
