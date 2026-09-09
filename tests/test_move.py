"""Advection and boundary conditions (spec section 9.5), parametrised over D."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jax_dsmc import boundary  # noqa: E402
from jax_dsmc.boundary import BC  # noqa: E402
from jax_dsmc.move import move  # noqa: E402
from jax_dsmc.state import GridSpec, ParticleState  # noqa: E402

DIMS = [1, 2, 3]


def grid_of(D, lo=-1.0, hi=3.0):
    return GridSpec(lo=jnp.full(D, lo), hi=jnp.full(D, hi), n=(2,) * D)


def state_of(x, v):
    n = x.shape[0]
    return ParticleState(
        x=x, v=v, species=jnp.zeros(n, jnp.int32), alive=jnp.ones(n, bool), logh_acc=jnp.zeros(n)
    )


# --------------------------- periodic ---------------------------------------


@pytest.mark.parametrize("D", DIMS)
def test_periodic_keeps_particles_inside(D):
    g = grid_of(D)
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (500, D)) * (g.hi - g.lo) + g.lo
    v = 50.0 * jax.random.normal(jax.random.fold_in(key, 1), (500, 3))
    s = move(state_of(x, v), g, jnp.float64(0.05), BC.PERIODIC)
    assert bool(jnp.all(s.x >= g.lo)) and bool(jnp.all(s.x < g.hi))
    assert bool(jnp.array_equal(s.v, v)), "periodic must not touch velocity"


@pytest.mark.parametrize("D", DIMS)
def test_periodic_is_translation_invariant_by_a_full_period(D):
    """Shifting a start point by exactly one period must land in the same place."""
    g = grid_of(D)
    L = g.hi - g.lo
    x = jnp.full((3, D), 0.3) + g.lo
    v = jnp.zeros((3, 3)).at[:, :D].set(1.0)
    a = move(state_of(x, v), g, jnp.float64(0.1), BC.PERIODIC).x
    b = move(state_of(x + L, v), g, jnp.float64(0.1), BC.PERIODIC).x
    assert float(jnp.max(jnp.abs(a - b))) < 1e-12


@pytest.mark.parametrize("D", DIMS)
def test_periodic_adjoint_is_the_identity(D):
    """[P3] section 3.1: the wrap is a translation, so the cotangent passes through."""
    g = grid_of(D)
    x = jnp.full((4, D), 0.5) + g.lo
    v = jnp.zeros((4, 3)).at[:, :D].set(2.0)
    j = jax.jacrev(lambda xx: move(state_of(xx, v), g, jnp.float64(0.1), BC.PERIODIC).x)(x)
    j = np.asarray(j).reshape(4 * D, 4 * D)
    assert np.allclose(j, np.eye(4 * D), atol=1e-12)


# --------------------------- specular ---------------------------------------


def brute_force_fold(x, lo, hi, n_iter=200):
    """Iterative reference: mirror repeatedly until inside.  The spec's own check."""
    x = np.array(x, dtype=float)
    sign = np.ones_like(x)
    for _ in range(n_iter):
        below, above = x < lo, x > hi
        if not (below.any() or above.any()):
            break
        x = np.where(below, 2 * lo - x, x)
        x = np.where(above, 2 * hi - x, x)
        sign = np.where(below | above, -sign, sign)
    return x, sign


@pytest.mark.parametrize("D", DIMS)
def test_specular_matches_a_brute_force_multibounce_reference(D):
    """The closed-form triangle wave must equal iterative mirroring, many bounces deep."""
    g = grid_of(D)
    lo, hi = float(g.lo[0]), float(g.hi[0])
    # spread far outside the box so several reflections are required
    x_new = jnp.asarray(np.linspace(lo - 4.5 * (hi - lo), hi + 4.5 * (hi - lo), 400))
    x_new = jnp.stack([x_new] * D, axis=-1)
    v = jnp.ones((x_new.shape[0], 3))
    x, vv = boundary.apply_specular(x_new, v, g)
    ref_x, ref_sign = brute_force_fold(np.asarray(x_new), lo, hi)
    assert np.allclose(np.asarray(x), ref_x, atol=1e-12)
    assert np.allclose(np.asarray(vv)[:, :D], ref_sign, atol=1e-12)
    assert bool(jnp.all((x >= g.lo - 1e-12) & (x <= g.hi + 1e-12)))


def test_specular_worked_example_from_the_spec():
    """Section 9.2 verifies two cases by hand; both must hold."""
    g = GridSpec(lo=jnp.zeros(1), hi=jnp.ones(1), n=(1,))
    for x_in, want_x, want_sgn in ((1.2, 0.8, -1.0), (-0.3, 0.3, -1.0), (0.4, 0.4, 1.0)):
        x, v = boundary.apply_specular(jnp.array([[x_in]]), jnp.ones((1, 3)), g)
        assert float(x[0, 0]) == pytest.approx(want_x, abs=1e-12)
        assert float(v[0, 0]) == pytest.approx(want_sgn, abs=1e-12)


@pytest.mark.parametrize("D", DIMS)
def test_specular_adjoint_flips_sign_with_the_fold_parity(D):
    """``dx/dx_new == dv/dv == sgn`` -- [P3] eq. ``C_ki``, free from AD."""
    g = grid_of(D)
    for x_in in (0.4, 1.2, -0.3, 2.9):
        xx = jnp.full((1, D), x_in)
        dxdx = jax.jacrev(lambda a: boundary.apply_specular(a, jnp.ones((1, 3)), g)[0])(xx)
        _, v_out = boundary.apply_specular(xx, jnp.ones((1, 3)), g)
        sgn = float(v_out[0, 0])
        assert float(np.asarray(dxdx).reshape(D, D)[0, 0]) == pytest.approx(sgn, abs=1e-12)


@pytest.mark.parametrize("D", DIMS)
def test_specular_never_leaves_the_domain(D):
    g = grid_of(D)
    key = jax.random.PRNGKey(4)
    x = jax.random.uniform(key, (2000, D)) * (g.hi - g.lo) + g.lo
    v = 500.0 * jax.random.normal(jax.random.fold_in(key, 1), (2000, 3))
    s = move(state_of(x, v), g, jnp.float64(0.02), BC.SPECULAR)
    assert bool(jnp.all(s.x >= g.lo - 1e-12)) and bool(jnp.all(s.x <= g.hi + 1e-12))


@pytest.mark.parametrize("D", DIMS)
def test_specular_preserves_speed(D):
    g = grid_of(D)
    key = jax.random.PRNGKey(5)
    x = jax.random.uniform(key, (500, D)) * (g.hi - g.lo) + g.lo
    v = 400.0 * jax.random.normal(jax.random.fold_in(key, 1), (500, 3))
    s = move(state_of(x, v), g, jnp.float64(0.03), BC.SPECULAR)
    assert (
        float(jnp.max(jnp.abs(jnp.linalg.norm(s.v, axis=-1) - jnp.linalg.norm(v, axis=-1)))) < 1e-12
    )


# --------------------------- outflow ----------------------------------------


@pytest.mark.parametrize("D", DIMS)
def test_outflow_kills_only_particles_that_left(D):
    g = grid_of(D)
    inside = jnp.full((1, D), 0.5)
    outside = jnp.full((1, D), 99.0)
    x = jnp.concatenate([inside, outside])
    v = jnp.zeros((2, 3))
    s = move(state_of(x, v), g, jnp.float64(0.0), BC.OUTFLOW)
    assert bool(s.alive[0]) and not bool(s.alive[1])


@pytest.mark.parametrize("D", DIMS)
def test_outflow_alive_is_monotone_non_increasing(D):
    """True through M4.  From M5 inflow refills slots and this is replaced by a
    mass balance -- see section 9.3 and gap U6."""
    g = grid_of(D)
    key = jax.random.PRNGKey(6)
    x = jax.random.uniform(key, (300, D)) * (g.hi - g.lo) + g.lo
    v = 200.0 * jax.random.normal(jax.random.fold_in(key, 1), (300, 3))
    s = state_of(x, v)
    prev = int(jnp.sum(s.alive))
    for _ in range(5):
        s = move(s, g, jnp.float64(0.01), BC.OUTFLOW)
        now = int(jnp.sum(s.alive))
        assert now <= prev
        prev = now


# --------------------------- contract ---------------------------------------


@pytest.mark.parametrize("D", DIMS)
@pytest.mark.parametrize("bc", [BC.PERIODIC, BC.SPECULAR, BC.OUTFLOW])
def test_move_is_jittable_and_shape_stable(D, bc):
    g = grid_of(D)
    x = jnp.full((64, D), 0.5)
    v = jnp.ones((64, 3))
    f = jax.jit(move, static_argnames=("bc",))
    s = f(state_of(x, v), g, jnp.float64(0.1), bc=bc)
    assert s.x.shape == (64, D) and s.v.shape == (64, 3)


@pytest.mark.parametrize("D", DIMS)
def test_only_the_first_D_velocity_components_advect(D):
    """Velocities stay 3-component whatever D is (section 0.2)."""
    g = grid_of(D, lo=-1e6, hi=1e6)
    x = jnp.zeros((1, D))
    v = jnp.arange(3.0).reshape(1, 3) + 1.0
    s = move(state_of(x, v), g, jnp.float64(1.0), BC.PERIODIC)
    assert float(jnp.max(jnp.abs(s.x[0] - v[0, :D]))) < 1e-9


def test_unknown_bc_is_rejected():
    g = grid_of(1)
    with pytest.raises(ValueError, match="unknown boundary"):
        boundary.apply(jnp.zeros((1, 1)), jnp.zeros((1, 3)), jnp.ones(1, bool), g, "nope")
