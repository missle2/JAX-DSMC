"""P3 disjoint forward driver checks independent of wall score identities."""

import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc import paper_walls as pw


def test_disjoint_maxwell_driver_conserves_momentum_and_energy():
    key = jax.random.PRNGKey(31)
    cfg = pw.Config(particles=128, cells=4, dt=0.5)
    x = jax.random.uniform(key, (128, 1))
    v = jax.random.normal(jax.random.fold_in(key, 1), (128, 3))
    out, invalid = pw.collision(x, v, key, cfg)
    assert int(invalid) == 0
    assert np.any(np.asarray(v) != np.asarray(out))
    np.testing.assert_allclose(out.sum(axis=0), v.sum(axis=0), atol=1e-12)
    np.testing.assert_allclose((out * out).sum(), (v * v).sum(), rtol=1e-13)
    derivative = jax.grad(lambda scale: jnp.sum(pw.collision(x, scale * v, key, cfg)[0] ** 2))(1.0)
    np.testing.assert_allclose(derivative, 2 * jnp.sum(v * v), rtol=1e-13)


def test_paper_driver_anisotropic_wall_gradients_are_finite_and_value_neutral():
    cfg = pw.Config(particles=64, cells=2, steps=2, observable="heat")
    temp = jnp.array([[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]])
    (value, (plain, invalid)), gradient = jax.value_and_grad(pw.solve, has_aux=True)(
        temp, 0.0, jax.random.PRNGKey(7), cfg
    )
    assert int(invalid) == 0
    assert float(value) == float(plain)
    assert gradient.shape == (2, 3)
    assert np.isfinite(gradient).all()
