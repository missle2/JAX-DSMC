"""Device inlet reconstruction compared with the independent SciPy oracle."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_dsmc import ensemble_inflow as host
from jax_dsmc import ensemble_inflow_jax as device


def test_tiled_affine_matches_scipy_with_padding():
    rng = np.random.default_rng(5)
    points = rng.normal(size=(77, 4))
    values = np.sin(points[:, 1]) + points[:, 0] ** 2
    queries = rng.normal(size=(19, 4))
    valid = np.arange(77) < 65
    expected = host.LocalAffine(points[valid], values[valid], 16)(queries)
    fit = jax.jit(lambda p, v, q: device.local_affine(p, v, valid, q, 16,
                                                     jnp.ones(4), tile=32))
    np.testing.assert_allclose(fit(points, values, queries), expected, atol=2e-10)


def test_empty_and_singleton_reconstruction():
    points = jnp.zeros((8, 4))
    query = jnp.ones((3, 4))
    for n in (0, 1):
        got = device.local_affine(points, jnp.full(8, 7.), jnp.arange(8) < n,
                                  query, 4, jnp.ones(4), tile=4)
        np.testing.assert_allclose(got, 7. if n else 0., atol=1e-8)


@pytest.mark.parametrize('reconstruction', ['affine', 'ballistic'])
def test_jitted_forward_backward_masks_and_normalization(reconstruction):
    cfg = host.Config(n_initial=64, cells=4, steps=2, retained=8,
                      source_samples=8, partners=2, neighbors=8,
                      reconstruction=reconstruction)
    run = jax.jit(lambda t, key: device.forward(cfg, t, key, (8, 8), capacity=96))
    trace = run(jnp.full((2, 3), 4.), jax.random.key(5))
    assert int(trace.normalization) == 96
    assert int(trace.frames.counts[0].sum()) == 64
    assert not bool(trace.invalid)
    grad = jax.jit(lambda tr: device.backward(tr, cfg, jax.random.key(6)))(trace)
    assert grad.shape == (2, 3)
    assert np.isfinite(grad).all()


def test_capacity_failure_is_explicit():
    cfg = host.Config(n_initial=16, cells=2, steps=1, retained=4, source_samples=4)
    tr = device.forward(cfg, jnp.ones((2, 3)), jax.random.key(1), (16, 16), capacity=16)
    assert bool(tr.invalid)
