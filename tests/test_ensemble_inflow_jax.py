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
    fit = jax.jit(lambda p, v, q: device.local_affine(p, v, valid, q, 16, jnp.ones(4), tile=32))
    np.testing.assert_allclose(fit(points, values, queries), expected, atol=2e-10)


def test_empty_and_singleton_reconstruction():
    points = jnp.zeros((8, 4))
    query = jnp.ones((3, 4))
    for n in (0, 1):
        got = device.local_affine(
            points, jnp.full(8, 7.0), jnp.arange(8) < n, query, 4, jnp.ones(4), tile=4
        )
        np.testing.assert_allclose(got, 7.0 if n else 0.0, atol=1e-8)


@pytest.mark.parametrize("reconstruction", ["affine", "ballistic"])
def test_jitted_forward_backward_masks_and_normalization(reconstruction):
    cfg = host.Config(
        n_initial=64,
        cells=4,
        steps=2,
        retained=8,
        source_samples=8,
        partners=2,
        neighbors=8,
        reconstruction=reconstruction,
    )
    run = jax.jit(lambda t, key: device.forward(cfg, t, key, (8, 8), capacity=96))
    trace = run(jnp.full((2, 3), 4.0), jax.random.key(5))
    assert int(trace.normalization) == 96
    assert int(trace.frames.counts[0].sum()) == 64
    assert not bool(trace.invalid)
    grad = jax.jit(lambda tr: device.backward(tr, cfg, jax.random.key(6)))(trace)
    assert grad.shape == (2, 3)
    assert np.isfinite(grad).all()


@pytest.mark.parametrize("n", [16, 64])
def test_capacity_failure_is_explicit(n):
    if (
        n == 16
        and jax.__version__ == "0.6.2"
        and any(d.platform == "gpu" and "3050" in d.device_kind for d in jax.devices())
    ):
        pytest.skip(
            "RTX 3050/JAX 0.6.2: 16-particle source compile exceeded 180 s; 64-particle overflow still tested"
        )
    cfg = host.Config(
        n_initial=n, cells=4, steps=2, retained=8, source_samples=8, partners=2, neighbors=8
    )
    tr = device.forward(cfg, jnp.ones((2, 3)), jax.random.key(1), (n, n), capacity=n)
    assert bool(tr.invalid)


def test_collisionless_backward_on_host_trace_matches_exact_characteristics():
    cfg = host.Config(n_initial=32, cells=2, steps=3, retained=8, source_samples=256, q0=0.0)
    tr = host.forward(
        cfg, np.array([[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]]), seed=7, injection_counts=(256, 256)
    )
    frames = device.Frame(
        *(
            jnp.asarray(np.stack([getattr(f, field) for f in tr.frames]))
            for field in ("points", "valid", "counts")
        )
    )
    sources = jnp.asarray([[s.points for s in step] for step in tr.sources])
    converted = device.Trace(
        frames,
        sources,
        jnp.ones(sources.shape[:-1], bool),
        jnp.asarray([[s.weight for s in step] for step in tr.sources]),
        jnp.asarray(tr.temperatures),
        jnp.asarray(tr.normalization),
        jnp.asarray(tr.objective),
        jnp.asarray(False),
    )
    got = jax.jit(lambda trace: device.backward(trace, cfg, jax.random.key(2)))(converted)
    np.testing.assert_allclose(got, host.backward(tr), rtol=1e-12, atol=1e-12)


def test_custom_reward_is_carried_and_configuration_mismatch_rejected():
    from dataclasses import replace  # noqa: PLC0415 -- optional reference dependency

    cfg = host.Config(n_initial=32, cells=2, steps=1, retained=4, source_samples=4, q0=0.0)

    def fn(p):
        return jnp.ones(p.shape[:-1])

    tr = device.forward(cfg, jnp.ones((2, 3)), jax.random.key(1), (4, 4), 40, terminal_reward=fn)
    a = device.backward(tr, cfg, jax.random.key(2))
    b = device.backward(tr, cfg, jax.random.key(2), terminal_reward=fn)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="configuration"):
        device.backward(tr, replace(cfg, dt=0.1), jax.random.key(2))
