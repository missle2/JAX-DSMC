"""jax_dsmc.wall_knn: the tiled exact kNN against brute force, the own-key exclusion, and the
local fit recovering a known linear gradient (2026-11-05)."""

import jax.numpy as jnp
import numpy as np

from jax_dsmc import wall_knn


def _data(seed=0, n=6000, q=300, f=5):
    rng = np.random.default_rng(seed)
    z_n = rng.uniform(0.0, 12.0, (n, f))
    z_n[:, 2:] = rng.normal(0.0, 0.5, (n, f - 2))  # untiled (velocity-like) coordinates
    key_n = np.arange(n, dtype=np.int64)
    pick = rng.choice(n, q, replace=False)
    z_q = z_n[pick] + rng.normal(0.0, 0.05, (q, f))
    key_q = key_n[pick]  # each query's own node must be skipped
    return z_n, key_n, z_q, key_q


def test_exact_knn_matches_brute_force():
    z_n, key_n, z_q, key_q = _data()
    k = 16
    idx, d2, _tile = wall_knn.knn_exact(z_n, jnp.asarray(key_n), z_q, jnp.asarray(key_q),
                                        tile=2.0, k=k, block_q=64, block_n=512, n_pos=2,
                                        low_precision=False)  # fmt: skip
    idx = np.asarray(idx)
    dist = np.sum((z_q[:, None, :] - z_n[None, :, :]) ** 2, axis=-1)
    dist[np.arange(len(z_q)), key_q] = np.inf  # the own node
    ref = np.argsort(dist, axis=1)[:, :k]
    assert np.all(np.sort(idx, axis=1) == np.sort(ref, axis=1))
    assert np.all(idx != key_q[:, None])
    np.testing.assert_allclose(np.sort(np.asarray(d2), axis=1),
                               np.sort(np.take_along_axis(dist, ref, axis=1), axis=1),
                               rtol=1e-12)  # fmt: skip


def test_local_fit_recovers_a_linear_gradient():
    rng = np.random.default_rng(1)
    n, q, k = 4000, 50, 64
    feat_n = rng.normal(0.0, 1.0, (n, 3))
    slope = np.array([2.0, -1.5, 0.5])
    lam = feat_n @ slope + 0.3
    feat_q = rng.normal(0.0, 0.3, (q, 3))
    d = np.sum((feat_q[:, None, :] - feat_n[None, :, :]) ** 2, axis=-1)
    idx = np.argsort(d, axis=1)[:, :k].astype(np.int32)
    jac = np.broadcast_to(np.eye(3)[None, :, :2], (q, 3, 2))  # x = the first two features
    g, bad = wall_knn.local_gradients(jnp.asarray(idx), jnp.asarray(feat_n), jnp.asarray(lam),
                                      jnp.asarray(feat_q), jnp.asarray(jac), 2, 1e-12, 1e8,
                                      chunk=16)  # fmt: skip
    np.testing.assert_allclose(np.asarray(g), np.broadcast_to(slope[:2], (q, 2)), rtol=1e-6)
    assert not np.any(np.asarray(bad))
