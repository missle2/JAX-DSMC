"""The device backward reproduces the host backward on the same captured records."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples import ensemble_relaxation as er
from examples.ensemble_relaxation import CONTROLS, relaxation_run, species
from examples.gradient_convergence import (
    COUETTE_CONTROLS,
    INLET_CONTROLS,
    couette_run,
    inlet_shock_geometry,
    inlet_shock_reward,
    inlet_shock_run,
)
from jax_dsmc import ensemble_backward as host
from jax_dsmc import ensemble_device as device
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.species_data import ARGON

jax.config.update("jax_enable_x64", True)


def _compare(state0, out, rec, grid, phi, n_nom, v_scale, npc_max, tol=1e-8):
    host.configure("sequential")
    try:
        adj = host.backward(rec, state0, out, phi, n_nom, grid, v_scale)
    finally:
        host.configure()
    dx = (np.asarray(grid.hi) - np.asarray(grid.lo)) / np.asarray(grid.n)
    scale = jnp.concatenate((jnp.asarray(dx), jnp.full(3, v_scale)))
    dev = device.backward(rec, state0.alive, out, phi(out.x, out.v) / n_nom, grid, scale, npc_max)
    np.testing.assert_allclose(
        np.asarray(dev.lambda0), adj.lambda0, rtol=tol, atol=tol * np.abs(adj.lambda0).max()
    )
    for b in adj.births:
        np.testing.assert_allclose(
            np.asarray(dev.lam[b.step])[b.slots], b.values, rtol=tol, atol=1e-14
        )
    for h in adj.hits:
        np.testing.assert_allclose(
            np.asarray(dev.lam[h.step])[h.slots], h.values, rtol=tol, atol=1e-14
        )
    for pr in adj.pairs:
        got = float(jnp.sum(dev.four_term[pr.step]))
        np.testing.assert_allclose(got, pr.four_term.sum(), rtol=1e-7, atol=1e-12)
    return adj, dev


def test_device_matches_host_on_the_inlet_shock():
    # dt = 0.5 flights with 1024 particles: ~14 entrants per face per step, so the
    # source path is exercised (the bench's 128-particle case injects 0.17 per step)
    geometry = inlet_shock_geometry(1024, 8, 0.5)
    state0, out, diag, rec = inlet_shock_run(
        dict(INLET_CONTROLS),
        jax.random.key(2),
        geometry,
        8,
        2048,
        4,
        "certified",
        "paper",
        capture=True,
    )
    length = geometry["length"]
    v_scale = float(np.sqrt(K_BOLTZ * ARGON.tref / ARGON.mass))
    adj, dev = _compare(
        state0,
        out,
        rec,
        geometry["grid"],
        lambda x, v: inlet_shock_reward(x, v, length),
        1024.0,
        v_scale,
        1024,
    )
    assert sum(len(p.i) for p in adj.pairs) > 20 and len(adj.births) > 0


def test_device_matches_host_on_couette_with_walls():
    t = dict(COUETTE_CONTROLS)
    state0, out, diag, rec, grid = couette_run(
        t, jax.random.key(5), 256, 4, 4, 128, 0.05, 0.005, "certified", "paper", capture=True
    )
    adj, dev = _compare(state0, out, rec, grid, lambda x, v: v[:, 1] ** 2, 256.0, 1.0, 256)
    assert len(adj.hits) > 0


@pytest.mark.parametrize("basis", ["affine", "quadratic", "quadratic_full"])
def test_field_query_is_exact_for_fields_in_its_span(basis):
    key = jax.random.key(0)
    kx, kv = jax.random.split(key)
    x = jax.random.uniform(kx, (1, 200, 1), dtype=jnp.float64)
    v = jax.random.normal(kv, (1, 200, 3), dtype=jnp.float64)
    valid = jnp.ones((1, 200), bool)

    def field(x, v):
        f = 0.3 + 0.5 * x[..., 0] + jnp.sum(jnp.array([1.0, -2.0, 0.5]) * v, axis=-1)
        if basis == "quadratic":
            f = f + 0.7 * jnp.sum(v**2, axis=-1) + 0.2 * x[..., 0] ** 2
        if basis == "quadratic_full":
            f = f + 0.7 * v[..., 0] ** 2 - 0.4 * v[..., 0] * v[..., 2] + 0.2 * x[..., 0] ** 2
        return f

    vals = field(x, v)
    qx, qv = jnp.array([[0.4]]), jnp.array([[0.2, -0.3, 0.1]])
    value, _, grad = device.field_query(
        x, v, valid, vals, qx, qv, jnp.array([1.0, 1.0, 1.0, 1.0]), 32, basis
    )
    np.testing.assert_allclose(value, field(qx, qv), rtol=1e-6)
    expected_grad = jax.grad(lambda vv: field(qx, vv[None])[0])(qv[0])
    np.testing.assert_allclose(grad[0], expected_grad, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("nodes", ["post", "staged"])
def test_device_virtual_pullback_carries_an_invariant_exactly(monkeypatch, nodes):
    """Same invariance as the host test: with |v|^2 / N as the reward every virtual
    four-term vanishes and the quadratic fit is exact, so lambda_0 = |v_0|^2 / N."""
    monkeypatch.setattr(device, "RIDGE_REL", 0.0)
    n = 400
    t = dict(CONTROLS)
    state0, out, _diag, rec, grid = relaxation_run(
        t, jax.random.key(3), n, 3, 512, 0.2, "certified", capture=True
    )
    dev = device.backward_virtual(
        rec,
        state0.alive,
        out,
        jnp.sum(out.v**2, axis=1) / n,
        grid,
        jnp.ones(4),
        species(t),
        jax.random.key(0),
        512,
        32,
        "quadratic_full",
        256,
        2,
        4,
        nodes,
        None,
    )
    np.testing.assert_allclose(
        np.asarray(dev.lambda0), np.sum(np.asarray(state0.v) ** 2, axis=1) / n, rtol=1e-9
    )
    assert dev.kernel.shape == (3,) and float(jnp.sum(jnp.abs(dev.kernel))) == 0.0


def test_device_staged_velocities_match_the_host_replay():
    n = 300
    t = dict(CONTROLS)
    _state0, _out, _diag, rec, grid = relaxation_run(
        t, jax.random.key(4), n, 1, 1024, 0.4, "certified", capture=True
    )
    pairs = jax.tree.map(lambda a: a[0], rec.pairs)
    _ids, _valid, cell, _rank, _count = device.cell_nodes(rec.x[0], rec.alive[0], grid, 512)
    got = np.asarray(device.staged_velocities(pairs, rec.v[0], rec.v_post[0], cell, 4))
    want = host._staged_velocities(
        rec.pairs, 0, np.asarray(rec.v[0]), np.asarray(rec.v_post[0]), n, 4
    )
    np.testing.assert_array_equal(got, want)
    assert not np.array_equal(want[2], want[0]) and not np.array_equal(want[2], want[-1])


def test_device_virtual_kernel_term_matches_the_host_form_in_expectation():
    """Relaxation ``diameter`` gradient: device and host draw different virtual partners,
    so they agree only statistically -- but the ensemble means over 4 seeds must be
    within the noise of the difference."""
    cfg = {"particles": 300, "steps": 2, "attempts": 512, "dt": 0.2, "majorant": "certified"}
    host.configure("virtual", 2, 8, "staged")
    try:
        diffs = []
        for seed in range(4):
            key = jax.random.key(seed)
            _, g_host, _ = er.ensemble_gradient("diameter", key, cfg, 32, device=False)
            _, g_dev, _ = er.ensemble_gradient("diameter", key, cfg, 32, device=True)
            diffs.append(g_dev - g_host)
    finally:
        host.configure()
    diffs = np.array(diffs)
    assert abs(diffs.mean()) < 3 * diffs.std(ddof=1) / np.sqrt(len(diffs)) + 1e-3
