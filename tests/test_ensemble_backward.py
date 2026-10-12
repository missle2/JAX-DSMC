"""Particle-carried ensemble adjoint on captured production (VSS/NTC/inflow) records."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.ensemble_relaxation import CONTROLS, relaxation_run, species
from examples.gradient_convergence import (
    COUETTE_CONTROLS,
    INLET_CONTROLS,
    couette_run,
    inlet_shock_geometry,
    inlet_shock_run,
)
from jax_dsmc import ensemble_backward as eb
from jax_dsmc.constants import K_BOLTZ
from jax_dsmc.species_data import ARGON

jax.config.update("jax_enable_x64", True)

V_SCALE = float(np.sqrt(K_BOLTZ * ARGON.tref / ARGON.mass))


def _run(steps=3, particles=96, cells=4, diam=None, seed=1):
    geometry = inlet_shock_geometry(particles, cells, 0.05)
    t = dict(INLET_CONTROLS)
    if diam is not None:
        t["diam"] = diam
    state0, out, diag, rec = inlet_shock_run(
        t, jax.random.key(seed), geometry, cells, 128, steps, "certified", "paper", capture=True
    )
    return geometry, state0, out, rec


def test_capture_is_consistent_with_the_states():
    geometry, state0, out, rec = _run()
    steps = rec.x.shape[0]
    # the state at t_{k+1} is (x_k, v_post_k, alive_k); at the end it is the output
    np.testing.assert_array_equal(np.asarray(rec.x[-1]), np.asarray(out.x))
    np.testing.assert_array_equal(np.asarray(rec.v_post[-1]), np.asarray(out.v))
    np.testing.assert_array_equal(np.asarray(rec.alive[-1]), np.asarray(out.alive))
    for k in range(steps):
        acc = np.asarray(rec.pairs.accept[k]).reshape(-1)
        pi = np.asarray(rec.pairs.pi[k]).reshape(-1)[acc]
        pj = np.asarray(rec.pairs.pj[k]).reshape(-1)[acc]
        assert np.all(np.asarray(rec.alive[k])[pi]) and np.all(np.asarray(rec.alive[k])[pj])
        # accepted pairs conserve energy between their recorded pre and post velocities
        vi, vj = (
            np.asarray(rec.pairs.vi[k]).reshape(-1, 3)[acc],
            np.asarray(rec.pairs.vj[k]).reshape(-1, 3)[acc],
        )
        vi_n, vj_n = (
            np.asarray(rec.pairs.vi_post[k]).reshape(-1, 3)[acc],
            np.asarray(rec.pairs.vj_post[k]).reshape(-1, 3)[acc],
        )
        np.testing.assert_allclose(
            np.sum(vi_n**2 + vj_n**2, axis=1), np.sum(vi**2 + vj**2, axis=1), rtol=1e-12
        )
        # every slot's LAST accepted collision (attempt order within its cell) ends at
        # its recorded post-collision velocity
        v_post = np.asarray(rec.v_post[k])
        order = np.argsort(
            np.flatnonzero(acc), kind="stable"
        )  # attempt-major == chronological per cell
        members = np.stack((pi[order], pj[order]), axis=1).reshape(-1)
        posts = np.stack((vi_n[order], vj_n[order]), axis=1).reshape(-1, 3)
        for slot in np.unique(members):
            last = np.flatnonzero(members == slot)[-1]
            np.testing.assert_allclose(v_post[slot], posts[last], rtol=1e-12)
    assert int(np.asarray(rec.born_face >= 0).sum()) > 0


def test_without_collisions_the_adjoint_is_pure_bookkeeping():
    """Tiny diameter: no accepted pairs.  Every carried value is the terminal reward of
    the same slot if it survived to the end, else zero."""
    geometry, state0, out, rec = _run(diam=1e-14)
    assert int(np.asarray(rec.pairs.accept).sum()) == 0
    n_nom = 96
    phi = lambda x, v: jnp.sum(v**2, axis=1)  # noqa: E731
    eb.configure("sequential")
    try:
        adj = eb.backward(rec, state0, out, phi, n_nom, geometry["grid"], V_SCALE)
    finally:
        eb.configure()
    final = np.where(np.asarray(out.alive), np.asarray(phi(out.x, out.v)) / n_nom, 0.0)
    # a slot alive at t_0 that is still the SAME particle at t_M carries its reward
    alive0 = np.asarray(state0.alive)
    reborn = np.zeros_like(alive0)
    for k in range(rec.x.shape[0]):
        reborn |= np.asarray(rec.born_face[k]) >= 0
    same = alive0 & ~reborn & np.asarray(out.alive)
    np.testing.assert_allclose(adj.lambda0[same], final[same], rtol=1e-12)
    assert np.all(adj.lambda0[~alive0] == 0.0)
    # births: the last step's newborns carry the terminal reward of their slot
    last = [b for b in adj.births if b.step == rec.x.shape[0] - 1]
    for b in last:
        np.testing.assert_allclose(b.values, final[b.slots], rtol=1e-12)


def test_collision_pullback_reproduces_the_pre_collision_field_for_an_invariant(monkeypatch):
    """With the field known exactly and a collision-invariant reward (|v|^2, no window),
    ``mu_i + sum[Lam(partner post) - Lam(partner pre)]`` equals |v_i^pre|^2 / N to
    round-off -- including slots that collide more than once in a step."""
    geometry, state0, out, rec = _run(steps=2, particles=256, cells=2)
    n_nom = 256.0
    monkeypatch.setattr(
        eb,
        "_field",
        lambda nx, nv, nc, vals, alive, qx, qv, qc, k, s: np.sum(qv**2, axis=1) / n_nom,
    )
    k = rec.x.shape[0] - 1
    x = np.asarray(rec.x[k]).reshape(-1, 1)
    v_pre = np.asarray(rec.v[k])
    v_post = np.asarray(rec.v_post[k])
    alive = np.asarray(rec.alive[k])
    mu = np.where(alive, np.sum(v_post**2, axis=1) / n_nom, 0.0)
    lam, _ = eb._collision_pullback(
        rec.pairs, k, x, v_post, alive, np.zeros(len(x), int), mu, 32, np.ones(4)
    )
    acc = np.asarray(rec.pairs.accept[k]).reshape(-1)
    pi = np.asarray(rec.pairs.pi[k]).reshape(-1)[acc]
    pj = np.asarray(rec.pairs.pj[k]).reshape(-1)[acc]
    assert len(pi) > 5
    multi = len(np.concatenate((pi, pj))) > len(np.unique(np.concatenate((pi, pj))))
    assert multi, "the case must contain a slot that collides twice"
    expected = np.where(alive, np.sum(v_pre**2, axis=1) / n_nom, 0.0)
    np.testing.assert_allclose(lam, expected, rtol=1e-10, atol=1e-18)


class _ExactEnergyField:
    """Stand-in for LocalFit returning |v|^2 / N at the queries (an exact quadratic field)."""

    def __init__(self, points, values, neighbors, scale, basis):
        del points, values, neighbors, scale, basis

    def __call__(self, query):
        return np.sum(np.asarray(query)[:, 1:] ** 2, axis=1) / 256.0


def test_sequential_pullback_reproduces_the_pre_collision_field_for_an_invariant(monkeypatch):
    """Reverse-order attempts with an exact field: lam == |v_pre|^2 / N to round-off, and
    the per-pair four-term values vanish (an invariant's four-term is zero exactly)."""
    geometry, state0, out, rec = _run(steps=2, particles=256, cells=2)
    monkeypatch.setattr(eb, "LocalFit", _ExactEnergyField)
    k = rec.x.shape[0] - 1
    x = np.asarray(rec.x[k]).reshape(-1, 1)
    v_pre, v_post, alive = np.asarray(rec.v[k]), np.asarray(rec.v_post[k]), np.asarray(rec.alive[k])
    mu = np.where(alive, np.sum(v_post**2, axis=1) / 256.0, 0.0)
    lam, pairs = eb._collision_pullback_sequential(
        rec.pairs, k, x, v_post, alive, np.zeros(len(x), int), mu, 32, np.ones(4)
    )
    assert len(pairs.i) > 5
    np.testing.assert_allclose(
        lam, np.where(alive, np.sum(v_pre**2, axis=1) / 256.0, 0.0), rtol=1e-10, atol=1e-18
    )
    # cancellation of O(|v|^2 / N ~ 1e3) terms leaves round-off of that scale
    np.testing.assert_allclose(pairs.four_term, 0.0, atol=1e-9)


def test_every_wall_bounce_is_recorded_and_scored_separately():
    """Hot walls and a long step make double hits common; each must appear as its own
    Hits entry with its own face and branch, weighted by the same end-of-move value."""
    t = {**COUETTE_CONTROLS, "temperature_left": 4.0, "temperature_right": 4.0}
    state0, out, diag, rec, grid = couette_run(
        t, jax.random.key(3), 256, 2, 3, 128, 0.6, 0.0, "certified", "paper", capture=True
    )
    face = np.asarray(rec.walls.face)  # (steps, K_bounce, N)
    per_slot = (face >= 0).sum(axis=1)  # hits per slot per step
    assert per_slot.max() >= 2, "the case must produce double hits"
    eb.configure("sequential")
    try:
        adj = eb.backward(rec, state0, out, lambda x, v: v[:, 1] ** 2, 256, grid, 1.0)
    finally:
        eb.configure()
    for k in range(face.shape[0]):
        counted = np.zeros(256, int)
        for h in adj.hits:
            if h.step == k:
                np.add.at(counted, h.slots, 1)
                # a bounce's record carries the face it hit, not the last one
                assert np.all(h.faces >= 0)
        alive = np.asarray(rec.alive[k])
        np.testing.assert_array_equal(counted[alive], per_slot[k][alive])
    # first and second hits of a double-hit slot are on opposite faces (a 1-D gap)
    k, slot = np.argwhere(per_slot >= 2)[0]
    faces = face[k][:, slot][face[k][:, slot] >= 0]
    assert faces[0] != faces[1]


def test_lean_record_rebuilds_v_post_bitwise():
    """``simulate.post_collision_v``: with ``v_post`` dropped (a lean record) the replay of
    each step's collision writes from ``v`` and the pair record is **bitwise** the stored
    post-collision state -- on the inlet shock (walls, entrants) and on a dense relaxation
    where particles collide several times in one step (later attempts see earlier writes)."""
    from jax_dsmc import simulate  # noqa: PLC0415
    from jax_dsmc.collide.ntc import step_pairs  # noqa: PLC0415

    _g, _s0, _out, shock = _run(steps=5, particles=256, cells=4)
    _s, _o, _d, dense, _grid = relaxation_run(
        dict(CONTROLS), jax.random.key(3), 400, 3, 512, 0.2, "certified", capture=True
    )
    for rec in (shock, dense):
        lean = simulate.lean_record(rec)
        rebuilt = np.asarray(simulate.post_collision_v(lean))
        np.testing.assert_array_equal(rebuilt, np.asarray(rec.v_post))
        for k in range(rec.x.shape[0]):  # the per-step accessors the device code uses
            step = jax.tree.map(lambda a, k=k: a[k], lean)
            got = np.asarray(simulate.step_v_post(step))
            np.testing.assert_array_equal(got, np.asarray(rec.v_post[k]))
            pairs = step_pairs(step)  # the sequential pull-back's partner ghosts
            np.testing.assert_array_equal(np.asarray(pairs.vi), np.asarray(rec.pairs.vi[k]))
            np.testing.assert_array_equal(np.asarray(pairs.vj), np.asarray(rec.pairs.vj[k]))
        assert np.any(rebuilt != np.asarray(rec.v))  # collisions happened
    # several accepted collisions of one particle within a step: the case slot order matters
    acc = np.asarray(dense.pairs.accept[0])
    pi, pj = np.asarray(dense.pairs.pi[0])[acc], np.asarray(dense.pairs.pj[0])[acc]
    assert np.bincount(np.concatenate([pi, pj])).max() >= 2
    with pytest.raises(ValueError, match="full record"):
        eb.from_device(simulate.lean_record(dense), None, 1.0)


def test_capture_works_without_transport():
    """``collide_only`` runs have no wall or source events; capture must still work."""
    state0, out, diag, rec, grid = relaxation_run(
        dict(CONTROLS), jax.random.key(1), 256, 2, 256, 0.2, "certified", capture=True
    )
    assert (
        int(np.asarray(rec.walls.face).max()) == -1 and int(np.asarray(rec.born_face).max()) == -1
    )
    assert int(np.asarray(rec.pairs.accept).sum()) > 0


@pytest.mark.parametrize("nodes", ["post", "staged"])
@pytest.mark.parametrize("ridge", [0.0, eb.RIDGE_REL])
def test_virtual_pullback_carries_an_invariant_exactly(monkeypatch, nodes, ridge):
    """The virtual-partner pull-back of |v|^2 / N: every virtual four-term vanishes
    (energy is conserved by every VSS pair) and the quadratic fit is exact, so lambda at
    the initial state is |v_0|^2 / N to round-off, at any substep count or node choice.
    The shipped slope ridge (1e-8 of the mean diagonal) costs ~1e-4 of that exactness."""
    monkeypatch.setattr(eb, "RIDGE_REL", ridge)
    n = 400
    t = dict(CONTROLS)
    state0, out, _diag, rec, grid = relaxation_run(
        t, jax.random.key(3), n, 3, 512, 0.2, "certified", capture=True
    )
    tag = eb.configure("virtual", 3, 4, nodes)
    try:
        assert tag == "_virt3x4" + ("st" if nodes == "staged" else "") + eb.basis_tag(eb.CELL_BASIS)
        assert eb.basis_tag("quadratic_full") == ""
        adj = eb.backward(
            rec, state0, out, lambda x, v: jnp.sum(v**2, axis=1), n, grid, 1.0, 32, sp=species(t)
        )
    finally:
        eb.configure()
    assert sum(len(p.i) for p in adj.pairs) == 3 * 3 * n * 4
    tol = 1e-9 if ridge == 0.0 else 1e-3
    np.testing.assert_allclose(
        adj.lambda0, np.sum(np.asarray(state0.v) ** 2, axis=1) / n, rtol=tol, atol=tol * 1e-3
    )
    for p in adj.pairs:
        np.testing.assert_allclose(p.four_term, 0.0, atol=tol)


def test_staged_velocities_replay_the_attempt_record():
    """Boundary 0 is the pre-collision state, the last boundary the post-collision one,
    and a boundary in between reproduces a sequential replay of the accepted slots."""
    n = 300
    t = dict(CONTROLS)
    _state0, _out, _diag, rec, _grid = relaxation_run(
        t, jax.random.key(4), n, 1, 1024, 0.4, "certified", capture=True
    )
    v_pre, v_post = np.asarray(rec.v[0]), np.asarray(rec.v_post[0])
    stages = eb._staged_velocities(rec.pairs, 0, v_pre, v_post, n, 4)
    assert np.array_equal(stages[0], v_pre) and np.array_equal(stages[-1], v_post)
    pi, pj = np.asarray(rec.pairs.pi[0])[:, 0], np.asarray(rec.pairs.pj[0])[:, 0]
    accept = np.asarray(rec.pairs.accept[0])[:, 0] & (pi < n)
    n_att = int((pi < n).sum())
    assert accept.sum() > 20
    for b in (1, 2, 3):
        replay = v_pre.copy()
        for a in range(int(np.ceil(n_att * b / 4))):
            if accept[a] and a / n_att < b / 4:
                replay[pi[a]] = np.asarray(rec.pairs.vi_post[0])[a, 0]
                replay[pj[a]] = np.asarray(rec.pairs.vj_post[0])[a, 0]
        np.testing.assert_array_equal(stages[b], replay)
    assert not np.array_equal(stages[2], v_pre) and not np.array_equal(stages[2], v_post)
