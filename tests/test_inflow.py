"""Inflow slot reuse and the gap-U6 guard (spec sections 9.3, 19).

U6's failure mode has no symptom: a dead slot refilled without clearing ``logh_acc``
gives the injected particle the score history of whatever occupied that index, and it
then contributes an ``eta`` term for collisions it never took part in.  No error, no
``NaN``, no shape change -- just a wrong gradient.  The register's gate is therefore
"inject into a dead slot; assert accumulator 0 **and gradient equals a fresh
particle's**", and both halves are asserted here.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from scipy import integrate  # noqa: E402

from jax_dsmc import adjoint, inflow  # noqa: E402
from jax_dsmc.boundary import BC, Faces, WallParams, apply_faces  # noqa: E402
from jax_dsmc.constants import K_BOLTZ  # noqa: E402
from jax_dsmc.state import GridSpec, ParticleState  # noqa: E402

N = 64


def _state(key, n=N, dim=1, stale=True):
    """A state whose slots 0..n/4 are dead and carry a *nonzero* accumulator."""
    k1, k2 = jax.random.split(key)
    dead = jnp.arange(n) < n // 4
    return ParticleState(
        x=jax.random.uniform(k1, (n, dim), dtype=jnp.float64),
        v=jax.random.normal(k2, (n, 3), dtype=jnp.float64),
        species=jnp.zeros(n, jnp.int32),
        alive=~dead,
        # the poison: history left behind by the slot's previous occupant
        logh_acc=jnp.where(dead, 3.7, 0.0) if stale else jnp.zeros(n),
    )


def test_revived_slot_has_a_zeroed_accumulator():
    key = jax.random.PRNGKey(0)
    st = _state(key)
    revive = ~st.alive
    new = inflow.revive_slots(st, revive, jnp.zeros_like(st.x), jnp.ones_like(st.v))
    assert int(inflow.stale_accumulator_count(new, revive)) == 0
    assert float(jnp.max(jnp.abs(new.logh_acc[revive]))) == 0.0
    assert bool(jnp.all(new.alive))


def test_revive_leaves_live_slots_completely_untouched():
    """Including their accumulators -- a live particle's history is not a slot reuse."""
    key = jax.random.PRNGKey(1)
    st = dataclasses.replace(_state(key), logh_acc=jnp.linspace(0.0, 1.0, N))
    live = st.alive
    new = inflow.revive_slots(st, ~st.alive, jnp.zeros_like(st.x), jnp.ones_like(st.v))
    np.testing.assert_array_equal(np.asarray(new.logh_acc[live]), np.asarray(st.logh_acc[live]))
    np.testing.assert_array_equal(np.asarray(new.x[live]), np.asarray(st.x[live]))
    np.testing.assert_array_equal(np.asarray(new.v[live]), np.asarray(st.v[live]))


def test_revive_refuses_to_resurrect_a_live_slot():
    """``revive & ~alive``: asking to revive a living particle is a no-op, not a reset."""
    key = jax.random.PRNGKey(2)
    st = dataclasses.replace(_state(key), logh_acc=jnp.full((N,), 2.5))
    new = inflow.revive_slots(st, jnp.ones(N, bool), jnp.zeros_like(st.x), jnp.zeros_like(st.v))
    live = st.alive
    np.testing.assert_array_equal(np.asarray(new.logh_acc[live]), np.asarray(st.logh_acc[live]))


def _objective(theta, state, inherited):
    """``J~`` where the score accumulator is a genuine function of ``theta``.

    This is what makes the U6 check real.  ``magic(L) = exp(L - stop_grad(L))`` has
    gradient ``dL/dtheta``, so a stale accumulator that were merely a *constant* would
    differentiate to zero and prove nothing.  The actual hazard is that the previous
    occupant's score was itself built from ``theta``-dependent collisions, so the
    inherited term carries a nonzero ``dL/dtheta`` into the new particle's reward.
    """
    own = theta * jnp.linspace(0.1, 0.9, N)  # the new particle's own collisions
    logh = state.logh_acc + own + theta * inherited
    return adjoint.surrogate(state.v[:, 0] ** 2, logh, state.alive, rho=1.0)


def test_revived_slot_gradient_equals_a_fresh_particles():
    """**The U6 gate.**  A revived slot must be indistinguishable from a fresh one.

    Two states reach identical positions, velocities and ``alive`` masks: one by
    reviving dead slots that carry a stale accumulator, the other by being born that
    way.  Objective *and* gradient must agree exactly.
    """
    key = jax.random.PRNGKey(3)
    st = _state(key, stale=True)
    revive = ~st.alive
    dead = np.asarray(revive)
    x_new = jnp.full_like(st.x, 0.25)
    v_new = jnp.full_like(st.v, 2.0)

    revived = inflow.revive_slots(st, revive, x_new, v_new)
    fresh = ParticleState(
        x=revived.x,
        v=revived.v,
        species=revived.species,
        alive=revived.alive,
        logh_acc=jnp.zeros(N),
    )
    none_inherited = jnp.zeros(N)

    j_r, g_r = jax.value_and_grad(_objective)(1.0, revived, none_inherited)
    j_f, g_f = jax.value_and_grad(_objective)(1.0, fresh, none_inherited)
    assert j_r == pytest.approx(j_f, rel=1e-14)
    assert float(abs(g_r - g_f)) == 0.0

    # ... and the guard is load-bearing.  Skip the revival, so the slot keeps the
    # previous occupant's theta-dependent score, and the gradient moves.
    inherited = jnp.where(revive, 1.3, 0.0)
    poisoned = dataclasses.replace(revived, logh_acc=st.logh_acc)
    g_bad = jax.grad(_objective)(1.0, poisoned, inherited)
    assert abs(float(g_bad - g_f)) > 1e-9, (
        "a stale accumulator did not change the gradient -- this test proves nothing"
    )
    # the contamination is exactly the inherited score times the revived slots' reward
    phi = np.asarray(fresh.v[:, 0] ** 2)
    expected = float(np.sum(1.3 * phi[dead]) / N)
    assert float(g_bad - g_f) == pytest.approx(expected, rel=1e-10)


def test_every_accumulator_is_registered():
    """A new per-particle history field must be added to ``ACCUMULATOR_FIELDS``.

    Reflection rather than a hand-kept list: the U6 failure recurs verbatim for any
    accumulator added later, and nothing else would catch it.
    """
    known = {"x", "v", "species", "alive"} | set(inflow.ACCUMULATOR_FIELDS)
    fields = {f.name for f in dataclasses.fields(ParticleState)}
    assert fields <= known, (
        f"unregistered ParticleState field(s) {sorted(fields - known)}: if any accumulates "
        "over a run it must be listed in inflow.ACCUMULATOR_FIELDS and zeroed on revival"
    )


def test_mass_balance_replaces_the_monotone_alive_invariant():
    """Section 9.3: from M5 ``alive`` is not monotone; the balance is what holds instead."""
    assert int(inflow.mass_balance_residual(100, 105, injected=12, deleted=7)) == 0
    assert int(inflow.mass_balance_residual(100, 100, injected=0, deleted=0)) == 0
    # a lost slot is caught
    assert int(inflow.mass_balance_residual(100, 104, injected=12, deleted=7)) != 0


# --- the flux sampler and the injection stage -------------------------------

MASS = 6.63e-26
T_RES = 300.0
L_BOX = 1e-3
N_CELLS = 8


def _vscale(temp=T_RES):
    return math.sqrt(2.0 * K_BOLTZ * temp / MASS)


@pytest.mark.parametrize("s", [-1.0, 0.0, 0.5, 2.0, 5.0])
def test_normal_beta_matches_the_analytic_inflow_flux(s):
    """``beta ~ (beta + s) exp(-beta^2)`` on ``[max(-3,-s), 3]`` -- Bird's bounded scheme.

    Checked against numerical quadrature of the density rather than a stored baseline,
    so a change of sampler that preserved the distribution passes and one that broke it
    does not.
    """
    n = 200_000
    beta, accepted = inflow.sample_normal_beta(jax.random.PRNGKey(0), n, s)
    b = np.asarray(beta)[np.asarray(accepted)]
    assert accepted.mean() > 0.99, f"acceptance {float(accepted.mean()):.3f} too low"

    lo = max(-3.0, -s)
    w = lambda x: (x + s) * np.exp(-x * x)  # noqa: E731
    z = integrate.quad(w, lo, 3.0)[0]
    m1 = integrate.quad(lambda x: x * w(x), lo, 3.0)[0] / z
    m2 = integrate.quad(lambda x: x * x * w(x), lo, 3.0)[0] / z
    assert b.mean() == pytest.approx(m1, rel=0.01)
    assert b.std() == pytest.approx(math.sqrt(m2 - m1 * m1), rel=0.02)
    assert b.min() >= lo - 1e-12, "sampled below the truncation, so the flux weight is wrong"


def test_entrants_have_an_inward_normal_velocity():
    """Every injected particle must be travelling *into* the domain."""
    res = inflow.InflowParams.uniform(1, 1e20, T_RES)
    for side, sign in ((0, +1.0), (1, -1.0)):
        v, accepted = inflow.sample_face_velocities(
            jax.random.PRNGKey(1), 20_000, res, 0, side, jnp.float64(MASS)
        )
        vn = np.asarray(v[:, 0])[np.asarray(accepted)]
        assert np.all(sign * vn > 0.0), f"side {side}: {int(np.sum(sign * vn <= 0))} wrong-way"


def test_mol_inflow_matches_the_sparta_expression():
    """``fix_emit.cpp:166-174``, including the ``s < -3`` cutoff."""
    for s in (-0.5, 0.0, 1.0, 3.0):
        want = (math.exp(-s * s) + math.sqrt(math.pi) * s * (1.0 + math.erf(s))) / (
            2.0 * math.sqrt(math.pi)
        )
        assert float(inflow.mol_inflow(s)) == pytest.approx(want, rel=1e-12)
    assert float(inflow.mol_inflow(-4.0)) == 0.0


def _uniform_inflow_case(n_part=20_000, dim=1):
    grid = GridSpec(lo=jnp.zeros(dim), hi=jnp.full(dim, L_BOX), n=(N_CELLS,) * dim)
    faces = Faces.uniform(BC.INFLOW, dim)
    res = inflow.InflowParams.uniform(dim, 1e20, T_RES)
    fnum = 1e20 * float(L_BOX) ** dim / (n_part * 0.5)
    dt = 0.15 * (L_BOX / N_CELLS) / _vscale()
    return grid, faces, res, fnum, dt


def test_inflow_reaches_a_steady_state_that_reproduces_the_reservoir():
    """Inflow at both ends, outflow of whatever leaves: the box must hold the reservoir.

    This is the inflow analogue of "start Maxwellian, stay Maxwellian" -- a physics
    gate rather than a regression baseline.  It also asserts the section-9.3 mass
    balance at **every** step, which is the invariant that replaces the M1-M4
    monotone-``alive`` assertion once slots can be refilled.
    """
    n_part = 20_000
    grid, faces, res, fnum, dt = _uniform_inflow_case(n_part)
    wall = WallParams.uniform(1, T_RES)
    key = jax.random.PRNGKey(0)

    @jax.jit
    def step(state, k):
        before = jnp.sum(state.alive)
        x, v, alive, _, _ = apply_faces(
            state.x,
            state.v,
            state.alive,
            grid,
            faces,
            wall,
            jnp.float64(MASS),
            jnp.float64(dt),
            k,
        )
        moved = ParticleState(x=x, v=v, species=state.species, alive=alive, logh_acc=state.logh_acc)
        deleted = before - jnp.sum(alive)
        out, injected, shortfall = inflow.inject(
            moved,
            grid,
            faces,
            res,
            jnp.float64(MASS),
            jnp.float64(fnum),
            jnp.float64(dt),
            k,
            capacity=64,
        )
        resid = inflow.mass_balance_residual(before, jnp.sum(out.alive), injected, deleted)
        return out, jnp.sum(out.alive), resid, shortfall

    state = ParticleState(
        x=L_BOX * jax.random.uniform(key, (n_part, 1), dtype=jnp.float64),
        v=math.sqrt(K_BOLTZ * T_RES / MASS)
        * jax.random.normal(jax.random.fold_in(key, 1), (n_part, 3), dtype=jnp.float64),
        species=jnp.zeros(n_part, jnp.int32),
        alive=jnp.arange(n_part) < n_part // 2,
        logh_acc=jnp.zeros(n_part),
    )
    counts = []
    for s in range(400):
        state, n_alive, resid, shortfall = step(state, jax.random.fold_in(key, s))
        assert int(resid) == 0, f"mass balance broken at step {s}: {int(resid)}"
        assert float(shortfall) == 0.0
        counts.append(int(n_alive))

    tail = np.array(counts[-100:])
    assert tail.std() / tail.mean() < 0.02, "population is not steady"
    assert abs(tail.mean() - counts[0]) / counts[0] < 0.05, "population drifted"
    temp = float(MASS * jnp.mean(jnp.sum(state.v[state.alive] ** 2, axis=1)) / (3.0 * K_BOLTZ))
    assert temp == pytest.approx(T_RES, rel=0.03), f"steady T={temp:.1f}"


def test_entrants_are_spread_across_the_step_not_piled_on_the_face():
    """The randomised first-move fraction (``fix_emit_face.cpp:600``) is anti-pileup.

    Without it every entrant sits exactly on the face at the start of the step and the
    first cell shows an artificial density spike.
    """
    n_part = 20_000
    grid, faces, res, fnum, dt = _uniform_inflow_case(n_part)
    state = ParticleState(
        x=jnp.zeros((n_part, 1)),
        v=jnp.zeros((n_part, 3)),
        species=jnp.zeros(n_part, jnp.int32),
        alive=jnp.zeros(n_part, bool),
        logh_acc=jnp.zeros(n_part),
    )
    out, injected, _ = inflow.inject(
        state,
        grid,
        faces,
        res,
        jnp.float64(MASS),
        jnp.float64(fnum),
        jnp.float64(dt),
        jax.random.PRNGKey(4),
        capacity=64,
    )
    assert int(injected) > 0
    entered = np.asarray(out.x[out.alive, 0])
    on_face = np.sum((entered == 0.0) | (entered == L_BOX))
    assert on_face == 0, f"{on_face} entrants left sitting exactly on a face"
    assert entered.std() > 0.0


# --- the inflow-driven shock, structurally ----------------------------------


def test_two_sided_inflow_shock_delivers_its_upstream_state_and_conserves_mass():
    """A cheap version of ``examples/shock/inflow_driven``.

    The full Rankine-Hugoniot comparison needs 20 000 warm-up steps and lives in the
    example (all six moments land within 0.7 %).  What is affordable here is the part
    that is specifically about the *boundary condition*: the inlet must reproduce the
    state it was asked for, no injection may be dropped, and mass flux must be
    conserved through the domain.

    Mass flux is the sharp one.  When this case was first run the shock had parked 6
    mean free paths from the outlet -- inside its own thickness -- and the downstream
    plateau read 9 % off Rankine-Hugoniot **while the flux stayed conserved to 0.07 %**.
    That is what identified the fault as the domain layout rather than the solver.
    """
    from examples.shock import inflow_driven as sh  # noqa: PLC0415

    # `a_max` must be sized for the DOWNSTREAM side, which is 2.3x denser here, and the
    # NTC attempt count goes as npc^2 -- so ~5x the upstream figure.  At `a_max=64` this
    # clipped 5328 attempts, which is gap U4's exact silent-under-collision failure mode
    # and is caught only because `attempts_clipped` is asserted.
    case = sh.Case(n_cells=40, npc=100, gap_mfp=30.0, a_max=256, capacity=128)
    r = sh.profile(case, n_warm=1200, n_sample=800)

    assert r["shortfall"] == 0.0, "the per-face capacity bound was hit"
    assert r["clipped"] == 0, "A_max clipped, so the collision rate is wrong"

    m, t = r["measured"], r["target"]
    assert m["nrho1"] == pytest.approx(t["nrho1"], rel=0.05)
    assert m["temp1"] == pytest.approx(t["temp1"], rel=0.05)
    assert m["ux1"] == pytest.approx(t["ux1"], rel=0.05)
    # the shock compresses: downstream must be denser, hotter and slower
    assert m["nrho2"] > 1.5 * m["nrho1"]
    assert m["temp2"] > 1.5 * m["temp1"]
    assert m["ux2"] < 0.75 * m["ux1"]
    assert r["flux_ratio"] == pytest.approx(1.0, rel=0.05), (
        f"mass flux not conserved end to end: ratio {r['flux_ratio']:.4f}"
    )
