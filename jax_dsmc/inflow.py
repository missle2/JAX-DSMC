"""Inflow: reviving dead slots, and the flux-weighted sampler (spec sections 9.3, 5.4).

**Gap U6 lives here.**  Through M4 particles are only ever destroyed, so ``alive`` is
monotone non-increasing and a dead slot is never touched again.  Inflow breaks that: it
refills dead slots, and a refilled slot that keeps its old ``logh_acc`` gives the newly
injected particle the score history of whatever previously occupied that index.  It then
contributes an ``eta`` term for collisions it never took part in.

That failure is silent -- no error, no ``NaN``, no shape change, just a wrong gradient --
which is the same shape as the two worst defects this project has found (section 8.1's
exactly-zero gradients and section 8.4.1's momentum violation), both of which survived
careful reading.  So the rule is structural rather than remembered: :func:`revive_slots`
is the **only** sanctioned way to set ``alive`` from ``False`` to ``True``, and it zeroes
every per-particle accumulator itself.  :func:`stale_accumulator_count` is the runtime
check, and ``tests/test_inflow.py`` asserts the gradient of a revived slot equals a fresh
particle's -- which is the gate section 19 states for U6.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

from jax_dsmc import sampling  # noqa: E402
from jax_dsmc.boundary import BC  # noqa: E402
from jax_dsmc.constants import K_BOLTZ  # noqa: E402
from jax_dsmc.safe import safe_log  # noqa: E402
from jax_dsmc.state import ParticleState  # noqa: E402

ACCUMULATOR_FIELDS = ("logh_acc",)
"""Per-particle state that is a *history* and so must be reset on slot reuse.

Listed explicitly so that adding a second accumulator is a one-line change here rather
than a silent bug at every injection site.  If a field is added to
:class:`~jax_dsmc.state.ParticleState` that accumulates over a run, it belongs in this
tuple; :func:`test_every_accumulator_is_registered` fails if one is missed.
"""


def revive_slots(state: ParticleState, revive, x_new, v_new, species=None) -> ParticleState:
    """Bring dead slots back to life.  **The only sanctioned way to do so.**

    Args:
        state: current ``ParticleState``.
        revive: ``bool[N]``, true where a dead slot is to be filled.
        x_new: ``f64[N, D]`` positions to use where ``revive``.
        v_new: ``f64[N, 3]`` velocities to use where ``revive``.
        species: optional ``i32[N]``; defaults to keeping the slot's existing species.

    Every field in :data:`ACCUMULATOR_FIELDS` is zeroed on the revived slots, in the
    same update that sets ``alive`` -- that simultaneity is the whole point of gap U6.
    Slots not being revived are untouched, including their accumulators.
    """
    revive = revive & ~state.alive
    logh = jnp.where(revive, 0.0, state.logh_acc)
    return ParticleState(
        x=jnp.where(revive[:, None], x_new, state.x),
        v=jnp.where(revive[:, None], v_new, state.v),
        species=state.species if species is None else jnp.where(revive, species, state.species),
        alive=state.alive | revive,
        logh_acc=logh,
    )


def stale_accumulator_count(state: ParticleState, revived) -> jnp.ndarray:
    """How many just-revived slots still carry a nonzero accumulator.  Must be ``0``.

    Cheap enough to assert in tests and to surface as a diagnostic; the failure it
    catches produces no other symptom.
    """
    return jnp.sum(revived & (state.logh_acc != 0.0))


def mass_balance_residual(n_alive_before, n_alive_after, injected, deleted):
    """``injected - deleted - (alive_after - alive_before)``; must be exactly ``0``.

    Section 9.3: inflow retires the M1-M4 invariant that the alive count is monotone
    non-increasing.  This is its replacement, and it is strictly stronger -- it fails on
    a double-count or a lost slot as well as on a sign error.
    """
    return (injected - deleted) - (n_alive_after - n_alive_before)


# --- the flux-weighted sampler (spec section 5.4) ---------------------------

N_TRIALS = 16
"""Outer rejection trials.  See :func:`sample_normal_beta` for why this suffices."""


def mol_inflow(s):
    """Bird eq. 4.22 inward number flux, per unit ``vscale``.  ``fix_emit.cpp:166-174``.

    ``s`` is the speed ratio ``vstream . n_inward / vscale``.  SPARTA returns ``0``
    below ``s = -3``: the stream is then blowing hard *out* of the face and the inward
    flux is smaller than the sampler could resolve anyway.
    """
    s = jnp.asarray(s, dtype=jnp.float64)
    root_pi = jnp.sqrt(jnp.pi)
    flux = (jnp.exp(-(s**2)) + root_pi * s * (1.0 + jax.scipy.special.erf(s))) / (2.0 * root_pi)
    return jnp.where(s < -3.0, 0.0, flux)


def _beta_trial(carry, key, scosine):
    """One rejection trial for the normal component.  Module-level: it is a scan body."""
    accepted, beta = carry
    k1, k2 = jax.random.split(key)
    # SPARTA draws `beta_un = 6U - 3` and rejects `beta_un + scosine < 0` in an inner
    # do-while (`fix_emit_face.cpp:576-577`).  Sampling the truncated range directly is
    # exactly the same distribution with the inner loop removed -- which matters here,
    # because a nested data-dependent loop is inadmissible (section 12).
    lo = jnp.maximum(-3.0, -scosine)
    beta_un = lo + (3.0 - lo) * jax.random.uniform(k1, beta.shape, dtype=jnp.float64)
    root = jnp.sqrt(scosine * scosine + 2.0)
    p = (
        2.0
        * (beta_un + scosine)
        / (scosine + root)
        * jnp.exp(0.5 + 0.5 * scosine * (scosine - root) - beta_un * beta_un)
    )
    take = (~accepted) & (jax.random.uniform(k2, beta.shape, dtype=jnp.float64) <= p)
    return (accepted | take, jnp.where(take, beta_un, beta)), None


def sample_normal_beta(key, n, scosine, n_trials=N_TRIALS):
    """Normal-velocity ratio ``beta = v_n / vscale`` of an entering particle.

    Returns ``(beta, accepted)``.  Bird's bounded rejection scheme, as a ``lax.scan``
    of a **static** ``n_trials`` with an acceptance mask -- never a ``while_loop``,
    which would be neither reverse-differentiable nor statically shaped (section 12).
    ``accepted`` must be reported: an unaccepted slot is a particle that was not
    injected, which is a silent mass sink otherwise.
    """
    scosine = jnp.asarray(scosine, dtype=jnp.float64)
    init = (jnp.zeros(n, bool), jnp.zeros(n, dtype=jnp.float64))
    body = functools.partial(_beta_trial, scosine=scosine)
    (accepted, beta), _ = lax.scan(body, init, jax.random.split(key, n_trials))
    return beta, accepted


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class InflowParams:
    """**Traced** reservoir state behind each face.  All ``2 D``-long.

    Fields:
        nrho:    ``f64[2D]`` reservoir number density.
        temp:    ``f64[2D]`` reservoir temperature.
        vstream: ``f64[2D, 3]`` reservoir bulk velocity.

    Traced because these are the design variables an inflow sensitivity is taken with
    respect to -- and, per [P3] section 5, the ones whose perturbation changes particle
    *counts* and so needs the ensemble adjoint rather than a pathwise one (M8).
    """

    nrho: jax.Array
    temp: jax.Array
    vstream: jax.Array

    @classmethod
    def uniform(cls, dim, nrho, temp, vstream=None):
        n = 2 * dim
        vs = (
            jnp.zeros((n, 3))
            if vstream is None
            else jnp.broadcast_to(jnp.asarray(vstream, dtype=jnp.float64), (n, 3))
        )
        return cls(
            nrho=jnp.full((n,), nrho, dtype=jnp.float64),
            temp=jnp.full((n,), temp, dtype=jnp.float64),
            vstream=jnp.asarray(vs, dtype=jnp.float64),
        )


def face_area(grid, axis: int) -> jnp.ndarray:
    """Area of a face normal to ``axis``.  In 1-D this is 1 (a point), by convention."""
    extent = grid.hi - grid.lo
    area = jnp.array(1.0, dtype=jnp.float64)
    for other in range(len(grid.n)):
        if other != axis:
            area = area * extent[other]
    return area


def target_count(res, grid, axis, side, mass, fnum, dt):
    """Expected number of simulator particles entering face ``(axis, side)`` in ``dt``.

    ``nrho * vscale * mol_inflow(s) * A * dt / fnum``, with ``vscale = sqrt(2kT/m)``
    (Bird's most-probable speed, which is the scale ``mol_inflow`` is written in) and
    ``s`` the speed ratio along the **inward** normal.
    """
    f = 2 * axis + side
    inward = 1.0 if side == 0 else -1.0
    vscale = jnp.sqrt(2.0 * K_BOLTZ * res.temp[f] / mass)
    s = inward * res.vstream[f, axis] / vscale
    return res.nrho[f] * vscale * mol_inflow(s) * face_area(grid, axis) * dt / fnum


def sample_face_velocities(key, n, res, axis, side, mass):
    """Velocities of ``n`` candidate entrants at face ``(axis, side)``, ``f64[n, 3]``.

    Normal component from the flux-weighted rejection sampler; tangential components
    Maxwellian about the reservoir stream, matching ``fix_emit_face.cpp:584-587``
    (whose polar form is Box-Muller for the same pair of Gaussians).
    """
    f = 2 * axis + side
    inward = 1.0 if side == 0 else -1.0
    vscale = jnp.sqrt(2.0 * K_BOLTZ * res.temp[f] / mass)
    k_beta, k_tan = jax.random.split(key)
    s = inward * res.vstream[f, axis] / vscale
    beta, accepted = sample_normal_beta(k_beta, n, s)
    # tangential sigma is sqrt(kT/m) = vscale/sqrt(2)
    v = (vscale / jnp.sqrt(2.0)) * jax.random.normal(k_tan, (n, 3), dtype=jnp.float64)
    v = v + res.vstream[f]
    v = v.at[:, axis].set(inward * beta * vscale + res.vstream[f, axis])
    return v, accepted


def sample_face_positions(key, n, grid, axis, side):
    """Uniform positions **on** face ``(axis, side)``, ``f64[n, D]``."""
    D = len(grid.n)
    u = jax.random.uniform(key, (n, D), dtype=jnp.float64)
    x = grid.lo + u * (grid.hi - grid.lo)
    return x.at[:, axis].set(grid.lo[axis] if side == 0 else grid.hi[axis])


def inject(state, grid, faces, res, mass, fnum, dt, key, capacity):
    """Emit through every ``INFLOW`` face into dead slots.  Static shapes throughout.

    Returns ``(state, injected, shortfall)``.

    ``capacity`` is the static per-face bound on how many particles may enter in one
    step; the realised count is ``stochastic_round(target_count)`` and slots beyond it
    are masked off.  ``shortfall`` reports emissions that could not be placed -- either
    because the step wanted more than ``capacity`` or because there were not enough dead
    slots.  Like ``attempts_clipped`` it must be surfaced rather than swallowed: a
    silently dropped entrant is a mass sink that looks exactly like correct physics.

    Every revival goes through :func:`revive_slots`, so the U6 accumulator reset cannot
    be skipped at this call site or any future one.

    Entrants get a **randomised first-move fraction** ``dtremain = dt * U``
    (``fix_emit_face.cpp:600``, [DF] ``dsmcFreeStreamInflowPatch.C``) and are placed at
    ``x_face + dtremain * v``.  Without it every entrant would sit exactly on the face
    at the start of the step and produce an artificial density spike there.
    """
    n_part = state.x.shape[0]
    # dead slots first; `argsort` on a bool puts False (dead) ahead of True
    order = jnp.argsort(state.alive.astype(jnp.int32), stable=True)
    n_dead = jnp.sum(~state.alive)

    revive = jnp.zeros(n_part, bool)
    x_new = state.x
    v_new = state.v
    # M8: the score of the injection-count decision.  `stochastic_round` is a frozen
    # decision that carries no tangent, so `nrho` (and `dt`, `fnum`, `face_area`) reach
    # the objective through this log-probability and through nothing else -- without it
    # `dJ/d(nrho)` reads exactly 0.0, which looks like an answer rather than an absence.
    logh_new = jnp.zeros(n_part, dtype=jnp.float64)
    injected = jnp.array(0, jnp.int32)
    shortfall = jnp.array(0.0, jnp.float64)
    used = jnp.array(0, jnp.int32)

    for axis in range(len(grid.n)):
        for side in (0, 1):
            if faces.of(axis, side) is not BC.INFLOW:
                continue
            f = 2 * axis + side
            k_n, k_x, k_v, k_t = jax.random.split(jax.random.fold_in(key, f), 4)
            target = target_count(res, grid, axis, side, mass, fnum, dt)
            u_round = jax.random.uniform(k_n, (), dtype=jnp.float64)
            n_emit = sampling.stochastic_round(target, u_round).astype(jnp.int32)
            # P(round up) = frac(target); the realised direction is data, the probability
            # of it is what depends on `nrho`
            frac = target - jnp.floor(target)
            log_p_count = jnp.where(u_round < frac, safe_log(frac), safe_log(1.0 - frac))
            shortfall = shortfall + jnp.maximum(0.0, target - capacity)

            xf = sample_face_positions(k_x, capacity, grid, axis, side)
            vf, accepted = sample_face_velocities(k_v, capacity, res, axis, side, mass)
            frac = dt * jax.random.uniform(k_t, (capacity,), dtype=jnp.float64)
            xf = xf + vf[:, : len(grid.n)] * frac[:, None]

            wanted = (jnp.arange(capacity) < n_emit) & accepted
            slots = jax.lax.dynamic_slice(order, (used,), (capacity,))
            slot_is_dead = (jnp.arange(capacity) + used) < n_dead
            take = wanted & slot_is_dead
            # Slot starvation is the OTHER way an entrant goes missing, and until this
            # line it was invisible: `shortfall` counted only `target > capacity` even
            # though its contract says it reports both.  A saturated particle array then
            # looks exactly like correct physics -- measured on a Mach-4 two-sided shock,
            # the downstream density ratio flattened at 1.83 against a Rankine-Hugoniot
            # 3.37 (-46 %) with every diagnostic reading zero.
            shortfall = shortfall + jnp.sum((wanted & ~slot_is_dead).astype(jnp.float64))

            revive = revive.at[slots].set(jnp.where(take, True, revive[slots]))
            x_new = x_new.at[slots].set(jnp.where(take[:, None], xf, x_new[slots]))
            v_new = v_new.at[slots].set(jnp.where(take[:, None], vf, v_new[slots]))
            logh_new = logh_new.at[slots].set(jnp.where(take, log_p_count, logh_new[slots]))
            injected = injected + jnp.sum(take)
            used = used + capacity

    # AFTER `revive_slots`, which zeroes the accumulator for revived slots (gap U6): an
    # entrant has no inherited history, but the decision that created it is its own.
    state = revive_slots(state, revive, x_new, v_new)
    state = dataclasses.replace(state, logh_acc=state.logh_acc + logh_new)
    return state, injected, shortfall
