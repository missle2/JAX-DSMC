"""Coarse training windows from fine reference snapshots (plan step 6).

A window starts from a fine snapshot (:mod:`.reference`) **Poisson-thinned** to the coarse
particle count -- each particle kept with probability ``p = N_coarse / N_fine`` and the
weight ``fnum / p``, unbiased in ``f`` -- runs a forward-only **burn-in** of ``burn``
sample blocks (so training sees the operator's own state distribution, not the fine
run's), then a differentiable **rollout** of ``blocks`` sample blocks with the operator under
training, recording the coarse moments (:mod:`.moments`) at the reference's sample times.

The loss is the **two-replica** product ``sum_f w_f (m_a - m_ref)(m_b - m_ref)`` over sample
times and coarse cells: two independently thinned and seeded replicas, so its expectation is
the squared bias alone -- plain MSE would reward the operator for suppressing its own
fluctuations.  Reverse-mode AD through the rollout gives ``dL/dtheta`` (exact for the
window; its length is limited by chaos, plan "Objective and gradients").
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from examples.learned_collision import moments
from examples.learned_collision.reference import _scale
from examples.learned_collision.shear_wave import BLOCK, setup
from jax_dsmc import simulate
from jax_dsmc.collide import vremax_init
from jax_dsmc.state import CollideState, ParticleState

LOSS_FIELDS = (7, 10)
"""Fields the default loss compares: the shear stress ``pxy`` and heat flux ``qx``."""


class Case(NamedTuple):
    """A coarse window configuration (everything but the particles and theta)."""

    grid: object
    sp: object
    params: object
    static: object
    lo: float
    hi: float
    n_coarse: int
    scale: jax.Array
    p_keep: float
    lam: float
    vth: float


def coarse_case(wave, length_mfp, dx_c, ppc_c, n_fine, a_max=0, pairing="uniform",
                height_mfp=None):  # fmt: skip
    """The coarse grid, gas and parameters of a window, with particle capacity for a
    ``p = cells * ppc_c / n_fine`` thinning of an ``n_fine``-particle snapshot (six-sigma
    headroom; overflow is counted by :func:`thin`).  ``height_mfp``: the reference slab's
    height (2-D; it must be a multiple of ``dx_c``), ``None`` for a 1-D reference."""
    ny = 1 if height_mfp is None else round(height_mfp / dx_c)
    if height_mfp is not None and not math.isclose(ny * dx_c, height_mfp):
        raise ValueError(f"slab height {height_mfp} is not a multiple of dx_c {dx_c}")
    _, _, grid, sp, params, static, _, c = setup(dx_c, ppc_c, 0, a_max, wave, length_mfp, ny)
    expected = c["n_part"]
    cap = math.ceil(expected + 6.0 * math.sqrt(expected) + 8)
    static = dataclasses.replace(static, N=cap, pairing=pairing)
    n_coarse = c["cells"]
    if ny > 1:
        static = dataclasses.replace(static, D=2)
    return Case(grid, sp, params, static, float(grid.lo[0]), float(grid.hi[0]), n_coarse,
                _scale(expected / n_coarse, c["vth"]), expected / n_fine, c["lambda"],
                c["vth"])  # fmt: skip


def thin(x, v, p, key, cap):
    """Keep each particle with probability ``p``: ``(x[cap], v[cap], alive[cap], overflow)``.
    The kept particles come first in their original order; ``overflow`` counts the kept
    ones beyond ``cap`` (dropped -- must be zero)."""
    keep = jax.random.uniform(key, (x.shape[0],)) < p
    count = jnp.sum(keep)
    idx = jnp.nonzero(keep, size=cap, fill_value=0)[0]
    alive = jnp.arange(cap) < count
    return x[idx], v[idx], alive, jnp.maximum(count - cap, 0)


def initial_state(snap_x, snap_v, case, key):
    """A thinned coarse ``(state, cstate, overflow)`` from one snapshot in reference units
    (``x / lambda``, ``v / v_th``)."""
    cap = case.static.N
    x, v, alive, overflow = thin(snap_x * case.lam, snap_v * case.vth, case.p_keep, key, cap)
    if x.ndim == 1:
        x = x[:, None]
    state = ParticleState(x=x, v=v, species=jnp.zeros(cap, jnp.int32), alive=alive,
                          logh_acc=jnp.zeros(cap))  # fmt: skip
    n_cells = case.params.fnum.shape[0]
    temp = jnp.full((n_cells,), 273.15)
    sp = case.sp
    cstate = CollideState(vremax=vremax_init(temp, sp.omega[0, 0], sp.tref[0, 0], sp.mr[0, 0],
                                             sp.prefactor[0, 0], n_cells, 1))  # fmt: skip
    return state, cstate, overflow


def _block(carry, step0, case, key, collide_fn, theta):
    """One sample block: ``BLOCK`` steps, then the nondimensional coarse moments."""
    state, cstate = carry
    (state, cstate), _ = simulate.run(
        state, cstate, case.grid, case.sp, case.params, case.static, key, step0=step0,
        emit_moments="none", collide_fn=collide_fn, collision_params=theta,
    )  # fmt: skip
    m = moments.coarse_moments(state, case.lo, case.hi, case.n_coarse) * case.scale
    return (state, cstate), m


def rollout(theta, state, cstate, key, case, collide_fn, blocks, step0=0):
    """``(state, cstate, m[blocks, C, 13])``: ``blocks`` sample blocks from step ``step0``."""
    body = functools.partial(_block, case=case, key=key, collide_fn=collide_fn, theta=theta)
    steps = step0 + BLOCK * jnp.arange(blocks)
    (state, cstate), ms = jax.lax.scan(body, (state, cstate), steps)
    return state, cstate, ms


def burn_in(theta, state, cstate, key, case, collide_fn, blocks):
    """The forward-only burn-in: :func:`rollout` with no gradient to anything."""
    state, cstate, _ = rollout(jax.lax.stop_gradient(theta), state, cstate, key, case,
                               collide_fn, blocks)  # fmt: skip
    return jax.lax.stop_gradient(state), jax.lax.stop_gradient(cstate)


def wave_basis(n_coarse):
    """``f64[2, C]``: the box's first Fourier pair ``(2/C) (cos k x_c, sin k x_c)`` at the
    coarse cell centres -- projecting a field on it gives its wave amplitudes."""
    phase = 2.0 * np.pi * (np.arange(n_coarse) + 0.5) / n_coarse
    return jnp.asarray(2.0 / n_coarse * np.stack([np.cos(phase), np.sin(phase)]))


def pair_loss(ms, ref, weights, fields=LOSS_FIELDS, basis=None):
    """The two-replica loss of ``ms[2, T, C, 13]`` against ``ref[T, C, 13]``: the mean over
    times (and cells, or with ``basis[K, C]`` the projections on its rows) of
    ``sum_f w_f (m_a - r)(m_b - r)``.  A single coarse cell's instantaneous flux is mostly
    noise (SNR ~1e-2 at 40 particles per cell in a 0.2 v_th shear wave); projecting on the
    flow's modes first averages it over every cell."""
    idx = jnp.array(fields)
    da = ms[0][..., idx] - ref[..., idx]
    db = ms[1][..., idx] - ref[..., idx]
    if basis is not None:
        da = jnp.einsum("kc,tcf->tkf", basis, da)
        db = jnp.einsum("kc,tcf->tkf", basis, db)
    return jnp.mean(jnp.sum(jnp.asarray(weights) * da * db, axis=-1))


def window_loss(theta, states, cstates, keys, ref, weights, case, collide_fn, blocks,
                step0=0, fields=LOSS_FIELDS, basis=None):  # fmt: skip
    """:func:`pair_loss` of two replicas (leading axis 2 of ``states``, ``cstates``,
    ``keys``) rolled out ``blocks`` sample blocks from step ``step0``."""
    run = functools.partial(rollout, theta, case=case, collide_fn=collide_fn, blocks=blocks,
                            step0=step0)  # fmt: skip
    _, _, ms = jax.vmap(run)(states, cstates, keys)
    return pair_loss(ms, ref, weights, fields, basis)


def field_weights(ref, fields=LOSS_FIELDS, basis=None):
    """``1 / mean(ref_f^2)`` per loss field over the window (of its projections with
    ``basis``): each field's relative error counts equally."""
    ref = np.asarray(ref)
    if basis is not None:
        ref = np.einsum("kc,tcf->tkf", np.asarray(basis), ref)
    return np.array([1.0 / max(np.mean(ref[..., f] ** 2), 1e-30) for f in fields])
