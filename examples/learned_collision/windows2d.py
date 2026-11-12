"""Coarse training windows of the round-1 2-D flows (:mod:`.flows`, :mod:`.refdata`).

A window takes one reference snapshot's particle pool (replica ``r`` reads pool ``r``, so
the two replicas share no particle), a random stripe of the slab ``H_w`` high (the coarse
window height, :func:`.flows.window_height`; ``y`` is periodic and the flow planar, so any
stripe is the same in distribution), thins it to the coarse count, burns in forward-only,
then rolls out ``blocks`` sample blocks with the operator under training.

**Targets.**  A shear wave's are the reference's seed-mean coarse moments at the window's
sample times; a shock's the reference's shock-frame profile placed at the snapshot's shock
position and summed into the coarse cells (:func:`shock_target`), the same at every time.

**Loss fields.**  Shear: ``pxy``, projected on the box's first Fourier pair.  Shock: the
normal stress ``tau_xx = pxx - tr(p) / 3`` and heat flux ``qx`` per coarse cell, or with
``Case.modes = K > 0`` projected on the first ``K`` cosine modes of the domain
(:func:`shock_basis`; ``K = 1``: the integrals ``int tau_xx dx`` and ``int q_x dx`` alone,
``K = C``: the per-cell loss again, by Parseval).  2026-11-12: the per-cell loss traded heat
flux for stress (lctime3: int qx 0.95 -> 0.87 at M10) while the steady integrals are jointly
fixable by the two rates -- profile *shape* errors the rates cannot fix pull the heat-flux
rate the wrong way; the low modes keep the magnitudes and drop the shape (``lcloss``).

**Gradient estimators** (:func:`window_objective`):

* ``pair`` -- ``d/dtheta`` of the two-replica product ``mean(w d_a d_b)``, ``d = m - ref``:
  unbiased for the squared bias, but each factor ``d`` is mostly noise (window gradients
  stayed below 2 sigma over 64 windows, 2026-11-05).
* ``bias`` -- ``d/dtheta`` of ``mean(w b (m_a + m_b) / 2)`` with ``b`` a stop-gradient
  running estimate of ``E[m - ref]`` (per case, sample time and mode or cell) kept by the
  trainer from every window: the same expectation ``E[d] grad m`` with the noisy ``d``
  replaced by its average over many windows.  It lags ``theta`` (an exponential average),
  which a small learning rate makes harmless.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from examples.learned_collision import flows, moments
from examples.learned_collision.reference import _scale
from examples.learned_collision.shear_wave import BLOCK
from examples.learned_collision.windows import wave_basis
from jax_dsmc import simulate
from jax_dsmc.state import ParticleState

LOSS_FIELDS = {"shear": ("pxy",), "shock": ("txx", "qx")}


class Case(NamedTuple):
    """A coarse window configuration (everything but particles and theta)."""

    name: str
    kind: str
    grid: object
    sp: object
    params: object
    static: object
    bc: dict
    lo: float
    hi: float
    n_coarse: int
    scale: jax.Array
    p_rel: float
    lam: float
    vth: float
    height: float
    ref_height: float
    dx_c: float
    ppc_up: float
    cstate0: object
    modes: int = 0


def make_case(name, meta, ppc_c=flows.PPC_COARSE, a_max=0, pairing="uniform"):
    """The coarse window :class:`Case` of :data:`.flows.CASES` ``name`` against its
    reference's ``meta`` (pool fraction, fine resolution)."""
    ref_name, dx_c, _ = flows.CASES[name]
    ref = flows.REFERENCES[ref_name]
    height = flows.window_height(dx_c, ppc_c, meta.get("height", ref["height"]))
    _, cstate0, grid, sp, params, static, _, c, bc = flows.setup(ref, dx_c, ppc_c, 0, height,
                                                                a_max)  # fmt: skip
    expected = c["n_part"]
    if ref["kind"] == "shear":  # the wave setup sizes N exactly; leave room for the thinning
        static = dataclasses.replace(static, N=math.ceil(expected + 6.0 * math.sqrt(expected) + 8))
    static = dataclasses.replace(static, pairing=pairing)
    p_rel = flows.thinning(dx_c, ppc_c, meta["dx_fine"], meta["ppc"]) / meta["p_pool"]
    if p_rel > 1.0 + 1e-9:
        raise ValueError(f"{name}: the pools hold too few particles (p_rel {p_rel})")
    n_coarse = c["cells"]
    length = n_coarse * dx_c * c["lambda"]
    return Case(name, ref["kind"], grid, sp, params, static, bc, 0.0, length, n_coarse,
                _scale(c["ppc_up"], c["vth"]), min(p_rel, 1.0), c["lambda"], c["vth"],
                height, meta.get("height", ref["height"]), dx_c, c["ppc_up"], cstate0)  # fmt: skip


def pool_state(px, pv, pa, case, key):
    """A coarse ``(state, cstate, overflow)`` from one pool (``px[P, 2]`` in ``lambda``,
    ``pv[P, 3]`` in ``v_th``, ``pa[P]``): a random stripe ``case.height`` high, thinned by
    ``case.p_rel``."""
    k_off, k_thin = jax.random.split(key)
    px = jnp.asarray(px, jnp.float64)
    y = jnp.mod(px[:, 1] - jax.random.uniform(k_off) * case.ref_height, case.ref_height)
    keep = (jnp.asarray(pa) & (y < case.height)
            & (jax.random.uniform(k_thin, y.shape) < case.p_rel))  # fmt: skip
    cap = case.static.N
    count = jnp.sum(keep)
    idx = jnp.nonzero(keep, size=cap, fill_value=0)[0]
    state = ParticleState(
        x=jnp.stack([px[idx, 0], y[idx]], axis=1) * case.lam,
        v=jnp.asarray(pv, jnp.float64)[idx] * case.vth,
        species=jnp.zeros(cap, jnp.int32), alive=jnp.arange(cap) < count,
        logh_acc=jnp.zeros(cap),
    )  # fmt: skip
    return state, case.cstate0, jnp.maximum(count - cap, 0)


def placed_raw(prof_raw, x_s, case, meta):
    """``f64[C, 16]``: the reference's shock-frame mean raw sums (fine bins aligned on
    ``x_shock``) placed at shock position ``x_s`` [lambda] and summed into the coarse cells
    (beyond the profile's ends, its end bins: the plateaus)."""
    dx_f = meta["dx_fine"]
    k = round(case.dx_c / dx_f)
    shift = round((x_s - meta["x_shock"]) / dx_f)
    fine = np.arange(case.n_coarse * k) - shift
    rows = prof_raw[np.clip(fine, 0, prof_raw.shape[0] - 1)]
    return rows.reshape(case.n_coarse, k, 16).sum(axis=1)


def shock_target(prof_raw, x_s, case, meta, blocks):
    """``f64[blocks, C, 13]``: :func:`placed_raw` as the coarse moments' units -- population
    moments, the count as density relative to upstream -- the same at every time."""
    rows = placed_raw(prof_raw, x_s, case, meta)
    k = round(case.dx_c / meta["dx_fine"])
    m = np.asarray(moments.central(jnp.asarray(rows[:, 0]), jnp.asarray(rows[:, 1:4]),
                                   jnp.asarray(rows[:, 4:13].reshape(-1, 3, 3)),
                                   jnp.asarray(rows[:, 13:16]), unbiased=False))  # fmt: skip
    m = m * np.asarray(_scale(1.0, meta["vth_ms"]))
    m[:, 0] = rows[:, 0] / (k * meta["ppc_up"])
    return np.broadcast_to(m, (blocks, *m.shape)).copy()


def flux(m, kind):
    """``[..., F]``: the loss fields (:data:`LOSS_FIELDS`) of moments ``m[..., 13]``."""
    if kind == "shear":
        return m[..., 7:8]
    txx = m[..., 4] - (m[..., 4] + m[..., 5] + m[..., 6]) / 3.0
    return jnp.stack([txx, m[..., 10]], axis=-1)


def shock_basis(n_coarse, modes):
    """``f64[K, C]``: the first ``modes`` orthonormal cosine (DCT-II) rows over the coarse
    cells -- row 0 the integral (the cell sum over ``sqrt(C)``), row ``k`` ``k`` half-waves."""
    c = np.arange(n_coarse) + 0.5
    rows = [np.full(n_coarse, 1.0 / math.sqrt(n_coarse))]
    rows += [math.sqrt(2.0 / n_coarse) * np.cos(np.pi * k * c / n_coarse)
             for k in range(1, modes)]  # fmt: skip
    return jnp.asarray(np.stack(rows))


def loss_rows(case):
    """``K``: the rows of a case's projected deviation (:func:`project`)."""
    if case.kind == "shear":
        return 2
    return case.modes if case.modes > 0 else case.n_coarse


def project(d, case):
    """``[T, K, F]``: ``d[T, C, F]`` projected on the wave modes (shear), on the first
    ``case.modes`` cosine modes (shock, ``modes > 0``) or as is (shock)."""
    if case.kind == "shear":
        return jnp.einsum("kc,tcf->tkf", wave_basis(case.n_coarse), d)
    if case.modes > 0:
        return jnp.einsum("kc,tcf->tkf", shock_basis(case.n_coarse, case.modes), d)
    return d


def field_weights(target, case):
    """``1 / mean(f^2)`` of each loss field of the target (projected for a wave)."""
    f = np.asarray(project(flux(jnp.asarray(target), case.kind), case))
    return np.array([1.0 / max(float(np.mean(f[..., i] ** 2)), 1e-30) for i in range(f.shape[-1])])


def _block(carry, step0, case, key, collide_fn, theta):
    state, cstate = carry
    faces, capacity, res, wall = flows.run_kwargs(case.bc)
    (state, cstate), _ = simulate.run(
        state, cstate, case.grid, case.sp, case.params, case.static, key, step0=step0,
        emit_moments="none", faces=faces, wall=wall, res=res, capacity=capacity,
        collide_fn=collide_fn, collision_params=theta,
    )  # fmt: skip
    m = moments.coarse_moments(state, case.lo, case.hi, case.n_coarse) * case.scale
    return (state, cstate), m


def rollout(theta, state, cstate, key, case, collide_fn, blocks, step0=0):
    """``(state, cstate, m[blocks, C, 13])``."""
    body = functools.partial(_block, case=case, key=key, collide_fn=collide_fn, theta=theta)
    steps = step0 + BLOCK * jnp.arange(blocks)
    (state, cstate), ms = jax.lax.scan(body, (state, cstate), steps)
    return state, cstate, ms


def burn_in(theta, state, cstate, key, case, collide_fn, blocks):
    """Forward-only :func:`rollout`, no gradient to anything."""
    state, cstate, _ = rollout(jax.lax.stop_gradient(theta), state, cstate, key, case,
                               collide_fn, blocks)  # fmt: skip
    return jax.lax.stop_gradient(state), jax.lax.stop_gradient(cstate)


def window_objective(theta, states, cstates, keys, target, weights, bias, case, collide_fn,
                     blocks, step0, estimator, final=False):  # fmt: skip
    """``(objective, (pair, d_mean))`` of two replicas (leading axis 2): ``pair`` the
    two-replica loss, ``d_mean`` the replicas' mean projected deviation ``[T, K, F]`` (the
    trainer's running-bias update), and the objective whose gradient is the chosen
    estimator's (module docstring; ``bias`` ``[T, K, F]`` is ignored by ``pair``).
    ``final``: the aux also carries the replicas' end ``(states, cstates)`` (no gradient) --
    a persistent chain's next start (:mod:`.train` ``--warm-pool``)."""
    run = functools.partial(rollout, theta, case=case, collide_fn=collide_fn, blocks=blocks,
                            step0=step0)  # fmt: skip
    end_s, end_c, ms = jax.vmap(run)(states, cstates, keys)
    ref = flux(target, case.kind)
    d_a = project(flux(ms[0], case.kind) - ref, case)
    d_b = project(flux(ms[1], case.kind) - ref, case)
    w = jnp.asarray(weights)
    pair = jnp.mean(jnp.sum(w * d_a * d_b, axis=-1))
    d_mean = 0.5 * (d_a + d_b)
    if estimator == "pair":
        obj = pair
    else:
        obj = jnp.mean(jnp.sum(w * jax.lax.stop_gradient(bias) * d_mean, axis=-1))
    aux = (jax.lax.stop_gradient(pair), jax.lax.stop_gradient(d_mean))
    if final:
        aux = (*aux, jax.lax.stop_gradient((end_s, end_c)))
    return obj, aux
