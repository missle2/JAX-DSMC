"""No Time Counter collision driver (spec section 8).

Three things in this file are load-bearing and were each established by
measurement rather than by reading a reference code:

* **stochastic rounding** of the attempt count (8.1).  ``n = floor(f)`` freezes
  ``dt``, ``fnum`` and ``cell_volume`` out of the gradient *entirely* -- measured
  as exactly ``[0.0, 0.0, 0.0]``.  The fix makes the count a scored random
  decision, which is where those three sensitivities live.
* **the scratch row** at index ``N`` and the clamped ``npc`` (8.4.1).  Without
  them an empty cell's attempt writes stale velocities over a neighbouring
  cell's particles -- silently, and in 176/300 runs.
* **the a-priori thermal majorant** (7.2.3).  A pure ratchet lags a transient and
  permits majorant violations, each of which contributes a zero score and biases
  the gradient.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax, random

from jax_dsmc.collide.vss import (
    acceptance_prob,
    collision_rate,
    frame_is_well_defined,
    log_h,
    scatter,
)
from jax_dsmc.constants import EPS, K_BOLTZ
from jax_dsmc.grid import cell_volume, num_cells
from jax_dsmc.rng import (
    STAGE_ACCEPT,
    STAGE_NTC_ROUND,
    STAGE_PAIR,
    STAGE_SCATTER,
    key_for,
)
from jax_dsmc.safe import safe_log, safe_pow
from jax_dsmc.state import CollideState, Diagnostics, ParticleState

jax.config.update("jax_enable_x64", True)  # section 0.3: f64 throughout

K_EV = 5.0
"""Extreme-value multiplier for the a-priori majorant (section 7.2.3).

``P(|vr| > K_EV * v_ref) * npc^2 / 2 ~ 1.6e-4`` per cell per step.  Raising it
widens the margin at a linear cost in attempts.
"""

F_UP = 2.0
"""Safety factor on the majorant -- the 2-5x band required by section 7.2.1.

Measured (U3, section 17.5): ``F_UP = 2`` holds a minimum margin of 2.15 with
**zero** violations across equilibrium, cold and counter-streaming-beam ICs, each
with and without a heating transient.
"""


def thermal_majorant(temp, omega, tref, mr, prefactor, k_ev=K_EV, f_up=F_UP):
    """A-priori upper bound on ``q`` from the cell temperature.  ``f64[C]``.

    ``q(vr) = prefactor * (vr2)^(1-omega)`` is monotone in ``|vr|``, so over the
    reachable interval its supremum sits at an endpoint: the top one for
    ``omega < 1``, the clamp for ``omega > 1``.  ``temp`` is known at the **start**
    of the step (``sample.py`` computes it anyway), so unlike a ratchet on the
    observed maximum this does not lag a transient.

    This is the U3 gate's answer (section 7.2.3 / 17.5).  A pure stale-max ratchet
    seeded SPARTA's way measured a ``1.0e-2`` violation rate on a beam IC with a
    minimum margin of 0.62 -- i.e. ``Sigma`` sat *below* the true maximum.
    """
    vref2 = 2.0 * K_BOLTZ * temp / mr
    eps_abs = 1e-8 * (2.0 * K_BOLTZ * tref / mr)  # EPS_REL_VR2 * v_ref^2
    hi = prefactor * safe_pow(k_ev * k_ev * vref2, 1.0 - omega, EPS)
    lo = prefactor * safe_pow(eps_abs, 1.0 - omega, EPS)
    return f_up * jnp.maximum(hi, lo)


def vremax_init(temp0, omega, tref, mr, prefactor, n_cells, g_groups, k_ev=K_EV, f_up=F_UP):
    """Seed for ``CollideState.vremax``.  ``f64[C, G, G]``.

    **Not** SPARTA's ``vremax = 2 * cxs * vscale`` (``collide_vss.cpp:107-133``).
    That seed fails the U3 gate: with the stale-max rule it permits a ``1.1e-3``
    violation rate in equilibrium and ``1.0e-2`` under beams.  The a-priori
    thermal majorant is used instead, for both the seed and the per-step floor.
    """
    sigma = thermal_majorant(temp0, omega, tref, mr, prefactor, k_ev, f_up)
    return jnp.broadcast_to(jnp.reshape(sigma, (n_cells, 1, 1)), (n_cells, g_groups, g_groups))


def attempt_count(counts, vremax, volume, dt, fnum, key):
    """Number of candidate pairs per cell, by **stochastic rounding**.

    Args:
        counts: ``i32[C]`` particles per cell.
        vremax: ``f64[C]`` the majorant actually used this step (already floored
            by :func:`thermal_majorant`; pass ``Sigma``, not the raw carry).
        volume: ``f64[C]`` cell volumes.
        dt, fnum: TRACED scalars / ``f64[C]``.
        key: ``STAGE_NTC_ROUND`` key.

    Returns:
        ``(n, logP)`` -- ``i32[C]`` frozen counts and the ``f64[C]`` score.

    ``n = floor(f + U)``, SPARTA's *default* path (``collide_vss.cpp:147-149``),
    not the ``remainflag`` carry.  The carry version has no probability to score --
    ``P(n | f, remain)`` is degenerate -- so ``dt``, ``fnum`` and ``cell_volume``
    have **no gradient path at all** under it, measured as exactly ``0.0`` while
    the velocity path was healthy at ``9.5``.  ``logP`` is that missing path.

    The returned ``logP`` must be added to ``logh_acc`` for *every particle in the
    cell* -- it is a per-cell decision, not a per-pair one (section 7.3, ``j=1``).
    """
    npc = counts.astype(jnp.float64)
    f = 0.5 * npc * (npc - 1.0) * vremax * dt * fnum / volume
    u = random.uniform(key, npc.shape, dtype=jnp.float64)
    lower = jnp.floor(f)
    frac = f - lower
    up = u < frac  # FROZEN (bool)
    n = lax.stop_gradient(lower + up).astype(jnp.int32)  # FROZEN
    logP = jnp.where(up, safe_log(frac), safe_log(1.0 - frac))
    return n, logP


def pair_ntc(key, npc):
    """Pair selection **with replacement** -- SPARTA / [DF] ``noTimeCounter``.

    The production default, and the correct scheme for inhomogeneous flow where the
    attempt count must be set by the physical rate rather than the particle count.

    SPARTA resamples until ``i != j`` (``collide.cpp:530-533``): a ``while_loop``
    with a data-dependent trip count, neither reverse-differentiable nor statically
    shaped.  ``j = (i + 1 + r) mod npc`` with ``r`` on ``npc-1`` values covers every
    index except ``i`` exactly once, so ``j`` is uniform on the complement of
    ``{i}`` -- verified by chi-square (section 17).

    ``npc`` is **clamped to >= 2**: ``jax.random.randint(k, shape, 0, 0)`` silently
    returns 0 rather than erroring (2000/2000 seeds, JAX 0.6.2), and ``x % 0``
    returns 0 too, so an empty cell would produce ``i == j == 0`` and index into a
    neighbouring cell's particles.  See section 8.4.1.
    """
    k1, k2 = random.split(key)
    npc_safe = jnp.maximum(npc, 2)
    return pair_from_draws(
        random.randint(k1, npc.shape, 0, npc_safe),
        random.randint(k2, npc.shape, 0, npc_safe - 1),
        npc,
    )


def pair_from_draws(u_i, u_r, npc):
    """:func:`pair_ntc` from pre-drawn integers -- the form the scan uses.

    ``u_i`` must be uniform on ``[0, npc_safe)`` and ``u_r`` on
    ``[0, npc_safe - 1)``.  Split out so the attempt-slot scan can draw every
    slot's integers in one batched call (section 1 rule 3) rather than paying a
    kernel launch per slot.  Measured effect of the hoist: compile 17.0 s -> 2.4 s
    and per-slot cost 102 us -> 53 us (section 8.6).

    The ``npc_safe`` clamp is applied here too, and is load-bearing for exactly
    the reason given in :func:`pair_ntc`.
    """
    npc_safe = jnp.maximum(npc, 2)
    j = (u_i + 1 + u_r) % npc_safe
    return u_i, j  # FROZEN (ints)


def pair_perm(key, npc):
    """Disjoint pairing -- [DF] ``noRepeatCollisions``, and what both papers assume.

    [P2] eq. ``factorizationEa`` writes ``E_a^k = Prod_{i=1}^{N/2} E_ai^k``, a
    product over ``N/2`` *disjoint* pairs; [P1] Algorithm 1 says "without
    replacement".  The section-13 oracles run under this so they are faithful
    implementations of Algorithms 2 and 3 rather than approximations of them
    applied to a different pairing scheme.

    Not the production default: it fixes the collision count at ``npc/2`` per cell
    per step, decoupling it from ``sigma_T c_r dt`` -- correct only for a carefully
    tuned ``dt``.
    """
    perm = random.permutation(key, npc)
    return perm[0::2], perm[1::2]


#: Golden-ratio low-discrepancy weights for :func:`decision_weights`.  Any fixed
#: sequence of distinct irrational-ish multipliers works; this one needs no RNG,
#: is reproducible across devices, and makes a compensating cancellation between
#: two flipped decisions a measure-zero coincidence rather than a plausible one.
PHI_INV = 0.6180339887498949


def decision_weights(k):
    """``f64[k]`` distinct, **strictly positive** weights for a decision checksum.

    A plain ``sum`` of decision indices is not a fingerprint: swapping two
    particles' cells leaves it unchanged.  Weighting each slot by an irrational
    rotation makes that cancellation vanishingly unlikely while costing one
    ``mod``.

    **The ``+ 1`` and the ``+ 0.5`` are both load-bearing.**  An earlier version
    used ``mod(arange(k) * PHI_INV, 1.0)``, whose first entry is exactly ``0.0`` --
    so every decision taken in **cell 0** was multiplied by zero and vanished from
    the fingerprint.  In a 1-D reflected shock cell 0 is the cell against the wall,
    which is precisely where accept bits flip, and the guard reported "no flips"
    across a finite-difference estimate that was wrong by a factor of three
    (``346`` against a true ``127``).  A checksum with a zero weight is not a
    checksum; the offset keeps every weight in ``[0.5, 1.5)``.
    """
    return jnp.mod((jnp.arange(k, dtype=jnp.float64) + 1.0) * PHI_INV, 1.0) + 0.5


def _slot(carry, xs, n, table, sigma, isp, jsp, sp, n_part, score_mode="paper"):
    """One attempt slot, vectorised across all cells.  Body of the section-8.4 scan.

    **Draws nothing.**  Every random number it needs arrives through ``xs``, drawn
    in one batched call per stage before the scan (section 1 rule 3).  An earlier
    version called ``fold_in`` + ``randint``/``uniform`` inside this body -- about
    nine tiny kernels per slot -- which halved throughput and cost 7x in compile
    time.  What remains after the hoist is a ~40 us per-slot floor that is flat in
    ``A_max``; it is amortised across cells, so it only hurts when ``C`` is small
    (section 8.6).

    Module-level and bound with ``functools.partial`` rather than nested, per
    decision 14: the captured state is explicit and the body is independently
    testable.
    """
    v, logh, qmax, nviol, margin, chk = carry
    a, u_i, u_r, u_acc, u_chi, u_eps = xs
    active = (a < n) & (table.counts >= 2)  # bool[C]

    i, j = pair_from_draws(u_i, u_r, table.counts)
    # inactive cells address the SCRATCH ROW at index n_part, never a neighbour's
    # particles.  Both `where`s are essential: see 8.4.1.
    pi = jnp.where(active, table.order[table.starts + i], n_part)
    pj = jnp.where(active, table.order[table.starts + j], n_part)
    vi, vj = v[pi], v[pj]

    q = collision_rate(vi, vj, isp, jsp, sp)
    p = acceptance_prob(q, sigma)  # FIXED Sigma -- stale-max rule
    accept = active & (u_acc < p)  # FROZEN (bool)

    vi_n, vj_n = scatter(vi, vj, isp, jsp, sp, u_chi, u_eps)
    # `vi` and `vj` are both gathered from the SAME pre-scatter `v`, so for an
    # inactive slot where pi == pj the second `.set` would write stale data over
    # the first.  The scratch row is what makes that harmless.
    v = v.at[pi].set(jnp.where(accept[..., None], vi_n, vi))
    v = v.at[pj].set(jnp.where(accept[..., None], vj_n, vj))

    # BOTH partners get the SAME log h, and BOTH branches contribute (section 7.3).
    lh = jnp.where(active, log_h(q, sigma, accept), 0.0)
    if score_mode == "trajectory":
        logh = logh + jnp.sum(lh)
    else:
        logh = logh.at[pi].add(lh).at[pj].add(lh)

    # masked so an untested slot's meaningless q cannot inflate the next majorant
    qmax = jnp.where(active, jnp.maximum(qmax, q), qmax)
    nviol = nviol + jnp.sum(active & (q > sigma)).astype(jnp.int32)
    # margin = min over TESTED pairs of Sigma/q.  Target >= 2 (7.2.2/7.2.3).  q == 0
    # only below the safe_pow clamp, where the pair is degenerate and the ratio is
    # meaningless, so those slots are excluded rather than producing an inf.
    tested = active & (q > 0.0)
    q_safe = jnp.where(tested, q, 1.0)
    margin = jnp.minimum(margin, jnp.min(jnp.where(tested, sigma / q_safe, jnp.inf)))

    # Fingerprint of this slot's frozen decisions (pair choice and accept bit),
    # weighted per cell and per slot so a flip cannot cancel against another.
    # `stop_gradient` is belt-and-braces -- every input is already integer or
    # boolean -- and states the intent: this is bookkeeping, not physics.
    wc = decision_weights(table.counts.shape[0])
    wa = jnp.mod(a.astype(jnp.float64) * PHI_INV, 1.0) + 0.5
    chk = chk + wa * jnp.sum(
        wc
        * (
            pi.astype(jnp.float64)
            + 0.25 * pj.astype(jnp.float64)
            + 0.5 * accept
            + 0.125 * frame_is_well_defined(vi, vj)
        )
    )
    return (v, logh, qmax, nviol, margin, lax.stop_gradient(chk)), None


def collide(
    state: ParticleState,
    table,
    cstate: CollideState,
    grid,
    sp,
    params,
    static,
    cell_temp,
    step,
    base_key,
    score_mode="paper",
):
    """One collision step.  Returns ``(state, cstate, diag)``.

    Sequential within a cell -- slot ``a+1`` sees the velocities written by slot
    ``a``, so a particle can collide twice in a step, matching both reference codes
    -- while fully parallel across cells.  It is a ``scan``, so it is
    reverse-differentiable and rematerialisable.

    Spec: section 8.4.
    """
    if score_mode not in ("paper", "trajectory"):
        raise ValueError("score_mode must be paper or trajectory")
    n_part = static.N
    n_cells = num_cells(grid)
    volume = cell_volume(grid)
    isp, jsp = 0, 0  # single species (decision 6); shaped for (S, S) throughout

    # Stale-max rule (7.2): Sigma is read ONCE and held fixed for the whole loop.
    # The carried vremax is only a floored ratchet -- on its own it lags a
    # transient and fails the U3 gate (7.2.3).  The a-priori thermal majorant is
    # what supplies the 2-5x margin.
    sigma = lax.stop_gradient(
        jnp.maximum(
            cstate.vremax[:, 0, 0],
            thermal_majorant(
                cell_temp,
                sp.omega[isp, jsp],
                sp.tref[isp, jsp],
                sp.mr[isp, jsp],
                sp.prefactor[isp, jsp],
            ),
        )
    )

    k_cnt = key_for(base_key, step, STAGE_NTC_ROUND)
    n, logP = attempt_count(table.counts, sigma, volume, params.dt, params.fnum, k_cnt)

    # Section 1 rule 3: each stage draws ALL its randomness in ONE batched call,
    # shaped (A_max, C), consumed by the scan as `xs`.  This is a performance
    # requirement, not a style preference -- see `_slot`.
    A, C = static.A_max, table.counts.shape[0]
    npc_safe = jnp.maximum(table.counts, 2)
    k_pi, k_pr = random.split(key_for(base_key, step, STAGE_PAIR))
    k_chi, k_eps = random.split(key_for(base_key, step, STAGE_SCATTER))
    u_i = random.randint(k_pi, (A, C), 0, npc_safe)
    u_r = random.randint(k_pr, (A, C), 0, npc_safe - 1)
    u_acc = random.uniform(key_for(base_key, step, STAGE_ACCEPT), (A, C), dtype=jnp.float64)
    u_chi = random.uniform(k_chi, (A, C), dtype=jnp.float64)
    u_eps = random.uniform(k_eps, (A, C), dtype=jnp.float64)

    # v and logh carry an extra SCRATCH ROW at index N (8.4.1).
    v0 = jnp.concatenate([state.v, jnp.zeros((1, 3), dtype=state.v.dtype)])
    logh0 = jnp.concatenate([state.logh_acc, jnp.zeros((1,), dtype=state.logh_acc.dtype)])

    body = functools.partial(
        _slot,
        n=n,
        table=table,
        sigma=sigma,
        isp=isp,
        jsp=jsp,
        sp=sp,
        n_part=n_part,
        score_mode=score_mode,
    )
    init = (v0, logh0, sigma, jnp.int32(0), jnp.float64(jnp.inf), jnp.float64(0.0))
    xs = (jnp.arange(A), u_i, u_r, u_acc, u_chi, u_eps)
    (v, logh, qmax, nviol, margin, chk), _ = lax.scan(body, init, xs)

    # --- the COUNT score (8.1) ------------------------------------------------
    # logP is a PER-CELL term: the decision "how many pairs to attempt" applies to
    # every particle in that cell, so each of them carries it.  This is the j=1
    # branch of 7.3, and it is where the entire dt / fnum / cell_volume
    # sensitivity enters.  Omitting this line reproduces the revision-2 defect
    # exactly -- those three gradients read 0.0.
    if score_mode == "trajectory":
        logh = logh + jnp.sum(logP)
    else:
        logh = logh.at[:n_part].add(jnp.where(state.alive, logP[table.cell], 0.0))

    # Cell membership and the attempt count are decided once per step, outside the
    # slot scan, so they are folded in here rather than per slot.
    chk = chk + jnp.sum(decision_weights(n_part) * jnp.where(state.alive, table.cell, -1))
    chk = chk + jnp.sum(decision_weights(n.shape[0]) * n)
    diag = Diagnostics(
        attempts_clipped=jnp.sum(jnp.maximum(n - static.A_max, 0)),
        max_attempts=jnp.max(n),
        majorant_violations=nviol,
        min_majorant_margin=margin,
        decision_checksum=lax.stop_gradient(chk),
        wall_recrossed=jnp.array(0, jnp.int32),
        inflow_injected=jnp.array(0, jnp.int32),
        inflow_shortfall=jnp.array(0.0, jnp.float64),
    )
    new_state = ParticleState(
        x=state.x, v=v[:n_part], species=state.species, alive=state.alive, logh_acc=logh[:n_part]
    )
    vremax_next = jnp.maximum(cstate.vremax, qmax[:, None, None])
    del n_cells, C
    return new_state, CollideState(vremax=vremax_next), diag
