"""Particle-recursion and full-trajectory score objectives.

``surrogate`` retains the terminal-label P2 comparison estimator. Agreement with
that recursion is not proof of an expected-gradient identity after arbitrary
partner changes. ``trajectory_surrogate`` scores a complete scalar objective;
its log probability must count each event exactly once. Neither repairs frozen
geometric support or integer-count support changes. P3 inflow instead uses the
separate distributional solver in ``ensemble_inflow``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax

from jax_dsmc.simulate import run
from jax_dsmc.state import ParticleState

jax.config.update("jax_enable_x64", True)  # section 0.3: f64 throughout


# --- the magic box ---------------------------------------------------------


def magic(logh):
    """DiCE operator: value identically ``1``, gradient identically ``d logh``.

    Sometimes written ``(x)_|``.  ``exp(L - stop_gradient(L))`` evaluates to
    ``exp(0) == 1`` while carrying ``exp(...) * dL == dL`` through reverse mode.

    Preferred over the simpler ``sum(phi) + sum(stop_gradient(phi) * L)`` for two
    reasons: the primal value stays *equal* to ``J`` so the objective can be
    reported directly, and higher derivatives stay correct (the naive form is right
    only to first order).

    Spec: section 11.1.
    """
    return jnp.exp(logh - lax.stop_gradient(logh))


def alive_weights(alive):
    """``1/N_alive`` on live slots, ``0`` on dead ones.  ``f64[N]``.

    The count is an integer reduction over a boolean, so no gradient path runs
    through it -- ``N`` is a frozen decision like every other one.  Clamped away
    from zero so an empty state returns ``0.0`` rather than ``nan``.
    """
    n = jnp.maximum(jnp.sum(alive), 1).astype(jnp.float64)
    return jnp.where(alive, 1.0, 0.0) / n


def objective(phi, alive, rho=1.0):
    """``J = (rho/N) sum_i phi_i`` over live particles.  **Pathwise gradient only.**

    Args:
        phi: ``f64[N]`` per-particle objective values.
        alive: ``bool[N]``.
        rho: the total mass/number the ``N`` particles represent ([P2]
            eq. ``obj1``).  Constant for a fixed particle count; carried
            explicitly so ``J`` stays dimensionally correct as ``N`` varies.

    This is the derivative of *one realisation* at frozen randomness.  Use it when
    the comparison is a fixed-key finite difference; use :func:`surrogate` when the
    comparison is ``dE[J]/dtheta``.
    """
    return rho * jnp.sum(alive_weights(alive) * phi)


def surrogate(phi, logh_acc, alive, rho=1.0, baseline=None):
    """Terminal-label P2 surrogate, with the same primal value as ``objective``.

    Each collision deposits its log probability on both participating labels.
    This reproduces the tested paper recursion; it is not a general expected
    derivative for arbitrary later partner changes or population-normalized
    open-boundary rewards. ``None`` or a constant independent baseline is allowed.
    A same-trajectory reward mean is biased and rejected.
    """
    w = alive_weights(alive)
    b = _resolve_baseline(baseline, phi, w)
    return rho * jnp.sum(w * (magic(logh_acc) * (phi - b) + b))


def _resolve_baseline(baseline, phi, w):
    """``b`` as a traced scalar, ``stop_gradient``-ed so it cannot carry a path."""
    if baseline is None:
        return 0.0
    if isinstance(baseline, str):
        raise ValueError("Use an independent baseline; the same-trajectory mean is biased")
    return lax.stop_gradient(jnp.float64(baseline))


def trajectory_surrogate(value, log_probability, baseline=0.0):
    """Full-trajectory likelihood estimator of a scalar expected objective.

    Each random event must occur exactly once in ``log_probability``. The baseline
    must be independent of the scored outcomes (or measurable before those events).
    This covers nonlinear/population-normalized rewards, unlike terminal-label
    attribution. It does not supply missing geometric support derivatives.
    """
    b = _resolve_baseline(baseline, None, None)
    return magic(log_probability) * (value - b) + b


# --- objective functionals -------------------------------------------------
#
# Each has signature ``(x, v) -> f64[N]`` and is bound with ``functools.partial``
# rather than closed over (decision 14: no nested function definitions).


def phi_power(x, v, component=0, power=2):
    """``v_l^p`` -- the objective both papers use ([P1] section 6: ``v_l^2``, ``v_l^4``)."""
    del x
    return v[:, component] ** power


def phi_kinetic(x, v):
    """``|v|^2`` -- total kinetic energy per unit mass, up to a factor 2."""
    del x
    return jnp.sum(v * v, axis=-1)


def phi_windowed(x, v, inner, axis=0, center=0.0, width=1.0):
    """``inner(x, v)`` weighted by a Gaussian window in ``x[:, axis]``.

    The window is what makes a *spatial* objective differentiable in position.  A
    hard indicator ``lo < x < hi`` has zero derivative almost everywhere and a
    delta at the edges, so it would freeze the transport half of the gradient out
    exactly the way ``floor`` freezes the count (section 8.1).  A Gaussian is
    smooth, strictly positive, and its width is a physical choice -- for a shock,
    a few mean free paths.
    """
    z = (x[:, axis] - center) / width
    return inner(x, v) * jnp.exp(-z * z)


# --- entry points ----------------------------------------------------------


def dice_reward(state: ParticleState, table, phi, rho=1.0, baseline=None, score_mode="paper"):
    """One timestep's contribution to a **windowed** objective, causal DiCE form.

    Signature matches what :func:`jax_dsmc.simulate.run` expects of its ``reward``
    argument.  Bind ``phi``/``rho``/``baseline`` with ``functools.partial``.

    ``state.logh_acc`` *is already* the running prefix ``L_{<=k}`` -- it is
    accumulated in place as the scan proceeds -- so reading it here, inside the
    loop, gives ``sum_k magic(L_{<=k}) r_k`` with no extra carry.  Reading it once
    at the end instead would score each reward against collisions that happened
    **after** it was emitted: still unbiased, but pure added noise (section 11.5,
    measured at ~3x variance and flat in the horizon).
    """
    del table
    if score_mode == "trajectory":
        value = objective(phi(state.x, state.v), state.alive, rho)
        return trajectory_surrogate(value, state.logh_acc[0], baseline)
    return surrogate(phi(state.x, state.v), state.logh_acc, state.alive, rho, baseline)


def terminal_objective(
    state,
    cstate,
    grid,
    sp,
    params,
    static,
    base_key,
    phi,
    rho=1.0,
    bc=None,
    collide_only=False,
    baseline=None,
    scored=True,
    score_mode="trajectory",
):
    """Run the solve and return ``J = (rho/N) sum_i phi(x_M, v_M)``.

    ``scored`` selects which of the two gradients this function carries:
    ``True`` uses full trajectory likelihood by default; ``score_mode="paper"``
    selects the terminal-label comparison estimator. Count probabilities are valid
    only within a fixed rounding-support interval; mesh memberships remain frozen.
    ``False`` gives :func:`objective` (pathwise only, what a fixed-key finite
    difference measures).  The **value is identical either way**.

    Returns ``(J, (state, cstate, diag))``.
    """
    kw = {} if bc is None else {"bc": bc}
    (st, cs), (_, diag) = run(
        state,
        cstate,
        grid,
        sp,
        params,
        static,
        base_key,
        collide_only=collide_only,
        emit_moments="none",
        score_mode=score_mode,
        **kw,
    )
    vals = phi(st.x, st.v)
    j = (
        (
            trajectory_surrogate(objective(vals, st.alive, rho), st.logh_acc[0], baseline)
            if score_mode == "trajectory"
            else surrogate(vals, st.logh_acc, st.alive, rho, baseline)
        )
        if scored
        else objective(vals, st.alive, rho)
    )
    return j, (st, cs, diag)


def windowed_objective(
    state,
    cstate,
    grid,
    sp,
    params,
    static,
    base_key,
    phi,
    rho=1.0,
    bc=None,
    collide_only=False,
    baseline=None,
    score_mode="trajectory",
):
    """Run the solve accumulating ``J = sum_k r(x_k, v_k)`` in the causal DiCE form.

    The time-averaged objectives that actually matter -- integrated drag, a moment
    averaged over a sampling window -- are of this shape.  Section 11.5.

    Returns ``(J, (state, cstate, diag))``.
    """
    kw = {} if bc is None else {"bc": bc}
    reward = functools.partial(
        dice_reward, phi=phi, rho=rho, baseline=baseline, score_mode=score_mode
    )
    (st, cs, j), (_, diag) = run(
        state,
        cstate,
        grid,
        sp,
        params,
        static,
        base_key,
        collide_only=collide_only,
        emit_moments="none",
        reward=reward,
        score_mode=score_mode,
        **kw,
    )
    return j, (st, cs, diag)
