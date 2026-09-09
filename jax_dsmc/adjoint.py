"""The surrogate objective, and the gradient entry points (spec section 11).

This module is small on purpose.  **No hand-written backward pass exists in the
production path**: the adjoint recursions of [P1] Algorithm 2 and [P2] Algorithm 3
are what reverse-mode AD already computes for the forward solve, provided two
things are true, and both are properties of code that lives elsewhere:

1. the randomness is *frozen* -- keys come from ``(base_key, step, stage)`` and
   never from state, so a perturbed run consumes an identical stream (section 1);
2. every stochastic branch deposits a differentiable ``log h`` into
   ``ParticleState.logh_acc`` (sections 7.3 and 8.1).

Given those, the entire content of the adjoint is the **DiCE magic box** applied to
that accumulator.  ``magic(L)`` is identically 1 in value and ``dL`` in gradient, so

    J~ = (rho/N) sum_i magic(L_i) phi_i

has the same *value* as the plain objective and a gradient that is
``pathwise + score``.  Section 11.2 shows term by term that this is [P2]
eq. ``gamma_rules``; ``tests/test_adjoint_oracle.py`` asserts it against both
papers' hand-coded recursions.

**Two gradients, and they are not the same quantity.**  :func:`objective` and
:func:`surrogate` return identical values and *different* derivatives, and mixing
them up is the most likely way to misread a verification run:

===================  ==========================================================
``objective``        ``d/dtheta`` of *this realisation* at frozen randomness.
                     Matches a fixed-key finite difference to round-off, and is
                     what a step-function-free FD sweep measures.
``surrogate``        ``d/dtheta`` of ``E[J]``.  Includes the score terms, so it
                     sees parameters that reach ``J`` **only** through frozen
                     decisions -- ``diam``, ``omega``, ``tref``, ``dt``,
                     ``fnum``.  A fixed-key FD reports exactly ``0.0`` for those
                     (section 13.4), so it must be checked against an *ensemble*
                     FD instead.
===================  ==========================================================
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
    """``J~``, the section-11.1 surrogate.  Value ``== J``, gradient ``dE[J]``.

    Args:
        phi: ``f64[N]`` per-particle objective values.
        logh_acc: ``f64[N]`` the score accumulator ``L``, filled by ``collide``.
        alive: ``bool[N]``.
        rho: as in :func:`objective`.
        baseline: reward baseline ``b``.  ``None`` reproduces the papers'
            estimator exactly (``b = 0``); ``"mean"`` uses the live-particle mean
            of ``phi``; a float is used as given.

    The gradient is ``(rho/N) sum_i [grad(phi_i) + (phi_i - b) grad(L_i)]``.  The
    first term is the pathwise ``D`` recursion of [P2] eq. ``back_D``; the second
    is ``eta``, because ``L_i`` and ``L_i1`` each accumulated the *same*
    ``log h_i^k``, so the coefficient of ``d log h/d v_i`` is ``phi_i + phi_i1`` --
    exactly [P2] eq. ``back_eta``.

    **The baseline is value-neutral and bias-free**: ``magic == 1`` in value so the
    ``- b`` and ``+ b`` cancel, and ``E[grad L] == 0`` so subtracting a constant
    leaves the estimator unbiased.  Section 11.4 measures **84x** variance
    reduction from ``b = mean(phi)`` on the count score, and calls it mandatory
    there rather than optional.
    """
    w = alive_weights(alive)
    b = _resolve_baseline(baseline, phi, w)
    return rho * jnp.sum(w * (magic(logh_acc) * (phi - b) + b))


def _resolve_baseline(baseline, phi, w):
    """``b`` as a traced scalar, ``stop_gradient``-ed so it cannot carry a path."""
    if baseline is None:
        return 0.0
    if baseline == "mean":
        return lax.stop_gradient(jnp.sum(w * phi))
    return jnp.float64(baseline)


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


def dice_reward(state: ParticleState, table, phi, rho=1.0, baseline=None):
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
):
    """Run the solve and return ``J = (rho/N) sum_i phi(x_M, v_M)``.

    ``scored`` selects which of the two gradients this function carries:
    ``True`` gives :func:`surrogate` (``dE[J]/dtheta``, the production estimator),
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
        **kw,
    )
    vals = phi(st.x, st.v)
    j = (
        surrogate(vals, st.logh_acc, st.alive, rho, baseline)
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
):
    """Run the solve accumulating ``J = sum_k r(x_k, v_k)`` in the causal DiCE form.

    The time-averaged objectives that actually matter -- integrated drag, a moment
    averaged over a sampling window -- are of this shape.  Section 11.5.

    Returns ``(J, (state, cstate, diag))``.
    """
    kw = {} if bc is None else {"bc": bc}
    reward = functools.partial(dice_reward, phi=phi, rho=rho, baseline=baseline)
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
        **kw,
    )
    return j, (st, cs, diag)
