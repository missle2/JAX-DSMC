"""Singularity-safe primitives.

Spec: PSEUDOCODE.md section 2.

DSMC is full of expressions whose *value* is fine at a removable singularity but
whose *derivative* is ``NaN``. Because ``jnp.where`` evaluates both branches and
``NaN * 0 == NaN`` in the backward pass, the obvious guard does not work::

    # WRONG -- NaN gradient at d == 0 even though the primal is correct
    result = jnp.where(d > eps, num / d, fallback)

    # RIGHT -- make the argument safe BEFORE the singular operation
    d_safe = jnp.where(d > eps, d, 1.0)
    result = jnp.where(d > eps, num / d_safe, fallback)

Every function here uses that double-``where`` pattern.

A clamp that merely makes the gradient *finite* is not enough (section 2.3):
just above ``EPS = 1e-30``, ``d/dx sqrt(x) = 5e14``. Each function therefore
carries a stated derivative bound, asserted in ``tests/test_safe.py``:

===========================  ===============================================
function                     derivative bound
===========================  ===============================================
``safe_sqrt`` via norm       ``1.0``  (structural, independent of ``eps``)
``safe_sqrt`` bare           ``1 / (2 * sqrt(eps))``
``safe_pow`` (on ``vr2``)    ``2 * |p| * eps**(p - 0.5)`` w.r.t. ``|vr|``
``safe_log`` reject branch   ``1 / (Sigma - max q)``
===========================  ===============================================

All arguments of every function here are TRACED; ``axis`` is STATIC.
"""

import jax.numpy as jnp

from jax_dsmc.constants import EPS

__all__ = [
    "safe_div",
    "safe_log",
    "safe_norm",
    "safe_normalize",
    "safe_pow",
    "safe_recip_count",
    "safe_sqrt",
]


def safe_div(num, den, eps=EPS, fallback=0.0):
    """``num / den``, with a finite gradient where ``|den| <= eps``.

    Args:
        num: f64, any shape broadcastable against ``den``. TRACED.
        den: f64, any shape broadcastable against ``num``. TRACED.
        eps: f64 scalar magnitude below which ``den`` is treated as zero.
            TRACED (but normally a Python constant).
        fallback: f64 value returned where the guard fires. TRACED.

    Returns:
        f64, the broadcast shape of ``num`` and ``den``.

    Spec: section 2.2.
    """
    ok = jnp.abs(den) > eps
    den_safe = jnp.where(ok, den, 1.0)
    return jnp.where(ok, num / den_safe, fallback)


def safe_sqrt(x, eps=EPS):
    """``sqrt(x)`` for ``x >= 0``, with a finite gradient at ``x == 0``.

    The derivative of ``sqrt`` is ``1 / (2 sqrt(x))``, which diverges at 0, so
    the *argument* is clamped rather than the result. Bare, the derivative is
    bounded by ``1 / (2 sqrt(eps))`` -- large for the default ``eps``. That is
    acceptable only because every use in the spec reaches ``safe_sqrt`` through
    :func:`safe_norm`, whose composite derivative is structurally ``1.0``
    (section 2.3).

    Args:
        x: f64, any shape, assumed non-negative. TRACED.
        eps: f64 scalar clamp. TRACED (normally a Python constant).

    Returns:
        f64, same shape as ``x``; ``0.0`` where ``x <= eps``.

    Spec: section 2.2. Singularities: ``collide_vss.cpp:214`` (relative speed),
    ``collide_vss.cpp:417,427`` (``sqrt(1 - cosX**2)``).
    """
    ok = x > eps
    x_safe = jnp.where(ok, x, 1.0)
    return jnp.where(ok, jnp.sqrt(x_safe), 0.0)


def safe_norm(v, axis=-1, eps=EPS):
    """Euclidean norm of ``v`` along ``axis``, with a finite gradient at ``v == 0``.

    NOT ``jnp.sqrt(jnp.sum(v * v))`` -- that has a ``NaN`` gradient at the
    origin. The composite ``d||v|| / dv = v / ||v||`` has norm exactly ``1.0``
    wherever the guard does not fire and exactly ``0.0`` where it does, so this
    is bounded by ``1.0`` regardless of ``eps`` (section 2.3, measured across 14
    decades of ``|v|``).

    Args:
        v: f64, any shape. TRACED.
        axis: int, the axis to reduce. STATIC (it selects a shape).
        eps: f64 scalar clamp on ``sum(v * v)``. TRACED.

    Returns:
        f64 with ``axis`` removed from ``v.shape``.

    Spec: section 2.2.
    """
    return safe_sqrt(jnp.sum(v * v, axis=axis), eps)


def safe_pow(x, p, eps):
    """``x ** p`` for ``x >= 0`` and traced, possibly negative or fractional ``p``.

    ``eps`` is REQUIRED and positional: it must be chosen *relatively*, scaled
    to the physical quantity being raised. For the collision rate the caller
    passes ``eps = EPS_REL_VR2 * (2 * K_BOLTZ * tref / mr)`` -- a fixed fraction
    of the reference thermal energy scale, NOT an absolute ``1e-30``, which
    would leave ``d q / d|vr|`` unbounded at ~2.5e16 (section 2.3).

    The fallback is ``p``-dependent: ``x**0 == 1`` for every ``x``, including
    ``x == 0``. A blanket ``0.0`` is wrong exactly at ``omega == 1`` (Maxwell
    molecules), where ``q = prefactor * vr2**0`` must equal ``prefactor`` even
    for a coincident pair -- which is precisely Oracle 1's configuration
    (section 13.1).

    Computed as ``exp(p * log(x))`` so that a traced, non-integer, possibly
    negative ``p`` never reaches ``jnp.power``'s negative-base branch.

    Args:
        x: f64, any shape, assumed non-negative. TRACED.
        p: f64, broadcastable against ``x``. TRACED (this is ``1 - omega``).
        eps: f64 absolute clamp on ``x``. TRACED. No default -- see above.

    Returns:
        f64, the broadcast shape of ``x`` and ``p``.

    Spec: section 2.2. Source: ``collide_vss.cpp:222``.
    """
    ok = x > eps
    x_safe = jnp.where(ok, x, 1.0)
    fallback = jnp.where(p == 0.0, 1.0, 0.0)
    return jnp.where(ok, jnp.exp(p * jnp.log(x_safe)), fallback)


def safe_normalize(v, axis=-1, eps=EPS):
    """Unit vector along ``v``, returning zeros (not ``NaN``) at ``v == 0``.

    Args:
        v: f64, any shape. TRACED.
        axis: int, the axis holding the vector components. STATIC.
        eps: f64 scalar clamp, applied to both the norm-squared and the
            division. TRACED.

    Returns:
        f64, same shape as ``v``.

    Spec: section 2.2. Used by the VSS scatter frame,
    ``collide_vss.cpp:429-437``.
    """
    n = safe_norm(v, axis, eps)
    return safe_div(v, jnp.expand_dims(n, axis), eps, fallback=0.0)


def safe_log(p, eps=EPS):
    """``log(p)`` for score terms, where ``p in (0, 1]`` can underflow to 0.

    The clamp bounds the *value*; it does not bound the derivative in any useful
    way (``1 / eps`` is ``1e30``). What actually bounds the score gradient is the
    majorant gap: on the reject branch ``d log(1 - q/Sigma) / dq = -1/(Sigma - q)``,
    controlled by keeping ``Sigma`` clear of ``max q`` (target a 2-5x safety
    factor, section 7.2.2), not by ``eps``.

    Args:
        p: f64, any shape, assumed non-negative. TRACED.
        eps: f64 scalar floor. TRACED.

    Returns:
        f64, same shape as ``p``.

    Spec: section 2.2. Source: [P2] eq. ``back_eta``.
    """
    return jnp.log(jnp.maximum(p, eps))


def safe_recip_count(c):
    """``1 / c`` for integer occupancy counts, ``0`` for empty cells.

    Args:
        c: i32, any shape -- particle counts, ``>= 0``. FROZEN (an integer
            decision, section 0.5); carries no tangent.

    Returns:
        f64, same shape as ``c``.

    Spec: section 2.2. Used by the moment normalisation in ``sample.py``
    (section 10).
    """
    cf = jnp.asarray(c, dtype=jnp.float64)
    return jnp.where(cf > 0.0, 1.0 / jnp.maximum(cf, 1.0), 0.0)
