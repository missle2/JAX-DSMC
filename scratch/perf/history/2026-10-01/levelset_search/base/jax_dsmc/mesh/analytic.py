"""Analytic circle / sphere bodies: exact hit times, normals and measures.

The polygonal machinery of :mod:`jax_dsmc.mesh.geometry` approximates a curved body by
chords; this module is the exact reference it is validated against (and the adjoint
reference, since the hit time is a smooth function of ``centre`` and ``radius``).

Conventions are those of :mod:`jax_dsmc.mesh.types`: the body is solid, its normal
points out of the body into the flow, and a hit only counts from the flow side
(``v . n < 0`` at the hit point).  Every function is pure and jit-able; ``dim`` is a
Python int (a shape), everything else is traced.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jax_dsmc.mesh.types import AnalyticBody

jax.config.update("jax_enable_x64", True)


def hit_body(x: jax.Array, v: jax.Array, body: AnalyticBody, exclude_body: jax.Array) -> jax.Array:
    """Time along the ray ``x + t v`` to the body surface from outside, ``f64[N]``.

    Args:
        x: ``f64[N, D]`` positions.
        v: ``f64[N, 3]`` or ``f64[N, D]`` velocities; only the first ``D`` components
            are used.
        body: the circle / sphere.
        exclude_body: ``bool[N]`` -- force ``inf`` (the element-``exclude`` rule of
            :func:`jax_dsmc.mesh.geometry.first_hit`; a particle that has just left the
            surface has ``v . n > 0`` and never re-hits anyway, the flag exists for
            symmetry with the polygonal path).

    Returns:
        ``f64[N]`` -- the smaller root ``t >= 0`` of ``|x + t v - c|^2 = r^2``, which
        is the hit with ``v . n < 0``; ``inf`` when the ray misses, starts inside the
        body, or moves away from it.  A particle exactly on the surface moving inward
        hits at ``t = 0`` (inclusive, as the polygonal test).
    """
    d = x.shape[1]
    vd = v[:, :d]
    rel = x - body.centre
    a = jnp.sum(vd * vd, axis=1)
    b = jnp.sum(rel * vd, axis=1)
    c = jnp.sum(rel * rel, axis=1) - body.radius * body.radius
    disc = b * b - a * c
    safe_a = jnp.where(a > 0.0, a, 1.0)
    # a masked sqrt(0) would still send NaN through the mask's gradient (0 * inf)
    root = jnp.sqrt(jnp.where(disc > 0.0, disc, 1.0))
    root = jnp.where(disc > 0.0, root, 0.0)
    if body.grazing > 0.0:
        # d root = d disc / (2 root) and root = |v| r |cos(incidence)|: the grazing
        # regularisation clips that denominator in the tangent only (primal exact)
        floor = 2.0 * body.grazing * jnp.sqrt(safe_a) * body.radius
        root = jax.lax.stop_gradient(root) + _regularised_half_quotient(disc, root, floor)
    t = (-b - root) / safe_a
    # `c >= 0` (outside or on the surface) and `b < 0` (moving toward the centre) are
    # what make the smaller root the from-outside hit: the surface is then crossed with
    # `v . n < 0`.  A particle on the surface (`c == 0`) moving inward has `t == 0`.
    ok = (a > 0.0) & (disc >= 0.0) & (c >= 0.0) & (b < 0.0) & (t >= 0.0) & ~exclude_body
    return jnp.where(ok, t, jnp.inf)


def _regularised_half_quotient(disc, root, floor):
    """Zero primal; tangent ``d disc / max(2 root, floor)`` (see
    :func:`jax_dsmc.mesh.levelset.regularised_quotient`)."""
    den = jnp.maximum(2.0 * jax.lax.stop_gradient(root), floor)
    return (disc - jax.lax.stop_gradient(disc)) / den


def normal(body: AnalyticBody, x_hit: jax.Array) -> jax.Array:
    """Unit outward normal ``(x_hit - c) / r`` at points on the surface, ``f64[N, D]``."""
    return (x_hit - body.centre) / body.radius


def inside(body: AnalyticBody, x: jax.Array) -> jax.Array:
    """``|x - c| < r``, ``bool[N]`` (points on the surface count as outside)."""
    rel = x - body.centre
    return jnp.sum(rel * rel, axis=1) < body.radius * body.radius


def volume(body: AnalyticBody, dim: int) -> jax.Array:
    """``pi r^2`` (``dim == 2``) or ``4 pi r^3 / 3`` (``dim == 3``); TRACED in ``radius``."""
    r = body.radius
    if dim == 2:
        return math.pi * r * r
    if dim == 3:
        return 4.0 * math.pi * r * r * r / 3.0
    raise ValueError(f"analytic bodies exist for dim in (2, 3), got {dim}")


def area(body: AnalyticBody, dim: int) -> jax.Array:
    """``2 pi r`` (``dim == 2``) or ``4 pi r^2`` (``dim == 3``); TRACED in ``radius``."""
    r = body.radius
    if dim == 2:
        return 2.0 * math.pi * r
    if dim == 3:
        return 4.0 * math.pi * r * r
    raise ValueError(f"analytic bodies exist for dim in (2, 3), got {dim}")
