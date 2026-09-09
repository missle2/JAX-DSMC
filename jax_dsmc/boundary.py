"""Boundary conditions (spec section 9.1-9.3).

Forward-only in the sense that matters: the periodic and specular adjoints are
*free* -- they are the VJPs of a translation and a mirror, and reverse-mode AD
produces [P3]'s ``C = diag([-1, 1, 1])`` with no special handling.  Diffuse and
thermal walls are section 9.4 and are deferred to M4 (forward) / M7 (adjoint),
because the *event* of hitting a wall is a discontinuous function of position.

Only the first ``D`` components of ``v`` participate: velocities are always
3-component whatever the spatial dimension (section 0.2).
"""

from __future__ import annotations

import enum

import jax
import jax.numpy as jnp

from jax_dsmc.state import GridSpec

jax.config.update("jax_enable_x64", True)  # section 0.3: f64 throughout


class BC(enum.Enum):
    """Which boundary rule to apply.  STATIC -- it selects a code path."""

    PERIODIC = "periodic"
    SPECULAR = "specular"
    OUTFLOW = "outflow"


def apply_periodic(x_new, v, grid: GridSpec):
    """``domain.cpp:301-330``.  Adjoint: the identity, matching [P3] section 3.1.

    ``mod`` is differentiable a.e. with derivative 1, which is correct -- the wrap
    is a translation, so it must not change a cotangent.
    """
    L = grid.hi - grid.lo
    return grid.lo + jnp.mod(x_new - grid.lo, L), v


def apply_specular(x_new, v, grid: GridSpec):
    """Closed-form multi-bounce fold (section 9.2).

    ``domain.cpp:337-357`` mirrors once and assumes at most one crossing; [P3]
    section 3.2 makes the same assumption explicitly.  A triangle-wave fold handles
    **any** number of bounces in one branch-free expression, at no extra cost.

    Adjoint: ``dx/dx_new == dv/dv == sgn``, i.e. [P3] eq. ``C_ki`` applied to both
    the position and velocity adjoints -- which AD produces from the code below.
    """
    D = x_new.shape[-1]
    L = grid.hi - grid.lo
    y = (x_new - grid.lo) / L
    m = jnp.floor(y)  # fold index, FROZEN
    frac = y - m
    even = jnp.mod(m, 2.0) == 0.0
    pos = jnp.where(even, frac, 1.0 - frac)
    sgn = jnp.where(even, 1.0, -1.0)
    x = grid.lo + pos * L
    v = v.at[:, :D].multiply(sgn)
    return x, v


def apply_outflow(x_new, v, alive, grid: GridSpec):
    """``domain.cpp:294-295``.  Dead particles keep their slots (static shapes).

    Through M4 ``alive`` is monotone non-increasing, so a dead slot is never
    touched again.  **From M5 that changes**: inflow refills dead slots, and any
    path that sets ``alive[i] = True`` for a previously dead ``i`` must zero
    ``logh_acc[i]`` in the same update, or the new particle inherits the score
    history of whatever occupied that index (section 9.3, gap U6).
    """
    inside = jnp.all((x_new >= grid.lo) & (x_new <= grid.hi), axis=-1)
    return x_new, v, alive & inside


def fold_index(x_new, grid: GridSpec, bc: BC):
    """The **frozen integer** that says which side of a boundary discontinuity we sit on.

    ``i32[N, D]``.  For :data:`BC.SPECULAR` this is exactly the ``m`` of
    :func:`apply_specular` -- and its *parity* flips the sign of ``v``, so a particle
    that changes fold index between two evaluations has had its velocity negated.
    That is a discontinuity of the map every bit as sharp as an accept-bit flip, and
    it is **invisible in the cell index**: a particle just inside the wall and one
    just outside both fold into cell 0.

    This function exists because that hole was found the hard way.  A
    finite-difference check whose flip guard covered cells, attempt counts, pairs
    and accept bits but *not* this reported "no flips" while FD and AD disagreed by
    a factor of three (``370.5`` vs ``127.2``).

    For :data:`BC.PERIODIC` the same quantity is the wrap count.  A wrap is a
    translation on the torus and does not break continuity, but it is included
    anyway: it costs one ``floor`` and it can only make the guard stricter.
    """
    if bc is BC.OUTFLOW:
        inside = jnp.all((x_new >= grid.lo) & (x_new <= grid.hi), axis=-1)
        return jnp.broadcast_to(inside[:, None], x_new.shape).astype(jnp.int32)
    return jnp.floor((x_new - grid.lo) / (grid.hi - grid.lo)).astype(jnp.int32)


def apply(x_new, v, alive, grid: GridSpec, bc: BC):
    """Dispatch on ``bc``.  ``bc`` is STATIC -- a Python enum, never traced."""
    if bc is BC.PERIODIC:
        x, v = apply_periodic(x_new, v, grid)
        return x, v, alive
    if bc is BC.SPECULAR:
        x, v = apply_specular(x_new, v, grid)
        return x, v, alive
    if bc is BC.OUTFLOW:
        return apply_outflow(x_new, v, alive, grid)
    raise ValueError(f"unknown boundary condition: {bc!r}")
