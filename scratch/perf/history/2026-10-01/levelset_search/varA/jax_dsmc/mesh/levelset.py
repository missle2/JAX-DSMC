"""C1 level-set bodies: ``phi(x) < 0`` with ``phi`` a uniform cubic B-spline.

The shape representation of the optimiser (PSEUDOCODE section 4.4).  A flat-normal
polygon's shape derivative carries a delta at every vertex (the normal jumps there: under
uniform scaling that corner term is the whole ``dF/dr``); on a C1 surface the hit normal
varies smoothly with the shape and the pathwise derivative of the replayed wall map is
complete for specular walls.  A C0 (multilinear) interpolant would only move the jumps to
the level-set cell faces, so the cubic B-spline (C2 in ``phi``, C1 in the normal) is the
minimum.

Conventions are those of :mod:`jax_dsmc.mesh.types`: the body is ``phi < 0``, the normal
``grad phi / |grad phi|`` points into the flow, a hit only counts from the flow side.  The
surface id is ``2D`` and the model row ``2D`` (model 0), exactly as the analytic body, so
the tracker, tallies, kernel scores and the replayed shape VJP (``shape.geometry`` returns
``coef``) reuse the analytic path.

Hit search (``hit_body``): along ``x + s v_hat``, ``s in [0, L]`` with ``L = min(reach,
|v| remaining)`` and ``reach = n_sample h / 2``, the first ``+ -> -`` sign change between
consecutive samples brackets the entry root; ``n_bisect`` bisections; then **one Newton
step from the bracket, the bracket search being a function of ``stop_gradient`` inputs**.
The bisection's ``where`` selections carry no derivative at all -- the Newton step is what
restores the implicit-function derivative ``dt/dc_j = -B_j(x_hit) / (v . grad phi)`` (and
``dt/dx``, ``dt/dv``); freezing the search's *inputs* (not only its output) keeps reverse
mode from tracing the two loops at all.  There is no
``exclude``: after a reflection ``phi`` increases along ``v_out``, so the reflection point
is never a ``+ -> -`` crossing; a start on the surface (``|phi| <= tol``) moving inward
hits at ``t = 0``; a start deeper inside returns ``inf`` and the tracker's post-step inside
check counts it.  A grazing dip shorter than the sample spacing ``h / 2`` is missed
(tunnelling); it is counted by the LS0 grazing family, not hidden.

Every function is pure and jit-able; ``dim`` and the coefficient shape are static.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from jax_dsmc.mesh.types import LevelSet

jax.config.update("jax_enable_x64", True)

SURFACE_TOL = 1e-8
"""``|phi| <= SURFACE_TOL * h`` counts as *on* the surface (the ``t = 0`` hit rule)."""


# ---------------------------------------------------------------------------
# B-spline evaluation.
# ---------------------------------------------------------------------------


def _basis(f: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Cubic B-spline weights and their ``f``-derivatives for the four coefficients
    ``i-1, i, i+1, i+2`` at fractional position ``f in [0, 1)``: ``(w, dw)``, each ``[..., 4]``."""
    g = 1.0 - f
    w = jnp.stack(
        [
            g * g * g / 6.0,
            (3.0 * f * f * f - 6.0 * f * f + 4.0) / 6.0,
            (-3.0 * f * f * f + 3.0 * f * f + 3.0 * f + 1.0) / 6.0,
            f * f * f / 6.0,
        ],
        axis=-1,
    )
    dw = jnp.stack(
        [
            -g * g / 2.0,
            (3.0 * f * f - 4.0 * f) / 2.0,
            (-3.0 * f * f + 2.0 * f + 1.0) / 2.0,
            f * f / 2.0,
        ],
        axis=-1,
    )
    return w, dw


def _block(ls: LevelSet, x: jax.Array):
    """The ``4^D`` coefficient block of every point and the per-axis weights.

    Returns ``(block f64[N, 4, ..., 4], w list[D] of f64[N, 4], dw list[D] of f64[N, 4])``.
    Indices are clipped to the coefficient grid, so a point outside the valid domain
    ``u in [1, m - 2]`` evaluates *something* rather than erroring; the constructor
    checks the simulation box against that domain.
    """
    d = ls.dim
    u = (x - ls.lo) / ls.h
    i = jnp.floor(u)
    f = u - i
    i = i.astype(jnp.int32)
    w, dw = _basis(f)  # [N, D, 4]
    offsets = jnp.arange(-1, 3, dtype=jnp.int32)
    index = []
    for axis in range(d):
        m = ls.coef.shape[axis]
        idx = jnp.clip(i[:, axis, None] + offsets[None, :], 0, m - 1)  # [N, 4]
        shape = [x.shape[0]] + [1] * d
        shape[axis + 1] = 4
        index.append(idx.reshape(shape))
    block = ls.coef[tuple(index)]
    return block, [w[:, axis] for axis in range(d)], [dw[:, axis] for axis in range(d)]


def _contract(block: jax.Array, weights: list[jax.Array]) -> jax.Array:
    """``sum_k prod_axis weights[axis][k_axis] block[..., k]`` -> ``f64[N]``."""
    out = block
    for w in weights:
        # contract the leading local axis; `w` is [N, 4]
        out = jnp.sum(out * w.reshape(w.shape + (1,) * (out.ndim - 2)), axis=1)
    return out


def value_and_grad(ls: LevelSet, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``phi`` and ``grad phi`` at ``x`` (``f64[N, D]``): ``(f64[N], f64[N, D])``."""
    block, w, dw = _block(ls, x)
    phi = _contract(block, w)
    grads = []
    for axis in range(ls.dim):
        weights = list(w)
        weights[axis] = dw[axis]
        grads.append(_contract(block, weights) / ls.h)
    return phi, jnp.stack(grads, axis=1)


def value(ls: LevelSet, x: jax.Array) -> jax.Array:
    """``phi(x)``, ``f64[N]``."""
    block, w, _dw = _block(ls, x)
    return _contract(block, w)


def normal(ls: LevelSet, x_hit: jax.Array) -> jax.Array:
    """Unit normal into the flow ``grad phi / |grad phi|`` at ``x_hit``, ``f64[N, D]``."""
    _phi, g = value_and_grad(ls, x_hit)
    return g / jnp.maximum(jnp.linalg.norm(g, axis=1, keepdims=True), 1e-300)


# ---------------------------------------------------------------------------
# Ray-surface hit.
# ---------------------------------------------------------------------------


def _sample_step(i, carry, ls, x, vhat, ds):
    """One sample of the bracket search (``fori_loop`` body): the first ``+ -> -`` crossing."""
    found, s_lo, s_hi, phi_prev = carry
    s = i * ds
    phi = value(ls, x + s[:, None] * vhat)
    cross = ~found & (phi_prev > 0.0) & (phi <= 0.0)
    s_lo = jnp.where(cross, s - ds, s_lo)
    s_hi = jnp.where(cross, s, s_hi)
    return found | cross, s_lo, s_hi, phi


def _bisect_step(_i, carry, ls, x, vhat):
    """One bisection of the bracket (``fori_loop`` body); ``phi(s_lo) > 0 >= phi(s_hi)``."""
    s_lo, s_hi = carry
    mid = 0.5 * (s_lo + s_hi)
    phi = value(ls, x + mid[:, None] * vhat)
    positive = phi > 0.0
    return jnp.where(positive, mid, s_lo), jnp.where(positive, s_hi, mid)


def _bracket(ls, x, vhat, ds, phi0):
    """Sampled ``+ -> -`` bracket and its bisection: ``(found bool[N], s_hi f64[N])``."""
    n = x.shape[0]
    carry = (jnp.zeros(n, bool), jnp.zeros(n), jnp.zeros(n), phi0)
    found, s_lo, s_hi, _ = jax.lax.fori_loop(
        1, ls.n_sample + 1, functools.partial(_sample_step, ls=ls, x=x, vhat=vhat, ds=ds), carry
    )
    _s_lo, s_hi = jax.lax.fori_loop(
        0, ls.n_bisect, functools.partial(_bisect_step, ls=ls, x=x, vhat=vhat), (s_lo, s_hi)
    )
    return found, s_hi


NEAR_MIN = 2048
"""Rays per :func:`hit_body` call below which the near rays are not compacted (the search
is skipped when no ray is near, else run over all of them)."""
NEAR_MARGIN = 1e-6
"""Relative slack of the near-ray test (:func:`_near`) over the round-off of ``phi``."""


def gradient_bound(ls: LevelSet) -> jax.Array:
    """An upper bound on ``|grad phi|`` over all of space: ``d_a phi`` is a convex
    combination of the coefficient differences along ``a`` over ``h`` (the derivative of a
    cubic B-spline is a quadratic one on the differences), so ``|d_a phi| <= max |Delta_a
    c| / h``.  It holds outside the lattice too: :func:`_block` clips every index, i.e.
    evaluates the spline of the constant extension of ``coef``, whose differences are a
    subset of these (and zeros)."""
    per = [jnp.max(jnp.abs(jnp.diff(ls.coef, axis=a))) for a in range(ls.dim)]
    return jnp.sqrt(sum(p * p for p in per)) / ls.h


LOCAL_RADIUS = 3
"""Half-width (in lattice cells) of the windows of :func:`local_gradient_bounds`."""


def local_gradient_bounds(ls: LevelSet) -> jax.Array:
    """``f64[m_1, ..., m_D]``: entry ``i`` bounds ``|grad phi|`` on every segment of length
    ``<= h`` that starts in lattice cell ``i`` (``floor((x - lo) / h)``, clipped to the
    lattice).  Such a segment stays in cells ``i - 1 .. i + 1``, where ``d_a phi`` is a
    convex combination of ``Delta_a c`` at indices ``i - 2 .. i + 2`` along ``a`` and ``i -
    2 .. i + 3`` across: the max of ``|Delta_a c|`` over the ``LOCAL_RADIUS = 3`` window
    covers them (beyond the lattice the clipped evaluation's differences are those of the
    edge cells, or zero).  Far tighter than :func:`gradient_bound` away from the few
    coefficients a shape's kinks inflate."""
    d = ls.dim
    r = LOCAL_RADIUS
    total = jnp.zeros(ls.coef.shape)
    for a in range(d):
        diff = jnp.abs(jnp.diff(ls.coef, axis=a))
        diff = jnp.pad(diff, [(0, 1) if ax == a else (0, 0) for ax in range(d)])
        pooled = jax.lax.reduce_window(
            diff, -jnp.inf, jax.lax.max, (2 * r + 1,) * d, (1,) * d, [(r, r)] * d
        )
        total = total + pooled * pooled
    return jnp.sqrt(total) / ls.h


def _near(phi_start, length, ls, x):
    """``bool[N]``: can the ray's search find a ``+ -> -`` crossing at all?  Every sample
    lies within ``length`` of the start, where ``phi >= phi_start - G length`` with ``G`` a
    bound on ``|grad phi|`` there (:func:`local_gradient_bounds` at the start's lattice cell
    for ``length <= h``, else :func:`gradient_bound`); a ray starting farther than ``G
    length`` from the surface (with ``NEAR_MARGIN`` of slack, far above the round-off of the
    evaluated ``phi``) has no sample with ``phi <= 0`` and the search would return "not
    found"."""
    table = local_gradient_bounds(ls)
    cell = jnp.floor((x - ls.lo) / ls.h).astype(jnp.int32)
    cell = jnp.clip(cell, 0, jnp.asarray(ls.coef.shape, jnp.int32) - 1)
    local = table[tuple(cell[:, a] for a in range(ls.dim))]
    g = jnp.where(length <= ls.h, local, gradient_bound(ls))
    return phi_start <= g * length * (1.0 + NEAR_MARGIN) + NEAR_MARGIN * ls.h


def _bracket_none(args):
    """The skipped search (no ray near): nothing found, ``s_hi = 0`` as :func:`_bracket`
    leaves it on a ray without a crossing."""
    _ls, x, _vhat, _ds, _phi0, _slot = args
    n = x.shape[0]
    return jnp.zeros(n, bool), jnp.zeros(n)


def _bracket_dense(args):
    """The search over every ray."""
    ls, x, vhat, ds, phi0, _slot = args
    return _bracket(ls, x, vhat, ds, phi0)


def _bracket_near(args):
    """The search over the compacted near rays ``slot`` (``-1`` = unused), scattered back;
    the others keep :func:`_bracket_none`'s values -- what the dense search gives them."""
    ls, x, vhat, ds, phi0, slot = args
    n = x.shape[0]
    take = jnp.maximum(slot, 0)
    found, s_hi = _bracket(ls, x[take], vhat[take], ds[take], phi0[take])
    put = jnp.where(slot >= 0, slot, n)
    return (
        jnp.zeros(n, bool).at[put].set(found, mode="drop"),
        jnp.zeros(n).at[put].set(s_hi, mode="drop"),
    )


def _near_slots(near, size):
    """The ``near`` rays compacted into ``size`` entries (``-1`` unused) and their count
    (the cumulative-sum compaction of ``wall_adjoint.lane_slots``)."""
    n = near.shape[0]
    pos = jnp.cumsum(near.astype(jnp.int32)) - 1
    target = jnp.where(near & (pos < size), pos, size)
    slot = (
        jnp.full((size,), -1, jnp.int32).at[target].set(jnp.arange(n, dtype=jnp.int32), mode="drop")
    )
    return slot, jnp.sum(near, dtype=jnp.int32)


def _culled_bracket(ls, x, vhat, ds, phi_start, near):
    """:func:`_bracket` run only where it can find something: skipped when no ray is
    ``near``, over the compacted near rays when they fit ``N // 8`` entries (``N >=
    NEAR_MIN``), else over all rays.  Every branch returns, ray for ray, the dense
    search's result (a far ray finds nothing and keeps ``s_hi = 0``); the branch is a
    runtime ``switch``."""
    n = x.shape[0]
    if n < NEAR_MIN:
        args = (ls, x, vhat, ds, phi_start, jnp.zeros(0, jnp.int32))
        return jax.lax.cond(jnp.any(near), _bracket_dense, _bracket_none, args)
    size = n // 8
    slot, count = _near_slots(near, size)
    branch = jnp.where(count == 0, 0, jnp.where(count <= size, 1, 2))
    args = (ls, x, vhat, ds, phi_start, slot)
    return jax.lax.switch(branch, (_bracket_none, _bracket_near, _bracket_dense), args)


# --- cull tables (convex hull of the coefficients) ---------------------------------------

CULL_SLACK = 1e-6
"""Slack, in lattice cells, of the segment-length tests of the cull (:func:`hit_body`):
far above the round-off of ``(x + s v_hat - lo) / h``."""


def block_min_table(ls: LevelSet, radius: int) -> jax.Array:
    """Per lattice cell, a lower bound of ``phi`` on the cells within ``radius`` of it.

    Entry ``i + radius + 2`` (per axis; ``i = floor((x - lo) / h)`` clipped to ``-2 - radius
    .. m + radius``, see :func:`_table_index`) is the minimum coefficient over the ``4^D``
    blocks of the cells ``i - radius .. i + radius``, i.e. over the coefficients ``i - 1 -
    radius .. i + 2 + radius`` (clipped to the grid).  The cubic B-spline weights are
    non-negative and sum to one (convex hull), so ``phi >= `` that minimum on those cells --
    for the *evaluated* ``phi`` too: every term of :func:`_contract` is a product of
    non-negative weights with the block's coefficients, so a positive minimum gives a
    positive float.  A cell beyond the lattice evaluates the block of the clipped cell
    (:func:`_block` clips each index), which the window then contains.  Separable: ``D``
    running minima of width ``2 radius + 4``."""
    d = ls.dim
    width = 2 * radius + 4
    out = ls.coef
    for axis in range(d):
        window = tuple(width if a == axis else 1 for a in range(d))
        pad = tuple((width - 1, width - 1) if a == axis else (0, 0) for a in range(d))
        out = jax.lax.reduce_window(out, jnp.inf, jax.lax.min, window, (1,) * d, pad)
    return out


def _table_index(ls: LevelSet, x: jax.Array, radius: int) -> tuple[jax.Array, ...]:
    """The :func:`block_min_table` entry of each point's lattice cell (the cell of
    :func:`_block`: the same ``floor((x - lo) / h)``)."""
    u = (x - ls.lo) / ls.h
    top = jnp.asarray(ls.coef.shape, u.dtype) + radius
    i = jnp.clip(jnp.floor(u), -2.0 - radius, top)
    return tuple((i[:, a] + (radius + 2)).astype(jnp.int32) for a in range(ls.dim))


def _cell_edge(ls: LevelSet, x: jax.Array) -> jax.Array:
    """``bool[N]``: ``x`` within ``CULL_SLACK`` cells of a lattice-cell face, where a
    rounding difference between two evaluations of ``floor((x - lo) / h)`` could pick the
    neighbouring cell."""
    u = (x - ls.lo) / ls.h
    f = u - jnp.floor(u)
    return jnp.any((f < CULL_SLACK) | (f > 1.0 - CULL_SLACK), axis=1)


CULL_MIN = 256
"""Rows per culled call (:func:`hit_body`, :func:`inside`) below which the candidates are
not compacted: the work is skipped when there are none, else done on every row."""
CULL_BUCKETS = (16, 4)
"""The compacted sizes ``N // b`` of the culled calls, smallest first; more candidates
than ``N // CULL_BUCKETS[-1]`` run on every row."""


def _rows_none(args, fill):
    """No candidate: every row gets ``fill``."""
    _ls, rows, _slot = args
    return jnp.full(rows[0].shape[0], fill)


def _rows_dense(args, fn):
    """``fn`` on every row."""
    ls, rows, _slot = args
    return fn(ls, *rows)


def _rows_compact(args, fn, fill, size):
    """``fn`` on the rows ``slot[:size]`` (``-1`` unused), scattered back into ``fill``."""
    ls, rows, slot = args
    n = rows[0].shape[0]
    slot = slot[:size]
    take = jnp.maximum(slot, 0)
    out = fn(ls, *(r[take] for r in rows))
    put = jnp.where(slot >= 0, slot, n)
    return jnp.full(n, fill, out.dtype).at[put].set(out, mode="drop")


def _culled(fn, fill, ls, rows, candidate):
    """``where(candidate, fn(ls, *rows), fill)`` with ``fn`` (row-wise) evaluated only on
    the candidate rows: skipped when there is none, on the candidates compacted into the
    smallest of the ``N // CULL_BUCKETS`` sizes that holds them, else on every row (a
    runtime ``switch``; ``N < CULL_MIN``: a ``cond`` between skipped and every row).  The
    caller guarantees that ``fn`` gives ``fill`` on a non-candidate row; every branch
    then returns, row for row, what ``fn`` does on all of them."""
    n = candidate.shape[0]
    none = functools.partial(_rows_none, fill=fill)
    dense = functools.partial(_rows_dense, fn=fn)
    if n < CULL_MIN:
        args = (ls, rows, jnp.zeros(0, jnp.int32))
        return jax.lax.cond(jnp.any(candidate), dense, none, args)
    sizes = tuple(n // b for b in CULL_BUCKETS)
    slot, count = _near_slots(candidate, sizes[-1])
    branch = jnp.where(count == 0, 0, 1 + sum((count > s).astype(jnp.int32) for s in sizes))
    compact = [functools.partial(_rows_compact, fn=fn, fill=fill, size=s) for s in sizes]
    return jax.lax.switch(branch, (none, *compact, dense), (ls, rows, slot))


def _inside_rows(ls, x):
    return value(ls, x) < 0.0


def inside(ls: LevelSet, x: jax.Array) -> jax.Array:
    """``phi(x) < 0``, ``bool[N]`` (points on the surface count as outside).

    Evaluated only where it can hold: a point whose cell's coefficient block is
    non-negative has ``phi >= 0`` (:func:`block_min_table`, radius 0; a point within
    ``CULL_SLACK`` of a cell face is tested against its neighbours' blocks too, radius 1),
    so the answer is exactly the dense ``value(ls, x) < 0``."""
    own = block_min_table(ls, 0)[_table_index(ls, x, 0)]
    near = block_min_table(ls, 1)[_table_index(ls, x, 1)]
    candidate = (own < 0.0) | (_cell_edge(ls, x) & (near < 0.0))
    return _culled(_inside_rows, False, ls, (x,), candidate)


def regularised_quotient(num: jax.Array, den: jax.Array, floor: jax.Array) -> jax.Array:
    """``num / den`` with the **primal** exact and the **tangent** of ``num`` divided by
    ``max(|den|, floor)`` (sign kept) -- the grazing regularisation of an implicit-function
    derivative: a hit time's derivative carries ``1 / (v . n)``, integrable in the ray
    measure but of infinite variance; clipping the denominator in the derivative only
    biases the estimate by ``O(floor)`` and bounds the per-hit weight.  ``den``'s own
    tangent is dropped (it multiplies ``num``, zero at a converged root)."""
    den_reg = jnp.where(jnp.abs(den) < floor, jnp.sign(den) * floor, den)
    den_reg = jnp.where(den_reg == 0.0, floor, den_reg)
    primal = jax.lax.stop_gradient(num / den)
    return primal + (num - jax.lax.stop_gradient(num)) / jax.lax.stop_gradient(den_reg)


def hit_body(
    x: jax.Array, v: jax.Array, ls: LevelSet, remaining: jax.Array | None = None
) -> jax.Array:
    """Time along the ray ``x + t v`` to the surface from outside, ``f64[N]``.

    Args:
        x: ``f64[N, D]`` positions.
        v: ``f64[N, 3]`` or ``f64[N, D]`` velocities; only the first ``D`` components
            are used.
        ls: the level set.
        remaining: ``f64[N]`` remaining move time (the search covers
            ``min(reach, |v| remaining)``); ``None`` -> ``reach``.

    Returns ``inf`` when no ``+ -> -`` crossing lies within the search length, when the
    ray starts inside the body beyond ``SURFACE_TOL``, or when ``v = 0``.
    """
    d = x.shape[1]
    vd = v[:, :d]
    speed = jnp.linalg.norm(vd, axis=1)
    moving = speed > 0.0
    vhat = vd / jnp.where(moving, speed, 1.0)[:, None]
    length = jnp.full_like(speed, ls.reach)
    if remaining is not None:
        length = jnp.minimum(length, jnp.where(moving, speed, 1.0) * remaining)
    ds = length / ls.n_sample
    phi0, g0 = value_and_grad(ls, x)
    tol = SURFACE_TOL * ls.h
    on_surface = (phi0 <= 0.0) & (phi0 >= -tol)
    inward = jnp.sum(vhat * g0, axis=1) < 0.0
    hit_now = on_surface & inward & moving
    deep = phi0 < -tol
    # a start on the surface moving out of the body -- the state right after a reflection,
    # phi = round-off of either sign -- is outside: its first sample interval must count
    # as a `+ -> -` bracket, or a re-entry closer than one sample spacing (a grazing
    # particle skimming a concave wall) is missed and the particle flies into the body
    # (2026-09-23, optc S4 gd iterate 6: 0.4 deg incidence, 0.015 chords, spacing 0.031)
    leaving = (jnp.abs(phi0) <= tol) & ~inward
    phi_start = jnp.where(leaving, jnp.maximum(phi0, jnp.finfo(phi0.dtype).tiny), phi0)
    # the bracket search is a function of constants to AD (its inputs are frozen, not
    # just its output): reverse mode never traces the loops, only the Newton step below.
    # It runs only on the rays that can cross the surface within their search length
    # (`_near`, a Lipschitz bound): the rest provably find nothing, as the dense search
    frozen = jax.lax.stop_gradient(ls)
    phi_s = jax.lax.stop_gradient(phi_start)
    near = _near(phi_s, jax.lax.stop_gradient(length), frozen, jax.lax.stop_gradient(x))
    near = moving & ~deep & ~hit_now & near
    found, s_b = _culled_bracket(
        frozen,
        jax.lax.stop_gradient(x),
        jax.lax.stop_gradient(vhat),
        jax.lax.stop_gradient(ds),
        phi_s,
        near,
    )
    # Newton polish with the bracket frozen: the implicit-function derivative of the root
    phi_b, g_b = value_and_grad(ls, x + s_b[:, None] * vhat)
    den = jnp.sum(vhat * g_b, axis=1)
    floor = 1e-12 * jnp.linalg.norm(g_b, axis=1)
    den = jnp.where(jnp.abs(den) < floor, -floor, den)
    den = jnp.where(den == 0.0, -1.0, den)
    s = s_b - regularised_quotient(phi_b, den, ls.grazing * jnp.linalg.norm(g_b, axis=1))
    s = jnp.maximum(s, 0.0)
    t = s / jnp.where(moving, speed, 1.0)
    ok = found & moving & ~deep & ~hit_now
    return jnp.where(hit_now, 0.0, jnp.where(ok, t, jnp.inf))


def _min_sample_step(i, carry, ls, x, vhat, ds):
    """One sample of the closest-approach search (``fori_loop`` body): track the minimum."""
    phi_min, s_min = carry
    s = i * ds
    phi = value(ls, x + s[:, None] * vhat)
    better = phi < phi_min
    return jnp.where(better, phi, phi_min), jnp.where(better, s, s_min)


def _golden_step(_i, carry, ls, x, vhat):
    """One golden-section contraction of ``[a, b]`` on ``phi`` along the ray."""
    a, b = carry
    g = 0.5 * (math.sqrt(5.0) - 1.0)
    c = b - g * (b - a)
    d = a + g * (b - a)
    fc = value(ls, x + c[:, None] * vhat)
    fd = value(ls, x + d[:, None] * vhat)
    take_left = fc < fd
    return jnp.where(take_left, a, c), jnp.where(take_left, d, b)


def closest_approach(
    x: jax.Array,
    v: jax.Array,
    ls: LevelSet,
    remaining: jax.Array | None = None,
    n_refine: int = 16,
) -> tuple[jax.Array, jax.Array]:
    """Minimum of ``phi`` along the undeflected segment ``x + s v_hat``, ``s in [0, L]``
    (``L = min(reach, |v| remaining)``): ``(phi_min f64[N], s_min f64[N])``.

    The sampled minimum (spacing ``h / 2``) is refined by ``n_refine`` golden-section
    steps on the bracket of neighbouring samples.  Positive for a ray that misses the body
    (its distance of closest approach, for a distance-like ``phi``), negative for one that
    enters it (the penetration depth of the *undeflected* ray).  Not differentiated:
    ``d phi_min / d coef = B(x + s_min v_hat)`` by the envelope theorem, which the caller
    takes from :func:`value` at the frozen point.
    """
    d = x.shape[1]
    vd = v[:, :d]
    speed = jnp.linalg.norm(vd, axis=1)
    moving = speed > 0.0
    vhat = vd / jnp.where(moving, speed, 1.0)[:, None]
    length = jnp.full_like(speed, ls.reach)
    if remaining is not None:
        length = jnp.minimum(length, jnp.where(moving, speed, 1.0) * remaining)
    ds = length / ls.n_sample
    phi0 = value(ls, x)
    phi_min, s_min = jax.lax.fori_loop(
        1,
        ls.n_sample + 1,
        functools.partial(_min_sample_step, ls=ls, x=x, vhat=vhat, ds=ds),
        (phi0, jnp.zeros_like(phi0)),
    )
    a = jnp.maximum(s_min - ds, 0.0)
    b = jnp.minimum(s_min + ds, length)
    a, b = jax.lax.fori_loop(
        0, n_refine, functools.partial(_golden_step, ls=ls, x=x, vhat=vhat), (a, b)
    )
    s_ref = 0.5 * (a + b)
    phi_ref = value(ls, x + s_ref[:, None] * vhat)
    better = phi_ref < phi_min
    return jnp.where(better, phi_ref, phi_min), jnp.where(better, s_ref, s_min)


# ---------------------------------------------------------------------------
# Construction: knots, interpolation, signed distances, reinitialisation.
# ---------------------------------------------------------------------------


def knots(ls: LevelSet) -> jax.Array:
    """Knot positions ``lo + j h`` of every coefficient, ``f64[m_1 * ... * m_D, D]``
    (last axis fastest, the flattening of ``coef``)."""
    axes = [ls.lo[a] + ls.h * jnp.arange(m) for a, m in enumerate(ls.coef.shape)]
    grid = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack([g.ravel() for g in grid], axis=1)


def _interpolation_inverse(m: int) -> np.ndarray:
    """Inverse of the 1-D system ``(c[j-1] + 4 c[j] + c[j+1]) / 6 = value[j]`` for the
    interior knots with natural end conditions (``c[0] - 2 c[1] + c[2] = 0`` and the
    mirror; the two end values are not interpolated)."""
    a = np.zeros((m, m))
    for j in range(1, m - 1):
        a[j, j - 1 : j + 2] = (1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0)
    a[0, :3] = (1.0, -2.0, 1.0)
    a[m - 1, m - 3 :] = (1.0, -2.0, 1.0)
    return np.linalg.inv(a)


def interpolate(values: jax.Array) -> jax.Array:
    """Coefficients whose spline takes ``values`` at the interior knots (separable solve;
    linear in ``values``, so differentiable).  ``values`` is ``f64[m_1, ..., m_D]``."""
    coef = values
    for axis, m in enumerate(values.shape):
        inv = jnp.asarray(_interpolation_inverse(m))
        # rows 0 and m-1 are the natural end conditions (right-hand side 0)
        rhs = jnp.moveaxis(coef, axis, 0)
        rhs = rhs.at[0].set(0.0).at[m - 1].set(0.0)
        coef = jnp.moveaxis(jnp.tensordot(inv, rhs, axes=(1, 0)), 0, axis)
    return coef


def grid_for_box(lo, hi, h: float, margin: int = 3) -> tuple[np.ndarray, tuple[int, ...]]:
    """Origin and coefficient shape whose valid domain ``u in [1, m - 2]`` covers the box
    ``[lo, hi]`` with ``margin`` knots to spare on every side."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    origin = lo - (margin + 1) * h
    m = tuple(math.ceil((hi[a] - lo[a]) / h) + 2 * (margin + 1) + 1 for a in range(lo.size))
    return origin, m


def check_covers(ls: LevelSet, lo, hi) -> None:
    """Raise unless the valid domain of ``ls`` contains the box ``[lo, hi]``."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    origin = np.asarray(ls.lo, dtype=np.float64)
    u_lo = (lo - origin) / ls.h
    u_hi = (hi - origin) / ls.h
    m = np.asarray(ls.coef.shape)
    if (u_lo < 1.0).any() or (u_hi > m - 2.0).any():
        raise ValueError(
            f"level set does not cover the box: u in [{u_lo}, {u_hi}] but the valid domain "
            f"is [1, {m - 2}]"
        )


def from_values(values, lo, h: float, n_sample: int = 8, n_bisect: int = 40) -> LevelSet:
    """A level set interpolating ``values`` at the knots (see :func:`interpolate`)."""
    return LevelSet(
        coef=interpolate(jnp.asarray(values, dtype=jnp.float64)),
        lo=jnp.asarray(lo, dtype=jnp.float64),
        h=float(h),
        n_sample=int(n_sample),
        n_bisect=int(n_bisect),
    )


def analytic_distance(centre, radius, x: jax.Array) -> jax.Array:
    """Signed distance ``|x - c| - r`` to a circle / sphere, ``f64[N]``."""
    return jnp.linalg.norm(x - jnp.asarray(centre)[None, :], axis=1) - radius


def from_analytic(
    centre, radius, lo, h: float, shape: tuple[int, ...], n_sample: int = 8, n_bisect: int = 40
) -> LevelSet:
    """The level set interpolating the exact signed distance of a circle / sphere."""
    template = LevelSet(coef=jnp.zeros(shape), lo=jnp.asarray(lo, dtype=jnp.float64), h=float(h))
    sdf = analytic_distance(centre, radius, knots(template)).reshape(shape)
    return from_values(sdf, lo, h, n_sample, n_bisect)


def surface_distance(points: jax.Array, elements: jax.Array, x: jax.Array, chunk: int = 4096):
    """Unsigned distance from ``x`` (``f64[N, D]``) to a polygonal surface, ``f64[N]``
    (chunked over points; JAX, differentiable in ``points``)."""
    from jax_dsmc.mesh import geometry as mesh_geometry  # noqa: PLC0415 -- cycle

    d = x.shape[1]
    verts = points[elements]  # [E, D, D]
    n = x.shape[0]
    pad = (-n) % chunk
    xp = jnp.concatenate([x, jnp.zeros((pad, d))], axis=0).reshape(-1, chunk, d)
    fn = functools.partial(_distance_chunk, verts=verts, mesh_geometry=mesh_geometry)
    return jax.lax.map(fn, xp).reshape(-1)[:n]


def _distance_chunk(xc, verts, mesh_geometry):
    p = jnp.broadcast_to(verts[None], (xc.shape[0], *verts.shape))
    if xc.shape[1] == 2:
        dist = mesh_geometry._segment_distance(xc, p[:, :, 0], p[:, :, 1])
    else:
        dist = mesh_geometry._triangle_distance(xc, p[:, :, 0], p[:, :, 1], p[:, :, 2])
    return jnp.min(dist, axis=1)


def from_surface(
    surface, lo, h: float, shape: tuple[int, ...], n_sample: int = 8, n_bisect: int = 40
) -> LevelSet:
    """The level set interpolating the signed distance of a polygonal :class:`Surface`
    (distance in JAX, sign by the host ray-parity test of :mod:`jax_dsmc.mesh.cut`)."""
    from jax_dsmc.mesh import cut  # noqa: PLC0415 -- cycle

    template = LevelSet(coef=jnp.zeros(shape), lo=jnp.asarray(lo, dtype=jnp.float64), h=float(h))
    x = knots(template)
    dist = surface_distance(surface.points, surface.elements, x)
    sign = jnp.where(jnp.asarray(cut._global_outside(surface, np.asarray(x))), 1.0, -1.0)
    return from_values((sign * dist).reshape(shape), lo, h, n_sample, n_bisect)


def _project_step(_i, x, ls, max_step):
    """One closest-point projection ``x <- x - phi grad phi / |grad phi|^2``, the step
    clipped to ``max_step`` (a knot far from the surface, or one where ``grad phi`` is
    small, must not be thrown across the domain)."""
    phi, g = value_and_grad(ls, x)
    step = -(phi / jnp.maximum(jnp.sum(g * g, axis=1), 1e-300))[:, None] * g
    size = jnp.linalg.norm(step, axis=1, keepdims=True)
    return x + step * jnp.minimum(1.0, max_step / jnp.maximum(size, 1e-300))


def project(ls: LevelSet, x: jax.Array, n_iter: int = 8) -> jax.Array:
    """Closest points on the surface of the points ``x`` (``f64[N, D]``): ``n_iter`` clipped
    projection steps along ``grad phi`` (exact for a distance-like ``phi``)."""
    fn = functools.partial(_project_step, ls=ls, max_step=ls.h)
    return jax.lax.fori_loop(0, n_iter, fn, x)


def reinitialise(ls: LevelSet, n_iter: int | None = None, band: float | None = None) -> LevelSet:
    """Refit ``phi`` to the signed distance of its own zero set.

    Knots within ``band`` of the surface (``|phi| <= band``, default ``3 h + reach``) are
    projected onto it by :func:`project`; their new value is the distance to the projected
    point.  The projected band points form a point cloud of the surface, and every other
    knot takes its distance to that cloud (host ``cKDTree``; the cloud's spacing ``~h`` is
    the error, far from the surface where only the sign and the Lipschitz bound matter).
    Signs are inherited from ``phi``.  Not differentiable (an optimiser calls it between
    iterates).

    ``n_iter`` (default ``ceil(band / h) + 4``) must let a band knot travel the band: the
    projection step is clipped to ``h``, and a knot that stops short would enter the cloud
    as a false surface point (2026-09-23: the fixed 8 steps against the 11 h band of
    ``n_sample = 16`` put rings of spurious zero crossings ~0.5-0.7 off the surface after
    two iterates).  Knots whose projection still misses the surface (``|phi| > 1e-6 h``
    there, e.g. near a medial axis) are left out of the cloud and take the cloud distance.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415 -- host-only dependency

    band = 3.0 * ls.h + ls.reach if band is None else float(band)
    n_iter = math.ceil(band / ls.h) + 4 if n_iter is None else int(n_iter)
    x = knots(ls)
    phi = np.asarray(value(ls, x))
    in_band = np.abs(phi) <= band
    xb = jnp.asarray(np.asarray(x)[in_band])
    proj = np.asarray(project(ls, xb, n_iter))
    converged = np.abs(np.asarray(value(ls, jnp.asarray(proj)))) <= 1e-6 * ls.h
    if not converged.any():
        raise ValueError("reinitialise: no band knot projected onto the surface")
    dist_band = np.linalg.norm(proj - np.asarray(xb), axis=1)
    dist = cKDTree(proj[converged]).query(np.asarray(x))[0]
    band_idx = np.nonzero(in_band)[0]
    dist[band_idx[converged]] = dist_band[converged]
    values = np.where(phi < 0.0, -dist, dist).reshape(ls.coef.shape)
    return LevelSet(interpolate(jnp.asarray(values)), ls.lo, ls.h, ls.n_sample, ls.n_bisect)


def body_volume(ls: LevelSet, lo, hi, sub: int, width: float | None = None) -> jax.Array:
    """Volume of ``phi < 0`` inside the box ``[lo, hi]`` by a ``sub``-per-knot-spacing
    centred sub-grid with a smoothed Heaviside of half-width ``width`` (default ``h / 2``):
    differentiable in ``coef`` (the constraint of the optimiser; ``O(width^2)`` bias for a
    distance-like ``phi``)."""
    box = np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64)  # static
    lo = jnp.asarray(lo, dtype=jnp.float64)
    hi = jnp.asarray(hi, dtype=jnp.float64)
    d = lo.shape[0]
    counts = [math.ceil(box[a] / ls.h) * sub for a in range(d)]
    axes = [lo[a] + (hi[a] - lo[a]) * (jnp.arange(counts[a]) + 0.5) / counts[a] for a in range(d)]
    grid = jnp.meshgrid(*axes, indexing="ij")
    x = jnp.stack([g.ravel() for g in grid], axis=1)
    w = ls.h / 2.0 if width is None else float(width)
    phi = value(ls, x)
    heaviside = 0.5 * (1.0 - jnp.tanh(phi / w))
    cell = jnp.prod((hi - lo) / jnp.asarray(counts, dtype=jnp.float64))
    return jnp.sum(heaviside) * cell


def coefficient_band(ls: LevelSet, width: float) -> jax.Array:
    """``bool[m_1, ..., m_D]``: coefficients whose knot lies within ``width`` of the
    surface (``|phi| <= width`` at the knot) -- the ones a shape step should move."""
    return (jnp.abs(value(ls, knots(ls))) <= width).reshape(ls.coef.shape)
