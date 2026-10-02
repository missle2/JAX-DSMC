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
CULL_BUCKETS = (32, 8)
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


def _near_slots(near, size):
    """The ``near`` rows compacted into ``size`` entries (``-1`` unused) and their count
    (the cumulative-sum compaction of ``wall_adjoint.lane_slots``)."""
    n = near.shape[0]
    pos = jnp.cumsum(near.astype(jnp.int32)) - 1
    target = jnp.where(near & (pos < size), pos, size)
    slot = (
        jnp.full((size,), -1, jnp.int32).at[target].set(jnp.arange(n, dtype=jnp.int32), mode="drop")
    )
    return slot, jnp.sum(near, dtype=jnp.int32)


NEAR_MIN = 2048
"""Rows below which the polygon search of :mod:`jax_dsmc.mesh.geometry` is not compacted."""


def _inside_rows(ls, x):
    return value(ls, x) < 0.0


def inside(ls: LevelSet, x: jax.Array) -> jax.Array:
    """``phi(x) < 0``, ``bool[N]`` (points on the surface count as outside).

    Evaluated only where it can hold: a point whose cell's coefficient block is
    non-negative has ``phi >= 0`` (:func:`block_min_table`, radius 0; a point on a cell
    face is always evaluated), so the answer is exactly the dense ``value(ls, x) < 0``."""
    table = block_min_table(ls, 0)
    candidate = (table[_table_index(ls, x, 0)] < 0.0) | _cell_edge(ls, x)
    return _culled(_inside_rows, False, ls, (x,), candidate)


# ---------------------------------------------------------------------------
# Ray-surface hit.
# ---------------------------------------------------------------------------


def _samples(ls, x, vhat, ds, phi_start):
    """The sampled bracket, all ``n_sample`` samples of a ray in one evaluation: the first
    ``+ -> -`` sign change between consecutive samples (the start's ``phi_start`` first).
    Returns ``(found bool[N], s_lo, s_hi, phi_lo, phi_hi)`` (``f64[N]`` each; meaningless
    where not found)."""
    n, d = x.shape
    k = jnp.arange(1, ls.n_sample + 1, dtype=ds.dtype)
    s = k[None, :] * ds[:, None]  # [N, S]
    points = x[:, None, :] + s[:, :, None] * vhat[:, None, :]
    phi = value(ls, points.reshape(-1, d)).reshape(n, ls.n_sample)
    prev = jnp.concatenate([phi_start[:, None], phi[:, :-1]], axis=1)
    cross = (prev > 0.0) & (phi <= 0.0)
    first = jnp.argmax(cross, axis=1)[:, None]
    s_hi = jnp.take_along_axis(s, first, axis=1)[:, 0]
    phi_lo = jnp.take_along_axis(prev, first, axis=1)[:, 0]
    phi_hi = jnp.take_along_axis(phi, first, axis=1)[:, 0]
    return jnp.any(cross, axis=1), s_hi - ds, s_hi, phi_lo, phi_hi


ROOT_PHI_TOL = 1e-15
"""``|phi| <= ROOT_PHI_TOL h`` at an iterate ends its refinement (the root is resolved to
the round-off of ``phi``; the Newton polish of :func:`hit_body` follows)."""


def _refine_more(carry, max_iter):
    """``while_loop`` condition of :func:`_refine`: iterations left and a ray unresolved."""
    k, _s, _a, _b, _dx, done = carry
    return (k < max_iter) & ~jnp.all(done)


def _refine_step(carry, ls, x, vhat, tol):
    """One safeguarded Newton iteration on the bracket ``phi(a) > 0 >= phi(b)`` (``rtsafe``):
    the Newton step when it stays inside the updated bracket and at most halves the previous
    step, else bisection."""
    k, s, a, b, dx_old, done = carry
    phi, g = value_and_grad(ls, x + s[:, None] * vhat)
    slope = jnp.sum(vhat * g, axis=1)
    positive = phi > 0.0
    a = jnp.where(positive, s, a)
    b = jnp.where(positive, b, s)
    step = phi / jnp.where(slope == 0.0, 1.0, slope)
    newton = s - step
    use_newton = (slope != 0.0) & (newton > a) & (newton < b) & (2.0 * jnp.abs(step) <= dx_old)
    s_new = jnp.where(use_newton, newton, 0.5 * (a + b))
    dx = jnp.abs(s_new - s)
    resolved = jnp.abs(phi) <= ROOT_PHI_TOL * ls.h
    keep = done | resolved
    s = jnp.where(keep, s, s_new)
    return k + 1, s, a, b, jnp.where(keep, dx_old, dx), keep | (dx <= tol)


def _refine(ls, x, vhat, found, s_lo, s_hi, phi_lo, phi_hi, tol):
    """The root in the sampled bracket to ``tol``: safeguarded Newton from the secant point,
    a ``while_loop`` that stops when every ray is resolved (at most ``n_bisect`` iterations;
    a pure bisection would reach ``tol = ds 2^-n_bisect`` in as many).  ``f64[N]``, ``0``
    where not found."""
    s0 = s_lo + (s_hi - s_lo) * (phi_lo / (phi_lo - phi_hi))
    s0 = jnp.where(found, jnp.clip(s0, s_lo, s_hi), 0.0)
    carry = (jnp.int32(0), s0, s_lo, s_hi, s_hi - s_lo, ~found)
    carry = jax.lax.while_loop(
        functools.partial(_refine_more, max_iter=ls.n_bisect),
        functools.partial(_refine_step, ls=ls, x=x, vhat=vhat, tol=tol),
        carry,
    )
    return carry[1]


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


def _hit_rows(ls, x, vhat, speed, ds):
    """:func:`hit_body` on every row given (``vhat`` the unit direction, ``0`` where
    ``speed = 0``; ``ds`` the sample spacing)."""
    moving = speed > 0.0
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
    # just its output): reverse mode never traces it, only the Newton step below
    frozen = jax.lax.stop_gradient(ls)
    xf = jax.lax.stop_gradient(x)
    vf = jax.lax.stop_gradient(vhat)
    found, s_lo, s_hi, phi_lo, phi_hi = _samples(
        frozen, xf, vf, ds, jax.lax.stop_gradient(phi_start)
    )
    s_b = _refine(frozen, xf, vf, found, s_lo, s_hi, phi_lo, phi_hi, ds * 2.0**-ls.n_bisect)
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


def _hit_candidates(ls, x, length, moving, bounded):
    """``bool[N]``: can the ray hit at all (``hit_now`` or a sampled ``+ -> -`` crossing)?
    Every point of ``x + s v_hat``, ``s in [0, L]``, lies in the cells within ``ceil(L /
    h)`` of the start's; where the coefficient minimum over their blocks
    (:func:`block_min_table`) is positive, ``phi > 0`` on the whole segment, so no sample
    is ``<= 0``, the start is neither on the surface nor inside, and :func:`_hit_rows`
    returns ``inf``.  Radius 1 for ``L <= h`` (``bounded``: the moves of a known remaining
    time), else ``n_sample // 2 + 1`` (covers ``reach = n_sample h / 2``)."""
    wide = ls.n_sample // 2 + 1
    lower = block_min_table(ls, wide)[_table_index(ls, x, wide)]
    if bounded:
        short = block_min_table(ls, 1)[_table_index(ls, x, 1)]
        lower = jnp.where(length <= (1.0 - CULL_SLACK) * ls.h, short, lower)
    return moving & (lower <= 0.0)


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
    ray starts inside the body beyond ``SURFACE_TOL``, or when ``v = 0``.  Only the rays
    that can hit (:func:`_hit_candidates`, exact) are searched (:func:`_culled`).
    """
    d = x.shape[1]
    vd = v[:, :d]
    speed = jnp.linalg.norm(vd, axis=1)
    moving = speed > 0.0
    vhat = vd / jnp.where(moving, speed, 1.0)[:, None]
    length = jnp.full_like(speed, ls.reach)
    if remaining is not None:
        length = jnp.minimum(length, jnp.where(moving, speed, 1.0) * remaining)
    ds = jax.lax.stop_gradient(length / ls.n_sample)
    candidate = _hit_candidates(
        jax.lax.stop_gradient(ls),
        jax.lax.stop_gradient(x),
        jax.lax.stop_gradient(length),
        moving,
        remaining is not None,
    )
    return _culled(_hit_rows, jnp.inf, ls, (x, vhat, speed, ds), candidate)


