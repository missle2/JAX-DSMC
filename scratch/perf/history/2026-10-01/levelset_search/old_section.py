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


