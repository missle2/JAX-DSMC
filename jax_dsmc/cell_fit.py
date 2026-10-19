"""Per-cell polynomial fits of the carried adjoint: ONE least-squares fit per collision
cell, evaluated at every query in the cell -- the alternative to
:func:`ensemble_device.field_query`'s weighted fit around each query from its ``k``
nearest nodes.

The virtual pull-back asks ~``(3 + 2 K) N`` field values per substep, every one of which
the per-query fit pays a ``k``-row normal matrix and a Cholesky solve for; here the
``C`` fits are formed once per substep and a query is a polynomial evaluation.  With
``k`` at or above the cell occupancy the per-query fit already uses the whole cell, and
its weights ``1 / (1 + (d / d_k)^2)`` then only range over ``[1/2, 1]`` -- the per-cell
fit drops them (or replaces them by a fixed kernel around the cell, ``pool``).

A fit is named by a ``basis`` string ``"cell:<poly>[:<option>]..."`` (static):

* ``<poly>``: the monomials, in scaled offsets ``o = (z - centre) / scale`` of
  ``z = (x, v)`` from the fit's node-mean centre -- ``mean`` (1), ``lin`` (``1, o``),
  ``quad`` (``lin`` + ``v (x) v`` + ``x_d^2``: the per-query fit's ``quadratic_full``),
  ``quadfull`` (every product ``o_a o_b``), ``cubicv`` (``quad`` + the cubic monomials
  of ``v``), ``cubic`` (``quadfull`` + every cubic monomial);
* ``pool<r>``: fit the ``(2 r + 1)^D`` block of cells around the cell (its value is
  still only evaluated inside the cell), weighted by a tricube in the position offset
  from the cell's node mean (support ``r + 1`` cells);
* ``adapt<a>``: the highest polynomial order whose cumulative column count is at most
  ``count / a`` per cell (the order groups 0, 1, 2, 3), lower orders' columns kept --
  a cell too sparse for its quadratic falls back to a plane or its mean instead of
  leaning on the ridge;
* ``loo``: per cell, the order (0 to the poly's) of least leave-one-out residual sum
  ``sum w (r_i / (1 - h_ii))^2`` -- each order's fit and hat diagonal from the same
  normal matrix with its higher-order columns masked;
* ``guard<h>``: per QUERY, the highest order (1 to the poly's, lower orders' columns kept, the
  ``adapt`` cap still applying) whose leverage ``h = f(o)^T M^-1 f(o)`` at the query is at most
  ``h``, the cell mean (order 0) if none is -- ``h`` is the fitted value's variance in units of
  one node's noise, so a query outside the region the cell's nodes determine (e.g. a
  post-collision velocity in the empty gap between a hypersonic beam and the wall-reflected
  gas, where a cubic's leverage reaches ~10^3) takes a lower order instead of an
  extrapolation the backward would compound every substep (the thin-body M 10 blow-up of
  ``cubicv``, 2026-10-18: scratch/thingate).  Not with ``loo``.  :data:`TRACE` counts the
  orders chosen;
* ``ridge<e>``: the relative ridge on the non-intercept columns (default ``1e-8``, as
  :data:`ensemble_device.RIDGE_REL`);
* ``rb``: the virtual partners' four-term values are the field's EXPECTATION over the
  scattering angle instead of its value at one sampled angle (Rao-Blackwellised:
  :func:`ensemble_device._scatter_sigma`) -- exact for a field quadratic in ``v``;
* ``own`` / ``align<r>`` / ``nofallback`` / ``both``: the split fits next to surfaces, below.

**Split fits next to surfaces** (the backward's ``normals``: each slot's nearest surface
normal within a cell width, zero elsewhere; any dimension -- normals and velocities are
3-vectors).  The carried value jumps across ``v . n = 0`` next to a surface (about to
hit / just left), which a polynomial in ``v`` cannot represent, so each cell is fitted
three times (:class:`SplitField`): ``ahead`` (moving away from the surface), ``behind``
(towards it) and ``both`` (every node: the field of a query with no surface near it, as
the per-query one-sided fit leaves such a query unrestricted).  The side of a node:

* default: ``v . n_c > 0`` with ``n_c`` the cell's mean normal (the per-query fit's rule
  ``v_node . n_q > 0`` when one normal describes the cell);
* ``own``: ``v_i . n_i > 0`` with the node's OWN nearest normal -- each particle relative
  to the surface it is about to hit / just left, which stays meaningful at corners, two
  facing surfaces and curvature at the cell scale; a node without a normal enters both
  sides.  A query's side is its own ``v . n_q``.

A cell where one normal does not describe the surface -- the resultant length
``R = |sum n_i| / #(n_i != 0)`` of its nodes' normals below ``align<r>`` (default 0: no
cell is) -- is **complex**: its queries are answered by the per-query one-sided fit
instead (:func:`ensemble_device.backward_virtual`'s ``fallback_capacity``), unless
``nofallback`` keeps the split fit there or ``both`` uses the cell's unsplit fit there
(two-sided: the jump is not represented in those cells -- an accepted error).  The
per-query fit is not exact in such a cell either (it classifies every node by the query's
normal), and its one-sided form is UNSTABLE on complicated geometry at ~40 particles per
cell (optc S1-adam iterate 100, Kn 0.5: lambda grows ~150x, 2026-10-11), so the
per-query fallback inherits that.

Value columns (``node_val`` ``(C, P, R)``) are right-hand sides of the same solve.
"""

from __future__ import annotations

import functools
import itertools
from typing import NamedTuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

POLYS = ("mean", "lin", "quad", "quadfull", "cubicv", "cubic")


class Spec(NamedTuple):
    poly: str
    pool: int = 0
    adapt: float = 0.0
    loo: bool = False
    ridge: float = 1e-8
    rb: bool = False
    own: bool = False
    align: float = 0.0
    fallback: bool = True
    both: bool = False
    guard: float = 0.0


# the orders a ``guard`` fit chose: None (off) or a dict ``{C: int64[C, G]}`` the evaluations
# add their per-cell counts to (a host callback per evaluation: diagnostics only; set it before
# the backward is first traced)
TRACE = None


class CellField(NamedTuple):
    """A fitted per-cell field: ``centre f64[C, D+3]`` (scaled), ``beta f64[C, p]`` (or
    ``f64[C, p, R]`` fitted to value columns).  A ``guard`` fit also carries every order's fit
    ``levels f64[C, G, p(, R)]`` (zero above the order) and the inverse normal matrices
    whitening ``whiten f64[C, p, p]`` (``L^-1`` of the normal matrix: :func:`_fit_guard`), and the
    highest order ``adapt`` lets each cell fit, ``top int[C]``."""

    centre: jax.Array
    beta: jax.Array
    levels: jax.Array | None = None
    whiten: jax.Array | None = None
    top: jax.Array | None = None


class SplitField(NamedTuple):
    """The three per-cell fields of a split fit (module docstring): ``ahead`` / ``behind``
    / ``both``, with the cells' unit mean normals ``normal f64[C, 3]`` (zero rows: no
    surface near), the ``complex bool[C]`` cells and which sides have no node
    (``ahead_empty`` / ``behind_empty`` ``bool[C]``: a query there takes ``both``)."""

    ahead: CellField
    behind: CellField
    both: CellField
    normal: jax.Array
    complex: jax.Array
    ahead_empty: jax.Array
    behind_empty: jax.Array


def cell_normals(node_normal, node_valid, align):
    """``(normal f64[C, 3], complex bool[C])`` of a node normal table ``(C, P, 3)``: the
    unit mean of the nodes' normals (zero rows without any) and ``R < align``."""
    n = jnp.where(node_valid[..., None], node_normal.astype(jnp.float64), 0.0)
    total = jnp.sum(n, axis=1)
    has = jnp.sum(jnp.any(n != 0.0, axis=-1), axis=1)
    length = jnp.sqrt(jnp.sum(total**2, axis=1))
    normal = jnp.where(length[:, None] > 0.0, total / jnp.maximum(length, 1e-300)[:, None], 0.0)
    resultant = length / jnp.maximum(has, 1)
    return normal, (has > 0) & (resultant < align)


def is_cell(basis: str) -> bool:
    return basis.startswith("cell:")


NUMERIC_OPTIONS = {"pool": int, "adapt": float, "guard": float, "ridge": float}


@functools.cache
def parse(basis: str) -> Spec:
    """The :class:`Spec` of a ``"cell:..."`` basis string."""
    head, poly, *opts = basis.split(":")
    if head != "cell" or poly not in POLYS:
        raise ValueError(f"not a per-cell basis: {basis!r} (polys {POLYS})")
    spec = Spec(poly)
    for o in opts:
        numeric = next((n for n in NUMERIC_OPTIONS if o.startswith(n)), None)
        if numeric:
            spec = spec._replace(**{numeric: NUMERIC_OPTIONS[numeric](o[len(numeric) :])})
        elif o == "loo":
            spec = spec._replace(loo=True)
        elif o == "rb":
            spec = spec._replace(rb=True)
        elif o == "own":
            spec = spec._replace(own=True)
        elif o.startswith("align"):
            spec = spec._replace(align=float(o[5:]))
        elif o == "nofallback":
            spec = spec._replace(fallback=False)
        elif o == "both":
            spec = spec._replace(fallback=False, both=True)
        else:
            raise ValueError(f"unknown per-cell fit option {o!r} in {basis!r}")
    if spec.guard and spec.loo:
        raise ValueError(f"guard and loo are exclusive: {basis!r}")
    return spec


@functools.cache
def monomials(poly: str, dim: int) -> tuple[tuple[int, ...], ...]:
    """The monomials of ``poly`` over ``dim + 3`` variables (positions first), each a
    sorted tuple of variable indices (``()`` the intercept), in order of degree."""
    nvar = dim + 3
    vel = tuple(range(dim, nvar))
    terms = [()]
    if poly == "mean":
        return tuple(terms)
    terms += [(a,) for a in range(nvar)]
    if poly == "lin":
        return tuple(terms)
    if poly in ("quad", "cubicv"):
        terms += list(itertools.combinations_with_replacement(vel, 2))
        terms += [(a, a) for a in range(dim)]
    else:
        terms += list(itertools.combinations_with_replacement(range(nvar), 2))
    if poly == "cubicv":
        terms += list(itertools.combinations_with_replacement(vel, 3))
    elif poly == "cubic":
        terms += list(itertools.combinations_with_replacement(range(nvar), 3))
    return tuple(terms)


def orders(poly: str, dim: int) -> tuple[int, ...]:
    return tuple(len(t) for t in monomials(poly, dim))


def features(offsets, poly):
    """The design ``(..., p)`` of scaled offsets ``(..., D+3)``."""
    dim = offsets.shape[-1] - 3
    cols = []
    for t in monomials(poly, dim):
        col = jnp.ones(offsets.shape[:-1])
        for a in t:
            col = col * offsets[..., a]
        cols.append(col)
    return jnp.stack(cols, axis=-1)


def _tricube(u):
    return jnp.where(u < 1.0, (1.0 - u**3) ** 3, 0.0)


def _masked_normal(design, weights, keep, ridge):
    """Weighted normal matrix ``(C, p, p)`` with the columns ``~keep`` (``(C, p)``) masked
    to an identity block (their coefficients come out 0) and the relative ridge."""
    d = jnp.where(keep[:, None, :], design, 0.0)
    m = jnp.einsum("cki,ck,ckj->cij", d, weights, d)
    p = design.shape[-1]
    diag_scale = jnp.trace(m, axis1=1, axis2=2) / jnp.maximum(jnp.sum(keep, axis=1), 1)
    ridge_vec = jnp.full(p, ridge).at[0].set(0.0)
    m = m + ridge_vec[None] * diag_scale[:, None, None] * jnp.eye(p)
    m = m + jnp.eye(p)[None] * (~keep)[:, :, None].astype(m.dtype)
    # an empty cell (or an empty side of a split one): the identity, coefficients 0
    empty = jnp.sum(weights, axis=1) <= 0.0
    return jnp.where(empty[:, None, None], jnp.eye(p)[None], m), d


def _solve(m, rhs):
    """``m^-1 rhs`` for ``C`` normal matrices: a batched LU (``C`` solves per substep, not
    one per query, so the unrolled Cholesky of the per-query fits buys nothing here and
    would unroll ``p^3`` operations for the cubic bases)."""
    return jnp.linalg.solve(m, rhs)


def _solve_masked(design, weights, vals, keep, ridge):
    m, d = _masked_normal(design, weights, keep, ridge)
    rhs = jnp.einsum("cki,ck,ckr->cir", d, weights, vals)
    return _solve(m, rhs), m, d


def _loo_press(design, weights, vals, beta, m, d):
    """Leave-one-out residual sum per cell and column-summed ``(C,)``."""
    resid = vals - jnp.einsum("cki,cir->ckr", d, beta)
    # h_ii = w_i d_i^T M^-1 d_i: M^-1 d^T with the P nodes as right-hand sides
    minv_d = _solve(m, jnp.swapaxes(d, 1, 2))  # (C, p, P)
    h = weights * jnp.einsum("cki,cik->ck", d, minv_d)
    loo = resid / jnp.maximum(1.0 - h, 1e-3)[..., None]
    return jnp.sum(weights[..., None] * loo**2, axis=(1, 2))


def _fit_guard(design, weights, vals, keep, order, spec, squeeze, centre):
    """A ``guard`` fit's :class:`CellField`.  The monomials are ordered by degree, so every
    lower order's design is a leading block of the full one, and with ``M = L L^T`` (Cholesky)
    the leading block of ``W = L^-1`` whitens it: a query's leverage under order ``g`` is
    ``sum_{i < p_g} (W f)_i^2`` and that order's fit ``W_g^T (W F^T w y)_g`` -- one factorisation
    for every order (the ridge's scale is the full order's: < 1e-8 relative by default)."""
    m, d = _masked_normal(design, weights, keep, spec.ridge)
    eye = jnp.broadcast_to(jnp.eye(m.shape[-1]), m.shape)
    whiten = jax.scipy.linalg.solve_triangular(jnp.linalg.cholesky(m), eye, lower=True)
    u = jnp.einsum("cij,ckj,ck,ckr->cir", whiten, d, weights, vals)  # (C, p, R)
    levels = []
    for size in _prefix_sizes(spec.poly, centre.shape[-1] - 3):
        w_g = whiten[:, :size, :size]
        beta = jnp.einsum("cji,cjr->cir", w_g, u[:, :size])
        levels.append(jnp.pad(beta, ((0, 0), (0, m.shape[-1] - size), (0, 0))))
    levels = jnp.stack(levels, axis=1)  # (C, G, p, R)
    return CellField(
        centre,
        levels[:, -1, :, 0] if squeeze else levels[:, -1],
        levels[..., 0] if squeeze else levels,
        # single precision: only compared with the guard (~1e-3 relative at cond 1e8), and
        # the per-query gather of this matrix is the guard's cost
        whiten.astype(jnp.float32),
        jnp.max(jnp.where(keep, order[None, :], 0), axis=1),
    )


@functools.cache
def _prefix_sizes(poly, dim):
    """The column count of each order ``0 .. max`` (the monomials are ordered by degree)."""
    o = orders(poly, dim)
    return tuple(sum(1 for x in o if x <= g) for g in range(max(o) + 1))


def fit(node_x, node_v, node_valid, node_val, scale, basis, own_x=None, own_valid=None, adapt=0.0):
    """:class:`CellField` of the node tables ``(C, P, .)`` (``node_val`` ``(C, P)`` or
    ``(C, P, R)``).  A pooled fit passes the widened tables
    (:func:`ensemble_device.pooled_tables`) and the cell's own ``own_x`` / ``own_valid``
    (the kernel's centre).  ``adapt`` is a floor under the basis' ``adapt<a>``."""
    spec = parse(basis)
    spec = spec._replace(adapt=max(spec.adapt, adapt))
    vals = node_val if node_val.ndim == 3 else node_val[..., None]
    pts = jnp.concatenate((node_x, node_v), axis=-1) / scale
    weights = node_valid.astype(pts.dtype)
    dim = node_x.shape[-1]
    if spec.pool:
        own = own_x / scale[:dim]
        n_own = jnp.maximum(jnp.sum(own_valid, axis=1, keepdims=True), 1)
        anchor = jnp.sum(jnp.where(own_valid[..., None], own, 0.0), axis=1) / n_own
        u = jnp.abs(pts[..., :dim] - anchor[:, None, :]) / (spec.pool + 1.0)
        weights = weights * jnp.prod(_tricube(u), axis=-1)
    total = jnp.maximum(jnp.sum(weights, axis=1, keepdims=True), 1e-300)
    centre = jnp.sum(weights[..., None] * pts, axis=1) / total
    design = features(pts - centre[:, None, :], spec.poly)
    vals = jnp.where(node_valid[..., None], vals, 0.0)
    order = jnp.asarray(orders(spec.poly, dim))
    count = jnp.sum(node_valid, axis=1)
    if spec.adapt:
        cum = jnp.asarray([sum(o <= g for o in orders(spec.poly, dim)) for g in range(4)])
        allowed = jnp.sum(cum[None, :] * spec.adapt <= count[:, None], axis=1) - 1
        keep = order[None, :] <= jnp.maximum(allowed, 0)[:, None]
    else:
        keep = jnp.ones((count.shape[0], order.shape[0]), bool)
    squeeze = node_val.ndim == 2
    if spec.guard:
        return _fit_guard(design, weights, vals, keep, order, spec, squeeze, centre)
    if not spec.loo:
        beta, _, _ = _solve_masked(design, weights, vals, keep, spec.ridge)
        return CellField(centre, beta[..., 0] if squeeze else beta)
    best, best_press = None, None
    for g in range(int(max(orders(spec.poly, dim))) + 1):
        keep_g = keep & (order[None, :] <= g)
        beta, m, d = _solve_masked(design, weights, vals, keep_g, spec.ridge)
        press = _loo_press(design, weights, vals, beta, m, d)
        # an order with no more nodes than columns has h_ii -> 1: not eligible
        press = jnp.where(count > jnp.sum(keep_g, axis=1), press, jnp.inf)
        if best is None:
            best, best_press = beta, press
        else:
            better = press < best_press
            best = jnp.where(better[:, None, None], beta, best)
            best_press = jnp.where(better, press, best_press)
    return CellField(centre, best[..., 0] if squeeze else best)


def fit_split(
    node_x,
    node_v,
    node_valid,
    node_val,
    node_normal,
    normal,
    complex_,
    scale,
    basis,
    own_x=None,
    own_valid=None,
):
    """:class:`SplitField` of node tables (as :func:`fit`) with the nodes' normals
    ``node_normal (C, P, 3)`` and the cells' :func:`cell_normals`.  A pooled table's nodes
    take the side of the cell they are pooled INTO under the default rule, their own under
    ``own``.  A side holds about half the cell's nodes, so its polynomial order is capped
    at no more columns than nodes (``adapt`` floor 1: a sparse side fits a plane or its
    mean instead of an underdetermined quadratic); an empty side defers to ``both``."""
    if parse(basis).own:
        dot = jnp.sum(node_v * node_normal, axis=-1)
        free = jnp.all(node_normal == 0.0, axis=-1)
        ahead_side, behind_side = (dot > 0.0) | free, (dot <= 0.0) | free
    else:
        ahead_side = jnp.einsum("cpk,ck->cp", node_v, normal) > 0.0
        behind_side = ~ahead_side
    part = functools.partial(fit, node_x, node_v, node_val=node_val, scale=scale, basis=basis,
                             own_x=own_x, own_valid=own_valid)  # fmt: skip
    ahead_valid, behind_valid = node_valid & ahead_side, node_valid & behind_side
    return SplitField(
        part(node_valid=ahead_valid, adapt=1.0),
        part(node_valid=behind_valid, adapt=1.0),
        part(node_valid=node_valid),
        normal,
        complex_,
        ~jnp.any(ahead_valid, axis=1),
        ~jnp.any(behind_valid, axis=1),
    )


def take_trace():
    """The :data:`TRACE` counts so far, ``int64[C, G]`` summed over the field sizes seen (None
    if no guard fit was evaluated), and an emptied :data:`TRACE` for the next run."""
    jax.effects_barrier()
    if not TRACE:
        return None
    counts = TRACE[max(TRACE)]
    TRACE.clear()
    return counts


def _record(counts):
    counts = jax.device_get(counts)
    total = TRACE.setdefault(counts.shape[0], 0)
    TRACE[counts.shape[0]] = total + counts


GUARD_BATCH = 8192


def _leverage(args, whiten, sizes):
    """Every order's leverage at one query: prefix sums of its whitened design squared."""
    c, f = args
    z2 = jnp.cumsum((whiten[c] @ f.astype(whiten.dtype)) ** 2)
    return z2[jnp.asarray(sizes) - 1]


def _guard_level(field, c, f, guard, q_valid, sizes):
    """The order ``int[Q]`` a guard fit uses at each query (design ``f (Q, p)``)."""
    # batched: one gathered (p, p) whitening per query (576 floats for cubicv in 2-D)
    h = jax.lax.map(
        functools.partial(_leverage, whiten=field.whiten, sizes=sizes),
        (c, f),
        batch_size=GUARD_BATCH,
    )  # (Q, G)
    levels = jnp.arange(h.shape[1])
    ok = ((h <= guard) & (levels[None, :] <= field.top[c][:, None])) | (levels[None, :] == 0)
    level = jnp.max(jnp.where(ok, levels[None, :], 0), axis=1)
    if TRACE is not None:
        hit = jax.nn.one_hot(level, h.shape[1], dtype=jnp.int64) * q_valid[:, None]
        counts = jax.ops.segment_sum(hit, c, num_segments=field.centre.shape[0])
        jax.debug.callback(_record, counts)
    return level


def evaluate(
    field, c, q_x, q_v, scale, basis, gradients=False, q_normal=None, q_valid=None
):  # fmt: skip
    """The fitted value at ``Q`` queries in cells ``c`` -- ``f64[Q]`` (``f64[Q, R]`` for a
    field fitted to value columns) -- and with ``gradients`` its ``(grad_x, grad_v)`` per
    unscaled coordinate (``f64[Q, D]``, ``f64[Q, 3]``, trailing ``R`` likewise).  A
    :class:`SplitField` needs the queries' own surface normals ``q_normal (Q, 3)`` (zero
    rows: no surface near, the ``both`` field).  ``q_valid bool[Q]`` only masks a ``guard`` fit's
    :data:`TRACE` counts (dead slots' sentinel queries)."""
    if isinstance(field, SplitField):
        near = jnp.any(q_normal != 0.0, axis=1)
        side_normal = q_normal.astype(q_v.dtype) if parse(basis).own else field.normal[c]
        ahead = jnp.sum(q_v * side_normal, axis=1) > 0.0
        parts = [
            evaluate(f, c, q_x, q_v, scale, basis, gradients, q_valid=q_valid)
            for f in (field.ahead, field.behind, field.both)
        ]
        sided = jax.tree.map(functools.partial(_pick, ahead), parts[0], parts[1])
        empty = jnp.where(ahead, field.ahead_empty[c], field.behind_empty[c])
        use_sided = near & ~empty
        if parse(basis).both:
            use_sided = use_sided & ~field.complex[c]
        return jax.tree.map(functools.partial(_pick, use_sided), sided, parts[2])
    spec = parse(basis)
    q = jnp.concatenate((q_x, q_v), axis=-1) / scale
    o = q - field.centre[c]
    if spec.guard:
        f = features(o, spec.poly)
        valid = jnp.ones(c.shape, bool) if q_valid is None else q_valid
        level = _guard_level(
            field, c, f, spec.guard, valid, _prefix_sizes(spec.poly, q_x.shape[-1])
        )
        # only the chosen order's coefficients: (C, G, ...) flattened to (C G, ...)
        flat = field.levels.reshape((-1, *field.levels.shape[2:]))
        beta = flat[c * field.levels.shape[1] + level]
    else:
        beta = field.beta[c]  # (Q, p[, R])
    squeeze = beta.ndim == 2
    beta = beta[..., None] if squeeze else beta
    value = jnp.einsum("qi,qir->qr", features(o, spec.poly), beta)
    value = value[:, 0] if squeeze else value
    if not gradients:
        return value
    jac = jax.vmap(jax.jacfwd(functools.partial(features, poly=spec.poly)))(o)  # (Q, p, n)
    grad = jnp.einsum("qin,qir->qnr", jac, beta) / scale[None, :, None]
    grad = grad[..., 0] if squeeze else grad
    dim = q_x.shape[-1]
    return value, grad[:, :dim], grad[:, dim:]


def _pick(side, a, b):
    return jnp.where(side.reshape(side.shape + (1,) * (a.ndim - 1)), a, b)
