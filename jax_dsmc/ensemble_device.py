"""Device (JIT) implementation of the particle-carried ensemble adjoint's backward sweep.

The algorithm is :mod:`jax_dsmc.ensemble_backward` (route b, full quadratic ghost
basis) with either collision pull-back, as nested scans under one reverse scan over the
captured steps:

* ``sequential`` -- a reverse scan over the attempt slots that mirrors the forward's
  ``ntc._slot`` scan, every cell handling its slot's accepted pair in parallel (the
  exact discrete adjoint of the NTC step);
* ``virtual`` -- :func:`backward_virtual`: the substepped operator ``exp(dt L*)`` with
  ``K`` virtual partners per particle and NTC weights, every field value from the
  smooth per-cell fit (the host's ``_collision_pullback_virtual``, ``"post"`` or
  ``"staged"`` nodes), the partner loop a scan so the query memory is ``O(N)`` per pass
  and the queries themselves chunked through ``lax.map``.

The one primitive everything else is built on is :func:`field_query`: the value **and
the velocity gradient** of the carried adjoint at arbitrary ``(x, v)`` from the current
particle nodes of the same cell.  The pathwise terms (``grad_v λ · ∂_θ v'``) and the shape
terms (``grad_x λ`` at reflected states) consume the same primitive; this module emits
only ``λ`` on the slots per step and the four-term of every accepted attempt, so the
parameter scoring stays outside it.

Value columns: the virtual backward carries ``R`` value fields at once when its rewards
have a trailing axis -- ``λ`` is ``f64[N, R]``, every fit factorised once for all columns
(:func:`field_query`, :func:`_spd_solve`).  The sweep decides nothing from the values, so
column ``r`` is the backward of reward ``r``; a shape seed's two value backwards run as
one (PSEUDOCODE section 9.7.1).

Memory: the whole ``EventRecord`` stack is resident (``O(M·N)``); block re-capture of
the forward is the known fix for long horizons and is not done here.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from jax_dsmc import cell_fit, sort, step_axis
from jax_dsmc.collide.ntc import step_pairs, step_v_post
from jax_dsmc.collide.vss import collision_rate, scatter
from jax_dsmc.grid import cell_neighbourhood as grid_neighbourhood
from jax_dsmc.grid import collision_cell, spacing
from jax_dsmc.state import ParticleState

jax.config.update("jax_enable_x64", True)

RIDGE_REL = 1e-8
FIT_PAIRS = 1 << 20
"""(query, node) pairs one :func:`cell_query` chunk may gather: a chunk materialises
~70-100 bytes per pair (the scaled node block, distances, masks, top-k), so the fit's
working set stays ~100 MB whatever the node table width ``P`` (which grows with the
particles per cell -- ~17 k candidates for a pooled fit at the fine circle's density)."""


class Backward(NamedTuple):
    """The backward's outputs.  Under :func:`backward_virtual` with value columns (rewards
    with a trailing axis ``R``) every value carries it: ``lambda0`` ``f64[N, R]``, ``lam``
    ``f64[M, N, R]``, ``kernel`` ``f64[M, R]`` -- column ``r`` the backward of reward
    column ``r``."""

    lambda0: jax.Array
    """``f64[N]`` adjoint at the initial state, on the initially alive slots (else 0)."""
    lam: jax.Array
    """``f64[M, N]`` adjoint at each step's post-move state (what sources and walls weight)."""
    four_term: jax.Array
    """``f64[M, A, C]`` ``Lam(v_i')+Lam(v_j')-Lam(v_i)-Lam(v_j)`` per accepted attempt (0
    elsewhere); zeros under the virtual pull-back, whose pairs are not the record's."""
    kernel: jax.Array
    """``f64[M]`` per-step ``sum 1/2 w * four_term * pair_score(v_i, v_j)`` over the virtual
    pairs -- the kernel-parameter gradient's collision term when ``pair_score`` is
    ``d/dtheta log q``; zeros under the sequential pull-back (score the record's pairs)."""
    collision: jax.Array | None = None
    """``f32[M, N]`` (``backward_virtual(collision_out=True)``) each slot's collision
    increment of the step (value column 0), ``sum_substeps sum_kappa w four`` =
    ``dt (L* lam)(x, v)`` at its post-move state: the along-flight identity's collision
    term (``shape.gradient_variants(collision=)``).  ``None`` otherwise."""
    cell_collision: jax.Array | None = None
    """``f64[M, C]`` (``collision_out=True``) half the sum of :attr:`collision` over each
    collision cell: the expected change of the objective by the cell's collisions in the
    step, ``C_kc`` of the cut-volume term (``shape.volume_weights``)."""


def cell_nodes(x, alive, grid, npc_max):
    """Padded per-cell node table: ``ids i32[C, P]`` (slot of node), ``valid bool[C, P]``,
    the cell of every particle (sentinel ``C`` when dead), its rank within the cell and
    the cell occupancies ``count i32[C]``."""
    n = x.shape[0]
    table = sort.build_table(
        ParticleState(x, jnp.zeros((n, 3)), jnp.zeros(n, jnp.int32), alive, jnp.zeros(n)), grid
    )
    rank = jnp.arange(npc_max)[None, :]
    ids = table.order[jnp.minimum(table.starts[:, None] + rank, n - 1)]
    valid = rank < table.counts[:, None]
    position = jnp.zeros(n, jnp.int32).at[table.order].set(jnp.arange(n, dtype=jnp.int32))
    c = jnp.minimum(table.cell, table.starts.shape[0] - 1)
    particle_rank = position - table.starts[c]
    return ids, valid, table.cell, particle_rank, table.counts


def _design(offsets, basis):
    """Local polynomial design over ``(Q, k, D+3)`` scaled offsets; column 0 is the intercept."""
    cols = [jnp.ones((*offsets.shape[:2], 1)), offsets]
    if basis in ("quadratic", "quadratic_full"):
        dv = offsets[..., -3:]
        if basis == "quadratic":
            cols.append(jnp.sum(dv**2, axis=-1, keepdims=True))
        else:
            iu = jnp.triu_indices(3)
            cols.append((dv[..., :, None] * dv[..., None, :])[..., iu[0], iu[1]])
        cols.append(offsets[..., :-3] ** 2)
    return jnp.concatenate(cols, axis=-1)


def _spd_solve(m, rhs):
    """``m^-1 rhs`` for a batch of small symmetric positive-definite matrices: ``m``
    ``f64[Q, p, p]``, ``rhs`` ``f64[Q, p, R]`` -> ``f64[Q, p, R]``.

    The Cholesky factor ``m = L L^T`` written out over the (static, small) dimension, then
    the forward and back substitutions for all ``R`` columns: every operation is
    elementwise over the ``Q`` queries, which XLA fuses into a few kernels -- where
    ``jnp.linalg.solve``'s batched LU of 14 x 14 matrices costs 2.3x as much on the GPU
    (`scratch/adjoint_fit/`, 2026-10-02) and refactorises per right-hand side.  The
    normal matrices of :func:`field_query` are positive definite (a weighted Gram matrix
    plus the relative ridge on the slopes); a pivot that rounding drives to zero or below
    -- a fit with fewer nodes than columns at the ridge's 1e-8 -- is floored at the
    smallest normal float rather than turned into a NaN.  Agrees with the LU solve to
    ~1e-13 relative on the fits' values and slopes."""
    p = m.shape[-1]
    tiny = jnp.finfo(m.dtype).tiny
    low = [[None] * p for _ in range(p)]  # low[i][j], j <= i: the entries of L, f64[Q, 1]
    for j in range(p):
        pivot = m[:, j, j, None] - sum(low[j][t] ** 2 for t in range(j))
        low[j][j] = jnp.sqrt(jnp.maximum(pivot, tiny))
        for i in range(j + 1, p):
            dot = sum(low[i][t] * low[j][t] for t in range(j))
            low[i][j] = (m[:, i, j, None] - dot) / low[j][j]
    y = []  # forward substitution, L y = rhs
    for i in range(p):
        y.append((rhs[:, i] - sum(low[i][t] * y[t] for t in range(i))) / low[i][i])
    x = [None] * p  # back substitution, L^T x = y
    for i in reversed(range(p)):
        x[i] = (y[i] - sum(low[t][i] * x[t] for t in range(i + 1, p))) / low[i][i]
    return jnp.stack(x, axis=1)


def field_query(node_x, node_v, node_valid, node_val, q_x, q_v, scale, k, basis="quadratic_full"):
    """Weighted local-polynomial fit of ``node_val`` at each query; returns
    ``(value, grad_x, grad_v)`` -- the intercept and the linear slopes of the fit at the
    query, per unscaled coordinate (``grad_x`` is ``f64[Q, D]``, ``grad_v`` ``f64[Q, 3]``).

    Shapes: nodes ``(Q, P, ·)`` gathered per query, queries ``(Q, ·)``.  Exactly the
    host :class:`ensemble_backward.LocalFit`: the ``k`` nearest valid nodes in the scaled
    ``(x, v)`` metric, weights ``1/(1+(d/d_k)^2)``, ridge ``1e-8`` (relative) on the slopes.
    The shape adjoint consumes ``grad_x`` (the position cotangent of the replayed wall
    map); the pathwise and collision terms consume the value and ``grad_v``.

    **Value columns.**  ``node_val`` may carry trailing value columns, ``f64[Q, P, R]``:
    ``R`` fields of the same nodes (the two value backwards of a shape seed,
    :func:`backward_virtual`).  The nearest nodes, the weights, the design and the normal
    matrix depend on the node and query states only, so they are formed and factorised
    once and every column is a right-hand side of the same solve: ``value`` is then
    ``f64[Q, R]``, ``grad_x`` ``f64[Q, D, R]``, ``grad_v`` ``f64[Q, 3, R]`` -- column ``r``
    exactly the single-column fit of ``node_val[..., r]`` (the same arithmetic per column).
    """
    columns = node_val.ndim == 3
    vals = node_val if columns else node_val[..., None]
    dim = q_x.shape[-1]
    pts = jnp.concatenate((node_x, node_v), axis=-1) / scale
    q = jnp.concatenate((q_x, q_v), axis=-1) / scale
    d2 = jnp.sum((pts - q[:, None, :]) ** 2, axis=-1)
    d2 = jnp.where(node_valid, d2, jnp.inf)
    neg, ids = lax.top_k(-d2, k)
    dist = jnp.sqrt(jnp.maximum(-neg, 0.0))
    good = jnp.isfinite(dist)
    dk = jnp.max(jnp.where(good, dist, 0.0), axis=1, keepdims=True)
    weights = jnp.where(good, 1.0 / (1.0 + (dist / jnp.maximum(dk, 1e-12)) ** 2), 0.0)
    chosen = jnp.take_along_axis(pts, ids[..., None], axis=1)
    values = jnp.take_along_axis(vals, ids[..., None], axis=1)  # (Q, k, R)
    design = _design(chosen - q[:, None, :], basis)
    design = jnp.where(good[..., None], design, 0.0)
    matrix = jnp.einsum("qki,qk,qkj->qij", design, weights, design)
    ridge = jnp.full(design.shape[-1], RIDGE_REL).at[0].set(0.0)
    diag_scale = jnp.trace(matrix, axis1=1, axis2=2) / design.shape[-1]
    matrix = matrix + ridge[None] * diag_scale[:, None, None] * jnp.eye(design.shape[-1])
    # a query with no valid node at all gets value 0 rather than a singular solve
    empty = ~jnp.any(good, axis=1)
    matrix = matrix.at[:, 0, 0].add(empty.astype(jnp.float64))
    # the right-hand sides as a weighted sum over the k nodes (Q, p, R); the named scopes
    # label these ops in a profile (scratch/perf/profile_report.py)
    with jax.named_scope("fit_rhs"):
        weighted = design * weights[..., None]
        rhs = jnp.sum(
            weighted[..., None] * jnp.where(good[..., None], values, 0.0)[:, :, None, :], axis=1
        )
    with jax.named_scope("spd_solve"):
        beta = _spd_solve(matrix, rhs)  # (Q, p, R)
    value = jnp.where(empty[:, None], 0.0, beta[:, 0])
    grad_x = beta[:, 1 : 1 + dim] / scale[:dim, None]
    grad_v = beta[:, 1 + dim : 1 + dim + 3] / scale[-3:, None]
    if not columns:
        return value[:, 0], grad_x[..., 0], grad_v[..., 0]
    return value, grad_x, grad_v


def _slot_pullback(carry, xs, x, ids, valid, scale, k, basis):
    """One attempt slot, all cells in parallel, in reverse order.  ``scan`` body.

    A slot in which no cell accepted an attempt (most slots: the attempt count per
    cell is far below ``A_max``) is skipped through a ``lax.cond`` -- the same measured
    trick as the wall tracker's empty passes.
    """
    _, _, accept, _, _ = xs
    body = functools.partial(_slot_events, x=x, ids=ids, valid=valid, scale=scale, k=k, basis=basis)
    return lax.cond(jnp.any(accept), body, _slot_skip, carry, xs)


def _slot_skip(carry, xs):
    return carry, jnp.zeros(xs[2].shape[0])


def _slot_events(carry, xs, x, ids, valid, scale, k, basis):
    lam, v_cur = carry
    pi, pj, accept, vi, vj = xs  # (C,), (C,), (C,) bool, (C, 3), (C, 3)
    # the partner ghosts: (x_j, v_j^pre) queried from cell c's current nodes, and (x_i, v_i^pre)
    node_x = x[ids]  # (C, P, D)
    node_v = v_cur[ids]
    node_val = lam[ids]
    q_x = jnp.concatenate((x[pj], x[pi]))
    q_v = jnp.concatenate((vj, vi))
    nx = jnp.concatenate((node_x, node_x))
    nv = jnp.concatenate((node_v, node_v))
    nval = jnp.concatenate((node_val, node_val))
    nvalid = jnp.concatenate((valid, valid))
    ghost, _, _ = field_query(nx, nv, nvalid, nval, q_x, q_v, scale, k, basis)
    ghost_j, ghost_i = ghost[: pi.shape[0]], ghost[pi.shape[0] :]
    li, lj = lam[pi], lam[pj]
    new_i = jnp.where(accept, li + lj - ghost_j, li)
    new_j = jnp.where(accept, lj + li - ghost_i, lj)
    four = jnp.where(accept, li + lj - ghost_i - ghost_j, 0.0)
    # inactive attempts address the scratch row n_part, which is never read back
    lam = lam.at[pi].set(new_i).at[pj].set(new_j)
    v_cur = v_cur.at[pi].set(jnp.where(accept[:, None], vi, v_cur[pi]))
    v_cur = v_cur.at[pj].set(jnp.where(accept[:, None], vj, v_cur[pj]))
    return (lam, v_cur), four


def _per_slot(a, lam):
    """A per-slot array ``a`` (``(N,)``, a mask or a weight) shaped to multiply or select
    ``lam``, which may carry trailing value columns ``(N, R)``."""
    return a.reshape(a.shape + (1,) * (lam.ndim - a.ndim))


def _add_step_reward(mu, rec_k, phi_k):
    """Value form of a per-step reward: the carried value at step ``k``'s post-collision
    state includes the reward collected there, so ``phi_k`` (``f64[N]``, per slot, already
    divided by the normalization) joins ``mu`` BEFORE the step's pull-back."""
    if phi_k is None:
        return mu
    return jnp.where(_per_slot(rec_k.alive, mu), mu + phi_k, mu)


def _move_pullback(lam, rec_k, r_k):
    """The move pull-back: survivors carry their post-move value back to the pre-move state,
    plus the rewards of the hits they took during the move (``r_k`` ``f64[K, N]``, one per
    tracker pass, zero where there was no hit).  Newborns (``born_face >= 0``) have no
    pre-move state; their value and birth-step hit rewards weight the source score."""
    born = rec_k.born_face >= 0
    lam = jnp.where(_per_slot(rec_k.alive, lam), lam, 0.0)
    total = lam if r_k is None else lam + jnp.sum(r_k, axis=0)
    return jnp.where(_per_slot(rec_k.alive & ~born, lam), total, 0.0), lam


def _step_pullback(mu, xs, grid, npc_max, scale, k, basis):
    """One captured step backward: sequential collision pull-back, then the move pull-back."""
    rec_k, phi_k, r_k = xs
    mu = _add_step_reward(mu, rec_k, phi_k)
    n = rec_k.x.shape[0]
    ids, valid, cell, _, _ = cell_nodes(rec_k.x, rec_k.alive, grid, npc_max)
    # scratch row for inactive attempts, as in the forward slot scan
    lam0 = jnp.concatenate((mu, jnp.zeros(1)))
    v0 = jnp.concatenate((step_v_post(rec_k), jnp.zeros((1, 3))))
    x = jnp.concatenate((rec_k.x, jnp.zeros((1, rec_k.x.shape[1]))))
    del cell
    body = functools.partial(
        _slot_pullback, x=x, ids=ids, valid=valid, scale=scale, k=k, basis=basis
    )
    p = step_pairs(rec_k)
    slots = (p.pi, p.pj, p.accept, p.vi, p.vj)
    (lam, _), four_rev = lax.scan(body, (lam0, v0), slots, reverse=True)
    mu_prev, lam = _move_pullback(lam[:n], rec_k, r_k)
    return mu_prev, (lam, four_rev)


def _scan_body(mu, xs, grid, npc_max, scale, k, basis):
    return _step_pullback(mu, xs, grid, npc_max, scale, k, basis)


@functools.partial(jax.jit, static_argnames=("npc_max", "k", "basis"))
def backward(
    records,
    initial_alive,
    final_state,
    phi_final,
    grid,
    scale,
    npc_max,
    k=32,
    basis="quadratic_full",
    phi_step=None,
    hit_reward=None,
):
    """Reverse scan over the captured steps.  ``phi_final`` is ``f64[N]`` = reward / normalization.

    Value-form rewards (plan section 2): ``phi_step`` ``f64[M, N]`` is a per-step reward at
    each step's post-collision state (the state ``simulate.run``'s ``reward`` hook sees),
    added to the carried value before that step's pull-back; ``hit_reward`` ``f64[M, K, N]``
    is a per-hit reward (one slot per tracker pass, as ``EventRecord.walls``), added on the
    move pull-back -- only its sum over the pass axis is used, so a lane replay passes its
    per-step totals ``shape.HitRewards.totals()[:, None]`` (``f64[M, 1, N]``).  Both are per slot and already divided by the normalization; a windowed
    objective is nonzero only on its window.  The returned ``lam`` stays the value at the
    post-move state; :func:`ensemble_backward.from_device` forms the per-hit weights
    ``lam + (rewards of the later hits of the step)`` from ``hit_reward``.
    """
    mu = jnp.where(final_state.alive, phi_final, 0.0)
    body = functools.partial(_scan_body, grid=grid, npc_max=npc_max, scale=scale, k=k, basis=basis)
    mu0, (lam, four) = lax.scan(body, mu, (records, phi_step, hit_reward), reverse=True)
    return Backward(jnp.where(initial_alive, mu0, 0.0), lam, four, jnp.zeros(lam.shape[0]))


# --- virtual-partner pull-back ------------------------------------------------------


def _complex_load(carry, xs, grid, npc_max, basis):
    """Alive slots in the complex cells of one step (:func:`fallback_capacity`)."""
    x, alive, normals = xs
    ids, valid, cell, _, count = cell_nodes(x, alive, grid, npc_max)
    _, _, complex_ = _split_context(normals, ids, valid, basis)
    c = jnp.minimum(cell, count.shape[0] - 1)
    return carry, jnp.sum(alive & complex_[c] & (cell < count.shape[0]))


@functools.partial(jax.jit, static_argnames=("npc_max", "basis"))
def complex_load(records, normals, grid, npc_max, basis):
    """``i32[M]``: per captured step, the alive slots in the complex cells of a split
    per-cell fit (``basis`` ``"cell:...:align<r>"``) -- the queries per field evaluation
    that its per-query fallback answers."""
    body = functools.partial(_complex_load, grid=grid, npc_max=npc_max, basis=basis)
    _, load = lax.scan(body, None, (records.x, records.alive, normals))
    return load


FALLBACK_BUCKET = 256


def fallback_capacity(records, normals, grid, npc_max, basis):
    """The static ``fallback_capacity`` of :func:`backward_virtual` for ``basis`` (host):
    the largest :func:`complex_load` over the steps rounded up to
    :data:`FALLBACK_BUCKET` (one compilation for similar loads); 0 when the basis is not a
    split per-cell fit with a fallback (no normals, ``align`` 0, ``nofallback``)."""
    if normals is None or not cell_fit.is_cell(basis):
        return 0
    spec = cell_fit.parse(basis)
    if spec.align <= 0.0 or not spec.fallback:
        return 0
    worst = int(jnp.max(complex_load(records, normals, grid, npc_max, basis)))
    return -(-worst // FALLBACK_BUCKET) * FALLBACK_BUCKET if worst else 0


def _chunk_query(
    args, node_x, node_v, node_valid, node_val, node_class, scale, k, basis, gradients, halfspace
):
    """Field values (and slopes) at one chunk of queries, each from its own cell's nodes;
    ``direction`` (``f64[chunk, 3]``, zero = no restriction) keeps only the nodes with
    ``v . direction > 0`` -- the one-sided fit of the shape adjoint -- and, with
    ``halfspace`` (a length), only those with ``(x_node - x_q) . direction >= -halfspace``
    (not behind the surface's tangent plane: a thin body's far side); ``node_class`` /
    ``q_class`` (``bool``) keep only the nodes of the query's class."""
    c, q_x, q_v, direction, q_class = args
    valid = node_valid[c]
    one_sided = jnp.any(direction != 0.0, axis=1)
    same_side = jnp.einsum("qpk,qk->qp", node_v[c], direction) > 0.0
    if halfspace is not None:
        d = q_x.shape[1]
        ahead = jnp.einsum("qpk,qk->qp", node_x[c] - q_x[:, None, :], direction[:, :d])
        same_side = same_side & (ahead >= -halfspace)
    valid = valid & (same_side | ~one_sided[:, None])
    if node_class is not None:
        valid = valid & (node_class[c] == q_class[:, None])
    if cell_fit.is_cell(basis):
        out = _polynomial_query(node_x[c], node_v[c], valid, node_val[c], q_x, q_v, scale, basis)
        return out if gradients else out[0]
    value, grad_x, grad_v = field_query(
        node_x[c], node_v[c], valid, node_val[c], q_x, q_v, scale, k, basis
    )
    return (value, grad_x, grad_v) if gradients else value


def _polynomial_query(node_x, node_v, valid, node_val, q_x, q_v, scale, basis):
    """``(value, grad_x, grad_v)`` at each query from the per-cell polynomial of a
    ``"cell:..."`` ``basis`` (:mod:`cell_fit`) fitted to that query's own selected nodes
    ``(Q, P, .)`` -- every valid node, unweighted, with the basis' order cap and leverage
    guard -- and differentiated analytically (:func:`cell_fit.evaluate`): the collision
    field's polynomial as a slope estimator (2026-10-29).  A ``pool<r>`` basis weights the
    pooled block by its tricube around the query's own position."""
    own = {}
    if cell_fit.parse(basis).pool:
        own = {"own_x": q_x[:, None, :], "own_valid": jnp.ones((q_x.shape[0], 1), bool)}
    field = cell_fit.fit(node_x, node_v, valid, node_val, scale, basis, **own)
    rows = jnp.arange(q_x.shape[0])
    return cell_fit.evaluate(field, rows, q_x, q_v, scale, basis, gradients=True)


def fit_chunk(q, width, chunk):
    """Queries per :func:`cell_query` pass: at most ``chunk``, at most ``FIT_PAIRS / width``
    (at least 64) for a node table ``width`` wide, and the ``Q`` queries split into equal
    passes so the padding stays below one pass.  Each query's fit reads only the tables
    and its own state, so the chunking changes nothing but XLA's batching."""
    cap = max(1, min(chunk, max(64, FIT_PAIRS // max(width, 1)), q))
    passes = -(-q // cap)
    return -(-q // passes)


def cell_query(
    tables,
    q_c,
    q_x,
    q_v,
    scale,
    k,
    basis,
    chunk,
    gradients=False,
    direction=None,
    q_class=None,
    halfspace=None,
):
    """Field value at ``Q`` queries in cells ``q_c`` from per-cell node tables ``(C, P, .)``,
    at most ``chunk`` queries at a time (the per-query node gather is ``P x (D+3)``
    floats; :func:`fit_chunk` bounds the pass by ``FIT_PAIRS``).
    ``gradients=True`` returns ``(value, grad_x, grad_v)`` instead; ``direction``
    ``f64[Q, 3]`` restricts a query's nodes to those with ``v . direction > 0`` (rows of
    zeros: no restriction) and, with ``halfspace``, to the nodes not more than that far
    behind the query's plane normal to ``direction``.  A fifth table ``node_class`` (``bool[C, P]``) with ``q_class``
    ``bool[Q]`` restricts them to the query's class -- the stratified fit of the shape
    adjoint (:func:`node_gradients`).  A node-value table with trailing columns
    ``(C, P, R)`` gives ``R`` columns per query (:func:`field_query`)."""
    node_x, node_v, node_valid, node_val, *rest = tables
    node_class = rest[0] if rest else None
    q = q_c.shape[0]
    chunk = fit_chunk(q, node_valid.shape[1], chunk)
    pad = (-q) % chunk
    c = jnp.pad(q_c, (0, pad)).reshape(-1, chunk)
    x = jnp.pad(q_x, ((0, pad), (0, 0))).reshape(-1, chunk, q_x.shape[1])
    v = jnp.pad(q_v, ((0, pad), (0, 0))).reshape(-1, chunk, 3)
    if direction is None:
        direction = jnp.zeros((q, 3))
    d = jnp.pad(direction, ((0, pad), (0, 0))).reshape(-1, chunk, 3)
    if q_class is None:
        q_class = jnp.zeros(q, dtype=bool)
    qc = jnp.pad(q_class, (0, pad)).reshape(-1, chunk)
    body = functools.partial(
        _chunk_query,
        node_x=node_x,
        node_v=node_v,
        node_valid=node_valid,
        node_val=node_val,
        node_class=node_class,
        scale=scale,
        k=k,
        basis=basis,
        gradients=gradients,
        halfspace=halfspace,
    )
    out = lax.map(body, (c, x, v, d, qc))
    return jax.tree.map(functools.partial(_unchunk, q=q), out)


def _unchunk(a, q):
    """``(passes, chunk, ...)`` -> the first ``q`` rows of ``(passes * chunk, ...)``."""
    return a.reshape(-1, *a.shape[2:])[:q]


def pooled_tables(tables, grid, pool, cells=None):
    """Per-cell node tables ``(C, P, .)`` widened to the ``(2 pool + 1)^D`` block of cells
    around each cell, ``(C, B P, .)``: the pooled fit draws its ``k`` nearest nodes from the
    neighbourhood instead of the cell alone (more nodes and a wider position range for the
    slope, at the cost of a farther, less local kernel).  Block positions outside the box
    contribute no valid nodes.  ``cells`` ``i32[K]`` widens only those cells (row ``j`` of the
    result is ``cells[j]``'s block)."""
    node_x, node_v, node_valid, node_val, *rest = tables
    nbr, exists = grid_neighbourhood(grid, pool)  # (C, B)
    if cells is not None:
        nbr, exists = nbr[cells], exists[cells]
    c, b = nbr.shape

    def widen(t):
        out = t[nbr]  # (C, B, P, ...)
        return out.reshape(c, b * out.shape[2], *out.shape[3:])

    valid = (node_valid[nbr] & exists[:, :, None]).reshape(c, -1)
    return (widen(node_x), widen(node_v), valid, widen(node_val), *[widen(t) for t in rest])


def node_gradients(
    x,
    v,
    alive,
    lam,
    grid,
    npc_max,
    scale,
    k,
    basis="quadratic_full",
    chunk=4096,
    direction=None,
    pool=0,
    klass=None,
    queries=None,
    halfspace=None,
):
    """``(grad_x f64[N, D], grad_v f64[N, 3])`` of the carried adjoint at every alive slot's
    own state, from the per-cell fit of ``lam`` over the slot's cell (the slot is one of
    its own nodes).  Zero on dead slots.  These are the cotangents the shape adjoint
    (:mod:`jax_dsmc.shape`) puts on the replayed wall map's post-move state.

    ``direction`` ``f64[N, 3]`` (the normal of the slot's last hit; zero rows = none) makes
    the fit **one-sided**: only nodes moving away from that surface (``v . n > 0``) are
    used.  Next to a wall the post-move population splits into particles about to hit
    (value of the hit's reward) and particles that just left (value of what follows),
    an O(reward) contrast that a polynomial in ``v`` cannot represent and that is
    correlated with the distance to the wall (within one flight of it the approaching
    side is depleted) -- the x-slope of a two-sided fit then reads the population mix,
    not ``grad_x lambda`` (gate G3: the sign of the plane's shear gradient came out
    wrong).  ``pool = 1`` draws the ``k`` nearest nodes from the ``3^D`` block of cells
    around the slot's cell (:func:`pooled_tables`) instead of the cell alone.
    ``halfspace`` (in cells; one-sided queries only) also drops the nodes farther than
    that behind the plane through the query normal to ``direction``: a pooled block around
    a body thinner than the block reaches the body's far side, whose particles have their
    own, unrelated values (the squash ellipse at ``s = 0.35``: ``one_pool`` -15 %).

    ``klass`` ``bool[N]`` **stratifies** the fit: a slot's nodes are the slots of its own
    class.  **Diagnostic only** when the class depends on the particle's future (the shape
    adjoint's hit-ahead class): the fit then drops ``grad P (lam_1 - lam_0)`` and is biased
    by construction (``shape.gradient``, ``value_class``).  The shape adjoint passes "has a body hit ahead in the window": a particle's
    value jumps by its next hit's reward across the hit / no-hit boundary in phase space,
    and next to a curved surface the velocity half-space of ``direction`` does not
    separate the two (particles leaving a cap's tangent plane still hit the flat face
    below it) -- the fitted ``grad_v lambda`` of a reflected particle then reads the
    approaching population's ``d(2 m v_x)/dv``, a -27 % bias on the specular capsule
    whose true post-hit value gradient is zero (review section 7.8).

    ``queries`` ``i32[Q]`` (slot ids, ``-1`` = unused entry) fits only at those slots and
    returns zeros elsewhere; the nodes are still every alive slot.  Each query's fit reads
    only the node tables and its own state, so the fitted values at the queried slots are
    those of the full call (up to the last bit, if XLA batches a different query count
    differently) -- the shape adjoint uses it for the slots whose post-move state depends
    on the geometry (:func:`jax_dsmc.shape.gradient`, ``hit_fits``).
    """
    ids, valid, cell, _, count = cell_nodes(x, alive, grid, npc_max)
    c = jnp.minimum(cell, count.shape[0] - 1)
    halfspace = _halfspace_length(halfspace, grid)
    tables = (x[ids], v[ids], valid, lam[ids])
    if klass is not None:
        tables = (*tables, klass[ids])
    if pool and queries is not None:
        return _pooled_query_gradients(
            tables, grid, pool, c, x, v, alive, scale, k, basis, chunk, direction, klass,
            queries, halfspace,
        )  # fmt: skip
    if pool:
        tables = pooled_tables(tables, grid, pool)
    if queries is None:
        _, grad_x, grad_v = cell_query(
            tables, c, x, v, scale, k, basis, chunk, gradients=True, direction=direction,
            q_class=klass, halfspace=halfspace,
        )  # fmt: skip
        return jnp.where(alive[:, None], grad_x, 0.0), jnp.where(alive[:, None], grad_v, 0.0)
    return _query_gradients(tables, c, x, v, alive, scale, k, basis, chunk, direction, klass,
                            queries, halfspace)  # fmt: skip


POOL_QUERY_CELLS = 256
"""Cells widened per step by the pooled slope fits at the hit queries (the cells next to the
body; a step with more distinct query cells widens every cell instead, exactly)."""


def _query_gradients(tables, c, x, v, alive, scale, k, basis, chunk, direction, klass, queries,
                     halfspace):  # fmt: skip
    """:func:`node_gradients`' query branch on prepared ``tables`` indexed by ``c``."""
    n = x.shape[0]
    used = queries >= 0
    q = jnp.where(used, queries, 0)
    _, qx, qv = cell_query(
        tables, c[q], x[q], v[q], scale, k, basis, chunk, gradients=True,
        direction=None if direction is None else direction[q],
        q_class=None if klass is None else klass[q], halfspace=halfspace,
    )  # fmt: skip
    keep = (used & alive[q])[:, None]
    target = jnp.where(used, queries, n)
    grad_x = jnp.zeros((n, x.shape[1])).at[target].set(jnp.where(keep, qx, 0.0), mode="drop")
    grad_v = jnp.zeros((n, 3)).at[target].set(jnp.where(keep, qv, 0.0), mode="drop")
    return grad_x, grad_v


def _pooled_query_gradients(tables, grid, pool, c, x, v, alive, scale, k, basis, chunk,
                            direction, klass, queries, halfspace):  # fmt: skip
    """The pooled fits at the queries with only the queried cells widened (at most
    :data:`POOL_QUERY_CELLS` of them; more, and every cell is widened): the same fits, without
    widening the whole grid every step."""
    n_cells = tables[0].shape[0]
    used = queries >= 0
    cq = jnp.where(used, c[jnp.where(used, queries, 0)], n_cells)
    cap = min(POOL_QUERY_CELLS, n_cells)
    cells = jnp.unique(cq, size=cap, fill_value=n_cells)  # sorted; n_cells = none
    pos = jnp.clip(jnp.searchsorted(cells, c), 0, cap - 1)  # every slot's row (queries' exact)
    fits = cells[jnp.clip(jnp.searchsorted(cells, cq), 0, cap - 1)] == cq
    common = (x, v, alive, scale, k, basis, chunk, direction, klass, queries, halfspace)

    def compact(_):
        wide = pooled_tables(tables, grid, pool, jnp.minimum(cells, n_cells - 1))
        return _query_gradients(wide, pos, *common)

    def full(_):
        return _query_gradients(pooled_tables(tables, grid, pool), c, *common)

    return lax.cond(step_axis.all_(fits | ~used), compact, full, None)


def value_query(
    x,
    v,
    alive,
    lam,
    grid,
    q_x,
    q_v,
    npc_max,
    scale,
    k,
    basis="quadratic_full",
    chunk=4096,
    direction=None,
    pool=0,
    halfspace=None,
):
    """``f64[Q]``: the carried adjoint at arbitrary query states ``(q_x, q_v)`` from the
    per-cell fit of ``lam`` over the population ``(x, v, alive)`` -- the value of a
    *virtual* post-move state (the silhouette term of :mod:`jax_dsmc.shape` asks for the
    value a particle would carry had its hit / miss gone the other way).  Queries outside
    the box are clipped to it.  ``direction`` / ``pool`` / ``halfspace`` as
    :func:`node_gradients`."""
    ids, valid, _cell, _, count = cell_nodes(x, alive, grid, npc_max)
    tables = (x[ids], v[ids], valid, lam[ids])
    if pool:
        tables = pooled_tables(tables, grid, pool)
    lo, hi = grid.lo, grid.hi
    q_x = jnp.clip(q_x, lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
    q_c = jnp.minimum(collision_cell(q_x, grid), count.shape[0] - 1)
    return cell_query(
        tables, q_c, q_x, q_v, scale, k, basis, chunk, direction=direction,
        halfspace=_halfspace_length(halfspace, grid),
    )  # fmt: skip


def _halfspace_length(halfspace, grid):
    """``halfspace`` (in cells: the smallest cell spacing) as a length, or ``None``."""
    return None if halfspace is None else halfspace * jnp.min(spacing(grid))


def staged_velocities(pairs_k, v_pre, v_post, cell, substeps):
    """Velocities at the ``substeps + 1`` slot boundaries of a step (host
    ``_staged_velocities``): ``v_pre`` overwritten by the post-collision velocity of each
    particle's last accepted slot before the boundary.  ``f64[substeps + 1, N, 3]``."""
    n = v_pre.shape[0]
    pi, pj = pairs_k.pi, pairs_k.pj
    active = pi < n
    accept = pairs_k.accept & active
    n_att = jnp.sum(active, axis=0)
    a = jnp.arange(pi.shape[0], dtype=jnp.int32)[:, None]
    frac = a / jnp.maximum(n_att, 1)[None, :]
    c = jnp.minimum(cell, pi.shape[1] - 1)
    own = jnp.arange(n, dtype=jnp.int32)
    stages = [v_pre]
    for b in range(1, substeps):
        slot = jnp.where(accept & (frac < b / substeps), a, -1)
        last_i = jax.ops.segment_max(slot.ravel(), pi.ravel(), num_segments=n + 1)
        last_j = jax.ops.segment_max(slot.ravel(), pj.ravel(), num_segments=n + 1)
        last = jnp.maximum(last_i, last_j)[:n]
        la = jnp.maximum(last, 0)
        was_i = pi[la, c] == own
        vel = jnp.where(was_i[:, None], pairs_k.vi_post[la, c], pairs_k.vj_post[la, c])
        stages.append(jnp.where((last >= 0)[:, None], vel, v_pre))
    stages.append(v_post)
    return jnp.stack(stages)


class _Hybrid(NamedTuple):
    """A split per-cell field whose complex cells' queries are answered by the per-query
    one-sided fit from the own-cell ``tables``, at most ``capacity`` queries per call."""

    split: cell_fit.SplitField
    tables: tuple
    capacity: int


def _field(tables, scale, basis, grid, split=None, fallback=0):
    """The fitted field of node tables ``(x, v, valid, lam)`` ``(C, P, .)``: the tables
    themselves for the per-query fit (:func:`cell_query`), or a per-cell
    :class:`cell_fit.CellField` for a ``"cell:..."`` basis (a pooled one over the
    :func:`pooled_tables` block of ``grid``).  With surface normals, ``split`` is
    :func:`_split_context`'s ``(node_normal, cell normal, complex)``: the
    :class:`cell_fit.SplitField`, and with ``fallback`` (a query capacity) the per-query
    fit in the complex cells (:class:`_Hybrid`)."""
    if not cell_fit.is_cell(basis):
        return tables
    pool = cell_fit.parse(basis).pool
    own = {}
    wide = tables if split is None else (*tables, split[0])
    if pool:
        own = {"own_x": tables[0], "own_valid": tables[2]}
        wide = pooled_tables(wide, grid, pool)
    if split is None:
        return cell_fit.fit(*wide, scale, basis, **own)
    field = cell_fit.fit_split(*wide, split[1], split[2], scale, basis, **own)
    return _Hybrid(field, tables, fallback) if fallback else field


def _split_context(normals, ids, valid, basis):
    """``(node_normal f64[C, P, 3], normal f64[C, 3], complex bool[C])`` of a step's slot
    normals for a split per-cell fit (:func:`cell_fit.cell_normals`); ``None`` without
    normals or for the per-query fit."""
    if normals is None or not cell_fit.is_cell(basis):
        return None
    node_normal = jnp.where(valid[..., None], normals[ids].astype(jnp.float64), 0.0)
    normal, complex_ = cell_fit.cell_normals(node_normal, valid, cell_fit.parse(basis).align)
    return node_normal, normal, complex_


def _n_cells(field):
    if isinstance(field, _Hybrid):
        return field.tables[2].shape[0]
    if isinstance(field, cell_fit.SplitField):
        return field.normal.shape[0]
    if isinstance(field, cell_fit.CellField):
        return field.centre.shape[0]
    return field[2].shape[0]


def _query(field, cell, q_x, q_v, q_normal, scale, k, basis, chunk, layout=None):
    """Field values at queries in cells ``cell`` (the sentinel ``C`` for dead slots) with
    their surface normals ``q_normal`` (``None``: two-sided) from :func:`_field`'s output.
    ``layout = (ids, rank)``: the queries are slot-aligned (:func:`cell_fit.evaluate`)."""
    n_cells = _n_cells(field)
    c = jnp.minimum(cell, n_cells - 1)
    direction = _side(q_v, q_normal)
    if not cell_fit.is_cell(basis):
        return cell_query(field, c, q_x, q_v, scale, k, basis, chunk, direction=direction)
    live = cell < n_cells
    if not isinstance(field, _Hybrid):
        return cell_fit.evaluate(
            field, c, q_x, q_v, scale, basis, q_normal=q_normal, q_valid=live, layout=layout
        )
    value = cell_fit.evaluate(
        field.split, c, q_x, q_v, scale, basis, q_normal=q_normal, q_valid=live, layout=layout
    )
    q = q_x.shape[0]
    need = field.split.complex[c] & live
    idx = jnp.nonzero(need, size=field.capacity, fill_value=q)[0]
    safe = jnp.minimum(idx, q - 1)
    sub = cell_query(
        field.tables, c[safe], q_x[safe], q_v[safe], scale, k, "quadratic_full", chunk,
        direction=direction[safe],
    )  # fmt: skip
    return value.at[idx].set(sub, mode="drop")


def _pair_kernel(vi, vj, u_chi, u_eps, sp):
    q = collision_rate(vi, vj, 0, 0, sp)
    vi_n, vj_n = scatter(vi, vj, 0, 0, sp, u_chi, u_eps)
    return q, vi_n, vj_n


def _partner_pass(
    carry,
    key,
    x,
    v_nodes,
    field,
    f_node,
    ids,
    cell,
    rank,
    count,
    weight,
    sp,
    scale,
    k,
    basis,
    chunk,
    pair_score,
    normals,
):
    """One virtual partner per particle: ``(delta lam, kernel term)`` accumulated."""
    delta, kern = carry
    n = x.shape[0]
    k_j, k_chi, k_eps = jax.random.split(key, 3)
    c = jnp.minimum(cell, count.shape[0] - 1)
    # a partner among the OTHER members of the cell: index shift, never self
    u = jax.random.randint(k_j, (n,), 0, jnp.maximum(count[c] - 1, 1))
    j = ids[c, (rank + 1 + u) % jnp.maximum(count[c], 1)]
    u_chi = jax.random.uniform(k_chi, (n,))
    u_eps = jax.random.uniform(k_eps, (n,))
    if cell_fit.is_cell(basis) and cell_fit.parse(basis).rb:
        q = collision_rate(v_nodes, v_nodes[j], 0, 0, sp)
        vi_pts, vj_pts = _scatter_sigma(v_nodes, v_nodes[j], sp)
    else:
        q, vi_n, vj_n = jax.vmap(functools.partial(_pair_kernel, sp=sp))(
            v_nodes, v_nodes[j], u_chi, u_eps
        )
        vi_pts, vj_pts = (vi_n,), (vj_n,)
    # j shares i's cell (the sentinel for a dead i, whose increment is dropped)
    normal_j = None if normals is None else normals[j]
    query = functools.partial(
        _query, field, scale=scale, k=k, basis=basis, chunk=chunk, layout=(ids, rank)
    )
    f_i = sum(query(cell, x, p, normals) for p in vi_pts) / len(vi_pts)
    f_j = sum(query(cell, x[j], p, normal_j) for p in vj_pts) / len(vj_pts)
    four = f_i + f_j - f_node - f_node[j]
    w = weight * q
    delta = delta + _per_slot(w, four) * four
    if pair_score is not None:
        score = jax.vmap(pair_score)(v_nodes, v_nodes[j])
        kern = kern + jnp.sum(_per_slot(0.5 * w * score, four) * four, axis=0)
    return (delta, kern), None


def _scatter_sigma(vi, vj, sp):
    """Six post-collision velocity pairs of ``(vi, vj)`` (``f64[N, 3]``, species 0) whose
    equal-weight average of any function quadratic in the post-collision velocity is its
    expectation over the VSS scattering angle -- the symmetric sigma points of the
    post-collision relative direction ``n``, whose mean is ``E[cos chi] g_hat`` and
    covariance ``var(cos chi) g_hat g_hat^T + (1 - E[cos^2 chi]) / 2 (I - g_hat g_hat^T)``
    (``cos chi = 2 u^(1/alpha) - 1``: ``E[cos chi] = (alpha - 1)/(alpha + 1)``,
    ``E[cos^2 chi] = 1 - 4 alpha / ((alpha + 1)(alpha + 2))``).  Returns two tuples of six
    ``f64[N, 3]``: the ``i`` and ``j`` velocities (:func:`jax_dsmc.collide.vss.scatter`:
    ``v_cm +- m_other / (m_i + m_j) |v_r| n``)."""
    alpha = sp.alpha[0, 0]
    vr = vi - vj
    g = jnp.sqrt(jnp.sum(vr * vr, axis=1, keepdims=True))
    ghat = vr / jnp.maximum(g, jnp.finfo(vr.dtype).tiny)
    # two unit vectors perpendicular to g_hat (any pair: the law is axisymmetric)
    helper = jnp.where(
        jnp.abs(ghat[:, :1]) < 0.9, jnp.asarray([[1.0, 0.0, 0.0]]), jnp.asarray([[0.0, 1.0, 0.0]])
    )
    e1 = helper - jnp.sum(helper * ghat, axis=1, keepdims=True) * ghat
    e1 = e1 / jnp.sqrt(jnp.sum(e1 * e1, axis=1, keepdims=True))
    e2 = jnp.cross(ghat, e1)
    mean_cos = (alpha - 1.0) / (alpha + 1.0)
    mean_cos2 = 1.0 - 4.0 * alpha / ((alpha + 1.0) * (alpha + 2.0))
    s_par = jnp.sqrt(jnp.maximum(mean_cos2 - mean_cos**2, 0.0))
    s_perp = jnp.sqrt((1.0 - mean_cos2) / 2.0)
    vcm = 0.5 * (vi + vj)  # equal masses (species 0 with itself)
    root3 = jnp.sqrt(3.0)
    vi_pts, vj_pts = [], []
    for axis, spread in ((ghat, s_par), (e1, s_perp), (e2, s_perp)):
        for sign in (1.0, -1.0):
            n = mean_cos * ghat + sign * root3 * spread * axis
            vi_pts.append(vcm + 0.5 * g * n)
            vj_pts.append(vcm - 0.5 * g * n)
    return tuple(vi_pts), tuple(vj_pts)


def _side(v, normals):
    """One-sided fit direction of a query: ``sign(v . n) n`` (zero rows: unrestricted)."""
    if normals is None:
        return None
    return jnp.sign(jnp.sum(v * normals, axis=1))[:, None] * normals


def _boundary_move(
    lam, x, v_prev, v_nodes, ids, valid, cell, rank, moved, normals, scale, k, basis, chunk,
    grid, split, fallback,
):  # fmt: skip
    tables = (x[ids], v_nodes[ids], valid, lam[ids])
    field = _field(tables, scale, basis, grid, split, fallback)
    layout = (ids, rank)
    f_prev = _query(field, cell, x, v_prev, normals, scale, k, basis, chunk, layout)
    f_here = _query(field, cell, x, v_nodes, normals, scale, k, basis, chunk, layout)
    return jnp.where(_per_slot(moved, lam), lam + f_prev - f_here, lam)


def _boundary_skip(lam, *args):
    del args
    return lam


def _substep(
    carry,
    xs,
    x,
    ids,
    valid,
    cell,
    rank,
    count,
    weight,
    alive,
    sp,
    scale,
    k,
    basis,
    chunk,
    partners,
    pair_score,
    normals,
    grid,
    split,
    fallback,
):
    """One backward substep: ``lam += (dt/S) L* lam`` with ``partners`` virtual partners per
    particle, then the value move of particles whose state differs across the boundary."""
    lam, key = carry
    v_nodes, v_prev = xs
    key, sub = jax.random.split(key)
    field = _field((x[ids], v_nodes[ids], valid, lam[ids]), scale, basis, grid, split, fallback)
    f_node = _query(field, cell, x, v_nodes, normals, scale, k, basis, chunk, (ids, rank))
    body = functools.partial(
        _partner_pass,
        x=x,
        v_nodes=v_nodes,
        field=field,
        f_node=f_node,
        ids=ids,
        cell=cell,
        rank=rank,
        count=count,
        weight=weight,
        sp=sp,
        scale=scale,
        k=k,
        basis=basis,
        chunk=chunk,
        pair_score=pair_score,
        normals=normals,
    )
    (delta, kern), _ = lax.scan(
        body, (jnp.zeros_like(lam), jnp.zeros(lam.shape[1:])), jax.random.split(sub, partners)
    )
    delta = jnp.where(_per_slot(alive, delta), delta, 0.0)
    lam = lam + delta
    moved = alive & jnp.any(v_prev != v_nodes, axis=1)
    lam = lax.cond(
        jnp.any(moved),
        functools.partial(
            _boundary_move,
            scale=scale,
            k=k,
            basis=basis,
            chunk=chunk,
            grid=grid,
            split=split,
            fallback=fallback,
        ),
        _boundary_skip,
        lam,
        x,
        v_prev,
        v_nodes,
        ids,
        valid,
        cell,
        rank,
        moved,
        normals,
    )
    return (lam, key), (kern, delta)


def _step_pullback_virtual(
    carry,
    xs,
    grid,
    npc_max,
    scale,
    k,
    basis,
    chunk,
    sp,
    substeps,
    partners,
    nodes,
    pair_score,
    collision_out,
    columns_out,
    fallback_capacity,
):
    """One captured step backward under the virtual pull-back, then the move pull-back."""
    mu, key = carry
    rec_k, phi_k, r_k, normals_k = xs
    mu = _add_step_reward(mu, rec_k, phi_k)
    key, step_key = jax.random.split(key)
    n = rec_k.x.shape[0]
    ids, valid, cell, rank, count = cell_nodes(rec_k.x, rec_k.alive, grid, npc_max)
    c = jnp.minimum(cell, count.shape[0] - 1)
    n_att = jnp.sum(rec_k.pairs.pi < n, axis=0)
    per_particle = jnp.where(count > 0, 2.0 * n_att / jnp.maximum(count, 1), 0.0)
    weight = per_particle[c] / rec_k.sigma[c] / (partners * substeps)
    weight = jnp.where(rec_k.alive & (count[c] >= 2), weight, 0.0)
    if nodes == "staged":
        stages = staged_velocities(rec_k.pairs, rec_k.v, step_v_post(rec_k), cell, substeps)
    else:
        stages = jnp.concatenate(
            (rec_k.v[None], jnp.broadcast_to(step_v_post(rec_k), (substeps, n, 3)))
        )
    body = functools.partial(
        _substep,
        x=rec_k.x,
        ids=ids,
        valid=valid,
        cell=cell,
        rank=rank,
        count=count,
        weight=weight,
        alive=rec_k.alive,
        sp=sp,
        scale=scale,
        k=k,
        basis=basis,
        chunk=chunk,
        partners=partners,
        pair_score=pair_score,
        normals=normals_k,
        grid=grid,
        split=_split_context(normals_k, ids, valid, basis),
        fallback=fallback_capacity,
    )
    (lam, _), (kern, delta) = lax.scan(
        body, (mu, step_key), (stages[1:], stages[:-1]), reverse=True
    )
    mu_prev, lam = _move_pullback(lam, rec_k, r_k)
    coll = None
    if collision_out:
        # the first value column only (the objective's; the others are auxiliary values):
        # per slot in float32 (it feeds the along-flight identity's parallel component, a
        # memory peak in f64 at the production occupancy), per collision cell in f64
        per = jnp.sum(delta, axis=0)
        per = per[..., 0] if per.ndim > 1 else per
        cells = jnp.zeros(count.shape[0] + 1).at[cell].add(0.5 * per)[:-1]
        coll = (per.astype(jnp.float32), cells)
    if columns_out:
        # one stacked output per value column: the [M, N, R] array is never formed
        lam = tuple(lam[..., r] for r in range(lam.shape[-1]))
    return (mu_prev, key), (lam, jnp.sum(kern, axis=0), coll)


@functools.partial(
    jax.jit,
    static_argnames=(
        "npc_max",
        "k",
        "basis",
        "chunk",
        "substeps",
        "partners",
        "nodes",
        "pair_score",
        "collision_out",
        "columns_out",
        "fallback_capacity",
    ),
)
def backward_virtual(
    records,
    initial_alive,
    final_state,
    phi_final,
    grid,
    scale,
    sp,
    key,
    npc_max,
    k=32,
    basis="quadratic_full",
    chunk=4096,
    substeps=4,
    partners=8,
    nodes="post",
    pair_score=None,
    phi_step=None,
    hit_reward=None,
    normals=None,
    collision_out=False,
    columns_out=False,
    fallback_capacity=0,
):
    """Reverse scan over the captured steps with the virtual-partner collision pull-back.

    ``sp`` is the species table of the forward (the virtual pairs use its VSS kernel),
    ``key`` seeds the partner and angle draws, ``pair_score(v_i, v_j)`` an optional
    per-pair score whose weighted sum over the virtual pairs is returned per step.
    ``phi_step`` / ``hit_reward`` are the value-form rewards of :func:`backward`.
    ``normals`` ``f64[M, N, 3]`` (the nearest surface normal at each slot's post-move
    position, zero rows away from surfaces: :func:`jax_dsmc.mesh.geometry.nearest_normal`)
    makes every fit query one-sided -- the nodes on the query velocity's side of
    ``v . n = 0`` only; ``None`` keeps the two-sided fit (bitwise the old results).
    ``basis`` ``"cell:..."`` replaces the per-query fit by one fit per cell
    (:mod:`jax_dsmc.cell_fit`; split at ``v . n = 0`` with ``normals``); its complex cells
    (``align``) are answered by the per-query one-sided fit, at most
    ``fallback_capacity`` queries per field evaluation (:func:`fallback_capacity`; 0: none,
    the split fit everywhere).
    ``collision_out=True`` also returns each slot's per-step collision increment of the
    first value column and its cell sums (:attr:`Backward.collision`, ``f32[M, N]``;
    :attr:`Backward.cell_collision`, ``f64[M, C]``).  ``columns_out=True`` (value columns
    only) returns ``lam`` as a tuple of the ``R`` columns ``f64[M, N]`` -- the stacked
    ``f64[M, N, R]`` and the copies its slicing makes are never formed (the shape seed's
    memory peak at the production occupancy).

    **Several value fields in one sweep.**  ``phi_final`` ``f64[N, R]``, ``phi_step``
    ``f64[M, N, R]`` and ``hit_reward`` ``f64[M, K, N, R]`` carry ``R`` reward columns;
    ``lam`` is then ``f64[N, R]`` throughout and every output gains the trailing axis
    (:class:`Backward`).  The sweep is linear in the rewards, and nothing it decides --
    the cell tables, the partners and their draws, the scattered velocities, the nodes
    and weights of each fit -- depends on the values, so column ``r`` is the single-column
    backward of reward column ``r`` (to the solve's rounding, ~1e-13).  What the columns
    share is most of the cost: every fit forms its nearest nodes, design and normal
    matrix and its factorisation once and solves for all columns (:func:`field_query`),
    and the partner draws and VSS scatterings are made once.  The shape seed's two value
    backwards (hit rewards, timing rewards: ``examples/ensemble_mesh._backwards``) run
    as one ``R = 2`` sweep, 2.1x faster than two (2026-10-02, PSEUDOCODE section 9.7.1).
    """
    mu = jnp.where(_per_slot(final_state.alive, phi_final), phi_final, 0.0)
    body = functools.partial(
        _step_pullback_virtual,
        grid=grid,
        npc_max=npc_max,
        scale=scale,
        k=k,
        basis=basis,
        chunk=chunk,
        sp=sp,
        substeps=substeps,
        partners=partners,
        nodes=nodes,
        pair_score=pair_score,
        collision_out=collision_out,
        columns_out=columns_out,
        fallback_capacity=fallback_capacity,
    )
    (mu0, _), (lam, kern, coll) = lax.scan(
        body, (mu, key), (records, phi_step, hit_reward, normals), reverse=True
    )
    m, a, c = records.pairs.pi.shape
    lambda0 = jnp.where(_per_slot(initial_alive, mu0), mu0, 0.0)
    per, cells = (None, None) if coll is None else coll
    return Backward(lambda0, lam, jnp.zeros((m, a, c)), kern, per, cells)
