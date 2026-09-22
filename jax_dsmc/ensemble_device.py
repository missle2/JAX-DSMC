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

Memory: the whole ``EventRecord`` stack is resident (``O(M·N)``); block re-capture of
the forward is the known fix for long horizons and is not done here.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from jax_dsmc import sort
from jax_dsmc.collide.vss import collision_rate, scatter
from jax_dsmc.grid import cell_neighbourhood as grid_neighbourhood
from jax_dsmc.grid import cell_of
from jax_dsmc.state import ParticleState

jax.config.update("jax_enable_x64", True)

RIDGE_REL = 1e-8


class Backward(NamedTuple):
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


def field_query(node_x, node_v, node_valid, node_val, q_x, q_v, scale, k, basis="quadratic_full"):
    """Weighted local-polynomial fit of ``node_val`` at each query; returns
    ``(value, grad_x, grad_v)`` -- the intercept and the linear slopes of the fit at the
    query, per unscaled coordinate (``grad_x`` is ``f64[Q, D]``, ``grad_v`` ``f64[Q, 3]``).

    Shapes: nodes ``(Q, P, ·)`` gathered per query, queries ``(Q, ·)``.  Exactly the
    host :class:`ensemble_backward.LocalFit`: the ``k`` nearest valid nodes in the scaled
    ``(x, v)`` metric, weights ``1/(1+(d/d_k)^2)``, ridge ``1e-8`` (relative) on the slopes.
    The shape adjoint consumes ``grad_x`` (the position cotangent of the replayed wall
    map); the pathwise and collision terms consume the value and ``grad_v``.
    """
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
    values = jnp.take_along_axis(node_val, ids, axis=1)
    design = _design(chosen - q[:, None, :], basis)
    design = jnp.where(good[..., None], design, 0.0)
    matrix = jnp.einsum("qki,qk,qkj->qij", design, weights, design)
    ridge = jnp.full(design.shape[-1], RIDGE_REL).at[0].set(0.0)
    diag_scale = jnp.trace(matrix, axis1=1, axis2=2) / design.shape[-1]
    matrix = matrix + ridge[None] * diag_scale[:, None, None] * jnp.eye(design.shape[-1])
    # a query with no valid node at all gets value 0 rather than a singular solve
    empty = ~jnp.any(good, axis=1)
    matrix = matrix.at[:, 0, 0].add(empty.astype(jnp.float64))
    rhs = jnp.einsum("qki,qk,qk->qi", design, weights, jnp.where(good, values, 0.0))
    beta = jnp.linalg.solve(matrix, rhs[..., None])[..., 0]
    value = jnp.where(empty, 0.0, beta[:, 0])
    grad_x = beta[:, 1 : 1 + dim] / scale[:dim]
    grad_v = beta[:, 1 + dim : 1 + dim + 3] / scale[-3:]
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


def _add_step_reward(mu, rec_k, phi_k):
    """Value form of a per-step reward: the carried value at step ``k``'s post-collision
    state includes the reward collected there, so ``phi_k`` (``f64[N]``, per slot, already
    divided by the normalization) joins ``mu`` BEFORE the step's pull-back."""
    if phi_k is None:
        return mu
    return jnp.where(rec_k.alive, mu + phi_k, mu)


def _move_pullback(lam, rec_k, r_k):
    """The move pull-back: survivors carry their post-move value back to the pre-move state,
    plus the rewards of the hits they took during the move (``r_k`` ``f64[K, N]``, one per
    tracker pass, zero where there was no hit).  Newborns (``born_face >= 0``) have no
    pre-move state; their value and birth-step hit rewards weight the source score."""
    born = rec_k.born_face >= 0
    lam = jnp.where(rec_k.alive, lam, 0.0)
    total = lam if r_k is None else lam + jnp.sum(r_k, axis=0)
    return jnp.where(rec_k.alive & ~born, total, 0.0), lam


def _step_pullback(mu, xs, grid, npc_max, scale, k, basis):
    """One captured step backward: sequential collision pull-back, then the move pull-back."""
    rec_k, phi_k, r_k = xs
    mu = _add_step_reward(mu, rec_k, phi_k)
    n = rec_k.x.shape[0]
    ids, valid, cell, _, _ = cell_nodes(rec_k.x, rec_k.alive, grid, npc_max)
    # scratch row for inactive attempts, as in the forward slot scan
    lam0 = jnp.concatenate((mu, jnp.zeros(1)))
    v0 = jnp.concatenate((rec_k.v_post, jnp.zeros((1, 3))))
    x = jnp.concatenate((rec_k.x, jnp.zeros((1, rec_k.x.shape[1]))))
    del cell
    body = functools.partial(
        _slot_pullback, x=x, ids=ids, valid=valid, scale=scale, k=k, basis=basis
    )
    p = rec_k.pairs
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
    move pull-back.  Both are per slot and already divided by the normalization; a windowed
    objective is nonzero only on its window.  The returned ``lam`` stays the value at the
    post-move state; :func:`ensemble_backward.from_device` forms the per-hit weights
    ``lam + (rewards of the later hits of the step)`` from ``hit_reward``.
    """
    mu = jnp.where(final_state.alive, phi_final, 0.0)
    body = functools.partial(_scan_body, grid=grid, npc_max=npc_max, scale=scale, k=k, basis=basis)
    mu0, (lam, four) = lax.scan(body, mu, (records, phi_step, hit_reward), reverse=True)
    return Backward(jnp.where(initial_alive, mu0, 0.0), lam, four, jnp.zeros(lam.shape[0]))


# --- virtual-partner pull-back ------------------------------------------------------


def _chunk_query(
    args, node_x, node_v, node_valid, node_val, node_class, scale, k, basis, gradients
):
    """Field values (and slopes) at one chunk of queries, each from its own cell's nodes;
    ``direction`` (``f64[chunk, 3]``, zero = no restriction) keeps only the nodes with
    ``v . direction > 0`` -- the one-sided fit of the shape adjoint; ``node_class`` /
    ``q_class`` (``bool``) keep only the nodes of the query's class."""
    c, q_x, q_v, direction, q_class = args
    valid = node_valid[c]
    one_sided = jnp.any(direction != 0.0, axis=1)
    same_side = jnp.einsum("qpk,qk->qp", node_v[c], direction) > 0.0
    valid = valid & (same_side | ~one_sided[:, None])
    if node_class is not None:
        valid = valid & (node_class[c] == q_class[:, None])
    value, grad_x, grad_v = field_query(
        node_x[c], node_v[c], valid, node_val[c], q_x, q_v, scale, k, basis
    )
    return (value, grad_x, grad_v) if gradients else value


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
):
    """Field value at ``Q`` queries in cells ``q_c`` from per-cell node tables ``(C, P, .)``,
    ``chunk`` queries at a time (the per-query node gather is ``P x (D+3)`` floats).
    ``gradients=True`` returns ``(value, grad_x, grad_v)`` instead; ``direction``
    ``f64[Q, 3]`` restricts a query's nodes to those with ``v . direction > 0`` (rows of
    zeros: no restriction).  A fifth table ``node_class`` (``bool[C, P]``) with ``q_class``
    ``bool[Q]`` restricts them to the query's class -- the stratified fit of the shape
    adjoint (:func:`node_gradients`)."""
    node_x, node_v, node_valid, node_val, *rest = tables
    node_class = rest[0] if rest else None
    q = q_c.shape[0]
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
    )
    out = lax.map(body, (c, x, v, d, qc))
    if gradients:
        value, grad_x, grad_v = out
        return (
            value.reshape(-1)[:q],
            grad_x.reshape(-1, grad_x.shape[-1])[:q],
            grad_v.reshape(-1, 3)[:q],
        )
    return out.reshape(-1)[:q]


def pooled_tables(tables, grid, pool):
    """Per-cell node tables ``(C, P, .)`` widened to the ``(2 pool + 1)^D`` block of cells
    around each cell, ``(C, B P, .)``: the pooled fit draws its ``k`` nearest nodes from the
    neighbourhood instead of the cell alone (more nodes and a wider position range for the
    slope, at the cost of a farther, less local kernel).  Block positions outside the box
    contribute no valid nodes."""
    node_x, node_v, node_valid, node_val, *rest = tables
    nbr, exists = grid_neighbourhood(grid, pool)  # (C, B)
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

    ``klass`` ``bool[N]`` **stratifies** the fit: a slot's nodes are the slots of its own
    class.  The shape adjoint passes "has a body hit ahead in the window": a particle's
    value jumps by its next hit's reward across the hit / no-hit boundary in phase space,
    and next to a curved surface the velocity half-space of ``direction`` does not
    separate the two (particles leaving a cap's tangent plane still hit the flat face
    below it) -- the fitted ``grad_v lambda`` of a reflected particle then reads the
    approaching population's ``d(2 m v_x)/dv``, a -27 % bias on the specular capsule
    whose true post-hit value gradient is zero (review section 7.8).
    """
    ids, valid, cell, _, count = cell_nodes(x, alive, grid, npc_max)
    c = jnp.minimum(cell, count.shape[0] - 1)
    tables = (x[ids], v[ids], valid, lam[ids])
    if klass is not None:
        tables = (*tables, klass[ids])
    if pool:
        tables = pooled_tables(tables, grid, pool)
    _, grad_x, grad_v = cell_query(
        tables, c, x, v, scale, k, basis, chunk, gradients=True, direction=direction, q_class=klass
    )
    return jnp.where(alive[:, None], grad_x, 0.0), jnp.where(alive[:, None], grad_v, 0.0)


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
):
    """``f64[Q]``: the carried adjoint at arbitrary query states ``(q_x, q_v)`` from the
    per-cell fit of ``lam`` over the population ``(x, v, alive)`` -- the value of a
    *virtual* post-move state (the silhouette term of :mod:`jax_dsmc.shape` asks for the
    value a particle would carry had its hit / miss gone the other way).  Queries outside
    the box are clipped to it.  ``direction`` / ``pool`` as :func:`node_gradients`."""
    ids, valid, _cell, _, count = cell_nodes(x, alive, grid, npc_max)
    tables = (x[ids], v[ids], valid, lam[ids])
    if pool:
        tables = pooled_tables(tables, grid, pool)
    lo, hi = grid.lo, grid.hi
    q_x = jnp.clip(q_x, lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
    q_c = jnp.minimum(cell_of(q_x, grid), count.shape[0] - 1)
    return cell_query(tables, q_c, q_x, q_v, scale, k, basis, chunk, direction=direction)


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


def _pair_kernel(vi, vj, u_chi, u_eps, sp):
    q = collision_rate(vi, vj, 0, 0, sp)
    vi_n, vj_n = scatter(vi, vj, 0, 0, sp, u_chi, u_eps)
    return q, vi_n, vj_n


def _partner_pass(
    carry,
    key,
    x,
    v_nodes,
    tables,
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
    q, vi_n, vj_n = jax.vmap(functools.partial(_pair_kernel, sp=sp))(
        v_nodes, v_nodes[j], u_chi, u_eps
    )
    f_i = cell_query(tables, c, x, vi_n, scale, k, basis, chunk, direction=_side(vi_n, normals))
    f_j = cell_query(
        tables,
        c,
        x[j],
        vj_n,
        scale,
        k,
        basis,
        chunk,
        direction=None if normals is None else _side(vj_n, normals[j]),
    )
    four = f_i + f_j - f_node - f_node[j]
    w = weight * q
    delta = delta + w * four
    if pair_score is not None:
        kern = kern + jnp.sum(0.5 * w * four * jax.vmap(pair_score)(v_nodes, v_nodes[j]))
    return (delta, kern), None


def _side(v, normals):
    """One-sided fit direction of a query: ``sign(v . n) n`` (zero rows: unrestricted)."""
    if normals is None:
        return None
    return jnp.sign(jnp.sum(v * normals, axis=1))[:, None] * normals


def _boundary_move(lam, x, v_prev, v_nodes, ids, valid, c, moved, normals, scale, k, basis, chunk):
    tables = (x[ids], v_nodes[ids], valid, lam[ids])
    f_prev = cell_query(
        tables, c, x, v_prev, scale, k, basis, chunk, direction=_side(v_prev, normals)
    )
    f_here = cell_query(
        tables, c, x, v_nodes, scale, k, basis, chunk, direction=_side(v_nodes, normals)
    )
    return jnp.where(moved, lam + f_prev - f_here, lam)


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
):
    """One backward substep: ``lam += (dt/S) L* lam`` with ``partners`` virtual partners per
    particle, then the value move of particles whose state differs across the boundary."""
    lam, key = carry
    v_nodes, v_prev = xs
    key, sub = jax.random.split(key)
    tables = (x[ids], v_nodes[ids], valid, lam[ids])
    c = jnp.minimum(cell, count.shape[0] - 1)
    f_node = cell_query(
        tables, c, x, v_nodes, scale, k, basis, chunk, direction=_side(v_nodes, normals)
    )
    body = functools.partial(
        _partner_pass,
        x=x,
        v_nodes=v_nodes,
        tables=tables,
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
        body, (jnp.zeros(lam.shape[0]), 0.0), jax.random.split(sub, partners)
    )
    lam = jnp.where(alive, lam + delta, lam)
    moved = alive & jnp.any(v_prev != v_nodes, axis=1)
    lam = lax.cond(
        jnp.any(moved),
        functools.partial(_boundary_move, scale=scale, k=k, basis=basis, chunk=chunk),
        _boundary_skip,
        lam,
        x,
        v_prev,
        v_nodes,
        ids,
        valid,
        c,
        moved,
        normals,
    )
    return (lam, key), kern


def _step_pullback_virtual(
    carry, xs, grid, npc_max, scale, k, basis, chunk, sp, substeps, partners, nodes, pair_score
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
        stages = staged_velocities(rec_k.pairs, rec_k.v, rec_k.v_post, cell, substeps)
    else:
        stages = jnp.concatenate((rec_k.v[None], jnp.broadcast_to(rec_k.v_post, (substeps, n, 3))))
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
    )
    (lam, _), kern = lax.scan(body, (mu, step_key), (stages[1:], stages[:-1]), reverse=True)
    mu_prev, lam = _move_pullback(lam, rec_k, r_k)
    return (mu_prev, key), (lam, jnp.sum(kern))


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
    """
    mu = jnp.where(final_state.alive, phi_final, 0.0)
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
    )
    (mu0, _), (lam, kern) = lax.scan(
        body, (mu, key), (records, phi_step, hit_reward, normals), reverse=True
    )
    m, a, c = records.pairs.pi.shape
    return Backward(jnp.where(initial_alive, mu0, 0.0), lam, jnp.zeros((m, a, c)), kern)
