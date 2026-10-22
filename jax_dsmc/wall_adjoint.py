"""P3 random-time walls, with the conditional-time derivative made explicit.

A realized 1-D boundary history restricts the Gaussian travel time to an
interval. Reparameterizing the truncated Gaussian in that interval and scoring
the interval probability gives a consistent derivative of the regularized
transition, including position observables. Merely scoring the hit probability
while freezing an unconditional travel time omits this conditional derivative.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_dsmc import sampling, step_axis
from jax_dsmc.compaction import prefix_count, ranked_slots
from jax_dsmc.geometry import first_hit
from jax_dsmc.grid import cell_of
from jax_dsmc.mesh import geometry as mesh_geometry
from jax_dsmc.safe import safe_log
from jax_dsmc.state import lane_capacity
from jax_dsmc.surface import SurfaceTallies, zero


def conditional_time(raw, z, lower, upper, dt, epsilon):
    """Value-preserving conditional Gaussian sample and interval log probability."""
    eps = jnp.where(epsilon > 0.0, epsilon, 1.0)
    finite_lo, finite_hi = jnp.isfinite(lower), jnp.isfinite(upper)
    zl = (jnp.where(finite_lo, lower, dt) - dt) / eps
    zu = (jnp.where(finite_hi, upper, dt) - dt) / eps
    fl = jnp.where(finite_lo, jax.scipy.special.ndtr(zl), 0.0)
    fu = jnp.where(finite_hi, jax.scipy.special.ndtr(zu), 1.0)
    probability = jnp.maximum(fu - fl, 1e-30)
    within = jax.lax.stop_gradient(
        jnp.clip((jax.scipy.special.ndtr(z) - fl) / probability, 0.0, 1.0)
    )
    quantile = jnp.clip(fl + within * probability, 1e-15, 1.0 - 1e-15)
    sampled = dt + eps * jax.scipy.special.ndtri(quantile)
    # Preserve the original random realization exactly, but use its conditional tangent.
    corrected = jax.lax.stop_gradient(raw) + (sampled - jax.lax.stop_gradient(sampled))
    time = jnp.maximum(jnp.where(epsilon > 0.0, corrected, dt), 0.0)
    logp = jnp.where(epsilon > 0.0, safe_log(probability), 0.0)
    return time, logp


def _fixed_time(raw, z, lower, upper, dt, epsilon):
    """:func:`conditional_time` at ``epsilon = 0``: the move time is ``dt`` and the interval
    carries no log-probability -- the same values, without the special functions over
    every slot."""
    del z, lower, upper, epsilon
    return jnp.maximum(jnp.zeros_like(raw) + dt, 0.0), jnp.zeros_like(raw)


def _time_draws(key, n):
    """The standard normals of the regularised flight time (``epsilon > 0``)."""
    return jax.random.normal(jax.random.fold_in(key, 0), (n,), dtype=jnp.float64)


def _no_time_draws(key, n):
    """``epsilon = 0``: the draws would enter as ``epsilon * z = 0``; none are made."""
    del key
    return jnp.zeros(n)


def next_hit(x, v, lo, hi):
    vd = v[:, 0]
    moving = vd != 0.0
    side = vd > 0.0
    bound = jnp.where(side, hi, lo)
    distance = (bound - x[:, 0]) / jnp.where(moving, vd, 1.0)
    return jnp.where(moving, jnp.maximum(distance, 0.0), jnp.inf), side


class WallEvents(NamedTuple):
    """Per-slot, per-bounce record of the step's diffuse-face hits, for the ensemble adjoint.

    Shapes ``(K_bounce, N)`` (``(K_bounce, N, 3)`` for ``v_out``): pass ``k`` of the
    chronological tracker records the hit it resolved -- the DIFFUSE face (``-1`` if the
    pass resolved no diffuse hit for that slot), the laboratory velocity the particle left
    with, and whether the accommodation draw took the diffuse branch.  Every hit is scored
    separately, weighted by λ at the end of the move: the realized remainder of the flight
    is the unbiased sample of the value after the hit, so nothing is linearised in the
    number of hits per step.

    With an embedded surface (``mesh``), **every** element hit is recorded too, specular
    or diffuse, under the surface id ``2D + e`` (``2D`` for an analytic body): the shape
    adjoint replays the step's wall map from the element ids alone (plan section 0), so
    nothing else about the hit is stored in a forward record.

    The three trailing fields are ``None`` in a record and filled only by a *traced*
    replay (``advance(trace=True)``, :mod:`jax_dsmc.shape`): ``x_hit`` ``f64[K, N, D]``
    the hit point, ``v_in`` ``f64[K, N, 3]`` the incident velocity, ``normal``
    ``f64[K, N, 3]`` the unit normal used (zero-padded past ``D``) -- what per-hit
    rewards and the kernel's normal score need.
    """

    face: jax.Array
    v_out: jax.Array
    diffuse: jax.Array
    x_hit: jax.Array | None = None
    v_in: jax.Array | None = None
    normal: jax.Array | None = None
    time: jax.Array | None = None
    """``f64[K, N]`` time since the start of the step at which pass ``k``'s event happened
    (every event, box faces included; 0 where there was none) -- the hit-timing band
    term of the shape adjoint puts its cotangent here."""


class SplitEvents(NamedTuple):
    """The events of a lane-tracked step: ``head`` (``WallEvents`` with one row, every
    slot) is pass 0; ``tail`` (``WallEvents`` with ``K - 1`` rows over the ``B`` lane
    entries) holds passes ``1 .. K - 1`` of the particles that had a pass-0 event, entry
    ``j`` belonging to slot ``slot[j]`` (``-1`` = unused entry).  A dense record (the
    stencil / walk trackers) is the special case ``tail = passes 1..``, ``slot =
    arange(N)`` (:func:`as_split`).  Per-event arrays of a consumer have the shapes of
    ``head.face`` and ``tail.face``; :func:`particle_sum` folds them per slot."""

    head: WallEvents
    tail: WallEvents
    slot: jax.Array


class RecordEvents(NamedTuple):
    """The **stored** events of a lane-tracked step (``EventRecord.walls``): pass 0 dense
    (``head``, ``WallEvents`` with one row over every slot) and the later passes as a flat
    list of the events actually recorded -- ``slot`` / ``pass_`` (``i32[E]``, ``-1`` for
    unused entries; ``pass_`` counts from 1), ``face``, ``v_out`` (``f64[E, 3]``),
    ``diffuse``.  :class:`SplitEvents` keeps ``K - 1`` rows over the whole lane, almost all
    empty (a particle rarely has a recorded event after its first); nothing downstream of
    the forward needs the empty rows -- the shape adjoint's replay recomputes its events --
    so the record keeps only the list (:func:`compact_record`)."""

    head: WallEvents
    slot: jax.Array
    pass_: jax.Array
    face: jax.Array
    v_out: jax.Array
    diffuse: jax.Array


def _put_flat(values, target, capacity, fill, dtype):
    """``values`` (``R, W, ...``) flattened and written to the list entries ``target``
    (``capacity`` = dropped); unused entries hold ``fill``."""
    rows, width = values.shape[:2]
    flat = values.reshape((rows * width, *values.shape[2:]))
    empty = jnp.full((capacity, *values.shape[2:]), fill, dtype)
    return empty.at[target].set(flat.astype(dtype), mode="drop")


def compact_record(events, capacity):
    """:class:`RecordEvents` of split ``events`` with room for ``capacity`` later-pass
    events (in pass order, then lane order), and the number that did not fit (the record
    would lose them: an error, ``Diagnostics.record_overflow``)."""
    tail = events.tail
    rows, width = tail.face.shape
    has = (tail.face >= 0).reshape(-1)
    pos = prefix_count(has) - 1
    fits = has & (pos < capacity)
    target = jnp.where(fits, pos, capacity)

    slot = jnp.broadcast_to(events.slot[None, :], (rows, width))
    pass_ = jnp.broadcast_to(jnp.arange(1, rows + 1, dtype=jnp.int32)[:, None], (rows, width))
    put = functools.partial(_put_flat, target=target, capacity=capacity)
    record = RecordEvents(
        events.head,
        put(slot, fill=-1, dtype=jnp.int32),
        put(pass_, fill=-1, dtype=jnp.int32),
        put(tail.face, fill=-1, dtype=jnp.int32),
        put(tail.v_out, fill=0.0, dtype=jnp.float64),
        put(tail.diffuse, fill=False, dtype=bool),
    )
    return record, jnp.sum(has, dtype=jnp.int32) - jnp.sum(fits, dtype=jnp.int32)


def as_split(events):
    """:class:`SplitEvents` view of ``events`` (a dense ``WallEvents`` or already split)."""
    if isinstance(events, SplitEvents):
        return events
    n = events.face.shape[1]
    head = jax.tree.map(lambda a: a[:1], events)
    tail = jax.tree.map(lambda a: a[1:], events)
    return SplitEvents(head, tail, jnp.arange(n, dtype=jnp.int32))


def _drop_slot(slot, n):
    """Lane slot ids with unused entries (``-1``) sent out of range for ``mode="drop"``."""
    return jnp.where(slot >= 0, slot, n)


def particle_sum(head_values, tail_values, slot):
    """Per-slot total of a per-event quantity: ``head_values`` (``1, N``) summed over its
    row plus ``tail_values`` (``K, B``) summed over passes and scattered to ``slot``."""
    n = head_values.shape[1]
    total = jnp.sum(head_values, axis=0)
    return total.at[_drop_slot(slot, n)].add(jnp.sum(tail_values, axis=0), mode="drop")


def to_dense(events):
    """Dense ``WallEvents`` (``1 + K`` rows over every slot) of split ``events`` -- for
    tests and small cases; the tail rows are scattered to their slots, unused lane entries
    dropped, and slots outside the lane keep the empty-event values."""
    if not isinstance(events, SplitEvents):
        return events
    n = events.head.face.shape[1]
    idx = _drop_slot(events.slot, n)

    def spread(h, t):
        empty = jnp.full((t.shape[0], n, *t.shape[2:]), -1 if h.dtype == jnp.int32 else 0, h.dtype)
        full = empty.at[:, idx].set(t, mode="drop")
        return jnp.concatenate((h, full))

    return jax.tree.map(spread, events.head, events.tail)


def lane_slots(active, size):
    """Compact the ``active`` slots into a lane of ``size`` entries (cumulative-sum
    compaction, :func:`jax_dsmc.compaction.ranked_slots`): ``(slot i32[size], overflow bool[N])`` --
    ``slot[j]`` the ``j``-th active slot in slot order or ``-1``, ``overflow`` the active
    slots that did not fit."""
    pos = prefix_count(active) - 1
    return ranked_slots(pos, 0, size), active & (pos >= size)


class TrackerCounts(NamedTuple):
    """Per-step tracking diagnostics of the mesh path (all ``i32`` scalars).

    ``exhausted``: particles with an unresolved hit after ``max_bounces`` passes (stencil
    or lane; deleted, SPARTA's stuck particle)
    or ``max_walk`` iterations (walk); ``surface_hits``: element / body hits resolved;
    ``stencil_overrun``: particles whose remaining move exceeded the stencil dilation;
    ``inside_body``: particles found inside the body after the move and killed;
    ``lane_overflow``: particles with a pass-0 event that did not fit the lane
    (``tracker="lane"``) and were deleted.
    """

    exhausted: jax.Array
    surface_hits: jax.Array
    stencil_overrun: jax.Array
    inside_body: jax.Array
    lane_overflow: jax.Array | int = 0


def no_wall_events(n, bounces, trace_dim=None):
    """Empty events; ``trace_dim=D`` allocates the traced-replay fields as well."""
    trace = (
        (
            jnp.zeros((bounces, n, trace_dim)),
            jnp.zeros((bounces, n, 3)),
            jnp.zeros((bounces, n, 3)),
            jnp.zeros((bounces, n)),
        )
        if trace_dim is not None
        else (None, None, None, None)
    )
    return WallEvents(
        jnp.full((bounces, n), -1, jnp.int32),
        jnp.zeros((bounces, n, 3)),
        jnp.zeros((bounces, n), bool),
        *trace,
    )


def _trace_hit(events, k, selected, x_hit, v_in, normal):
    """Fill pass ``k``'s traced fields for ``selected`` (no-op on an untraced record)."""
    if events.x_hit is None:
        return events
    sel = selected[:, None]
    return events._replace(
        x_hit=events.x_hit.at[k].set(jnp.where(sel, x_hit, events.x_hit[k])),
        v_in=events.v_in.at[k].set(jnp.where(sel, v_in, events.v_in[k])),
        normal=events.normal.at[k].set(jnp.where(sel, normal, events.normal[k])),
    )


def _face_tally(tally, f, selected, incident, outgoing, mass, velocity_wall):
    """Extensive momentum / energy / heat transferred to face ``f`` by ``selected``."""
    impulse = jnp.where(selected[:, None], mass * (incident - outgoing), 0.0).sum(axis=0)
    energy = jnp.where(
        selected,
        0.5 * mass * (jnp.sum(incident**2, axis=1) - jnp.sum(outgoing**2, axis=1)),
        0.0,
    ).sum()
    return SurfaceTallies(
        tally.hits.at[f].add(selected.sum(dtype=jnp.int32)),
        tally.momentum.at[f].add(impulse),
        tally.energy.at[f].add(energy),
        tally.heat.at[f].add(energy - jnp.dot(velocity_wall, impulse)),
    )


def _face_tallies(tally, rows, slot, mass, velocity_wall):
    """:func:`_face_tally` of every box face over every row: ``rows = (box_hit, surface,
    incident, v, alive)``, the outgoing velocity ``v`` where ``alive`` (else zero)."""
    del slot
    box_hit, surface, incident, v, alive = rows
    outgoing = jnp.where(alive[:, None], v, 0.0)
    for f, vw in enumerate(velocity_wall):
        tally = _face_tally(tally, f, box_hit & (surface == f), incident, outgoing, mass, vw)
    return tally


def _compact_face_tallies(tally, rows, slot, mass, velocity_wall):
    """:func:`_face_tallies` of the box-face hits among the compacted hit rows ``slot``
    (``-1`` unused): the same per-row terms summed over the hit rows only, so the totals
    differ from the dense sums only in the order of the additions."""
    _box_hit, surface, incident, v, alive = rows
    take = jnp.maximum(slot, 0)
    valid = slot >= 0
    surface, incident, v, alive = surface[take], incident[take], v[take], alive[take]
    outgoing = jnp.where(alive[:, None], v, 0.0)
    for f, vw in enumerate(velocity_wall):
        tally = _face_tally(tally, f, valid & (surface == f), incident, outgoing, mass, vw)
    return tally


def _box_tallies(tally, rows, hit_slot, faces, wall, mass):
    """The box faces' tallies of one pass (``rows`` of :func:`_face_tallies`), on the
    compacted hits ``hit_slot = (slot, lost)`` unless they overflowed or are ``None``."""
    from jax_dsmc.boundary import BC  # noqa: PLC0415 -- break the boundary import cycle

    velocity_wall = tuple(
        wall.vwall[f] if faces.of(f // 2, f % 2) is BC.DIFFUSE else jnp.zeros(3)
        for f in range(len(faces.bcs))
    )
    dense = functools.partial(_face_tallies, mass=mass, velocity_wall=velocity_wall)
    if hit_slot is None:
        return dense(tally, rows, None)
    slot, lost = hit_slot
    compact = functools.partial(_compact_face_tallies, mass=mass, velocity_wall=velocity_wall)
    return jax.lax.cond(step_axis.any_(lost), dense, compact, tally, rows, slot)


def _surface_tally(tally, surface, selected, incident, outgoing, mass, velocity_wall):
    """:func:`_face_tally` for a per-particle surface id (scatter-add over ``surface``)."""
    impulse = jnp.where(selected[:, None], mass * (incident - outgoing), 0.0)
    energy = jnp.where(
        selected,
        0.5 * mass * (jnp.sum(incident**2, axis=1) - jnp.sum(outgoing**2, axis=1)),
        0.0,
    )
    heat = energy - jnp.sum(velocity_wall * impulse, axis=1)
    return SurfaceTallies(
        tally.hits.at[surface].add(selected.astype(jnp.int32)),
        tally.momentum.at[surface].add(impulse),
        tally.energy.at[surface].add(energy),
        tally.heat.at[surface].add(heat),
    )


COMPACT_MIN = 2048
"""Rows per pass from which the face tallies and element hits are formed on the compacted
hit rows only (:func:`_apply_events`); below it (lane passes, entrants) every row."""


def _reemission(rows, mass):
    """The diffuse re-emission of each row from its uniforms ``u_z, u_r`` at its hit
    (``normal``, wall ``temp`` and ``vwall``, incident ``v``), ``f64[R, 3]``."""
    normal, v, u_z, u_r, temp, vwall = rows
    z, r = sampling.diffuse_draws_from_uniforms(u_z, u_r)
    return sampling.diffuse_velocity_general(z, r, normal, temp[:, None], mass, vwall, v)


def _hit_rows(rows, grid, wall, mass, acc_delta, mesh):
    """The element hit of each row -- ``rows = (x_at, v, surface, u_z, u_r, u, normal)``,
    ``normal`` ``None`` to take the surface's -- as ``(e, normal, took_diffuse, w, v_out,
    log_p, fresh_traced, v_spec, vwall)``; meaningful on the rows that hit an element."""
    x_at, v, surface, u_z, u_r, u, normal = rows
    dim = x_at.shape[1]
    e = jnp.clip(surface - 2 * dim, 0, mesh.n_elements - 1)
    model = mesh.body.model[e] if mesh.kind == "surface" else jnp.zeros_like(e)
    row = 2 * dim + model
    temp_rows = wall.temp if wall.temp.ndim == 1 else wall.temp[:, 0]
    temp = temp_rows[row]
    acc = wall.acc[row]
    vwall = wall.vwall[row]
    if normal is None:
        normal = _hit_normal(x_at, surface, grid, mesh)
    fresh = _reemission((normal, v, u_z, u_r, temp, vwall), mass)
    vn = jnp.sum(v * normal, axis=1, keepdims=True)
    v_spec = v - 2.0 * vn * normal
    # the diffuse branch is a KERNEL to the adjoint: its parameters (temperature, drift,
    # accommodation) and the normal enter the VALUE after the hit through the score of
    # `sampling.log_diffuse_density`, never pathwise through the sampled velocity (review
    # section 5.9: a field's fitted derivative is a poor source term); the post-hit state
    # and the events therefore carry the frozen sample.  The hit's OWN reward is exact in
    # `v_out`, so the surface tally -- the pathwise entry of a per-hit reward -- is formed
    # from the traced sample: the reparameterised `d/dn E[r(v_out)]` with bounded weights,
    # which the shape adjoint pairs with the score weighted by the value-to-go AFTER this
    # hit (review section 7.8, GP)
    fresh_traced = fresh
    fresh = jax.lax.stop_gradient(fresh)
    # `surf_collide_diffuse.cpp:211` -- `random->uniform() > acc` takes the specular branch
    took_diffuse = u <= acc
    safe_delta = jnp.where(acc_delta > 0.0, acc_delta, 1.0)
    w = jnp.where(
        acc_delta > 0.0,
        jax.nn.sigmoid((acc - u) / safe_delta),
        took_diffuse.astype(jnp.float64),
    )
    v_out = w[:, None] * fresh + (1.0 - w)[:, None] * v_spec
    log_p = jnp.where(took_diffuse, safe_log(acc), safe_log(1.0 - acc))
    log_p = jnp.where(acc_delta > 0.0, 0.0, log_p)
    return e, normal, took_diffuse, w, v_out, log_p, fresh_traced, v_spec, vwall


def _hit_normal(x_at, surface, grid, mesh):
    """Unit normal into the flow of each row's surface, zero-padded to 3, ``f64[R, 3]``."""
    dim = x_at.shape[1]
    return (
        jnp.zeros((x_at.shape[0], 3))
        .at[:, :dim]
        .set(mesh_geometry.surface_normal(mesh, surface, x_at, grid))
    )


def _elements_dense(state, rows, slot, k, operands):
    """The element hits of one pass over every row: the per-row results of
    :func:`_hit_rows` selected into the state where ``selected``."""
    del slot
    v, logp, tally, events, exclude = state
    grid, wall, mass, acc_delta, tallies, mesh = operands
    x_at, surface, selected, *draws, normal = rows
    e, normal, took_diffuse, w, v_out, log_p, fresh_traced, v_spec, vwall = _hit_rows(
        (x_at, v, surface, *draws, normal), grid, wall, mass, acc_delta, mesh
    )
    sel = selected[:, None]
    v_new = jnp.where(sel, v_out, v)
    logp = logp + jnp.where(selected, log_p, 0.0)
    events = events._replace(
        face=events.face.at[k].set(jnp.where(selected, surface, events.face[k])),
        v_out=events.v_out.at[k].set(jnp.where(sel, v_out, events.v_out[k])),
        diffuse=events.diffuse.at[k].set(jnp.where(selected, took_diffuse, events.diffuse[k])),
    )
    events = _trace_hit(events, k, selected, x_at, v, normal)
    # SPARTA `update.cpp`: the element just left is excluded from the next search
    exclude = jnp.where(selected, e, exclude)
    if tallies:
        outgoing = jnp.where(sel, w[:, None] * fresh_traced + (1.0 - w)[:, None] * v_spec, v)
        tally = _surface_tally(tally, surface, selected, v, outgoing, mass, vwall)
    return v_new, logp, tally, events, exclude


def _elements_compact(state, rows, slot, k, operands):
    """:func:`_elements_dense` on the element hits among the compacted hit rows ``slot``
    (``-1`` unused), scattered back: row for row the same arithmetic, and the rows it
    skips are the ones the dense pass leaves unchanged (its surface tally adds exact
    zeros for them, and the scatter-add visits the rest in slot order)."""
    v, logp, tally, events, exclude = state
    grid, wall, mass, acc_delta, tallies, mesh = operands
    n = v.shape[0]
    take = jnp.maximum(slot, 0)
    x_at, surface, _selected, *draws, normal = rows
    x_at, surface, vt = x_at[take], surface[take], v[take]
    valid = (slot >= 0) & (surface >= 2 * x_at.shape[1])
    put = jnp.where(valid, slot, n)
    draws = [a[take] for a in draws]
    normal = None if normal is None else normal[take]
    e, normal, took_diffuse, w, v_out, log_p, fresh_traced, v_spec, vwall = _hit_rows(
        (x_at, vt, surface, *draws, normal), grid, wall, mass, acc_delta, mesh
    )
    events = events._replace(
        face=events.face.at[k, put].set(surface, mode="drop"),
        v_out=events.v_out.at[k, put].set(v_out, mode="drop"),
        diffuse=events.diffuse.at[k, put].set(took_diffuse, mode="drop"),
    )
    if events.x_hit is not None:
        events = events._replace(
            x_hit=events.x_hit.at[k, put].set(x_at, mode="drop"),
            v_in=events.v_in.at[k, put].set(vt, mode="drop"),
            normal=events.normal.at[k, put].set(normal, mode="drop"),
        )
    if tallies:
        outgoing = jnp.where(
            valid[:, None], w[:, None] * fresh_traced + (1.0 - w)[:, None] * v_spec, vt
        )
        tally = _surface_tally(tally, surface, valid, vt, outgoing, mass, vwall)
    return (
        v.at[put].set(v_out, mode="drop"),
        logp.at[put].add(log_p, mode="drop"),
        tally,
        events,
        exclude.at[put].set(e, mode="drop"),
    )


def _element_events(
    x_at, v, logp, tally, events, exclude, hit, surface, draws, k, operands, hit_slot
):
    """The element branch of one pass: reflect off the embedded surface (SPARTA
    ``surf_collide_specular.cpp:75-150`` / ``surf_collide_diffuse.cpp:206-306``).

    Every element carries the ``diffuse T acc`` law of its model row of ``WallParams``
    (``2D + model``; ``acc = 0`` is specular).  The normal is the flat element normal
    (``mesh.normals[e]``) or the exact one of an analytic body, in the flow direction;
    the diffuse frame is :func:`sampling._tangent_frame`.  Element models use the
    isotropic wall temperature (``wall.temp[row]`` or its first component).
    ``hit_slot``: ``(slot, lost)`` of the pass's hits compacted (:func:`lane_slots`), or
    ``None`` (every row).
    """
    grid, _wall, _mass, _acc_delta, _tallies, mesh = operands
    dim = x_at.shape[1]
    selected = hit & (surface >= 2 * dim)
    u_z, u_r, u = draws  # uniforms: transformed on the rows that use them (_hit_frames)
    # on the compacted hits (`hit_slot`), a polygon's normal is a gather and joins them; a
    # curved body's stays dense: a level set's is a sum of products whose FMA contraction
    # XLA:CPU chooses per kernel
    normal = None if mesh.kind == "surface" else _hit_normal(x_at, surface, grid, mesh)
    rows = (x_at, surface, selected, u_z, u_r, u, normal)
    state = (v, logp, tally, events, exclude)
    if hit_slot is None:
        return _elements_dense(state, rows, None, k, operands)
    slot, lost = hit_slot
    return jax.lax.cond(
        step_axis.any_(lost),
        functools.partial(_elements_dense, k=k, operands=operands),
        functools.partial(_elements_compact, k=k, operands=operands),
        state,
        rows,
        slot,
    )


def _slot_uniforms(pass_key, slot):
    """The re-emission uniforms ``(u_z f64[R, 3], u_r f64[R], u_acc f64[R])`` of the rows
    whose slots are ``slot`` (``i32[R]``), counter-based: row ``j`` draws five uniforms from
    ``fold_in(pass_key, slot[j])``.  A particle therefore meets the same randomness in a
    pass whatever row it occupies -- in any tracker, the lane's entries included -- and
    only the rows that are searched pay for draws (not every slot of the population).
    ``u_z`` is on ``(nextafter(-1, 0), 1)``, the domain of the normal transform
    (:func:`jax_dsmc.sampling.diffuse_draws_from_uniforms`)."""
    keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(pass_key, slot)
    u = jax.vmap(functools.partial(jax.random.uniform, shape=(5,), dtype=jnp.float64))(keys)
    u_z = jnp.maximum(sampling.NORMAL_LO, 2.0 * u[:, :3] - 1.0)
    return u_z, u[:, 3], u[:, 4]


def _pass_draws(pass_key, n, faces, draw_slot, draw_n, counter=False):
    """One pass's re-emission randomness: the uniforms ``(u_z, u_r, u_acc)`` and the box
    faces' ``draws`` ``(z, r, u_acc)``.  ``counter`` (an embedded surface): per-slot
    counter-based draws (:func:`_slot_uniforms`) at the rows' slots ``draw_slot`` (``None``
    = row ``i`` is slot ``i``).  Else the uniforms are drawn over every slot (a lane entry
    gathers its slot's, so a particle meets the same randomness in any tracker).  The
    transforms to normal / Rayleigh variates run only where a re-emission can use them --
    every row when a box face is diffuse, the element hit rows (:func:`_hit_frames`)."""
    from jax_dsmc.boundary import BC  # noqa: PLC0415 -- break the boundary import cycle

    if counter:
        slot = jnp.arange(n, dtype=jnp.int32) if draw_slot is None else draw_slot
        u_z, u_r, u_acc = _slot_uniforms(pass_key, slot)
    else:
        k_draw, k_acc = jax.random.split(pass_key)
        n_draw = n if draw_slot is None else draw_n
        u_z, u_r = sampling.standard_diffuse_uniforms(k_draw, n_draw)
        u_acc = jax.random.uniform(k_acc, (n_draw,), dtype=jnp.float64)
        if draw_slot is not None:
            u_z, u_r, u_acc = u_z[draw_slot], u_r[draw_slot], u_acc[draw_slot]
    z = r = None
    if BC.DIFFUSE in faces.bcs:
        z, r = sampling.diffuse_draws_from_uniforms(u_z, u_r)
    return u_z, u_r, u_acc, (z, r, u_acc)


def _apply_events(
    operands,
    grid,
    faces,
    wall,
    mass,
    acc_delta,
    tallies,
    mesh=None,
    draw_slot=None,
    draw_n=0,
    hit_slot=None,
):
    """Resolve the events of one pass for the particles flagged ``hit``.

    ``draw_slot`` (lane passes): the diffuse draws are the pass's draws over all
    ``draw_n`` slots gathered at each lane entry's slot, so a particle meets exactly the
    randomness the dense tracker would give it in that pass.  ``hit_slot``: the hits
    already compacted (:func:`lane_slots` at ``lane_capacity(N)``), else made here."""
    from jax_dsmc.boundary import (  # noqa: PLC0415 -- break the boundary dispatch import cycle
        BC,
        diffuse_from_draws,
    )

    x, v, alive, logp, tally, events, hit, surface, used, pass_key, k, exclude, at_time = operands
    dim = x.shape[1]
    n = x.shape[0]
    u_z, u_r, u_acc, draws = _pass_draws(
        pass_key, n, faces, draw_slot, draw_n, counter=mesh is not None
    )
    x_at = x + used[:, None] * v[:, :dim]
    v_start = v
    # over a whole population the hits are a few in a thousand: the face tallies and the
    # element hits are formed on them compacted (the lane's size; an overflow: every row)
    if hit_slot is None and n >= COMPACT_MIN and (tallies or mesh is not None):
        hit_slot = lane_slots(hit, lane_capacity(n, 0))
    if events.time is not None:
        events = events._replace(time=events.time.at[k].set(jnp.where(hit, at_time, 0.0)))
    for axis in range(dim):
        for s in (0, 1):
            f = 2 * axis + s
            selected = hit & (surface == f)
            rule = faces.of(axis, s)
            incident = v
            if rule is BC.DIFFUSE:
                new_v, lp = diffuse_from_draws(v, wall, mass, axis, s, draws, acc_delta)
                v = jnp.where(selected[:, None], new_v, v)
                logp = logp + jnp.where(selected, lp, 0.0)
                events = events._replace(
                    face=events.face.at[k].set(jnp.where(selected, f, events.face[k])),
                    v_out=events.v_out.at[k].set(
                        jnp.where(selected[:, None], new_v, events.v_out[k])
                    ),
                    diffuse=events.diffuse.at[k].set(
                        jnp.where(selected, u_acc <= wall.acc[f], events.diffuse[k])
                    ),
                )
                n_face = jnp.zeros((n, 3)).at[:, axis].set(-1.0 if s else 1.0)
                events = _trace_hit(events, k, selected, x_at, incident, n_face)
            elif rule is BC.SPECULAR:
                v = v.at[:, axis].set(jnp.where(selected, -v[:, axis], v[:, axis]))
            elif rule is BC.PERIODIC:
                x_at = x_at.at[:, axis].set(
                    jnp.where(selected, grid.lo[axis] if s else grid.hi[axis], x_at[:, axis])
                )
            else:
                alive = alive & ~selected
    if tallies:
        # a particle meets one face per pass: its face's incident velocity is the pass's
        # `v_start`, its outgoing one the face loop's result
        rows = (hit & (surface < 2 * dim), surface, v_start, v, alive)
        tally = _box_tallies(tally, rows, hit_slot, faces, wall, mass)
    # a box-face event ends the exclusion (SPARTA resets `exclude` on leaving the cell)
    exclude = jnp.where(hit & (surface < 2 * dim), -1, exclude)
    if mesh is not None:
        v, logp, tally, events, exclude = _element_events(
            x_at,
            v,
            logp,
            tally,
            events,
            exclude,
            hit,
            surface,
            (u_z, u_r, u_acc),
            k,
            (grid, wall, mass, acc_delta, tallies, mesh),
            hit_slot,
        )
    x = jnp.where(hit[:, None], x_at, x)
    return x, v, alive, logp, tally, events, exclude


def _skip_events(operands):
    x, v, alive, logp, tally, events, *_rest, exclude, _at_time = operands
    return x, v, alive, logp, tally, events, exclude


def _bounce(
    k,
    carry,
    grid,
    faces,
    wall,
    mass,
    key,
    acc_delta,
    tallies,
    mesh=None,
    row_shift=0,
    draw_slot=None,
    draw_n=0,
    with_hit=False,
    with_slot=False,
    tables=None,
):
    """One chronological event pass over every particle.  ``fori_loop`` body.

    Module-level and bound with ``functools.partial`` (decision 14).  Two cost
    decisions, both measured (scratch/bench/b6_wall_cost.py):

    * the diffuse randomness -- a standard normal triple, a standard Rayleigh speed
      and the accommodation uniform -- is drawn **once per pass**, not once per
      face: a particle meets at most one face per pass, so every face's re-emission
      law is unchanged and the pass costs one ``(n, 3)`` draw instead of ``2D``;
    * a pass in which **no** particle reaches a face (the usual case for every pass
      after the first) skips the draws, the face loop and the tallies through a
      ``lax.cond``; only the ``first_hit`` query is paid.  Draws happen inside the
      taken branch, so a skipped pass consumes no random numbers.

    Lane passes (``tracker="lane"``) run this body on the compacted lane: ``k`` is the
    pass number (it keys the draws), the event row is ``k - row_shift``, and
    ``draw_slot`` / ``draw_n`` gather the pass's full-width draws at the lane's slots.
    ``with_slot`` (pass 0 of the lane, ``N >= COMPACT_MIN``) also returns the pass's hits
    compacted, ``(slot, lost)`` of :func:`lane_slots`.  ``tables``: the polygon search's
    cull tables (:func:`jax_dsmc.mesh.geometry.near_tables`), formed outside the pass.
    """
    x, v, alive, elapsed, logp, had_hit, tally, events, budget, exclude, overrun, n_surf = carry
    distance, surface = _next_surface(x, v, grid, mesh, exclude, budget - elapsed, tables)
    hit = alive & (elapsed + distance <= budget)
    used = jnp.where(hit, distance, 0.0)
    if mesh is not None:
        # the stencil's sufficient condition, tested on the whole remaining move
        ok = mesh_geometry.stencil_sufficient(v, budget - elapsed, grid, mesh, x)
        overrun = overrun | (alive & ~ok)
        n_surf = n_surf + jnp.sum(hit & (surface >= 2 * x.shape[1]), dtype=jnp.int32)
    operands = (
        x,
        v,
        alive,
        logp,
        tally,
        events,
        hit,
        surface,
        used,
        jax.random.fold_in(key, 1 + k),
        k - row_shift,
        exclude,
        elapsed + used,
    )
    hit_slot = lane_slots(hit, lane_capacity(x.shape[0], 0)) if with_slot else None
    x, v, alive, logp, tally, events, exclude = jax.lax.cond(
        step_axis.any_(hit),
        functools.partial(
            _apply_events,
            grid=grid,
            faces=faces,
            wall=wall,
            mass=mass,
            acc_delta=acc_delta,
            tallies=tallies,
            mesh=mesh,
            draw_slot=draw_slot,
            draw_n=draw_n,
            hit_slot=hit_slot,
        ),
        _skip_events,
        operands,
    )
    elapsed = elapsed + used
    had_hit = had_hit | hit
    carry = (x, v, alive, elapsed, logp, had_hit, tally, events, budget, exclude, overrun, n_surf)
    if with_slot:
        return carry, hit_slot
    return (carry, hit & alive) if with_hit else carry


def _lane_pass(k, state, body):
    """One lane pass, skipped outright (``lax.cond``, so the hit search is not paid) when
    no lane particle had an event in the previous pass: one without an event in pass
    ``k - 1`` has none in pass ``k``, so the skip is exact.  ``state = (carry, pending)``;
    reverse-mode differentiable, unlike a ``while_loop``, so the shape adjoint's replay
    runs it too."""
    carry, pending = state
    return jax.lax.cond(
        step_axis.any_(pending),
        functools.partial(body, k, with_hit=True),
        functools.partial(_lane_idle, pending=pending),
        carry,
    )


def _lane_more(state, max_bounces):
    """``while_loop`` condition of the untraced lane: passes left and a particle pending."""
    k, _carry, pending = state
    return (k < max_bounces) & jnp.any(pending)


def _lane_step(state, body):
    """``while_loop`` body of the untraced lane: one pass, the particles it resolved."""
    k, carry, _pending = state
    carry, hit = body(k, carry, with_hit=True)
    return k + 1, carry, hit


def _lane_idle(carry, pending):
    """The skipped lane pass: the carry unchanged, no particle pending."""
    return carry, jnp.zeros_like(pending)


def _next_surface(x, v, grid, mesh, exclude, remaining, tables=None):
    """Earliest surface along the ray: box faces only, or box faces and the stencil's
    elements (:func:`jax_dsmc.mesh.geometry.first_hit`) when a mesh is present."""
    if mesh is None:
        return first_hit(x, v, grid)
    return mesh_geometry.first_hit(x, v, grid, mesh, exclude, remaining, tables)


def _dense_distance(args, grid, mesh):
    """The post-loop surface query over every slot (the full search reach: the conditional
    time's upper end)."""
    x, v, exclude, _slot, _left = args
    return _next_surface(x, v, grid, mesh, exclude, None)[0]


def _left_distance(args):
    """The post-loop query of the remaining moves themselves (``args[-1]``), when ``epsilon
    = 0`` makes it the only one needed."""
    return args[-1]


def _lane_distance(args, grid, mesh):
    """The post-loop surface query over the lane only, ``inf`` elsewhere, along each lane
    particle's REMAINING move (``remaining``, ``f64[N]``): it decides which particles are
    still unresolved (``exhausted``), and a slot outside the lane had no event in pass 0 --
    whose search covered its whole move -- so it has nothing left to resolve.

    The remaining move, not the search reach: a level set's hit search samples its length at
    ``n_sample`` points, so over the reach (0.5 m at h = 0.0625) the spacing is 3 cm and a
    re-entry into a sliver of the body closer than that goes unseen -- the particle is then
    not counted unresolved and its final move ends inside (2026-10-21, optrules2 M 10
    iterate 68: 32 specular bounces along a concave stretch behind the nose, the 33rd hit
    1 cm away, ``inside_body`` fired).  Over the remaining move the end point is a sample, so
    a move that ends inside is always bracketed."""
    x, v, exclude, slot, remaining = args
    n = x.shape[0]
    take = jnp.maximum(slot, 0)
    d, _ = _next_surface(x[take], v[take], grid, mesh, exclude[take], remaining[take])
    return jnp.full((n,), jnp.inf).at[_drop_slot(slot, n)].set(d, mode="drop")


def _lane_dense(active, hit_slot, size):
    """:func:`lane_slots` of ``active`` over every slot."""
    del hit_slot
    return lane_slots(active, size)


def _lane_from_hits(active, hit_slot, size):
    """:func:`lane_slots` of ``active`` when every ``active`` slot is among the compacted
    ``hit_slot`` (slot order, ``-1`` unused): the compaction of those entries -- the same
    lane and overflow, from a pass over the hits instead of every slot."""
    n = active.shape[0]
    take = jnp.maximum(hit_slot, 0)
    entry, lost = lane_slots((hit_slot >= 0) & active[take], size)
    slot = jnp.where(entry >= 0, hit_slot[jnp.maximum(entry, 0)], -1)
    lost_slot = jnp.zeros(n, bool).at[jnp.where(lost, hit_slot, n)].set(True, mode="drop")
    return slot, lost_slot


NEAR_LANE = 16
"""The near lane of ``tracker="lane"`` (:func:`_lane_passes`) holds ``N // NEAR_LANE``
slots per chunk from ``COMPACT_MIN`` slots (every slot below).  Measured 2026-10-01 on the
SPARTA circle (53 k slots, steady state, 46.5 k alive): 1.8 k slots near the polygon,
2.5 k near the level set (h 0.0625) -- 4 % / 5 %; more than a chunk's worth run in
further chunks."""


def near_lane_size(n: int) -> int:
    """Slots per chunk of the near lane for ``n`` particle slots (:data:`NEAR_LANE`)."""
    return n if n < COMPACT_MIN else max(COMPACT_MIN // 2, n // NEAR_LANE)


def _box_rules(x, v, alive, used, hit, face, faces):
    """The first box event ``(used, hit, face)`` of every slot applied in closed form, as
    pass 0 of the stencil tracker applies it: ``(x_at, v_out, alive_out, full)`` -- the
    position at the event, the velocity after a specular face, the outflow / inflow
    faces' deletions, and ``full``: the events this pass leaves to the full tracker (a
    diffuse face draws and records; a periodic face's wrapped flight starts elsewhere)."""
    from jax_dsmc.boundary import BC  # noqa: PLC0415 -- break the boundary import cycle

    dim = x.shape[1]
    x_at = x + used[:, None] * v[:, :dim]
    v_out, alive_out = v, alive
    full = jnp.zeros_like(alive)
    for axis in range(dim):
        for s in (0, 1):
            selected = hit & (face == 2 * axis + s)
            rule = faces.of(axis, s)
            if rule is BC.SPECULAR:
                v_out = v_out.at[:, axis].set(jnp.where(selected, -v[:, axis], v_out[:, axis]))
            elif rule in (BC.DIFFUSE, BC.PERIODIC):
                full = full | selected
            else:
                alive_out = alive_out & ~selected
    return x_at, v_out, alive_out, full


def _box_events(x, v, alive, budget, grid, faces):
    """The dense pass's box events of ``tracker="lane"``: ``(used, hit, face, full)`` --
    every slot's first box event (time, ``0`` without one; flag; face) and ``full``, the
    slots the closed form cannot finish (:func:`_box_rules`, or a second box event within
    the move)."""
    t_box, face = first_hit(x, v, grid)
    face = face.astype(jnp.int32)
    hit = alive & (t_box <= budget)
    used = jnp.where(hit, t_box, 0.0)
    x_at, v_out, alive_out, full = _box_rules(x, v, alive, used, hit, face, faces)
    t_next, _ = first_hit(x_at, v_out, grid)
    full = full | (hit & alive_out & (used + t_next <= budget))
    return used, hit, face, full


def _chunk_slots(rank, near, c, size):
    """The slots of chunk ``c`` of the near lane, ``i32[size]`` (``-1`` unused): the near
    slots of rank ``c size .. (c + 1) size - 1`` in slot order (``rank`` the inclusive
    cumulative count minus one, computed once for every chunk; :func:`ranked_slots`)."""
    del near
    return ranked_slots(rank, c * size, size)


def _put_row(target, values, put):
    """Row 0 of ``values`` (``1, B, ...``) written to the entries ``put`` of row 0 of
    ``target`` (``1, N, ...``)."""
    return target.at[0, put].set(values[0], mode="drop")


def _near_chunk(c, state, init, rank, near, size, body, trace_dim):
    """Pass 0 of the stencil tracker on chunk ``c`` of the near lane (the slots the dense
    pass could not finish), from their pre-move state ``init``; the results scattered into
    ``state`` (the tracker carry less ``budget``).  Returns ``(state, slot)``."""
    x, v, alive, elapsed, logp, had_hit, _tally, _events, budget, exclude, _over, _ns = init
    n = x.shape[0]
    slot = _chunk_slots(rank, near, c, size)
    valid = slot >= 0
    take = jnp.maximum(slot, 0)
    sx, sv, salive, selapsed, slogp, shad, tally, head, sexclude, soverrun, n_surf = state
    lane = (
        x[take],
        # an unused entry does not move: no search reaches anything from it
        jnp.where(valid[:, None], v[take], 0.0),
        alive[take] & valid,
        elapsed[take],
        logp[take],
        had_hit[take] & valid,
        tally,
        no_wall_events(size, 1, trace_dim),
        budget[take],
        exclude[take],
        jnp.zeros(size, bool),
        n_surf,
    )
    if size >= COMPACT_MIN:
        lane, _hits = body(0, lane, draw_slot=take, with_slot=True)
    else:
        lane = body(0, lane, draw_slot=take)
    lx, lv, la, le, llp, lh, tally, events, _, lex, lover, n_surf = lane
    put = _drop_slot(slot, n)
    state = (
        sx.at[put].set(lx, mode="drop"),
        sv.at[put].set(lv, mode="drop"),
        salive.at[put].set(la, mode="drop"),
        selapsed.at[put].set(le, mode="drop"),
        slogp.at[put].set(llp, mode="drop"),
        shad.at[put].set(lh, mode="drop"),
        tally,
        jax.tree.map(functools.partial(_put_row, put=put), head, events),
        sexclude.at[put].set(lex, mode="drop"),
        soverrun.at[put].max(lover, mode="drop"),
        n_surf,
    )
    return state, slot


def _chunk_more(state, count, size):
    """``while_loop`` condition of the near lane's further chunks (untraced)."""
    return state[0] * size < count


def _chunk_step(state, chunk):
    """``while_loop`` body of the near lane's further chunks (untraced)."""
    c, carry = state
    carry, _slot = chunk(c, carry)
    return c + 1, carry


def _chunk_run(carry, c, chunk):
    return chunk(c, carry)[0]


def _chunk_skip(carry):
    return carry


def _chunk_maybe(c, carry, chunk, count, size):
    """``fori_loop`` body of the near lane's further chunks (traced: a fixed trip count,
    reverse-mode differentiable), an empty chunk skipped by ``lax.cond``."""
    return jax.lax.cond(
        c * size < count,
        functools.partial(_chunk_run, c=c, chunk=chunk),
        _chunk_skip,
        carry,
    )


def _routing_code(args, grid, mesh):
    """``i32[N]``: ``near + 2 box`` of every alive slot (:func:`_dense_pass`) -- ``near``:
    the move may meet the embedded surface (:func:`jax_dsmc.mesh.geometry.near_surface`);
    ``box``: otherwise, its straight end point is outside the box or within round-off of
    a face (a box event is possible).  No division, one table read."""
    x, v, alive, budget, tables = args
    dim = x.shape[1]
    end = x + budget[:, None] * v[:, :dim]
    slack = BOX_SLACK * (grid.hi - grid.lo)
    box = jnp.any((end <= grid.lo + slack) | (end >= grid.hi - slack), axis=1)
    near = alive & mesh_geometry.near_surface(x, v, budget, grid, mesh, tables)
    return near.astype(jnp.int32) + 2 * (alive & box & ~near).astype(jnp.int32)


BOX_SLACK = 1e-9
"""Relative slack of the dense pass's box test (:func:`_routing_code`) over the round-off of
the straight end point: a slot within it of a face takes the exact event search."""


def _no_routing(args):
    """No slot alive: nothing to route."""
    return jnp.zeros(args[0].shape[0], jnp.int32)


def _box_compact(args, grid, faces, wall, mass, tallies):
    """The dense pass's box candidates resolved on their compaction (the few candidates'
    work instead of every slot's), scattered back; one beyond the compaction's room, or
    one the closed form cannot finish, joins ``near``."""
    x, v, alive, elapsed, had_hit, head, tally, budget, box, near = args
    n = x.shape[0]
    slot, lost = lane_slots(box, lane_capacity(n, 0))
    valid = slot >= 0
    take = jnp.maximum(slot, 0)
    bx, bv, balive, bbudget = x[take], v[take], alive[take] & valid, budget[take]
    used, hit, face, full = _box_events(bx, bv, balive, bbudget, grid, faces)
    x_at, v_out, alive_out, _full = _box_rules(bx, bv, balive, used, hit, face, faces)
    dense_hit = hit & ~full
    if tallies:
        rows = (dense_hit, face, bv, v_out, alive_out)
        tally = _box_tallies(tally, rows, None, faces, wall, mass)
    put = jnp.where(dense_hit, slot, n)
    if head.time is not None:
        head = head._replace(time=head.time.at[0, put].set(used, mode="drop"))
    near = near | lost
    near = near.at[jnp.where(valid & full, slot, n)].set(True, mode="drop")
    return (
        x.at[put].set(x_at, mode="drop"),
        v.at[put].set(v_out, mode="drop"),
        alive.at[put].set(alive_out, mode="drop"),
        elapsed.at[put].add(used, mode="drop"),
        had_hit.at[put].set(True, mode="drop"),
        head,
        tally,
        near,
    )


def _dense_pass(init, grid, faces, wall, mass, tallies, mesh, tables):
    """Step 1 of :func:`_lane_passes`: route every slot, and resolve the box events of the
    slots that cannot reach the embedded surface.

    Over every slot, one elementwise kernel (:func:`_routing_code`, inside a conditional
    branch so that its result is materialised -- an elementwise result is recomputed
    inside every fusion that reads it on XLA:CPU, where `optimization_barrier` is expanded
    away before fusion): ``near`` or a possible box event.  A slot with neither keeps its
    state (the epilogue moves it).  The box candidates (a few per thousand) are compacted
    and resolved exactly as pass 0 of the stencil tracker would (:func:`_box_events`: the
    first event in closed form, the face tallies), scattered back; one the closed form
    cannot finish (a diffuse or periodic face, a second box event) joins ``near``, as does
    every candidate beyond the compaction's room.  Returns ``(state, near, rank, count)``:
    the tracker carry less ``budget``, the near flags, their inclusive rank minus one and
    their count."""
    x, v, alive, elapsed, logp, had_hit, tally, head, budget, exclude, overrun, n_surf = init
    frozen = jax.lax.stop_gradient((x, v, alive, budget))
    code = jax.lax.cond(
        jnp.any(alive),
        functools.partial(_routing_code, grid=grid, mesh=mesh),
        _no_routing,
        (*frozen, tables),
    )
    near = (code & 1) == 1
    box = (code & 2) == 2
    x, v, alive, elapsed, had_hit, head, tally, near = _box_compact(
        (x, v, alive, elapsed, had_hit, head, tally, budget, box, near),
        grid=grid,
        faces=faces,
        wall=wall,
        mass=mass,
        tallies=tallies,
    )
    rank = prefix_count(near) - 1
    count = rank[-1] + 1
    state = (x, v, alive, elapsed, logp, had_hit, tally, head, exclude, overrun, n_surf)
    return state, near, rank, count


def _first_pass_routed(init, body, grid, faces, wall, mass, tallies, mesh, tables, size, trace):
    """Pass 0 of :func:`_lane_passes` as SPARTA moves (XLA:CPU): the dense pass, then the
    near lane in chunks; and the event lane's slots.  Returns ``(state, (slot, lost),
    routing)`` -- the carry less ``budget``, the event lane, and ``(near, slot0, count,
    chunk)`` for the inside check."""
    n = init[0].shape[0]
    # 1. the dense pass
    state, near, rank, count = _dense_pass(init, grid, faces, wall, mass, tallies, mesh, tables)
    # 2. the near lane, chunk 0 unconditionally, further chunks only when it overflows
    chunk = near_lane_size(n)
    run = functools.partial(
        _near_chunk, init=init, rank=rank, near=near, size=chunk, body=body, trace_dim=trace
    )
    state, slot0 = run(0, state)
    n_chunks = -(-n // chunk)
    if n_chunks > 1:
        if trace is None:
            _, state = jax.lax.while_loop(
                functools.partial(_chunk_more, count=count, size=chunk),
                functools.partial(_chunk_step, chunk=run),
                (jnp.int32(1), state),
            )
        else:
            state = jax.lax.fori_loop(
                1,
                n_chunks,
                functools.partial(_chunk_maybe, chunk=run, count=count, size=chunk),
                state,
            )
    alive, had_hit = state[2], state[5]
    # 3. the event lane: an outflow ends the flight, every other pass-0 event of a near slot
    # may be followed by more (a dense slot's may not); compacted from the first chunk
    # unless the near slots overflowed it
    lane = jax.lax.cond(
        count <= chunk,
        functools.partial(_lane_from_hits, size=size),
        functools.partial(_lane_dense, size=size),
        alive & had_hit & near,
        slot0,
    )
    return state, lane, (near, slot0, count, chunk)


def _first_pass_dense(init, body, grid, faces, wall, mass, tallies, mesh, tables, size, trace):
    """Pass 0 of :func:`_lane_passes` over every slot (GPU: elementwise work over all
    slots is a few kernels, where the routed pass's gathers and scatters of a dozen arrays
    are many launches); and the event lane's slots, compacted from the pass's hits.  The
    same results as :func:`_first_pass_routed`; ``routing`` is ``None``."""
    del grid, faces, wall, mass, tallies, mesh, tables, trace
    n = init[0].shape[0]
    if n >= COMPACT_MIN:
        carry, (hit_slot, hit_lost) = body(0, init, with_slot=True)
    else:
        carry, hit_slot = body(0, init), None
    x, v, alive, elapsed, logp, had_hit, tally, head, _budget, exclude, overrun, n_surf = carry
    if hit_slot is None:
        lane = lane_slots(alive & had_hit, size)
    else:
        lane = jax.lax.cond(
            step_axis.any_(hit_lost),
            functools.partial(_lane_dense, size=size),
            functools.partial(_lane_from_hits, size=size),
            alive & had_hit,
            hit_slot,
        )
    state = (x, v, alive, elapsed, logp, had_hit, tally, head, exclude, overrun, n_surf)
    return state, lane, None


def _lane_passes(
    init, grid, faces, wall, mass, key, acc_delta, tallies, mesh, max_bounces, size, trace_dim
):
    """SPARTA's move (``update.cpp:879-1547``) as a dense pass and two compacted lanes
    (XLA:CPU, :func:`_first_pass_routed`; on GPU pass 0 runs over every slot instead,
    :func:`_first_pass_dense` -- the same results, fewer kernel launches).

    1. **Dense pass** over every slot (:func:`_box_events`, :func:`near_surface
       <jax_dsmc.mesh.geometry.near_surface>`): the straight move's first box event in
       closed form and the flag of the slots it cannot finish -- a path that may reach the
       embedded surface (the cut-free ring of SPARTA's optimised move, as a distance), a
       diffuse or periodic face, a second box event.  Elementwise, one fused kernel.
    2. **Near lane**: those slots compacted (chunks of :func:`near_lane_size`; more than
       one only when they overflow it) run pass 0 of the stencil tracker (box faces and
       elements, the earliest event, :func:`_apply_events`), scattered back.
    3. **Event lane**: the near slots with a pass-0 event still alive (``size`` entries,
       compacted from the first chunk) run passes ``1 .. max_bounces - 1``.

    A dense slot's move is exactly the stencil tracker's (no element within reach: its
    pass-0 event is the box face, its pass 1 finds nothing); every other slot runs the
    stencil passes with the same per-slot draws, so the result equals the stencil
    tracker's.  Returns the tracker carry with :class:`SplitEvents` in the events place
    (``slot`` = the event lane), the ``bool[N]`` slots that overflowed the event lane
    (deleted), and the routing ``(near, slot0, count, chunk)`` -- the near flags, the first
    chunk's slots, the near count and the chunk size (the inside check runs on them;
    ``None`` on GPU, where every alive slot is checked)."""
    tables = mesh_geometry.near_tables(mesh, grid)
    body = functools.partial(
        _bounce,
        grid=grid,
        faces=faces,
        wall=wall,
        mass=mass,
        key=key,
        acc_delta=acc_delta,
        tallies=tallies,
        mesh=mesh,
        tables=tables if mesh.kind == "surface" else None,
    )
    n = init[0].shape[0]
    budget = init[8]
    first = _first_pass_routed if jax.default_backend() == "cpu" else _first_pass_dense
    state, (slot, lost), routing = first(
        init, body, grid, faces, wall, mass, tallies, mesh, tables, size, trace_dim
    )
    x, v, alive, elapsed, logp, had_hit, tally, head, exclude, overrun, n_surf = state
    alive = alive & ~lost
    valid = slot >= 0
    take = jnp.maximum(slot, 0)
    lane = (
        x[take],
        jnp.where(valid[:, None], v[take], 0.0),
        alive[take] & valid,
        elapsed[take],
        logp[take],
        had_hit[take] & valid,
        tally,
        no_wall_events(size, max_bounces - 1, trace_dim),
        budget[take],
        exclude[take],
        jnp.zeros(size, bool),
        n_surf,
    )
    pass_body = functools.partial(body, row_shift=1, draw_slot=take, draw_n=n)
    if trace_dim is None:
        # an untraced move (the forward) stops after the last pass any lane particle needs;
        # the traced replay (differentiated) keeps the fixed trip count with skipped passes
        state = (jnp.int32(1), lane, lane[2])
        state = jax.lax.while_loop(
            functools.partial(_lane_more, max_bounces=max_bounces),
            functools.partial(_lane_step, body=pass_body),
            state,
        )
        lane = state[1]
    else:
        lane_body = functools.partial(_lane_pass, body=pass_body)
        lane, _ = jax.lax.fori_loop(1, max_bounces, lane_body, (lane, lane[2]))
    lx, lv, la, le, llp, lh, tally, tail, _, lex, lover, n_surf = lane
    put = _drop_slot(slot, n)
    carry = (
        x.at[put].set(lx, mode="drop"),
        v.at[put].set(lv, mode="drop"),
        alive.at[put].set(la, mode="drop"),
        elapsed.at[put].set(le, mode="drop"),
        logp.at[put].set(llp, mode="drop"),
        had_hit.at[put].set(lh, mode="drop"),
        tally,
        SplitEvents(head, tail, slot),
        budget,
        exclude.at[put].set(lex, mode="drop"),
        overrun.at[put].max(lover, mode="drop"),
        n_surf,
    )
    return carry, lost, routing


def _inside_rows(args, mesh, grid):
    """The inside check on the near lane's first chunk ``slot`` only, scattered back
    (``False`` elsewhere): every other slot is dense, provably outside."""
    x, active, slot = args
    n = x.shape[0]
    take = jnp.maximum(slot, 0)
    xs = x[take]
    inside = mesh_geometry.inside_polygon(
        xs, cell_of(xs, grid), mesh, grid, active[take] & (slot >= 0)
    )
    return jnp.zeros(n, bool).at[_drop_slot(slot, n)].set(inside, mode="drop")


def _inside_all(args, mesh, grid):
    """The inside check on every ``active`` slot."""
    x, active, _slot = args
    return mesh_geometry.inside_polygon(x, cell_of(x, grid), mesh, grid, active)


def _advance_impl(  # noqa: PLR0915 -- the three trackers' dispatch and one shared epilogue
    x,
    v,
    alive,
    grid,
    faces,
    wall,
    mass,
    dt,
    key,
    epsilon=0.0,
    acc_delta=0.0,
    max_bounces=4,
    return_tallies=False,
    return_events=False,
    mesh=None,
    tracker="stencil",
    max_walk=8,
    trace=False,
    lane_size=0,
):
    """Chronological first-event tracking and the P3 wall gradient; returns the
    ``apply_faces`` tuple.

    Specular, diffuse, periodic and absorbing faces participate in the same
    chronological event search in any ``D``. ``recrossed`` reports exhausted event
    capacity.  All parameters use physical units, including the wall temperature in
    kelvin.  The conditional-time estimator is validated by one-particle quadrature
    in 1-D; the deterministic tracking agrees with the legacy fold path in D=1,2,3.

    With ``mesh`` (a :class:`jax_dsmc.mesh.types.Mesh`) the embedded surface joins the
    same event search.  ``tracker="stencil"`` keeps the pass structure above with the
    per-cell candidate list (``mesh.candidates``) and counts ``stencil_overrun``;
    ``tracker="walk"`` is SPARTA's cell walk (:mod:`jax_dsmc.mesh.walk`, ``max_walk``
    iterations of one cell crossing or one hit each), exact for any move length.  All
    three trackers end with the inside-body check that kills and counts a particle a missed hit left in
    the body.  The fourth element of the result is then a :class:`TrackerCounts`
    instead of the bare ``exhausted`` count, and the tallies have ``2D + E`` rows.

    ``trace=True`` (with ``return_events``) also fills the hit point, incident velocity
    and normal of every recorded hit (:class:`WallEvents`); the shape adjoint's replay
    uses it, a forward record never does.

    ``tracker="lane"`` (mesh only) is SPARTA's move structure (:func:`_lane_passes`): a
    dense closed-form pass over every particle for the moves that provably cannot reach
    the embedded surface (SPARTA's optimised move), pass 0 of the stencil tracker on the
    compacted rest (the near lane), and passes ``1 .. max_bounces - 1`` on the particles
    with a pass-0 event (a lane of ``lane_capacity(N, lane_size)`` entries); a particle
    still unresolved after them is deleted (``exhausted``, SPARTA MAXSTUCK) and an
    overfull event lane deletes and counts the rest (``lane_overflow``).  With an embedded
    surface the re-emission draws are per slot (:func:`_slot_uniforms`), so the result
    equals the stencil tracker's with the same ``max_bounces``; the events are
    :class:`SplitEvents`.
    """
    n = x.shape[0]
    dim = x.shape[1]
    # the regularised flight time only exists for epsilon > 0; at 0 every quantity below
    # is exactly its dt value, so the draws and special functions are skipped (a runtime
    # branch: epsilon is traced)
    regularised = jnp.asarray(epsilon) > 0.0
    z = jax.lax.cond(
        regularised,
        functools.partial(_time_draws, n=n),
        functools.partial(_no_time_draws, n=n),
        key,
    )
    raw = dt + epsilon * z
    budget = jnp.maximum(raw, 0.0)
    originally_alive = alive
    n_surfaces = 2 * dim + (mesh.n_elements if mesh is not None else 0)
    init = (
        x,
        v,
        alive,
        jnp.zeros(n),
        jnp.zeros(n),
        jnp.zeros(n, bool),
        zero(n_surfaces),
        no_wall_events(
            n,
            {"stencil": max_bounces, "lane": 1}.get(tracker, max_walk),
            dim if trace else None,
        ),
        budget,
        jnp.full((n,), -1, jnp.int32),
        jnp.zeros(n, bool),
        jnp.int32(0),
    )
    if mesh is None or tracker == "stencil":
        body = functools.partial(
            _bounce,
            grid=grid,
            faces=faces,
            wall=wall,
            mass=mass,
            key=key,
            acc_delta=acc_delta,
            tallies=return_tallies,
            mesh=mesh,
        )
        carry = jax.lax.fori_loop(0, max_bounces, body, init)
        x, v, alive, elapsed, logp, had_hit, tally, events, _, exclude, overrun, n_surf = carry
        distance, _ = _next_surface(x, v, grid, mesh, exclude, None)
        # unresolved: a hit left within the REMAINING move, searched along that move (its
        # end point a sample; :func:`_lane_distance`)
        left, _ = _next_surface(x, v, grid, mesh, exclude, budget - elapsed)
        exhausted = alive & (elapsed + left <= budget)
        if mesh is not None:
            # SPARTA `MAXSTUCK`: a particle still bouncing in a concave pocket after
            # `max_bounces` passes is deleted and counted (`surface_stuck`), not left with
            # an unresolved hit
            alive = alive & ~exhausted
    elif tracker == "lane":
        carry, lost, routing = _lane_passes(
            init,
            grid,
            faces,
            wall,
            mass,
            key,
            acc_delta,
            return_tallies,
            mesh,
            max_bounces,
            lane_capacity(n, lane_size),
            dim if trace else None,
        )
        x, v, alive, elapsed, logp, had_hit, tally, events, _, exclude, overrun, n_surf = carry
        # the lane particles' remaining moves (only they can be left unresolved), and the
        # next surface over every slot when epsilon > 0 (the conditional time's upper end)
        left = _lane_distance((x, v, exclude, events.slot, budget - elapsed), grid, mesh)
        distance = jax.lax.cond(
            jnp.asarray(epsilon) > 0.0,
            functools.partial(_dense_distance, grid=grid, mesh=mesh),
            _left_distance,
            (x, v, exclude, events.slot, left),
        )
        # SPARTA MAXSTUCK, as for the stencil: the lane particles still unresolved
        exhausted = alive & (elapsed + left <= budget)
        alive = alive & ~exhausted
    elif tracker == "walk":
        from jax_dsmc.mesh.walk import walk  # noqa: PLC0415 -- walk imports this module

        carry, stuck = walk(
            init, grid, faces, wall, mass, key, acc_delta, return_tallies, mesh, max_walk
        )
        x, v, alive, elapsed, logp, had_hit, tally, events, _, exclude, overrun, n_surf = carry
        # SPARTA `MAXSTUCK`: a particle the walk could not finish is discarded
        exhausted = stuck
        alive = alive & ~stuck
        # the conditional-time interval needs the next surface; the stencil query is exact
        # for it whenever a near-miss within the regularisation is at stake
        distance, _ = _next_surface(x, v, grid, mesh, exclude, None)
    else:
        raise ValueError(f"tracker must be 'stencil', 'lane' or 'walk', got {tracker!r}")
    lower = jnp.where(had_hit, elapsed, -jnp.inf)
    upper = jnp.where(alive, elapsed + distance, jnp.inf)
    time, interval_score = jax.lax.cond(
        regularised, conditional_time, _fixed_time, raw, z, lower, upper, dt, epsilon
    )
    x = x + jnp.where(alive, time - elapsed, 0.0)[:, None] * v[:, :dim]
    logp = logp + interval_score
    x = jnp.clip(x, grid.lo, grid.hi)
    counts = jnp.sum(exhausted)
    lane_lost = lost if (mesh is not None and tracker == "lane") else None
    if mesh is not None:
        # SPARTA's catch-all: a particle inside the body after the move was mis-tracked
        if tracker == "lane" and routing is not None:
            # a dense slot's move provably stays clear of the body: only the near lane's
            # slots are tested (the first chunk's, every near slot when it overflowed)
            near, slot0, n_near, chunk = routing
            inside = jax.lax.cond(
                n_near <= chunk,
                functools.partial(_inside_rows, mesh=mesh, grid=grid),
                functools.partial(_inside_all, mesh=mesh, grid=grid),
                (x, alive & near, slot0),
            )
        else:
            inside = mesh_geometry.inside_polygon(x, cell_of(x, grid), mesh, grid, alive)
        alive = alive & ~inside
        counts = TrackerCounts(
            jnp.sum(exhausted, dtype=jnp.int32),
            n_surf,
            jnp.sum(overrun, dtype=jnp.int32),
            jnp.sum(inside, dtype=jnp.int32),
            jnp.int32(0) if lane_lost is None else jnp.sum(lane_lost, dtype=jnp.int32),
        )
    result = (
        x,
        v,
        alive,
        counts,
        jnp.where(originally_alive, logp, 0.0),
    )

    if return_events:
        result = (*result, events)
    return (*result, tally) if return_tallies else result


_advance_jitted = jax.jit(
    _advance_impl,
    static_argnames=(
        "faces",
        "max_bounces",
        "return_tallies",
        "return_events",
        "tracker",
        "max_walk",
        "trace",
        "lane_size",
    ),
)


@functools.wraps(_advance_impl)
def advance(
    x,
    v,
    alive,
    grid,
    faces,
    wall,
    mass,
    dt,
    key,
    epsilon=0.0,
    acc_delta=0.0,
    max_bounces=4,
    return_tallies=False,
    return_events=False,
    mesh=None,
    tracker="stencil",
    max_walk=8,
    trace=False,
    lane_size=0,
):
    # Eager calls need a stable compilation boundary; an enclosing jit/grad already
    # supplies tracing. Bypass the extra boundary there to keep its AD graph small.
    leaves = jax.tree.leaves((x, v, alive, grid, wall, mass, dt, key, epsilon, acc_delta, mesh))
    traced = any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)
    kernel = _advance_impl if traced else _advance_jitted
    return kernel(
        x,
        v,
        alive,
        grid,
        faces,
        wall,
        mass,
        dt,
        key,
        epsilon=epsilon,
        acc_delta=acc_delta,
        max_bounces=max_bounces,
        return_tallies=return_tallies,
        return_events=return_events,
        mesh=mesh,
        tracker=tracker,
        max_walk=max_walk,
        trace=trace,
        lane_size=lane_size,
    )
