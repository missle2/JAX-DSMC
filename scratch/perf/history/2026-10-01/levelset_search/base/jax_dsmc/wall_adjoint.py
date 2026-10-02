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

from jax_dsmc import sampling
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
    pos = jnp.cumsum(has.astype(jnp.int32)) - 1
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
    compaction, out-of-range writes dropped): ``(slot i32[size], overflow bool[N])`` --
    ``slot[j]`` the ``j``-th active slot in slot order or ``-1``, ``overflow`` the active
    slots that did not fit."""
    n = active.shape[0]
    pos = jnp.cumsum(active.astype(jnp.int32)) - 1
    fits = active & (pos < size)
    target = jnp.where(fits, pos, size)
    slot = (
        jnp.full((size,), -1, jnp.int32).at[target].set(jnp.arange(n, dtype=jnp.int32), mode="drop")
    )
    return slot, active & ~fits


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
"""Rows per pass from which the diffuse re-emissions are computed on the compacted element
hit rows only (:func:`_element_events`); below it (lane passes, entrants) every row."""


def _reemission(rows, mass):
    """The diffuse re-emission of each row from its uniforms ``u_z, u_r`` at its hit
    (``normal``, wall ``temp`` and ``vwall``, incident ``v``), ``f64[R, 3]``."""
    normal, v, u_z, u_r, temp, vwall = rows
    z, r = sampling.diffuse_draws_from_uniforms(u_z, u_r)
    return sampling.diffuse_velocity_general(z, r, normal, temp[:, None], mass, vwall, v)


def _dense_reemission(operands, mass):
    """:func:`_reemission` of every row."""
    return _reemission(operands[:-1], mass)


def _compact_reemission(operands, mass):
    """:func:`_reemission` of the compacted rows ``slot`` (``-1`` unused), scattered back
    over zeros."""
    *rows, slot = operands
    n = rows[0].shape[0]
    take = jnp.maximum(slot, 0)
    fresh = _reemission(tuple(a[take] for a in rows), mass)
    return jnp.zeros((n, 3)).at[_drop_slot(slot, n)].set(fresh, mode="drop")


def _element_events(x_at, v, logp, tally, events, exclude, hit, surface, draws, k, operands):
    """The element branch of one pass: reflect off the embedded surface (SPARTA
    ``surf_collide_specular.cpp:75-150`` / ``surf_collide_diffuse.cpp:206-306``).

    Every element carries the ``diffuse T acc`` law of its model row of ``WallParams``
    (``2D + model``; ``acc = 0`` is specular).  The normal is the flat element normal
    (``mesh.normals[e]``) or the exact one of an analytic body, in the flow direction;
    the diffuse frame is :func:`sampling._tangent_frame`.  Element models use the
    isotropic wall temperature (``wall.temp[row]`` or its first component).
    """
    grid, wall, mass, acc_delta, tallies, mesh = operands
    dim = x_at.shape[1]
    selected = hit & (surface >= 2 * dim)
    e = jnp.clip(surface - 2 * dim, 0, mesh.n_elements - 1)
    model = mesh.body.model[e] if mesh.kind == "surface" else jnp.zeros_like(e)
    row = 2 * dim + model
    temp_rows = wall.temp if wall.temp.ndim == 1 else wall.temp[:, 0]
    temp = temp_rows[row]
    acc = wall.acc[row]
    vwall = wall.vwall[row]
    u_z, u_r, u = draws  # uniforms: transformed on the rows that use them (_hit_frames)
    normal = (
        jnp.zeros((x_at.shape[0], 3))
        .at[:, :dim]
        .set(mesh_geometry.surface_normal(mesh, surface, x_at, grid))
    )
    # the diffuse re-emission matters only on the rows that hit an element; over a whole
    # population those are a few in a thousand, so they are compacted (the lane's size; an
    # overflow computes every row) -- row for row the same arithmetic, zeros elsewhere
    # (never read: every use below is masked by `selected`).  The normal stays dense: a
    # level set's is a sum of products whose FMA contraction XLA:CPU chooses per kernel
    rows = (normal, v, u_z, u_r, temp, vwall)
    n = x_at.shape[0]
    if n >= COMPACT_MIN:
        slot, lost = lane_slots(selected, lane_capacity(n, 0))
        fresh = jax.lax.cond(
            jnp.any(lost),
            functools.partial(_dense_reemission, mass=mass),
            functools.partial(_compact_reemission, mass=mass),
            (*rows, slot),
        )
    else:
        fresh = _reemission(rows, mass)
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
    v_new = jnp.where(selected[:, None], v_out, v)
    v_new_tally = jnp.where(
        selected[:, None], w[:, None] * fresh_traced + (1.0 - w)[:, None] * v_spec, v
    )
    logp = logp + jnp.where(selected, log_p, 0.0)
    events = events._replace(
        face=events.face.at[k].set(jnp.where(selected, surface, events.face[k])),
        v_out=events.v_out.at[k].set(jnp.where(selected[:, None], v_out, events.v_out[k])),
        diffuse=events.diffuse.at[k].set(jnp.where(selected, took_diffuse, events.diffuse[k])),
    )
    events = _trace_hit(events, k, selected, x_at, v, normal)
    # SPARTA `update.cpp`: the element just left is excluded from the next search
    exclude = jnp.where(selected, e, exclude)
    if tallies:
        tally = _surface_tally(tally, surface, selected, v, v_new_tally, mass, vwall)
    return v_new, logp, tally, events, exclude


def _pass_draws(pass_key, n, faces, draw_slot, draw_n):
    """One pass's re-emission randomness: the uniforms ``(u_z, u_r, u_acc)`` and the box
    faces' ``draws`` ``(z, r, u_acc)``.  The uniforms are drawn over every slot (a lane
    entry gathers its slot's, so a particle meets the same randomness in any tracker); the
    transforms to normal / Rayleigh variates run only where a re-emission can use them --
    every row when a box face is diffuse, the element hit rows (:func:`_hit_frames`)."""
    from jax_dsmc.boundary import BC  # noqa: PLC0415 -- break the boundary import cycle

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
    operands, grid, faces, wall, mass, acc_delta, tallies, mesh=None, draw_slot=None, draw_n=0
):
    """Resolve the events of one pass for the particles flagged ``hit``.

    ``draw_slot`` (lane passes): the diffuse draws are the pass's draws over all
    ``draw_n`` slots gathered at each lane entry's slot, so a particle meets exactly the
    randomness the dense tracker would give it in that pass."""
    from jax_dsmc.boundary import (  # noqa: PLC0415 -- break the boundary dispatch import cycle
        BC,
        diffuse_from_draws,
    )

    x, v, alive, logp, tally, events, hit, surface, used, pass_key, k, exclude, at_time = operands
    dim = x.shape[1]
    n = x.shape[0]
    u_z, u_r, u_acc, draws = _pass_draws(pass_key, n, faces, draw_slot, draw_n)
    x_at = x + used[:, None] * v[:, :dim]
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
                outgoing = jnp.where(alive[:, None], v, 0.0)
                velocity_wall = wall.vwall[f] if rule is BC.DIFFUSE else jnp.zeros(3)
                tally = _face_tally(tally, f, selected, incident, outgoing, mass, velocity_wall)
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
    """
    x, v, alive, elapsed, logp, had_hit, tally, events, budget, exclude, overrun, n_surf = carry
    distance, surface = _next_surface(x, v, grid, mesh, exclude, budget - elapsed)
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
    x, v, alive, logp, tally, events, exclude = jax.lax.cond(
        jnp.any(hit),
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
        ),
        _skip_events,
        operands,
    )
    elapsed = elapsed + used
    had_hit = had_hit | hit
    carry = (x, v, alive, elapsed, logp, had_hit, tally, events, budget, exclude, overrun, n_surf)
    return (carry, hit & alive) if with_hit else carry


def _lane_pass(k, state, body):
    """One lane pass, skipped outright (``lax.cond``, so the hit search is not paid) when
    no lane particle had an event in the previous pass: one without an event in pass
    ``k - 1`` has none in pass ``k``, so the skip is exact.  ``state = (carry, pending)``;
    reverse-mode differentiable, unlike a ``while_loop``, so the shape adjoint's replay
    runs it too."""
    carry, pending = state
    return jax.lax.cond(
        jnp.any(pending),
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


def _next_surface(x, v, grid, mesh, exclude, remaining):
    """Earliest surface along the ray: box faces only, or box faces and the stencil's
    elements (:func:`jax_dsmc.mesh.geometry.first_hit`) when a mesh is present."""
    if mesh is None:
        return first_hit(x, v, grid)
    return mesh_geometry.first_hit(x, v, grid, mesh, exclude, remaining)


def _dense_distance(args, grid, mesh):
    """The post-loop surface query over every slot."""
    x, v, exclude, _slot = args
    return _next_surface(x, v, grid, mesh, exclude, None)[0]


def _lane_distance(args, grid, mesh):
    """The post-loop surface query over the lane only, ``inf`` elsewhere: with ``epsilon =
    0`` the query decides only which particles are still unresolved (``exhausted``), and a
    slot outside the lane had no event in pass 0 -- whose search covered its whole move --
    so it has nothing left to resolve."""
    x, v, exclude, slot = args
    n = x.shape[0]
    take = jnp.maximum(slot, 0)
    d, _ = _next_surface(x[take], v[take], grid, mesh, exclude[take], None)
    return jnp.full((n,), jnp.inf).at[_drop_slot(slot, n)].set(d, mode="drop")


def _lane_passes(
    init, grid, faces, wall, mass, key, acc_delta, tallies, mesh, max_bounces, size, trace_dim
):
    """Pass 0 of the stencil tracker over every slot, then passes ``1 .. max_bounces - 1``
    on the lane of slots with a pass-0 event (:func:`lane_slots`), scattered back.
    Returns the tracker carry with :class:`SplitEvents` in the events place (its ``slot``
    the lane), and the ``bool[N]`` slots that overflowed the lane (deleted)."""
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
    )
    carry = body(0, init)
    x, v, alive, elapsed, logp, had_hit, tally, head, budget, exclude, overrun, n_surf = carry
    n = x.shape[0]
    # an outflow ends the flight; every other pass-0 event may be followed by more
    slot, lost = lane_slots(alive & had_hit, size)
    alive = alive & ~lost
    valid = slot >= 0
    take = jnp.maximum(slot, 0)
    lane = (
        x[take],
        v[take],
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
    return carry, lost


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

    ``tracker="lane"`` (mesh only) is SPARTA's move loop: pass 0 over every particle as
    the stencil does; the particles with a pass-0 event are compacted into a lane of
    ``lane_capacity(N, lane_size)`` entries and passes ``1 .. max_bounces - 1`` run on the
    lane alone (one hit search over the lane per pass instead of over all ``N``); a
    particle still unresolved after them is deleted (``exhausted``, SPARTA MAXSTUCK) and
    an overfull lane deletes and counts the rest (``lane_overflow``).  Each lane pass uses
    the draws the dense pass would give that slot, so the result equals the stencil
    tracker's with the same ``max_bounces``; the events are :class:`SplitEvents`.
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
        exhausted = alive & (elapsed + distance <= budget)
        if mesh is not None:
            # SPARTA `MAXSTUCK`: a particle still bouncing in a concave pocket after
            # `max_bounces` passes is deleted and counted (`surface_stuck`), not left with
            # an unresolved hit
            alive = alive & ~exhausted
    elif tracker == "lane":
        carry, lost = _lane_passes(
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
        # the next surface: over every slot when epsilon > 0 (the conditional time's upper
        # end), else over the lane alone (only its particles can be left unresolved)
        distance = jax.lax.cond(
            jnp.asarray(epsilon) > 0.0,
            functools.partial(_dense_distance, grid=grid, mesh=mesh),
            functools.partial(_lane_distance, grid=grid, mesh=mesh),
            (x, v, exclude, events.slot),
        )
        # SPARTA MAXSTUCK, as for the stencil: the lane particles still unresolved
        exhausted = alive & (elapsed + distance <= budget)
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
        inside = alive & mesh_geometry.inside_polygon(x, cell_of(x, grid), mesh, grid)
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
