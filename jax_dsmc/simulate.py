"""The time loop (spec section 12).

Intra-step ordering is **`move -> sort -> collide`**, matching both reference
codes (SPARTA ``update.cpp:337-364``; [DF]'s ``move -> buildCellOccupancy ->
collisions``).

> **This is NOT [P3]'s ordering, and the difference is real.** [P3] specifies
> `collide -> move`, which puts the transport term *inside* ``B_k``:
> ``beta_k = B(beta_{k+1} + dt P* alpha_{k+1})``.  Ours puts it outside:
> ``beta_k = B beta_{k+1} + dt P* alpha_{k+1})``.  Verified by taking real
> ``jax.vjp``s of both (exact match one way, `2.45e-02` mismatch the other), and
> the forms differ by `O(dt ||B - I||)`, which does **not** vanish at realistic
> timesteps.
>
> This costs nothing in production -- ``jax.grad`` differentiates the loop as
> written, so the computed gradient is correct either way.  It matters only to
> someone hand-checking the adjoint against the paper's formula.  Note that the 0D
> oracles of section 13.1/13.2 provably **cannot** detect it (``alpha == 0`` there,
> so both forms collapse), which is why section 13.3's 1D two-step oracle exists
> (gap U5).

Every loop here is a ``scan`` with a **static** trip count -- the time loop, the
attempt-slot loop, and the wall-bounce loop.  No ``while_loop`` anywhere: a
data-dependent trip count is neither reverse-differentiable nor statically shaped,
and the jaxpr audit in the test suite enforces it (for the collision step).  One
exception (2026-09-24): the *untraced* move of ``tracker="lane"`` stops its lane passes
with a ``while_loop`` once no particle needs another; the traced replay the shape adjoint
differentiates keeps the fixed trip count (``wall_adjoint._lane_passes``).
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from jax import lax

from jax_dsmc import boundary, inflow, sample, sort
from jax_dsmc.boundary import BC
from jax_dsmc.collide.ntc import (
    collide,
    decision_weights,
    post_collision_velocities,
    step_v_post,
)
from jax_dsmc.move import move_with_decisions
from jax_dsmc.rng import STAGE_INFLOW, STAGE_MOVE, key_for
from jax_dsmc.state import CollideState, ParticleState, lane_capacity
from jax_dsmc.validation import require_valid, validate_inputs
from jax_dsmc.wall_adjoint import (
    RecordEvents,
    SplitEvents,
    WallEvents,
    compact_record,
    lane_slots,
    no_wall_events,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class EventRecord:
    """One step's stochastic events, as the ensemble adjoint needs them (stacked over steps).

    ``x``/``v``/``alive`` are the state the collisions were drawn from (after transport,
    walls and injection); the post-collision state is the *next* step's pre-move state.
    ``born_face`` is ``-1`` except on slots born this step, whose sampled velocity is
    ``born_v``.  ``pairs`` is :class:`jax_dsmc.collide.ntc.PairRecord`.
    """

    x: jax.Array
    v: jax.Array
    alive: jax.Array
    born_face: jax.Array
    born_v: jax.Array | None
    """``None`` in a lean record (``SimStatic.record = "lean"``): read by the host adjoint
    only."""
    pairs: object
    v_post: jax.Array | None
    """velocities after the collisions: with ``x`` and ``alive`` this is the state at
    ``t_{k+1}``.  ``None`` in a lean record -- :func:`post_collision_v` rebuilds it bitwise
    from ``v`` and ``pairs``."""
    walls: object
    """The step's recorded wall events: diffuse-face hits and, with a mesh, every element
    hit -- :class:`jax_dsmc.wall_adjoint.WallEvents` (``K, N``) for the stencil and walk
    trackers, :class:`jax_dsmc.wall_adjoint.RecordEvents` (pass 0 dense, later passes as a
    list, :func:`jax_dsmc.wall_adjoint.compact_record`) for ``tracker="lane"``."""
    sigma: jax.Array
    """``f64[C]`` the collision majorant the step's attempts were accepted against."""


jax.config.update("jax_enable_x64", True)  # section 0.3: f64 throughout


def post_collision_v(rec):
    """``v_post`` of one step's (or a stack of steps') record: the stored array, or for a
    lean record the exact replay of the step's collision writes
    (:func:`jax_dsmc.collide.ntc.post_collision_velocities`, vmapped over a leading step
    axis when ``rec`` is stacked)."""
    if rec.v_post is not None:
        return rec.v_post
    if rec.v.ndim == 3:
        return jax.vmap(post_collision_velocities)(rec.v, rec.pairs)
    return step_v_post(rec)


def _lean_faces(face, dim):
    """A lean record's surface ids: ``-1`` (no event), the box face ``0..2D-1``, or ``2D``
    for any body element (the element index is read by the host adjoint only), ``i8``."""
    return jnp.minimum(face, 2 * dim).astype(jnp.int8)


def lean_record(record):
    """``record`` (one step's or a stack's :class:`EventRecord`) reduced to what the device
    backward and the shape VJP read: no ``v_post`` (:func:`post_collision_v`), ``born_v``,
    pair ``vi`` / ``vj`` (:func:`jax_dsmc.collide.ntc.step_pairs`) or wall ``v_out`` /
    ``diffuse``; ``born_face`` (a box face, ``< 2D``) and the wall surface ids as ``i8``
    (:func:`_lean_faces` -- ``>= 0`` and ``>= 2D`` keep their meaning).  The forward is
    untouched: this only selects what the captured run keeps."""
    dim = record.x.shape[-1]
    walls = record.walls
    if isinstance(walls, RecordEvents):
        walls = walls._replace(
            head=WallEvents(_lean_faces(walls.head.face, dim), None, None),
            face=_lean_faces(walls.face, dim),
            v_out=None,
            diffuse=None,
        )
    elif isinstance(walls, WallEvents):
        walls = WallEvents(_lean_faces(walls.face, dim), None, None)
    pairs = record.pairs
    if pairs is not None:
        pairs = pairs._replace(vi=None, vj=None)
    return dataclasses.replace(
        record,
        born_face=record.born_face.astype(jnp.int8),
        born_v=None,
        v_post=None,
        pairs=pairs,
        walls=walls,
    )


def _merge_born_split(walls, born_walls, born, size):
    """Entrants' dense birth-step events (``K_bounce`` rows over every slot) into a lane
    record: pass 0 into the head row, the later passes of the entrants that have any
    compacted into ``size`` extra tail entries (appended to the movers' lane).  Returns
    ``(SplitEvents, lost)`` with ``lost`` the entrants whose later events did not fit."""
    head = WallEvents(
        jnp.where(born[None, :], born_walls.face[:1], walls.head.face),
        jnp.where(born[None, :, None], born_walls.v_out[:1], walls.head.v_out),
        jnp.where(born[None, :], born_walls.diffuse[:1], walls.head.diffuse),
    )
    later = born & jnp.any(born_walls.face[1:] >= 0, axis=0)
    slot, lost = lane_slots(later, size)
    take = jnp.maximum(slot, 0)
    valid = (slot >= 0)[None, :]
    tail = WallEvents(
        jnp.concatenate((walls.tail.face, jnp.where(valid, born_walls.face[1:, take], -1)), axis=1),
        jnp.concatenate(
            (walls.tail.v_out, jnp.where(valid[..., None], born_walls.v_out[1:, take], 0.0)),
            axis=1,
        ),
        jnp.concatenate((walls.tail.diffuse, valid & born_walls.diffuse[1:, take]), axis=1),
    )
    return SplitEvents(head, tail, jnp.concatenate((walls.slot, slot))), jnp.sum(
        lost, dtype=jnp.int32
    )


def _born_lane(static):
    """Tail entries reserved for entrants with more than one recorded birth-step event."""
    return max(64, lane_capacity(static.N, static.lane_size) // 4)


def _empty_split(static):
    """The empty lane record of a step without events (the shapes of a real one)."""
    n, k = static.N, static.K_bounce
    size = lane_capacity(n, static.lane_size) + _born_lane(static)
    return SplitEvents(
        no_wall_events(n, 1), no_wall_events(size, k - 1), jnp.full((size,), -1, jnp.int32)
    )


def _record_capacity(static):
    """Later-pass events a lane record stores per step (:func:`compact_record`): a quarter
    of the lane.  A particle rarely has a recorded event (an element hit or a diffuse face)
    after its first; measured on the roughest optimised shape, the deepest recorded pass in
    20 steady moves was 1 -- the list replaces ``K - 1`` mostly empty rows over the lane."""
    return max(64, (lane_capacity(static.N, static.lane_size) + _born_lane(static)) // 4)


def _event_record(moved, source, wall_events, captured, state, static, diffuse, mesh=None):
    """Assemble one step's :class:`EventRecord`; entrants' birth-step hits are theirs.
    Also returns ``(lost, overflow)``: the entrants lost to a full lane and the later-pass
    events a lane record had no room for (``tracker="lane"``; else 0)."""
    lane = mesh is not None and static.tracker == "lane"
    k_events = static.K_walk if static.tracker == "walk" else static.K_bounce
    lost = jnp.int32(0)
    if lane:
        walls = wall_events[0] if diffuse else _empty_split(static)
    else:
        walls = wall_events[0] if diffuse else no_wall_events(static.N, k_events)
    if diffuse and source is not None:
        born = source.born_face >= 0
        if lane:
            walls, lost = _merge_born_split(walls, source.born_walls, born, _born_lane(static))
        else:
            walls = WallEvents(
                jnp.where(born[None, :], source.born_walls.face, walls.face),
                jnp.where(born[None, :, None], source.born_walls.v_out, walls.v_out),
                jnp.where(born[None, :], source.born_walls.diffuse, walls.diffuse),
            )
    elif lane and diffuse:
        # no sources: pad the tail to the record's fixed width
        pad = _born_lane(static)
        walls = SplitEvents(
            walls.head,
            jax.tree.map(
                lambda a, b: jnp.concatenate((a, b), axis=1),
                walls.tail,
                no_wall_events(pad, static.K_bounce - 1),
            ),
            jnp.concatenate((walls.slot, jnp.full((pad,), -1, jnp.int32))),
        )
    record = EventRecord(
        x=moved.x,
        v=moved.v,
        alive=moved.alive,
        born_face=source.born_face if source is not None else jnp.full(static.N, -1, jnp.int32),
        born_v=source.born_v if source is not None else jnp.zeros((static.N, 3)),
        pairs=captured[0],
        v_post=state.v,
        walls=walls,
        sigma=captured[1],
    )
    overflow = jnp.int32(0)
    if lane:
        # store the later passes as a list of the recorded events, not K - 1 lane rows
        walls, overflow = compact_record(walls, _record_capacity(static))
        record = dataclasses.replace(record, walls=walls)
    if static.record == "lean":
        record = lean_record(record)
    return record, (lost, overflow)


def _tracking_diagnostics(counts, tracker):
    """Split a ``wall_adjoint.TrackerCounts`` into the old ``recrossed`` slot and the mesh
    diagnostics of :class:`Diagnostics` (all of which must stay zero)."""
    tracking = {
        "surface_hits": counts.surface_hits,
        "stencil_overrun": counts.stencil_overrun,
        "inside_body": counts.inside_body,
    }
    if tracker == "walk":
        tracking["walk_exhausted"] = counts.exhausted
        return counts.exhausted, tracking
    tracking["lane_overflow"] = jnp.asarray(counts.lane_overflow, jnp.int32)
    # stencil: the stuck particles were deleted (SPARTA MAXSTUCK), a rate, not a failure
    tracking["surface_stuck"] = counts.exhausted
    return jnp.zeros_like(counts.exhausted), tracking


def _step(  # noqa: PLR0912, PLR0915 -- one timestep, kept whole so the event order is readable
    carry,
    k,
    grid,
    sp,
    params,
    static,
    base_key,
    bc,
    collide_only,
    emit_moments,
    reward,
    faces=None,
    wall=None,
    res=None,
    capacity=0,
    score_spread="particle",
    acc_delta=0.0,
    score_mode="paper",
    capture=False,
    mesh=None,
    emit_surface=False,
):
    """One timestep.  Module-level and bound with ``partial`` (decision 14).

    ``collide_only`` short-circuits advection for the 0D milestone-1 case, where
    there is a single cell and transport is identically a no-op.  ``emit_moments``
    selects whether the per-step output is the full :class:`CellMoments` or just
    the per-cell temperature.  Both are STATIC Python bools, so they select a code
    path at trace time rather than branching on a traced value.

    ``reward`` is the third: ``None`` (the default) leaves the carry as
    ``(state, cstate)``; a callable ``(state, table) -> f64`` extends it to
    ``(state, cstate, J)`` and accumulates a **windowed** objective in the causal
    DiCE form of section 11.5.  It is called *after* ``collide``, so
    ``state.logh_acc`` is the running prefix ``L_{<=k}`` and each reward is scored
    only against the decisions that preceded it.  ``table`` is still valid there:
    ``collide`` rewrites velocities but never positions, so cell occupancy is
    unchanged.
    """
    partition = sort.partition_active(static)
    if partition:
        # the carried cell order rides last in the carry (`run` adds and removes it)
        *carry, sort_carry = carry
    if reward is None:
        state, cstate = carry
    else:
        state, cstate, j_acc = carry
    bc_chk = jnp.float64(0.0)
    recrossed = jnp.array(0, jnp.int32)
    injected = jnp.array(0, jnp.int32)
    shortfall = jnp.float64(0.0)
    wall_events = []
    tracking = {}
    tallies = None
    if not collide_only:
        if faces is None:
            state, fold = move_with_decisions(state, grid, params.dt, bc)
            # the boundary fold index is a frozen decision too -- see boundary.fold_index
            w = decision_weights(fold.size).reshape(fold.shape)
            bc_chk = lax.stop_gradient(jnp.sum(w * fold))
        else:
            # per-face path (section 9.4): thermal walls, inflow, mixed geometries
            k_move = key_for(base_key, k, STAGE_MOVE)
            record_walls = capture and (BC.DIFFUSE in faces.bcs or mesh is not None)
            x, v, alive, recrossed, logh_bc, *extra = boundary.apply_faces(
                state.x,
                state.v,
                state.alive,
                grid,
                faces,
                wall,
                sp.mass[0],
                params.dt,
                k_move,
                epsilon=params.epsilon,
                score_spread=score_spread,
                acc_delta=acc_delta,
                max_bounces=static.K_bounce,
                return_events=record_walls,
                mesh=mesh,
                tracker=static.tracker,
                max_walk=static.K_walk,
                return_tallies=emit_surface,
                lane_size=static.lane_size,
            )
            # `advance` appends the events, then the tallies
            if emit_surface:
                tallies = extra.pop()
            wall_events = extra
            if mesh is not None:
                recrossed, tracking = _tracking_diagnostics(recrossed, static.tracker)
            # No fold term here (2026-10-01): `apply_faces` returns resolved in-box
            # positions, whose fold index is 0 for every live particle bar one exactly on
            # an upper face -- the term fingerprinted dead slots' stale positions, and
            # deaths are already in the collision checksum's cell term.  It cost 0.5 ms
            # per step at N = 6e4 on one CPU core.
            # M7: the boundary decisions' score joins the collision score in the same
            # accumulator, because `surrogate` weights the reward by `grad` of the WHOLE
            # prefix `L_{<=k}` -- a separate accumulator would drop the cross terms.
            state = ParticleState(
                x=x,
                v=v,
                species=state.species,
                alive=alive,
                logh_acc=state.logh_acc
                + (jnp.sum(logh_bc) if score_mode == "trajectory" else logh_bc),
            )
            if res is not None:
                state, injected, shortfall, source_details = inflow.inject(
                    state,
                    grid,
                    faces,
                    res,
                    sp.mass[0],
                    params.fnum[0],
                    params.dt,
                    key_for(base_key, k, STAGE_INFLOW),
                    capacity,
                    wall=wall,
                    max_bounces=static.K_bounce,
                    track=True,
                    score_mode=score_mode,
                    return_diagnostics=True,
                    epsilon=params.epsilon,
                    acc_delta=acc_delta,
                    mesh=mesh,
                    # newborns take the same tracker; `inject` expands a lane's split
                    # events over its entrants and `_event_record` splits them again
                    tracker=static.tracker,
                    max_walk=static.K_walk,
                )
    if partition:
        table, sort_carry = sort.update_table(state, grid, sort_carry)
    else:
        table = sort.build_table(state, grid)
    # Only the cell temperature feeds the a-priori majorant floor (section 7.2.3);
    # the full moments -- pressure tensor and heat flux, an (N, 3, 3) intermediate --
    # are formed once, AFTER the collisions, so the emitted snapshot describes the
    # returned state (and costs nothing when `emit_moments == "none"`).
    temp = sample.cell_temperature(state, table, grid, sp)
    moved = state  # post-move, post-injection, pre-collision: what the collisions were drawn from
    state, cstate, diag, *captured = collide(
        state,
        table,
        cstate,
        grid,
        sp,
        params,
        static,
        temp,
        k,
        base_key,
        score_mode=score_mode,
        capture=capture,
    )
    record = None
    if capture:
        record, (born_lost, record_overflow) = _event_record(
            moved,
            source_details if res is not None else None,
            wall_events,
            captured,
            state,
            static,
            faces is not None
            and not collide_only
            and (BC.DIFFUSE in faces.bcs or mesh is not None),
            mesh,
        )
        if "lane_overflow" in tracking:
            tracking["lane_overflow"] = tracking["lane_overflow"] + born_lost
            tracking["record_overflow"] = record_overflow
    if res is not None:
        diag = dataclasses.replace(
            diag,
            inflow_requested=source_details.requested,
            inflow_escaped=source_details.escaped,
            inflow_sampler_failed=source_details.sampler_failed,
            inflow_candidate_overflow=source_details.candidate_overflow,
            inflow_storage_overflow=source_details.storage_overflow,
            inflow_tracking_exhausted=source_details.tracking_exhausted,
        )
        source_support = jnp.int32(0)
        for axis in range(len(grid.n)):
            for side in (0, 1):
                if faces.of(axis, side) is boundary.BC.INFLOW:
                    mean = inflow.target_count(
                        res, grid, axis, side, sp.mass[0], params.fnum[0], params.dt
                    )
                    source_support += jnp.asarray((mean > 0) & (mean == jnp.floor(mean)), jnp.int32)
        diag = dataclasses.replace(
            diag, count_support_boundary=diag.count_support_boundary + source_support
        )
    diag = dataclasses.replace(
        diag,
        decision_checksum=diag.decision_checksum + bc_chk,
        wall_recrossed=recrossed,
        inflow_injected=injected,
        inflow_shortfall=shortfall,
        **tracking,
    )
    new_carry = (state, cstate) if reward is None else (state, cstate, j_acc + reward(state, table))
    if partition:
        new_carry = (*new_carry, sort_carry)
    if emit_moments == "none":
        # Emit nothing per step.  `scan` stacks its ys over time, so a full
        # CellMoments costs `n_steps * C * 6 * 8` bytes -- 29 GB for 8 steps at
        # C = 1.5e8, which is the regime the multi-device runs live in.  The
        # diagnostics stay: they are per-step SCALARS, and `A_max` resizing needs
        # them.  Snapshots come from `chunk_moments` instead.
        m = None
    else:
        m = sample.moments(state, table, grid, params, sp)
    out = (m, diag) if record is None else (m, diag, record)
    if emit_surface:
        out = (*out, tallies)
    return new_carry, out


def _probe_step(body, step0, carry):
    return body(carry, step0)


def _zeros_like_shape(a):
    return jnp.zeros(a.shape, a.dtype)


def _flatten_blocks(a, n_steps):
    return a.reshape((-1, *a.shape[2:]))[:n_steps]


def _padded_step(carry, offset, body, step0, size, n_steps, padding, block_index):
    """One step of a checkpoint block; steps past ``n_steps`` are no-ops."""
    index = block_index * size + offset
    return lax.cond(
        index < n_steps,
        functools.partial(_live_step, body=body, index=step0 + index),
        functools.partial(_dead_step, padding=padding),
        carry,
    )


def _live_step(carry, body, index):
    return body(carry, index)


def _dead_step(carry, padding):
    return carry, padding


def _block(carry, block_index, body, step0, size, n_steps, padding):
    """A block of ``size`` steps, rematerialised as a unit (module-level, decision 14)."""
    step = functools.partial(
        _padded_step,
        body=body,
        step0=step0,
        size=size,
        n_steps=n_steps,
        padding=padding,
        block_index=block_index,
    )
    return lax.scan(step, carry, jnp.arange(size))


def run(
    state: ParticleState,
    cstate: CollideState,
    grid,
    sp,
    params,
    static,
    base_key,
    bc: BC = BC.PERIODIC,
    collide_only: bool = False,
    emit_moments: bool = True,
    step0: int = 0,
    reward=None,
    faces=None,
    wall=None,
    res=None,
    capacity: int = 0,
    score_spread: str = "particle",
    acc_delta: float = 0.0,
    score_mode: str = "paper",
    capture: bool = False,
    mesh=None,
    emit_surface: bool = False,
):
    """Run ``static.n_steps`` timesteps.

    ``capture=True`` appends a stacked :class:`EventRecord` to the per-step output --
    the post-move/pre-collision state, the newborn faces and sampled velocities, and
    every attempt slot's pair decision -- which is what the ensemble adjoint
    (:mod:`jax_dsmc.ensemble_backward`) consumes.  The forward itself is unchanged.

    ``score_mode="paper"`` keeps terminal-label accumulators. ``"trajectory"``
    stores the same accumulated stochastic correction in every slot, counting each event
    once. Initialize the accumulator to zero or continue with the same mode.
    Consume the latter with ``adjoint.trajectory_surrogate``. This likelihood
    does not differentiate hard cell memberships or changing rounding support.

    Returns:
        ``((state, cstate), (out, diagnostics))`` where the second element is
        stacked over time.  ``out`` is the full :class:`CellMoments` per step when
        ``emit_moments`` (the default), or just ``f64[n_steps, C]`` temperature
        when not -- which skips the ``(N,3,3)`` pressure-tensor intermediate that
        the majorant does not need (section 8.6).  Every ``diagnostics`` field is
        ``[n_steps]``.

    ``jax.checkpoint`` wraps the step body so a reverse pass rematerialises rather
    than storing every intermediate.  It is a no-op in the forward pass, so it
    costs nothing until a gradient is actually taken.

    The RNG contract (section 1) means the step index ``k`` is the *only* thing
    that varies the random stream: keys are recomputed from ``(base_key, k, stage)``
    and never carried, which is what makes the rematerialised backward pass replay
    the same draws the forward pass took.

    ``step0`` offsets that index.  It exists so a long run can be executed in
    chunks -- see :func:`run_adaptive` -- without the chunk boundaries changing a
    single random draw: chunking ``[0, 200)`` into two halves consumes exactly the
    stream that one 200-step call would have.

    ``mesh`` embeds a surface (:func:`jax_dsmc.mesh.cut.build_mesh`); it needs ``faces``
    and a ``grid`` carrying the mesh's flow volumes (:func:`jax_dsmc.mesh.cut.with_volume`).
    ``static.tracker`` selects the stencil, lane or cell-walk tracker.  ``emit_surface``
    appends the step's :class:`jax_dsmc.surface.SurfaceTallies` (rows ``2D`` box faces
    then the elements; extensive momentum / energy / heat of the resident particles'
    hits -- entrants' birth-step hits are not tallied) as the last element of the
    per-step output.
    """
    if res is not None and (faces is None or collide_only or capacity < 1):
        raise ValueError("Sources require faces, transport, and positive candidate capacity")
    if mesh is not None and (faces is None or collide_only):
        raise ValueError("An embedded surface requires per-face rules and transport")
    # Concrete public calls reject unsupported physics immediately. Traced calls
    # must be constructed through a validated setup or checked_run.
    if not any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves((sp, params))):
        validate_inputs(sp, params, static)
    if score_mode not in ("paper", "trajectory"):
        raise ValueError("score_mode must be paper or trajectory")
    if score_mode == "trajectory" and score_spread != "particle":
        raise ValueError("Trajectory scoring requires each boundary event exactly once")
    body = functools.partial(
        _step,
        grid=grid,
        sp=sp,
        params=params,
        static=static,
        base_key=base_key,
        bc=bc,
        collide_only=collide_only,
        emit_moments=emit_moments,
        reward=reward,
        score_spread=score_spread,
        acc_delta=acc_delta,
        score_mode=score_mode,
        capture=capture,
        faces=faces,
        wall=wall,
        res=res,
        capacity=capacity,
        mesh=mesh,
        emit_surface=emit_surface,
    )
    policy = getattr(jax.checkpoint_policies, static.remat_policy, None)
    body = jax.checkpoint(body, policy=policy, static_argnums=())
    init = (state, cstate) if reward is None else (state, cstate, jnp.float64(0.0))
    partition = sort.partition_active(static)
    if partition:
        # the carried cell order lives inside the scan only: a run starts from the full
        # sort of its initial state and returns the carry it was given (structure too)
        init = (*init, sort.initial_carry(state, grid))
    if static.checkpoint_block_size == 0:
        final, outputs = lax.scan(body, init, step0 + jnp.arange(static.n_steps))
        return (final[:-1] if partition else final), outputs
    size = static.checkpoint_block_size
    blocks = (static.n_steps + size - 1) // size
    output_shape = jax.eval_shape(functools.partial(_probe_step, body, step0), init)[1]
    padding = jax.tree.map(_zeros_like_shape, output_shape)
    block = functools.partial(
        _block, body=body, step0=step0, size=size, n_steps=static.n_steps, padding=padding
    )
    final, outputs = lax.scan(jax.checkpoint(block), init, jnp.arange(blocks))
    outputs = jax.tree.map(functools.partial(_flatten_blocks, n_steps=static.n_steps), outputs)
    return (final[:-1] if partition else final), outputs


def run_adaptive(  # noqa: PLR0915 -- the resize/retry loop reads best as one unit
    state: ParticleState,
    cstate: CollideState,
    grid,
    sp,
    params,
    static,
    base_key,
    bc: BC = BC.PERIODIC,
    collide_only: bool = False,
    emit_moments: bool | str = True,
    chunk: int = 32,
    growth: float = 1.4,
    max_resizes: int = 8,
    verbose: bool = False,
    sample_stride: int = 1,
    chunk_moments: bool = False,
):
    """Run ``static.n_steps`` steps, growing ``A_max`` whenever the flow outgrows it.

    **Why this exists.** ``A_max`` is the trip count of the attempt-slot ``scan``,
    so it is baked into the compiled program and cannot change while that program
    runs.  But the *right* value is a property of the flow: the attempt count goes
    as ``npc^2`` at fixed cell volume, so compression drives it, and for a shape
    being optimised nobody can know it in advance.  Guessing it is not a strategy;
    the failure mode is silent under-collision (section 8.2, gap U4).

    **What makes resizing cheap.** The simulation state is just arrays, so
    continuing after a recompile costs *nothing but the compile* -- there is no
    restart, no re-initialisation, no lost physics.  So the loop runs in chunks of
    ``chunk`` steps, reads ``attempts_clipped`` on the host, and if the capacity was
    exceeded it grows ``A_max`` and **re-runs that chunk from the state it started
    from**.  Only the offending chunk is redone, never the whole run.  Each distinct
    ``A_max`` compiles once and is then reused (and the on-disk compilation cache
    carries it across processes), so a run settles onto its working capacity after
    a couple of resizes and pays nothing thereafter.

    ``step0`` keeps the random stream identical to an unchunked run (every draw is keyed by
    the absolute step index), so chunking does not change the results.

    **Keeping the output affordable.**  ``scan`` stacks its per-step output, so a
    long run at a large cell count cannot emit full moments every step.  Two knobs
    control that, and at scale both are needed:

    ``sample_stride``
        keep only every ``stride``-th step of each chunk.  Reduces what is
        *retained*, not what is computed, so it does not help the peak inside a
        chunk -- shrink ``chunk`` for that.
    ``chunk_moments``
        with ``emit_moments="none"``, compute one :class:`CellMoments` snapshot from
        the state at the end of each chunk.  Peak output memory then depends on the
        cell count alone, not on the step count, which is what makes a
        billion-particle run's field output tractable.  Set ``chunk`` to the frame
        interval.

    Returns:
        ``((state, cstate), (out, diag), history)`` where ``history`` is the list of
        ``(step, A_max, clipped, peak_attempts)`` resize events.

    Not for use inside a differentiated run: the host-side branch is Python control
    flow.  Calibrate with this, then take gradients through :func:`run` at the
    settled ``A_max`` -- which is the natural split anyway, since an optimiser's
    outer loop can resize between iterations.
    """
    total = static.n_steps
    a_max = static.A_max
    done = 0
    outs, diags, history = [], [], []
    resizes = 0

    while done < total:
        n = min(chunk, total - done)
        st = dataclasses.replace(static, A_max=a_max, n_steps=n)
        saved = (state, cstate)
        (state, cstate), (out, diag) = run(
            state,
            cstate,
            grid,
            sp,
            params,
            st,
            base_key,
            bc=bc,
            collide_only=collide_only,
            emit_moments=emit_moments,
            step0=done,
        )
        clipped = int(jnp.sum(diag.attempts_clipped))
        if clipped > 0:
            peak = int(jnp.max(diag.max_attempts))
            new_a = max(a_max + 1, math.ceil(growth * peak))
            history.append((done, a_max, clipped, peak))
            if verbose:
                print(
                    f"  step {done}: A_max={a_max:,} clipped {clipped:,} "
                    f"(peak wanted {peak:,}) -> {new_a:,}"
                )
            resizes += 1
            if resizes > max_resizes:
                raise RuntimeError(
                    f"A_max failed to converge after {max_resizes} resizes "
                    f"(now {new_a:,}); the flow is outgrowing capacity faster than "
                    "the growth factor can follow."
                )
            a_max = new_a
            state, cstate = saved  # redo THIS chunk only, from where it began
            continue

        require_valid(diag)
        if chunk_moments:
            table = sort.build_table(state, grid)
            out = jax.tree.map(
                lambda a: a[None, ...], sample.moments(state, table, grid, params, sp)
            )
        elif sample_stride > 1:
            out = jax.tree.map(lambda a: a[sample_stride - 1 :: sample_stride], out)
        if out is not None:
            outs.append(out)
        diags.append(diag)
        done += n

    if not outs:
        cat = None
    elif len(outs) > 1:
        cat = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *outs)
    else:
        cat = outs[0]
    dcat = (
        jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *diags)
        if len(diags) > 1
        else diags[0]
    )
    return (state, cstate), (cat, dcat), history


def checked_run(state, cstate, grid, sp, params, static, base_key, **kwargs):
    """Host wrapper: validate configuration and all returned trajectory diagnostics."""

    validate_inputs(sp, params, static)
    result = run(state, cstate, grid, sp, params, static, base_key, **kwargs)
    require_valid(result[1][1])
    return result
