"""Shape adjoint: the VJP of the replayed wall map with respect to the geometry.

The ensemble adjoint (:mod:`jax_dsmc.ensemble_device`) carries the value ``lambda`` on the
particles and never linearises the geometry.  The geometry enters the forward only
through the move (:func:`jax_dsmc.boundary.apply_faces` -> :func:`jax_dsmc.wall_adjoint.advance`):
hit times, hit points, specular reflections and the normals of diffuse re-emissions.
Under the RNG contract that move is a pure function of the pre-move state, the step's
key and the geometry, so it can be **replayed** from a captured
:class:`jax_dsmc.simulate.EventRecord` and differentiated by ``jax.vjp`` with the
cotangents the ensemble backward provides (plan section 2, item 4):

* ``grad_x lambda``, ``grad_v lambda`` at every survivor's own post-move state (from the
  per-cell fit, :func:`jax_dsmc.ensemble_device.node_gradients`) on the replayed
  post-move state -- the pathwise term of hit times and specular reflections;
* ``lambda + r`` (the value after the move plus the step's hit rewards) on the move's
  log-probability (the epsilon hit-timing and accommodation branches);
* the objective's own weights on the step's surface tallies -- the pathwise part of a
  per-hit reward such as drag;
* ``(lambda + r_after) d/dn log q(v_out | n)`` on the normal of every diffuse-branch
  element hit (``r_after``: the rewards of the step's later hits; the hit's own reward
  is exact in ``v_out`` and enters pathwise through the tally of the traced sample --
  the reparameterised form with bounded weights, review section 7.8 -- when a tally
  cotangent is given, and joins the score weight otherwise);
* optionally the **hit-timing band term** on the event times: a reward that jumps at a
  hit (a velocity-dependent per-step reward; a per-hit reward at the window's edges)
  makes the expected objective depend on which step a hit falls in, a dependence the
  pathwise derivative of a fixed hit set cannot see.  Its density is estimated from the
  hits within a band ``Delta dt`` of the step end (:func:`timing_cotangent`), the
  value-preserving analogue of the epsilon hit-timing score with no forward
  regularisation (gate G3: the pathwise term alone read ``+0.82 +- 0.23`` for the
  plane's windowed ``v_y^2`` against an FD of ``+0.05 +- 0.15``; the band term is
  ``-0.43 +- 0.02``). -- the kernel score of the re-emission law's dependence on the normal
  (:func:`jax_dsmc.sampling.log_diffuse_density`; the sampled velocity itself is
  ``stop_gradient`` in :func:`jax_dsmc.wall_adjoint._element_events`, review section 5.9).

The geometry is the body's traced pytree: ``AnalyticBody`` (centre, radius) or the
``points`` of a ``Surface``; :func:`with_geometry` rebuilds the metric tables (normals,
areas) from it.  Cut volumes are not re-derived by the replay; their dependence -- the
NTC attempt count carries ``1 / V_c`` -- is the separate **cut-volume term**
(:func:`volume_weights`, :func:`volume_cotangent`: ``-sum_k sum_c (dV_c / V_c) C_kc``
with ``C_kc`` the expected change of the objective by cell ``c``'s collisions at step
``k``, from the virtual pull-back's collision increments).  The cell tables are fixed by
design (no membership term).  What the VJP cannot see -- the element-assignment jumps of a flat-normal
polygon and the silhouette of a diffuse body -- is the edge-band score of plan
section 2, item 5: the polygon's is not built; the level set's silhouette band term is
:func:`silhouette_cotangent` (``gradient(silhouette_band=)``, 2026-09-21).

Everything here is module-level and ``lax.scan``-driven (decision 14); the per-step
replay is validated against the record (``replay_error``, roundoff at most).
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jax_dsmc import boundary, ensemble_device, sampling, step_axis
from jax_dsmc.collide.ntc import step_v_post
from jax_dsmc.ensemble_device import node_gradients
from jax_dsmc.grid import collision_cell, spacing
from jax_dsmc.mesh import geometry as mesh_geometry
from jax_dsmc.mesh import levelset as levelset_mod
from jax_dsmc.mesh import surface as surface_mod
from jax_dsmc.mesh.types import Mesh
from jax_dsmc.rng import STAGE_MOVE, key_for
from jax_dsmc.sampling import log_diffuse_density
from jax_dsmc.wall_adjoint import RecordEvents, SplitEvents, as_split, lane_slots, particle_sum

jax.config.update("jax_enable_x64", True)


class EventArray(NamedTuple):
    """A per-event quantity of a lane-tracked step, aligned with
    :class:`jax_dsmc.wall_adjoint.SplitEvents`: ``head`` ``f64[1, N]`` (pass 0, every
    slot), ``tail`` ``f64[K - 1, B]`` (the lane's later passes, entry ``j`` belonging to
    the events' ``slot[j]``).  Stacked over steps the leading axis is ``M``."""

    head: jax.Array
    tail: jax.Array


class HitRewards(NamedTuple):
    """Per-hit rewards of a lane-tracked replay (stacked over the ``M`` steps): the
    per-event ``values`` (:class:`EventArray`) and the lane's ``slot`` (``i32[M, B]``).
    :func:`gradient` reads ``values``, :func:`jax_dsmc.ensemble_backward.from_device` both;
    the device backward needs only each slot's per-step total, :meth:`totals` -- formed on
    demand rather than stored next to the values (the replay's outputs stay alive through
    both backwards and the shape VJP)."""

    values: object
    slot: jax.Array

    def totals(self):
        """``f64[M, N]``: each slot's summed rewards per step."""
        return jax.vmap(particle_sum)(self.values.head, self.values.tail, self.slot)


def _tail_of(values, slot):
    """Per-slot ``values`` (``N, ...``) at the lane's entries (zero / False where unused)."""
    got = values[jnp.maximum(slot, 0)]
    valid = (slot >= 0).reshape(slot.shape + (1,) * (got.ndim - 1))
    return jnp.where(valid, got, jnp.zeros_like(got))


def _as_event_array(values, events):
    """``values`` (a dense ``K, N`` array or already an :class:`EventArray`) split like
    ``events``; ``None`` -> zeros."""
    split = as_split(events)
    if values is None:
        return EventArray(jnp.zeros(split.head.face.shape), jnp.zeros(split.tail.face.shape))
    if isinstance(values, HitRewards):
        return values.values
    if isinstance(values, EventArray):
        return values
    return EventArray(values[:1], values[1:])


def _per_part(fn, events, *args):
    """``fn(part_events, *args)`` on the head and the tail of split ``events``."""
    split = as_split(events)
    return EventArray(fn(split.head, *args), fn(split.tail, *args))


def _weighted(fn, setup):
    """The reward function ``fn`` with the objective's traced weights bound
    (``setup["reward_weight"]``: ``f64[D]``, ``dJ/d`` each component of the momentum the body
    receives -- a ratio objective's chain rule at this seed's forces, e.g. lift / drag), or
    ``fn`` itself without them.  Traced, so new weights per seed do not recompile."""
    weight = setup.get("reward_weight")
    if fn is None or weight is None:
        return fn
    return functools.partial(fn, weight=weight)


def _masked(values, survivors, slot):
    """An :class:`EventArray` with the non-survivors' events zeroed."""
    return EventArray(
        jnp.where(survivors[None, :], values.head, 0.0),
        jnp.where(_tail_of(survivors, slot)[None, :], values.tail, 0.0),
    )


def _cotangent_field(events, field, values):
    """``zeros``-structured cotangent for ``events`` with ``field`` set from the
    :class:`EventArray` ``values`` (dense events: head and tail rows stacked back)."""
    zeros = _zero_cotangent(events)
    if isinstance(events, SplitEvents):
        return zeros._replace(
            head=zeros.head._replace(**{field: values.head}),
            tail=zeros.tail._replace(**{field: values.tail}),
        )
    return zeros._replace(**{field: jnp.concatenate((values.head, values.tail))})


class Replay(NamedTuple):
    hit_reward: object
    """Per-hit rewards of the objective (zeros without ``reward``): ``f64[M, K, N]`` for
    the dense trackers, :class:`HitRewards` for ``tracker="lane"`` -- either is what
    :func:`gradient` and the device backward take."""
    tallies: object
    """Per-step :class:`SurfaceTallies` of the replayed moves."""
    replay_error: jax.Array
    """As :class:`ShapeTerms`."""
    surface_hits: jax.Array
    """``i32[M]``."""
    timing_reward: jax.Array
    """The hit-timing band term per hit (``timing_cotangent`` of the objective's
    ``timing_jump``), laid out as ``hit_reward``: the change of the objective per unit
    delay of the particle's trajectory, what the along-flight identity's second backward
    carries."""
    patch: jax.Array | None = None
    """``f64[P]``: the survivors' hits summed onto the nodes of ``patch_tally`` over the
    replayed steps (``None`` without one): e.g. the surface heat flux on a node lattice."""

    @property
    def hit_total(self):
        """``f64[M, N]`` each slot's summed hit rewards per step -- all the device backward's
        move pull-back uses (formed on demand)."""
        return _totals(self.hit_reward)

    @property
    def timing_total(self):
        """``f64[M, N]`` as :attr:`hit_total` for the timing rewards."""
        return _totals(self.timing_reward)


def _totals(rewards):
    """Per-slot per-step totals of a replay's rewards (dense ``M, K, N`` or lane)."""
    if isinstance(rewards, HitRewards):
        return rewards.totals()
    return jnp.sum(rewards, axis=1)


class ShapeTerms(NamedTuple):
    pathwise: object
    """Geometry-shaped: ``grad lambda`` on the replayed post-move state and the
    objective's weights on the tallies (the sum of the three parts below)."""
    pathwise_x: object
    """Geometry-shaped: the ``grad_x lambda`` part alone (hit times and hit points)."""
    pathwise_v: object
    """Geometry-shaped: the ``grad_v lambda`` part alone (reflected velocities)."""
    pathwise_tally: object
    """Geometry-shaped: the objective's tally weights alone (a per-hit reward's own
    dependence on the geometry through the reflected velocity)."""
    score: object
    """Geometry-shaped: ``lambda + r`` on the move's log-probability and the normal
    score of the diffuse-branch element hits."""
    timing: object
    """Geometry-shaped: the hit-timing band term (:func:`timing_cotangent`); zero unless
    ``gradient`` was given ``timing_jump``."""
    silhouette: object
    """Geometry-shaped: the silhouette band term (:func:`silhouette_cotangent`): the value
    difference between a hit and a miss for the particles whose undeflected flight passes
    within ``silhouette_band`` of the surface, times the rate at which the surface's
    motion converts one into the other.  Zero unless ``gradient`` was given a band."""
    replay_error: jax.Array
    """``f64`` largest relative ``|x_post - record|`` (per box extent) / ``|v_post - record|``
    (per the record's largest speed) over survivors, all steps.  The replay is the same
    arithmetic as the forward under the RNG contract but compiled in another context, so
    a few ulp is roundoff (amplified to ~1e-12 by grazing reflections on concave walls);
    ``1`` means a discrete outcome differs (:func:`_mismatch`).  The mesh runners reject
    anything above ``REPLAY_TOL = 1e-9``."""
    surface_hits: jax.Array
    """``i32[M]`` element hits per replayed step."""
    order: object = None
    """Geometry-shaped: the collision-order term (:func:`order_jump`): a body hit that slips
    from the end of step ``k`` into step ``k + 1`` collides in step ``k`` as the incoming
    particle instead of as the reflected one.  Zero without collisions; ``None`` unless
    ``gradient`` was given the collision increments."""


def geometry(mesh: Mesh | None):
    """The traced geometry of a mesh: the analytic body itself, the surface points or the
    level-set coefficients (``None`` without a mesh: the box-only replay has no geometry
    control)."""
    if mesh is None:
        return None
    if mesh.kind == "surface":
        return mesh.body.points
    if mesh.kind == "levelset":
        return mesh.body.coef
    return mesh.body


def with_geometry(mesh: Mesh, geom) -> Mesh:
    """``mesh`` with its body and metric tables (normals, areas) taken from ``geom``
    (cell tables and cut volumes unchanged -- see the module docstring)."""
    if mesh.kind == "surface":
        body = dataclasses.replace(mesh.body, points=geom)
        return dataclasses.replace(
            mesh,
            body=body,
            normals=surface_mod.normals(geom, body.elements),
            areas=surface_mod.areas(geom, body.elements),
        )
    if mesh.kind == "levelset":
        return dataclasses.replace(mesh, body=dataclasses.replace(mesh.body, coef=geom))
    return dataclasses.replace(mesh, body=geom)


def wall_map(
    geom,
    x,
    v,
    alive,
    key,
    grid,
    faces,
    wall,
    mass,
    dt,
    mesh,
    epsilon=0.0,
    acc_delta=0.0,
    max_bounces=4,
    tracker="stencil",
    max_walk=8,
    lane_size=0,
):
    """The step's move as a function of the geometry: ``(x_post, v_post, alive_post,
    logp, tallies, events)`` of :func:`jax_dsmc.boundary.apply_faces` with the traced
    events (hit points, incident velocities, normals) filled."""
    mesh_g = None if mesh is None else with_geometry(mesh, geom)
    x_post, v_post, alive_post, _, logp, events, tallies = boundary.apply_faces(
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
        return_events=True,
        mesh=mesh_g,
        tracker=tracker,
        max_walk=max_walk,
        return_tallies=True,
        trace=True,
        lane_size=lane_size,
    )
    return x_post, v_post, alive_post, logp, tallies, events


def _normals_step(carry, x, grid, mesh, diffuse_faces, within):
    del carry
    return None, mesh_geometry.nearest_normal(x, grid, mesh, diffuse_faces, within)


@functools.partial(jax.jit, static_argnames=("faces", "within"))
def local_normals(records, grid, mesh, faces, within=1.0):
    """``f64[M, N, 3]``: the nearest surface normal (elements and DIFFUSE box faces within
    ``within`` cell widths, else zero) at every slot's post-move position of every captured
    step -- the ``normals`` argument that makes the backward's fits one-sided."""
    diffuse = tuple(f for f in range(2 * len(grid.n)) if faces.bcs[f] is boundary.BC.DIFFUSE)
    body = functools.partial(
        _normals_step, grid=grid, mesh=mesh, diffuse_faces=diffuse, within=within
    )
    _, out = lax.scan(body, None, records.x)
    return out


def _zero_cotangent(primal):
    """Zeros of the right dtype for every leaf (``float0`` for integer / boolean leaves)."""
    return jax.tree.map(
        lambda p: (
            jnp.zeros(p.shape, jax.dtypes.float0)
            if not jnp.issubdtype(p.dtype, jnp.inexact)
            else jnp.zeros_like(p)
        ),
        primal,
    )


def _hit_rows(events, mesh, dim):
    """``WallParams`` row of every recorded hit (``K, N``): the face id for box faces,
    ``2D + model`` for elements; 0 where there is no hit."""
    face = events.face
    e = jnp.clip(face - 2 * dim, 0, mesh.n_elements - 1)
    model = mesh.body.model[e] if mesh.kind == "surface" else jnp.zeros_like(e)
    return jnp.where(face >= 2 * dim, 2 * dim + model, jnp.maximum(face, 0))


def _log_density_rows(v_out, normal, temp, vwall, floor, mass):
    return jnp.sum(log_diffuse_density(v_out, normal, temp, mass, vwall, floor))


def normal_score(events, weight, wall, mesh, mass, dim):
    """Cotangent on ``events.normal`` (``K, N, 3``): ``weight[k, i] * grad_n log q(v_out | n)``
    at every diffuse-branch ELEMENT hit (box-face normals carry no geometry).  The body's
    ``grazing`` cut-off ``eps`` (level set / analytic body; ``0`` for a polygon) floors the
    score's ``1 / (v_out . n)`` at ``eps |v_out|`` -- the emission-side twin of the
    hit-time regularisation (:func:`jax_dsmc.mesh.levelset.regularised_quotient`)."""
    rows = _hit_rows(events, mesh, dim)
    temp_rows = wall.temp if wall.temp.ndim == 1 else wall.temp[:, 0]
    temp = temp_rows[rows]
    vwall = wall.vwall[rows]
    k, n = rows.shape
    floor = getattr(mesh.body, "grazing", 0.0) * jnp.linalg.norm(events.v_out, axis=-1)
    flat = jax.vmap(jax.grad(_log_density_rows, argnums=1), in_axes=(0, 0, 0, 0, 0, None))
    grad_n = flat(
        events.v_out.reshape(k * n, 1, 3),
        events.normal.reshape(k * n, 1, 3),
        temp.reshape(k * n, 1),
        vwall.reshape(k * n, 1, 3),
        floor.reshape(k * n, 1),
        mass,
    ).reshape(k, n, 3)
    scored = (events.face >= 2 * dim) & events.diffuse
    return jnp.where(scored[..., None], weight[..., None] * grad_n, 0.0)


def timing_cotangent(events, jump, dt, band):
    """Cotangent on ``events.time`` (``K, N``) of the hit-timing band term.

    ``jump`` (``K, N``) is the change of the objective when the hit of pass ``k`` moves from
    the end of this step to the start of the next (the per-step reward's ``phi(v_out) -
    phi(v_in)``; a per-hit reward's ``+r`` at the window's last step and ``-r`` on the step
    before the window; zero elsewhere).  Hits within ``band * dt`` of the step end estimate
    the density of hit times at the boundary: moving the surface so a hit crosses it
    loses ``jump``, hence ``-jump / (band dt)`` per band hit on that hit's time.
    """
    inside = (events.face >= 0) & (events.time > dt * (1.0 - band))
    return jnp.where(inside, -jump / (band * dt), 0.0)


def _last_normal(events):
    """Normal of each slot's last recorded hit of the step (``f64[N, 3]``; zero rows for
    slots without one); dense or split events."""
    split = as_split(events)
    head = split.head
    n = head.face.shape[1]
    out = jnp.where((head.face[0] >= 0)[:, None], head.normal[0], 0.0)
    put = jnp.where(split.slot >= 0, split.slot, n)
    tail = split.tail
    for k in range(tail.face.shape[0]):
        hit = tail.face[k] >= 0
        cur = out[jnp.maximum(split.slot, 0)]
        out = out.at[put].set(jnp.where(hit[:, None], tail.normal[k], cur), mode="drop")
    return out


class ToGo(NamedTuple):
    """Reward-to-go of every event (:func:`_to_go`): ``pre`` / ``post`` are
    :class:`EventArray` -- ``lam`` plus the rewards of this event and every later one of
    the slot's step (``pre``, the value from this event's draw onward), or of the later
    ones only (``post``); ``start`` ``f64[N]`` is the value at the pre-move state."""

    pre: EventArray
    post: EventArray
    start: jax.Array


def _to_go(lam, rewards, slot):
    """:class:`ToGo` of the per-event ``rewards`` (an :class:`EventArray`) for the
    post-move values ``lam`` (``f64[N]``) -- the suffix sums of each slot's rewards in
    pass order, pass 0 in the head, the later passes in the lane's tail."""
    n = lam.shape[0]
    suffix = jnp.cumsum(rewards.tail[::-1], axis=0)[::-1]
    tail_total = jnp.sum(rewards.tail, axis=0)
    later = jnp.zeros(n).at[jnp.where(slot >= 0, slot, n)].add(tail_total, mode="drop")
    head_post = lam + later
    head_pre = head_post + rewards.head[0]
    tail_pre = _tail_of(lam, slot)[None, :] + suffix
    return ToGo(
        EventArray(head_pre[None], tail_pre),
        EventArray(head_post[None], tail_pre - rewards.tail),
        head_pre,
    )


def _step_rewards(events, tallies, rec_k, step, survivors, reward, timing_jump, setup, band):
    """Per-event hit rewards and timing rewards of one replayed step (:class:`EventArray`,
    non-survivors zeroed) and its survivors' body hits."""
    split = as_split(events)
    dim = rec_k.x.shape[1]
    r = _as_event_array(None, events)
    reward, timing_jump = _weighted(reward, setup), _weighted(timing_jump, setup)
    if reward is not None:
        r = _masked(_per_part(reward, events, tallies, rec_k, step), survivors, split.slot)
    tau = _as_event_array(None, events)
    if timing_jump is not None:
        jump = _masked(_per_part(timing_jump, events, rec_k, step), survivors, split.slot)
        tau = EventArray(
            timing_cotangent(split.head, jump.head, setup["map"]["dt"], band),
            timing_cotangent(split.tail, jump.tail, setup["map"]["dt"], band),
        )
    body = _masked(
        EventArray(
            (split.head.face >= 2 * dim).astype(jnp.float64),
            (split.tail.face >= 2 * dim).astype(jnp.float64),
        ),
        survivors,
        split.slot,
    )
    n_hits = (jnp.sum(body.head) + jnp.sum(body.tail)).astype(jnp.int32)
    return r, tau, n_hits


def _patch_totals(events, survivors, step, patch_tally):
    """``f64[P]``: ``patch_tally(part_events, step) -> (index, value)`` (``i32`` / ``f64`` of
    the events' shape plus a trailing corner axis) scattered over its ``n_nodes`` nodes, the
    non-survivors' events left out."""
    split = as_split(events)
    total = jnp.zeros(patch_tally.keywords["n_nodes"])
    parts = ((split.head, survivors[None, :]), (split.tail, _tail_of(survivors, split.slot)[None, :]))
    for part, keep in parts:
        index, value = patch_tally(part, step)
        value = jnp.where(keep[..., None], value, 0.0)
        total = total.at[index.reshape(-1)].add(value.reshape(-1))
    return total


def _replay_step(carry, xs, geom, reward, timing_jump, timing_band, setup, patch_tally=None):
    """Forward-only traced replay of one step: hit rewards, timing rewards, replay check
    (and the step's ``patch_tally`` totals, :func:`_patch_totals`)."""
    x, v, alive, err = carry
    rec_k, step = xs
    key = key_for(setup["base_key"], step, STAGE_MOVE)
    x_post, v_post, alive_post, _logp, tallies, events = wall_map(
        geom, x, v, alive, key, **setup["map"]
    )
    survivors = alive & rec_k.alive & (rec_k.born_face < 0)
    err = jnp.maximum(
        err, _mismatch(x_post, v_post, alive, alive_post, rec_k, survivors, setup["map"]["grid"])
    )
    r, tau, n_hits = _step_rewards(
        events, tallies, rec_k, step, survivors, reward, timing_jump, setup, timing_band
    )
    slot = as_split(events).slot
    patch = jnp.zeros(0) if patch_tally is None else _patch_totals(
        events, survivors, step, patch_tally
    )
    if isinstance(events, SplitEvents):
        r = HitRewards(r, slot)
        tau = HitRewards(tau, slot)
    else:
        # dense trackers: the (K, N) layout the callers have always had
        r = jnp.concatenate((r.head, r.tail))
        tau = jnp.concatenate((tau.head, tau.tail))
    return (rec_k.x, step_v_post(rec_k), rec_k.alive, err), (r, tallies, n_hits, tau, patch)


def _mismatch(x_post, v_post, alive, alive_post, rec_k, survivors, grid):
    """The replay's disagreement with the record: ``1`` if any particle's discrete outcome
    differs (alive before the move, and alive after it in the replay but not in the record
    or the reverse -- an outflow, a deletion, a missed hit; a slot the inflow refilled is
    the record's newborn), else the largest relative error over survivors: positions per
    box extent, velocities per the record's largest speed component.

    The two parts are checked differently on purpose.  A discrete divergence (a different
    draw, a hit / miss flip) moves a particle by a sizeable fraction of its step; the
    continuous part is round-off of one arithmetic compiled twice, which grazing
    reflections on a concave wall amplify: each changes the hit point by ``delta / theta``
    along a curved wall, so four at 0.4 degrees turn one ulp into ~1e-12
    (``scratch/mesh/probe_amplification.py``, 2026-09-24; the optc runs of optimised shapes
    reported 1.8e-12 .. 7.6e-12).  Callers hold the continuous part to a tolerance far above
    that and far below any discrete divergence (``REPLAY_TOL = 1e-9``)."""
    expected = rec_k.alive & (rec_k.born_face < 0)
    flipped = jnp.any(alive & (alive_post != expected))
    dx = jnp.max(
        jnp.where(survivors[:, None], jnp.abs(x_post - rec_k.x) / (grid.hi - grid.lo), 0.0)
    )
    v_scale = jnp.maximum(jnp.max(jnp.abs(jnp.where(survivors[:, None], rec_k.v, 0.0))), 1e-300)
    dv = jnp.max(jnp.where(survivors[:, None], jnp.abs(v_post - rec_k.v), 0.0)) / v_scale
    return jnp.where(flipped, 1.0, jnp.maximum(dx, dv))


def along_flight(gx, v, lam_timing):
    """Replace the component of the fitted ``grad_x lambda`` along the flight by the
    along-flight identity: a particle at ``x - s v`` is the particle at ``x`` delayed by
    ``s``, so ``v . grad_x V = -T`` with ``T`` the expected change of the objective per
    unit delay of the trajectory (``lam_timing``: the value of the timing rewards, carried
    by a second backward).  ``gx`` (``N, D``), ``v`` (``N, 3``): the perpendicular part
    stays fitted.

    In a collisional gas the backward equation ``d_t V + v . grad_x V + L* V + r = 0``
    gives ``v . grad_x V = -T - L* V``: the extra flight time ``s`` is spent where the
    particle collides at the rate of its state.  The caller adds ``L* V`` to
    ``lam_timing`` (:func:`gradient_variants` ``collision=``: the virtual pull-back's
    per-slot increment ``dt L* lam``, :attr:`ensemble_device.Backward.collision`, over
    ``dt``); without it the identity holds only in free flight."""
    d = gx.shape[1]
    vd = v[:, :d]
    v2 = jnp.maximum(jnp.sum(vd * vd, axis=1), 1e-300)
    parallel = jnp.sum(gx * vd, axis=1) / v2
    return gx - parallel[:, None] * vd - (lam_timing / v2)[:, None] * vd


def _virtual_hit(x_star, n_star, v, key, wall, mass, dim):
    """The re-emitted velocity of a virtual hit at ``x_star`` with normal ``n_star``
    (``N, D``) under the body's wall model (row ``2D``): specular, or diffuse with the
    accommodation probability, from fresh draws.  ``f64[N, 3]``."""
    n = x_star.shape[0]
    normal = jnp.zeros((n, 3)).at[:, :dim].set(n_star)
    vn = jnp.sum(v * normal, axis=1, keepdims=True)
    v_spec = v - 2.0 * vn * normal
    temp_rows = wall.temp if wall.temp.ndim == 1 else wall.temp[:, 0]
    temp = temp_rows[2 * dim]
    acc = wall.acc[2 * dim]
    vwall = wall.vwall[2 * dim]
    k_draw, k_acc = jax.random.split(key)
    z, r = sampling.standard_diffuse_draws(k_draw, n)
    u = jax.random.uniform(k_acc, (n,), dtype=jnp.float64)
    fresh = sampling.diffuse_velocity_general(
        z, r, normal, jnp.full((n, 1), temp), mass, jnp.broadcast_to(vwall, (n, 3)), v
    )
    return jnp.where((u <= acc)[:, None], fresh, v_spec)


def silhouette_cotangent(
    ls, x, v, survivors, events, to_go, rec_k, lam_k, step, key, setup, fit, band, reward
):
    """The silhouette band term of one step as a cotangent ``c_i`` on ``phi`` at the
    closest-approach point ``x*_i`` of every survivor's *undeflected* flight, and the
    points: ``(c f64[N], x_star f64[N, D])``.

    A particle whose undeflected flight passes within ``band`` of the surface
    (``|phi_min| < band``; ``phi_min < 0`` = it hit on its first pass with that
    penetration depth, ``> 0`` = it missed by that distance) changes outcome when the
    surface moves by more than ``phi_min`` there: a miss becomes a hit, a hit a miss.
    ``phi_min`` is taken along the undeflected *line* within the search reach (a hit's
    fragility is its whole chord's depth) and every passing is counted once: a miss in
    the step whose move contains the closest-approach point (strictly inside -- a
    particle starting there has just left the surface), a hit in the step it hits.  The
    change of the objective is ``V_hit - V_miss``, the rate ``-d phi_min / d theta`` times
    the density of ``phi_min`` at zero, which the band's particles on *both* sides
    estimate (a window of width ``2 band``; the miss side alone describes an outward
    move, the hit side alone an inward one, and the derivative is the same from either
    side -- value-preserving, like :func:`timing_cotangent`); ``d phi_min / d coef_j =
    B_j(x*)`` by the envelope theorem, so the term is the VJP of ``phi(x*)`` with
    cotangent ``c = -(V_hit - V_miss) / (2 band)``.  ``V`` of the realised outcome is the particle's own
    reward-to-go; the virtual outcome's value is the fitted adjoint at the virtual
    post-move state (:func:`ensemble_device.value_query`) plus the virtual hit's reward
    (``reward``, the objective's per-hit reward).  For a specular wall ``V_hit -> V_miss``
    at grazing incidence and the term vanishes; for a diffuse wall it is the O(1) term of
    the silhouette (a grazing hit re-emits thermally).
    """
    dim = x.shape[1]
    grid, wall, mass, dt = setup["grid"], setup["wall"], setup["mass"], setup["dt"]
    n = x.shape[0]
    frozen = jax.lax.stop_gradient(ls)
    # the closest approach of the undeflected LINE within the search reach (not just this
    # step's segment): a hit's fragility is the depth of its whole chord
    phi_min, s_min = levelset_mod.closest_approach(
        jax.lax.stop_gradient(x), jax.lax.stop_gradient(v), frozen, None
    )
    vd = v[:, :dim]
    speed = jnp.maximum(jnp.linalg.norm(vd, axis=1), 1e-300)
    vhat = vd / speed[:, None]
    length = speed * dt
    x_star = jax.lax.stop_gradient(x + s_min[:, None] * vhat)
    n_star = levelset_mod.normal(frozen, x_star)
    split = as_split(events)
    hit_first = split.head.face[0] >= 2 * dim
    hit_later = (
        particle_sum(
            jnp.zeros((1, n)), (split.tail.face >= 2 * dim).astype(jnp.float64), split.slot
        )
        > 0
    )
    # one passing per trajectory: a miss counts in the step whose move contains the
    # closest-approach point (strictly inside: a particle that starts at its closest
    # approach is one that just left the surface, and one still approaching at the end
    # of the step is counted when it gets there); a hit counts in the step it hits
    interior = (s_min > 1e-3 * ls.reach) & (s_min < length)
    passing = jnp.where(hit_first, s_min > 1e-3 * ls.reach, interior)
    in_band = survivors & (jnp.abs(phi_min) < band) & ~hit_later & passing
    # the virtual outcome
    t_star = s_min / speed
    v_hit = _virtual_hit(x_star, n_star, v, key, wall, mass, dim)
    x_after_hit = x_star + v_hit[:, :dim] * jnp.maximum(dt - t_star, 0.0)[:, None]
    x_miss = x + vd * dt
    x_alt = jnp.where(hit_first[:, None], x_miss, x_after_hit)
    v_alt = jnp.where(hit_first[:, None], v, v_hit)
    # the virtual state's value from the nodes that have LEFT the surface (v . n > 0):
    # a re-emitted particle has, and a particle that grazes past the tip is from then on
    # one that left the surface there -- the approaching population carries the value of
    # the hit it is about to make, which is exactly what the virtual miss does not get
    side = jnp.zeros((n, 3)).at[:, :dim].set(n_star)
    lam_alt = ensemble_device.value_query(
        rec_k.x, rec_k.v, rec_k.alive, lam_k, grid, x_alt, v_alt, direction=side, **fit
    )
    r_alt = jnp.zeros(n)
    if reward is not None:
        head = split.head
        virtual = head._replace(
            face=jnp.full_like(head.face, 2 * dim),
            v_out=v_hit[None],
            diffuse=jnp.zeros_like(head.diffuse),
            x_hit=x_star[None],
            v_in=v[None],
            normal=side[None],
            time=t_star[None],
        )
        r_alt = reward(virtual, None, rec_k, step)[0]
    v_actual = to_go.start
    v_virtual = lam_alt + jnp.where(hit_first, 0.0, r_alt)
    diff = jnp.where(hit_first, v_actual - v_virtual, v_virtual - v_actual)  # V_hit - V_miss
    c = jnp.where(in_band, -diff / (2.0 * band), 0.0)
    return c, x_star


def _phi_at(coef, ls, x_star):
    return levelset_mod.value(dataclasses.replace(ls, coef=coef), x_star)


def _body_hit_slots(events, survivors, dim):
    """The surviving slots with a body (element) event this step, compacted into the
    lane's size (``i32[B]``, ``-1`` unused), and the number that did not fit.  A body hit
    is a pass-0 event that leaves the particle alive, so every such slot is in the
    tracker's lane and a valid forward never overflows; the count is still returned so a
    violation is an error, not a silently dropped cotangent."""
    split = as_split(events)
    head = jnp.any(split.head.face >= 2 * dim, axis=0).astype(jnp.float64)
    tail = (split.tail.face >= 2 * dim).astype(jnp.float64)
    hit = particle_sum(head[None], tail, split.slot) > 0
    slots, lost = lane_slots(hit & survivors, split.slot.shape[0])
    return slots, jnp.sum(lost, dtype=jnp.int32)


def _inexact_outputs(fn, geom):
    """``fn(geom)``'s floating leaves (the differentiable outputs) and, as auxiliary data,
    the whole output: the wall map's VJP restricted to real-valued cotangents, so a stack
    of them can be pulled back in one batched transpose (:func:`_pullbacks`)."""
    out = fn(geom)
    leaves = [a for a in jax.tree.leaves(out) if jnp.issubdtype(a.dtype, jnp.inexact)]
    return leaves, out


def _pullbacks(vjp, cotangents):
    """``[vjp(c) for c in cotangents]`` as ONE transpose of the wall map batched over the
    cotangents (``vmap``): the pull-back is linear in its cotangent, so each result is the
    separate call's up to summation order.  ``cotangents`` are full-structured (``float0``
    at the integer / boolean outputs); only their floating leaves are pulled back."""
    parts = [[a for a in jax.tree.leaves(c) if a.dtype != jax.dtypes.float0] for c in cotangents]
    stacked = [jnp.stack(column) for column in zip(*parts, strict=True)]
    (batched,) = jax.vmap(vjp)(stacked)
    return [jax.tree.map(lambda a, j=j: a[j], batched) for j in range(len(cotangents))]


ORDER_FIT = {"k": 32, "pool": 0}
"""Fit of the collision-increment field for :func:`order_jump`: the hit's own cell only
(no pooled block, which reaches across a thin body; the position half-space of the pooled
shape fits would drop every flow node for the approaching side's query)."""


def order_jump(events, survivors, rec_k, coll_k, grid, npc_max, scale, chunk):
    """Per pass-0 body hit, ``J = c(x_w, v_in) - c(x_w, v_out)`` (``f64[N]``, zero off the
    hits): the change of the objective when the hit moves from the end of this step to the
    start of the next.  The move precedes the step's collisions, so the hit that slips
    collides here as the incoming particle (``v_in``) and hits at the start of the next
    step, instead of hitting now and colliding as the reflected particle (``v_out``).
    Both futures are the same after the next step's move, so the difference is the step's
    expected collision effect, ``c = dt L* lambda`` -- the backward's per-slot increment
    ``coll_k`` (:attr:`ensemble_device.Backward.collision`) fitted at the hit point on the
    approaching (``v . n < 0``) and leaving sides.  Without collisions ``c = 0``: the
    hit-timing band term (:func:`timing_cotangent`) is then the whole hit-step dependence;
    with them this is its collisional part, at every step boundary, not only at the
    window's edges."""
    dim = rec_k.x.shape[1]
    head = as_split(events).head
    body = (head.face[0] >= 2 * dim) & survivors
    slots, _lost = lane_slots(body, as_split(events).slot.shape[0])
    used = slots >= 0
    q = jnp.where(used, slots, 0)
    n = head.normal[0][q]
    x_w = head.x_hit[0][q]
    lam = coll_k.astype(jnp.float64)
    fit = {"npc_max": npc_max, "scale": scale, "chunk": chunk, **ORDER_FIT}
    c_in = ensemble_device.value_query(
        rec_k.x, rec_k.v, rec_k.alive, lam, grid, x_w, head.v_in[0][q], direction=-n, **fit
    )
    c_out = ensemble_device.value_query(
        rec_k.x, rec_k.v, rec_k.alive, lam, grid, x_w, head.v_out[0][q], direction=n, **fit
    )
    n_slots = rec_k.x.shape[0]
    target = jnp.where(used, slots, n_slots)
    return jnp.zeros(n_slots).at[target].set(jnp.where(used, c_in - c_out, 0.0), mode="drop")


def _gradient_step(  # noqa: PLR0915 -- one step's cotangents, kept in one place
    carry,
    xs,
    geom,
    setup,
    fits,
    tally_cotangent,
    timing_jump,
    timing_band,
    silhouette_band,
    reward,
    hit_fits,
):
    """One step of the shape VJP (module-level ``scan`` body).  ``fits`` is a tuple of
    ``grad lambda`` estimators (the fit settings plus ``"along"``); the per-estimator
    carries ``g_x``, ``g_v``, ``g_sil`` are tuples over it, the rest is shared."""
    x, v, alive, g_x, g_v, g_tally, g_score, g_time, g_sil, g_order, err = carry
    rec_k, lam_k, step, r_k, tally_weight, lam_timing_k, klass_k, coll_k = xs
    reward, timing_jump = _weighted(reward, setup), _weighted(timing_jump, setup)
    key = key_for(setup["base_key"], step, STAGE_MOVE)
    f = functools.partial(wall_map, x=x, v=v, alive=alive, key=key, **setup["map"])
    _floats, vjp, primal = jax.vjp(functools.partial(_inexact_outputs, f), geom, has_aux=True)
    x_post, v_post, alive_post, _logp, _tallies, events = primal
    dim = x.shape[1]
    survivors = alive & rec_k.alive & (rec_k.born_face < 0)
    err = jnp.maximum(
        err, _mismatch(x_post, v_post, alive, alive_post, rec_k, survivors, setup["map"]["grid"])
    )
    # value gradients at the post-move state (the record's), on survivors only, from the
    # one-sided fit over the nodes on the hit particle's side of its last surface, of the
    # particle's own class (a body hit ahead or not).  `hit_fits`: only where the post-move
    # state depends on the geometry -- with epsilon = 0 that is the slots with a body
    # event this step; every other slot's cotangent multiplies an exact zero in the VJP
    queries = None
    if hit_fits:
        queries, lost = _body_hit_slots(events, survivors, dim)
        err = jnp.maximum(err, jnp.where(lost > 0, 1.0, 0.0))
    direction = _last_normal(events)
    zeros = _zero_cotangent(primal)
    cotangents = []
    for fit in fits:
        gx, gv = node_gradients(
            rec_k.x,
            rec_k.v,
            rec_k.alive,
            lam_k,
            setup["map"]["grid"],
            direction=direction,
            klass=klass_k,
            queries=queries,
            **_fit_settings(fit),
        )
        if fit["along"]:
            # v . grad_x V = -T - L* V: the collision increment dt L* lam over dt
            t_k = lam_timing_k if coll_k is None else lam_timing_k + coll_k / setup["map"]["dt"]
            gx = along_flight(gx, rec_k.v, t_k)
        gx = jnp.where(survivors[:, None], gx, 0.0)
        gv = jnp.where(survivors[:, None], gv, 0.0)
        cotangents.append((gx, zeros[1], zeros[2], zeros[3], zeros[4], zeros[5]))
        cotangents.append((zeros[0], gv, zeros[2], zeros[3], zeros[4], zeros[5]))
    split = as_split(events)
    rewards = _masked(_as_event_array(r_k, events), survivors, split.slot)
    to_go = _to_go(jnp.where(survivors, lam_k, 0.0), rewards, split.slot)
    # --- score: the move's log-probability and the diffuse normals ---
    # with a tally cotangent the hit's own reward is pathwise through the traced sample
    # (`wall_adjoint._element_events`), so the normal score carries the value-to-go
    # AFTER the hit; without one the reward's whole dependence is in the score
    score_weight = to_go.pre if tally_cotangent is None else to_go.post
    scored = functools.partial(
        normal_score,
        wall=setup["map"]["wall"],
        mesh=setup["map"]["mesh"],
        mass=setup["map"]["mass"],
        dim=dim,
    )
    cot_events = _cotangent_field(
        events,
        "normal",
        EventArray(scored(split.head, score_weight.head), scored(split.tail, score_weight.tail)),
    )
    cotangents.append((zeros[0], zeros[1], zeros[2], to_go.start, zeros[4], cot_events))
    # --- pathwise: the objective's tally weights ---
    if tally_cotangent is not None:
        cot_tally = zeros[4]._replace(
            momentum=tally_weight * tally_cotangent.momentum,
            energy=tally_weight * tally_cotangent.energy,
            heat=tally_weight * tally_cotangent.heat,
        )
        cotangents.append((zeros[0], zeros[1], zeros[2], zeros[3], cot_tally, zeros[5]))
    if timing_jump is not None:
        jump = _masked(_per_part(timing_jump, events, rec_k, step), survivors, split.slot)
        dt_ = setup["map"]["dt"]
        cot_time = _cotangent_field(
            events,
            "time",
            EventArray(
                timing_cotangent(split.head, jump.head, dt_, timing_band),
                timing_cotangent(split.tail, jump.tail, dt_, timing_band),
            ),
        )
        cotangents.append((zeros[0], zeros[1], zeros[2], zeros[3], zeros[4], cot_time))
    if coll_k is not None:
        # the collisional part of the hit-step dependence (order_jump), whole-step band:
        # the hit times of a stationary flow are uniform within the step
        fit0 = fits[0]
        jump_o = order_jump(
            events, survivors, rec_k, coll_k, setup["map"]["grid"], fit0["npc_max"],
            fit0["scale"], fit0["chunk"],
        )  # fmt: skip
        # timing_cotangent takes the objective's loss (before - after) when the hit slips
        head_o = timing_cotangent(split.head, -jump_o[None, :], setup["map"]["dt"], 1.0)
        head_o = jnp.where(split.head.face >= 2 * dim, head_o, 0.0)
        cot_order = _cotangent_field(
            events, "time", EventArray(head_o, jnp.zeros_like(split.tail.time))
        )
        cotangents.append((zeros[0], zeros[1], zeros[2], zeros[3], zeros[4], cot_order))
    pulled = _pullbacks(vjp, cotangents)
    n_fit = 2 * len(fits)
    g_x = tuple(jax.tree.map(jnp.add, g, d) for g, d in zip(g_x, pulled[0:n_fit:2], strict=True))
    g_v = tuple(jax.tree.map(jnp.add, g, d) for g, d in zip(g_v, pulled[1:n_fit:2], strict=True))
    g_score = jax.tree.map(jnp.add, g_score, pulled[n_fit])
    rest = pulled[n_fit + 1 :]
    if tally_cotangent is not None:
        g_tally = jax.tree.map(jnp.add, g_tally, rest.pop(0))
    if timing_jump is not None:
        g_time = jax.tree.map(jnp.add, g_time, rest.pop(0))
    if coll_k is not None:
        g_order = jax.tree.map(jnp.add, g_order, rest.pop(0))
    if silhouette_band > 0.0:
        mesh_g = with_geometry(setup["map"]["mesh"], geom)
        sil = []
        for fit, g in zip(fits, g_sil, strict=True):
            c_sil, x_star = silhouette_cotangent(
                mesh_g.body,
                x,
                v,
                survivors,
                events,
                to_go,
                rec_k,
                lam_k,
                step,
                jax.random.fold_in(key, 7919),
                setup["map"],
                _fit_settings(fit),
                silhouette_band,
                reward,
            )
            _phi, vjp_phi = jax.vjp(functools.partial(_phi_at, ls=mesh_g.body, x_star=x_star), geom)
            (d_sil,) = vjp_phi(c_sil)
            sil.append(jax.tree.map(jnp.add, g, d_sil))
        g_sil = tuple(sil)
    hit_tail = (split.tail.face >= 2 * dim) & _tail_of(survivors, split.slot)[None, :]
    n_hits = jnp.sum((split.head.face >= 2 * dim) & survivors[None, :], dtype=jnp.int32)
    n_hits = n_hits + jnp.sum(hit_tail, dtype=jnp.int32)
    return (
        rec_k.x,
        step_v_post(rec_k),
        rec_k.alive,
        g_x,
        g_v,
        g_tally,
        g_score,
        g_time,
        g_sil,
        g_order,
        err,
    ), n_hits


def _fit_settings(fit):
    """The :func:`ensemble_device.node_gradients` / ``value_query`` keywords of an
    estimator entry (everything but ``"along"``)."""
    return {name: value for name, value in fit.items() if name != "along"}


def _setup(
    base_key, grid, faces, wall, mass, dt, mesh, epsilon, acc_delta, static, reward_weight=None
):
    return {
        "base_key": base_key,
        "reward_weight": reward_weight,
        "map": {
            "grid": grid,
            "faces": faces,
            "wall": wall,
            "mass": mass,
            "dt": dt,
            "mesh": mesh,
            "epsilon": epsilon,
            "acc_delta": acc_delta,
            "max_bounces": static.K_bounce,
            "tracker": static.tracker,
            "max_walk": static.K_walk,
            "lane_size": static.lane_size,
        },
    }


def _step_alone(xs_k, pre, body, zeros):
    """One step of the shape VJP's ``body`` from its own pre-move state ``pre`` with zero
    accumulators: the step's contributions (the carry less the state) and its hit count."""
    carry, hits = body((*pre, *zeros), xs_k)
    return carry[3:], hits


def _batched_chunk(carry, c, body, xs, size):
    """``scan`` body of :func:`_batched_scan`: chunk ``c`` of ``size`` steps through
    ``vmap``."""
    xs_c = jax.tree.map(lambda a: lax.dynamic_slice_in_dim(a, c * size, size), xs)
    x, v, alive, *acc = carry
    rec = xs_c[0]
    x_post, v_post, alive_post = rec.x, jax.vmap(step_v_post)(rec), rec.alive
    pre = tuple(
        jnp.concatenate((a[None], b[:-1]))
        for a, b in ((x, x_post), (v, v_post), (alive, alive_post))
    )
    zeros = jax.tree.map(jnp.zeros_like, tuple(acc))
    batched = jax.vmap(
        functools.partial(_step_alone, body=body, zeros=zeros), axis_name=step_axis.STEPS
    )
    parts, hits = batched(xs_c, pre)
    *sums, err = acc
    *steps, errs = parts
    sums = jax.tree.map(lambda a, d: a + jnp.sum(d, axis=0), tuple(sums), tuple(steps))
    err = jnp.maximum(err, jnp.max(errs))
    return (x_post[-1], v_post[-1], alive_post[-1], *sums, err), hits


def _batched_scan(body, init, xs, batch):
    """``lax.scan(body, init, xs)`` of the shape VJP with ``batch`` steps at a time (the
    largest divisor of the step count up to it).  The steps are independent: a step's
    pre-move state is the previous step's recorded post-move state (the scan carries
    nothing else but sums), so a chunk's steps run side by side through ``vmap`` -- one
    kernel launch for ``batch`` steps' work.  The per-step replay and pull-back are
    hundreds of small kernels and data-dependent conditionals (each a device-to-host
    sync on GPU); batched, each conditional is taken once for the chunk, on its predicate
    reduced over the steps (:mod:`jax_dsmc.step_axis`).  Sums differ from the sequential
    scan's in their order only."""
    m = jax.tree.leaves(xs[0])[0].shape[0]
    b = max(d for d in range(1, min(batch, m) + 1) if m % d == 0)
    chunk = functools.partial(_batched_chunk, body=body, xs=xs, size=b)
    carry, hits = lax.scan(chunk, init, jnp.arange(m // b, dtype=jnp.int32))
    return carry, hits.reshape(m)


@functools.partial(
    jax.jit,
    static_argnames=("faces", "static", "reward", "timing_jump", "timing_band", "patch_tally"),
)
def replay(
    records,
    steps,
    state0,
    base_key,
    grid,
    faces,
    wall,
    mass,
    dt,
    mesh,
    epsilon,
    acc_delta,
    static,
    reward=None,
    timing_jump=None,
    timing_band=0.2,
    reward_weight=None,
    patch_tally=None,
):
    """Traced replay of the captured steps: a :class:`Replay`.

    ``records`` is the stacked :class:`EventRecord`, ``steps`` ``i32[M]`` their absolute
    step indices (the key contract), ``state0`` the pre-move state of the first captured
    step.  ``reward(events, tallies, rec_k, step) -> f64[K, N]`` (module-level, constants
    bound with ``partial``) is the per-hit reward of the objective, already divided by
    its normalization; a survivor's rewards are kept, a newborn's are the source's.
    ``timing_jump`` (as in :func:`gradient`) adds the per-hit timing rewards.
    ``reward_weight`` (``f64[D]``, or any pytree the reward takes, traced) is bound to both as
    ``weight=`` (:func:`_weighted`).  ``patch_tally`` (module-level, a ``partial`` carrying
    ``n_nodes``) sums the survivors' hits onto a node lattice (:attr:`Replay.patch`).
    """
    setup = _setup(
        base_key, grid, faces, wall, mass, dt, mesh, epsilon, acc_delta, static, reward_weight
    )
    body = functools.partial(
        _replay_step,
        geom=geometry(mesh),
        reward=reward,
        timing_jump=timing_jump,
        timing_band=timing_band,
        setup=setup,
        patch_tally=patch_tally,
    )
    init = (state0.x, state0.v, state0.alive, jnp.float64(0.0))
    (_, _, _, err), (r, tallies, hits, tau, patch) = lax.scan(body, init, (records, steps))
    return Replay(r, tallies, err, hits, tau, None if patch_tally is None else patch.sum(axis=0))


@functools.partial(
    jax.jit,
    static_argnames=(
        "faces",
        "static",
        "npc_max",
        "basis",
        "chunk",
        "timing_jump",
        "estimators",
        "silhouette_band",
        "reward",
        "value_class",
        "hit_fits",
        "batch",
    ),
)
def _gradient(
    records,
    lam,
    steps,
    state0,
    base_key,
    grid,
    faces,
    wall,
    mass,
    dt,
    mesh,
    epsilon,
    acc_delta,
    static,
    scale,
    npc_max,
    hit_reward=None,
    tally_cotangent=None,
    tally_weight=None,
    basis="quadratic_full",
    chunk=4096,
    timing_jump=None,
    timing_band=0.2,
    estimators=((0, 32, False),),
    lam_timing=None,
    silhouette_band=0.0,
    reward=None,
    value_class="none",
    hit_fits=False,
    collision=None,
    batch=1,
    reward_weight=None,
):
    """:func:`gradient` for several ``grad lambda`` estimators ``(pool, k, along)`` at once
    (a tuple of :class:`ShapeTerms`, one per estimator): one wall-map replay and one
    batched pull-back per step serve them all -- only the fits differ.  ``batch > 1``
    evaluates that many steps at once (:func:`_batched_scan`)."""
    if value_class not in ("hit_ahead", "none"):
        raise ValueError(value_class)
    setup = _setup(
        base_key, grid, faces, wall, mass, dt, mesh, epsilon, acc_delta, static, reward_weight
    )
    geom = geometry(mesh)
    fits = tuple(
        {"npc_max": npc_max, "scale": scale, "k": k_e, "basis": basis, "chunk": chunk,
         "pool": pool, "along": along, "halfspace": e[3] if len(e) > 3 else None}
        for e in estimators
        for pool, k_e, along in (e[:3],)
    )  # fmt: skip
    if any(e[2] for e in estimators) and lam_timing is None:
        raise ValueError("an along-flight estimator needs lam_timing")
    body = functools.partial(
        _gradient_step,
        geom=geom,
        setup=setup,
        fits=fits,
        tally_cotangent=tally_cotangent,
        timing_jump=timing_jump,
        timing_band=timing_band,
        silhouette_band=silhouette_band if mesh is not None and mesh.kind == "levelset" else 0.0,
        reward=reward,
        hit_fits=hit_fits,
    )
    m = lam.shape[0]
    if tally_weight is None:
        tally_weight = jnp.ones(m)
    klass = hit_ahead(records, mesh) if value_class == "hit_ahead" else None
    zero_g = jax.tree.map(jnp.zeros_like, geom)
    per = (zero_g,) * len(fits)
    init = (
        state0.x,
        state0.v,
        state0.alive,
        per,
        per,
        zero_g,
        zero_g,
        zero_g,
        per,
        zero_g,
        jnp.float64(0.0),
    )
    xs = (records, lam, steps, hit_reward, tally_weight, lam_timing, klass, collision)
    if batch > 1:
        (_, _, _, g_x, g_v, g_tally, g_score, g_time, g_sil, g_order, err), hits = _batched_scan(
            body, init, xs, batch
        )
    else:
        (_, _, _, g_x, g_v, g_tally, g_score, g_time, g_sil, g_order, err), hits = lax.scan(
            body, init, xs
        )
    return tuple(
        ShapeTerms(
            jax.tree.map(lambda a, b, c: a + b + c, gx, gv, g_tally),
            gx,
            gv,
            g_tally,
            g_score,
            g_time,
            gs,
            err,
            hits,
            None if collision is None else g_order,
        )
        for gx, gv, gs in zip(g_x, g_v, g_sil, strict=True)
    )


def gradient(
    records,
    lam,
    steps,
    state0,
    base_key,
    grid,
    faces,
    wall,
    mass,
    dt,
    mesh,
    epsilon,
    acc_delta,
    static,
    scale,
    npc_max,
    hit_reward=None,
    tally_cotangent=None,
    tally_weight=None,
    k=32,
    basis="quadratic_full",
    chunk=4096,
    timing_jump=None,
    timing_band=0.2,
    pool=0,
    lam_timing=None,
    silhouette_band=0.0,
    reward=None,
    value_class="none",
    hit_fits=False,
    batch=1,
    reward_weight=None,
):
    """The shape gradient of the captured window: :class:`ShapeTerms` shaped like
    :func:`geometry` ``(mesh)``.  ``silhouette_band > 0`` (a length, level-set bodies
    only) switches on :func:`silhouette_cotangent` with ``reward`` the objective's per-hit
    reward function (as :func:`replay`) for the virtual hits.

    ``lam`` ``f64[M, N]`` is the ensemble backward's adjoint at the post-move states
    (``Backward.lam``), ``hit_reward`` the array the backward was given (``None`` = no hit
    rewards), ``tally_cotangent`` a :class:`SurfaceTallies`-shaped ``dJ/d tally`` (its
    ``hits`` ignored) applied with the per-step factor ``tally_weight`` ``f64[M]`` (the
    window indicator), ``scale``/``npc_max``/``k``/``basis``/``chunk`` the fit settings
    of the backward.  ``timing_jump(events, rec_k, step) -> f64[K, N]`` (module-level)
    switches on the hit-timing band term of width ``timing_band`` (a fraction of ``dt``),
    see :func:`timing_cotangent`.  ``pool`` widens the fit's node set to the ``(2 pool +
    1)^D`` block of cells; ``lam_timing`` ``f64[M, N]`` (the value of the timing rewards of
    :func:`replay`, from a second backward) switches the along-flight component of
    ``grad_x lambda`` to the identity of :func:`along_flight`.

    ``value_class`` stratifies the ``grad lambda`` fits (:func:`node_gradients`):
    ``"none"`` (default) fits from the mixed pool.  ``"hit_ahead"`` is **diagnostic only,
    never a correct estimator**: it fits a particle's value from the particles of its own
    class, class = a body hit at a later captured step -- a property of the particle's
    FUTURE, not of its state.  With ``lambda = P lam_1 + (1 - P) lam_0`` (``P`` the
    probability of a later hit) a fit within the class estimates ``P grad lam_1 + (1 - P)
    grad lam_0`` and drops ``grad P (lam_1 - lam_0)``; restoring it needs ``grad P`` from
    the same local regression, which turns the estimator back into the mixed-pool fit with
    more variance.  Measured: coarse ``cos 2 theta`` +40 %, free-molecular squash +4..20 %
    (2026-10-02/03; docs/math/shape_validation.tex, Finding 1).  Stratification is
    legitimate only by a function of the current state (the one-sided ``v . n`` rule, the
    ``halfspace`` plane), which is what the fits use.

    ``hit_fits=True`` fits ``grad lambda`` only at the slots with a body event in the step
    (:func:`_body_hit_slots`) instead of every slot -- the same gradient (to the last bit)
    at ~1/200 of the fitting work, **valid only for ``epsilon = 0``**: with the regularised
    flight time every particle's post-move state depends on the geometry through the hit
    set, and the restriction would drop those terms.  (``epsilon`` is traced, so this is
    the caller's declaration; ``examples.ensemble_mesh.shape_terms`` runs at 0.)

    ``batch > 1`` evaluates that many captured steps side by side (:func:`_batched_scan`):
    the same terms up to summation order, fewer kernel launches, more memory.
    ``reward_weight`` as :func:`replay`.
    """
    _warn_value_class(value_class)
    (terms,) = _gradient(
        records,
        lam,
        steps,
        state0,
        base_key,
        grid,
        faces,
        wall,
        mass,
        dt,
        mesh,
        epsilon,
        acc_delta,
        static,
        scale,
        npc_max,
        hit_reward=hit_reward,
        tally_cotangent=tally_cotangent,
        tally_weight=tally_weight,
        basis=basis,
        chunk=chunk,
        timing_jump=timing_jump,
        timing_band=timing_band,
        estimators=((pool, k, lam_timing is not None),),
        lam_timing=lam_timing,
        silhouette_band=silhouette_band,
        reward=reward,
        value_class=value_class,
        hit_fits=hit_fits,
        batch=batch,
        reward_weight=reward_weight,
    )
    return terms


def hit_ahead(records, mesh):
    """``bool[M, N]``: does the slot hit the body (a surface id ``>= 2D``) at a captured
    step *after* this one -- the value class of the stratified fit (the reward of that
    hit is in the slot's value; a slot without one carries only what collisions and the
    window edges give it)."""
    dim = records.x.shape[-1]
    del mesh
    walls = records.walls
    if isinstance(walls, RecordEvents):
        head = jnp.any(walls.head.face >= 2 * dim, axis=1)  # (M, N)
        n = head.shape[1]
        # the record's later-pass events folded onto their slots, step by step
        body = (walls.face >= 2 * dim) & (walls.slot >= 0)  # (M, E)
        target = jnp.where(body, walls.slot, n)
        steps = jnp.arange(head.shape[0])[:, None]
        hit = head.at[steps, target].set(True, mode="drop")
    elif isinstance(walls, SplitEvents):
        head = jnp.any(walls.head.face >= 2 * dim, axis=1)  # (M, N)
        tail = jnp.any(walls.tail.face >= 2 * dim, axis=1).astype(jnp.float64)  # (M, B)
        # the lane's hits folded onto their slots, step by step
        folded = jax.vmap(particle_sum)(
            jnp.zeros_like(head, jnp.float64)[:, None], tail[:, None], walls.slot
        )
        hit = head | (folded > 0)
    else:
        hit = jnp.any(walls.face >= 2 * dim, axis=1)  # (M, N)
    later = jnp.flip(jnp.cumsum(jnp.flip(hit, 0), axis=0), 0) - hit.astype(jnp.int32)
    return later > 0


VARIANTS = {
    # name: (pool, k, along-flight) -- the estimators of grad_x lambda compared in review
    # section 7.6 item 5; every one of them is one-sided (the last hit's normal)
    "one": (0, 32, False),
    "one_pool": (1, 96, False),
    "along": (0, 32, True),
    "along_pool": (1, 96, True),
    # the pooled fits without the nodes more than a quarter cell behind the hit's tangent
    # plane: a body thinner than the 3^D block puts its far side's particles in the pool
    # (2026-10-02 bias study)
    "one_pool_plane": (1, 96, False, 0.25),
    "along_pool_plane": (1, 96, True, 0.25),
}


def _warn_value_class(value_class):
    if value_class == "hit_ahead":
        warnings.warn(
            'value_class="hit_ahead" is biased by construction (it conditions the fits on '
            "the particle's future); diagnostic use only -- see shape.gradient",
            RuntimeWarning,
            stacklevel=3,
        )


def gradient_variants(lam_timing, names=None, collision=None, **kwargs):
    """:func:`gradient` for every entry of :data:`VARIANTS` (the same records, cotangents
    and rewards; only the estimate of ``grad_x lambda`` differs): ``{name: ShapeTerms}``.
    ``lam_timing`` is the timing-reward value of the along-flight variants; ``names``
    restricts the set.  One scan serves all of them (:func:`_gradient`): the wall-map
    replay, its pull-back and the shared terms are computed once per step.

    ``collision`` ``f64[M, N]`` (:attr:`ensemble_device.Backward.collision` of the
    objective's value column, ``dt L* lam``) adds the collision term of the along-flight
    identity, ``v . grad_x V = -T - L* V`` (:func:`along_flight`); required for a
    collisional run, a no-op (zeros) without collisions."""
    _warn_value_class(kwargs.get("value_class", "none"))
    chosen = [name for name in VARIANTS if names is None or name in names]
    estimators = tuple(VARIANTS[name] for name in chosen)
    along = any(e[2] for e in estimators)
    for name in ("pool", "k", "lam_timing"):
        kwargs.pop(name, None)
    terms = _gradient(
        **kwargs,
        estimators=estimators,
        lam_timing=lam_timing if along else None,
        collision=collision,
    )
    return dict(zip(chosen, terms, strict=True))


# --- the cut-volume term ------------------------------------------------------------------


def volume_weights(cell_collision, grid):
    """``g_c = -sum_k C_kc / V_c`` (``f64[C]``, ``C`` the collision cells): the cut-volume
    term's weight on each cell's flow volume, ``dJ/dtheta |_V = sum_c g_c dV_c/dtheta``.

    NTC attempts ``N_c (N_c - 1) F (sigma g)_max dt / (2 V_c)`` pairs per step, so every
    collision probability of cell ``c`` scales as ``1 / V_c``, and the expected objective
    changes by ``-(dV_c / V_c) C_kc`` per step with ``C_kc`` the expected change of the
    objective by the cell's collisions -- the same form as the kernel-parameter term with
    ``d log q`` replaced by ``-d log V_c``.  ``cell_collision`` ``f64[M, C]`` is
    :attr:`ensemble_device.Backward.cell_collision`: ``C_kc = 1/2 sum_{i in c}
    collision_ki``, each slot's increment being ``sum_j P_ij Phi_ij`` and every pair seen
    from both ends."""
    total = jnp.sum(cell_collision, axis=0)
    volume = grid.volume
    return jnp.where(volume > 0.0, -total / jnp.where(volume > 0.0, volume, 1.0), 0.0)


VOLUME_CHUNK_POINTS = 1 << 20
"""Sample points per chunk of :func:`volume_cotangent` (2026-10-24: every Cartesian cell's
``32^3`` samples at once was 6.3 GB on the 3-D sphere, the first 3-D optimiser run)."""
VOLUME_SATURATION = 20.0
"""Cells whose centre is further than this many Heaviside widths (plus the half-diagonal)
from the surface are left out of the cut-volume term: ``tanh`` is 1 there to double precision,
the samples' derivative ``~ 4 exp(-40)``."""


def _cell_samples(cells, grid, sub):
    """Centred ``sub^D`` sample points of the Cartesian cells ``cells`` (flat indices,
    row-major: grid.cell_of's order), ``f64[len(cells) sub^D, D]``, the collision cell each
    lies in (a split cell's sub-cell by its region raster) and the volume each stands for
    (with ``grid.transverse``)."""
    d = len(grid.n)
    dx = spacing(grid)
    idx = jnp.stack(jnp.unravel_index(cells, grid.n), axis=-1)
    offs = jnp.stack(jnp.meshgrid(*[(jnp.arange(sub) + 0.5) / sub] * d, indexing="ij"), axis=-1)
    x = grid.lo + (idx.reshape(-1, 1, d) + offs.reshape(1, -1, d)) * dx
    x = x.reshape(-1, d)
    return x, collision_cell(x, grid), jnp.prod(dx) / sub**d * grid.transverse


def _near_surface_cells(ls, grid, width):
    """Flat indices of the Cartesian cells within :data:`VOLUME_SATURATION` widths of the
    surface (centre ``|phi|`` against the half-diagonal plus that margin), padded with ``-1``
    to a power of two (iterates share compilations)."""
    d = len(grid.n)
    dx = np.asarray(spacing(grid))
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in grid.n], indexing="ij"), axis=-1)
    centres = np.asarray(grid.lo) + (idx.reshape(-1, d) + 0.5) * dx
    phi = np.asarray(levelset_mod.value(ls, jnp.asarray(centres)))
    reach = 0.5 * float(np.linalg.norm(dx)) + VOLUME_SATURATION * width
    near = np.nonzero(np.abs(phi) <= reach)[0]
    size = 1 << max(int(near.size - 1).bit_length(), 0)
    return np.concatenate((near, np.full(size - near.size, -1)))


def _weighted_flow_volume(coef, ls, x, w_sample, width):
    """``sum_c weights_c V_c`` with ``V_c`` the smoothed flow volume of collision cell
    ``c``: its samples' ``1 - H_w(-phi)`` (``phi < 0`` is the body), ``w_sample`` the
    weight of each sample's cell times its volume."""
    phi = levelset_mod.value(dataclasses.replace(ls, coef=coef), x)
    flow = 0.5 * (1.0 + jnp.tanh(phi / width))
    return jnp.sum(w_sample * flow)


def _chunk_cotangent(cot, cells, coef, ls, grid, weights, sub, width):
    """:func:`lax.scan` body of :func:`volume_cotangent`: one chunk of cells' samples
    (``-1`` padding: weight 0) added to the cotangent."""
    valid = cells >= 0
    x, cell, area = _cell_samples(jnp.where(valid, cells, 0), grid, sub)
    keep = jnp.repeat(valid, sub ** len(grid.n))
    w_sample = jnp.where(keep, weights[cell] * area, 0.0)
    part = jax.grad(_weighted_flow_volume)(coef, ls, x, w_sample, width)
    return cot + part, None


def volume_cotangent(geom, mesh, grid, weights, sub=32, width=None):
    """The cut-volume term as a cotangent on the level-set coefficients ``geom``:
    ``sum_c weights_c dV_c / dcoef`` with ``V_c`` the smoothed flow volume of each
    collision cell (``sub^D`` centred samples per Cartesian cell, each assigned to its
    collision cell -- a split cell's samples to their sub-cells by the region raster --
    Heaviside half-width ``width``, default a quarter of the knot spacing), i.e.
    ``int_c delta(phi) B_j`` -- the shape derivative of the cut volume,
    ``int_{Gamma cap c} B_j / |grad phi|``.  ``weights`` from :func:`volume_weights`.  The
    sample spacing must resolve ``width`` (the default ``sub = 32`` gives 4 samples per
    ``width`` on a cell of 8 knots).  Only the cells near the surface are sampled
    (:func:`_near_surface_cells`; the rest add nothing), :data:`VOLUME_CHUNK_POINTS` samples
    at a time."""
    if mesh.kind != "levelset":
        raise ValueError("the cut-volume term is built for level-set bodies")
    ls = mesh.body
    width = ls.h / 4.0 if width is None else float(width)
    cells = _near_surface_cells(dataclasses.replace(ls, coef=geom), grid, width)
    per = max(1, VOLUME_CHUNK_POINTS // sub ** len(grid.n))
    per = min(per, cells.size)
    cells = np.concatenate((cells, np.full(-cells.size % per, -1)))
    body = functools.partial(
        _chunk_cotangent, coef=geom, ls=ls, grid=grid, weights=weights, sub=sub, width=width
    )
    cot, _ = lax.scan(body, jnp.zeros_like(geom), jnp.asarray(cells.reshape(-1, per)))
    return cot
